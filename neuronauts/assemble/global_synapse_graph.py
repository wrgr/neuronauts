"""Global synapse graph with DNA node features (Phase 2 input).

Every synapse in a Region becomes a graph node.  Node features are the DNA
embedding of the seg root the synapse belongs to (via Fragment.synapse_indices).
Edges connect each synapse to its k spatially nearest neighbours; edge features
are the log-normalised pairwise distance.

This replaces the box-local ``build_synapse_graph`` with a graph that has no
spatial boundary and uses learned morphological identity (DNA) instead of raw
position as the primary node signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..schemas import Fragment, Region
from ..represent.enrich import build_synapse_dna_matrix


@dataclass
class GlobalSynapseGraph:
    """Sparse synapse graph ready for CellGNN message-passing.

    Attributes
    ----------
    node_feat : ndarray [N, D]
        DNA embedding per synapse (float32).  Synapses without a matching
        fragment get zero vectors.
    node_pos : ndarray [N, 3]
        Synapse centroid in nm (float32).
    edge_src : ndarray [E]
        Source node indices (int64) — directed edges (both directions kept).
    edge_dst : ndarray [E]
        Destination node indices (int64).
    edge_feat : ndarray [E, 1]
        Log-normalised distance feature (float32).
    pre_root_id : ndarray [N]
        Label-version root ID per synapse (int64).  Used as ground truth
        during training and evaluation.
    """

    node_feat: np.ndarray
    node_pos: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_feat: np.ndarray
    pre_root_id: np.ndarray

    @property
    def n_synapses(self) -> int:
        return len(self.node_feat)

    @property
    def n_edges(self) -> int:
        return len(self.edge_src)

    @property
    def dna_dim(self) -> int:
        return self.node_feat.shape[1]


def build_global_synapse_graph(
    region: Region,
    fragments: Sequence[Fragment],
    *,
    k_neighbors: int = 8,
    max_dist_nm: float | None = None,
    dist_scale_nm: float = 10_000.0,
) -> GlobalSynapseGraph:
    """Build a k-NN synapse graph with DNA node features.

    Parameters
    ----------
    region:
        Region with pre_pt_nm, post_pt_nm, pre_root_id set.
    fragments:
        Fragments with dna filled by encode_fragments.
    k_neighbors:
        Spatial nearest neighbours per synapse (both directions kept → up to
        2·k edges per node).
    max_dist_nm:
        Prune edges beyond this distance.  Default: no pruning.
    dist_scale_nm:
        Reference distance for log-normalisation.  Edge feature =
        ``log(1 + dist / dist_scale_nm)``.

    Returns
    -------
    GlobalSynapseGraph with directed edges.
    """
    from .._scipy_compat import cKDTree

    node_feat = build_synapse_dna_matrix(region, fragments)
    pos = ((region.pre_pt_nm + region.post_pt_nm) / 2.0).astype(np.float32)

    N = len(pos)
    k = min(k_neighbors + 1, N)

    tree = cKDTree(pos)
    dists, idxs = tree.query(pos, k=k, workers=-1)

    src_list: list[int] = []
    dst_list: list[int] = []
    dist_list: list[float] = []

    for i in range(N):
        for slot in range(1, k):
            j = int(idxs[i, slot])
            d = float(dists[i, slot])
            if max_dist_nm is not None and d > max_dist_nm:
                continue
            src_list.append(i)
            dst_list.append(j)
            dist_list.append(d)

    edge_src = np.array(src_list, dtype=np.int64)
    edge_dst = np.array(dst_list, dtype=np.int64)
    dist_arr = np.array(dist_list, dtype=np.float32)
    edge_feat = np.log1p(dist_arr / dist_scale_nm).reshape(-1, 1)

    pre_root_id = (
        region.pre_root_id.copy()
        if region.pre_root_id is not None
        else np.zeros(N, dtype=np.int64)
    )

    return GlobalSynapseGraph(
        node_feat=node_feat,
        node_pos=pos,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_feat=edge_feat,
        pre_root_id=pre_root_id,
    )
