"""Synapse co-assignment graph.

Synapses are the invariant nodes. v117 segment IDs and DNA embeddings
supply the evidence for which synapses share a neuron.

No feature engineering here — raw inputs only. Position normalisation
and feature weighting are the model's job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynapseGraph:
    """A synapse graph for co-assignment.

    Attributes
    ----------
    node_pos  : [N, 3]  float32  raw nm positions
    node_dna  : [N, D]  float32  segment DNA (zeros when unavailable)
    seg_id    : [N]     int64    v117 segment assignment (noisy evidence)
    labels    : [N]     int64    ground-truth neuron IDs (0 = unknown)
    edge_src  : [E]     int64
    edge_dst  : [E]     int64
    same_seg  : [E]     float32  1.0 if both nodes share a v117 segment
    """

    node_pos: np.ndarray
    node_dna: np.ndarray
    seg_id: np.ndarray
    labels: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    same_seg: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.node_pos)

    @property
    def n_edges(self) -> int:
        return len(self.edge_src)

    @property
    def dna_dim(self) -> int:
        return self.node_dna.shape[1]

    @property
    def node_dim(self) -> int:
        return 3 + self.dna_dim


def build_synapse_graph(
    positions: np.ndarray,            # [N, 3] nm
    seg_ids: np.ndarray,              # [N] int64
    labels: np.ndarray,               # [N] int64, 0 = unknown
    seg_dna: dict[int, np.ndarray],   # seg_id → DNA embedding [D]
    *,
    k_spatial: int = 8,
    max_same_seg_pairs: int = 200,
) -> SynapseGraph:
    """Build synapse graph from positions, segment assignments, and DNA.

    Two edge types, both represented uniformly — the model learns to weight them:
      same-segment : both nodes share a v117 segment  (same_seg = 1)
      spatial k-NN : positionally close nodes          (same_seg = 0)

    Parameters
    ----------
    positions:
        Synapse positions in nanometres.
    seg_ids:
        v117 segment ID for each synapse. 0 = no segment assigned.
    labels:
        Ground-truth neuron IDs for supervision. 0 = unknown (masked in loss).
    seg_dna:
        DNA embedding per segment from SkeletonGNN. Synapses whose segment
        has no DNA entry get a zero vector.
    k_spatial:
        Spatial k-NN degree.
    max_same_seg_pairs:
        Cap on directed same-segment pairs per segment, preventing O(N²)
        blowup from large frankenmerge segments.
    """
    from .._scipy_compat import cKDTree

    N = len(positions)
    if len(seg_ids) != N or len(labels) != N:
        raise ValueError(
            f"positions, seg_ids, and labels must all have the same length; "
            f"got {N}, {len(seg_ids)}, {len(labels)}"
        )
    pos = positions.astype(np.float32)

    # DNA lookup — dimension inferred from first entry
    dna_dim = next(iter(seg_dna.values())).shape[0] if seg_dna else 0
    node_dna = np.zeros((N, dna_dim), dtype=np.float32)
    for i, sid in enumerate(seg_ids):
        vec = seg_dna.get(int(sid))
        if vec is not None:
            node_dna[i] = vec

    # Same-segment edges
    seg_groups: dict[int, list[int]] = {}
    for i, sid in enumerate(seg_ids):
        if int(sid) != 0:
            seg_groups.setdefault(int(sid), []).append(i)

    ss_src: list[int] = []
    ss_dst: list[int] = []
    rng_ss = np.random.default_rng(0)
    for sid, idxs in seg_groups.items():
        g = len(idxs)
        if g < 2:
            continue
        pairs = [(idxs[a], idxs[b]) for a in range(g) for b in range(a + 1, g)]
        if len(pairs) > max_same_seg_pairs:
            sel = rng_ss.choice(len(pairs), max_same_seg_pairs, replace=False)
            pairs = [pairs[int(k)] for k in sel]
        for u, v in pairs:
            ss_src += [u, v]
            ss_dst += [v, u]

    # Spatial k-NN edges
    k = min(k_spatial + 1, N)
    _, nbr_idxs = cKDTree(pos).query(pos, k=k)
    sp_src: list[int] = []
    sp_dst: list[int] = []
    for i in range(N):
        for j in nbr_idxs[i, 1:]:
            sp_src.append(i)
            sp_dst.append(int(j))

    n_ss = len(ss_src)
    edge_src = np.array(ss_src + sp_src, dtype=np.int64)
    edge_dst = np.array(ss_dst + sp_dst, dtype=np.int64)
    same_seg = np.concatenate([
        np.ones(n_ss, dtype=np.float32),
        np.zeros(len(sp_src), dtype=np.float32),
    ])

    return SynapseGraph(
        node_pos=pos,
        node_dna=node_dna,
        seg_id=seg_ids.astype(np.int64),
        labels=labels.astype(np.int64),
        edge_src=edge_src,
        edge_dst=edge_dst,
        same_seg=same_seg,
    )
