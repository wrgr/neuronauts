"""Half-synapse graph for neuron-partition learning (Phase 2.1).

Each synapse contributes two independent nodes — a pre-half and a post-half.
Pre-halves are partitioned independently from post-halves; each partition groups
nodes by the neuron that owns them.

Node features: concat(normalised_position [3], seg_dna [D])
    - position is the measured synapse location (nm), normalised by pos_scale_nm
    - seg_dna is the SkeletonGNN embedding of the v117 segment that owns the node
      (zero vector when no skeleton is available)

Edge types:
    0 — same-segment: both nodes share the same v117 seg_id (strong evidence;
        may span a frankenmerge, so the GNN must learn to override when DNA
        signals disagree)
    1 — spatial k-NN: positionally close nodes (weak positional evidence)

Edge features [E, 3]:
    [is_same_seg (0/1), is_spatial (0/1), cosine_similarity(dna_i, dna_j)]
    DNA cosine similarity is the physically-grounded signal that two nodes may
    share a neuron arbor (neurons are morphologically consistent trees).

Supervision labels (ground truth, never used as input features):
    pre-side labels  = Region.pre_root_id  (label-version neuron IDs)
    post-side labels = Region.post_root_id
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..schemas import Fragment, Region


@dataclass
class HalfSynapseGraph:
    """Half-synapse graph for one side (pre or post).

    Attributes
    ----------
    node_feat : ndarray [N, 3+D]
        Concat of normalised position and seg DNA (zeros when DNA unavailable).
    node_pos : ndarray [N, 3]
        Raw nm coordinates (pre_pt_nm or post_pt_nm).
    edge_src : ndarray [E] int64
        Source node indices.
    edge_dst : ndarray [E] int64
        Destination node indices.
    edge_type : ndarray [E] int64
        0 = same-segment, 1 = spatial k-NN.
    edge_feat : ndarray [E, 3] float32
        [is_same_seg, is_spatial, dna_cos_sim].
    labels : ndarray [N] int64
        Label-version root IDs (supervision only, not input features).
    seg_id : ndarray [N] int64
        Seg-version segment IDs (the noisy evidence channel).
    side : str
        "pre" or "post".
    """

    node_feat: np.ndarray
    node_pos: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_type: np.ndarray
    edge_feat: np.ndarray
    labels: np.ndarray
    seg_id: np.ndarray
    side: str

    @property
    def n_nodes(self) -> int:
        return len(self.node_feat)

    @property
    def n_edges(self) -> int:
        return len(self.edge_src)

    @property
    def node_dim(self) -> int:
        return self.node_feat.shape[1]

    @property
    def edge_dim(self) -> int:
        return self.edge_feat.shape[1] if self.edge_feat.ndim == 2 else 0

    @property
    def dna_dim(self) -> int:
        return self.node_dim - 3


def build_half_synapse_graph(
    region: Region,
    fragments: Sequence[Fragment],
    *,
    side: str = "pre",
    k_spatial: int = 8,
    max_dist_nm: float | None = None,
    pos_scale_nm: float = 50_000.0,
    max_same_seg_pairs: int = 200,
) -> HalfSynapseGraph:
    """Build a typed half-synapse graph with DNA-enriched node features.

    Parameters
    ----------
    region:
        Region providing synapse positions, seg IDs, and label IDs.
    fragments:
        Fragments with ``dna`` filled by ``encode_fragments_gnn``.  Each
        fragment's ``base_root_id`` (seg-version) is the key for DNA lookup.
    side:
        ``"pre"`` or ``"post"`` — which half of each synapse to build nodes for.
    k_spatial:
        Number of spatial nearest neighbours per node.
    max_dist_nm:
        Prune spatial edges beyond this distance.  Default: no pruning.
    pos_scale_nm:
        Divisor for position normalisation.  Default 50 µm keeps position and
        DNA features on similar scales.
    max_same_seg_pairs:
        Cap on same-segment directed pairs *per segment* to avoid O(N²) blowup
        from frankenmerge segments with many synapses.

    Returns
    -------
    HalfSynapseGraph
    """
    from .._scipy_compat import cKDTree

    if side not in ("pre", "post"):
        raise ValueError(f"side must be 'pre' or 'post', got {side!r}")

    if side == "pre":
        pos = region.pre_pt_nm.astype(np.float32)
        seg_ids = (
            region.pre_seg_id.astype(np.int64).copy()
            if region.pre_seg_id is not None
            else np.zeros(len(pos), dtype=np.int64)
        )
        labels = region.pre_root_id.astype(np.int64).copy()
    else:
        pos = region.post_pt_nm.astype(np.float32)
        seg_ids = (
            region.post_seg_id.astype(np.int64).copy()
            if region.post_seg_id is not None
            else np.zeros(len(pos), dtype=np.int64)
        )
        labels = region.post_root_id.astype(np.int64).copy()

    N = len(pos)

    # -----------------------------------------------------------------
    # DNA lookup: base_root_id (seg-version) → dna embedding
    # -----------------------------------------------------------------
    dna_lookup: dict[int, np.ndarray] = {}
    dna_dim = 0
    for frag in fragments:
        if frag.dna is not None:
            dna_lookup[int(frag.base_root_id)] = frag.dna
            if dna_dim == 0:
                dna_dim = len(frag.dna)

    node_dna = np.zeros((N, dna_dim), dtype=np.float32)
    for i in range(N):
        sid = int(seg_ids[i])
        if sid in dna_lookup:
            node_dna[i] = dna_lookup[sid]

    # Precompute L2-normalised DNA for cosine similarity in edge features.
    if dna_dim > 0:
        norms = np.linalg.norm(node_dna, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        node_dna_norm = node_dna / norms
    else:
        node_dna_norm = np.zeros((N, 0), dtype=np.float32)

    # -----------------------------------------------------------------
    # Node features: [normalised_position (3), seg_dna (D)]
    # -----------------------------------------------------------------
    pos_norm = pos / pos_scale_nm
    node_feat = np.concatenate([pos_norm, node_dna], axis=1).astype(np.float32)

    # -----------------------------------------------------------------
    # Same-segment edges (edge type 0)
    # -----------------------------------------------------------------
    seg_groups: dict[int, list[int]] = {}
    for i in range(N):
        sid = int(seg_ids[i])
        if sid != 0:
            seg_groups.setdefault(sid, []).append(i)

    same_seg_src: list[int] = []
    same_seg_dst: list[int] = []

    for sid, idxs in seg_groups.items():
        g = len(idxs)
        if g < 2:
            continue
        pairs = [(idxs[a], idxs[b]) for a in range(g) for b in range(a + 1, g)]
        if len(pairs) > max_same_seg_pairs:
            rng_ss = np.random.default_rng(sid % (2**32))
            sel = rng_ss.choice(len(pairs), max_same_seg_pairs, replace=False)
            pairs = [pairs[int(k)] for k in sel]
        for u, v in pairs:
            same_seg_src.extend([u, v])
            same_seg_dst.extend([v, u])

    # -----------------------------------------------------------------
    # Spatial k-NN edges (edge type 1)
    # -----------------------------------------------------------------
    k = min(k_spatial + 1, N)
    tree = cKDTree(pos)
    dists, nbr_idxs = tree.query(pos, k=k)

    spatial_src: list[int] = []
    spatial_dst: list[int] = []

    for i in range(N):
        for slot in range(1, k):
            j = int(nbr_idxs[i, slot])
            d = float(dists[i, slot])
            if max_dist_nm is not None and d > max_dist_nm:
                continue
            spatial_src.append(i)
            spatial_dst.append(j)

    # -----------------------------------------------------------------
    # Assemble edges + compute edge features
    # -----------------------------------------------------------------
    def _cos_sim(srcs: list[int], dsts: list[int]) -> np.ndarray:
        if dna_dim == 0 or len(srcs) == 0:
            return np.zeros(len(srcs), dtype=np.float32)
        s = np.array(srcs, dtype=np.int64)
        d = np.array(dsts, dtype=np.int64)
        return (node_dna_norm[s] * node_dna_norm[d]).sum(axis=1).astype(np.float32)

    n_ss = len(same_seg_src)
    n_sp = len(spatial_src)
    total = n_ss + n_sp

    if total == 0:
        edge_src = np.zeros(0, dtype=np.int64)
        edge_dst = np.zeros(0, dtype=np.int64)
        edge_type = np.zeros(0, dtype=np.int64)
        edge_feat = np.zeros((0, 3), dtype=np.float32)
    else:
        edge_src = np.array(same_seg_src + spatial_src, dtype=np.int64)
        edge_dst = np.array(same_seg_dst + spatial_dst, dtype=np.int64)
        edge_type = np.concatenate([
            np.zeros(n_ss, dtype=np.int64),
            np.ones(n_sp, dtype=np.int64),
        ])

        ss_cos = _cos_sim(same_seg_src, same_seg_dst)
        sp_cos = _cos_sim(spatial_src, spatial_dst)
        cos_sim_all = np.concatenate([ss_cos, sp_cos])

        type_onehot = np.column_stack([
            (edge_type == 0).astype(np.float32),
            (edge_type == 1).astype(np.float32),
        ])
        edge_feat = np.column_stack([type_onehot, cos_sim_all]).astype(np.float32)

    return HalfSynapseGraph(
        node_feat=node_feat,
        node_pos=pos,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_type=edge_type,
        edge_feat=edge_feat,
        labels=labels,
        seg_id=seg_ids,
        side=side,
    )
