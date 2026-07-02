"""Compartment labeling: the enriched alphabet for the compartment-augmented PCFG.

Assign every skeleton vertex a compartment symbol {SOMA, AXON, DEND, UNKNOWN},
derived (no training) from two sources the plan settled on:

* **soma**  — large-radius vertex clusters (``neuronauts.soma_clusters``) and,
  optionally, the nucleus/soma table.
* **axon vs dendrite** — synapse polarity.  For this neuron, its **pre**-synapses
  (it is presynaptic → output) mark **axonal** cable; its **post**-synapses
  (postsynaptic → input) mark **dendritic** cable.  Most vertices carry no
  synapse, so the sparse per-vertex pre/post counts are diffused along the
  skeleton tree (edge-length weighted) and each vertex takes the dominant sign.

The result feeds the grammar's productions: an edge whose two geodesic windows are
dominantly AXON vs DEND *without passing through a soma* is a candidate merge seam
(axon and dendrite both emanate from the soma, so a direct A–D fusion is illegal),
and >1 soma cluster is a multi-soma merge.

Units (the #1 footgun): skeleton vertices/radius are in nm; synapse ``pre_pt``/
``post_pt`` are MIP-``mip`` voxels → nm via ``MIP_VOXEL_SIZES[mip]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuronauts.fetch import MIP_VOXEL_SIZES
from neuronauts.soma_clusters import soma_clusters

SOMA, AXON, DEND, UNKNOWN = 0, 1, 2, 3
_NAMES = {SOMA: "soma", AXON: "axon", DEND: "dend", UNKNOWN: "unknown"}


@dataclass
class CompartmentLabels:
    root_id: int
    vertices_nm: np.ndarray      # [V, 3]
    edges: np.ndarray            # [E, 2]
    radius: np.ndarray | None    # [V] or None
    label: np.ndarray            # [V] int8 in {SOMA, AXON, DEND, UNKNOWN}
    axon_mass: np.ndarray        # [V] diffused pre (axonal) mass
    dend_mass: np.ndarray        # [V] diffused post (dendritic) mass
    pre_count: np.ndarray        # [V] raw pre-synapse count
    post_count: np.ndarray       # [V] raw post-synapse count
    soma_vertex_sets: list       # list of [k] int index arrays (one per soma cluster)

    @property
    def n_soma(self) -> int:
        return len(self.soma_vertex_sets)

    def label_names(self) -> list[str]:
        return [_NAMES[int(x)] for x in self.label]

    def summary(self) -> dict:
        V = len(self.label)
        counts = {n: int((self.label == c).sum()) for c, n in _NAMES.items()}
        return {"root_id": self.root_id, "n_vertices": V,
                "n_soma_clusters": self.n_soma,
                **{f"n_{k}_verts": v for k, v in counts.items()}}


def _build_weighted_adjacency(vertices_nm, edges, *, lam_nm: float):
    """Row-normalized sparse adjacency with weight exp(-edge_len / lam)."""
    from scipy import sparse

    V = len(vertices_nm)
    if len(edges) == 0:
        return sparse.csr_matrix((V, V))
    e0, e1 = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    length = np.linalg.norm(vertices_nm[e0] - vertices_nm[e1], axis=1)
    w = np.exp(-length / lam_nm)
    rows = np.concatenate([e0, e1])
    cols = np.concatenate([e1, e0])
    data = np.concatenate([w, w])
    A = sparse.coo_matrix((data, (rows, cols)), shape=(V, V)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    Dinv = sparse.diags(1.0 / deg)
    return Dinv @ A


def _diffuse(sources: np.ndarray, W, *, beta: float, iters: int) -> np.ndarray:
    """Source-reinjecting diffusion: steady state of m = beta*sources + (1-beta)*W m.

    beta controls locality (higher = stays near sources); iters is the number of
    Jacobi sweeps.  Keeps mass anchored to the synapse-bearing vertices instead of
    washing out to a global average.
    """
    m = sources.astype(np.float64).copy()
    for _ in range(iters):
        m = beta * sources + (1.0 - beta) * (W @ m)
    return m


def _snap_counts(vertices_nm, points_nm, *, cap_nm: float) -> np.ndarray:
    """Accumulate, per vertex, the number of points whose nearest vertex is it
    (within cap_nm).  Points beyond the cap are dropped (off-skeleton clutter)."""
    from scipy.spatial import cKDTree

    counts = np.zeros(len(vertices_nm), np.float64)
    if len(points_nm) == 0:
        return counts
    dist, idx = cKDTree(vertices_nm).query(points_nm, k=1)
    ok = dist <= cap_nm
    np.add.at(counts, idx[ok], 1.0)
    return counts


def label_compartments(
    sk,
    syn,
    *,
    root_id: int | None = None,
    mip: int = 2,
    nucleus_pos_nm: np.ndarray | None = None,
    max_snap_nm: float = 1500.0,
    lam_nm: float = 5000.0,
    diffuse_beta: float = 0.15,
    diffuse_iters: int = 25,
    dominance: float = 0.60,
    nucleus_ball_nm: float = 5000.0,
) -> CompartmentLabels:
    """Label each vertex of skeleton ``sk`` using synapses ``syn`` + soma caliber.

    Parameters
    ----------
    sk : SkeletonData
        Vertices (nm), edges, radius (nm).
    syn : SynapseTable
        Synapses touching this root; ``pre_pt``/``post_pt`` in MIP-``mip`` voxels.
    root_id : int
        The neuron whose skeleton this is; selects which synapse side is axonal
        (pre_root_id == root_id) vs dendritic (post_root_id == root_id).  Defaults
        to ``sk.root_id``.
    nucleus_pos_nm : [3] or [M,3], optional
        Nucleus centroid(s) in nm; vertices within ``nucleus_ball_nm`` are marked soma.
    """
    root_id = int(root_id if root_id is not None else sk.root_id)
    V = np.asarray(sk.vertices, dtype=np.float64)
    edges = np.asarray(sk.edges, dtype=np.int64).reshape(-1, 2)
    radius = None if sk.radius is None else np.asarray(sk.radius, dtype=np.float64)
    nv = len(V)

    vox = np.asarray(MIP_VOXEL_SIZES[mip], dtype=np.float64)

    # --- synapse polarity -> nm, snapped to vertices -----------------------
    pre_count = np.zeros(nv)
    post_count = np.zeros(nv)
    if syn is not None and syn.n_synapses:
        pre_mask = np.asarray(syn.pre_root_id) == root_id     # this cell is PREsynaptic -> axon
        post_mask = np.asarray(syn.post_root_id) == root_id   # this cell is POSTsynaptic -> dend
        pre_nm = np.asarray(syn.pre_pt, dtype=np.float64)[pre_mask] * vox
        post_nm = np.asarray(syn.post_pt, dtype=np.float64)[post_mask] * vox
        pre_count = _snap_counts(V, pre_nm, cap_nm=max_snap_nm)
        post_count = _snap_counts(V, post_nm, cap_nm=max_snap_nm)

    # --- diffuse along the tree -------------------------------------------
    W = _build_weighted_adjacency(V, edges, lam_nm=lam_nm)
    axon_mass = _diffuse(pre_count, W, beta=diffuse_beta, iters=diffuse_iters)
    dend_mass = _diffuse(post_count, W, beta=diffuse_beta, iters=diffuse_iters)

    # --- soma seeding ------------------------------------------------------
    soma_sets = soma_clusters(V, radius)
    soma_mask = np.zeros(nv, bool)
    for s in soma_sets:
        soma_mask[s] = True
    if nucleus_pos_nm is not None and nv:
        from scipy.spatial import cKDTree

        nuc = np.atleast_2d(np.asarray(nucleus_pos_nm, dtype=np.float64))
        near = cKDTree(V).query_ball_point(nuc, r=nucleus_ball_nm)
        for idxs in np.atleast_1d(near):
            soma_mask[np.asarray(idxs, dtype=int)] = True

    # --- per-vertex label --------------------------------------------------
    label = np.full(nv, UNKNOWN, dtype=np.int8)
    total = axon_mass + dend_mass
    with np.errstate(invalid="ignore", divide="ignore"):
        axon_frac = np.where(total > 0, axon_mass / total, 0.0)
    has_signal = total > 0
    label[has_signal & (axon_frac >= dominance)] = AXON
    label[has_signal & (axon_frac <= 1.0 - dominance)] = DEND
    label[soma_mask] = SOMA  # soma overrides

    return CompartmentLabels(
        root_id=root_id, vertices_nm=V, edges=edges, radius=radius, label=label,
        axon_mass=axon_mass, dend_mass=dend_mass, pre_count=pre_count,
        post_count=post_count, soma_vertex_sets=soma_sets,
    )
