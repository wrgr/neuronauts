"""Compartment-augmented grammar: productions that flag false merges + locate the seam.

Operates on a `CompartmentLabels` object (see ``compartments.py``) and produces:

* **MULTI_SOMA**  — >1 soma-caliber cluster in one object (reuses the verified
  ``soma_clusters`` count).  Two cells merged → two somas.
* **AXON↔DENDRITE crossing** — a skeleton edge whose two sides are dominantly
  different compartments (axon vs dendrite) **not mediated by a soma**.  Axon and
  dendrite both emanate from the soma, so a direct A–D fusion away from the soma
  is a merge seam.  We locate the offending edge.

These are combined into a per-object ``merge_score`` and a candidate seam edge.
SegCLR discordance across the seam can be added as a corroborating term (see
``object_signals(..., segclr=...)``); the grammar is fully functional without it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.pcfg.compartments import CompartmentLabels, SOMA, AXON, DEND


@dataclass
class ObjectSignals:
    root_id: int
    n_vertices: int
    cable_um: float
    n_soma: int
    ad_score: float                 # best guarded A↔D crossing contrast (0..1)
    ad_edge: tuple | None           # (u, v) offending skeleton edge
    ad_soma_dist_um: float          # geodesic dist of that edge to nearest soma
    segclr_discord: float | None    # 1 - cos across the seam (None if unavailable)
    features: dict = field(default_factory=dict)

    def merge_score(self, w=None) -> float:
        """Transparent combination (default weights); a fitted combiner can replace this."""
        w = w or {"soma": 1.0, "ad": 1.0, "seg": 0.5}
        s = w["soma"] * min(max(self.n_soma - 1, 0), 3) + w["ad"] * self.ad_score
        if self.segclr_discord is not None:
            s += w["seg"] * self.segclr_discord
        return float(s)


MERGED_ID = -999  # sentinel root id for a synthetic merged object


def _edge_lengths(V, edges):
    return np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)


def extract_axon_piece(skB, synB, rB, *, min_vertices=80):
    """Extract cell B's largest connected AXON subtree (+ its pre-synapses) as a
    standalone piece — for grafting onto another cell's dendrite to synthesize a
    realistic 1-soma false merge (an axon fragment on the wrong cell)."""
    from types import SimpleNamespace
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    from experiments.pcfg.compartments import label_compartments, AXON

    lab = label_compartments(skB, synB, root_id=rB, mip=2)
    axon = lab.label == AXON
    if axon.sum() < min_vertices:
        return None
    V = np.asarray(skB.vertices, float)
    E = np.asarray(skB.edges, np.int64).reshape(-1, 2)
    keep = axon[E[:, 0]] & axon[E[:, 1]]
    sub = E[keep]
    if len(sub) < min_vertices:
        return None
    nv = len(V)
    g = csr_matrix((np.ones(len(sub) * 2),
                    (np.concatenate([sub[:, 0], sub[:, 1]]),
                     np.concatenate([sub[:, 1], sub[:, 0]]))), shape=(nv, nv))
    ncc, comp = connected_components(g, directed=False)
    # largest axon component
    axon_idx = np.where(axon)[0]
    vals, cnts = np.unique(comp[axon_idx], return_counts=True)
    big = vals[np.argmax(cnts)]
    sel = np.where(comp == big)[0]
    if len(sel) < min_vertices:
        return None
    remap = -np.ones(nv, int); remap[sel] = np.arange(len(sel))
    subedges = np.array([[remap[a], remap[b]] for a, b in sub
                         if comp[a] == big and comp[b] == big], int)
    subV = V[sel]
    subR = (skB.radius[sel] if skB.radius is not None else np.full(len(sel), np.nan))
    subsk = SimpleNamespace(root_id=rB, vertices=subV, edges=subedges, radius=subR)

    # pre-synapses of B that snap to the axon piece
    from scipy.spatial import cKDTree
    vox = np.array([32.0, 32.0, 40.0])
    pre_mask = np.asarray(synB.pre_root_id) == rB
    pre_nm = np.asarray(synB.pre_pt, float)[pre_mask] * vox
    if len(pre_nm):
        d, _ = cKDTree(subV).query(pre_nm, k=1)
        pre_nm = pre_nm[d <= 1500.0]
    subsyn = SimpleNamespace(
        n_synapses=len(pre_nm),
        pre_pt=pre_nm / vox if len(pre_nm) else np.zeros((0, 3)),
        post_pt=np.zeros((len(pre_nm), 3)),
        pre_root_id=np.array([rB] * len(pre_nm)),
        post_root_id=np.array([0] * len(pre_nm)),
    )
    return subsk, subsyn


def build_merged_object(skA, synA, rA, skB, synB, rB):
    """Bridge two neurons into one synthetic merged object (a false merge).

    Concatenates the two skeletons and joins them with a single nearest-vertex
    bridge edge (so it isn't a trivially-disconnected giveaway), and builds a
    combined synapse table whose axonal (pre) / dendritic (post) sides are the
    union of both cells' — labeled with the sentinel ``MERGED_ID``.

    Returns ``(sk, syn, MERGED_ID)`` as SimpleNamespaces usable by
    ``label_compartments``.  The true seam is the bridge edge.
    """
    from types import SimpleNamespace
    from scipy.spatial import cKDTree

    Va = np.asarray(skA.vertices, float); Vb = np.asarray(skB.vertices, float)
    Ea = np.asarray(skA.edges, np.int64).reshape(-1, 2)
    Eb = np.asarray(skB.edges, np.int64).reshape(-1, 2) + len(Va)
    V = np.vstack([Va, Vb])
    # nearest cross pair -> one bridge edge
    d, i = cKDTree(Va).query(Vb, k=1)
    j = int(np.argmin(d)); ia = int(i[j]); ib = int(j) + len(Va)
    bridge = np.array([[ia, ib]], np.int64)
    edges = np.vstack([Ea, Eb, bridge])
    ra = skA.radius if skA.radius is not None else np.full(len(Va), np.nan)
    rb = skB.radius if skB.radius is not None else np.full(len(Vb), np.nan)
    radius = np.concatenate([np.asarray(ra, float), np.asarray(rb, float)])
    sk = SimpleNamespace(root_id=MERGED_ID, vertices=V, edges=edges, radius=radius)

    def pre_pts(syn, r):
        m = np.asarray(syn.pre_root_id) == r
        return np.asarray(syn.pre_pt, float)[m]

    def post_pts(syn, r):
        m = np.asarray(syn.post_root_id) == r
        return np.asarray(syn.post_pt, float)[m]

    pre = np.vstack([pre_pts(synA, rA), pre_pts(synB, rB)]) if (synA.n_synapses or synB.n_synapses) else np.zeros((0, 3))
    post = np.vstack([post_pts(synA, rA), post_pts(synB, rB)]) if (synA.n_synapses or synB.n_synapses) else np.zeros((0, 3))
    npre, npost = len(pre), len(post)
    syn = SimpleNamespace(
        n_synapses=npre + npost,
        pre_pt=np.vstack([pre, np.zeros((npost, 3))]),
        post_pt=np.vstack([np.zeros((npre, 3)), post]),
        pre_root_id=np.array([MERGED_ID] * npre + [0] * npost),
        post_root_id=np.array([0] * npre + [MERGED_ID] * npost),
    )
    return sk, syn, MERGED_ID


def geodesic_to_soma(labels: CompartmentLabels) -> np.ndarray:
    """Geodesic distance (nm) from each vertex to the nearest soma vertex.

    inf where there is no soma or a vertex is in a soma-free component.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra

    V = labels.vertices_nm
    edges = labels.edges
    nv = len(V)
    soma_idx = np.concatenate(labels.soma_vertex_sets) if labels.soma_vertex_sets else np.array([], int)
    if len(edges) == 0 or len(soma_idx) == 0:
        return np.full(nv, np.inf)
    length = _edge_lengths(V, edges)
    g = sparse.csr_matrix(
        (np.concatenate([length, length]),
         (np.concatenate([edges[:, 0], edges[:, 1]]),
          np.concatenate([edges[:, 1], edges[:, 0]]))),
        shape=(nv, nv))
    d = dijkstra(g, directed=False, indices=soma_idx, min_only=True)
    return np.asarray(d)


def detect_ad_crossings(
    labels: CompartmentLabels,
    *,
    dominance: float = 0.60,
    min_soma_dist_nm: float = 15_000.0,
    min_mass: float | None = None,
) -> list[tuple[tuple, float, float]]:
    """Find A↔D crossing edges: one side **strongly** axon, the other **strongly**
    dendrite, both far (> ``min_soma_dist_nm``) from any soma (not soma-mediated).

    Score = ``min(axon_mass on the axon side, dend_mass on the dend side)`` — a real
    graft has *both* a strong axon and a strong dendrite signal abutting; a clean
    neuron does not (its only A–D junction is at the soma, which is guarded).  This
    absolute-mass requirement (not just a fraction) suppresses the low-signal
    fraction noise that fires on clean distal cable.

    Returns ``[((u, v), score, soma_dist_nm), ...]`` sorted by score desc.
    """
    V = labels.vertices_nm
    edges = labels.edges
    if len(edges) == 0:
        return []
    am, dm = labels.axon_mass, labels.dend_mass
    total = am + dm
    with np.errstate(invalid="ignore", divide="ignore"):
        axon_frac = np.where(total > 0, am / total, np.nan)
    if min_mass is None:
        nz = total[total > 0]
        min_mass = float(np.median(nz)) if len(nz) else 0.0
    dsoma = geodesic_to_soma(labels)

    out = []
    for u, v in edges:
        fu, fv = axon_frac[u], axon_frac[v]
        if np.isnan(fu) or np.isnan(fv):
            continue
        s = 0.0
        if fu >= dominance and fv <= 1 - dominance:          # u axon, v dend
            s = min(am[u], dm[v])
        elif fv >= dominance and fu <= 1 - dominance:        # v axon, u dend
            s = min(am[v], dm[u])
        if s < min_mass:
            continue  # need real, opposite signal on BOTH sides
        edge_soma_dist = float(min(dsoma[u], dsoma[v]))
        if edge_soma_dist < min_soma_dist_nm:
            continue  # soma-mediated -> legal (axon & dendrite meet at the soma)
        out.append(((int(u), int(v)), float(s), edge_soma_dist))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def object_signals(
    labels: CompartmentLabels,
    *,
    segclr=None,
    seg_window_nm: float = 8000.0,
    dominance: float = 0.60,
    min_soma_dist_nm: float = 15_000.0,
) -> ObjectSignals:
    """Compute the per-object grammar signals + candidate seam edge."""
    V = labels.vertices_nm
    edges = labels.edges
    cable_um = float(_edge_lengths(V, edges).sum() / 1000.0) if len(edges) else 0.0
    n_soma = labels.n_soma

    crossings = detect_ad_crossings(labels, dominance=dominance,
                                    min_soma_dist_nm=min_soma_dist_nm)
    ad_score = crossings[0][1] if crossings else 0.0
    ad_edge = crossings[0][0] if crossings else None
    ad_soma_dist = crossings[0][2] if crossings else float("inf")

    segclr_discord = None
    if segclr is not None and ad_edge is not None and getattr(segclr, "covered", None) is not None:
        segclr_discord = _segclr_discord_at_edge(V, ad_edge, segclr, seg_window_nm)

    feats = {
        "n_soma": float(n_soma),
        "ad_score": float(ad_score),
        "ad_soma_dist_um": float(ad_soma_dist / 1000.0) if np.isfinite(ad_soma_dist) else 0.0,
        "cable_um": cable_um,
        "log_nv": float(np.log1p(len(V))),
        "n_ad_crossings": float(len(crossings)),
        "segclr_discord": float(segclr_discord) if segclr_discord is not None else 0.0,
    }
    return ObjectSignals(
        root_id=labels.root_id, n_vertices=len(V), cable_um=cable_um, n_soma=n_soma,
        ad_score=ad_score, ad_edge=ad_edge,
        ad_soma_dist_um=float(ad_soma_dist / 1000.0) if np.isfinite(ad_soma_dist) else 0.0,
        segclr_discord=segclr_discord, features=feats,
    )


def _segclr_discord_at_edge(V, edge, segclr, window_nm) -> float | None:
    """Pool SegCLR embeddings of covered vertices within ``window_nm`` (spatial) of
    each edge endpoint; return 1 - cos(poolA, poolB), or None if coverage too low."""
    from scipy.spatial import cKDTree

    covered = segclr.covered
    emb = segclr.embedding
    if covered.sum() < 4:
        return None
    tree = cKDTree(V)
    a = tree.query_ball_point(V[edge[0]], window_nm)
    b = tree.query_ball_point(V[edge[1]], window_nm)
    a = [i for i in a if covered[i]]
    b = [i for i in b if covered[i]]
    if len(a) < 2 or len(b) < 2:
        return None
    pa = emb[a].mean(0)
    pb = emb[b].mean(0)
    cos = float(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-9))
    return 1.0 - cos
