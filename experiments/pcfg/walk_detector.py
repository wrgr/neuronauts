"""Walk-based SegCLR detector for BOTH proofreading error types.

Lay the SegCLR embeddings along the skeleton and treat them as a 1-D identity
signal.  A rolling average denoises the per-node jitter (a single node is only
~87% self-consistent); the *step* statistic — cosine distance between the rolling
mean just before vs just after a point — spikes where local identity changes.

* **Merge error** (two cells fused): the step spikes mid-cable.  A clean neuron's
  only legitimate identity shift is at the **soma** (axon meets dendrite), so we
  *gate* the step score by geodesic distance to the soma: a spike far from any
  soma = a merge seam → cut there.

* **Split error** (one cell broken into fragments): a fragment's trace should
  *continue* across the gap into a neighbouring fragment.  At each fragment
  endpoint we look at nearby endpoints of other fragments and pick the one whose
  local SegCLR trace best matches (top-1 cosine) — the true continuation → join.

SegCLR is keyed to the m343 segmentation, so a current neuron spans many m343
fragments; we enumerate them with a sparse ``seg_m343`` volume lookup (cached per
root) and assign their embeddings to skeleton vertices spatially.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from neuronauts.segclr import assign_points_to_vertices, SegCLRAssignment


# --------------------------------------------------------------------------- #
# SegCLR -> skeleton assignment (via seg_m343 fragment enumeration, cached)
# --------------------------------------------------------------------------- #

def neuron_fragment_ids(root_id, vertices_nm, cv, *, cache_dir="cache/m343_frags",
                        n_probe=400, min_hits=1, rng=None) -> np.ndarray:
    """Enumerate the m343 segment ids covering a neuron by sampling ``n_probe``
    skeleton vertices in the ``seg_m343`` volume.  Cached per root.

    A current neuron spans ~40-90 m343 fragments, so dense probing is needed for
    good SegCLR coverage of the skeleton."""
    os.makedirs(cache_dir, exist_ok=True)
    cp = os.path.join(cache_dir, f"{int(root_id)}.npy")
    if os.path.exists(cp):
        return np.load(cp)
    rng = rng or np.random.default_rng(0)
    res = np.array(cv.resolution)
    V = np.asarray(vertices_nm, float)
    idx = rng.choice(len(V), min(n_probe, len(V)), replace=False)
    ids = []
    for p in V[idx]:
        vox = np.round(p / res).astype(int)
        try:
            ids.append(int(cv[vox[0], vox[1], vox[2]][0, 0, 0, 0]))
        except Exception:
            ids.append(0)
    ids = np.array(ids)
    u, ct = np.unique(ids[ids > 0], return_counts=True)
    frags = u[ct >= min_hits].astype(np.int64)
    np.save(cp, frags)
    return frags


def assign_segclr_to_skeleton(vertices_nm, frag_ids, reader, *, max_dist_nm=2000.0):
    """Fetch SegCLR for the given m343 fragments and assign to skeleton vertices."""
    pts_list, emb_list = [], []
    for fid in frag_ids:
        p, e = reader.read_segment(int(fid))
        if len(p):
            pts_list.append(p); emb_list.append(e)
    if not pts_list:
        V = len(vertices_nm)
        return SegCLRAssignment(np.asarray(vertices_nm), np.full((V, 64), np.nan, np.float32),
                                np.zeros(V, bool), 0.0, 0, "nm_coord")
    return assign_points_to_vertices(vertices_nm, np.vstack(pts_list),
                                     np.vstack(emb_list).astype(np.float32),
                                     max_dist_nm=max_dist_nm)


# --------------------------------------------------------------------------- #
# Tree walk
# --------------------------------------------------------------------------- #

def _adjacency(nv, edges):
    adj = [[] for _ in range(nv)]
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    return adj


def soma_to_tip_paths(vertices_nm, edges, root_vertex) -> list[np.ndarray]:
    """Root the tree at ``root_vertex`` (soma) and return every soma→leaf path as a
    vertex-index array.  (BFS parents; each degree-1 leaf yields one path.)"""
    nv = len(vertices_nm)
    adj = _adjacency(nv, edges)
    parent = np.full(nv, -1, int)
    seen = np.zeros(nv, bool)
    order = []
    stack = [root_vertex]; seen[root_vertex] = True
    while stack:
        u = stack.pop(); order.append(u)
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True; parent[v] = u; stack.append(v)
    leaves = [i for i in range(nv) if len(adj[i]) == 1 and i != root_vertex and seen[i]]
    paths = []
    for lf in leaves:
        p = [lf]; cur = lf
        while parent[cur] != -1:
            cur = parent[cur]; p.append(cur)
        paths.append(np.array(p[::-1]))  # soma -> tip
    return paths


def _rolling_step(embn, W):
    """cosine distance between mean(prev W) and mean(next W) at each path index."""
    n = len(embn); out = np.zeros(n)
    for i in range(n):
        a = embn[max(0, i - W):i]; b = embn[i:i + W]
        if len(a) < 2 or len(b) < 2:
            continue
        ma = a.mean(0); mb = b.mean(0)
        out[i] = 1 - float(ma @ mb / (np.linalg.norm(ma) * np.linalg.norm(mb) + 1e-9))
    return out


@dataclass
class WalkResult:
    step_score: np.ndarray       # [V] max gated step score seen at each vertex
    best_vertex: int             # vertex of the global max gated step
    best_score: float
    soma_dist_nm: np.ndarray     # [V]


def walk_merge_score(labels, seg_asg, *, W=6, min_soma_dist_nm=15000.0) -> WalkResult:
    """Per-vertex gated rolling-average step score along soma→tip paths.

    Only covered vertices contribute embeddings; the score at a vertex is gated to
    0 within ``min_soma_dist_nm`` of a soma (legal axon↔dendrite junction)."""
    from experiments.pcfg.compartment_grammar import geodesic_to_soma

    V = labels.vertices_nm
    edges = labels.edges
    nv = len(V)
    dsoma = geodesic_to_soma(labels)
    # root at the largest soma vertex, else max-radius vertex, else vertex 0
    if labels.soma_vertex_sets:
        root = int(labels.soma_vertex_sets[0][0])
    elif labels.radius is not None and np.isfinite(labels.radius).any():
        root = int(np.nanargmax(labels.radius))
    else:
        root = 0

    emb = seg_asg.embedding
    covered = seg_asg.covered
    embn = np.where(covered[:, None], emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9), 0.0)

    score = np.zeros(nv)
    for path in soma_to_tip_paths(V, edges, root):
        cov = covered[path]
        if cov.sum() < 2 * W:
            continue
        # step score computed on the covered subsequence, mapped back to vertices
        sub = path[cov]
        st = _rolling_step(embn[sub], W)
        # gate by soma distance
        gated = np.where(dsoma[sub] >= min_soma_dist_nm, st, 0.0)
        score[sub] = np.maximum(score[sub], gated)

    best = int(np.argmax(score)) if nv else 0
    return WalkResult(step_score=score, best_vertex=best,
                      best_score=float(score[best]) if nv else 0.0, soma_dist_nm=dsoma)


# --------------------------------------------------------------------------- #
# Split fixing: stitch fragments by SegCLR continuation (top-1, comparative)
# --------------------------------------------------------------------------- #

def fragment_contact_score(pa, ea, pb, eb, *, contact_nm=3000.0, k=5):
    """Join score for two fragment point clouds: mean cosine of the ``k`` nearest
    cross-fragment point pairs' (L2-normalised) embeddings.  ``None`` if the
    fragments never come within ``contact_nm``.

    Use this **comparatively** (each fragment's *highest*-scoring neighbour is its
    continuation) — the absolute value barely separates same- vs different-cell
    contacts (both ~0.92), but the top-1 ranking is ~0.9 correct.
    """
    from scipy.spatial import cKDTree

    d, idx = cKDTree(pb).query(pa, k=1)
    near = d <= contact_nm
    if near.sum() == 0:
        return None
    order = np.argsort(d[near])
    ai = np.where(near)[0][order][:k]
    bi = idx[near][order][:k]
    ean = ea / (np.linalg.norm(ea, axis=1, keepdims=True) + 1e-9)
    ebn = eb / (np.linalg.norm(eb, axis=1, keepdims=True) + 1e-9)
    return float(np.einsum("ij,ij->i", ean[ai], ebn[bi]).mean())


def stitch_fragments(fragments, *, contact_nm=3000.0, min_score=None):
    """Greedy top-1 fragment stitching.  ``fragments`` = list of ``(id, points_nm,
    embeddings)``.  Returns a list of proposed join edges ``(id_a, id_b, score)``:
    each fragment is joined to its single highest-scoring contacting neighbour
    (optionally thresholded by ``min_score``).  Comparative, so it does not rely on
    an absolute similarity cutoff."""
    joins = []
    for i, (ida, pa, ea) in enumerate(fragments):
        best = None
        for j, (idb, pb, eb) in enumerate(fragments):
            if i == j:
                continue
            s = fragment_contact_score(pa, ea, pb, eb, contact_nm=contact_nm)
            if s is not None and (best is None or s > best[1]):
                best = (idb, s)
        if best is not None and (min_score is None or best[1] >= min_score):
            joins.append((ida, best[0], best[1]))
    return joins
