"""Skeleton-based grammar for neuron half-partitions.

Computes the same F/B/L/R bigram grammar as pcfg_partitions.py, but applies
the tokenization to the SKELETON vertex sequence instead of the synapse cloud.

Synapse grammar (extrinsic): PCA-orders synapses in the local bounding box.
  Encodes local circuit context -> changes across boxes -> cross-box AUC 0.51.

Skeleton grammar (intrinsic): DFS-traverses the skeleton from soma.
  Encodes the neuron's own morphology -> same neuron looks the same in any box.

Both grammars produce the same 17-dim feature (16 bigrams + entropy),
enabling direct numerical comparison.

Pre vs post: pass side='pre' or side='post' to restrict to vertices near the
axonal or dendritic synapse positions.  Default side='both' uses full skeleton.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from experiments.pcfg.pcfg_partitions import (
    FEAT_DIM,
    PAIR_DIM,
    bigram_features,
    cond_entropy,
    root_groups,
    tokenize,
)


@dataclass
class SkeletonPartition:
    """One root's skeleton + synapse positions + GT label.

    Mirrors HalfPartition so merge-pair construction and cross-box analysis
    can be adapted with minimal changes.
    """
    root_id: int        # v117 root ID
    v18xx_root: int     # remapped v1718 GT root ID
    side: str           # 'pre' | 'post' | 'both'
    pts: np.ndarray     # (N_syn, 3) synapse positions nm -- for centroid distance
    skel_verts: np.ndarray   # (V, 3) float64 skeleton vertex positions nm
    skel_edges: np.ndarray   # (E, 2) int64 edge index pairs


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _build_adj(edges: np.ndarray, n: int) -> list:
    adj = [[] for _ in range(n)]
    for row in edges.tolist():
        u, v = int(row[0]), int(row[1])
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _find_soma(vertices: np.ndarray, radius) -> int:
    if radius is not None and len(radius) == len(vertices) and float(radius.max()) > 0:
        return int(np.argmax(radius))
    centroid = vertices.mean(axis=0)
    return int(np.argmin(np.linalg.norm(vertices - centroid, axis=1)))


def _dfs_order(adj: list, root: int) -> list:
    n = len(adj)
    visited = [False] * n
    order = []
    stack = [root]
    while stack:
        v = stack.pop()
        if visited[v]:
            continue
        visited[v] = True
        order.append(v)
        for u in reversed(adj[v]):
            if not visited[u]:
                stack.append(u)
    return order


def _restrict_near_synapses(vertices: np.ndarray, order: list,
                             syn_pts: np.ndarray, radius_nm: float = 5000.0) -> list:
    """Keep only vertices within radius_nm of any synapse -- isolates one compartment."""
    if len(syn_pts) == 0 or len(order) < 2:
        return order
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(syn_pts)
        dists, _ = tree.query(vertices[order], k=1, workers=1)
        return [order[i] for i, d in enumerate(dists.tolist()) if d <= radius_nm]
    except ImportError:
        return order


# ---------------------------------------------------------------------------
# Grammar features
# ---------------------------------------------------------------------------

def skeleton_features(sp: SkeletonPartition, *, radius=None,
                       side_radius_nm: float = 5000.0) -> np.ndarray:
    """17-dim skeleton grammar: 16 bigrams + conditional entropy.

    Uses DFS-ordered skeleton vertices as the token sequence, applying the
    same tokenize() -> bigram_features() -> cond_entropy() pipeline as the
    synapse grammar.  Returns zeros if skeleton is too small.
    """
    verts = sp.skel_verts.astype(np.float64)
    edges = sp.skel_edges
    if len(verts) < 3 or len(edges) == 0:
        return np.zeros(FEAT_DIM, dtype=np.float64)

    adj = _build_adj(edges, len(verts))
    soma = _find_soma(verts, radius)
    order = _dfs_order(adj, soma)

    if sp.side != 'both':
        order = _restrict_near_synapses(verts, order, sp.pts, side_radius_nm)

    if len(order) < 3:
        return np.zeros(FEAT_DIM, dtype=np.float64)

    tokens = tokenize(verts[order])
    if len(tokens) < 2:
        return np.zeros(FEAT_DIM, dtype=np.float64)

    return np.append(bigram_features(tokens), cond_entropy(tokens))


# ---------------------------------------------------------------------------
# Partition extraction
# ---------------------------------------------------------------------------

def extract_skeleton_partitions(pre_pt, post_pt, pre_root_id, post_root_id,
                                  root_remap, skeletons, *,
                                  min_synapses=4, sides='both'):
    """Build SkeletonPartition list from a synapse table + fetched skeletons.

    Drops roots with missing/empty skeletons or no valid v1718 label.
    skeletons: dict[int, SkeletonData] from fetch_root_skeletons().
    """
    partitions = []
    candidates = []
    if sides in ('pre', 'both'):
        candidates.append(('pre', pre_pt, pre_root_id))
    if sides in ('post', 'both'):
        candidates.append(('post', post_pt, post_root_id))

    for side, pts, root_ids in candidates:
        for rid, indices in root_groups(root_ids).items():
            if rid == 0 or len(indices) < min_synapses:
                continue
            target = root_remap.get(rid, 0)
            if target == 0:
                continue
            sk = skeletons.get(rid)
            if sk is None or len(sk.vertices) < 3:
                continue
            partitions.append(SkeletonPartition(
                root_id=rid,
                v18xx_root=target,
                side=side,
                pts=pts[indices].astype(np.float64),
                skel_verts=sk.vertices.astype(np.float64),
                skel_edges=sk.edges.astype(np.int64),
            ))
    return partitions


# ---------------------------------------------------------------------------
# Merge pair construction  (mirrors build_merge_pairs)
# ---------------------------------------------------------------------------

def build_skeleton_merge_pairs(partitions, *, max_neg_ratio=3.0, rng=None,
                                 radii=None):
    """Build (X, y) from skeleton grammar features.

    Same 35-dim layout: feat_a(17) + feat_b(17) + log_dist(1).
    Positives = different v117 roots -> same v1718 root.
    Negatives = KD-tree nearest different-neuron pairs.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    _radii = radii or {}

    feats = [skeleton_features(p, radius=_radii.get(p.root_id)) for p in partitions]
    centroids = np.array([p.pts.mean(axis=0) for p in partitions], dtype=np.float64)
    v18xx = [p.v18xx_root for p in partitions]
    root_ids = [p.root_id for p in partitions]

    # Positives
    pos_rows = []
    by_v18 = defaultdict(list)
    for i, v in enumerate(v18xx):
        by_v18[v].append(i)
    for group in by_v18.values():
        if len(group) < 2:
            continue
        for i, j in combinations(group, 2):
            if root_ids[i] == root_ids[j]:
                continue
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            pos_rows.append((feats[i], feats[j], dist, 1))

    if not pos_rows:
        return np.zeros((0, PAIR_DIM), dtype=np.float64), np.zeros(0, dtype=np.int64)

    n_neg_target = max(1, int(len(pos_rows) * max_neg_ratio))

    # Negatives: KD-tree k-NN
    neg_rows = []
    try:
        from scipy.spatial import cKDTree
        k = min(51, len(partitions))
        tree = cKDTree(centroids)
        dists_nn, idx_nn = tree.query(centroids, k=k, workers=-1)
        seen = set()
        for i in rng.permutation(len(partitions)):
            for slot in range(1, k):
                j = int(idx_nn[i, slot])
                pair = (min(i, j), max(i, j))
                if pair in seen:
                    continue
                if root_ids[i] == root_ids[j] or v18xx[i] == v18xx[j]:
                    continue
                seen.add(pair)
                neg_rows.append((feats[i], feats[j], float(dists_nn[i, slot]), 0))
                if len(neg_rows) >= n_neg_target * 5:
                    break
            if len(neg_rows) >= n_neg_target * 5:
                break
    except ImportError:
        pass

    neg_rows.sort(key=lambda r: r[2])
    neg_rows = neg_rows[:n_neg_target]

    all_rows = pos_rows + neg_rows
    shuffled = [all_rows[k] for k in rng.permutation(len(all_rows))]
    X = np.array([np.concatenate([fa, fb, [np.log1p(d)]]) for fa, fb, d, _ in shuffled],
                 dtype=np.float64)
    y = np.array([lbl for _, _, _, lbl in shuffled], dtype=np.int64)
    return X, y
