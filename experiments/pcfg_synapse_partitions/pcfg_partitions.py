"""PCFG synapse partition grammar.

Berlin-style bigram grammar over synapse half-partitions, ported to the
neuronauts synapse domain.  No torch dependency — numpy + collections only.

Analogy to Berlin grammar_probe.py
-----------------------------------
  Berlin tokens   : N (navigate)  S (segment)  A (annotate)  O (other)
  Synapse tokens  : F (forward)   B (backward) L (lateral-L) R (lateral-R)

  Berlin features : bigram(16) + cond_entropy(1) per proofreader session
  Here            : bigram(16) + cond_entropy(1) per synapse half-partition

  Berlin positive : expert proofreader (manual label, n~15)
  Here positive   : two v117 roots that map to the same v18xx root (GT merge)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

# ---------------------------------------------------------------------------
# Alphabet
# ---------------------------------------------------------------------------
# 4 direction tokens derived from step direction in PCA-aligned space:
#   F = Forward   — step projects along PCA1, |c1| > threshold and c1 > 0
#   B = Backward  — step projects against PCA1, |c1| > threshold and c1 ≤ 0
#   L = Lateral-L — |c1| ≤ threshold, PCA2 component ≥ 0
#   R = Lateral-R — |c1| ≤ threshold, PCA2 component < 0
ALPH: list[str] = ['F', 'B', 'L', 'R']
_ALPH_IDX: dict[str, int] = {c: i for i, c in enumerate(ALPH)}
N_ALPH: int = len(ALPH)
BIGRAM_DIM: int = N_ALPH * N_ALPH  # 16
FEAT_DIM: int = BIGRAM_DIM + 1     # 17 (bigram + entropy)
PAIR_DIM: int = FEAT_DIM * 2 + 1   # 35 (feat_a + feat_b + log_dist)


@dataclass
class HalfPartition:
    """All synapses on one side (pre or post) of a single v117 root ID."""

    root_id: int        # v117 root ID
    v18xx_root: int     # remapped v18xx root ID — ground-truth label
    pts: np.ndarray     # (N, 3) float64 synapse positions in nm
    side: str           # 'pre' | 'post'


# ---------------------------------------------------------------------------
# Low-level spatial helpers
# ---------------------------------------------------------------------------

def root_groups(root_ids: np.ndarray) -> dict[int, list[int]]:
    """Group synapse indices by root ID."""
    groups: dict[int, list[int]] = {}
    for idx, rid in enumerate(root_ids.tolist()):
        groups.setdefault(int(rid), []).append(idx)
    return groups


def ordered_pts(
    pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA-order points along their main spatial axis.

    Returns
    -------
    pts_sorted : (N, 3) array ordered along PCA1 ascending
    pca1       : (3,) unit vector — first principal axis
    pca2       : (3,) unit vector — second principal axis
    """
    pts = pts.astype(np.float64)
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca1, pca2 = Vt[0], Vt[1]
    order = np.argsort(centered @ pca1)
    return pts[order], pca1, pca2


def tokenize(pts: np.ndarray, threshold: float = 0.4) -> list[str]:
    """Convert synapse positions to a direction-token sequence.

    Steps:
    1. PCA-order points along the main axis.
    2. Compute normalized step vectors between consecutive ordered synapses.
    3. Project each step onto PCA1 and PCA2.
    4. |c1| > threshold -> 'F' (c1 > 0) or 'B'; else -> 'L' (c2 >= 0) or 'R'.

    Returns [] when len(pts) < 2.
    """
    if len(pts) < 2:
        return []

    pts_sorted, pca1, pca2 = ordered_pts(pts)
    steps = np.diff(pts_sorted, axis=0)                     # (N-1, 3)
    norms = np.linalg.norm(steps, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    steps_norm = steps / norms

    c1 = steps_norm @ pca1  # projection onto main axis
    c2 = steps_norm @ pca2  # projection onto second axis

    tokens: list[str] = []
    for f, lat in zip(c1.tolist(), c2.tolist()):
        if abs(f) > threshold:
            tokens.append('F' if f > 0 else 'B')
        else:
            tokens.append('L' if lat >= 0 else 'R')
    return tokens


# ---------------------------------------------------------------------------
# Grammar features  (mirrors Berlin grammar_probe.py logic)
# ---------------------------------------------------------------------------

def bigram_features(tokens: list[str]) -> np.ndarray:
    """Normalized 16-dim bigram frequency vector.

    Identical to Berlin's ``ngram(a, n=2)`` computation.
    """
    cnt: Counter = Counter()
    for i in range(len(tokens) - 1):
        cnt[(tokens[i], tokens[i + 1])] += 1
    total = sum(cnt.values()) or 1
    return np.array(
        [cnt[(a, b)] / total for a, b in product(ALPH, repeat=2)],
        dtype=np.float64,
    )


def cond_entropy(tokens: list[str]) -> float:
    """H(next_token | current_token) over the token sequence.

    Identical to Berlin's ``cond_ent()`` computation.
    """
    nxt: dict[str, Counter] = defaultdict(Counter)
    for i in range(len(tokens) - 1):
        nxt[tokens[i]][tokens[i + 1]] += 1
    num = den = 0.0
    for _st, c in nxt.items():
        tot = sum(c.values())
        p = np.array(list(c.values()), dtype=np.float64) / tot
        num += -(p * np.log2(p + 1e-9)).sum() * tot
        den += tot
    return num / max(1.0, den)


def partition_features(p: HalfPartition) -> np.ndarray:
    """17-dim feature vector: 16 bigram probs + 1 conditional entropy.

    Returns zeros when the partition has fewer than 3 synapses (too few
    steps to compute meaningful bigrams).
    """
    if len(p.pts) < 3:
        return np.zeros(FEAT_DIM, dtype=np.float64)
    tokens = tokenize(p.pts)
    if len(tokens) < 2:
        return np.zeros(FEAT_DIM, dtype=np.float64)
    bg = bigram_features(tokens)
    ent = cond_entropy(tokens)
    return np.append(bg, ent)


# ---------------------------------------------------------------------------
# Partition extraction
# ---------------------------------------------------------------------------

def extract_partitions(
    pre_pt: np.ndarray,
    post_pt: np.ndarray,
    pre_root_id: np.ndarray,
    post_root_id: np.ndarray,
    root_remap: dict[int, int],
    *,
    min_synapses: int = 4,
    sides: str = 'both',
) -> list[HalfPartition]:
    """Extract half-partitions from a box synapse table.

    Parameters
    ----------
    pre_pt / post_pt     : (N, 3) synapse position arrays in nm
    pre_root_id / post_root_id : (N,) int64 v117 root ID arrays
    root_remap           : mapping v117 root ID -> v18xx root ID
    min_synapses         : drop partitions with fewer synapses
    sides                : 'pre', 'post', or 'both'
    """
    partitions: list[HalfPartition] = []
    candidates = []
    if sides in ('pre', 'both'):
        candidates.append(('pre', pre_pt, pre_root_id))
    if sides in ('post', 'both'):
        candidates.append(('post', post_pt, post_root_id))

    for side, pts, root_ids in candidates:
        for rid, indices in root_groups(root_ids).items():
            if rid == 0:
                continue
            if len(indices) < min_synapses:
                continue
            target = root_remap.get(rid, 0)
            if target == 0:
                continue
            partitions.append(HalfPartition(
                root_id=rid,
                v18xx_root=target,
                pts=pts[indices].astype(np.float64),
                side=side,
            ))
    return partitions


# ---------------------------------------------------------------------------
# Merge pair construction
# ---------------------------------------------------------------------------

def _artificial_positives(
    partitions: list[HalfPartition],
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """PCA-midpoint split of each partition as fallback positive examples.

    Returns list of (feat_left, feat_right) pairs.
    """
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for p in partitions:
        if len(p.pts) < 6:
            continue
        pts_sorted, _pca1, _pca2 = ordered_pts(p.pts)
        mid = len(pts_sorted) // 2
        left_pts, right_pts = pts_sorted[:mid], pts_sorted[mid:]
        if len(left_pts) < 3 or len(right_pts) < 3:
            continue
        fl = partition_features(HalfPartition(p.root_id, p.v18xx_root, left_pts, p.side))
        fr = partition_features(HalfPartition(p.root_id, p.v18xx_root, right_pts, p.side))
        rows.append((fl, fr))
    return rows


def build_merge_pairs(
    partitions: list[HalfPartition],
    *,
    max_neg_ratio: float = 3.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (X, y) arrays for merge binary classification.

    Positive pairs (y=1)
        Two partitions from *different* v117 root IDs that map to the *same*
        v18xx root -- real false-splits corrected by proofreading.

    Negative pairs (y=0)
        Two partitions with different v18xx roots.  Subsampled to
        ``max_neg_ratio * n_positives``, preferring spatially closest pairs.

    Feature vector layout (35-dim total):
        cols  0-15 : bigram features of partition A          (16)
        col  16    : conditional entropy of partition A       (1)
        cols 17-32 : bigram features of partition B          (16)
        col  33    : conditional entropy of partition B       (1)
        col  34    : log1p(centroid distance in nm)           (1)

    Falls back to same-root PCA-midpoint artificial positives when real
    positives number fewer than 2, so there is always a training signal
    even in boxes where v117 false-splits are rare.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    feats = [partition_features(p) for p in partitions]
    centroids = [p.pts.mean(axis=0) for p in partitions]
    v18xx = [p.v18xx_root for p in partitions]

    # Tuples: (feat_a, feat_b, dist_nm, label)
    pos_rows: list[tuple[np.ndarray, np.ndarray, float, int]] = []
    neg_rows: list[tuple[np.ndarray, np.ndarray, float, int]] = []

    for i, j in combinations(range(len(partitions)), 2):
        if partitions[i].root_id == partitions[j].root_id:
            continue  # same root — skip (identity pair)
        dist = float(np.linalg.norm(centroids[i] - centroids[j]))
        if v18xx[i] == v18xx[j]:
            pos_rows.append((feats[i], feats[j], dist, 1))
        else:
            neg_rows.append((feats[i], feats[j], dist, 0))

    # Fall back to artificial positives if real ones are sparse
    if len(pos_rows) < 2:
        for fl, fr in _artificial_positives(partitions, rng):
            pos_rows.append((fl, fr, 0.0, 1))

    if not pos_rows:
        return (
            np.zeros((0, PAIR_DIM), dtype=np.float64),
            np.zeros(0, dtype=np.int64),
        )

    # Subsample negatives: prefer spatially close pairs, cap ratio
    neg_rows.sort(key=lambda r: r[2])
    n_neg = min(len(neg_rows), max(1, int(len(pos_rows) * max_neg_ratio)))
    neg_rows = neg_rows[:n_neg]

    all_rows = pos_rows + neg_rows
    order = rng.permutation(len(all_rows))
    all_rows = [all_rows[k] for k in order]

    X = np.array(
        [np.concatenate([fa, fb, [np.log1p(d)]]) for fa, fb, d, _ in all_rows],
        dtype=np.float64,
    )
    y = np.array([lbl for _, _, _, lbl in all_rows], dtype=np.int64)
    return X, y
