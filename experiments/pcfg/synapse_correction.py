"""Synapse-level proofreading correction model: learn f(v117) -> proofread partition.

Where ``pcfg_partitions.py`` models only false-splits (merges) at the *half-partition*
level, this module works at the **synapse level** and captures BOTH proofreading
directions from a single label.

Construction
------------
Each synapse is a stable physical landmark.  Its supervoxel is immutable, so its root
ID at any materialization is well defined and *single-valued* (unlike ``get_latest_roots``
on a whole root, which goes multi-valued exactly when a split happened).  Joining a
synapse-side to itself across versions gives, per side (pre / post)::

    (root_v117, root_later)

The pairwise label is simply **"do these two synapse-sides share the same later root?"**
and it unifies both corrections:

  * within-v117-root pair, later roots DIFFER  -> a SPLIT  (false-merge corrected)
  * cross-v117-root pair,  later roots SAME    -> a MERGE  (false-split corrected)
  * same/same or diff/diff                     -> stable (no correction)

So the learned affinity ``P(same later root | v117 features)`` *is* the correction
operator; XOR-ing it against the v117 relation yields the edit.

Grouping
--------
Cross-validation groups are connected components of the bipartite
(v117-root <-> later-root) co-occurrence graph -- the physical "cells".  This prevents
the cell-identity leakage that inflated the berlin risk AUC (0.92 -> 0.76).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .pcfg_partitions import (
    FEAT_DIM,
    HalfPartition,
    ordered_pts,
    partition_features,
)

# Pair feature layout (see _pair_features):
#   0      log1p(distance nm)
#   1      same v117 root (0/1)
#   2      log1p(n synapses in root A)
#   3      log1p(n synapses in root B)
#   4 .. 4+FEAT_DIM-1            grammar features of root A   (17)
#   ..     grammar features of root B                         (17)
#   -5     axial separation (normalized by root-A extent)
#   -4     lateral separation (normalized by root-A extent)
#   -3     gap occupancy along A->B segment (within-root continuity proxy)
#   -2     local kNN density at A
#   -1     local kNN density at B
PAIR_DIM: int = 4 + 2 * FEAT_DIM + 5


@dataclass
class SideTable:
    """Synapse half-sides (pre and/or post) joined across two materializations."""

    syn_id: np.ndarray      # (M,) int64 synapse id (repeated for pre & post)
    side: np.ndarray        # (M,) int8   0 = pre, 1 = post
    pt: np.ndarray          # (M, 3) float64 position in nm
    root_v117: np.ndarray   # (M,) int64
    root_later: np.ndarray  # (M,) int64  (0 = body absent at later version)

    def __len__(self) -> int:
        return len(self.syn_id)

    def mask(self, m: np.ndarray) -> "SideTable":
        return SideTable(self.syn_id[m], self.side[m], self.pt[m],
                         self.root_v117[m], self.root_later[m])


# ---------------------------------------------------------------------------
# Build a SideTable from per-synapse cross-version arrays
# ---------------------------------------------------------------------------

def build_side_table(
    pre_pt: np.ndarray,
    post_pt: np.ndarray,
    pre_root_v117: np.ndarray,
    post_root_v117: np.ndarray,
    pre_root_later: np.ndarray,
    post_root_later: np.ndarray,
    syn_id: np.ndarray,
    *,
    sides: str = "both",
) -> SideTable:
    """Stack pre/post synapse sides into one long SideTable.

    Drops sides with a zero v117 root (no segmentation) up front; later-root 0
    (body deleted at the target version) is kept and handled as "no label".
    """
    blocks: list[tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]] = []
    if sides in ("pre", "both"):
        blocks.append((syn_id, 0, pre_pt, pre_root_v117, pre_root_later))
    if sides in ("post", "both"):
        blocks.append((syn_id, 1, post_pt, post_root_v117, post_root_later))

    sid, sd, pt, rv, rl = [], [], [], [], []
    for ids, side_code, pts, rv117, rlater in blocks:
        keep = np.asarray(rv117, dtype=np.int64) > 0
        sid.append(np.asarray(ids, dtype=np.int64)[keep])
        sd.append(np.full(int(keep.sum()), side_code, dtype=np.int8))
        pt.append(np.asarray(pts, dtype=np.float64)[keep])
        rv.append(np.asarray(rv117, dtype=np.int64)[keep])
        rl.append(np.asarray(rlater, dtype=np.int64)[keep])

    return SideTable(
        syn_id=np.concatenate(sid) if sid else np.zeros(0, np.int64),
        side=np.concatenate(sd) if sd else np.zeros(0, np.int8),
        pt=np.concatenate(pt) if pt else np.zeros((0, 3), np.float64),
        root_v117=np.concatenate(rv) if rv else np.zeros(0, np.int64),
        root_later=np.concatenate(rl) if rl else np.zeros(0, np.int64),
    )


# ---------------------------------------------------------------------------
# Edit summary -- the "identify merges and splits at synapse level" deliverable
# ---------------------------------------------------------------------------

def summarize_edits(tab: SideTable) -> dict[str, int]:
    """Count synapse-level merge/split corrections implied by the v117->later join.

    A v117 root is *split* if its synapse-sides span >= 2 distinct nonzero later
    roots.  A later root is a *merge* target if it gathers >= 2 distinct v117 roots.
    Counts are reported per side-stream pooled together.
    """
    valid = tab.root_later > 0
    rv, rl = tab.root_v117[valid], tab.root_later[valid]

    # v117 root -> set of later roots
    split_roots = 0
    by_v117: dict[int, set[int]] = {}
    for a, b in zip(rv.tolist(), rl.tolist()):
        by_v117.setdefault(a, set()).add(b)
    split_roots = sum(1 for s in by_v117.values() if len(s) >= 2)

    # later root -> set of v117 roots
    by_later: dict[int, set[int]] = {}
    for a, b in zip(rv.tolist(), rl.tolist()):
        by_later.setdefault(b, set()).add(a)
    merge_targets = sum(1 for s in by_later.values() if len(s) >= 2)

    n_changed = int((rv != rl).sum())  # crude: root id relabeled (incl. pure renames)
    return {
        "sides": int(len(tab)),
        "sides_with_later_label": int(valid.sum()),
        "v117_roots": len(by_v117),
        "later_roots": len(by_later),
        "split_roots": split_roots,        # false merges to be cut
        "merge_targets": merge_targets,     # false splits to be joined
    }


# ---------------------------------------------------------------------------
# Leakage-safe grouping: connected components of (v117-root <-> later-root)
# ---------------------------------------------------------------------------

class _DSU:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def cell_components(tab: SideTable) -> dict[int, int]:
    """Map each v117 root id -> a "cell" component id (stable small int).

    Components are the connected parts of the bipartite graph linking every v117
    root to every later root that shares a synapse-side with it.
    """
    dsu = _DSU()
    valid = tab.root_later > 0
    for a, b in zip(tab.root_v117[valid].tolist(), tab.root_later[valid].tolist()):
        dsu.union(("v", int(a)), ("l", int(b)))
    # also register v117 roots that have no later label so they still get a group
    for a in np.unique(tab.root_v117).tolist():
        dsu.find(("v", int(a)))

    comp_id: dict = {}
    out: dict[int, int] = {}
    for a in np.unique(tab.root_v117).tolist():
        root = dsu.find(("v", int(a)))
        if root not in comp_id:
            comp_id[root] = len(comp_id)
        out[int(a)] = comp_id[root]
    return out


# ---------------------------------------------------------------------------
# Per-root geometry / grammar cache
# ---------------------------------------------------------------------------

@dataclass
class _RootCtx:
    pts: np.ndarray         # (n, 3)
    centroid: np.ndarray    # (3,)
    pca: np.ndarray         # (3, 3) rows = principal axes
    extent: float           # span along pca1 (nm), >0
    feat: np.ndarray        # (FEAT_DIM,) grammar features
    knn_mean: np.ndarray    # (n,) mean dist to up-to-k nearest same-root neighbours
    index_of: dict          # row-index-in-tab -> local index in pts


def _build_root_ctx(tab: SideTable, idxs: list[int], k: int = 4) -> _RootCtx:
    pts = tab.pt[idxs].astype(np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    if len(pts) >= 2:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        pca = np.vstack([Vt, np.zeros((3 - Vt.shape[0], 3))])[:3]
    else:
        pca = np.eye(3)
    proj1 = centered @ pca[0]
    extent = float(proj1.max() - proj1.min()) if len(pts) else 1.0
    extent = extent if extent > 1e-6 else 1.0
    feat = partition_features(HalfPartition(0, 0, pts, "x"))
    # local density: mean distance to up to k nearest same-root neighbours
    if len(pts) >= 2:
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        kk = min(k, len(pts) - 1)
        knn_mean = np.sort(d, axis=1)[:, :kk].mean(axis=1)
    else:
        knn_mean = np.zeros(len(pts))
    index_of = {row: i for i, row in enumerate(idxs)}
    return _RootCtx(pts, centroid, pca, extent, feat, knn_mean, index_of)


def _gap_occupancy(ctx: _RootCtx, ia: int, ib: int) -> float:
    """Fraction of the A->B segment (in root A's frame) backed by same-root points.

    A clean continuation has points all along the segment (occupancy ~1); a merge
    seam has a void between the two lobes (occupancy ~0).
    """
    a, b = ctx.pts[ia], ctx.pts[ib]
    seg = b - a
    L = float(np.linalg.norm(seg))
    if L < 1e-6:
        return 1.0
    u = seg / L
    rel = ctx.pts - a
    t = rel @ u                      # projection along the segment
    perp = np.linalg.norm(rel - np.outer(t, u), axis=1)
    tube = max(0.15 * L, 500.0)      # lateral tolerance in nm
    nbins = 6
    on_seg = (t >= 0) & (t <= L) & (perp <= tube)
    if not on_seg.any():
        return 0.0
    bins = np.clip((t[on_seg] / L * nbins).astype(int), 0, nbins - 1)
    return float(len(set(bins.tolist())) / nbins)


def _pair_features(
    ca: _RootCtx, ia: int, cb: _RootCtx, ib: int, *, same_root: bool,
    na: int, nb: int,
) -> np.ndarray:
    pa, pb = ca.pts[ia], cb.pts[ib]
    diff = pb - pa
    dist = float(np.linalg.norm(diff))
    # relative geometry in root A's frame
    axial = abs(float(diff @ ca.pca[0])) / ca.extent
    lateral = float(np.linalg.norm(diff - (diff @ ca.pca[0]) * ca.pca[0])) / ca.extent
    gap = _gap_occupancy(ca, ia, ib) if same_root else 0.0
    dens_a = float(ca.knn_mean[ia]) if ia < len(ca.knn_mean) else 0.0
    dens_b = float(cb.knn_mean[ib]) if ib < len(cb.knn_mean) else 0.0
    return np.concatenate([
        [np.log1p(dist), 1.0 if same_root else 0.0,
         np.log1p(na), np.log1p(nb)],
        ca.feat, cb.feat,
        [axial, lateral, gap, np.log1p(dens_a), np.log1p(dens_b)],
    ]).astype(np.float64)


# ---------------------------------------------------------------------------
# Pair dataset (both strata)
# ---------------------------------------------------------------------------

def build_correction_pairs(
    tab: SideTable,
    *,
    min_synapses: int = 4,
    max_within_pairs_per_root: int = 120,
    cross_k_neighbors: int = 12,
    cross_radius_nm: float = 6000.0,
    max_neg_ratio: float = 3.0,
    rng: np.random.Generator | None = None,
):
    """Build (X, y, groups, strata) for the synapse-level correction classifier.

    y = 1  iff the two synapse-sides share the same nonzero later root.
    strata: 0 = SPLIT stratum (within v117 root), 1 = MERGE stratum (cross v117 root).
    groups: cell component id (for GroupKFold).

    Only sides with a nonzero later label participate (we need supervision).  Pre and
    post side-streams are paired separately (different root spaces) then pooled.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    comp = cell_components(tab)
    X: list[np.ndarray] = []
    y: list[int] = []
    groups: list[int] = []
    strata: list[int] = []

    try:
        from scipy.spatial import cKDTree
        have_kdtree = True
    except ImportError:
        have_kdtree = False

    for side_code in (0, 1):
        sel = (tab.side == side_code) & (tab.root_later > 0)
        rows = np.nonzero(sel)[0]
        if len(rows) < 2:
            continue
        sub = tab.mask(sel)
        local_rows = np.arange(len(sub))

        # group rows by v117 root, build a context per root
        by_root: dict[int, list[int]] = {}
        for li, rv in zip(local_rows.tolist(), sub.root_v117.tolist()):
            by_root.setdefault(int(rv), []).append(li)
        ctx: dict[int, _RootCtx] = {
            rv: _build_root_ctx(sub, idxs)
            for rv, idxs in by_root.items() if len(idxs) >= min_synapses
        }

        # ---- SPLIT stratum: within-root pairs ----
        for rv, c in ctx.items():
            idxs = by_root[rv]
            pairs = list(combinations(range(len(idxs)), 2))
            if len(pairs) > max_within_pairs_per_root:
                sel_p = rng.choice(len(pairs), max_within_pairs_per_root, replace=False)
                pairs = [pairs[p] for p in sel_p]
            for a, b in pairs:
                ra, rb = idxs[a], idxs[b]
                lbl = int(sub.root_later[ra] == sub.root_later[rb])
                X.append(_pair_features(c, c.index_of[ra], c, c.index_of[rb],
                                        same_root=True, na=len(idxs), nb=len(idxs)))
                y.append(lbl)
                strata.append(0)
                groups.append(comp.get(rv, -1))

        # ---- MERGE stratum: spatially near cross-root pairs ----
        rootids = list(ctx.keys())
        if len(rootids) >= 2 and have_kdtree:
            all_idx = np.concatenate([by_root[rv] for rv in rootids])
            pts_all = sub.pt[all_idx]
            tree = cKDTree(pts_all)
            kq = min(cross_k_neighbors + 1, len(all_idx))
            dnn, inn = tree.query(pts_all, k=kq, workers=-1)
            seen: set[tuple[int, int]] = set()
            cross_rows: list[tuple] = []
            for i in range(len(all_idx)):
                ri = int(all_idx[i])
                rvi = int(sub.root_v117[ri])
                for slot in range(1, kq):
                    if dnn[i, slot] > cross_radius_nm:
                        break
                    j = int(inn[i, slot])
                    rj = int(all_idx[j])
                    rvj = int(sub.root_v117[rj])
                    if rvi == rvj:
                        continue  # within-root handled above
                    key = (min(ri, rj), max(ri, rj))
                    if key in seen:
                        continue
                    seen.add(key)
                    lbl = int(sub.root_later[ri] == sub.root_later[rj])
                    cross_rows.append((ri, rj, rvi, rvj, lbl))
            # balance: keep all positives, subsample negatives
            pos = [r for r in cross_rows if r[4] == 1]
            neg = [r for r in cross_rows if r[4] == 0]
            n_neg = min(len(neg), max(1, int(max(1, len(pos)) * max_neg_ratio)))
            if len(neg) > n_neg:
                neg = [neg[p] for p in rng.choice(len(neg), n_neg, replace=False)]
            for ri, rj, rvi, rvj, lbl in pos + neg:
                ca, cb = ctx[rvi], ctx[rvj]
                X.append(_pair_features(ca, ca.index_of[ri], cb, cb.index_of[rj],
                                        same_root=False,
                                        na=len(by_root[rvi]), nb=len(by_root[rvj])))
                y.append(lbl)
                strata.append(1)
                groups.append(comp.get(rvi, -1))

    if not X:
        return (np.zeros((0, PAIR_DIM)), np.zeros(0, np.int64),
                np.zeros(0, np.int64), np.zeros(0, np.int64))
    return (np.array(X), np.array(y, np.int64),
            np.array(groups, np.int64), np.array(strata, np.int64))
