"""Per-atom, label-blind features for the object-level tasks (EXP-063 onward).

Three families, kept separate so an experiment can ablate them and so a gain
can be attributed:

``polarity``      from the synapse tallies alone. H1 established that an atom's
                  pre/post fraction identifies its compartment without labels;
                  H8 asks whether an atom whose fraction sits in the middle is
                  an axon merged to a dendrite.
``global_shape``  the ten whole-object descriptors of the synapse cloud from
                  ``experiments/pcfg/global_shape_merge.py`` -- the detector
                  the PCFG report measured at AUC 0.875 / precision 0.41 on
                  v117 roots. Ported to numpy here because scikit-learn is not
                  installed in this environment and the harness carries no new
                  dependency (see ``baselines.py``); the port is checked
                  against scikit-learn's own output in
                  ``tests/test_atom_features.py``.
``object_geometry`` from the L2 node cloud (``objgeom``), which EXP-070 showed
                  is a strictly tighter description of the object than its
                  endpoints. Extent, anisotropy, and how the cloud splits.

Everything here is a function of one atom's own points and counts. Nothing
looks at a neighbour, a label, or a candidate pair.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# polarity
# ---------------------------------------------------------------------------

POLARITY_COLS = ["pre_frac", "minority_frac", "log_n_pre", "log_n_post",
                 "log_n_syn"]


def polarity_features(n_pre: np.ndarray, n_post: np.ndarray) -> np.ndarray:
    """``[N, 5]`` from per-atom pre/post synapse counts.

    ``minority_frac`` is the smaller side's share: 0 for a purely axonal or
    purely dendritic atom, 0.5 for an even mix. It is the one-number form of
    H8's "pre-fraction in (0.3, 0.7)".
    """
    pre = np.asarray(n_pre, np.float64)
    post = np.asarray(n_post, np.float64)
    tot = pre + post
    frac = pre / np.maximum(tot, 1.0)
    return np.stack([frac, np.minimum(frac, 1.0 - frac),
                     np.log1p(pre), np.log1p(post), np.log1p(tot)], axis=1)


# ---------------------------------------------------------------------------
# global shape (numpy port of global_shape_merge.global_features)
# ---------------------------------------------------------------------------

GLOBAL_SHAPE_COLS = ["log_n", "log_extent", "pc1_rel", "pc2_rel", "pc3_rel",
                     "bimod", "center_gap", "balance", "log_n_blobs",
                     "largest_blob_frac"]
GLOBAL_SHAPE_SIZE_COLS = [0]            # log n -- the size confound, alone
GLOBAL_SHAPE_SHAPE_COLS = list(range(2, 10))

DBSCAN_EPS_NM = 5000.0
DBSCAN_MIN_SAMPLES = 3


def _kmeans2(pts: np.ndarray, *, n_init: int = 10, seed: int = 0,
             max_iter: int = 300, tol: float = 1e-4):
    """Lloyd's 2-means, greedy k-means++ seeding, best of ``n_init`` restarts.

    Same algorithm as ``sklearn.cluster.KMeans(2, n_init=2, random_state=0)``
    -- greedy k-means++ (two candidate second centres, keep the one with the
    lower potential), Lloyd iterations, convergence when the squared centre
    shift falls under ``tol`` times the data's mean per-axis variance or the
    labels stop changing -- but not the same RNG stream, so the two land in the
    same local optimum only most of the time. A first port with plain seeding
    and two restarts found a worse optimum than scikit-learn on 130 of 400 real
    clouds; five times the restarts closes that. The features that come out
    (inertia ratio, centre gap, balance) are validated against scikit-learn's
    on real synapse clouds in ``tests/test_atom_features.py``, with the rule
    that the port must reach an equal-or-better optimum.
    """
    rng = np.random.default_rng(seed)
    n = len(pts)
    scaled_tol = tol * float(np.var(pts, axis=0).mean())
    best = None
    # One deterministic restart from the principal axis: split the cloud at its
    # median along PC1 and seed from the two halves' means. For the elongated
    # clouds a neurite makes, this is the basin k-means++ sampling misses most
    # often; the remaining restarts are greedy k-means++.
    mu = pts.mean(0)
    if n > 1:
        _, _, vt = np.linalg.svd(pts - mu, full_matrices=False)
        proj = (pts - mu) @ vt[0]
        half = proj > np.median(proj)
        if half.any() and (~half).any():
            inits = [np.stack([pts[~half].mean(0), pts[half].mean(0)])]
        else:
            inits = []
    else:
        inits = []
    for r in range(n_init):
        if r < len(inits):
            centres = inits[r]
        else:
            c0 = pts[rng.integers(n)]
            d2 = ((pts - c0) ** 2).sum(1)
            tot = d2.sum()
            if tot <= 0:
                centres = np.stack([c0, c0])
            else:
                # greedy k-means++: sample 2 candidates ∝ D², keep the better one
                cand = rng.choice(n, size=2, p=d2 / tot)
                pot = [np.minimum(d2, ((pts - pts[c]) ** 2).sum(1)).sum()
                       for c in cand]
                centres = np.stack([c0, pts[cand[int(np.argmin(pot))]]])
        lab = None
        for _ in range(max_iter):
            d = ((pts[:, None, :] - centres[None, :, :]) ** 2).sum(2)
            new_lab = d.argmin(1)
            new = np.stack([pts[new_lab == k].mean(0) if (new_lab == k).any()
                            else centres[k] for k in range(2)])
            shift = float(((new - centres) ** 2).sum())
            same = lab is not None and np.array_equal(new_lab, lab)
            centres, lab = new, new_lab
            if same or shift <= scaled_tol:
                break
        d = ((pts[:, None, :] - centres[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        inertia = float(d[np.arange(n), lab].sum())
        if best is None or inertia < best[0]:
            best = (inertia, centres.copy(), lab.copy())
    return best


def _dbscan_labels(pts: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """DBSCAN cluster labels, -1 for noise. Same semantics as scikit-learn's:

    a point is *core* when at least ``min_samples`` points (itself included)
    lie within ``eps``; clusters are connected components of core points
    linked within ``eps``; a non-core point within ``eps`` of a core point
    joins that core's cluster (a *border* point); everything else is noise.
    """
    n = len(pts)
    if n == 0:
        return np.empty(0, np.int64)
    tree = cKDTree(pts)
    neigh = tree.query_ball_point(pts, r=eps)
    counts = np.fromiter((len(v) for v in neigh), np.int64, n)
    core = counts >= min_samples
    labels = np.full(n, -1, np.int64)
    if not core.any():
        return labels
    core_idx = np.flatnonzero(core)
    pos = {int(i): k for k, i in enumerate(core_idx.tolist())}
    rows, cols = [], []
    for i in core_idx.tolist():
        for j in neigh[i]:
            if core[j]:
                rows.append(pos[i]); cols.append(pos[j])
    g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)),
                   shape=(len(core_idx), len(core_idx)))
    _, comp = connected_components(g, directed=False)
    labels[core_idx] = comp
    # border points: attach to any core neighbour's cluster
    for i in np.flatnonzero(~core).tolist():
        for j in neigh[i]:
            if core[j]:
                labels[i] = labels[j]
                break
    return labels


def global_shape_features(pts: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """The ten whole-object descriptors of ``global_shape_merge.global_features``."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    if n == 0:
        return np.array([0.0, 0.0, 0, 0, 0, 0.0, 0.0, 0.0, np.log1p(1), 1.0])
    c = pts - pts.mean(0)
    cov = np.cov(c.T) if n > 1 else np.zeros((3, 3))
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1] if n > 1 else np.zeros(3)
    evals = np.clip(evals, 0, None)
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    bimod = center_gap = balance = 0.0
    ndb, largest_frac = 1, 1.0
    if n >= 6:
        i1 = float(((pts - pts.mean(0)) ** 2).sum())
        # default restarts, deliberately: a first version passed n_init=2 here
        # to mirror scikit-learn's call and lost to it on 13 of 400 real
        # clouds; the diagnostic showed every one reachable at the default.
        i2, centres, lab = _kmeans2(pts, seed=seed)
        bimod = float(1.0 - i2 / i1) if i1 > 0 else 0.0
        center_gap = float(np.linalg.norm(centres[0] - centres[1]) / (extent + 1.0))
        n0, n1 = int((lab == 0).sum()), int((lab == 1).sum())
        balance = min(n0, n1) / max(n0, n1) if max(n0, n1) else 0.0
        labs = _dbscan_labels(pts, DBSCAN_EPS_NM, DBSCAN_MIN_SAMPLES)
        labs = labs[labs >= 0]
        if len(labs):
            counts = np.bincount(labs)
            ndb = len(counts)
            largest_frac = float(counts.max() / counts.sum())
    return np.array([
        np.log1p(n), np.log1p(extent),
        np.sqrt(evals[0]) / (extent + 1.0), np.sqrt(evals[1]) / (extent + 1.0),
        np.sqrt(evals[2]) / (extent + 1.0),
        bimod, center_gap, balance, np.log1p(ndb), largest_frac,
    ])


# ---------------------------------------------------------------------------
# object geometry (L2 node cloud)
# ---------------------------------------------------------------------------

OBJECT_GEOMETRY_COLS = ["log_n_nodes", "log_extent_nm", "aniso_12", "aniso_13",
                        "log_mean_radius_nm", "radius_cv", "bimod_nodes",
                        "center_gap_nodes", "balance_nodes"]


def object_geometry_features(pts: np.ndarray, radii: Optional[np.ndarray] = None,
                             *, seed: int = 0, max_points: int = 4000) -> np.ndarray:
    """``[9]`` from an atom's L2 node positions (and radii, if given).

    Large atoms are subsampled to ``max_points`` for the 2-means term only;
    extent, PCA and radius statistics use every node.
    """
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    if n == 0:
        return np.zeros(len(OBJECT_GEOMETRY_COLS))
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    if n > 1:
        ev = np.clip(np.sort(np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T)))[::-1],
                     0, None)
        ev = np.sqrt(ev)
        a12 = ev[1] / (ev[0] + 1e-9)
        a13 = ev[2] / (ev[0] + 1e-9)
    else:
        a12 = a13 = 0.0
    if radii is not None and len(radii):
        r = np.asarray(radii, np.float64)
        r = r[np.isfinite(r)]
        lr = float(np.log1p(r.mean())) if len(r) else 0.0
        cv = float(r.std() / (r.mean() + 1e-9)) if len(r) else 0.0
    else:
        lr = cv = 0.0
    bimod = center_gap = balance = 0.0
    if n >= 6:
        P = pts
        if n > max_points:
            P = pts[np.random.default_rng(seed).choice(n, max_points, replace=False)]
        i1 = float(((P - P.mean(0)) ** 2).sum())
        i2, centres, lab = _kmeans2(P, seed=seed)
        bimod = float(1.0 - i2 / i1) if i1 > 0 else 0.0
        center_gap = float(np.linalg.norm(centres[0] - centres[1]) / (extent + 1.0))
        n0, n1 = int((lab == 0).sum()), int((lab == 1).sum())
        balance = min(n0, n1) / max(n0, n1) if max(n0, n1) else 0.0
    return np.array([np.log1p(n), np.log1p(extent), a12, a13, lr, cv,
                     bimod, center_gap, balance])


# ---------------------------------------------------------------------------
# topology columns already on the substrate
# ---------------------------------------------------------------------------

TOPOLOGY_COLS = ["n_l2", "n_edge", "n_comp", "n_iso", "n_end", "n_branch",
                 "n_seg", "n_leaf_seg", "n_cycle", "cable_nm", "cable_nan_seg",
                 "caliber_mean_nm"]


def topology_features(npz) -> np.ndarray:
    """``[A, 12]`` from a ``topology/k*.npz`` file, log1p'd where it is a count
    or a length so a boosted-stump model sees a sane range."""
    cols = []
    for c in TOPOLOGY_COLS:
        v = np.asarray(npz[c], np.float64)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        cols.append(np.log1p(np.maximum(v, 0.0)))
    return np.stack(cols, axis=1)


def precision_at_top(y_true: np.ndarray, score: np.ndarray, frac: float) -> dict:
    """Precision/recall when the top ``frac`` of scores is flagged.

    Same operating point as ``global_shape_merge._grouped_auc``: the threshold
    is the ``1 - frac`` quantile of the scores, ties included on the flagged
    side.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, np.float64)
    if not len(s):
        return {"precision": float("nan"), "recall": float("nan"), "n_flagged": 0}
    thr = np.quantile(s, 1.0 - frac)
    flagged = s >= thr
    nf = int(flagged.sum())
    prec = float(y[flagged].mean()) if nf else float("nan")
    rec = float((flagged & y).sum() / max(int(y.sum()), 1))
    return {"precision": prec, "recall": rec, "n_flagged": nf}
