#!/usr/bin/env python3
"""PCFG synapse partition grammar on real v117 MICrONS data.

Fetches the same spatial region as v117_coassign.py but uses the bigram
grammar instead of a GNN -- no skeletons, no torch, no GPU required.
Designed for a direct apples-to-apples comparison with v117_coassign.py.

Usage
-----
    # Single 20 um box (same defaults as v117_coassign.py):
    python scripts/v117_pcfg.py --token $CAVE_TOKEN

    # Larger box for more data:
    python scripts/v117_pcfg.py --token $CAVE_TOKEN --side-um 40

    # Pool multiple boxes:
    python scripts/v117_pcfg.py --token $CAVE_TOKEN --n-boxes 5

    # Custom center:
    python scripts/v117_pcfg.py --token $CAVE_TOKEN \\
        --center-nm 733592 513592 595640 --side-um 30

CAVE token
----------
Create a free account and token at https://global.daf-apis.com and pass
it via --token or the CAVE_TOKEN environment variable.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as a script directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Logging -- mirrors v117_coassign.py setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
for _noisy in ("caveclient", "urllib3", "CAVEclient"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("neuronauts").setLevel(logging.INFO)
log = logging.getLogger("v117_pcfg")
log.setLevel(logging.INFO)

# Same default as v117_coassign.py -- a validated densely-proofread region
DEFAULT_CENTER_NM = (733_592, 513_592, 595_640)
DEFAULT_SIDE_UM = 20.0

# MIP-2 voxel size in nm -- matches MIP_VOXEL_SIZES[2] in neuronauts/fetch.py
_MIP2_VOX = np.array([32.0, 32.0, 40.0], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PCFG synapse partition grammar on real v117 MICrONS data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--token",
        default=os.environ.get("CAVE_TOKEN"),
        help="CAVE auth token (or set CAVE_TOKEN env var)",
    )
    p.add_argument(
        "--center-nm",
        nargs=3,
        type=int,
        default=list(DEFAULT_CENTER_NM),
        metavar=("X", "Y", "Z"),
        help="Bounding box center in global nm",
    )
    p.add_argument(
        "--side-um",
        type=float,
        default=DEFAULT_SIDE_UM,
        help="Bounding box side length in micrometers",
    )
    p.add_argument(
        "--n-boxes",
        type=int,
        default=1,
        help="Number of boxes to fetch (>1 pools partitions from shifted regions)",
    )
    p.add_argument(
        "--side",
        default="both",
        choices=["pre", "post", "both"],
        help="Which synapse side to use for half-partitions",
    )
    p.add_argument(
        "--min-synapses",
        type=int,
        default=4,
        help="Minimum synapses per half-partition",
    )
    p.add_argument(
        "--max-neg-ratio",
        type=float,
        default=3.0,
        help="Max ratio of negative to positive pairs",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--match-distance",
        action="store_true",
        help="Also run with distance-matched negatives (honest grammar-only test)",
    )
    p.add_argument(
        "--use-skeleton",
        action="store_true",
        help="Also run skeleton grammar (fetches CAVE skeletons; slow on first run, fast with cache)",
    )
    p.add_argument(
        "--skeleton-cache-dir",
        default=None,
        help="Directory to cache skeleton fetches across runs",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-box stats",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _sample_centers(
    center_nm: list[int],
    n_boxes: int,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    """Return n_boxes centers: the requested center + random +/-30 um offsets."""
    centers: list[tuple[int, ...]] = [tuple(center_nm)]
    if n_boxes > 1:
        # +/-30,000 nm = +/-30 um shifts; stays within the proofread core
        offsets = rng.integers(-30_000, 30_000, size=(n_boxes - 1, 3))
        for off in offsets:
            c = tuple(int(center_nm[i] + int(off[i])) for i in range(3))
            centers.append(c)
    return centers


_GT_VERSION = 1718  # latest materialization; use 1412 for reproducibility vs older runs

def _fetch_one_box(
    center_nm: tuple[int, ...],
    side_um: float,
    token: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]] | None:
    """Fetch synapses + v117->v1718 remap for one box.

    Returns (pre_pt_nm, post_pt_nm, pre_root_id, post_root_id, remap)
    or None when the box yields no usable data.
    """
    from neuronauts.fetch import fetch_synapses, make_cube_bbox_nm
    from neuronauts.cave_root_mapping import map_roots_between_versions

    bbox_nm = make_cube_bbox_nm(tuple(center_nm), side_um=side_um)
    log.info("Fetching synapses in %.0f um box around %s ...", side_um, center_nm)

    try:
        syn = fetch_synapses(bbox_nm, version=117, token=token)
    except Exception as exc:
        log.warning("Synapse fetch failed for center %s: %s", center_nm, exc)
        return None

    if syn.n_synapses == 0:
        log.warning("No synapses in box %s -- skipping", center_nm)
        return None

    log.info(
        "  %d synapses  (%d unique pre roots, %d unique post roots)",
        syn.n_synapses,
        int(np.unique(syn.pre_root_id).shape[0]),
        int(np.unique(syn.post_root_id).shape[0]),
    )

    # Box-relative MIP-2 voxels -> global nm
    bbox_origin = np.array(bbox_nm[0], dtype=np.float64)
    pre_pt_nm  = syn.pre_pt.astype(np.float64)  * _MIP2_VOX + bbox_origin
    post_pt_nm = syn.post_pt.astype(np.float64) * _MIP2_VOX + bbox_origin

    # Map all unique root IDs to latest GT version (v1718)
    all_roots = list(
        set(syn.pre_root_id.tolist()) | set(syn.post_root_id.tolist())
    )
    log.info("  Mapping %d unique root IDs v117 -> v%d ...", len(all_roots), _GT_VERSION)
    try:
        remap = map_roots_between_versions(all_roots, 117, _GT_VERSION, token=token)
    except Exception as exc:
        log.warning("Root mapping failed for center %s: %s", center_nm, exc)
        return None

    n_mapped = sum(1 for v in remap.values() if v > 0)
    log.info("  %d / %d roots have a valid v%d label", n_mapped, len(all_roots), _GT_VERSION)

    return pre_pt_nm, post_pt_nm, syn.pre_root_id, syn.post_root_id, remap


# ---------------------------------------------------------------------------
# Cross-box analysis (honest grammar test)
# ---------------------------------------------------------------------------

def _cross_box_analysis(
    all_partitions: list,
    box_ids: list,
    n_folds: int,
    seed: int,
    rng: np.random.Generator,
    max_neg_ratio: float,
) -> None:
    """Grammar evaluation on cross-box positive pairs only.

    Cross-box positives: same v1718 root, partitions from *different* boxes.
    Their centroid distances are on the order of box offsets (~30 um), which
    overlaps the negative pair range — so distance carries no free signal and
    the grammar must do the work.
    """
    from collections import defaultdict
    from experiments.pcfg_synapse_partitions.pcfg_partitions import (
        partition_features, BIGRAM_DIM, FEAT_DIM,
    )

    n = len(all_partitions)
    feats_arr  = np.array([partition_features(p) for p in all_partitions])
    centroids  = np.array([p.pts.mean(axis=0) for p in all_partitions])
    v18xx_arr  = np.array([p.v18xx_root for p in all_partitions])
    root_arr   = np.array([p.root_id for p in all_partitions])
    box_arr    = np.array(box_ids)

    # -- Positives: O(N) groupby by v1718 root --------------------------------
    by_v18: dict = defaultdict(list)
    for idx, vr in enumerate(v18xx_arr.tolist()):
        by_v18[vr].append(idx)

    pos_i, pos_j = [], []
    for vr, idxs in by_v18.items():
        if len(idxs) < 2:
            continue
        for ai in range(len(idxs)):
            for bi in range(ai + 1, len(idxs)):
                i, j = idxs[ai], idxs[bi]
                if box_arr[i] == box_arr[j]:
                    continue  # same box — skip
                if root_arr[i] == root_arr[j]:
                    continue  # same v117 root (not a false split)
                pos_i.append(i)
                pos_j.append(j)

    n_pos_xb = len(pos_i)
    if n_pos_xb == 0:
        print()
        print("Cross-box analysis: no cross-box positive pairs found.")
        print(f"  (no v{_GT_VERSION} root appears in more than one box)")
        return

    # -- Negatives: vectorized random cross-box sampling ----------------------
    n_neg_target = min(int(n_pos_xb * max_neg_ratio), 100_000)
    neg_i, neg_j = [], []
    batch_sz = 50_000
    while len(neg_i) < n_neg_target:
        ii = rng.integers(0, n, size=batch_sz)
        jj = rng.integers(0, n, size=batch_sz)
        mask = (
            (ii != jj)
            & (box_arr[ii] != box_arr[jj])
            & (v18xx_arr[ii] != v18xx_arr[jj])
            & (root_arr[ii] != root_arr[jj])
        )
        valid_ii = ii[mask]
        valid_jj = jj[mask]
        remaining = n_neg_target - len(neg_i)
        neg_i.extend(valid_ii[:remaining].tolist())
        neg_j.extend(valid_jj[:remaining].tolist())

    # -- Build feature matrix -------------------------------------------------
    pos_ia = np.array(pos_i)
    pos_ja = np.array(pos_j)
    neg_ia = np.array(neg_i)
    neg_ja = np.array(neg_j)

    def _pair_feat(ia, ja):
        dists = np.linalg.norm(centroids[ia] - centroids[ja], axis=1)
        return np.column_stack([feats_arr[ia], feats_arr[ja], np.log1p(dists)])

    X_pos = _pair_feat(pos_ia, pos_ja)
    X_neg = _pair_feat(neg_ia, neg_ja)
    X_xb = np.vstack([X_pos, X_neg])
    y_xb = np.concatenate([
        np.ones(n_pos_xb, dtype=np.int64),
        np.zeros(len(neg_i), dtype=np.int64),
    ])
    order = rng.permutation(len(y_xb))
    X_xb = X_xb[order]
    y_xb = y_xb[order]

    bg_idx   = list(range(BIGRAM_DIM)) + list(range(FEAT_DIM, FEAT_DIM + BIGRAM_DIM))
    be_idx   = list(range(FEAT_DIM))   + list(range(FEAT_DIM, FEAT_DIM * 2))
    dist_idx = [X_xb.shape[1] - 1]

    pct_pos = 100.0 * n_pos_xb / len(y_xb)
    pos_dists_nm = np.expm1(X_pos[:, -1])
    neg_dists_nm = np.expm1(X_neg[:, -1])

    print()
    print(f"Cross-box pairs only (n={len(y_xb):,} pairs, {pct_pos:.1f}% positive):")
    print("  [honest grammar test: same-neuron fragments in spatially distinct boxes]")
    print(f"  Positive dist: min={pos_dists_nm.min()/1e3:.1f}  "
          f"med={np.median(pos_dists_nm)/1e3:.1f}  "
          f"max={pos_dists_nm.max()/1e3:.1f} µm")
    print(f"  Negative dist: min={neg_dists_nm.min()/1e3:.1f}  "
          f"med={np.median(neg_dists_nm)/1e3:.1f}  "
          f"max={neg_dists_nm.max()/1e3:.1f} µm")
    print()
    _run_cv(X_xb[:, dist_idx], y_xb, n_folds, seed, "distance only (1 feat)")
    print("-- LR --")
    _run_cv(X_xb[:, bg_idx],   y_xb, n_folds, seed, "bigram-syntax (16+16 feats)")
    _run_cv(X_xb[:, be_idx],   y_xb, n_folds, seed, "bigram + entropy (17+17 feats)")
    _run_cv(X_xb,               y_xb, n_folds, seed, "bigram + entropy + dist (35 feats)")
    print("-- RF --")
    _run_cv(X_xb[:, bg_idx],   y_xb, n_folds, seed, "bigram-syntax RF (16+16 feats)", classifier="rf")
    _run_cv(X_xb[:, be_idx],   y_xb, n_folds, seed, "bigram + entropy RF (17+17 feats)", classifier="rf")
    _run_cv(X_xb,               y_xb, n_folds, seed, "bigram + entropy + dist RF (35 feats)", classifier="rf")


# ---------------------------------------------------------------------------
# Partition-level merge evaluation
# ---------------------------------------------------------------------------

def _merge_report(
    X: np.ndarray,
    y: np.ndarray,
    partitions: list,
    n_folds: int,
    seed: int,
) -> None:
    """Report precision/recall/F1 of grammar-predicted merges vs the no-changes baseline.

    The 'no-changes' baseline makes zero merges: recall=0, F1=0. Any F1 above
    zero is the delta the grammar provides.  We also report how many merges the
    grammar suggests so the user can gauge the edit budget.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos < n_folds or n_neg < n_folds:
        return

    # Out-of-fold probabilities from the best classifier (full 35-dim features)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_prob = np.zeros(len(y), dtype=np.float64)
    for tr, va in skf.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(np.nan_to_num(X[tr]))
        X_va = sc.transform(np.nan_to_num(X[va]))
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        clf.fit(X_tr, y[tr])
        oof_prob[va] = clf.predict_proba(X_va)[:, 1]

    # Evaluate at threshold 0.5
    pred = (oof_prob >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    n_suggested = tp + fp

    print()
    print("-- Merge decision (threshold=0.50) --")
    print(f"  True merges needed:   {n_pos} pairs")
    print(f"  Grammar suggests:     {n_suggested} merges ({tp} correct, {fp} wrong)")
    print(f"  No-changes baseline:  P=1.00  R=0.00  F1=0.00  (0 merges made)")
    print(f"  Grammar:              P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}")
    print(f"  Delta F1 over no-changes: +{f1:.2f}")


# ---------------------------------------------------------------------------
# CV helper (identical logic to run_experiment.py)
# ---------------------------------------------------------------------------

def _run_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
    classifier: str = "lr",
) -> float:
    """Stratified k-fold CV; returns mean ROC-AUC.

    classifier: 'lr' = LogisticRegression (default), 'rf' = RandomForest
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print(f"  {label:47s} (sklearn not available)")
        return float("nan")

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos < n_folds or n_neg < n_folds:
        print(f"  {label:47s} skipped (too few: {n_pos}+/{n_neg}-)")
        return float("nan")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    probs = np.zeros(len(y), dtype=np.float64)
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(np.nan_to_num(X[train_idx]))
        X_va = scaler.transform(np.nan_to_num(X[val_idx]))
        if classifier == "rf":
            clf = RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1
            )
            clf.fit(X_tr, y[train_idx])
        else:
            clf = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=seed
            )
            clf.fit(X_tr, y[train_idx])
        probs[val_idx] = clf.predict_proba(X_va)[:, 1]

    auc = float(roc_auc_score(y, probs))
    print(f"  {label:47s} CV AUC = {auc:.2f}")
    return auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    from experiments.pcfg_synapse_partitions.pcfg_partitions import (
        BIGRAM_DIM,
        FEAT_DIM,
        build_merge_pairs,
        extract_partitions,
    )

    # -- Fetch one or more boxes ------------------------------------------
    centers = _sample_centers(args.center_nm, args.n_boxes, rng)
    all_partitions = []
    all_box_ids: list[int] = []
    n_boxes_used = 0

    for box_idx, center_nm in enumerate(centers):
        result = _fetch_one_box(center_nm, args.side_um, args.token)
        if result is None:
            continue
        pre_pt_nm, post_pt_nm, pre_root_id, post_root_id, remap = result

        parts = extract_partitions(
            pre_pt_nm,
            post_pt_nm,
            pre_root_id,
            post_root_id,
            remap,
            min_synapses=args.min_synapses,
            sides=args.side,
        )
        log.info("  -> %d half-partitions extracted", len(parts))
        if args.verbose:
            from collections import Counter as _Counter
            # Count unique v117 root IDs per GT root (true false-split pairs).
            # Note: sides='both' creates 2 partitions per v117 root (pre+post), both
            # sharing the same root_id — those are NOT positive pairs. Only partitions
            # with *different* root_ids mapping to the same GT root are positives.
            v18xx_by_root: dict = {}
            for p in parts:
                v18xx_by_root.setdefault(p.v18xx_root, set()).add(p.root_id)
            n_multi = sum(1 for roots in v18xx_by_root.values() if len(roots) >= 2)
            log.info(
                "    (%d v%d roots with >=2 distinct v117 root IDs = potential positives)",
                n_multi, _GT_VERSION,
            )

        all_partitions.extend(parts)
        all_box_ids.extend([box_idx] * len(parts))
        n_boxes_used += 1

    if not all_partitions:
        sys.exit(
            "No partitions found. Check your CAVE token and center coordinates.\n"
            "The default center requires a valid token for minnie65_public at v117."
        )

    log.info(
        "Total: %d half-partitions from %d box(es)",
        len(all_partitions),
        n_boxes_used,
    )

    # -- Build merge pair dataset -----------------------------------------
    log.info("Building merge pairs (max_neg_ratio=%.1f) ...", args.max_neg_ratio)
    X, y = build_merge_pairs(
        all_partitions, max_neg_ratio=args.max_neg_ratio, rng=rng, match_distance=False,
    )
    X_md, y_md = build_merge_pairs(
        all_partitions, max_neg_ratio=args.max_neg_ratio,
        rng=np.random.default_rng(args.seed), match_distance=True,
    )

    if len(y) == 0:
        sys.exit("No merge pairs generated. Try --min-synapses 2 or a larger box.")

    n_pos = int(y.sum())
    pct_pos = 100.0 * n_pos / len(y)
    t_data = time.time() - t0

    # -- Feature slices ---------------------------------------------------
    # Layout: [bigram_a(16) | entropy_a(1) | bigram_b(16) | entropy_b(1) | log_dist(1)]
    bg_idx   = list(range(BIGRAM_DIM)) + list(range(FEAT_DIM, FEAT_DIM + BIGRAM_DIM))
    be_idx   = list(range(FEAT_DIM)) + list(range(FEAT_DIM, FEAT_DIM * 2))
    dist_idx = [X.shape[1] - 1]  # last column: log1p(centroid_dist_nm)

    # Permuted-label baseline: AUC should be ~0.50 — confirms CV is unbiased
    y_perm = rng.permutation(y)

    # -- Results ----------------------------------------------------------
    print()
    print("=" * 60)
    print("PCFG synapse partition grammar -- live v117 region")
    print("=" * 60)
    print(f"  Box:         {args.side_um:.0f} um around {tuple(args.center_nm)}")
    print(f"  Boxes used:  {n_boxes_used}")
    print(f"  Partitions:  {len(all_partitions)}")
    print(f"  Wall time:   {t_data:.1f} s  (fetch + remap, no skeletons)")
    print()
    # Distance distribution diagnostic — shows whether distance alone separates
    # positives from negatives (a key confound in the within-box setting).
    pos_mask = (y == 1)
    pos_dists_nm = np.expm1(X[pos_mask, -1])   # undo log1p on the distance feature
    neg_dists_nm = np.expm1(X[~pos_mask, -1])

    print(f"Standard negatives (n={len(y):,} pairs, {pct_pos:.1f}% positive):")
    print(f"  Positive centroid dist: "
          f"min={pos_dists_nm.min()/1e3:.1f}  "
          f"med={np.median(pos_dists_nm)/1e3:.1f}  "
          f"max={pos_dists_nm.max()/1e3:.1f} µm")
    print(f"  Negative centroid dist: "
          f"min={neg_dists_nm.min()/1e3:.1f}  "
          f"med={np.median(neg_dists_nm)/1e3:.1f}  "
          f"max={neg_dists_nm.max()/1e3:.1f} µm")
    print("  [negatives = spatially nearest different-neuron pairs]")
    print()
    print("-- Baselines (delta-from-no-changes) --")
    _run_cv(X[:, dist_idx], y,      args.cv_folds, args.seed, "distance only (1 feat)")
    _run_cv(X,              y_perm, args.cv_folds, args.seed, "permuted labels (null)")
    print()
    print("-- Grammar (LR) --")
    _run_cv(X[:, bg_idx], y, args.cv_folds, args.seed, "bigram-syntax (16+16 feats)")
    _run_cv(X[:, be_idx], y, args.cv_folds, args.seed, "bigram + entropy (17+17 feats)")
    _run_cv(X,            y, args.cv_folds, args.seed, "bigram + entropy + dist (35 feats)")
    print("-- Grammar (RF, non-linear) --")
    _run_cv(X[:, bg_idx], y, args.cv_folds, args.seed, "bigram-syntax RF (16+16 feats)", classifier="rf")
    _run_cv(X[:, be_idx], y, args.cv_folds, args.seed, "bigram + entropy RF (17+17 feats)", classifier="rf")
    _run_cv(X,            y, args.cv_folds, args.seed, "bigram + entropy + dist RF (35 feats)", classifier="rf")
    _merge_report(X, y, all_partitions, args.cv_folds, args.seed)

    # -- Distance-matched negatives (honest grammar test) -----------------
    # Negatives are sampled to have the same centroid-distance distribution as
    # positives, so distance carries no signal.  Grammar must do the work.
    print()
    n_pos_md = int(y_md.sum())
    pct_pos_md = 100.0 * n_pos_md / len(y_md)
    print(f"Distance-matched negatives (n={len(y_md):,} pairs, {pct_pos_md:.1f}% positive):")
    print("  [negatives matched to same distance distribution as positives]")
    y_perm_md = np.random.default_rng(args.seed).permutation(y_md)
    print()
    print("-- Baselines --")
    _run_cv(X_md[:, dist_idx], y_md,      args.cv_folds, args.seed, "distance only (1 feat) [should be ~0.5]")
    _run_cv(X_md,              y_perm_md, args.cv_folds, args.seed, "permuted labels (null)")
    print()
    print("-- Grammar (honest, LR) --")
    _run_cv(X_md[:, bg_idx], y_md, args.cv_folds, args.seed, "bigram-syntax (16+16 feats)")
    _run_cv(X_md[:, be_idx], y_md, args.cv_folds, args.seed, "bigram + entropy (17+17 feats)")
    _run_cv(X_md,            y_md, args.cv_folds, args.seed, "bigram + entropy + dist (35 feats)")
    print("-- Grammar (honest, RF) --")
    _run_cv(X_md[:, bg_idx], y_md, args.cv_folds, args.seed, "bigram-syntax RF (16+16 feats)", classifier="rf")
    _run_cv(X_md[:, be_idx], y_md, args.cv_folds, args.seed, "bigram + entropy RF (17+17 feats)", classifier="rf")
    _run_cv(X_md,            y_md, args.cv_folds, args.seed, "bigram + entropy + dist RF (35 feats)", classifier="rf")

    # -- Skeleton grammar (optional, requires CAVE skeleton fetch) -----------
    if args.use_skeleton:
        _run_skeleton_grammar(
            args,
            all_partitions,
            all_box_ids,
            n_boxes_used,
            rng,
        )

    # -- Cross-box analysis (only meaningful when n_boxes > 1) -------------
    if n_boxes_used > 1:
        _cross_box_analysis(
            all_partitions,
            all_box_ids,
            args.cv_folds,
            args.seed,
            np.random.default_rng(args.seed),
            args.max_neg_ratio,
        )

    print()
    print("reference (same region, v117_coassign.py, 120 epochs, d_model=128):")
    print("  GNN  edge P/R = 0.82 / 0.92   partition F1 = 0.76   coverage@5 = False")
    print("  Berlin bigram AUC = 0.95  (n=15, LOO)")
    print("=" * 60)




# ---------------------------------------------------------------------------
# Skeleton grammar (--use-skeleton)
# ---------------------------------------------------------------------------

def _run_skeleton_grammar(args, all_partitions, all_box_ids, n_boxes_used, rng):
    """Fetch skeletons for all roots in all_partitions and run the grammar CV.

    Skeleton fetch is sequential (~1-3 s/root on first call, instant with cache).
    Only roots that appear in all_partitions (those with >=min_synapses) are
    fetched -- typically 500-2000 per 40um box, far fewer than all box roots.

    Both sides (pre+post) share one skeleton per root.  The grammar is computed
    once per root (not per side), using the full skeleton.  Two SkeletonPartitions
    for the same root (pre side and post side) will have the SAME skeleton grammar
    features -- the side distinction is still captured by the synapse-position
    centroid distance used for pairing.
    """
    from experiments.pcfg_synapse_partitions.skeleton_tokens import (
        extract_skeleton_partitions,
        build_skeleton_merge_pairs,
        skeleton_features,
    )
    from neuronauts.fetch import fetch_root_skeletons
    from experiments.pcfg_synapse_partitions.pcfg_partitions import (
        BIGRAM_DIM, FEAT_DIM,
    )

    # Collect unique root IDs across all partitions
    unique_roots = list({p.root_id for p in all_partitions})
    log.info("Fetching skeletons for %d unique roots (v117) ...", len(unique_roots))
    try:
        skeletons = fetch_root_skeletons(
            unique_roots,
            version=117,
            token=args.token,
            cache_dir=args.skeleton_cache_dir,
        )
    except Exception as exc:
        log.warning("Skeleton fetch failed: %s", exc)
        return

    n_ok = sum(1 for sk in skeletons.values() if len(sk.vertices) >= 3)
    log.info("  %d / %d roots have a usable skeleton", n_ok, len(unique_roots))

    # Re-extract partitions using skeleton data
    # We need the original synapse arrays; reconstruct them from all_partitions.
    # Each HalfPartition already holds the synapse pts and remap info we need.
    # Build SkeletonPartitions directly from HalfPartition + skeleton lookup.
    from experiments.pcfg_synapse_partitions.skeleton_tokens import SkeletonPartition
    sk_partitions = []
    for p in all_partitions:
        sk = skeletons.get(p.root_id)
        if sk is None or len(sk.vertices) < 3:
            continue
        sk_partitions.append(SkeletonPartition(
            root_id=p.root_id,
            v18xx_root=p.v18xx_root,
            side=p.side,
            pts=p.pts,
            skel_verts=sk.vertices.astype(float),
            skel_edges=sk.edges,
        ))

    radii = {rid: sk.radius for rid, sk in skeletons.items()
             if sk.radius is not None}

    if len(sk_partitions) < 10:
        print()
        print("Skeleton grammar: too few partitions with skeletons -- skipping.")
        return

    log.info("Building skeleton merge pairs (%d partitions) ...", len(sk_partitions))
    X_sk, y_sk = build_skeleton_merge_pairs(
        sk_partitions, max_neg_ratio=args.max_neg_ratio,
        rng=np.random.default_rng(args.seed), radii=radii,
    )

    if len(y_sk) == 0:
        print()
        print("Skeleton grammar: no merge pairs generated.")
        return

    n_pos_sk = int(y_sk.sum())
    pct_pos_sk = 100.0 * n_pos_sk / len(y_sk)
    bg_idx  = list(range(BIGRAM_DIM)) + list(range(FEAT_DIM, FEAT_DIM + BIGRAM_DIM))
    be_idx  = list(range(FEAT_DIM))   + list(range(FEAT_DIM, FEAT_DIM * 2))
    dist_idx = [X_sk.shape[1] - 1]
    y_perm_sk = np.random.default_rng(args.seed).permutation(y_sk)

    pos_dists_nm_sk = np.expm1(X_sk[y_sk == 1, -1])
    neg_dists_nm_sk = np.expm1(X_sk[y_sk == 0, -1])

    print()
    print("=" * 60)
    print("Skeleton grammar -- DFS-ordered skeleton vertices")
    print("=" * 60)
    print(f"  Partitions with skeleton: {len(sk_partitions)}")
    print(f"  Pairs:                    {len(y_sk):,} ({pct_pos_sk:.1f}% positive)")
    print(f"  Positive centroid dist:   ", end="")
    if len(pos_dists_nm_sk):
        print(f"min={pos_dists_nm_sk.min()/1e3:.1f}  "              f"med={float(np.median(pos_dists_nm_sk))/1e3:.1f}  "              f"max={pos_dists_nm_sk.max()/1e3:.1f} um")
    else:
        print("(none)")
    print(f"  Negative centroid dist:   ", end="")
    if len(neg_dists_nm_sk):
        print(f"min={neg_dists_nm_sk.min()/1e3:.1f}  "              f"med={float(np.median(neg_dists_nm_sk))/1e3:.1f}  "              f"max={neg_dists_nm_sk.max()/1e3:.1f} um")
    else:
        print("(none)")
    print()
    print("-- Baselines --")
    _run_cv(X_sk[:, dist_idx], y_sk,      args.cv_folds, args.seed, "distance only (1 feat)")
    _run_cv(X_sk,              y_perm_sk, args.cv_folds, args.seed, "permuted labels (null)")
    print("-- Skeleton grammar (LR) --")
    _run_cv(X_sk[:, bg_idx], y_sk, args.cv_folds, args.seed, "skeleton bigram (16+16 feats)")
    _run_cv(X_sk[:, be_idx], y_sk, args.cv_folds, args.seed, "skeleton bigram+entropy (17+17)")
    _run_cv(X_sk,            y_sk, args.cv_folds, args.seed, "skeleton + dist (35 feats)")
    print("-- Skeleton grammar (RF) --")
    _run_cv(X_sk[:, bg_idx], y_sk, args.cv_folds, args.seed, "skeleton bigram RF (16+16)",    classifier="rf")
    _run_cv(X_sk[:, be_idx], y_sk, args.cv_folds, args.seed, "skeleton bigram+entropy RF",    classifier="rf")
    _run_cv(X_sk,            y_sk, args.cv_folds, args.seed, "skeleton + dist RF (35 feats)", classifier="rf")

    # Cross-box honest test for skeleton grammar
    if n_boxes_used > 1:
        from collections import defaultdict as _dd
        n = len(sk_partitions)
        skel_feats = [skeleton_features(p, radius=radii.get(p.root_id)) for p in sk_partitions]
        skel_cents = np.array([p.pts.mean(axis=0) for p in sk_partitions])
        skel_v18   = np.array([p.v18xx_root for p in sk_partitions])
        skel_rids  = np.array([p.root_id for p in sk_partitions])
        # box_ids for sk_partitions: match by root_id to original all_partitions ordering
        pid_to_box = {p.root_id: all_box_ids[i] for i, p in enumerate(all_partitions)}
        skel_boxes = np.array([pid_to_box.get(p.root_id, 0) for p in sk_partitions])

        by_v18 = _dd(list)
        for idx, v in enumerate(skel_v18.tolist()):
            by_v18[v].append(idx)

        pos_i, pos_j = [], []
        for group in by_v18.values():
            if len(group) < 2:
                continue
            for ai in range(len(group)):
                for bi in range(ai + 1, len(group)):
                    i, j = group[ai], group[bi]
                    if skel_boxes[i] == skel_boxes[j]:
                        continue
                    if skel_rids[i] == skel_rids[j]:
                        continue
                    pos_i.append(i)
                    pos_j.append(j)

        n_pos_xb = len(pos_i)
        if n_pos_xb == 0:
            print()
            print("Skeleton cross-box: no cross-box positive pairs found.")
        else:
            n_neg_target = min(int(n_pos_xb * args.max_neg_ratio), 100_000)
            neg_i, neg_j = [], []
            batch_sz = 50_000
            while len(neg_i) < n_neg_target:
                ii = rng.integers(0, n, size=batch_sz)
                jj = rng.integers(0, n, size=batch_sz)
                mask = ((ii != jj) & (skel_boxes[ii] != skel_boxes[jj])
                        & (skel_v18[ii] != skel_v18[jj])
                        & (skel_rids[ii] != skel_rids[jj]))
                valid_ii = ii[mask]; valid_jj = jj[mask]
                remaining = n_neg_target - len(neg_i)
                neg_i.extend(valid_ii[:remaining].tolist())
                neg_j.extend(valid_jj[:remaining].tolist())

            fa_arr = np.array(skel_feats)
            pos_ia, pos_ja = np.array(pos_i), np.array(pos_j)
            neg_ia, neg_ja = np.array(neg_i), np.array(neg_j)

            def _pfeat(ia, ja):
                d = np.linalg.norm(skel_cents[ia] - skel_cents[ja], axis=1)
                return np.column_stack([fa_arr[ia], fa_arr[ja], np.log1p(d)])

            X_xb = np.vstack([_pfeat(pos_ia, pos_ja), _pfeat(neg_ia, neg_ja)])
            y_xb = np.concatenate([np.ones(n_pos_xb, dtype=np.int64),
                                    np.zeros(len(neg_i), dtype=np.int64)])
            shuf = rng.permutation(len(y_xb))
            X_xb, y_xb = X_xb[shuf], y_xb[shuf]

            pct_xb = 100.0 * n_pos_xb / len(y_xb)
            pos_d = np.expm1(X_xb[y_xb == 1, -1])
            neg_d = np.expm1(X_xb[y_xb == 0, -1])
            print()
            print(f"Skeleton cross-box honest test (n={len(y_xb):,}, {pct_xb:.1f}% pos):")
            print(f"  Pos dist: med={float(np.median(pos_d))/1e3:.1f} um  "                  f"Neg dist: med={float(np.median(neg_d))/1e3:.1f} um")
            dist_idx_xb = [X_xb.shape[1] - 1]
            bg_idx_xb  = list(range(BIGRAM_DIM)) + list(range(FEAT_DIM, FEAT_DIM + BIGRAM_DIM))
            be_idx_xb  = list(range(FEAT_DIM))   + list(range(FEAT_DIM, FEAT_DIM * 2))
            print()
            _run_cv(X_xb[:, dist_idx_xb], y_xb, args.cv_folds, args.seed, "distance only")
            print("-- LR --")
            _run_cv(X_xb[:, bg_idx_xb], y_xb, args.cv_folds, args.seed, "skeleton bigram (16+16)")
            _run_cv(X_xb[:, be_idx_xb], y_xb, args.cv_folds, args.seed, "skeleton bigram+ent (17+17)")
            _run_cv(X_xb,               y_xb, args.cv_folds, args.seed, "skeleton + dist (35 feats)")
            print("-- RF --")
            _run_cv(X_xb[:, bg_idx_xb], y_xb, args.cv_folds, args.seed, "skeleton bigram RF",       classifier="rf")
            _run_cv(X_xb[:, be_idx_xb], y_xb, args.cv_folds, args.seed, "skeleton bigram+ent RF",   classifier="rf")
            _run_cv(X_xb,               y_xb, args.cv_folds, args.seed, "skeleton + dist RF",       classifier="rf")

if __name__ == "__main__":
    main()
