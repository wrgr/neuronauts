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


def _fetch_one_box(
    center_nm: tuple[int, ...],
    side_um: float,
    token: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, int]] | None:
    """Fetch synapses + v117->v1412 remap for one box.

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

    # Map all unique root IDs to v1412
    all_roots = list(
        set(syn.pre_root_id.tolist()) | set(syn.post_root_id.tolist())
    )
    log.info("  Mapping %d unique root IDs v117 -> v1412 ...", len(all_roots))
    try:
        remap = map_roots_between_versions(all_roots, 117, 1412, token=token)
    except Exception as exc:
        log.warning("Root mapping failed for center %s: %s", center_nm, exc)
        return None

    n_mapped = sum(1 for v in remap.values() if v > 0)
    log.info("  %d / %d roots have a valid v1412 label", n_mapped, len(all_roots))

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

    Cross-box positives: same v1412 root, partitions from *different* boxes.
    Their centroid distances are on the order of box offsets (~30 um), which
    overlaps the negative pair range — so distance carries no free signal and
    the grammar must do the work.
    """
    from itertools import combinations as _comb
    from experiments.pcfg_synapse_partitions.pcfg_partitions import (
        partition_features, BIGRAM_DIM, FEAT_DIM,
    )

    feats     = [partition_features(p) for p in all_partitions]
    centroids = [p.pts.mean(axis=0) for p in all_partitions]
    v18xx     = [p.v18xx_root for p in all_partitions]

    pos_rows: list = []
    neg_rows: list = []

    for i, j in _comb(range(len(all_partitions)), 2):
        if box_ids[i] == box_ids[j]:
            continue  # same box — skip
        if all_partitions[i].root_id == all_partitions[j].root_id:
            continue  # identical root ID (shouldn't happen across boxes but guard it)
        dist = float(np.linalg.norm(centroids[i] - centroids[j]))
        if v18xx[i] == v18xx[j]:
            pos_rows.append((feats[i], feats[j], dist, 1))
        else:
            neg_rows.append((feats[i], feats[j], dist, 0))

    n_pos_xb = len(pos_rows)
    if n_pos_xb == 0:
        print()
        print("Cross-box analysis: no cross-box positive pairs found.")
        print("  (no v1412 root appears in more than one box)")
        return

    n_neg = min(len(neg_rows), max(1, int(n_pos_xb * max_neg_ratio)))
    neg_rows.sort(key=lambda r: r[2])
    neg_rows = neg_rows[:n_neg]

    all_rows = pos_rows + neg_rows
    order = rng.permutation(len(all_rows))
    all_rows = [all_rows[k] for k in order]

    X_xb = np.array(
        [np.concatenate([fa, fb, [np.log1p(d)]]) for fa, fb, d, _ in all_rows],
        dtype=np.float64,
    )
    y_xb = np.array([lbl for _, _, _, lbl in all_rows], dtype=np.int64)

    bg_idx   = list(range(BIGRAM_DIM)) + list(range(FEAT_DIM, FEAT_DIM + BIGRAM_DIM))
    be_idx   = list(range(FEAT_DIM))   + list(range(FEAT_DIM, FEAT_DIM * 2))
    dist_idx = [X_xb.shape[1] - 1]

    pct_pos = 100.0 * n_pos_xb / len(y_xb)
    pos_dists = np.array([r[2] for r in pos_rows])
    neg_dists = np.array([r[2] for r in neg_rows])

    print()
    print(f"Cross-box pairs only (n={len(y_xb):,} pairs, {pct_pos:.1f}% positive):")
    print(f"  [honest grammar test: same-neuron fragments in spatially distinct boxes]")
    print(f"  Positive dist: min={pos_dists.min()/1e3:.1f}  "
          f"med={np.median(pos_dists)/1e3:.1f}  "
          f"max={pos_dists.max()/1e3:.1f} µm")
    print(f"  Negative dist: min={neg_dists.min()/1e3:.1f}  "
          f"med={np.median(neg_dists)/1e3:.1f}  "
          f"max={neg_dists.max()/1e3:.1f} µm")
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
            # Count unique v117 root IDs per v1412 root (true false-split pairs).
            # Note: sides='both' creates 2 partitions per v117 root (pre+post), both
            # sharing the same root_id — those are NOT positive pairs. Only partitions
            # with *different* root_ids mapping to the same v1412 root are positives.
            v18xx_by_root: dict = {}
            for p in parts:
                v18xx_by_root.setdefault(p.v18xx_root, set()).add(p.root_id)
            n_multi = sum(1 for roots in v18xx_by_root.values() if len(roots) >= 2)
            log.info(
                "    (%d v1412 roots with >=2 distinct v117 root IDs = potential positives)",
                n_multi,
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


if __name__ == "__main__":
    main()
