"""PCFG synapse partition experiment runner.

Learns f(v117) -> v18xx merge prediction using Berlin-style bigram grammar
features over synapse half-partitions.  No neural network, no EM volume,
no agent simulation required.

Usage
-----
    python experiments/pcfg_synapse_partitions/run_experiment.py \\
      --cache-dir data/boxes_v117 \\
      --root-remap-tsv data/boxes_v117/root_remap_v117_to_v1412.tsv \\
      [--side both] [--min-synapses 4] [--cv-folds 5] [--seed 42] [--verbose]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Allow running as a script directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neuronauts.dataset_builder import BoxCache
from experiments.pcfg_synapse_partitions.pcfg_partitions import (
    BIGRAM_DIM,
    FEAT_DIM,
    build_merge_pairs,
    extract_partitions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_remap_tsv(path: str) -> dict[int, int]:
    """Load root_base -> root_target TSV (same format as scripts/train.py)."""
    mapping: dict[int, int] = {}
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh, delimiter='\t')
        if reader.fieldnames is None or not {'root_base', 'root_target'}.issubset(
            set(reader.fieldnames)
        ):
            raise SystemExit(
                f'--root-remap-tsv must have columns root_base and root_target. '
                f'Found: {reader.fieldnames}'
            )
        for row in reader:
            mapping[int(row['root_base'])] = int(row['root_target'])
    return mapping


def _run_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    seed: int,
    label: str,
) -> float:
    """Stratified k-fold CV with LogisticRegression; returns mean ROC-AUC."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print(f'  {label:45s} (sklearn not available)')
        return float('nan')

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos < n_folds or n_neg < n_folds:
        print(f'  {label:45s} skipped (too few examples: {n_pos}+/{n_neg}-)')
        return float('nan')

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    probs = np.zeros(len(y), dtype=np.float64)
    for train_idx, val_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(np.nan_to_num(X[train_idx]))
        X_va = scaler.transform(np.nan_to_num(X[val_idx]))
        clf = LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=seed
        )
        clf.fit(X_tr, y[train_idx])
        probs[val_idx] = clf.predict_proba(X_va)[:, 1]

    auc = float(roc_auc_score(y, probs))
    print(f'  {label:45s} CV AUC = {auc:.2f}')
    return auc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='PCFG synapse partition merge-prediction experiment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cache-dir', required=True,
                        help='BoxCache directory built with --cave-version 117')
    parser.add_argument('--root-remap-tsv', required=True,
                        help='TSV with root_base, root_target columns (v117 -> v18xx)')
    parser.add_argument('--side', default='both', choices=['pre', 'post', 'both'],
                        help='Which synapse side to use for half-partitions')
    parser.add_argument('--min-synapses', type=int, default=4,
                        help='Minimum synapses per half-partition')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of stratified CV folds')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-box partition counts')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Load remap ────────────────────────────────────────────────────────
    print(f'Loading root remap from {args.root_remap_tsv} ...', flush=True)
    root_remap = load_remap_tsv(args.root_remap_tsv)
    print(f'  {len(root_remap):,} root ID mappings loaded')

    # ── Load boxes and extract partitions ────────────────────────────────
    cache = BoxCache(args.cache_dir)
    records = cache.all_records()
    print(f'Processing {len(records)} boxes from {args.cache_dir} ...', flush=True)

    all_partitions = []
    n_boxes_used = 0

    for rec in records:
        try:
            _, synapses = cache.load(rec, load_volume=False)
        except Exception as exc:
            if args.verbose:
                print(f'  skip {rec.box_hash[:8]}: {exc}')
            continue

        parts = extract_partitions(
            synapses.pre_pt,
            synapses.post_pt,
            synapses.pre_root_id,
            synapses.post_root_id,
            root_remap,
            min_synapses=args.min_synapses,
            sides=args.side,
        )
        if args.verbose:
            print(f'  box {rec.box_hash[:8]} -> {len(parts)} partitions')
        all_partitions.extend(parts)
        n_boxes_used += 1

    print(
        f'Extracted {len(all_partitions):,} half-partitions '
        f'from {n_boxes_used} boxes'
    )

    if not all_partitions:
        print('No partitions found -- check that cache-dir and root-remap-tsv versions align.')
        return

    # ── Build merge pair dataset ─────────────────────────────────────────
    print('Building merge pair dataset ...', flush=True)
    X, y = build_merge_pairs(all_partitions, rng=rng)

    if len(y) == 0:
        print('No merge pairs generated.')
        return

    n_pos = int(y.sum())
    pct_pos = 100.0 * n_pos / len(y)

    # Count how many positives are real (from remap) vs artificial
    # We can approximate: real positives come from pairs where root_ids differ
    # but v18xx match.  The artificial fallback is used only when real < 2.
    print(f'  {len(y):,} pairs  ({n_pos:,} positive, {pct_pos:.1f}%)')

    # ── Feature slices ──────────────────────────────────────────────────
    # Layout: [bigram_a(16) | entropy_a(1) | bigram_b(16) | entropy_b(1) | log_dist(1)]
    bg_idx = list(range(BIGRAM_DIM)) + list(range(FEAT_DIM, FEAT_DIM + BIGRAM_DIM))
    be_idx = list(range(FEAT_DIM)) + list(range(FEAT_DIM, FEAT_DIM * 2))
    # full: all 35 columns

    # ── Run classifiers ─────────────────────────────────────────────────
    print()
    print('PCFG synapse partition grammar -- merge prediction')
    print(f'(n={len(y):,} pairs from {n_boxes_used} boxes, {pct_pos:.1f}% positive):')

    _run_cv(X[:, bg_idx], y, args.cv_folds, args.seed,
            'bigram-syntax (16+16 feats)')
    _run_cv(X[:, be_idx], y, args.cv_folds, args.seed,
            'bigram + entropy (17+17 feats)')
    _run_cv(X,            y, args.cv_folds, args.seed,
            'bigram + entropy + dist (35 feats)')

    print()
    print('reference: Berlin proofreader bigram AUC = 0.95  (n=15, LOO)')
    print('           Neuronauts PathEncoder merge acc = 0.856  (85 boxes)')


if __name__ == '__main__':
    main()
