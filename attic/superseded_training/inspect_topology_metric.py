#!/usr/bin/env python3
"""Inspect topology/atomicity batch balance across cached boxes.

Run this to diagnose why topology accuracy stays flat during training.
If pos_frac is near 0.9 or 0.1, predicting the majority class yields
high accuracy with no learning — the metric is misleading.

Usage:
  python attic/superseded_training/inspect_topology_metric.py --cache-dir data/boxes_v117
  python attic/superseded_training/inspect_topology_metric.py --cache-dir data/boxes_v117 --sample 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuronauts.dataset_builder import BoxCache, load_dataset
from neuronauts.topology_dataset import build_cluster_examples, inspect_topology_batch_balance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cache-dir",
        default="data/boxes",
        help="Box cache directory.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of boxes to sample (default: 10). Use 0 for all.",
    )
    parser.add_argument(
        "--min-synapses",
        type=int,
        default=15,
        help="Minimum synapses per box.",
    )
    parser.add_argument(
        "--max-synapses",
        type=int,
        default=200,
        help="Maximum synapses per box.",
    )
    args = parser.parse_args()

    cache, records = load_dataset(
        args.cache_dir,
        min_synapses=args.min_synapses,
        max_synapses=args.max_synapses,
    )
    if not records:
        print(f"No boxes in {args.cache_dir} with {args.min_synapses}–{args.max_synapses} synapses.")
        return 1

    if args.sample > 0:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(records), size=min(args.sample, len(records)), replace=False)
        records = [records[i] for i in idx]

    print(f"Inspecting topology batch balance across {len(records)} boxes:\n")

    pos_fracs: list[float] = []
    n_atomics: list[int] = []
    n_non_atomics: list[int] = []

    for record in records:
        try:
            _vol, synapses = cache.load(record)
        except Exception as exc:
            print(f"  [W] skip {record.box_hash[:12]}: {exc}")
            continue

        examples = build_cluster_examples(
            synapses,
            membrane_field=np.zeros((1, 1, 1), dtype=np.float32),
            min_cluster_size=2,
            max_negative_pairs_per_role=32,
            max_branches=32,
            seed=42,
        )
        if not examples:
            continue

        stats = inspect_topology_batch_balance(examples)
        pos_fracs.append(stats["pos_frac"])
        n_atomics.append(stats["n_atomic"])
        n_non_atomics.append(stats["n_non_atomic"])

        warn = ""
        if stats["pos_frac"] < 0.3 or stats["pos_frac"] > 0.7:
            warn = "  ⚠ likely trivial accuracy from class imbalance"
        print(f"  {record.box_hash[:14]}  n={stats['n']:4d}  "
              f"atomic={stats['n_atomic']:3d}  non_atomic={stats['n_non_atomic']:3d}  "
              f"pos_frac={stats['pos_frac']:.3f}{warn}")

    if not pos_fracs:
        print("No topology examples produced.")
        return 1

    mean_pos = float(np.mean(pos_fracs))
    print(f"\nSummary: {len(pos_fracs)} boxes")
    print(f"  mean pos_frac = {mean_pos:.3f}")
    if mean_pos < 0.3:
        print("  → Topology batches skew NEGATIVE. Model may predict 0 always → high acc, no learning.")
    elif mean_pos > 0.7:
        print("  → Topology batches skew POSITIVE. Model may predict 1 always → high acc, no learning.")
    else:
        print("  → Balance is reasonable. Flat topo_acc suggests other causes (e.g. task saturates early).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
