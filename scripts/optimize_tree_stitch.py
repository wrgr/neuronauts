#!/usr/bin/env python3
"""Optimize the tree-stitching pipeline on real Minnie65 data.

Loads real proofread neurons from the MICrONS Minnie65 dataset, splits
each skeleton into N pieces to simulate fragmentation, then runs random
hyperparameter search over the full treestitch pipeline.

The objective is ARI (Adjusted Rand Index) — how well the PartitionGNN
recovers the original neuron identities from the fragmented observations.

Usage
-----
  # Quick smoke test (3 trials)
  python scripts/optimize_tree_stitch.py --n-trials 3 --n-objects 10

  # Real optimization run (20 trials, within-type 23P)
  python scripts/optimize_tree_stitch.py \\
      --n-trials 20 --n-objects 20 --n-pieces 3 \\
      --cell-type 23P --device cpu

  # Single pipeline run with specific hyperparams (no search)
  python scripts/optimize_tree_stitch.py \\
      --n-trials 1 --n-objects 20 --n-pieces 3 \\
      --embed-epochs 40 --partition-epochs 40 \\
      --endpoint-radius-nm 10000 --threshold 0.87
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    p.add_argument("--n-objects", type=int, default=20,
                   help="Number of parent trees (neurons) to fetch")
    p.add_argument("--n-pieces", type=int, default=3,
                   help="Fragments per tree (simulated fragmentation level)")
    p.add_argument("--obs-per-piece", type=int, default=12,
                   help="Observations placed near each fragment skeleton")
    p.add_argument("--cell-type", default=None,
                   help="Filter by cell type (e.g. '23P'). None = all types")
    p.add_argument("--synapse-noise-nm", type=float, default=500.0)
    p.add_argument("--max-verts", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    # Search
    p.add_argument("--n-trials", type=int, default=20,
                   help="Number of random configurations to evaluate")
    p.add_argument("--objective", default="ari_after",
                   help="Metric to maximise (ari_after, delta_ari, v_measure)")
    # Fixed overrides — if ALL four are set, search is skipped and a single
    # pipeline run is done instead.
    p.add_argument("--embed-epochs", type=int, default=None)
    p.add_argument("--partition-epochs", type=int, default=None)
    p.add_argument("--endpoint-radius-nm", type=float, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--k-spatial", type=int, default=None)
    # Infra
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    from treestitch.data import load_minnie65_world
    from treestitch.pipeline import optimize, run_pipeline

    # Load data once (shared across all trials)
    fragments, region, label_map = load_minnie65_world(
        n_objects=args.n_objects,
        n_pieces=args.n_pieces,
        observations_per_piece=args.obs_per_piece,
        cell_type=args.cell_type,
        max_verts=args.max_verts,
        synapse_noise_nm=args.synapse_noise_nm,
        seed=args.seed,
    )

    # Check if user provided a full fixed config → single run
    fixed = {k: v for k, v in [
        ("embed_epochs",        args.embed_epochs),
        ("partition_epochs",    args.partition_epochs),
        ("endpoint_radius_nm",  args.endpoint_radius_nm),
        ("threshold",           args.threshold),
        ("k_spatial",           args.k_spatial),
    ] if v is not None}

    if len(fixed) > 0 and args.n_trials == 1:
        # Single run with specified (and default) values
        kwargs = dict(
            embed_epochs=args.embed_epochs or 40,
            partition_epochs=args.partition_epochs or 40,
            endpoint_radius_nm=args.endpoint_radius_nm,
            threshold=args.threshold or 0.87,
            k_spatial=args.k_spatial or 8,
            device=args.device,
            seed=args.seed,
            log_every=args.log_every,
        )
        print(f"\n{'='*60}")
        print(f"Single pipeline run")
        print(f"  {args.n_objects} objects × {args.n_pieces} pieces"
              f"  endpoint_radius={args.endpoint_radius_nm} nm")
        print(f"{'='*60}")
        result = run_pipeline(fragments, region, label_map, **kwargs)
        print(f"\nFinal ARI: {result['ari_before']:.4f} → {result['ari_after']:.4f}"
              f"  (Δ={result['delta_ari']:+.4f})")
        print(f"Clusters: {result['n_clusters_pred']}/{result['n_clusters_true']}"
              f"  H={result['homogeneity']:.3f}  C={result['completeness']:.3f}")
        return 0

    # Random search
    print(f"\n{'='*60}")
    print(f"Hyperparameter search: {args.n_trials} trials")
    print(f"  Objective: {args.objective}")
    print(f"  Data: {args.n_objects} objects × {args.n_pieces} pieces"
          + (f"  cell_type={args.cell_type}" if args.cell_type else ""))
    print(f"{'='*60}")

    best = optimize(
        fragments, region, label_map,
        n_trials=args.n_trials,
        objective=args.objective,
        seed=args.seed,
        device=args.device,
        log_every=args.log_every,
    )

    print(f"\n{'='*60}")
    print(f"Best {args.objective}: {best['best_score']:.4f}")
    print(f"Best config:")
    for k, v in sorted(best["best_config"].items()):
        print(f"  {k:30s} = {v}")
    print(f"\nTop 5 trials:")
    for cfg, score in best["all_results"][:5]:
        ep = cfg.get("endpoint_radius_nm")
        print(f"  {args.objective}={score:.4f}  "
              f"d={cfg['embed_d_model']}  out={cfg['embed_output_dim']}"
              f"  k={cfg['k_spatial']}  ep={ep}  thr={cfg['threshold']}"
              f"  emb_ep={cfg['embed_epochs']}  part_ep={cfg['partition_epochs']}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
