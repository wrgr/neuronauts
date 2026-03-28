#!/usr/bin/env python3
"""Fetch real MICrONS boxes from CAVE into a portable BoxCache.

This is a convenience wrapper around ``train.py build-dataset`` designed
for the common workflow where data is fetched on one machine (with
network access) and training happens on another (offline / GPU cluster).

Quick start (no CAVE token required for public data)::

    pip install caveclient
    python scripts/fetch_cave_boxes.py \
        --cache-dir data/boxes_cave \
        --n-boxes 80 \
        --no-em

The resulting ``data/boxes_cave/`` directory is self-contained and can be
copied to any machine for training::

    python scripts/train.py train-cell-gnn --cache-dir data/boxes_cave

Strategies
----------
synapse-seeded (default)
    Queries CAVE for real synapse positions and uses those as box
    centres.  Fast, guarantees non-empty boxes.  No token required for
    minnie65_public.

proofread-core
    Samples proofread anchor roots and fetches their local synapse
    neighborhoods.  Highest quality labels but requires a CAVE token
    and is slower (~3 min per root).

Offline transfer
----------------
The cache is stored as plain ``.npz`` + ``.json`` files plus an
``index.json`` manifest.  ``rsync`` or ``tar`` the directory::

    tar czf boxes_cave.tar.gz data/boxes_cave/
    # ... copy to training machine ...
    tar xzf boxes_cave.tar.gz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--cache-dir", required=True,
        help="Output directory for the BoxCache.",
    )
    ap.add_argument(
        "--n-boxes", type=int, default=80,
        help="Number of boxes to fetch (default 80).",
    )
    ap.add_argument(
        "--strategy", default="synapse-seeded",
        choices=["synapse-seeded", "proofread-core"],
        help="Box selection strategy.",
    )
    ap.add_argument(
        "--box-side-um", type=float, default=30.0,
        help="Box side length in microns (default 30).",
    )
    ap.add_argument(
        "--min-synapses", type=int, default=15,
        help="Minimum synapses per box.",
    )
    ap.add_argument(
        "--max-synapses", type=int, default=20000,
        help="Maximum synapses per box.",
    )
    ap.add_argument(
        "--min-positive-pairs", type=int, default=5,
        help="Minimum same-root pairs per box (default 5).",
    )
    ap.add_argument(
        "--no-em", action="store_true", default=True,
        help="Skip EM volume fetch (default: skip). Grammar and CellGNN "
             "training only need synapse geometry.",
    )
    ap.add_argument(
        "--with-em", dest="no_em", action="store_false",
        help="Also fetch EM volumes (slow, needed only for GAT/agent training).",
    )
    ap.add_argument(
        "--cave-version", type=int, default=1412,
        help="CAVE materialization version.",
    )
    ap.add_argument(
        "--cave-token", default=None,
        help="CAVE auth token (not needed for public minnie65_public).",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
    )
    # proofread-core specific
    ap.add_argument(
        "--proofread-n-roots", type=int, default=25,
        help="Number of proofread roots to sample (proofread-core only).",
    )
    ap.add_argument(
        "--proofread-radius-um", type=float, default=40.0,
        help="Neighborhood radius in microns (proofread-core only).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be fetched without actually fetching.",
    )
    args = ap.parse_args(argv)

    # Check dependencies
    try:
        import caveclient  # noqa: F401
    except ImportError:
        print(
            "caveclient is required for CAVE access.  Install with:\n"
            "  pip install caveclient\n"
            "\n"
            "No CAVE token is needed for the public minnie65_public datastack."
        )
        return 1

    from neuronauts.dataset_builder import BoxCache

    cache = BoxCache(args.cache_dir)
    existing = len(cache)
    print(f"Cache: {args.cache_dir}  ({existing} existing boxes)")
    print(f"Strategy: {args.strategy}")
    print(f"Target: {args.n_boxes} boxes, {args.box_side_um} µm side, "
          f"≥{args.min_synapses} syn, ≥{args.min_positive_pairs} pos pairs")
    print(f"CAVE version: v{args.cave_version}  "
          f"EM: {'yes' if not args.no_em else 'no (synapse-only)'}")

    if args.dry_run:
        print("\n[dry-run] Would fetch boxes with the above settings.")
        return 0

    # Delegate to train.py build-dataset
    from scripts.train import cmd_build_dataset
    import types

    build_args = types.SimpleNamespace(
        cache_dir=args.cache_dir,
        n_boxes=args.n_boxes,
        strategy=args.strategy,
        box_side_um=args.box_side_um,
        min_synapses=args.min_synapses,
        max_synapses=args.max_synapses,
        min_positive_pairs=args.min_positive_pairs,
        no_em=args.no_em,
        seed=args.seed,
        cave_token=args.cave_token,
        cave_version=args.cave_version,
        counts_tsv=None,
        nucleus_csv=None,
        # proofread-core args
        proofread_datastack="minnie65_public",
        proofread_n_roots=args.proofread_n_roots,
        proofread_roots_tsv=None,
        proofread_radius_um=args.proofread_radius_um,
        proofread_anchor_side="both",
        proofread_min_anchor_synapses=50,
        proofread_per_root_timeout_s=180,
        proofread_require_dendrite=True,
        proofread_require_axon=False,
    )

    t0 = time.time()
    rc = cmd_build_dataset(build_args)
    elapsed = time.time() - t0

    if rc == 0:
        refreshed = BoxCache(args.cache_dir)
        new_boxes = len(refreshed) - existing
        print(f"\nFetched {new_boxes} new boxes in {elapsed:.0f}s")
        print(f"Total: {len(refreshed)} boxes in {args.cache_dir}")
        print(f"\nTo train CellGNN:")
        print(f"  python scripts/train.py train-cell-gnn "
              f"--cache-dir {args.cache_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
