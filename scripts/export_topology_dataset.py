#!/usr/bin/env python3
"""Export a MICRONS/CAVE synapse-cluster atomicity dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuronauts.fetch import RealBoxSpec
from neuronauts.run import REAL_BOXES
from neuronauts.topology_dataset import build_cluster_examples_for_box, save_examples_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/topology_dataset.npz", help="Output dataset path.")
    parser.add_argument(
        "--box-indices",
        default="0,1,2",
        help="Comma-separated indices into neuronauts.run.REAL_BOXES.",
    )
    parser.add_argument("--membrane-source", choices=["auto", "cache", "sobel"], default="auto")
    parser.add_argument("--membrane-cache-dir", default="cache/membranes")
    parser.add_argument("--min-cluster-size", type=int, default=2)
    parser.add_argument("--max-negative-pairs-per-role", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    indices = [int(part.strip()) for part in args.box_indices.split(",") if part.strip()]
    examples = []
    manifest = []
    for box_idx in indices:
        box = REAL_BOXES[box_idx]
        box_examples = build_cluster_examples_for_box(
            box,
            membrane_source=args.membrane_source,
            membrane_cache_dir=args.membrane_cache_dir,
            min_cluster_size=args.min_cluster_size,
            max_negative_pairs_per_role=args.max_negative_pairs_per_role,
            seed=args.seed + box_idx,
        )
        examples.extend(box_examples)
        manifest.append(
            {
                "box_idx": box_idx,
                "center_nm": box.center_nm,
                "side_um": box.side_um,
                "mip": box.mip,
                "examples": len(box_examples),
            }
        )

    save_examples_npz(args.output, examples)
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pos = sum(example.label for example in examples)
    neg = len(examples) - pos
    print(f"saved dataset: {args.output}")
    print(f"examples={len(examples)} positives={pos} negatives={neg}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
