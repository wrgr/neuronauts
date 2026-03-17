#!/usr/bin/env python3
"""Train the membrane U-Net from a tif-based dataset repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuronauts.membrane_unet import TrainingConfig, train_membrane_unet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Path to the dataset repo root with train/images and train/labels.")
    parser.add_argument("--output", default="models/membrane_unet.pt", help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        seed=args.seed,
        base_channels=args.base_channels,
    )
    metrics = train_membrane_unet(args.dataset_dir, args.output, config=config)
    metrics_path = Path(args.output).with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved checkpoint: {args.output}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
