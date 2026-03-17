#!/usr/bin/env python3
"""Train a lightweight reranker over box-level assembly hypotheses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neuronauts.hypothesis_reranker import (
    LinearRerankerConfig,
    save_linear_reranker,
    train_linear_reranker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to hypothesis dataset .npz.")
    parser.add_argument("--output", default="models/assembly_reranker.npz", help="Output reranker path.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = np.load(args.dataset, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y_f1"].astype(np.float32)
    feature_names = [str(item) for item in data["feature_names"].tolist()]

    model, metrics = train_linear_reranker(
        x,
        y,
        feature_names,
        config=LinearRerankerConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            seed=args.seed,
        ),
    )
    save_linear_reranker(args.output, model)
    metrics_path = Path(args.output).with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved model: {args.output}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
