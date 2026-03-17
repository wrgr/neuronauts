#!/usr/bin/env python3
"""Train a simple synapse-cluster atomicity model from an exported dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neuronauts.topology_model import LogisticConfig, save_logistic_model, train_logistic_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to exported .npz dataset.")
    parser.add_argument("--output", default="models/topology_atomicity_model.npz", help="Model output path.")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = np.load(args.dataset, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64)
    feature_names = [str(name) for name in data["feature_names"].tolist()]

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - args.val_fraction)))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(indices) else indices[-1:]

    config = LogisticConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        seed=args.seed,
    )
    model, train_metrics = train_logistic_model(x[train_idx], y[train_idx], feature_names, config=config)
    val_probs = model.predict_proba(x[val_idx])
    val_preds = (val_probs >= 0.5).astype(np.int64)
    val_metrics = {
        "accuracy": float((val_preds == y[val_idx]).mean()),
        "loss": float(-(y[val_idx] * np.log(np.clip(val_probs, 1e-6, 1 - 1e-6)) + (1 - y[val_idx]) * np.log(np.clip(1 - val_probs, 1e-6, 1 - 1e-6))).mean()),
    }
    save_logistic_model(args.output, model)
    metrics_path = Path(args.output).with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "train": train_metrics,
                "val": val_metrics,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved model: {args.output}")
    print(json.dumps({"train": train_metrics, "val": val_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
