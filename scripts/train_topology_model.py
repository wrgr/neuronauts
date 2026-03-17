#!/usr/bin/env python3
"""Train the attention-based arbor validator on a multi-branch dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neuronauts.topology_model import AttentionArborValidator, TrainingConfig, save_validator, train_iteration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to exported multi-branch .npz dataset.")
    parser.add_argument("--output", default="models/topology_atomicity_model.pt", help="Model output path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def _bce_loss(probs, targets) -> float:
    probs_np = np.clip(np.asarray(probs, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    targets_np = np.asarray(targets, dtype=np.float32)
    return float(-(targets_np * np.log(probs_np) + (1.0 - targets_np) * np.log(1.0 - probs_np)).mean())


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required for topology training; install with `pip install -e .[topology]`.") from exc

    data = np.load(args.dataset, allow_pickle=True)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    mask = data["mask"].astype(bool)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(x))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - args.val_fraction)))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(indices) else indices[-1:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    torch.manual_seed(config.seed)

    embed_dim = int(x.shape[2])
    model = AttentionArborValidator(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    x_train = torch.from_numpy(x[train_idx]).to(device)
    y_train = torch.from_numpy(y[train_idx]).to(device)
    mask_train = torch.from_numpy(mask[train_idx]).to(device)

    for _ in range(config.epochs):
        batch_order = torch.randperm(len(x_train), device=device)
        for start in range(0, len(batch_order), config.batch_size):
            batch_ids = batch_order[start : start + config.batch_size]
            train_iteration(
                model,
                optimizer,
                x_train[batch_ids],
                y_train[batch_ids],
                mask=mask_train[batch_ids],
            )

    model.eval()
    x_val = torch.from_numpy(x[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)
    mask_val = torch.from_numpy(mask[val_idx]).to(device)

    with torch.no_grad():
        train_probs = model(x_train, mask=mask_train).squeeze(-1).detach().cpu().numpy()
        val_probs = model(x_val, mask=mask_val).squeeze(-1).detach().cpu().numpy()

    train_preds = (train_probs >= 0.5).astype(np.int64)
    val_preds = (val_probs >= 0.5).astype(np.int64)
    train_targets = y[train_idx].astype(np.int64)
    val_targets = y[val_idx].astype(np.int64)
    train_metrics = {
        "accuracy": float((train_preds == train_targets).mean()),
        "loss": _bce_loss(train_probs, train_targets),
    }
    val_metrics = {
        "accuracy": float((val_preds == val_targets).mean()),
        "loss": _bce_loss(val_probs, val_targets),
    }

    save_validator(args.output, model, embed_dim=embed_dim)
    metrics_path = Path(args.output).with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "train": train_metrics,
                "val": val_metrics,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "embed_dim": embed_dim,
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
