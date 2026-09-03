#!/usr/bin/env python3
"""Train one shared grammar on local merge and global atomicity supervision.

Thread-specific helper for the **grammar** experiment (see
``experiments/README.md``). The primary CLI is ``scripts/train.py``; this script
is kept for the standalone shared-grammar training flow it implements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neuronauts.grammar import (
    DEFAULT_PATH_FEATURE_MODE,
    LEGACY_PATH_FEATURE_MODE,
    path_feature_names,
)
from neuronauts.shared_grammar_model import (
    SharedGrammarModel,
    SharedTrainingConfig,
    multitask_train_step,
    save_shared_grammar_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-dataset", required=True, help="Path to merge dataset .npz.")
    parser.add_argument("--topology-dataset", required=True, help="Path to topology dataset .npz.")
    parser.add_argument("--output", default="models/shared_grammar_model.pt", help="Model output path.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--merge-loss-weight", type=float, default=1.0)
    parser.add_argument("--atomicity-loss-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _accuracy_from_logits(logits, targets) -> float:
    preds = (np.asarray(logits) >= 0.0).astype(np.int64)
    return float((preds == np.asarray(targets, dtype=np.int64)).mean())


def main() -> int:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("torch is required; install with `pip install -e .[topology]`.") from exc

    merge = np.load(args.merge_dataset, allow_pickle=True)
    topo = np.load(args.topology_dataset, allow_pickle=True)

    config = SharedTrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        merge_loss_weight=args.merge_loss_weight,
        atomicity_loss_weight=args.atomicity_loss_weight,
        seed=args.seed,
    )
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = int(merge["left_x"].shape[-1])
    feature_names = tuple(merge["feature_names"].tolist()) if "feature_names" in merge.files else ()
    if feature_names == path_feature_names(DEFAULT_PATH_FEATURE_MODE):
        path_feature_mode = DEFAULT_PATH_FEATURE_MODE
    elif feature_names == path_feature_names(LEGACY_PATH_FEATURE_MODE):
        path_feature_mode = LEGACY_PATH_FEATURE_MODE
    else:
        path_feature_mode = DEFAULT_PATH_FEATURE_MODE if input_dim != 3 else LEGACY_PATH_FEATURE_MODE

    model = SharedGrammarModel(
        input_dim=input_dim,
        path_feature_mode=path_feature_mode,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    merge_size = len(merge["y"])
    topo_size = len(topo["y"])
    if merge_size == 0 or topo_size == 0:
        raise SystemExit("Both merge and topology datasets must be non-empty.")

    last_metrics = {}
    for _ in range(config.epochs):
        merge_order = rng.permutation(merge_size)
        topo_order = rng.permutation(topo_size)
        steps = max((merge_size + config.batch_size - 1) // config.batch_size, (topo_size + config.batch_size - 1) // config.batch_size)
        for step in range(steps):
            merge_ids = merge_order[(step * config.batch_size) % merge_size : ((step + 1) * config.batch_size) % merge_size]
            if len(merge_ids) == 0:
                merge_ids = merge_order[: config.batch_size]
            topo_ids = topo_order[(step * config.batch_size) % topo_size : ((step + 1) * config.batch_size) % topo_size]
            if len(topo_ids) == 0:
                topo_ids = topo_order[: config.batch_size]

            merge_batch = {
                "left_x": torch.from_numpy(merge["left_x"][merge_ids]).to(device),
                "left_mask": torch.from_numpy(merge["left_mask"][merge_ids]).to(device),
                "right_x": torch.from_numpy(merge["right_x"][merge_ids]).to(device),
                "right_mask": torch.from_numpy(merge["right_mask"][merge_ids]).to(device),
                "y": torch.from_numpy(merge["y"][merge_ids].astype(np.float32)).to(device),
            }
            topology_batch = {
                "branch_x": torch.from_numpy(topo["branch_x"][topo_ids]).to(device),
                "branch_sequence_mask": torch.from_numpy(topo["branch_sequence_mask"][topo_ids]).to(device),
                "branch_mask": torch.from_numpy(topo["branch_mask"][topo_ids]).to(device),
                "y": torch.from_numpy(topo["y"][topo_ids].astype(np.float32)).to(device),
            }
            last_metrics = multitask_train_step(
                model,
                optimizer,
                merge_batch=merge_batch,
                topology_batch=topology_batch,
                merge_loss_weight=config.merge_loss_weight,
                atomicity_loss_weight=config.atomicity_loss_weight,
            )

    model.eval()
    with torch.no_grad():
        merge_logits = model.score_merge(
            torch.from_numpy(merge["left_x"]).to(device),
            torch.from_numpy(merge["left_mask"]).to(device),
            torch.from_numpy(merge["right_x"]).to(device),
            torch.from_numpy(merge["right_mask"]).to(device),
        ).detach().cpu().numpy()
        topo_logits = model.score_atomicity(
            torch.from_numpy(topo["branch_x"]).to(device),
            torch.from_numpy(topo["branch_sequence_mask"]).to(device),
            torch.from_numpy(topo["branch_mask"]).to(device),
        ).detach().cpu().numpy()

    metrics = {
        "last_step": last_metrics,
        "merge_accuracy": _accuracy_from_logits(merge_logits, merge["y"]),
        "atomicity_accuracy": _accuracy_from_logits(topo_logits, topo["y"]),
        "n_merge": int(merge_size),
        "n_topology": int(topo_size),
    }

    save_shared_grammar_model(args.output, model)
    metrics_path = Path(args.output).with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved model: {args.output}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
