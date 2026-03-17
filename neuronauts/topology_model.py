"""Minimal learned models for topology validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LogisticConfig:
    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    epochs: int = 500
    seed: int = 42


@dataclass(frozen=True)
class LogisticModel:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float
    feature_names: list[str]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x_norm = (x - self.mean) / self.std
        logits = x_norm @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(np.int64)


def _safe_standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def train_logistic_model(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    config: LogisticConfig | None = None,
) -> tuple[LogisticModel, dict[str, float]]:
    config = config or LogisticConfig()
    rng = np.random.default_rng(config.seed)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x_norm, mean, std = _safe_standardize(x)

    weights = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float32)
    bias = 0.0
    n = max(1, len(x_norm))

    for _ in range(config.epochs):
        logits = x_norm @ weights + bias
        probs = 1.0 / (1.0 + np.exp(-logits))
        error = probs - y
        grad_w = (x_norm.T @ error) / n + config.weight_decay * weights
        grad_b = float(error.mean())
        weights -= config.learning_rate * grad_w
        bias -= config.learning_rate * grad_b

    model = LogisticModel(mean=mean, std=std, weights=weights, bias=float(bias), feature_names=feature_names)
    probs = model.predict_proba(x)
    preds = (probs >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float((preds == y.astype(np.int64)).mean()),
        "loss": float(_binary_cross_entropy(probs, y)),
        "auroc": float(binary_auroc(probs, y.astype(np.int64))),
    }
    return model, metrics


def _binary_cross_entropy(probs: np.ndarray, targets: np.ndarray) -> float:
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    return float(-(targets * np.log(probs) + (1 - targets) * np.log(1 - probs)).mean())


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[pos].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def save_logistic_model(path: str | Path, model: LogisticModel) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=model.mean.astype(np.float32),
        std=model.std.astype(np.float32),
        weights=model.weights.astype(np.float32),
        bias=np.array([model.bias], dtype=np.float32),
        feature_names=np.array(model.feature_names, dtype=object),
    )


def load_logistic_model(path: str | Path) -> LogisticModel:
    data = np.load(path, allow_pickle=True)
    return LogisticModel(
        mean=data["mean"].astype(np.float32),
        std=data["std"].astype(np.float32),
        weights=data["weights"].astype(np.float32),
        bias=float(data["bias"][0]),
        feature_names=[str(name) for name in data["feature_names"].tolist()],
    )
