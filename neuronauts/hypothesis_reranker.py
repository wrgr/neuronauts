"""Lightweight reranker for box-level assembly hypotheses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LinearRerankerConfig:
    learning_rate: float = 0.05
    weight_decay: float = 1e-4
    epochs: int = 500
    seed: int = 42


@dataclass(frozen=True)
class LinearReranker:
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float
    feature_names: list[str]

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_norm = (x - self.mean) / self.std
        return x_norm @ self.weights + self.bias


def _safe_standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return (x - mean) / std, mean.astype(np.float32), std.astype(np.float32)


def train_linear_reranker(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    config: LinearRerankerConfig | None = None,
) -> tuple[LinearReranker, dict[str, float]]:
    config = config or LinearRerankerConfig()
    rng = np.random.default_rng(config.seed)
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x_norm, mean, std = _safe_standardize(x)
    weights = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float32)
    bias = 0.0
    n = max(1, len(x_norm))

    for _ in range(config.epochs):
        preds = x_norm @ weights + bias
        error = preds - y
        grad_w = (x_norm.T @ error) / n + config.weight_decay * weights
        grad_b = float(error.mean())
        weights -= config.learning_rate * grad_w
        bias -= config.learning_rate * grad_b

    model = LinearReranker(mean=mean, std=std, weights=weights, bias=float(bias), feature_names=feature_names)
    preds = model.predict(x)
    metrics = {
        "mse": float(np.mean((preds - y) ** 2)),
        "corr": float(np.corrcoef(preds, y)[0, 1]) if len(y) > 1 and np.std(preds) > 0 and np.std(y) > 0 else 0.0,
    }
    return model, metrics


def save_linear_reranker(path: str | Path, model: LinearReranker) -> None:
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


def load_linear_reranker(path: str | Path) -> LinearReranker:
    data = np.load(path, allow_pickle=True)
    return LinearReranker(
        mean=data["mean"].astype(np.float32),
        std=data["std"].astype(np.float32),
        weights=data["weights"].astype(np.float32),
        bias=float(data["bias"][0]),
        feature_names=[str(item) for item in data["feature_names"].tolist()],
    )
