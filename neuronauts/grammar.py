"""Shared connectome grammar models.

This module holds the main coordinate-free representation surfaces used by the
topology dataset path and, eventually, the global assembly path:

- ``PathEncoder``: sequential path-profile encoder
- ``MergeScorer``: pairwise fragment compatibility
- ``ArborEncoder``: cluster/arbor summarizer over multiple fragments

The implementation here remains lightweight and numpy-based so the package
stays runnable without optional training dependencies. The topology training
stack in ``neuronauts.topology_model`` consumes the embeddings exported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PathBatch:
    """Coordinate-free path descriptors for one or more candidate fragments."""

    edge_len: np.ndarray
    radius: np.ndarray
    curvature: np.ndarray


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("pip install torch or pip install -e .[topology]") from exc
    return torch, nn


class PathEncoder:
    """Sequential path encoder over raw path descriptors.

    Rather than collapsing an entire path into one global mean/std summary, the
    encoder keeps coarse beginning/middle/end structure by splitting the
    sequence into thirds and summarizing each segment independently.
    """

    def __init__(self, output_dim: int = 32) -> None:
        self.output_dim = int(output_dim)

    def encode(self, batch: PathBatch) -> np.ndarray:
        if batch.edge_len.size == 0:
            return np.zeros(self.output_dim, dtype=np.float32)

        stacked = np.stack([batch.edge_len, batch.radius, batch.curvature], axis=-1)
        parts = np.array_split(stacked, 3, axis=0)
        features = []
        for part in parts:
            if len(part) == 0:
                features.append(np.zeros(3, dtype=np.float32))
                features.append(np.zeros(3, dtype=np.float32))
                continue
            features.append(part.mean(axis=0).astype(np.float32, copy=False))
            features.append(part.std(axis=0).astype(np.float32, copy=False))

        feature_vec = np.concatenate(features, axis=0)
        if feature_vec.size >= self.output_dim:
            return feature_vec[: self.output_dim].astype(np.float32, copy=False)
        padded = np.zeros(self.output_dim, dtype=np.float32)
        padded[: feature_vec.size] = feature_vec.astype(np.float32, copy=False)
        return padded


class MergeScorer:
    """Baseline pairwise compatibility scorer over two path embeddings."""

    def score(self, left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 0.0:
            return 0.0
        return float(np.dot(left, right) / denom)


class ArborEncoder:
    """Cluster-level summarizer for global atomicity decisions.

    Mean pooling keeps the dominant structural trend while max pooling preserves
    sharper fragment-level motifs. This stays permutation-invariant and avoids
    introducing a quadratic attention dependency into the export path itself.
    """

    def __init__(self, output_dim: int = 64) -> None:
        self.output_dim = int(output_dim)

    def encode(self, embeddings: Sequence[np.ndarray]) -> np.ndarray:
        if not embeddings:
            return np.zeros(self.output_dim, dtype=np.float32)
        matrix = np.stack([np.asarray(item, dtype=np.float32) for item in embeddings], axis=0)
        mean_pool = matrix.mean(axis=0)
        max_pool = matrix.max(axis=0)
        feature_vec = np.concatenate([mean_pool, max_pool], axis=0)
        if feature_vec.size >= self.output_dim:
            return feature_vec[: self.output_dim].astype(np.float32, copy=False)
        padded = np.zeros(self.output_dim, dtype=np.float32)
        padded[: feature_vec.size] = feature_vec.astype(np.float32, copy=False)
        return padded


class TorchPathEncoder:
    """Factory for a torch-native sequential path encoder."""

    def __new__(cls, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 32):
        torch, nn = _require_torch()

        class _TorchPathEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.hidden_dim = int(hidden_dim)
                self.output_dim = int(output_dim)
                self._init_kwargs = {
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                }
                self.proj = nn.Sequential(
                    nn.Linear(self.input_dim * 6, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.output_dim),
                )

            def forward(self, x, mask=None):
                x = x.float()
                if mask is None:
                    mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
                valid = (~mask).float().unsqueeze(-1)
                pooled_parts = []
                n_steps = x.shape[1]
                boundaries = [0, n_steps // 3, (2 * n_steps) // 3, n_steps]
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    chunk = x[:, start:end, :]
                    chunk_valid = valid[:, start:end, :]
                    denom = chunk_valid.sum(dim=1).clamp_min(1.0)
                    mean = (chunk * chunk_valid).sum(dim=1) / denom
                    centered = (chunk - mean.unsqueeze(1)) * chunk_valid
                    var = (centered.pow(2).sum(dim=1) / denom).clamp_min(0.0)
                    std = torch.sqrt(var)
                    pooled_parts.extend([mean, std])
                features = torch.cat(pooled_parts, dim=-1)
                return self.proj(features)

        return _TorchPathEncoder()


class TorchMergeScorer:
    """Factory for a torch-native pairwise merge scorer."""

    def __new__(cls, embedding_dim: int = 32, hidden_dim: int = 64):
        torch, nn = _require_torch()

        class _TorchMergeScorer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding_dim = int(embedding_dim)
                self.hidden_dim = int(hidden_dim)
                self._init_kwargs = {
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
                }
                self.net = nn.Sequential(
                    nn.Linear(self.embedding_dim * 4, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1),
                )

            def forward(self, left, right):
                features = torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)
                return self.net(features).squeeze(-1)

        return _TorchMergeScorer()


class TorchArborEncoder:
    """Factory for a torch-native permutation-invariant arbor summarizer."""

    def __new__(cls, embedding_dim: int = 32, hidden_dim: int = 64, output_dim: int = 64):
        torch, nn = _require_torch()

        class _TorchArborEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding_dim = int(embedding_dim)
                self.hidden_dim = int(hidden_dim)
                self.output_dim = int(output_dim)
                self._init_kwargs = {
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                }
                self.proj = nn.Sequential(
                    nn.Linear(self.embedding_dim * 2, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.output_dim),
                )

            def forward(self, x, mask=None):
                x = x.float()
                if mask is None:
                    mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
                valid = (~mask).float().unsqueeze(-1)
                denom = valid.sum(dim=1).clamp_min(1.0)
                mean = (x * valid).sum(dim=1) / denom
                max_ready = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
                max_pool = max_ready.max(dim=1).values
                max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
                features = torch.cat([mean, max_pool], dim=-1)
                return self.proj(features)

        return _TorchArborEncoder()


def save_torch_grammar_component(path: str | Path, model) -> None:
    torch, _, = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_name": model.__class__.__name__,
            "init_kwargs": dict(getattr(model, "_init_kwargs", {})),
        },
        path,
    )


def load_torch_grammar_component(path: str | Path, factory):
    torch, _ = _require_torch()
    checkpoint = torch.load(path, map_location="cpu")
    model = factory(**checkpoint.get("init_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build_path_batch(
    edge_len: Iterable[float],
    radius: Iterable[float],
    curvature: Iterable[float],
) -> PathBatch:
    return PathBatch(
        edge_len=np.asarray(list(edge_len), dtype=np.float32),
        radius=np.asarray(list(radius), dtype=np.float32),
        curvature=np.asarray(list(curvature), dtype=np.float32),
    )
