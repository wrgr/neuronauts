"""Shared connectome grammar models for the unified neuronauts repo.

This module is the intended home for the learned middle layer:

- PathEncoder: local neurite path representation
- MergeScorer: pairwise fragment compatibility
- ArborEncoder: cluster atomicity / global arbor validity

The initial implementation is intentionally small. It provides stable class
boundaries for the optimizer and docs while the learned internals evolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PathBatch:
    """Coordinate-free path descriptors for one or more candidate fragments."""

    edge_len: np.ndarray
    radius: np.ndarray
    curvature: np.ndarray


class PathEncoder:
    """Simple baseline encoder over raw path descriptors."""

    def __init__(self, output_dim: int = 32) -> None:
        self.output_dim = int(output_dim)

    def encode(self, batch: PathBatch) -> np.ndarray:
        stacked = np.stack([batch.edge_len, batch.radius, batch.curvature], axis=-1)
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        features = np.concatenate([mean, std], axis=0)
        if features.size >= self.output_dim:
            return features[: self.output_dim].astype(np.float32, copy=False)
        padded = np.zeros(self.output_dim, dtype=np.float32)
        padded[: features.size] = features.astype(np.float32, copy=False)
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
    """Baseline cluster-level summarizer for global atomicity decisions."""

    def encode(self, embeddings: Sequence[np.ndarray]) -> np.ndarray:
        if not embeddings:
            return np.zeros(32, dtype=np.float32)
        matrix = np.stack([np.asarray(item, dtype=np.float32) for item in embeddings], axis=0)
        return matrix.mean(axis=0).astype(np.float32, copy=False)


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
