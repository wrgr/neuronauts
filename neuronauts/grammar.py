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
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PathBatch:
    """Coordinate-free path descriptors for one or more candidate fragments."""

    edge_len: np.ndarray
    radius: np.ndarray
    curvature: np.ndarray


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
