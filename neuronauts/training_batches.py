"""Small batch helpers for torch grammar training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathSequenceBatch:
    x: np.ndarray
    mask: np.ndarray


def _downsample_steps(seq: np.ndarray, max_steps: int) -> np.ndarray:
    """Downsample a path sequence to at most max_steps rows.

    Uses deterministic linear indexing (keeps endpoints) to avoid blowing up
    Transformer positional encodings on very large fragments in big boxes.
    """
    seq = np.asarray(seq, dtype=np.float32)
    steps = int(seq.shape[0])
    if steps <= max_steps:
        return seq
    idx = np.linspace(0, steps - 1, num=max_steps, dtype=np.int64)
    return seq[idx]


def pad_path_sequences(
    sequences: list[np.ndarray],
    *,
    feature_dim: int = 3,
    max_steps_cap: int = 512,
) -> PathSequenceBatch:
    """Pad variable-length path descriptor sequences to a dense batch.

    Each input sequence is expected to be shaped ``(steps, feature_dim)``.
    ``mask`` uses ``True`` for padding positions.
    """
    if not sequences:
        return PathSequenceBatch(
            x=np.zeros((0, 0, feature_dim), dtype=np.float32),
            mask=np.zeros((0, 0), dtype=bool),
        )

    # Cap max length to keep the Transformer encoder stable.
    sequences = [_downsample_steps(sequence, max_steps_cap) for sequence in sequences]
    max_steps = max(sequence.shape[0] for sequence in sequences)
    x = np.zeros((len(sequences), max_steps, feature_dim), dtype=np.float32)
    mask = np.ones((len(sequences), max_steps), dtype=bool)
    for idx, sequence in enumerate(sequences):
        seq = np.asarray(sequence, dtype=np.float32)
        steps = seq.shape[0]
        x[idx, :steps, : seq.shape[1]] = seq
        mask[idx, :steps] = False
    return PathSequenceBatch(x=x, mask=mask)


@dataclass(frozen=True)
class NestedPathSequenceBatch:
    x: np.ndarray
    sequence_mask: np.ndarray
    item_mask: np.ndarray


def pad_nested_path_sequences(
    items: list[list[np.ndarray]],
    *,
    max_items: int | None = None,
    feature_dim: int = 3,
    max_steps_cap: int = 512,
) -> NestedPathSequenceBatch:
    """Pad a batch of variable-length lists of variable-length sequences."""
    if not items:
        return NestedPathSequenceBatch(
            x=np.zeros((0, 0, 0, feature_dim), dtype=np.float32),
            sequence_mask=np.zeros((0, 0, 0), dtype=bool),
            item_mask=np.zeros((0, 0), dtype=bool),
        )

    item_cap = max_items if max_items is not None else max(len(group) for group in items)
    max_steps = 0
    for group in items:
        for sequence in group[:item_cap]:
            max_steps = max(
                max_steps,
                int(_downsample_steps(sequence, max_steps_cap).shape[0]),
            )

    x = np.zeros((len(items), item_cap, max_steps, feature_dim), dtype=np.float32)
    sequence_mask = np.ones((len(items), item_cap, max_steps), dtype=bool)
    item_mask = np.ones((len(items), item_cap), dtype=bool)

    for batch_idx, group in enumerate(items):
        for item_idx, sequence in enumerate(group[:item_cap]):
            seq = _downsample_steps(sequence, max_steps_cap)
            steps = seq.shape[0]
            x[batch_idx, item_idx, :steps, : seq.shape[1]] = seq
            sequence_mask[batch_idx, item_idx, :steps] = False
            item_mask[batch_idx, item_idx] = False

    return NestedPathSequenceBatch(x=x, sequence_mask=sequence_mask, item_mask=item_mask)
