"""Spatial train / validation split with a seam buffer.

Learned baselines must be fit on tissue they are not scored on. A random split
over atoms leaks: two atoms of one neuron land on both sides and the model
learns that neuron. Splitting along one axis with a buffer wider than a
candidate radius removes that leak for the pair task -- no candidate pair can
straddle the seam -- and leaves both halves dense enough to hold many neurons.

Atoms are assigned by centroid. An arbor can cross the seam; such an atom is
still assigned to one side, and a pair is kept only when both atoms are on
the same side, so a cross-seam pair never trains or tests anything.
"""

from __future__ import annotations

import numpy as np

SPLIT_BUFFER, SPLIT_TRAIN, SPLIT_VAL = -1, 0, 1


def assign_split(centroid_nm: np.ndarray, *, axis: int = 0,
                 centre_nm: float, buffer_nm: float) -> np.ndarray:
    """Side of the seam for each row: 0 train, 1 val, -1 inside the buffer."""
    c = np.asarray(centroid_nm, np.float64)[:, axis]
    out = np.full(len(c), SPLIT_BUFFER, np.int8)
    out[c < centre_nm - buffer_nm / 2.0] = SPLIT_TRAIN
    out[c >= centre_nm + buffer_nm / 2.0] = SPLIT_VAL
    return out


def pair_split(split_a: np.ndarray, split_b: np.ndarray) -> np.ndarray:
    """Split of a pair: the shared side, or -1 when the pair straddles."""
    a = np.asarray(split_a, np.int8)
    b = np.asarray(split_b, np.int8)
    return np.where((a == b) & (a >= 0), a, SPLIT_BUFFER).astype(np.int8)


def describe(split: np.ndarray) -> dict:
    s = np.asarray(split)
    return {"n_train": int((s == SPLIT_TRAIN).sum()),
            "n_val": int((s == SPLIT_VAL).sum()),
            "n_buffer": int((s == SPLIT_BUFFER).sum())}
