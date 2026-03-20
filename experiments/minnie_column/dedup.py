"""Dedup keys and overlap weights when synapses appear in multiple tubes."""

from __future__ import annotations

import numpy as np


def synapse_stable_key(
    ctr_pt_nm: np.ndarray,
    *,
    grid_nm: float = 16.0,
) -> np.ndarray:
    """Stable integer key from synapse center position (nm), shape (N, 3).

    Use when ``synapse_id`` is unavailable or to cross-check identity.
    Quantize to ``grid_nm`` to absorb tiny floating differences.
    """
    pts = np.asarray(ctr_pt_nm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("ctr_pt_nm must be (N, 3)")
    q = np.round(pts / float(grid_nm)).astype(np.int64)
    # Mix into one int64 key (may collide rarely — prefer synapse_id when present)
    return (q[:, 0] * 1_000_003 + q[:, 1]) * 1_000_003 + q[:, 2]


def tube_overlap_weights(
    synapse_keys: np.ndarray,
    tube_membership: list[list[int]] | None = None,
) -> np.ndarray:
    """Return weights ``1 / degree`` per row where degree = number of tubes containing key.

    If ``tube_membership`` is omitted, assume each synapse appears once (weight 1).

    Parameters
    ----------
    synapse_keys
        One key per synapse row (e.g. synapse_id or output of ``synapse_stable_key``).
    tube_membership
        Optional list parallel to synapses: list of tube/root ids that claimed this
        synapse; length used as degree. If not provided, degree = 1.
    """
    n = len(synapse_keys)
    if tube_membership is None:
        return np.ones(n, dtype=np.float64)
    if len(tube_membership) != n:
        raise ValueError("tube_membership length must match synapse_keys")
    deg = np.array([max(1, len(m)) for m in tube_membership], dtype=np.float64)
    return 1.0 / deg
