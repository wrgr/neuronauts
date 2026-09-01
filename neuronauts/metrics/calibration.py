"""Probability calibration from ``(probs, labels)`` arrays.

Model- and framework-free: :mod:`treestitch.calibration` runs the GNN and
fits the temperature, then hands the calibrated probabilities here. A
predicted confidence of 0.8 should be right about 80% of the time; ECE is the
count-weighted gap from that diagonal and Brier the mean squared error.
"""

from __future__ import annotations

import numpy as np

from ._core import NAN


def reliability_bins(probs, y_true, *, n_bins: int = 10) -> dict:
    """Bin predictions by confidence and record accuracy per bin.

    Returns ``bin_centers, mean_conf, frac_pos, counts`` (each ``[n_bins]``;
    ``frac_pos`` is NaN for empty bins) plus ``n_bins``. The last bin is
    closed on the right so ``p == 1.0`` is counted.
    """
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    if p.shape != y.shape:
        raise ValueError("probs and y_true must align")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_conf = centers.copy()
    frac_pos = np.full(n_bins, NAN)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (p >= lo) & ((p <= hi) if b == n_bins - 1 else (p < hi))
        if mask.any():
            mean_conf[b] = p[mask].mean()
            frac_pos[b] = y[mask].mean()
            counts[b] = int(mask.sum())
    return {"bin_centers": centers, "mean_conf": mean_conf, "frac_pos": frac_pos,
            "counts": counts, "n_bins": n_bins}


def expected_calibration_error(bins: dict) -> float:
    """Count-weighted mean |accuracy - confidence| over non-empty bins."""
    counts = np.asarray(bins["counts"])
    total = counts.sum()
    if total == 0:
        return NAN
    frac_pos = np.asarray(bins["frac_pos"], dtype=np.float64)
    mean_conf = np.asarray(bins["mean_conf"], dtype=np.float64)
    valid = ~np.isnan(frac_pos)
    return float(np.sum(np.abs(frac_pos[valid] - mean_conf[valid]) * counts[valid]) / total)


def brier_score(probs, y_true) -> float:
    """Mean squared error between probabilities and binary outcomes."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    if len(p) == 0:
        return NAN
    return float(np.mean((p - y) ** 2))


def calibration_metrics(probs, y_true, *, n_bins: int = 10) -> dict:
    """``ece, brier, n_scored`` plus the reliability bins under ``bins``."""
    bins = reliability_bins(probs, y_true, n_bins=n_bins)
    return {"ece": expected_calibration_error(bins), "brier": brier_score(probs, y_true),
            "n_scored": int(len(np.asarray(probs))), "bins": bins}


__all__ = ["brier_score", "calibration_metrics", "expected_calibration_error", "reliability_bins"]
