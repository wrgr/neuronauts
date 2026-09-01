"""Shared primitives every metric in :mod:`neuronauts.metrics` is built on.

Three things are centralised here so that no metric module re-derives them:

* **Undefined ratios.** ``0/0`` is *undefined*, not 1.0 and not 0.0. Every
  ratio goes through :func:`safe_div` / :func:`prf1`, which return NaN by
  default. Callers that must keep a historical convention (the treestitch
  edge metrics returned 1.0 for "no merges proposed") pass ``undefined=``
  explicitly, so the choice is visible at the call site instead of buried in
  each implementation.

* **Label alignment.** Every partition-style metric takes two integer (or
  string) label arrays over the same N items. :func:`align_labels` applies the
  single ``ignore`` convention (drop items whose *true* label is unknown) and
  the ``pred_ignore`` convention (an *abstained* prediction is a singleton,
  never a cluster of its own), so that the two are not re-implemented with
  subtly different semantics per module.

* **Sparse contingency.** All pair-counting metrics (ARI, pairwise P/R/F1,
  homogeneity, VI, cable-weighted variants, ERL) come from one
  :class:`Contingency` built in O(N log N) without materialising a dense
  ``n_true x n_pred`` matrix, so they stay usable at harness scale
  (tens of thousands of clusters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import numpy as np

NAN = float("nan")


# ---------------------------------------------------------------------------
# nan-safe ratios
# ---------------------------------------------------------------------------

def safe_div(num: float, den: float, *, undefined: float = NAN) -> float:
    """``num / den`` or ``undefined`` when ``den`` is zero."""
    return float(num) / float(den) if den else undefined


def prf1(tp: float, fp: float, fn: float, *, undefined: float = NAN
         ) -> Tuple[float, float, float]:
    """Precision, recall and F1 from confusion counts (or weights).

    Precision is undefined without positive predictions, recall without
    positive truth. F1 is NaN if either is NaN, and 0.0 when both are 0.0.
    """
    p = safe_div(tp, tp + fp, undefined=undefined)
    r = safe_div(tp, tp + fn, undefined=undefined)
    if math.isnan(p) or math.isnan(r):
        return p, r, NAN
    f = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ---------------------------------------------------------------------------
# label alignment
# ---------------------------------------------------------------------------

def align_labels(
    pred,
    true,
    *,
    ignore=0,
    pred_ignore=None,
    weights=None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Apply the shared ignore conventions and return aligned arrays.

    Parameters
    ----------
    pred, true:
        Label arrays over the same N items. Any dtype ``np.unique`` accepts
        (int64, uint64 root ids, strings).
    ignore:
        Items whose **true** label equals this are dropped: their ground truth
        is unknown, so no pair involving them can be scored. ``None`` keeps
        everything.
    pred_ignore:
        Items whose **predicted** label equals this are *abstentions*. They are
        kept (an abstained item with known truth is a recall miss) but relabelled
        as singletons so that two abstained items never count as merged.
        ``None`` treats the value as an ordinary cluster label.
    weights:
        Optional per-item non-negative weights (cable length, synapse count);
        filtered alongside the labels.
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    if pred.shape != true.shape or pred.ndim != 1:
        raise ValueError(
            f"pred and true must be 1-D and aligned; got {pred.shape} vs {true.shape}")
    w = None if weights is None else np.asarray(weights, dtype=np.float64)
    if w is not None and w.shape != true.shape:
        raise ValueError(f"weights must align with labels; got {w.shape} vs {true.shape}")

    if ignore is not None:
        keep = true != ignore
        pred, true = pred[keep], true[keep]
        if w is not None:
            w = w[keep]

    if pred_ignore is not None and len(pred):
        abstained = pred == pred_ignore
        if abstained.any():
            # Work in inverse-index space so this is dtype agnostic.
            _, inv = np.unique(pred, return_inverse=True)
            inv = inv.reshape(-1).astype(np.int64)
            fresh = inv.max() + 1 + np.arange(int(abstained.sum()), dtype=np.int64)
            inv[abstained] = fresh
            pred = inv
    return pred, true, w


def labels_from_maps(
    pred_map: Mapping, gt_map: Mapping
) -> Tuple[list, np.ndarray, np.ndarray]:
    """Adapter for ``{item: cluster}`` dicts: align on the common keys.

    Returns ``(keys, pred, true)`` with keys in sorted order so results are
    deterministic. Items missing from either map are not scored.
    """
    keys = sorted(set(pred_map) & set(gt_map))
    pred = np.asarray([pred_map[k] for k in keys])
    true = np.asarray([gt_map[k] for k in keys])
    return keys, pred, true


def joint_labels(a, b) -> np.ndarray:
    """One integer label per distinct ``(a[i], b[i])`` pair."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"joint_labels needs aligned arrays; got {a.shape} vs {b.shape}")
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    ai = ai.reshape(-1).astype(np.int64)
    bi = bi.reshape(-1).astype(np.int64)
    nb = int(bi.max()) + 1 if len(bi) else 1
    return ai * nb + bi


# ---------------------------------------------------------------------------
# sparse contingency table
# ---------------------------------------------------------------------------

@dataclass
class Contingency:
    """Sparse ``true x pred`` co-occurrence table with optional item weights.

    ``row[k], col[k], count[k]`` describe non-empty cell ``k``. Weighted sums
    (``wsum`` = sum of item weights in the cell, ``w2sum`` = sum of squared
    weights) are ``None`` without weights. All marginals are dense vectors.
    """
    n: int
    n_true: int
    n_pred: int
    row: np.ndarray
    col: np.ndarray
    count: np.ndarray
    row_count: np.ndarray
    col_count: np.ndarray
    wsum: Optional[np.ndarray] = None
    w2sum: Optional[np.ndarray] = None
    row_wsum: Optional[np.ndarray] = None
    row_w2sum: Optional[np.ndarray] = None
    col_wsum: Optional[np.ndarray] = None
    col_w2sum: Optional[np.ndarray] = None
    w_total: float = 0.0
    w2_total: float = 0.0

    @property
    def weighted(self) -> bool:
        return self.wsum is not None


def contingency(true, pred, weights=None) -> Contingency:
    """Build the sparse contingency table for two aligned label arrays."""
    true = np.asarray(true)
    pred = np.asarray(pred)
    n = len(true)
    if n == 0:
        z = np.zeros(0, dtype=np.int64)
        return Contingency(0, 0, 0, z, z, z, z, z)

    _, t_inv = np.unique(true, return_inverse=True)
    _, p_inv = np.unique(pred, return_inverse=True)
    t_inv = t_inv.reshape(-1).astype(np.int64)
    p_inv = p_inv.reshape(-1).astype(np.int64)
    n_true = int(t_inv.max()) + 1
    n_pred = int(p_inv.max()) + 1

    key = t_inv * n_pred + p_inv
    ukey, k_inv, count = np.unique(key, return_inverse=True, return_counts=True)
    k_inv = k_inv.reshape(-1)
    row = ukey // n_pred
    col = ukey % n_pred
    row_count = np.bincount(t_inv, minlength=n_true)
    col_count = np.bincount(p_inv, minlength=n_pred)

    ct = Contingency(
        n=n, n_true=n_true, n_pred=n_pred,
        row=row, col=col, count=count.astype(np.int64),
        row_count=row_count.astype(np.int64), col_count=col_count.astype(np.int64),
    )
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != true.shape:
            raise ValueError("weights must align with labels")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
        w2 = w * w
        ct.wsum = np.bincount(k_inv, weights=w, minlength=len(ukey))
        ct.w2sum = np.bincount(k_inv, weights=w2, minlength=len(ukey))
        ct.row_wsum = np.bincount(t_inv, weights=w, minlength=n_true)
        ct.row_w2sum = np.bincount(t_inv, weights=w2, minlength=n_true)
        ct.col_wsum = np.bincount(p_inv, weights=w, minlength=n_pred)
        ct.col_w2sum = np.bincount(p_inv, weights=w2, minlength=n_pred)
        ct.w_total = float(w.sum())
        ct.w2_total = float(w2.sum())
    return ct


def pairs_in(counts: np.ndarray) -> int:
    """Number of unordered item pairs inside groups of the given sizes."""
    c = np.asarray(counts, dtype=np.int64)
    return int(np.sum(c * (c - 1))) // 2


def weighted_pairs_in(wsum: np.ndarray, w2sum: np.ndarray) -> float:
    """Sum of ``w_i * w_j`` over unordered pairs inside each group, totalled.

    Uses ``((sum w)^2 - sum w^2) / 2`` per group, which is exact and O(groups)
    instead of O(pairs).
    """
    v = 0.5 * (np.asarray(wsum, np.float64) ** 2 - np.asarray(w2sum, np.float64))
    return float(np.maximum(v, 0.0).sum())


def pair_confusion(ct: Contingency) -> Tuple[int, int, int, int]:
    """``(tp, fp, fn, tn)`` over unordered item pairs.

    A pair is *positive* when both items share a cluster. TP = same in both,
    FP = same in pred only (false merge), FN = same in truth only (false
    split), TN = different in both.
    """
    tp = pairs_in(ct.count)
    tp_fp = pairs_in(ct.col_count)
    tp_fn = pairs_in(ct.row_count)
    total = ct.n * (ct.n - 1) // 2
    fp = tp_fp - tp
    fn = tp_fn - tp
    tn = total - tp - fp - fn
    return tp, fp, fn, tn


def weighted_pair_confusion(ct: Contingency) -> Tuple[float, float, float, float]:
    """Weighted ``(tp, fp, fn, tn)`` where each pair counts ``w_i * w_j``."""
    if not ct.weighted:
        raise ValueError("contingency was built without weights")
    tp = weighted_pairs_in(ct.wsum, ct.w2sum)
    tp_fp = weighted_pairs_in(ct.col_wsum, ct.col_w2sum)
    tp_fn = weighted_pairs_in(ct.row_wsum, ct.row_w2sum)
    total = max(0.5 * (ct.w_total ** 2 - ct.w2_total), 0.0)
    fp = max(tp_fp - tp, 0.0)
    fn = max(tp_fn - tp, 0.0)
    tn = max(total - tp - fp - fn, 0.0)
    return tp, fp, fn, tn


__all__ = [
    "NAN",
    "Contingency",
    "align_labels",
    "contingency",
    "is_nan",
    "joint_labels",
    "labels_from_maps",
    "pair_confusion",
    "pairs_in",
    "prf1",
    "safe_div",
    "weighted_pair_confusion",
    "weighted_pairs_in",
]
