"""Score-based metrics: ranking quality and edit decisions at a threshold.

Pair scorers (DNA cosine, merge-head probability, PCFG likelihood) produce a
score per candidate; these metrics judge the scores before any partition is
built. Two cautions the repo has already paid for are built in:

* AUC is rank-based and numpy-only (Mann-Whitney U with tie handling), so it
  does not depend on an optional scikit-learn install and agrees with it.
* **Do-nothing-relative metrics.** In this data ~99% of candidate decisions are
  "leave the segmentation as is", so accuracy and even AUC can look fine for a
  model that never proposes an edit. :func:`edit_metrics_vs_baseline` scores a
  model by the edits it proposes relative to the do-nothing baseline (which has
  edit recall 0 by construction) and reports the *net* change in errors.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import rankdata

from ._core import NAN, prf1, safe_div


def _as_binary(y) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype == bool:
        return y
    return y.astype(np.int64) == 1


def roc_auc(y_true, y_score) -> float:
    """Area under the ROC curve; NaN unless both classes are present.

    ``P(score_pos > score_neg) + 0.5 * P(tie)`` via average ranks.
    """
    y = _as_binary(y_true)
    s = np.asarray(y_score, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError("y_true and y_score must align")
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return NAN
    ranks = rankdata(s)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def average_precision(y_true, y_score) -> float:
    """Area under the precision-recall curve (step-wise, ties grouped).

    Equals ``sum_k (R_k - R_{k-1}) * P_k`` over distinct score thresholds,
    the definition scikit-learn uses. NaN without positives.
    """
    y = _as_binary(y_true)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return NAN
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order].astype(np.float64)
    # last index of each tie group (scores are descending)
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tp_cum = np.cumsum(y_sorted)[last]
    n_cum = (last + 1).astype(np.float64)
    precision = tp_cum / n_cum
    d_tp = np.diff(np.r_[0.0, tp_cum])
    return float(np.sum(precision * d_tp) / n_pos)


def threshold_metrics(y_true, y_score, thresholds: Iterable[float]) -> list[dict]:
    """P/R/F1 and counts of ``score >= t`` for each threshold."""
    y = _as_binary(y_true)
    s = np.asarray(y_score, dtype=np.float64)
    rows = []
    for t in thresholds:
        pred = s >= t
        tp = int(np.sum(pred & y))
        fp = int(np.sum(pred & ~y))
        fn = int(np.sum(~pred & y))
        p, r, f = prf1(tp, fp, fn)
        rows.append({"threshold": float(t), "precision": p, "recall": r, "f1": f,
                     "tp": tp, "fp": fp, "fn": fn, "n_pred_pos": tp + fp})
    return rows


def best_f1_threshold(y_true, y_score) -> dict:
    """The distinct score threshold maximising F1 (ties: highest threshold)."""
    s = np.asarray(y_score, dtype=np.float64)
    if len(s) == 0:
        return {"threshold": NAN, "precision": NAN, "recall": NAN, "f1": NAN,
                "tp": 0, "fp": 0, "fn": 0, "n_pred_pos": 0}
    rows = threshold_metrics(y_true, s, np.unique(s)[::-1])
    best = None
    for row in rows:
        if best is None or (not np.isnan(row["f1"]) and (np.isnan(best["f1"]) or row["f1"] > best["f1"])):
            best = row
    return best


def edit_metrics_vs_baseline(y_true, y_base, y_pred) -> dict:
    """Score proposed edits against a do-nothing baseline.

    All three inputs are ``[M]`` binary "same cluster" decisions on the same
    candidate pairs: ``y_true`` the truth, ``y_base`` what the input
    segmentation already says (do nothing), ``y_pred`` the model. A pair where
    ``y_true != y_base`` needs an edit; the model *proposes* an edit where
    ``y_pred != y_base``.

    Returns
    -------
    ``n_pairs, n_edits_needed, n_splits_needed, n_merges_needed,
    n_edits_proposed, edit_precision, edit_recall, split_recall, merge_recall,
    base_errors, model_errors, net_fixed, net_fixed_frac``.
    ``net_fixed > 0`` means the edited partition has fewer wrong pairs than the
    input; a model can have a fine AUC and still sit at ``net_fixed <= 0``.
    """
    t = _as_binary(y_true)
    b = _as_binary(y_base)
    p = _as_binary(y_pred)
    if not (t.shape == b.shape == p.shape):
        raise ValueError("y_true, y_base and y_pred must align")
    needs_edit = t != b
    proposed = p != b
    correct = proposed & (p == t)
    need_split = needs_edit & ~t      # input says same, truth says different
    need_merge = needs_edit & t       # input says different, truth says same
    base_err = int(needs_edit.sum())
    model_err = int((p != t).sum())
    return {
        "n_pairs": int(len(t)),
        "n_edits_needed": base_err,
        "n_splits_needed": int(need_split.sum()),
        "n_merges_needed": int(need_merge.sum()),
        "n_edits_proposed": int(proposed.sum()),
        "edit_precision": safe_div(int(correct.sum()), int(proposed.sum())),
        "edit_recall": safe_div(int(correct.sum()), base_err),
        "split_recall": safe_div(int((correct & need_split).sum()), int(need_split.sum())),
        "merge_recall": safe_div(int((correct & need_merge).sum()), int(need_merge.sum())),
        "base_errors": base_err,
        "model_errors": model_err,
        "net_fixed": base_err - model_err,
        "net_fixed_frac": safe_div(base_err - model_err, base_err),
    }


__all__ = [
    "average_precision",
    "best_f1_threshold",
    "edit_metrics_vs_baseline",
    "roc_auc",
    "threshold_metrics",
]
