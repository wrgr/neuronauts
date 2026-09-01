"""Partition-vs-partition metrics over N labelled items.

An *item* is whatever the experiment partitions: a synapse observation, a
v117 fragment, an L2 node, an atom. Every metric here compares a predicted
cluster label with a ground-truth label per item, through one sparse
contingency table, so a single call gives:

* **Pair confusion.** Over all unordered item pairs: TP (same cluster in both),
  FP (false merge), FN (false split), TN. ``pair_precision`` is the
  treestitch/global-merge *merge_P* ("Bar 1"), ``pair_recall`` is *merge_R*.
* **Adjusted Rand index** (chance-corrected pair agreement).
* **Homogeneity / completeness / V-measure** (entropy based).
* **Variation of information** split into ``vi_split`` = H(pred | true), the
  over-segmentation term, and ``vi_merge`` = H(true | pred), the false-merge
  term. Both are 0 for a perfect partition and additive in bits of confusion,
  which makes them the most interpretable pair for proofreading effort.
* **Weighted variants.** With per-item weights (cable length in µm is the
  harness default) each pair contributes ``w_i * w_j``; ``wpair_precision`` /
  ``wpair_recall`` are the cable-weighted merge metrics and ``erl`` is the
  expected run length: the weight of the predicted piece that a random unit
  of ground-truth weight lands in.

Undefined ratios are NaN (see :mod:`neuronauts.metrics._core`).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ._core import (
    NAN,
    Contingency,
    align_labels,
    contingency,
    pair_confusion,
    prf1,
    weighted_pair_confusion,
)


# ---------------------------------------------------------------------------
# scalar metrics from confusion counts / contingency
# ---------------------------------------------------------------------------

def adjusted_rand_from_confusion(tp: int, fp: int, fn: int, tn: int) -> float:
    """ARI from pair confusion counts (Hubert & Arabie; sklearn's form).

    The two trivial cases where truth and prediction agree on every pair
    (no clusters split, none merged) return 1.0 exactly, which is also what
    the limit of the formula gives.
    """
    if fp == 0 and fn == 0:
        return 1.0
    num = 2 * (tp * tn - fn * fp)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return float(num) / float(den)


def _entropy_terms(ct: Contingency) -> tuple[float, float, float, float]:
    """``(H(T), H(P), H(T|P), H(P|T))`` in nats from a contingency table."""
    n = float(ct.n)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    p_row = ct.row_count / n
    p_col = ct.col_count / n
    h_t = float(-np.sum(p_row[p_row > 0] * np.log(p_row[p_row > 0])))
    h_p = float(-np.sum(p_col[p_col > 0] * np.log(p_col[p_col > 0])))
    p_cell = ct.count / n
    # H(T|P) = -sum n_ij/n log(n_ij / n_j) ; H(P|T) = -sum n_ij/n log(n_ij / n_i)
    h_t_given_p = float(-np.sum(p_cell * np.log(ct.count / ct.col_count[ct.col])))
    h_p_given_t = float(-np.sum(p_cell * np.log(ct.count / ct.row_count[ct.row])))
    return h_t, h_p, max(h_t_given_p, 0.0), max(h_p_given_t, 0.0)


def homogeneity_completeness_v(ct: Contingency) -> tuple[float, float, float]:
    """Rosenberg & Hirschberg homogeneity, completeness and V-measure."""
    h_t, h_p, h_t_given_p, h_p_given_t = _entropy_terms(ct)
    homogeneity = 1.0 - h_t_given_p / h_t if h_t > 1e-12 else 1.0
    completeness = 1.0 - h_p_given_t / h_p if h_p > 1e-12 else 1.0
    denom = homogeneity + completeness
    v = 2.0 * homogeneity * completeness / denom if denom > 1e-12 else 0.0
    return homogeneity, completeness, v


def variation_of_information(ct: Contingency) -> tuple[float, float, float]:
    """``(vi, vi_split, vi_merge)`` in bits.

    ``vi_split = H(pred | true)`` grows when a true cluster is scattered over
    several predicted clusters (over-segmentation). ``vi_merge = H(true | pred)``
    grows when a predicted cluster spans several true clusters (false merge).
    """
    _, _, h_t_given_p, h_p_given_t = _entropy_terms(ct)
    split = h_p_given_t / math.log(2)
    merge = h_t_given_p / math.log(2)
    return split + merge, split, merge


def cluster_purity(ct: Contingency) -> tuple[float, float, float]:
    """``(purity_mass, purity_mean, frac_pure_clusters)``.

    For each predicted cluster, its majority true label. ``purity_mass`` is the
    share of all items sitting under their cluster's majority label (the
    scaffold-census "mass purity"); ``purity_mean`` averages the per-cluster
    fraction unweighted, so a thousand tiny pure clusters do not hide one
    large impure one; ``frac_pure_clusters`` counts clusters that are entirely
    one label.
    """
    if ct.n == 0:
        return NAN, NAN, NAN
    order = np.lexsort((-ct.count, ct.col))
    col_sorted = ct.col[order]
    first = np.r_[True, col_sorted[1:] != col_sorted[:-1]]
    majority = ct.count[order][first]
    sizes = ct.col_count[col_sorted[first]]
    n_pure = int(np.sum(majority == sizes))
    return (float(majority.sum() / ct.n),
            float(np.mean(majority / sizes)),
            float(n_pure / len(sizes)))


def expected_run_length(ct: Contingency) -> float:
    """Piece-weighted expected run length.

    For each (true, pred) cell the total weight ``L`` of items in it is a
    *piece* of a true cluster that was reconstructed contiguously. ERL is the
    expected piece size seen from a random unit of true weight,
    ``sum(L^2) / sum(L)``. In the same units as the weights.
    """
    if not ct.weighted or ct.w_total <= 0:
        return NAN
    return float(np.sum(ct.wsum ** 2) / ct.w_total)


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------

def adjusted_rand_index(true, pred, *, ignore=None) -> float:
    """ARI between two aligned label arrays (any dtype). 1.0 for ``n < 2``."""
    pred, true, _ = align_labels(pred, true, ignore=ignore)
    if len(true) < 2:
        return 1.0
    return adjusted_rand_from_confusion(*pair_confusion(contingency(true, pred)))


def rand_disagreement(true, pred, *, ignore=None) -> int:
    """Number of item pairs grouped together in exactly one of the labelings.

    ``FP + FN`` of the pair confusion: the count a proofreader has to fix.
    """
    pred, true, _ = align_labels(pred, true, ignore=ignore)
    _, fp, fn, _ = pair_confusion(contingency(true, pred))
    return int(fp + fn)


_EMPTY_KEYS = (
    "ari", "homogeneity", "completeness", "v_measure",
    "vi", "vi_split", "vi_merge",
    "pair_precision", "pair_recall", "pair_f1",
    "purity_mass", "purity_mean", "frac_pure_clusters",
)


def partition_metrics(
    pred,
    true,
    *,
    ignore=0,
    pred_ignore=None,
    weights: Optional[np.ndarray] = None,
    undefined: float = NAN,
) -> dict:
    """All partition-vs-partition metrics from one contingency table.

    Parameters
    ----------
    pred, true:
        Aligned per-item labels.
    ignore:
        True-label value meaning "unknown"; those items are dropped. Default 0
        (the repo-wide convention for unlabelled root ids). ``None`` keeps all.
    pred_ignore:
        Predicted-label value meaning "abstained"; those items become
        singletons. ``None`` (default) treats it as a normal cluster, which is
        the historical behaviour and the right one when no abstention exists.
    weights:
        Optional non-negative per-item weights for the ``wpair_*`` / ``erl``
        block (cable length in µm in the harness).
    undefined:
        Value for ratios with a zero denominator. NaN by default.

    Returns
    -------
    Flat dict. Always: ``n_items, n_clusters_pred, n_clusters_true, n_pairs,
    pair_tp, pair_fp, pair_fn, pair_tn, pair_precision, pair_recall, pair_f1,
    rand_disagreement, ari, homogeneity, completeness, v_measure, vi, vi_split,
    vi_merge, purity_mass, purity_mean, frac_pure_clusters``. With weights:
    ``wpair_precision, wpair_recall, wpair_f1, wpair_tp, wpair_fp, wpair_fn,
    erl, weight_total``.
    """
    pred, true, w = align_labels(
        pred, true, ignore=ignore, pred_ignore=pred_ignore, weights=weights)
    n = len(true)
    ct = contingency(true, pred, w)

    out: dict = {
        "n_items": n,
        "n_clusters_pred": ct.n_pred,
        "n_clusters_true": ct.n_true,
        "n_pairs": n * (n - 1) // 2,
    }
    if n == 0:
        p0, r0, f0 = prf1(0, 0, 0, undefined=undefined)
        out.update({k: NAN for k in _EMPTY_KEYS})
        out.update({"pair_tp": 0, "pair_fp": 0, "pair_fn": 0, "pair_tn": 0,
                    "rand_disagreement": 0,
                    "pair_precision": p0, "pair_recall": r0, "pair_f1": f0})
        if w is not None:
            wp0, wr0, wf0 = prf1(0.0, 0.0, 0.0, undefined=undefined)
            out.update({"wpair_precision": wp0, "wpair_recall": wr0, "wpair_f1": wf0,
                        "wpair_tp": 0.0, "wpair_fp": 0.0, "wpair_fn": 0.0,
                        "erl": NAN, "weight_total": 0.0})
        return out

    tp, fp, fn, tn = pair_confusion(ct)
    p, r, f = prf1(tp, fp, fn, undefined=undefined)
    h, c, v = homogeneity_completeness_v(ct)
    vi, vi_split, vi_merge = variation_of_information(ct)
    pur_mass, pur_mean, frac_pure = cluster_purity(ct)
    out.update({
        "pair_tp": tp, "pair_fp": fp, "pair_fn": fn, "pair_tn": tn,
        "pair_precision": p, "pair_recall": r, "pair_f1": f,
        "rand_disagreement": fp + fn,
        "ari": adjusted_rand_from_confusion(tp, fp, fn, tn),
        "homogeneity": h, "completeness": c, "v_measure": v,
        "vi": vi, "vi_split": vi_split, "vi_merge": vi_merge,
        "purity_mass": pur_mass, "purity_mean": pur_mean,
        "frac_pure_clusters": frac_pure,
    })
    if w is not None:
        wtp, wfp, wfn, _ = weighted_pair_confusion(ct)
        wp, wr, wf = prf1(wtp, wfp, wfn, undefined=undefined)
        out.update({
            "wpair_precision": wp, "wpair_recall": wr, "wpair_f1": wf,
            "wpair_tp": wtp, "wpair_fp": wfp, "wpair_fn": wfn,
            "erl": expected_run_length(ct),
            "weight_total": ct.w_total,
        })
    return out


__all__ = [
    "adjusted_rand_from_confusion",
    "adjusted_rand_index",
    "cluster_purity",
    "expected_run_length",
    "homogeneity_completeness_v",
    "partition_metrics",
    "rand_disagreement",
    "variation_of_information",
]
