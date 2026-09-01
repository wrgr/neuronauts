"""Edge-level merge/split confusion over a candidate-edge set.

Partition metrics score every item pair; these score only the pairs a model
actually decided on: the candidate edges of an observation graph, a stitch
proposal list, or the endpoint-pair candidates in the harness. That is the
right denominator when the candidate generator is fixed and two scorers are
compared on identical candidates, which is exactly the harness protocol.

An edge is a *merge* when both endpoints land in the same predicted cluster.
Over-merge (FP) is the expensive, hard-to-undo error; under-merge (FN) can be
repaired by a later pass, so the two rates are reported separately.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._core import NAN, prf1, safe_div


def edge_merge_metrics(
    src,
    dst,
    pred,
    true,
    *,
    ignore=0,
    same_fragment: Optional[np.ndarray] = None,
    abstain: Optional[np.ndarray] = None,
    undefined: float = NAN,
) -> dict:
    """Merge/split confusion over candidate edges.

    Parameters
    ----------
    src, dst:
        ``[E]`` item indices of each candidate edge.
    pred, true:
        ``[N]`` predicted cluster and true label per item.
    ignore:
        True-label value for unknown ground truth; edges touching such an
        item are not evaluated.
    same_fragment:
        Optional ``[E]`` bool: edges whose endpoints lie on the same input
        object (v117 fragment / atom). Among those, an edge whose endpoints
        have different truth is a *frankenmerge* the model must cut;
        ``frankenmerge_split_recall`` is the fraction it did cut ("Bar 3").
    abstain:
        Optional ``[N]`` bool. An edge with an abstained endpoint counts as a
        predicted split (an abstention is not a merge decision), and
        ``abstain_rate`` is reported over all items.
    undefined:
        Value for zero-denominator ratios.

    Returns
    -------
    ``merge_precision, merge_recall, merge_f1, over_merge_rate,
    under_merge_rate, n_edges_eval, n_merges_pred, n_splits_pred,
    n_true_merges, tp_merges, fp_merges, fn_merges, tn_splits`` and, when the
    optional inputs are given, ``frankenmerge_rate, n_frankenmerge_edges,
    frankenmerge_split_recall, abstain_rate``.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    pred = np.asarray(pred)
    true = np.asarray(true)
    if src.shape != dst.shape:
        raise ValueError("src and dst must align")

    lab_s, lab_d = true[src], true[dst]
    valid = np.ones(len(src), dtype=bool) if ignore is None else \
        (lab_s != ignore) & (lab_d != ignore)

    same_true = lab_s == lab_d
    same_pred = pred[src] == pred[dst]
    if abstain is not None:
        ab = np.asarray(abstain, dtype=bool)
        same_pred = same_pred & ~ab[src] & ~ab[dst]

    st, sp = same_true[valid], same_pred[valid]
    n = int(valid.sum())
    tp = int(np.sum(st & sp))
    fp = int(np.sum(~st & sp))
    fn = int(np.sum(st & ~sp))
    tn = n - tp - fp - fn
    p, r, f = prf1(tp, fp, fn, undefined=undefined)

    out = {
        "merge_precision": p,
        "merge_recall": r,
        "merge_f1": f,
        "over_merge_rate": safe_div(fp, n, undefined=undefined),
        "under_merge_rate": safe_div(fn, n, undefined=undefined),
        "n_edges_eval": n,
        "n_merges_pred": tp + fp,
        "n_splits_pred": fn + tn,
        "n_true_merges": tp + fn,
        "tp_merges": tp,
        "fp_merges": fp,
        "fn_merges": fn,
        "tn_splits": tn,
    }

    if same_fragment is not None:
        sf = np.asarray(same_fragment, dtype=bool) & valid
        franken = sf & ~same_true
        n_sf = int(sf.sum())
        n_fk = int(franken.sum())
        out["frankenmerge_rate"] = safe_div(n_fk, n_sf, undefined=undefined)
        out["n_frankenmerge_edges"] = n_fk
        out["frankenmerge_split_recall"] = safe_div(
            int(np.sum(franken & ~same_pred)), n_fk, undefined=undefined)

    if abstain is not None:
        out["abstain_rate"] = safe_div(int(ab.sum()), len(pred), undefined=undefined)
    return out


__all__ = ["edge_merge_metrics"]
