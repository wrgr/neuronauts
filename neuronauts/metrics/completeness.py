"""Fragment completeness: which input objects need no edit at all.

An input fragment (a v117 root, an atom) is *complete* when it maps 1-to-1
onto a single ground-truth neuron: it is that neuron's sole contributor and it
is not a frankenmerge straddling two neurons. Predicting completeness is the
"leave it alone" decision a proofreading pipeline makes most often, so it is
scored as its own binary task rather than folded into pair metrics.
"""

from __future__ import annotations

import numpy as np

from ._core import NAN, prf1


def fragment_completeness(root_label_map: dict) -> dict:
    """Ground-truth completeness from ``{fragment: set(gt_neurons)}``.

    ``True`` when the fragment maps to exactly one neuron and no other fragment
    maps to that neuron.
    """
    neuron_to_frags: dict = {}
    for frag, neurons in root_label_map.items():
        for nrn in neurons:
            neuron_to_frags.setdefault(nrn, []).append(frag)
    result: dict = {}
    for frag, neurons in root_label_map.items():
        if len(neurons) != 1:
            result[frag] = False
            continue
        (nrn,) = tuple(neurons)
        result[frag] = len(neuron_to_frags[nrn]) == 1
    return result


def pred_fragment_completeness(fragment_id, pred_labels, *, ignore_label=-1) -> dict:
    """Predicted completeness: all of a fragment's items in one cluster that
    contains no other fragment. Items with ``pred == ignore_label`` are
    unassigned and skipped.
    """
    frag_clusters: dict = {}
    cluster_frags: dict = {}
    for f, c in zip(np.asarray(fragment_id).tolist(), np.asarray(pred_labels).tolist()):
        if c == ignore_label:
            continue
        frag_clusters.setdefault(f, set()).add(c)
        cluster_frags.setdefault(c, set()).add(f)
    return {
        f: (len(cs) == 1 and len(cluster_frags[next(iter(cs))]) == 1)
        for f, cs in frag_clusters.items()
    }


def completeness_metrics(root_label_map: dict, pred_completeness: dict) -> dict:
    """Binary P/R/F1/accuracy of a completeness prediction against truth.

    Scored over the fragments present in both. Keys: ``precision, recall, f1,
    accuracy, n_complete_gt, n_fragments, tp_complete, fp_complete,
    fn_complete, tn_complete``.
    """
    gt = fragment_completeness(root_label_map)
    common = [f for f in gt if f in pred_completeness]
    if not common:
        return {"precision": NAN, "recall": NAN, "f1": NAN, "accuracy": NAN,
                "n_complete_gt": int(sum(gt.values())), "n_fragments": len(gt),
                "tp_complete": 0, "fp_complete": 0, "fn_complete": 0, "tn_complete": 0}
    y_true = np.array([gt[f] for f in common], dtype=bool)
    y_pred = np.array([pred_completeness[f] for f in common], dtype=bool)
    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    p, r, f = prf1(tp, fp, fn)
    return {
        "precision": p, "recall": r, "f1": f,
        "accuracy": float((y_true == y_pred).mean()),
        "n_complete_gt": int(y_true.sum()),
        "n_fragments": len(common),
        "tp_complete": tp, "fp_complete": fp, "fn_complete": fn, "tn_complete": tn,
    }


__all__ = ["completeness_metrics", "fragment_completeness", "pred_fragment_completeness"]
