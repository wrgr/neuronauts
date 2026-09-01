"""Frankenmerge metrics: did the model cut what the input had falsely joined.

A *parent* is an input object (v117 root, atom, segment) whose items carry
two or more ground-truth labels: a frankenmerge. Fixing it means separating
the labels into different predicted clusters. The repo has used three
denominators for this ("Bar 3"); the two that are well defined on items are
computed together here so they can never drift apart again:

* ``fk_pair_split_rate``: over all within-parent item pairs with *different*
  truth, the fraction the prediction puts in different clusters. This is the
  global-merge ``fk_split``.
* ``fk_separation``: the fraction of frankenmerge parents in which **no**
  predicted cluster holds two of the parent's truth labels. This is the
  treestitch ``fk_separation``; it is stricter (one leaked pair fails the
  whole parent).

The edge-level ``frankenmerge_split_recall`` in :mod:`neuronauts.metrics.edges`
is the third convention: the same question restricted to candidate edges.
"""

from __future__ import annotations

import numpy as np

from ._core import NAN, align_labels, joint_labels, safe_div


def _c2(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=np.int64)
    return c * (c - 1) // 2


def _group_c2_sum(keys: np.ndarray, group_of_key: np.ndarray, n_groups: int) -> np.ndarray:
    """``sum over distinct key of C(count,2)``, accumulated per group."""
    uk, first, cnt = np.unique(keys, return_index=True, return_counts=True)
    return np.bincount(group_of_key[first], weights=_c2(cnt), minlength=n_groups)


def frankenmerge_metrics(pred, true, parent, *, ignore=0, pred_ignore=None) -> dict:
    """Pair- and parent-level frankenmerge separation.

    Parameters
    ----------
    pred, true, parent:
        Aligned per-item predicted cluster, true label and input-object id.
    ignore:
        Unknown-truth value; such items take no part.
    pred_ignore:
        Abstained prediction value; an abstained item is its own singleton, so
        it never merges with anything (matches treestitch's ``pred < 0``).

    Returns
    -------
    ``fk_n_parents, fk_n_cross_pairs, fk_n_cross_pairs_split,
    fk_pair_split_rate, fk_n_separated, fk_separation, fk_parents`` where
    ``fk_parents`` is the list of frankenmerge parent ids (input dtype).
    """
    true = np.asarray(true)
    parent = np.asarray(parent)
    pred = np.asarray(pred)
    if not (pred.shape == true.shape == parent.shape):
        raise ValueError("pred, true and parent must align")
    keep = np.ones(len(true), dtype=bool) if ignore is None else (true != ignore)
    pred, true, _ = align_labels(pred[keep], true[keep], ignore=None, pred_ignore=pred_ignore)
    parent = parent[keep]

    empty = {"fk_n_parents": 0, "fk_n_cross_pairs": 0, "fk_n_cross_pairs_split": 0,
             "fk_pair_split_rate": NAN, "fk_n_separated": 0, "fk_separation": NAN,
             "fk_parents": []}
    if len(true) == 0:
        return empty

    upar, par = np.unique(parent, return_inverse=True)
    par = par.reshape(-1).astype(np.int64)
    n_par = len(upar)
    n_parent = np.bincount(par, minlength=n_par)

    jpt = joint_labels(par, true)               # (parent, true)
    jpp = joint_labels(par, pred)               # (parent, pred)
    jptp = joint_labels(jpt, pred)              # (parent, true, pred)

    same_truth_pairs = _group_c2_sum(jpt, par, n_par)
    cross_pairs = _c2(n_parent) - same_truth_pairs
    pred_pairs = _group_c2_sum(jpp, par, n_par)
    pure_pred_pairs = _group_c2_sum(jptp, par, n_par)
    merged_cross = pred_pairs - pure_pred_pairs   # cross-truth pairs the prediction kept together

    is_fk = cross_pairs > 0
    n_fk = int(is_fk.sum())
    if n_fk == 0:
        return empty
    total_cross = int(cross_pairs[is_fk].sum())
    merged = int(merged_cross[is_fk].sum())
    separated = int(np.sum(is_fk & (merged_cross == 0)))
    return {
        "fk_n_parents": n_fk,
        "fk_n_cross_pairs": total_cross,
        "fk_n_cross_pairs_split": total_cross - merged,
        "fk_pair_split_rate": safe_div(total_cross - merged, total_cross),
        "fk_n_separated": separated,
        "fk_separation": safe_div(separated, n_fk),
        "fk_parents": upar[is_fk].tolist(),
    }


__all__ = ["frankenmerge_metrics"]
