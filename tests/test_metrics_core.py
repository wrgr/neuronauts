"""Core metric primitives: undefined ratios, label alignment, contingency.

These are the pieces every other metric is built on, so their edge cases are
tested directly rather than through a caller.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuronauts.metrics import (
    align_labels,
    contingency,
    joint_labels,
    labels_from_maps,
    pair_confusion,
    prf1,
    safe_div,
    weighted_pair_confusion,
)
from neuronauts.metrics._core import pairs_in, weighted_pairs_in


# ---------------------------------------------------------------------------
# undefined ratios
# ---------------------------------------------------------------------------

def test_safe_div_zero_denominator_is_nan_by_default():
    assert math.isnan(safe_div(0, 0))
    assert math.isnan(safe_div(5, 0))


def test_safe_div_honours_explicit_undefined():
    assert safe_div(0, 0, undefined=1.0) == 1.0
    assert safe_div(0, 0, undefined=0.0) == 0.0


def test_prf1_no_predictions_leaves_precision_undefined():
    p, r, f = prf1(0, 0, 3)
    assert math.isnan(p)
    assert r == 0.0
    assert math.isnan(f)


def test_prf1_no_truth_leaves_recall_undefined():
    p, r, f = prf1(0, 4, 0)
    assert p == 0.0
    assert math.isnan(r)
    assert math.isnan(f)


def test_prf1_both_zero_gives_zero_f1_not_nan():
    p, r, f = prf1(0, 1, 1)
    assert (p, r, f) == (0.0, 0.0, 0.0)


def test_prf1_perfect():
    assert prf1(10, 0, 0) == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# align_labels
# ---------------------------------------------------------------------------

def test_align_drops_items_with_unknown_truth():
    pred, true, _ = align_labels([1, 1, 2], [0, 5, 5], ignore=0)
    assert list(true) == [5, 5]
    assert list(pred) == [1, 2]


def test_align_ignore_none_keeps_everything():
    pred, true, _ = align_labels([1, 1], [0, 0], ignore=None)
    assert len(true) == 2


def test_align_pred_ignore_makes_abstained_items_singletons():
    # both items abstained: they must NOT be counted as merged with each other
    pred, true, _ = align_labels([-1, -1], [5, 5], ignore=0, pred_ignore=-1)
    assert pred[0] != pred[1]


def test_align_pred_ignore_leaves_other_clusters_intact():
    pred, true, _ = align_labels([7, 7, -1], [5, 5, 5], ignore=0, pred_ignore=-1)
    assert pred[0] == pred[1]
    assert pred[2] not in (pred[0],)


def test_align_filters_weights_alongside_labels():
    pred, true, w = align_labels([1, 2, 3], [0, 5, 5], ignore=0, weights=[9.0, 1.0, 2.0])
    assert list(w) == [1.0, 2.0]


def test_align_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        align_labels([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        align_labels([1, 2], [1, 2], weights=[1.0])


def test_labels_from_maps_uses_only_common_keys_in_sorted_order():
    keys, pred, true = labels_from_maps({"b": "x", "a": "y", "c": "z"},
                                        {"a": "1", "b": "2"})
    assert keys == ["a", "b"]
    assert list(pred) == ["y", "x"]
    assert list(true) == ["1", "2"]


def test_joint_labels_distinguishes_pairs():
    j = joint_labels([1, 1, 1], [7, 7, 8])
    assert j[0] == j[1]
    assert j[1] != j[2]


def test_joint_labels_survives_real_root_id_magnitudes():
    # real CAVE root ids are ~8.6e17; a naive a*max+b product would overflow
    big = np.array([864691135000000001, 864691135000000001], dtype=np.int64)
    other = np.array([864691135000000002, 864691135000000003], dtype=np.int64)
    j = joint_labels(big, other)
    assert j[0] != j[1]
    assert np.all(j >= 0)


# ---------------------------------------------------------------------------
# contingency and pair counting
# ---------------------------------------------------------------------------

def test_contingency_cells_match_a_dense_crosstab():
    rng = np.random.default_rng(0)
    true = rng.integers(0, 5, size=200)
    pred = rng.integers(0, 4, size=200)
    ct = contingency(true, pred)
    dense = np.zeros((5, 4), dtype=np.int64)
    np.add.at(dense, (true, pred), 1)
    got = np.zeros((5, 4), dtype=np.int64)
    got[ct.row, ct.col] = ct.count
    assert np.array_equal(dense, got)
    assert ct.count.sum() == 200


def test_contingency_empty_input():
    ct = contingency([], [])
    assert ct.n == 0
    assert pair_confusion(ct) == (0, 0, 0, 0)


def test_pairs_in_matches_brute_force():
    labels = np.array([0, 0, 0, 1, 1, 2])
    _, counts = np.unique(labels, return_counts=True)
    brute = sum(1 for i in range(6) for j in range(i + 1, 6) if labels[i] == labels[j])
    assert pairs_in(counts) == brute


def test_pair_confusion_matches_brute_force():
    rng = np.random.default_rng(3)
    true = rng.integers(0, 4, size=60)
    pred = rng.integers(0, 3, size=60)
    tp = fp = fn = tn = 0
    for i in range(60):
        for j in range(i + 1, 60):
            st, sp = true[i] == true[j], pred[i] == pred[j]
            tp += st and sp
            fp += (not st) and sp
            fn += st and not sp
            tn += (not st) and (not sp)
    assert pair_confusion(contingency(true, pred)) == (tp, fp, fn, tn)


def test_weighted_pairs_in_matches_brute_force():
    w = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([0, 0, 1, 1])
    ct = contingency(labels, labels, w)
    brute = sum(w[i] * w[j] for i in range(4) for j in range(i + 1, 4)
                if labels[i] == labels[j])
    assert weighted_pairs_in(ct.row_wsum, ct.row_w2sum) == pytest.approx(brute)


def test_weighted_pair_confusion_matches_brute_force():
    rng = np.random.default_rng(7)
    n = 40
    true = rng.integers(0, 3, size=n)
    pred = rng.integers(0, 3, size=n)
    w = rng.uniform(0.5, 5.0, size=n)
    tp = fp = fn = tn = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            wij = w[i] * w[j]
            st, sp = true[i] == true[j], pred[i] == pred[j]
            if st and sp:
                tp += wij
            elif sp:
                fp += wij
            elif st:
                fn += wij
            else:
                tn += wij
    got = weighted_pair_confusion(contingency(true, pred, w))
    assert got[0] == pytest.approx(tp)
    assert got[1] == pytest.approx(fp)
    assert got[2] == pytest.approx(fn)
    assert got[3] == pytest.approx(tn)


def test_weighted_pair_confusion_requires_weights():
    with pytest.raises(ValueError):
        weighted_pair_confusion(contingency([1, 1], [1, 1]))


def test_contingency_rejects_negative_weights():
    with pytest.raises(ValueError):
        contingency([1, 1], [1, 1], [-1.0, 1.0])
