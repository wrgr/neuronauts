"""Frankenmerge separation: neuronauts.metrics.frankenmerge."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuronauts.metrics import frankenmerge_metrics


def test_no_frankenmerges_returns_nan_rates():
    pred = np.array([0, 0, 1, 1])
    true = np.array([5, 5, 6, 6])
    parent = np.array(["A", "A", "B", "B"])
    m = frankenmerge_metrics(pred, true, parent)
    assert m["fk_n_parents"] == 0
    assert math.isnan(m["fk_pair_split_rate"])
    assert math.isnan(m["fk_separation"])


def test_single_frankenmerge_fully_separated():
    # parent A holds both true labels 5 and 6; prediction splits them apart
    pred = np.array([0, 1])
    true = np.array([5, 6])
    parent = np.array(["A", "A"])
    m = frankenmerge_metrics(pred, true, parent)
    assert m["fk_n_parents"] == 1
    assert m["fk_separation"] == pytest.approx(1.0)
    assert m["fk_pair_split_rate"] == pytest.approx(1.0)


def test_single_frankenmerge_not_separated_when_kept_together():
    pred = np.array([0, 0])
    true = np.array([5, 6])
    parent = np.array(["A", "A"])
    m = frankenmerge_metrics(pred, true, parent)
    assert m["fk_separation"] == pytest.approx(0.0)
    assert m["fk_pair_split_rate"] == pytest.approx(0.0)


def test_partial_separation_counts_leaked_pairs_but_fails_the_parent():
    # parent A: 3 items with truth 5,5,6. Prediction splits one 5 from the
    # other 5+6 (still leaves one cross pair merged) -> 1/2 cross pairs split,
    # but the parent itself is NOT fully separated (one leaked pair).
    pred = np.array([0, 1, 1])
    true = np.array([5, 5, 6])
    parent = np.array(["A", "A", "A"])
    m = frankenmerge_metrics(pred, true, parent)
    assert m["fk_n_cross_pairs"] == 2       # (item0,item2) and (item1,item2)
    assert m["fk_n_cross_pairs_split"] == 1  # only (item0,item2) is split
    assert m["fk_pair_split_rate"] == pytest.approx(0.5)
    assert m["fk_separation"] == pytest.approx(0.0)  # the parent still leaked one pair


def test_abstained_items_never_count_as_merged():
    pred = np.array([-1, -1])
    true = np.array([5, 6])
    parent = np.array(["A", "A"])
    m = frankenmerge_metrics(pred, true, parent, pred_ignore=-1)
    assert m["fk_separation"] == pytest.approx(1.0)


def test_ignore_label_drops_unknown_truth_items():
    pred = np.array([0, 0, 1])
    true = np.array([0, 5, 6])
    parent = np.array(["A", "A", "A"])
    m = frankenmerge_metrics(pred, true, parent, ignore=0)
    assert m["fk_n_parents"] == 1  # only items 1,2 remain, still a frankenmerge


def test_empty_input():
    m = frankenmerge_metrics(np.array([]), np.array([]), np.array([]))
    assert m["fk_n_parents"] == 0


def test_handles_real_root_id_magnitudes_without_overflow():
    pred = np.array([0, 1])
    true = np.array([864691135000000001, 864691135000000002], dtype=np.int64)
    parent = np.array([864691136000000005, 864691136000000005], dtype=np.int64)
    m = frankenmerge_metrics(pred, true, parent)
    assert m["fk_separation"] == pytest.approx(1.0)


def test_matches_legacy_treestitch_atomize_frankenmerge_separation():
    from treestitch.atomize import frankenmerge_separation as legacy

    rng = np.random.default_rng(9)
    n = 100
    parent = rng.integers(0, 15, size=n).astype(np.int64)
    true = rng.integers(0, 8, size=n).astype(np.int64)
    pred = rng.integers(-1, 6, size=n).astype(np.int64)  # -1 = abstain

    legacy_out = legacy(pred, true, parent, ignore_true=0)
    new_out = frankenmerge_metrics(pred, true, parent, ignore=0, pred_ignore=-1)
    assert new_out["fk_n_parents"] == legacy_out["n_frankenmerges"]
    assert new_out["fk_n_separated"] == legacy_out["n_separated"]
    a, b = new_out["fk_separation"], legacy_out["fk_separation"]
    if isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b)
    else:
        assert a == pytest.approx(b)
