"""Score-based metrics: neuronauts.metrics.ranking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuronauts.metrics import (
    average_precision,
    best_f1_threshold,
    edit_metrics_vs_baseline,
    roc_auc,
    threshold_metrics,
)


def test_roc_auc_perfect_separation_is_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert roc_auc(y, s) == pytest.approx(1.0)


def test_roc_auc_random_is_half():
    y = np.array([0, 1, 0, 1])
    s = np.array([1, 1, 2, 2], dtype=float)  # tied scores -> chance
    assert roc_auc(y, s) == pytest.approx(0.5)


def test_roc_auc_nan_without_both_classes():
    assert math.isnan(roc_auc(np.zeros(5), np.arange(5)))


def test_roc_auc_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, size=300)
    s = rng.normal(size=300) + y * 0.7
    assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))


def test_roc_auc_matches_repo_rank_based_fallback():
    """Cross-check against neuronauts.represent.enrich's local AUC fallback,
    the sklearn-free implementation the new roc_auc replaces."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=200)
    s = rng.normal(size=200)

    from scipy.stats import rankdata
    ranks = rankdata(s)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    u = np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0
    legacy = float(u / (n_pos * n_neg))
    assert roc_auc(y, s) == pytest.approx(legacy)


def test_average_precision_perfect_is_one():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.2, 0.1])
    assert average_precision(y, s) == pytest.approx(1.0)


def test_average_precision_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, size=250)
    s = rng.normal(size=250) + y * 0.5
    assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s))


def test_average_precision_nan_without_positives():
    assert math.isnan(average_precision(np.zeros(5), np.arange(5)))


def test_threshold_metrics_counts():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.9, 0.4, 0.6, 0.1])
    rows = threshold_metrics(y, s, [0.5])
    assert rows[0]["tp"] == 1
    assert rows[0]["fp"] == 1
    assert rows[0]["fn"] == 1


def test_best_f1_threshold_finds_perfect_separator():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    best = best_f1_threshold(y, s)
    assert best["f1"] == pytest.approx(1.0)


def test_edit_metrics_vs_baseline_do_nothing_has_zero_recall():
    n = 20
    truth = np.zeros(n, dtype=bool)
    truth[:5] = True
    base = np.zeros(n, dtype=bool)   # never agrees with truth's positives
    pred = base.copy()               # model proposes nothing either
    m = edit_metrics_vs_baseline(truth, base, pred)
    assert m["n_edits_proposed"] == 0
    assert m["net_fixed"] == 0


def test_edit_metrics_vs_baseline_net_fixed_positive_when_model_helps():
    truth = np.array([1, 1, 0, 0], dtype=bool)
    base = np.array([0, 0, 0, 0], dtype=bool)   # do-nothing: misses both positives
    pred = np.array([1, 1, 0, 0], dtype=bool)   # model fixes both
    m = edit_metrics_vs_baseline(truth, base, pred)
    assert m["base_errors"] == 2
    assert m["model_errors"] == 0
    assert m["net_fixed"] == 2
    assert m["edit_recall"] == pytest.approx(1.0)
    assert m["merge_recall"] == pytest.approx(1.0)


def test_edit_metrics_vs_baseline_split_and_merge_recall_are_separated():
    # pair0 needs a split (base=True truth=False), pair1 needs a merge (base=False truth=True)
    truth = np.array([0, 1], dtype=bool)
    base = np.array([1, 0], dtype=bool)
    pred = np.array([0, 0], dtype=bool)   # only fixes the split, not the merge
    m = edit_metrics_vs_baseline(truth, base, pred)
    assert m["split_recall"] == pytest.approx(1.0)
    assert m["merge_recall"] == pytest.approx(0.0)


def test_edit_metrics_vs_baseline_bad_model_can_be_net_negative():
    truth = np.array([0, 0], dtype=bool)
    base = np.array([0, 0], dtype=bool)
    pred = np.array([1, 1], dtype=bool)   # proposes wrong edits
    m = edit_metrics_vs_baseline(truth, base, pred)
    assert m["net_fixed"] < 0
