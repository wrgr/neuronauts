"""Fragment completeness: neuronauts.metrics.completeness."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuronauts.metrics import (
    completeness_metrics,
    fragment_completeness,
    pred_fragment_completeness,
)


def test_fragment_completeness_sole_contributor_is_complete():
    # frag A maps only to neuron 1, and is the ONLY frag mapping to neuron 1
    gt = {"A": {1}, "B": {2}}
    out = fragment_completeness(gt)
    assert out == {"A": True, "B": True}


def test_fragment_completeness_needs_merge_when_neuron_has_other_frags():
    gt = {"A": {1}, "B": {1}}
    out = fragment_completeness(gt)
    assert out == {"A": False, "B": False}


def test_fragment_completeness_frankenmerge_is_never_complete():
    gt = {"A": {1, 2}}
    assert fragment_completeness(gt) == {"A": False}


def test_pred_fragment_completeness_requires_singleton_cluster():
    fragment_id = np.array(["A", "A", "B"])
    pred = np.array([0, 0, 1])
    out = pred_fragment_completeness(fragment_id, pred)
    assert out == {"A": True, "B": True}


def test_pred_fragment_completeness_false_when_cluster_spans_fragments():
    fragment_id = np.array(["A", "B"])
    pred = np.array([0, 0])
    out = pred_fragment_completeness(fragment_id, pred)
    assert out == {"A": False, "B": False}


def test_pred_fragment_completeness_ignores_unassigned_items():
    fragment_id = np.array(["A", "A"])
    pred = np.array([0, -1])
    out = pred_fragment_completeness(fragment_id, pred, ignore_label=-1)
    assert out == {"A": True}


def test_completeness_metrics_perfect_prediction():
    gt = {"A": {1}, "B": {2}, "C": {1}}
    pred = fragment_completeness(gt)
    m = completeness_metrics(gt, pred)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)


def test_completeness_metrics_no_common_fragments_returns_nan():
    m = completeness_metrics({"A": {1}}, {"Z": True})
    assert math.isnan(m["precision"])
    assert m["n_fragments"] == 1


def test_completeness_metrics_matches_treestitch_partition_module():
    """Cross-check against treestitch.partition (the pre-consolidation home)."""
    from treestitch import partition as legacy

    gt = {i: ({i // 2} if i % 3 != 0 else {i // 2, i // 2 + 100}) for i in range(30)}
    rng = np.random.default_rng(0)
    fragment_id = np.array(list(gt.keys()))
    pred_labels = rng.integers(0, 10, size=len(fragment_id))
    pred_cmpl_new = pred_fragment_completeness(fragment_id, pred_labels, ignore_label=-1)
    pred_cmpl_legacy = legacy.pred_fragment_completeness(fragment_id, pred_labels, ignore_label=-1)
    assert pred_cmpl_new == pred_cmpl_legacy

    m_new = completeness_metrics(gt, pred_cmpl_new)
    m_legacy = legacy.completeness_metrics(gt, pred_cmpl_legacy)
    for k in ("precision", "recall", "f1", "accuracy", "n_complete_gt", "n_fragments"):
        a, b = m_new[k], m_legacy[k]
        if isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), k
        else:
            assert a == pytest.approx(b), k
