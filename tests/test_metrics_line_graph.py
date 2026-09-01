"""Line-graph F1: neuronauts.metrics.line_graph.

Ports the coverage from the pre-consolidation tests/test_line_graph.py and
adds a cross-check that evaluate_suite's counting form agrees exactly with
the pair-set form for randomised inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.metrics.line_graph import (
    build_true_line_graph,
    build_true_pairs_and,
    build_true_pairs_post,
    build_true_pairs_pre,
    compute_line_graph_f1,
    compute_sampled_line_graph_f1,
    evaluate_from_root_ids,
    evaluate_suite,
    sample_synapse_pairs,
)


def test_two_synapses_same_pre_root_produce_one_edge():
    pre = np.array([1, 1])
    post = np.array([10, 11])
    assert build_true_line_graph(pre, post) == {(0, 1)}


def test_two_synapses_same_post_root_produce_one_edge():
    pre = np.array([1, 2])
    post = np.array([10, 10])
    assert build_true_line_graph(pre, post) == {(0, 1)}


def test_edges_are_canonical_min_max_pairs():
    pre = np.array([7, 7])
    post = np.array([99, 99])
    for a, b in build_true_line_graph(pre, post):
        assert a <= b


def test_compute_line_graph_f1_perfect():
    edges = {(0, 1), (2, 3)}
    m = compute_line_graph_f1(edges, edges, 4)
    assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0)


def test_compute_line_graph_f1_empty_sets_is_zero_not_nan():
    m = compute_line_graph_f1(set(), set(), 4)
    assert (m.precision, m.recall, m.f1) == (0.0, 0.0, 0.0)


def test_evaluate_from_root_ids_matches_build_true_line_graph():
    pre = np.array([1, 1, 2])
    post = np.array([10, 11, 12])
    m = evaluate_from_root_ids(pre, post, pre, post)
    assert m.f1 == pytest.approx(1.0)


def test_sample_synapse_pairs_returns_all_when_max_exceeds_total():
    pairs = sample_synapse_pairs(4, max_pairs=100)
    assert len(pairs) == 6


def test_sample_synapse_pairs_respects_max_pairs():
    pairs = sample_synapse_pairs(1000, max_pairs=50)
    assert len(pairs) == 50


def test_compute_sampled_line_graph_f1_on_full_sample_matches_exact():
    pre = np.array([1, 1, 2, 2, 3])
    true_edges = build_true_pairs_pre(pre)
    m_exact = compute_line_graph_f1(true_edges, true_edges, 5)
    m_sampled = compute_sampled_line_graph_f1(true_edges, true_edges, 5, max_pairs=1000)
    assert m_sampled.f1 == pytest.approx(m_exact.f1)


# ---------------------------------------------------------------------------
# evaluate_suite: single-side
# ---------------------------------------------------------------------------

def test_evaluate_suite_pre_only_matches_pair_set_form():
    pre = np.array([1, 1, 2, 2, 3])
    post = np.array([10, 11, 12, 13, 14])
    pred = np.array([1, 1, 9, 9, 9])
    suite = evaluate_suite(pred, pre, post)
    expected = compute_line_graph_f1(build_true_pairs_pre(pre), build_true_pairs_pre(pred), 5)
    assert suite.pre_only.f1 == pytest.approx(expected.f1)
    assert suite.pre_only.tp == expected.tp


def test_evaluate_suite_or_metric_matches_pair_set_form():
    pre = np.array([1, 1, 2, 2])
    post = np.array([10, 10, 11, 12])
    pred = np.array([1, 1, 9, 9])
    suite = evaluate_suite(pred, pre, post)
    true_or = build_true_line_graph(pre, post)
    est = build_true_pairs_pre(pred)
    expected = compute_line_graph_f1(true_or, est, 4)
    assert suite.or_metric.f1 == pytest.approx(expected.f1)


def test_evaluate_suite_without_pred_post_leaves_post_and_and_none():
    pre = np.array([1, 1])
    post = np.array([10, 10])
    pred = np.array([1, 1])
    suite = evaluate_suite(pred, pre, post)
    assert suite.post_only is None
    assert suite.and_metric is None


# ---------------------------------------------------------------------------
# evaluate_suite: dual-side
# ---------------------------------------------------------------------------

def test_evaluate_suite_post_only_matches_pair_set_form():
    pre = np.array([1, 2, 3, 4])
    post = np.array([10, 10, 11, 11])
    pred_pre = np.array([1, 2, 3, 4])
    pred_post = np.array([10, 10, 9, 9])
    suite = evaluate_suite(pred_pre, pre, post, pred_post)
    expected = compute_line_graph_f1(build_true_pairs_post(post), build_true_pairs_post(pred_post), 4)
    assert suite.post_only.f1 == pytest.approx(expected.f1)


def test_evaluate_suite_and_metric_matches_pair_set_form():
    pre = np.array([1, 1, 2, 2])
    post = np.array([10, 11, 10, 11])
    pred_pre = np.array([1, 1, 2, 2])
    pred_post = np.array([10, 11, 10, 11])
    suite = evaluate_suite(pred_pre, pre, post, pred_post)
    true_and = build_true_pairs_and(pre, post)
    est_and = build_true_pairs_and(pred_pre, pred_post)
    expected = compute_line_graph_f1(true_and, est_and, 4)
    assert suite.and_metric.f1 == pytest.approx(expected.f1)
    assert suite.and_metric.tp == expected.tp
    assert suite.and_metric.fp == expected.fp
    assert suite.and_metric.fn == expected.fn


def test_evaluate_suite_counting_form_matches_pair_set_form_on_random_data():
    rng = np.random.default_rng(42)
    n = 60
    pre = rng.integers(0, 12, size=n)
    post = rng.integers(0, 10, size=n)
    pred_pre = rng.integers(0, 9, size=n)
    pred_post = rng.integers(0, 7, size=n)
    suite = evaluate_suite(pred_pre, pre, post, pred_post)

    exp_pre = compute_line_graph_f1(build_true_pairs_pre(pre), build_true_pairs_pre(pred_pre), n)
    exp_or = compute_line_graph_f1(build_true_line_graph(pre, post), build_true_pairs_pre(pred_pre), n)
    exp_post = compute_line_graph_f1(build_true_pairs_post(post), build_true_pairs_post(pred_post), n)
    exp_and = compute_line_graph_f1(build_true_pairs_and(pre, post), build_true_pairs_and(pred_pre, pred_post), n)

    for got, exp, name in ((suite.pre_only, exp_pre, "pre_only"),
                           (suite.or_metric, exp_or, "or_metric"),
                           (suite.post_only, exp_post, "post_only"),
                           (suite.and_metric, exp_and, "and_metric")):
        assert got.tp == exp.tp, name
        assert got.fp == exp.fp, name
        assert got.fn == exp.fn, name
        assert got.f1 == pytest.approx(exp.f1), name


def test_to_dict_flattens_all_present_variants():
    pre = np.array([1, 1])
    post = np.array([10, 10])
    pred = np.array([1, 1])
    suite = evaluate_suite(pred, pre, post)
    d = suite.to_dict()
    assert "lg_pre_only_f1" in d
    assert "lg_or_metric_precision" in d
    assert "lg_post_only_f1" not in d


def test_matches_legacy_neuronauts_line_graph_module():
    """Cross-check against the pre-consolidation neuronauts.line_graph."""
    from neuronauts.line_graph import evaluate_suite as legacy_evaluate_suite

    rng = np.random.default_rng(1)
    n = 50
    pre = rng.integers(0, 10, size=n)
    post = rng.integers(0, 8, size=n)
    pred_pre = rng.integers(0, 7, size=n)
    pred_post = rng.integers(0, 6, size=n)

    new = evaluate_suite(pred_pre, pre, post, pred_post)
    legacy = legacy_evaluate_suite(pred_pre, pre, post, pred_post)
    for name in ("pre_only", "or_metric", "post_only", "and_metric"):
        a, b = getattr(new, name), getattr(legacy, name)
        assert (a.tp, a.fp, a.fn) == (b.tp, b.fp, b.fn), name
        assert a.f1 == pytest.approx(b.f1), name
