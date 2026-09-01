"""Connectome-level metrics: neuronauts.metrics.connectome."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from neuronauts.metrics import (
    connectome_metrics,
    dual_side_connectome_metrics,
    edge_set_prf1,
    match_clusters_majority,
    undirected_edge_set,
)


def test_undirected_edge_set_sums_reciprocal_counts():
    from collections import Counter
    counts = Counter({(1, 2): 2, (2, 1): 1, (3, 4): 1})
    edges = undirected_edge_set(counts, min_syn=3)
    assert edges == {(1, 2)}


def test_undirected_edge_set_keeps_autapses():
    from collections import Counter
    edges = undirected_edge_set(Counter({(5, 5): 2}), min_syn=1)
    assert edges == {(5, 5)}


def test_edge_set_prf1_nan_when_no_predictions():
    p, r, f = edge_set_prf1({(1, 2)}, set())
    assert math.isnan(p)
    assert r == 0.0


def test_match_clusters_majority_ignores_ignore_label():
    # cluster 2's items have true labels [8, 0]; the 0 is the ignore sentinel
    # and must not count toward the majority vote, leaving 8 as the winner.
    pred = np.array([3, 3, 2, 2])
    true = np.array([9, 9, 8, 0])
    m = match_clusters_majority(pred, true, ignore=0)
    assert m == {3: 9, 2: 8}


def test_connectome_metrics_perfect_partition():
    n = 10
    pre = np.arange(1, n + 1)
    post = np.arange(101, 101 + n)
    out = connectome_metrics(pre, pre, post, min_syn=1)
    assert out["synapse_attr_acc"] == pytest.approx(1.0)
    assert out["conn_edge_f1"] == pytest.approx(1.0)
    assert out["conn_edge_f1_undir"] == pytest.approx(1.0)


def test_connectome_metrics_false_merge_inflates_predicted_edges():
    # two pre-neurons (1 and 2) both talk to post-neuron 100; merging them
    # into one predicted cluster invents a spurious edge count but the
    # directed edge set itself is unaffected here (still 2 -> 100 edges)
    true_pre = np.array([1, 1, 2, 2])
    true_post = np.array([100, 100, 100, 100])
    pred = np.array([9, 9, 9, 9])  # both neurons merged into cluster 9
    out = connectome_metrics(pred, true_pre, true_post, min_syn=1)
    # cluster 9 majority-votes to neuron 1 (tie broken by Counter order); either
    # way, synapse_attr_acc must be < 1 since neuron 2's synapses get mis-attributed
    assert out["synapse_attr_acc"] < 1.0


def test_connectome_metrics_ignore_label_excludes_unknowns():
    # item 0 has pred cluster == ignore (0), so it drops out; items 1-2 remain.
    pre = np.array([0, 1, 1])
    post = np.array([0, 100, 100])
    out = connectome_metrics(pre, pre, post, min_syn=1, ignore=0)
    assert out["n_synapses_labelled"] == 2


def test_dual_side_connectome_perfect_recovers_ground_truth():
    n = 6
    syn_id = np.arange(n)
    true_pre = np.arange(1, n + 1)
    true_post = np.arange(101, 101 + n)
    out = dual_side_connectome_metrics(
        pred_pre=true_pre, syn_id_pre=syn_id, true_pre=true_pre, true_post=true_post,
        pred_post=true_post, syn_id_post=syn_id, true_post_on_post_side=true_post,
    )
    assert out["conn_edge_f1"] == pytest.approx(1.0)
    assert out["n_synapses_both_sides"] == n
    assert out["n_synapses_pre_only"] == 0


def test_dual_side_connectome_reports_one_sided_coverage():
    syn_id_pre = np.array([0, 1, 2])
    syn_id_post = np.array([1, 2, 3])
    true_pre = np.array([1, 1, 1])
    true_post = np.array([100, 100, 100])
    out = dual_side_connectome_metrics(
        pred_pre=true_pre, syn_id_pre=syn_id_pre, true_pre=true_pre, true_post=true_post,
        pred_post=true_post, syn_id_post=syn_id_post, true_post_on_post_side=true_post,
    )
    assert out["n_synapses_both_sides"] == 2
    assert out["n_synapses_pre_only"] == 1
    assert out["n_synapses_post_only"] == 1


def test_matches_legacy_connectome_accuracy():
    """Cross-check against treestitch.connectivity.connectome_accuracy."""
    from treestitch.connectivity import connectome_accuracy

    rng = np.random.default_rng(4)
    n = 80
    pre = rng.integers(1, 8, size=n).astype(np.int64)
    post = rng.integers(100, 106, size=n).astype(np.int64)
    pred = rng.integers(0, 6, size=n).astype(np.int64)
    region = SimpleNamespace(pre_root_id=pre, post_root_id=post)

    legacy = connectome_accuracy(pred, region, min_syn=1)
    new = connectome_metrics(pred, pre, post, min_syn=1)
    for k in ("synapse_attr_acc", "conn_edge_precision", "conn_edge_recall",
              "conn_edge_f1", "n_true_edges", "n_pred_edges",
              "conn_edge_precision_undir", "conn_edge_recall_undir",
              "conn_edge_f1_undir", "n_true_edges_undir", "n_pred_edges_undir",
              "n_synapses_labelled"):
        a, b = legacy[k], new[k]
        if isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), k
        else:
            assert a == pytest.approx(b), k
