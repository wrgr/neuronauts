"""Smoke tests for the treestitch.partition public-API wrappers.

The wrappers are thin delegators to neuronauts.assemble.edge_partition;
these tests verify the delegation contract: correct return types, key sets,
and shapes, without re-testing the underlying algorithm.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_edge_partition import _make_separable_graph


# ---------------------------------------------------------------------------
# train_edge_partition wrapper
# ---------------------------------------------------------------------------

def test_train_edge_partition_returns_model_and_history():
    pytest.importorskip("torch")
    from treestitch.partition import train_edge_partition
    g = _make_separable_graph(n_objects=2, per_object=4)
    model, hist = train_edge_partition(g, n_epochs=5, log_every=0)
    assert model is not None
    assert {"loss", "p_pos", "p_neg", "edge_acc"} <= set(hist.keys())
    assert len(hist["loss"]) == 5


def test_train_edge_partition_franken_hard_frac_kwarg():
    pytest.importorskip("torch")
    from treestitch.partition import train_edge_partition
    g = _make_separable_graph(n_objects=2, per_object=4)
    model, _ = train_edge_partition(g, n_epochs=3, log_every=0, franken_hard_frac=0.2)
    assert model is not None


# ---------------------------------------------------------------------------
# partition_observations_cc wrapper
# ---------------------------------------------------------------------------

def test_partition_observations_cc_returns_array():
    pytest.importorskip("torch")
    from treestitch.partition import partition_observations_cc, train_edge_partition
    g = _make_separable_graph(n_objects=2, per_object=4)
    model, _ = train_edge_partition(g, n_epochs=5, log_every=0)
    pred = partition_observations_cc(model, g)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (g.n_nodes,)


def test_partition_observations_cc_abstain_threshold_kwarg():
    pytest.importorskip("torch")
    from treestitch.partition import partition_observations_cc, train_edge_partition
    g = _make_separable_graph(n_objects=3, per_object=4)
    model, _ = train_edge_partition(g, n_epochs=10, log_every=0, seed=0)
    pred = partition_observations_cc(model, g, abstain_threshold=0.5)
    assert pred.shape == (g.n_nodes,)
    # Some nodes may be abstained (negative IDs); positive nodes ≥ 0
    assert (pred >= 0).any()


# ---------------------------------------------------------------------------
# partition_observations_soft wrapper
# ---------------------------------------------------------------------------

def test_partition_observations_soft_returns_dict():
    pytest.importorskip("torch")
    from treestitch.partition import partition_observations_soft, train_edge_partition
    g = _make_separable_graph(n_objects=2, per_object=4)
    model, _ = train_edge_partition(g, n_epochs=5, log_every=0)
    result = partition_observations_soft(model, g)
    assert isinstance(result, dict)
    expected_keys = {"pred", "cluster_conf", "membership_probs", "cluster_ids", "entropy", "abstain_mask"}
    assert expected_keys == set(result.keys())


def test_partition_observations_soft_membership_rows_sum_to_one():
    pytest.importorskip("torch")
    from treestitch.partition import partition_observations_soft, train_edge_partition
    g = _make_separable_graph(n_objects=2, per_object=4)
    model, _ = train_edge_partition(g, n_epochs=5, log_every=0)
    result = partition_observations_soft(model, g)
    row_sums = result["membership_probs"].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# merge_metrics wrapper
# ---------------------------------------------------------------------------

def test_merge_metrics_perfect_labels():
    from treestitch.partition import merge_metrics
    g = _make_separable_graph()
    m = merge_metrics(g, g.labels)
    assert m["merge_precision"] == pytest.approx(1.0)
    assert m["over_merge_rate"] == pytest.approx(0.0)


def test_merge_metrics_all_same_cluster():
    from treestitch.partition import merge_metrics
    g = _make_separable_graph()
    all_one = np.zeros(g.n_nodes, dtype=np.int64)
    m = merge_metrics(g, all_one)
    assert m["merge_recall"] == pytest.approx(1.0)
    assert m["over_merge_rate"] > 0.0


def test_merge_metrics_has_extended_keys():
    from treestitch.partition import merge_metrics
    g = _make_separable_graph()
    m = merge_metrics(g, g.labels)
    for key in ("frankenmerge_rate", "frankenmerge_split_recall", "abstain_rate"):
        assert key in m, f"Missing key: {key}"
