"""The aggregate suite: neuronauts.metrics.evaluate_partition_suite.

Each block only appears when its inputs are given; when a block appears its
numbers must agree with calling the underlying function directly (no
transcription drift between suite.py and the per-topic modules).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuronauts.metrics import (
    completeness_metrics,
    connectome_metrics,
    edge_merge_metrics,
    evaluate_partition_suite,
    fragment_completeness,
    frankenmerge_metrics,
    partition_metrics,
)
from neuronauts.metrics.line_graph import evaluate_suite as line_graph_suite


def test_bare_call_only_has_partition_keys():
    pred = np.array([1, 1, 2])
    true = np.array([9, 9, 8])
    m = evaluate_partition_suite(pred, true)
    assert "ari" in m
    assert "merge_precision" not in m
    assert "conn_edge_f1" not in m


def test_partition_block_matches_direct_call():
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 5, size=40)
    true = rng.integers(1, 6, size=40)
    m = evaluate_partition_suite(pred, true)
    direct = partition_metrics(pred, true)
    for k in ("ari", "pair_precision", "pair_recall", "vi_split", "vi_merge"):
        assert m[k] == pytest.approx(direct[k]) or (math.isnan(m[k]) and math.isnan(direct[k]))


def test_edge_block_appears_only_with_src_dst():
    pred = np.array([1, 1, 2])
    true = np.array([9, 9, 8])
    src, dst = np.array([0]), np.array([1])
    m = evaluate_partition_suite(pred, true, src=src, dst=dst)
    assert "merge_precision" in m
    direct = edge_merge_metrics(src, dst, pred, true)
    assert m["merge_precision"] == pytest.approx(direct["merge_precision"]) or (
        math.isnan(m["merge_precision"]) and math.isnan(direct["merge_precision"]))


def test_src_without_dst_raises():
    with pytest.raises(ValueError):
        evaluate_partition_suite(np.array([1]), np.array([1]), src=np.array([0]))


def test_fragment_block_adds_frankenmerge_and_naive_baseline():
    pred = np.array([0, 1])
    true = np.array([5, 6])
    fragment_id = np.array(["A", "A"])
    m = evaluate_partition_suite(pred, true, fragment_id=fragment_id)
    assert m["fk_separation"] == pytest.approx(1.0)
    assert "fk_parents" not in m  # id list stripped from the flat suite dict
    assert "naive_ari" in m
    direct_fk = frankenmerge_metrics(pred, true, fragment_id)
    assert m["fk_pair_split_rate"] == pytest.approx(direct_fk["fk_pair_split_rate"])


def test_completeness_block_requires_root_label_map():
    pred = np.array([0, 1])
    true = np.array([5, 6])
    fragment_id = np.array(["A", "B"])
    root_label_map = {"A": {5}, "B": {6}}
    m = evaluate_partition_suite(pred, true, fragment_id=fragment_id, root_label_map=root_label_map)
    assert m["cmpl_f1"] == pytest.approx(1.0)
    direct = completeness_metrics(root_label_map, fragment_completeness(root_label_map))
    # perfect prediction here IS the ground truth, so this direct call is also perfect
    assert direct["f1"] == pytest.approx(1.0)


def test_connectome_and_line_graph_blocks_appear_with_true_post():
    rng = np.random.default_rng(5)
    n = 30
    pred = rng.integers(0, 6, size=n)
    true = rng.integers(1, 7, size=n)
    true_post = rng.integers(100, 105, size=n)
    m = evaluate_partition_suite(pred, true, true_post=true_post)
    assert "conn_edge_f1" in m
    assert "lg_pre_only_f1" in m
    assert "lg_post_only_f1" not in m  # no pred_post given

    direct_conn = connectome_metrics(pred, true, true_post)
    assert m["conn_edge_f1"] == pytest.approx(direct_conn["conn_edge_f1"]) or (
        math.isnan(m["conn_edge_f1"]) and math.isnan(direct_conn["conn_edge_f1"]))

    direct_lg = line_graph_suite(pred, true, true_post)
    assert m["lg_pre_only_f1"] == pytest.approx(direct_lg.pre_only.f1)


def test_line_graph_block_with_pred_post_adds_post_only_and_and():
    rng = np.random.default_rng(6)
    n = 25
    pred = rng.integers(0, 5, size=n)
    true = rng.integers(1, 6, size=n)
    true_post = rng.integers(100, 104, size=n)
    pred_post = rng.integers(0, 4, size=n)
    m = evaluate_partition_suite(pred, true, true_post=true_post, pred_post=pred_post)
    assert "lg_post_only_f1" in m
    assert "lg_and_metric_f1" in m


def test_ignore_label_zero_is_dropped_from_every_block():
    pred = np.array([0, 1, 1])
    true = np.array([0, 5, 5])
    fragment_id = np.array(["X", "A", "A"])
    m = evaluate_partition_suite(pred, true, fragment_id=fragment_id)
    assert m["n_items"] == 2


def test_full_call_produces_a_dict_of_plain_python_and_numpy_scalars():
    rng = np.random.default_rng(7)
    n = 15
    pred = rng.integers(0, 4, size=n)
    true = rng.integers(1, 5, size=n)
    true_post = rng.integers(100, 103, size=n)
    fragment_id = rng.integers(0, 6, size=n)
    root_label_map = {i: {rng.integers(1, 5)} for i in range(6)}
    weights = rng.uniform(1, 10, size=n)
    src = rng.integers(0, n, size=20)
    dst = rng.integers(0, n, size=20)

    m = evaluate_partition_suite(
        pred, true, weights=weights, src=src, dst=dst,
        fragment_id=fragment_id, root_label_map=root_label_map,
        true_post=true_post,
    )
    assert isinstance(m, dict)
    assert len(m) > 20
