"""Candidate-edge merge/split confusion: neuronauts.metrics.edges."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neuronauts.metrics import edge_merge_metrics


def _chain_graph():
    # 4 items in 2 true objects {0,1} {2,3}; candidate edges are the 3 "chain" pairs
    src = np.array([0, 1, 2])
    dst = np.array([1, 2, 3])
    true = np.array([5, 5, 6, 6])
    return src, dst, true


def test_perfect_prediction_has_full_precision_and_recall():
    src, dst, true = _chain_graph()
    pred = true.copy()
    m = edge_merge_metrics(src, dst, pred, true)
    assert m["merge_precision"] == pytest.approx(1.0)
    assert m["merge_recall"] == pytest.approx(1.0)
    assert m["fp_merges"] == 0
    assert m["fn_merges"] == 0


def test_all_one_cluster_overmerges():
    src, dst, true = _chain_graph()
    pred = np.zeros(4, dtype=int)
    m = edge_merge_metrics(src, dst, pred, true)
    # all 3 edges predicted merged; edges (0,1) and (2,3) are truly merges,
    # (1,2) crosses the true boundary and is the one false merge.
    assert m["tp_merges"] == 2
    assert m["fp_merges"] == 1
    assert m["merge_recall"] == pytest.approx(1.0)
    assert m["over_merge_rate"] == pytest.approx(1 / 3)


def test_all_singletons_undermerges():
    src, dst, true = _chain_graph()
    pred = np.arange(4)
    m = edge_merge_metrics(src, dst, pred, true)
    assert m["tp_merges"] == 0
    assert m["fn_merges"] == 2
    # no predicted merges at all: precision is undefined (NaN) by default.
    assert math.isnan(m["merge_precision"])


def test_ignore_label_drops_edges_touching_unknown_truth():
    src = np.array([0, 1])
    dst = np.array([1, 2])
    true = np.array([0, 5, 5])   # item 0 unknown
    pred = np.array([9, 1, 1])
    m = edge_merge_metrics(src, dst, pred, true, ignore=0)
    assert m["n_edges_eval"] == 1


def test_frankenmerge_split_recall_perfect_when_labels_used():
    # same_fragment edge whose endpoints differ in truth AND in prediction -> split correctly
    src = np.array([0])
    dst = np.array([1])
    true = np.array([5, 6])
    pred = np.array([5, 6])
    m = edge_merge_metrics(src, dst, pred, true, same_fragment=np.array([True]))
    assert m["frankenmerge_rate"] == pytest.approx(1.0)
    assert m["n_frankenmerge_edges"] == 1
    assert m["frankenmerge_split_recall"] == pytest.approx(1.0)


def test_frankenmerge_split_recall_zero_when_merged():
    src = np.array([0])
    dst = np.array([1])
    true = np.array([5, 6])
    pred = np.array([9, 9])
    m = edge_merge_metrics(src, dst, pred, true, same_fragment=np.array([True]))
    assert m["frankenmerge_split_recall"] == pytest.approx(0.0)


def test_abstain_rate_and_abstained_edges_count_as_split():
    src = np.array([0])
    dst = np.array([1])
    true = np.array([5, 5])
    pred = np.array([-1, -1])
    m = edge_merge_metrics(src, dst, pred, true, abstain=np.array([True, True]))
    assert m["abstain_rate"] == pytest.approx(1.0)
    assert m["fn_merges"] == 1   # abstained edge is NOT counted as merged, even though pred[0]==pred[1]


def test_empty_edge_set():
    m = edge_merge_metrics(np.array([], dtype=int), np.array([], dtype=int),
                           np.array([1, 2]), np.array([1, 2]))
    assert m["n_edges_eval"] == 0
    assert math.isnan(m["merge_precision"])


def test_rejects_misaligned_src_dst():
    with pytest.raises(ValueError):
        edge_merge_metrics(np.array([0, 1]), np.array([0]), np.array([1, 1]), np.array([1, 1]))


def test_matches_legacy_edge_partition_edge_merge_metrics():
    """Cross-check against neuronauts.assemble.edge_partition.edge_merge_metrics
    (the pre-consolidation implementation), which the new function replaces via
    a delegating wrapper."""
    from neuronauts.assemble.half_synapse_graph import HalfSynapseGraph
    from neuronauts.assemble.edge_partition import edge_merge_metrics as legacy_edge_merge_metrics

    rng = np.random.default_rng(0)
    n = 30
    labels = rng.integers(1, 6, size=n).astype(np.int64)
    edge_src = rng.integers(0, n, size=60).astype(np.int64)
    edge_dst = rng.integers(0, n, size=60).astype(np.int64)
    edge_type = (rng.random(60) < 0.5).astype(np.int64)
    pred = rng.integers(0, 5, size=n).astype(np.int64)

    graph = HalfSynapseGraph(
        node_feat=np.zeros((n, 1), dtype=np.float32),
        node_pos=np.zeros((n, 3), dtype=np.float32),
        edge_src=edge_src, edge_dst=edge_dst,
        edge_type=edge_type, edge_feat=np.zeros((60, 3), dtype=np.float32),
        labels=labels,
        seg_id=np.zeros(n, dtype=np.int64),
        side="pre",
    )
    legacy = legacy_edge_merge_metrics(graph, pred)
    new = edge_merge_metrics(edge_src, edge_dst, pred, labels,
                             same_fragment=(edge_type == 0))
    for k in ("merge_precision", "merge_recall", "merge_f1",
              "over_merge_rate", "under_merge_rate", "n_edges_eval",
              "tp_merges", "fp_merges", "fn_merges", "tn_splits",
              "frankenmerge_rate", "frankenmerge_split_recall"):
        a, b = legacy[k], new[k]
        if isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b), k
        else:
            assert a == pytest.approx(b), k
