"""Tests for edge-classification + correlation clustering (learn f(117→1412))."""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.assemble.edge_partition import (
    EdgePartitionGNN,
    _abstain_uncertain,
    correlation_cluster,
    edge_merge_metrics,
    partition_by_correlation,
    soft_partition,
    train_edge_partition_gnn,
)
from neuronauts.assemble.half_synapse_graph import HalfSynapseGraph


# ---------------------------------------------------------------------------
# correlation_cluster (GAEC) — pure algorithm, no torch
# ---------------------------------------------------------------------------

def _undirected(pairs_weights):
    """Helper: build bidirectional src/dst/weight arrays from (u,v,w) tuples."""
    src, dst, w = [], [], []
    for u, v, ww in pairs_weights:
        src += [u, v]
        dst += [v, u]
        w += [ww, ww]
    return (np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64),
            np.array(w, dtype=np.float64))


def test_cc_two_positive_cliques_split_by_negative():
    # Two triangles, strong positive internal edges, strong negative bridge.
    edges = [
        (0, 1, 5.0), (1, 2, 5.0), (0, 2, 5.0),
        (3, 4, 5.0), (4, 5, 5.0), (3, 5, 5.0),
        (2, 3, -8.0),
    ]
    src, dst, w = _undirected(edges)
    labels = correlation_cluster(6, src, dst, w)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    assert len(np.unique(labels)) == 2


def test_cc_all_negative_stays_singletons():
    edges = [(0, 1, -1.0), (1, 2, -1.0), (0, 2, -1.0)]
    src, dst, w = _undirected(edges)
    labels = correlation_cluster(3, src, dst, w)
    assert len(np.unique(labels)) == 3


def test_cc_all_positive_merges_all():
    edges = [(0, 1, 2.0), (1, 2, 2.0), (2, 3, 2.0)]
    src, dst, w = _undirected(edges)
    labels = correlation_cluster(4, src, dst, w)
    assert len(np.unique(labels)) == 1


def test_cc_net_negative_overrides_single_positive_edge():
    # 0-1 strongly positive, but two negative edges to a third node mean the
    # net evidence keeps a different structure: 0 and 1 merge, 2 stays apart.
    edges = [(0, 1, 4.0), (0, 2, -3.0), (1, 2, -3.0)]
    src, dst, w = _undirected(edges)
    labels = correlation_cluster(3, src, dst, w)
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_cc_cut_high_edge_when_global_disagrees():
    # A greedy threshold would merge the chain 0-1-2-3 across the weak-positive
    # 1-2 bridge.  GAEC keeps {0,1} and {2,3} apart because the bridge net
    # weight is negative once the strong cluster-internal negatives aggregate.
    edges = [
        (0, 1, 10.0), (2, 3, 10.0),
        (1, 2, 1.0), (0, 2, -6.0), (1, 3, -6.0),
    ]
    src, dst, w = _undirected(edges)
    labels = correlation_cluster(4, src, dst, w)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_cc_empty_graph_all_singletons():
    labels = correlation_cluster(
        4, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64)
    )
    assert len(np.unique(labels)) == 4


# ---------------------------------------------------------------------------
# Synthetic graph builder for the model tests
# ---------------------------------------------------------------------------

def _make_separable_graph(n_objects=4, per_object=6, dim=8, seed=0):
    """Graph where each object has a distinct DNA direction.

    Same-object pairs share DNA; nodes within an object are fully connected
    (type 0); a few cross-object spatial edges (type 1) are the hard negatives.
    """
    rng = np.random.default_rng(seed)
    dna_dirs = np.eye(n_objects, dim, dtype=np.float32)

    node_dna, labels, seg = [], [], []
    for obj in range(n_objects):
        for _ in range(per_object):
            node_dna.append(dna_dirs[obj] + rng.normal(0, 0.05, dim).astype(np.float32))
            labels.append(obj + 1)
            seg.append(obj + 1)  # one seg per object here
    node_dna = np.asarray(node_dna, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    seg = np.asarray(seg, dtype=np.int64)
    N = len(labels)
    pos = rng.normal(0, 1, (N, 3)).astype(np.float32)
    node_feat = np.concatenate([pos / 50_000.0, node_dna], axis=1).astype(np.float32)

    src, dst, etype = [], [], []
    # same-object (type 0) within each object block
    for obj in range(n_objects):
        idxs = [i for i in range(N) if labels[i] == obj + 1]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                src += [idxs[a], idxs[b]]
                dst += [idxs[b], idxs[a]]
                etype += [0, 0]
    # a few cross-object spatial edges (type 1) — hard negatives
    for _ in range(N):
        i, j = int(rng.integers(N)), int(rng.integers(N))
        if labels[i] != labels[j]:
            src += [i, j]
            dst += [j, i]
            etype += [1, 1]

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    etype = np.asarray(etype, dtype=np.int64)
    dn = node_dna / (np.linalg.norm(node_dna, axis=1, keepdims=True) + 1e-8)
    cos = (dn[src] * dn[dst]).sum(1).astype(np.float32)
    onehot = np.column_stack([(etype == 0).astype(np.float32),
                              (etype == 1).astype(np.float32)])
    edge_feat = np.column_stack([onehot, cos]).astype(np.float32)

    return HalfSynapseGraph(
        node_feat=node_feat, node_pos=pos, edge_src=src, edge_dst=dst,
        edge_type=etype, edge_feat=edge_feat, labels=labels, seg_id=seg, side="pre",
    )


# ---------------------------------------------------------------------------
# EdgePartitionGNN model + training
# ---------------------------------------------------------------------------

def test_edge_model_forward_shapes():
    pytest.importorskip("torch")
    import torch

    g = _make_separable_graph()
    model = EdgePartitionGNN(input_dim=g.node_dim, n_edge_types=2, edge_feat_dim=3)
    emb, logits = model(
        torch.from_numpy(g.node_feat),
        torch.from_numpy(g.edge_src).long(),
        torch.from_numpy(g.edge_dst).long(),
        torch.from_numpy(g.edge_type).long(),
        torch.from_numpy(g.edge_feat).float(),
    )
    assert emb.shape[0] == g.n_nodes
    assert logits.shape[0] == g.n_edges
    assert torch.isfinite(logits).all()


def test_train_edge_partition_separates():
    pytest.importorskip("torch")
    g = _make_separable_graph()
    model, hist = train_edge_partition_gnn(g, n_epochs=40, log_every=0, seed=0)
    # After training, predicted same-object probability should be much higher
    # on true positives than on true negatives.
    assert hist["p_pos"][-1] > hist["p_neg"][-1] + 0.3
    assert hist["edge_acc"][-1] > 0.8


def test_edge_cc_recovers_partition():
    pytest.importorskip("torch")
    g = _make_separable_graph(n_objects=4, per_object=6)
    model, _ = train_edge_partition_gnn(g, n_epochs=60, log_every=0, seed=0)
    pred = partition_by_correlation(model, g)
    from neuronauts.assemble.partition_gnn import evaluate_partition_ari
    r = evaluate_partition_ari(pred, g.labels)
    assert r["ari"] > 0.7


def test_edge_partition_empty_graph_is_safe():
    pytest.importorskip("torch")
    g = HalfSynapseGraph(
        node_feat=np.zeros((3, 5), np.float32), node_pos=np.zeros((3, 3), np.float32),
        edge_src=np.zeros(0, np.int64), edge_dst=np.zeros(0, np.int64),
        edge_type=np.zeros(0, np.int64), edge_feat=np.zeros((0, 3), np.float32),
        labels=np.array([1, 1, 2], np.int64), seg_id=np.array([1, 1, 2], np.int64),
        side="pre",
    )
    model, hist = train_edge_partition_gnn(g, n_epochs=5, log_every=0)
    pred = partition_by_correlation(model, g)
    assert len(pred) == 3
    assert hist["loss"] == []  # no supervised edges → no training


# ---------------------------------------------------------------------------
# edge_merge_metrics
# ---------------------------------------------------------------------------

def test_edge_merge_metrics_perfect():
    g = _make_separable_graph()
    # Perfect prediction = the ground-truth labels themselves.
    m = edge_merge_metrics(g, g.labels)
    assert m["merge_precision"] == pytest.approx(1.0)
    assert m["over_merge_rate"] == pytest.approx(0.0)


def test_edge_merge_metrics_all_one_cluster_overmerges():
    g = _make_separable_graph()
    allone = np.zeros(g.n_nodes, dtype=np.int64)
    m = edge_merge_metrics(g, allone)
    # Everything merged → recall perfect, but cross-object spatial edges are
    # now false merges, so over-merge rate is positive.
    assert m["merge_recall"] == pytest.approx(1.0)
    assert m["over_merge_rate"] > 0.0


def test_edge_merge_metrics_all_singletons_undermerges():
    g = _make_separable_graph()
    sing = np.arange(g.n_nodes, dtype=np.int64)
    m = edge_merge_metrics(g, sing)
    assert m["merge_recall"] == pytest.approx(0.0)
    assert m["under_merge_rate"] > 0.0
    assert m["over_merge_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Frankenmerge graph fixture
# ---------------------------------------------------------------------------

def _make_frankenmerge_graph():
    """Frankenmerge: seg_id=1 spans neuron 1 (nodes 0-3) AND neuron 2 (nodes 4-7).

    The type-0 (same-fragment) edges within seg=1 include cross-neuron cuts —
    these are the frankenmerge cut edges the model must learn to detect.
    Nodes 8-9 form a clean seg=2 fragment for neuron 2.
    """
    labels  = np.array([1, 1, 1, 1,  2, 2, 2, 2,  2, 2], dtype=np.int64)
    seg_ids = np.array([1, 1, 1, 1,  1, 1, 1, 1,  2, 2], dtype=np.int64)
    N = 10
    node_feat = np.zeros((N, 5), dtype=np.float32)
    node_pos  = np.zeros((N, 3), dtype=np.float32)

    src_list, dst_list, etype_list = [], [], []
    # Type-0 within seg=1 (all 8-node pairs, incl. cross-neuron frankenmerge cuts)
    for i in range(8):
        for j in range(i + 1, 8):
            src_list += [i, j]; dst_list += [j, i]; etype_list += [0, 0]
    # Type-0 within seg=2
    src_list += [8, 9]; dst_list += [9, 8]; etype_list += [0, 0]
    # Type-1 cross-fragment spatial edge
    src_list += [0, 8]; dst_list += [8, 0]; etype_list += [1, 1]

    src   = np.array(src_list,   dtype=np.int64)
    dst   = np.array(dst_list,   dtype=np.int64)
    etype = np.array(etype_list, dtype=np.int64)
    return HalfSynapseGraph(
        node_feat=node_feat, node_pos=node_pos,
        edge_src=src, edge_dst=dst, edge_type=etype,
        edge_feat=np.zeros((len(src), 3), dtype=np.float32),
        labels=labels, seg_id=seg_ids, side="pre",
    )


# ---------------------------------------------------------------------------
# _abstain_uncertain
# ---------------------------------------------------------------------------

def test_abstain_uncertain_no_conflicting_edges_no_abstention():
    # Each cluster has only internal edges with high prob → no conflicting evidence.
    pred  = np.array([0, 0,  1, 1], dtype=np.int64)
    src   = np.array([0, 2], dtype=np.int64)
    dst   = np.array([1, 3], dtype=np.int64)
    probs = np.array([0.9, 0.9], dtype=np.float32)
    result = _abstain_uncertain(pred, src, dst, probs, threshold=0.5)
    assert (result >= 0).all()


def test_abstain_uncertain_conflicting_evidence_triggers_abstention():
    # Node 0 is in cluster 0; it has a high-p edge to node 1 (same cluster)
    # AND a high-p edge to node 2 (different cluster) → confidence = 0 → abstain.
    pred  = np.array([0, 0, 1], dtype=np.int64)
    src   = np.array([0, 0], dtype=np.int64)
    dst   = np.array([1, 2], dtype=np.int64)
    probs = np.array([0.9, 0.9], dtype=np.float32)
    result = _abstain_uncertain(pred, src, dst, probs, threshold=0.5)
    assert result[0] < 0, "Node 0 should abstain: equal same- and diff-cluster evidence"
    assert result[1] >= 0, "Node 1 unambiguous: only same-cluster edge"


def test_abstain_uncertain_negative_ids_are_unique():
    # All four nodes have cross-cluster edges → all abstain; IDs must be distinct.
    pred  = np.array([0, 0, 1, 1], dtype=np.int64)
    src   = np.array([0, 1, 2, 3], dtype=np.int64)
    dst   = np.array([2, 3, 0, 1], dtype=np.int64)
    probs = np.ones(4, dtype=np.float32) * 0.9
    result = _abstain_uncertain(pred, src, dst, probs, threshold=0.5)
    neg = result[result < 0]
    assert len(neg) == len(set(neg.tolist())), "Abstained IDs must be unique"


def test_abstain_threshold_zero_bypass_via_partition_by_correlation():
    # The threshold=0 bypass is enforced at the partition_by_correlation level.
    # Verify that abstain_threshold=0.0 produces no negative IDs.
    pytest.importorskip("torch")
    g = _make_separable_graph(n_objects=3, per_object=4)
    model, _ = train_edge_partition_gnn(g, n_epochs=5, log_every=0, seed=0)
    pred = partition_by_correlation(model, g, abstain_threshold=0.0)
    assert (pred >= 0).all(), "abstain_threshold=0 must produce no abstentions"


# ---------------------------------------------------------------------------
# soft_partition
# ---------------------------------------------------------------------------

def test_soft_partition_output_shapes():
    pytest.importorskip("torch")
    g = _make_separable_graph(n_objects=3, per_object=4)
    model, _ = train_edge_partition_gnn(g, n_epochs=5, log_every=0, seed=0)
    result = soft_partition(model, g)
    N = g.n_nodes
    K = len(result["cluster_ids"])
    assert result["pred"].shape == (N,)
    assert result["cluster_conf"].shape == (N,)
    assert result["membership_probs"].shape == (N, K)
    assert result["entropy"].shape == (N,)
    assert result["abstain_mask"].shape == (N,)
    assert result["abstain_mask"].dtype == bool


def test_soft_partition_membership_rows_sum_to_one():
    pytest.importorskip("torch")
    g = _make_separable_graph(n_objects=3, per_object=4)
    model, _ = train_edge_partition_gnn(g, n_epochs=5, log_every=0, seed=0)
    result = soft_partition(model, g)
    row_sums = result["membership_probs"].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_soft_partition_abstain_mask_consistent_with_pred():
    pytest.importorskip("torch")
    g = _make_separable_graph(n_objects=4, per_object=6)
    model, _ = train_edge_partition_gnn(g, n_epochs=30, log_every=0, seed=0)
    result = soft_partition(model, g, abstain_threshold=0.5)
    assert np.all(result["pred"][result["abstain_mask"]] < 0)
    assert np.all(result["pred"][~result["abstain_mask"]] >= 0)


def test_soft_partition_empty_graph_safe():
    pytest.importorskip("torch")
    g = HalfSynapseGraph(
        node_feat=np.zeros((3, 5), np.float32), node_pos=np.zeros((3, 3), np.float32),
        edge_src=np.zeros(0, np.int64), edge_dst=np.zeros(0, np.int64),
        edge_type=np.zeros(0, np.int64), edge_feat=np.zeros((0, 3), np.float32),
        labels=np.array([1, 1, 2], np.int64), seg_id=np.array([1, 1, 2], np.int64),
        side="pre",
    )
    model, _ = train_edge_partition_gnn(g, n_epochs=2, log_every=0)
    result = soft_partition(model, g)
    assert result["pred"].shape == (3,)
    assert result["membership_probs"].shape[0] == 3


# ---------------------------------------------------------------------------
# edge_merge_metrics — frankenmerge and abstain extensions
# ---------------------------------------------------------------------------

def test_edge_merge_metrics_frankenmerge_rate_nonzero():
    g = _make_frankenmerge_graph()
    m = edge_merge_metrics(g, g.labels)
    assert m["frankenmerge_rate"] > 0.0, "Frankenmerge cross-neuron type-0 edges must be detected"


def test_edge_merge_metrics_frankenmerge_split_recall_perfect_when_labels_used():
    g = _make_frankenmerge_graph()
    m = edge_merge_metrics(g, g.labels)
    # The ground-truth labels perfectly split the frankenmerge → recall = 1.0
    assert m["frankenmerge_split_recall"] == pytest.approx(1.0)


def test_edge_merge_metrics_frankenmerge_split_recall_zero_when_merged():
    g = _make_frankenmerge_graph()
    # Assign all nodes to a single cluster → frankenmerge never split
    all_one = np.zeros(g.n_nodes, dtype=np.int64)
    m = edge_merge_metrics(g, all_one)
    assert m["frankenmerge_split_recall"] == pytest.approx(0.0)


def test_edge_merge_metrics_abstain_rate():
    g = _make_separable_graph()
    pred = g.labels.copy()
    pred[:3] = np.array([-1, -2, -3], dtype=np.int64)
    m = edge_merge_metrics(g, pred)
    assert m["abstain_rate"] == pytest.approx(3 / g.n_nodes)


# ---------------------------------------------------------------------------
# Viability bars: integration test on synthetic data
# ---------------------------------------------------------------------------

def test_viability_bar1_edge_cc_beats_union_find():
    """Bar 1: edge_cc ARI ≥ union-find ARI AND merge_P ≥ union-find merge_P."""
    pytest.importorskip("torch")
    from treestitch.partition import (
        evaluate_partition,
        merge_metrics,
        partition_observations,
        partition_observations_cc,
        train_edge_partition,
        train_partition,
    )
    g = _make_separable_graph(n_objects=4, per_object=6)

    gnn_uf, _ = train_partition(g, n_epochs=60, seed=0, log_every=0)
    pred_uf   = partition_observations(gnn_uf, g, threshold=0.8)
    r_uf = evaluate_partition(pred_uf, g.labels)
    m_uf = merge_metrics(g, pred_uf)

    model_cc, _ = train_edge_partition(g, n_epochs=60, seed=0, log_every=0)
    pred_cc     = partition_observations_cc(model_cc, g)
    r_cc = evaluate_partition(pred_cc, g.labels)
    m_cc = merge_metrics(g, pred_cc)

    assert r_cc["ari"] >= r_uf["ari"] - 0.05, (
        f"Bar1 ARI: edge_cc={r_cc['ari']:.3f} union-find={r_uf['ari']:.3f}"
    )
    assert m_cc["merge_precision"] >= m_uf["merge_precision"] - 0.05, (
        f"Bar1 merge_P: edge_cc={m_cc['merge_precision']:.3f} uf={m_uf['merge_precision']:.3f}"
    )


def test_viability_bar2_merge_precision_high():
    """Bar 2: merge_P > 0.85 on a well-separated 4-object synthetic graph (60 epochs)."""
    pytest.importorskip("torch")
    from treestitch.partition import (
        merge_metrics, partition_observations_cc, train_edge_partition,
    )
    g = _make_separable_graph(n_objects=4, per_object=6)
    model, _ = train_edge_partition(g, n_epochs=60, seed=0, log_every=0)
    pred = partition_observations_cc(model, g)
    m = merge_metrics(g, pred)
    assert m["merge_precision"] > 0.85, f"Bar2 fail: merge_P={m['merge_precision']:.3f}"
