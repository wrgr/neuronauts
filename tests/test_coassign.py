"""Tests for the synapse co-assignment pipeline."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_graph(n=30, n_neurons=3, dna_dim=0, seed=0):
    """Make a small SynapseGraph for testing."""
    from neuronauts.coassign import build_synapse_graph

    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 50_000, (n, 3)).astype(np.float32)
    labels = np.repeat(np.arange(1, n_neurons + 1), n // n_neurons).astype(np.int64)
    labels = labels[:n]
    seg_ids = labels.copy()  # one segment per neuron for simplicity

    seg_dna = {}
    if dna_dim > 0:
        for sid in np.unique(seg_ids):
            seg_dna[int(sid)] = rng.normal(0, 1, dna_dim).astype(np.float32)

    return build_synapse_graph(pos, seg_ids, labels, seg_dna, k_spatial=4)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

class TestBuildSynapseGraph:
    def test_node_count(self):
        g = _small_graph(n=20)
        assert g.n_nodes == 20

    def test_node_dim_no_dna(self):
        g = _small_graph(n=10, dna_dim=0)
        assert g.node_dim == 3

    def test_node_dim_with_dna(self):
        g = _small_graph(n=10, dna_dim=16)
        assert g.node_dim == 19

    def test_same_seg_flag(self):
        g = _small_graph(n=20, n_neurons=4)
        # same_seg must be exactly 0 or 1
        assert set(g.same_seg.tolist()).issubset({0.0, 1.0})

    def test_same_seg_edges_exist(self):
        g = _small_graph(n=20, n_neurons=2)
        assert (g.same_seg == 1).sum() > 0

    def test_spatial_edges_exist(self):
        g = _small_graph(n=20)
        assert (g.same_seg == 0).sum() > 0

    def test_edge_indices_in_bounds(self):
        g = _small_graph(n=15)
        assert g.edge_src.max() < g.n_nodes
        assert g.edge_dst.max() < g.n_nodes

    def test_no_dna_gives_zero_embeddings(self):
        g = _small_graph(n=10, dna_dim=0)
        assert g.node_dna.shape == (10, 0)

    def test_dna_filled_for_known_segments(self):
        g = _small_graph(n=12, dna_dim=8)
        # All nodes have a seg_id in seg_dna so none should be all-zero
        assert not np.all(g.node_dna == 0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TestSynapseCoassigner:
    def test_forward_shape(self):
        torch = pytest.importorskip("torch")
        from neuronauts.coassign import SynapseCoassigner

        g = _small_graph(n=20, dna_dim=8)
        model = SynapseCoassigner(node_dim=g.node_dim)
        node_feat  = torch.from_numpy(np.concatenate([g.node_pos, g.node_dna], 1)).float()
        edge_src_t = torch.from_numpy(g.edge_src).long()
        edge_dst_t = torch.from_numpy(g.edge_dst).long()
        same_seg_t = torch.from_numpy(g.same_seg).float()
        out = model(node_feat, edge_src_t, edge_dst_t, same_seg_t)
        assert out.shape == (g.n_edges,)

    def test_edge_probs_in_range(self):
        torch = pytest.importorskip("torch")
        from neuronauts.coassign import SynapseCoassigner

        g = _small_graph(n=20)
        model = SynapseCoassigner(node_dim=g.node_dim)
        node_feat  = torch.from_numpy(np.concatenate([g.node_pos, g.node_dna], 1)).float()
        edge_src_t = torch.from_numpy(g.edge_src).long()
        edge_dst_t = torch.from_numpy(g.edge_dst).long()
        same_seg_t = torch.from_numpy(g.same_seg).float()
        probs = model.edge_probs(node_feat, edge_src_t, edge_dst_t, same_seg_t).numpy()
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_output_finite(self):
        torch = pytest.importorskip("torch")
        from neuronauts.coassign import SynapseCoassigner

        g = _small_graph(n=15, dna_dim=16)
        model = SynapseCoassigner(node_dim=g.node_dim)
        node_feat  = torch.from_numpy(np.concatenate([g.node_pos, g.node_dna], 1)).float()
        edge_src_t = torch.from_numpy(g.edge_src).long()
        edge_dst_t = torch.from_numpy(g.edge_dst).long()
        same_seg_t = torch.from_numpy(g.same_seg).float()
        out = model(node_feat, edge_src_t, edge_dst_t, same_seg_t)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTrain:
    def test_history_keys(self):
        pytest.importorskip("torch")
        from neuronauts.coassign import SynapseCoassigner, train

        g = _small_graph(n=30, n_neurons=3)
        model = SynapseCoassigner(node_dim=g.node_dim)
        h = train(model, [g], n_epochs=3, log_every=0)
        assert set(h.keys()) == {"loss", "precision", "recall"}
        assert len(h["loss"]) == 3

    def test_loss_finite(self):
        pytest.importorskip("torch")
        from neuronauts.coassign import SynapseCoassigner, train

        g = _small_graph(n=30, n_neurons=4, dna_dim=8)
        model = SynapseCoassigner(node_dim=g.node_dim)
        h = train(model, [g], n_epochs=5, log_every=0)
        assert all(np.isfinite(v) for v in h["loss"])

    def test_loss_decreases(self):
        pytest.importorskip("torch")
        from neuronauts.coassign import SynapseCoassigner, train

        # Place neurons in distinct spatial clusters so learning is easy
        rng = np.random.default_rng(42)
        n_neurons, n_per = 4, 15
        N = n_neurons * n_per
        centers = rng.uniform(0, 200_000, (n_neurons, 3)).astype(np.float32)
        pos = np.repeat(centers, n_per, axis=0) + rng.normal(0, 500, (N, 3)).astype(np.float32)
        labels = np.repeat(np.arange(1, n_neurons + 1), n_per).astype(np.int64)
        seg_ids = labels.copy()
        from neuronauts.coassign import build_synapse_graph
        g = build_synapse_graph(pos, seg_ids, labels, {}, k_spatial=4)
        model = SynapseCoassigner(node_dim=g.node_dim)
        h = train(model, [g], n_epochs=30, log_every=0)
        assert h["loss"][-1] < h["loss"][0]


# ---------------------------------------------------------------------------
# Clustering and metrics
# ---------------------------------------------------------------------------

class TestGreedyCluster:
    def test_output_shape(self):
        from neuronauts.coassign import greedy_cluster

        n = 20
        src = np.array([0, 1, 2], dtype=np.int64)
        dst = np.array([1, 2, 3], dtype=np.int64)
        probs = np.array([0.9, 0.9, 0.1], dtype=np.float32)
        labels = greedy_cluster(n, src, dst, probs)
        assert labels.shape == (n,)

    def test_high_prob_edges_merge(self):
        from neuronauts.coassign import greedy_cluster

        # Nodes 0-2 fully connected with p=0.95, isolated from node 3
        src = np.array([0, 1, 0], dtype=np.int64)
        dst = np.array([1, 2, 2], dtype=np.int64)
        probs = np.full(3, 0.95, dtype=np.float32)
        labels = greedy_cluster(4, src, dst, probs, threshold=0.5)
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] != labels[0]

    def test_low_prob_edges_separate(self):
        from neuronauts.coassign import greedy_cluster

        src = np.array([0], dtype=np.int64)
        dst = np.array([1], dtype=np.int64)
        probs = np.array([0.1], dtype=np.float32)
        labels = greedy_cluster(2, src, dst, probs, threshold=0.5)
        assert labels[0] != labels[1]


class TestMaterializations:
    def test_returns_k_results(self):
        from neuronauts.coassign import materializations

        g = _small_graph(n=20, n_neurons=3)
        probs = np.full(g.n_edges, 0.5, dtype=np.float32)
        mats = materializations(g.n_nodes, g.edge_src, g.edge_dst, probs, K=3)
        assert len(mats) <= 3
        assert len(mats) >= 1

    def test_sorted_by_score(self):
        from neuronauts.coassign import materializations

        g = _small_graph(n=20)
        probs = np.random.default_rng(0).uniform(0, 1, g.n_edges).astype(np.float32)
        mats = materializations(g.n_nodes, g.edge_src, g.edge_dst, probs, K=5)
        scores = [s for _, s in mats]
        assert scores == sorted(scores, reverse=True)


class TestPairwisePR:
    def test_perfect_partition(self):
        from neuronauts.coassign import pairwise_precision_recall

        true = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
        pred = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        r = pairwise_precision_recall(pred, true)
        assert abs(r["precision"] - 1.0) < 1e-6
        assert abs(r["recall"] - 1.0) < 1e-6
        assert abs(r["f1"] - 1.0) < 1e-6

    def test_single_cluster_perfect_recall(self):
        from neuronauts.coassign import pairwise_precision_recall

        true = np.array([1, 1, 2, 2], dtype=np.int64)
        pred = np.array([0, 0, 0, 0], dtype=np.int64)
        r = pairwise_precision_recall(pred, true)
        assert r["recall"] == 1.0
        assert r["precision"] < 1.0

    def test_ignore_label_zero(self):
        from neuronauts.coassign import pairwise_precision_recall

        true = np.array([0, 1, 1, 2, 2], dtype=np.int64)
        pred = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        r = pairwise_precision_recall(pred, true)
        # Node 0 is ignored; prediction on nodes 1-4 is perfect
        assert abs(r["f1"] - 1.0) < 1e-6


class TestCoverageAtK:
    def test_perfect_model_covered(self):
        from neuronauts.coassign import coverage_at_k

        true = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
        pred = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)   # perfect partition
        mats = [(pred, 0.0)]
        assert coverage_at_k(mats, true) is True

    def test_empty_mats_not_covered(self):
        from neuronauts.coassign import coverage_at_k

        true = np.array([1, 1, 2, 2], dtype=np.int64)
        assert coverage_at_k([], true) is False
