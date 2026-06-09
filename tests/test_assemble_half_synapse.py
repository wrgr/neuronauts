"""Tests for half-synapse graph construction, GNN, and partition evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.schemas import Fragment, Region


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_region(
    n_synapses: int = 12,
    n_neurons: int = 3,
    n_seg_ids: int = 4,
    with_seg_ids: bool = True,
    seed: int = 42,
) -> Region:
    """Build a small Region for testing."""
    rng = np.random.default_rng(seed)
    N = n_synapses
    pre_pt = rng.uniform(0, 100_000, (N, 3)).astype(np.float32)
    post_pt = rng.uniform(0, 100_000, (N, 3)).astype(np.float32)
    # Assign label-version root IDs (ground truth, 1-indexed)
    pre_root = rng.integers(1, n_neurons + 1, N, dtype=np.int64)
    post_root = rng.integers(1, n_neurons + 1, N, dtype=np.int64)
    syn_id = np.arange(N, dtype=np.int64)

    if with_seg_ids:
        pre_seg = rng.integers(1, n_seg_ids + 1, N, dtype=np.int64)
        post_seg = rng.integers(1, n_seg_ids + 1, N, dtype=np.int64)
    else:
        pre_seg = None
        post_seg = None

    return Region(
        region_id="test",
        bbox_nm=((0.0, 0.0, 0.0), (100_000.0, 100_000.0, 100_000.0)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=pre_pt,
        post_pt_nm=post_pt,
        pre_root_id=pre_root,
        post_root_id=post_root,
        synapse_id=syn_id,
        pre_seg_id=pre_seg,
        post_seg_id=post_seg,
    ).validate()


def _make_fragment(seg_id: int, dna_dim: int = 8, seed: int | None = None) -> Fragment:
    """Build a minimal Fragment with DNA filled."""
    rng = np.random.default_rng(seg_id if seed is None else seed)
    verts = rng.uniform(0, 100_000, (5, 3)).astype(np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    radii = np.ones(5, dtype=np.float32) * 300.0
    endpoints = verts[[0, 4]]
    dna = rng.normal(0, 1, dna_dim).astype(np.float32)
    dna /= np.linalg.norm(dna)
    return Fragment(
        fragment_id=seg_id,
        region_id="test",
        base_root_id=seg_id,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=endpoints,
        radius_nm=radii,
        synapse_indices=np.array([], dtype=np.int64),
        dna=dna,
    ).validate()


# ---------------------------------------------------------------------------
# HalfSynapseGraph construction tests
# ---------------------------------------------------------------------------

class TestBuildHalfSynapseGraph:
    def test_pre_node_count(self):
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        region = _make_region(n_synapses=10)
        graph = build_half_synapse_graph(region, [], side="pre")
        assert graph.n_nodes == 10

    def test_post_node_count(self):
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        region = _make_region(n_synapses=8)
        graph = build_half_synapse_graph(region, [], side="post")
        assert graph.n_nodes == 8

    def test_same_seg_edges_present(self):
        """3 synapses sharing the same pre_seg_id → 6 directed same-seg edges."""
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        N = 5
        pre_pt = np.random.default_rng(0).uniform(0, 1e5, (N, 3)).astype(np.float32)
        post_pt = pre_pt.copy()
        pre_seg = np.array([7, 7, 7, 2, 2], dtype=np.int64)  # seg 7 has 3, seg 2 has 2
        region = Region(
            region_id="t", bbox_nm=((0, 0, 0), (1e5, 1e5, 1e5)),
            voxel_size_nm=(8, 8, 40), seg_version=117, label_version=1412,
            pre_pt_nm=pre_pt, post_pt_nm=post_pt,
            pre_root_id=np.ones(N, dtype=np.int64),
            post_root_id=np.ones(N, dtype=np.int64),
            synapse_id=np.arange(N, dtype=np.int64),
            pre_seg_id=pre_seg, post_seg_id=np.zeros(N, dtype=np.int64),
        ).validate()

        graph = build_half_synapse_graph(region, [], side="pre", k_spatial=0)
        ss_mask = graph.edge_type == 0
        # seg 7: 3 nodes → 3 pairs × 2 directions = 6
        # seg 2: 2 nodes → 1 pair × 2 = 2
        assert ss_mask.sum() == 8

    def test_spatial_edges_present(self):
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        region = _make_region(n_synapses=10, with_seg_ids=False)
        graph = build_half_synapse_graph(region, [], side="pre", k_spatial=3)
        sp_mask = graph.edge_type == 1
        assert sp_mask.sum() > 0

    def test_edge_types_are_valid(self):
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        region = _make_region(n_synapses=15)
        graph = build_half_synapse_graph(region, [], side="pre")
        assert set(np.unique(graph.edge_type)).issubset({0, 1})

    def test_dna_in_node_feat(self):
        """Nodes whose seg_id has a fragment should have non-zero DNA part."""
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        N = 4
        rng = np.random.default_rng(1)
        pre_pt = rng.uniform(0, 1e5, (N, 3)).astype(np.float32)
        seg_ids = np.array([10, 10, 99, 99], dtype=np.int64)
        region = Region(
            region_id="t", bbox_nm=((0, 0, 0), (1e5, 1e5, 1e5)),
            voxel_size_nm=(8, 8, 40), seg_version=117, label_version=1412,
            pre_pt_nm=pre_pt, post_pt_nm=pre_pt.copy(),
            pre_root_id=np.array([1, 1, 2, 2], dtype=np.int64),
            post_root_id=np.ones(N, dtype=np.int64),
            synapse_id=np.arange(N, dtype=np.int64),
            pre_seg_id=seg_ids, post_seg_id=seg_ids,
        ).validate()

        dna_dim = 8
        frag10 = _make_fragment(10, dna_dim=dna_dim)
        graph = build_half_synapse_graph(region, [frag10], side="pre", k_spatial=1)
        # Nodes 0, 1 have seg_id=10 → should have DNA from frag10
        assert graph.node_dim == 3 + dna_dim
        np.testing.assert_allclose(
            graph.node_feat[0, 3:], frag10.dna, atol=1e-5
        )
        np.testing.assert_allclose(
            graph.node_feat[1, 3:], frag10.dna, atol=1e-5
        )
        # Nodes 2, 3 have seg_id=99 → no fragment → zeros in DNA part
        assert np.all(graph.node_feat[2, 3:] == 0)

    def test_zero_dna_when_no_fragment(self):
        """Synapse with unknown seg_id gets zero DNA in node features."""
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        region = _make_region(n_synapses=6)
        frag = _make_fragment(999, dna_dim=8)  # seg_id not in region
        graph = build_half_synapse_graph(region, [frag], side="pre", k_spatial=1)
        # All nodes should have zero DNA (no matching seg_id)
        assert np.all(graph.node_feat[:, 3:] == 0)

    def test_labels_are_label_version_ids(self):
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        region = _make_region(n_synapses=8)
        graph = build_half_synapse_graph(region, [], side="pre")
        np.testing.assert_array_equal(graph.labels, region.pre_root_id)

    def test_post_labels_are_post_root_ids(self):
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        region = _make_region(n_synapses=8)
        graph = build_half_synapse_graph(region, [], side="post")
        np.testing.assert_array_equal(graph.labels, region.post_root_id)

    def test_build_no_seg_ids_only_spatial(self):
        """When pre_seg_id is None there should be no same-seg edges."""
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        region = _make_region(n_synapses=10, with_seg_ids=False)
        graph = build_half_synapse_graph(region, [], side="pre", k_spatial=4)
        ss_mask = graph.edge_type == 0
        assert ss_mask.sum() == 0
        sp_mask = graph.edge_type == 1
        assert sp_mask.sum() > 0

    def test_edge_feat_shape(self):
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        region = _make_region(n_synapses=10)
        graph = build_half_synapse_graph(region, [], side="pre")
        assert graph.edge_feat.shape == (graph.n_edges, 3)

    def test_edge_cos_sim_in_range(self):
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

        region = _make_region(n_synapses=10)
        frags = [_make_fragment(i, dna_dim=8) for i in range(1, 5)]
        graph = build_half_synapse_graph(region, frags, side="pre")
        cos_col = graph.edge_feat[:, 2]
        assert np.all(cos_col >= -1.0 - 1e-5)
        assert np.all(cos_col <= 1.0 + 1e-5)


# ---------------------------------------------------------------------------
# HalfSynapseGNN tests
# ---------------------------------------------------------------------------

class TestHalfSynapseGNN:
    def test_forward_shape(self):
        pytest.importorskip("torch")
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph
        from neuronauts.assemble.partition_gnn import HalfSynapseGNN

        region = _make_region(n_synapses=12)
        graph = build_half_synapse_graph(region, [], side="pre", k_spatial=3)
        gnn = HalfSynapseGNN(input_dim=graph.node_dim, output_dim=16)
        import torch
        nf = torch.from_numpy(graph.node_feat)
        es = torch.from_numpy(graph.edge_src).long()
        ed = torch.from_numpy(graph.edge_dst).long()
        et = torch.from_numpy(graph.edge_type).long()
        out = gnn(nf, es, ed, et)
        assert out.shape == (12, 16)

    def test_forward_finite(self):
        pytest.importorskip("torch")
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph
        from neuronauts.assemble.partition_gnn import HalfSynapseGNN
        import torch

        region = _make_region(n_synapses=10)
        graph = build_half_synapse_graph(region, [], side="pre")
        gnn = HalfSynapseGNN(input_dim=graph.node_dim)
        nf = torch.from_numpy(graph.node_feat)
        es = torch.from_numpy(graph.edge_src).long()
        ed = torch.from_numpy(graph.edge_dst).long()
        et = torch.from_numpy(graph.edge_type).long()
        out = gnn(nf, es, ed, et)
        assert torch.isfinite(out).all()

    def test_output_l2_normalised(self):
        pytest.importorskip("torch")
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph
        from neuronauts.assemble.partition_gnn import HalfSynapseGNN
        import torch

        region = _make_region(n_synapses=8)
        graph = build_half_synapse_graph(region, [], side="pre")
        gnn = HalfSynapseGNN(input_dim=graph.node_dim, output_dim=16)
        with torch.no_grad():
            nf = torch.from_numpy(graph.node_feat)
            es = torch.from_numpy(graph.edge_src).long()
            ed = torch.from_numpy(graph.edge_dst).long()
            et = torch.from_numpy(graph.edge_type).long()
            out = gnn(nf, es, ed, et)
        norms = out.norm(dim=-1)
        np.testing.assert_allclose(norms.numpy(), np.ones(8), atol=1e-5)

    def test_train_history_keys(self):
        pytest.importorskip("torch")
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph
        from neuronauts.assemble.partition_gnn import train_partition_gnn

        region = _make_region(n_synapses=20, n_neurons=4)
        graph = build_half_synapse_graph(region, [], side="pre")
        _, history = train_partition_gnn(graph, n_epochs=3, log_every=0)
        assert set(history.keys()) == {"loss", "pos_sim", "neg_sim"}
        assert len(history["loss"]) == 3

    def test_train_history_finite(self):
        pytest.importorskip("torch")
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph
        from neuronauts.assemble.partition_gnn import train_partition_gnn

        region = _make_region(n_synapses=20, n_neurons=4)
        graph = build_half_synapse_graph(region, [], side="pre")
        _, history = train_partition_gnn(graph, n_epochs=5, log_every=0)
        for key in history:
            assert all(np.isfinite(v) for v in history[key])


# ---------------------------------------------------------------------------
# Partition tests
# ---------------------------------------------------------------------------

class TestPartitionHalfSynapses:
    def test_returns_int_array(self):
        pytest.importorskip("torch")
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph
        from neuronauts.assemble.partition_gnn import (
            HalfSynapseGNN,
            partition_half_synapses,
        )

        region = _make_region(n_synapses=10)
        graph = build_half_synapse_graph(region, [], side="pre")
        gnn = HalfSynapseGNN(input_dim=graph.node_dim)
        labels = partition_half_synapses(gnn, graph)
        assert labels.dtype == np.int64
        assert labels.shape == (10,)

    def test_labels_consecutive_from_zero(self):
        pytest.importorskip("torch")
        from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph
        from neuronauts.assemble.partition_gnn import (
            HalfSynapseGNN,
            partition_half_synapses,
        )

        region = _make_region(n_synapses=8)
        graph = build_half_synapse_graph(region, [], side="pre")
        gnn = HalfSynapseGNN(input_dim=graph.node_dim)
        labels = partition_half_synapses(gnn, graph)
        unique = np.unique(labels)
        assert unique[0] == 0
        assert len(unique) == unique[-1] + 1


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluatePartitionARI:
    def test_perfect_partition_ari(self):
        from neuronauts.assemble.partition_gnn import evaluate_partition_ari

        true = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
        pred = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        result = evaluate_partition_ari(pred, true)
        assert abs(result["ari"] - 1.0) < 1e-6

    def test_random_partition_ari_near_zero(self):
        from neuronauts.assemble.partition_gnn import evaluate_partition_ari

        rng = np.random.default_rng(0)
        true = np.repeat(np.arange(10), 20).astype(np.int64)
        pred = rng.integers(0, 30, len(true)).astype(np.int64)
        result = evaluate_partition_ari(pred, true)
        assert abs(result["ari"]) < 0.3

    def test_ignore_label_zero(self):
        from neuronauts.assemble.partition_gnn import evaluate_partition_ari

        true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        pred = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        result = evaluate_partition_ari(pred, true)
        # Only 4 labelled nodes (label != 0), perfect partition among them
        assert abs(result["ari"] - 1.0) < 1e-6
        assert result["n_nodes"] == 4

    def test_result_dict_keys(self):
        from neuronauts.assemble.partition_gnn import evaluate_partition_ari

        true = np.array([1, 1, 2, 2], dtype=np.int64)
        pred = np.array([0, 0, 1, 1], dtype=np.int64)
        result = evaluate_partition_ari(pred, true)
        expected_keys = {
            "ari", "homogeneity", "completeness", "v_measure",
            "n_clusters_pred", "n_clusters_true", "n_nodes",
        }
        assert set(result.keys()) == expected_keys

    def test_n_clusters_counts(self):
        from neuronauts.assemble.partition_gnn import evaluate_partition_ari

        true = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
        pred = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        result = evaluate_partition_ari(pred, true)
        assert result["n_clusters_true"] == 3
        assert result["n_clusters_pred"] == 3
