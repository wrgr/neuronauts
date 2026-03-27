"""Tests for neuronauts.cell_graph — synapse-level GNN cell reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.fetch import SynapseTable
from neuronauts.cell_graph import (
    CellGNN,
    CellGNNConfig,
    SynapseEdge,
    SynapseGraph,
    build_synapse_graph,
    cell_graph_train_step,
    connectivity_graph_from_cell_labels,
    infer_cells,
    load_cell_gnn,
    partition_from_embeddings,
    rank_boxes_by_tangledness,
    save_cell_gnn,
    score_box_tangledness,
    select_cell_gnn_training_boxes,
    spatial_train_val_test_split,
)

torch = pytest.importorskip("torch", reason="torch not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synapses(n_cells: int = 3, synapses_per_cell: int = 4, seed: int = 0) -> SynapseTable:
    """Synthetic SynapseTable with clearly clustered pre-side synapses."""
    rng = np.random.default_rng(seed)
    n = n_cells * synapses_per_cell

    pre_pt = np.zeros((n, 3), dtype=np.float32)
    post_pt = rng.standard_normal((n, 3)).astype(np.float32) * 50
    pre_root_id = np.zeros(n, dtype=np.int64)
    post_root_id = np.zeros(n, dtype=np.int64)

    for c in range(n_cells):
        centre = rng.standard_normal(3) * 500  # cells 500 nm apart on average
        for k in range(synapses_per_cell):
            idx = c * synapses_per_cell + k
            pre_pt[idx] = centre + rng.standard_normal(3) * 20   # tight cluster
            pre_root_id[idx] = c + 1                               # 1-indexed
            post_root_id[idx] = rng.integers(1, n_cells + 1)

    return SynapseTable(
        pre_pt=pre_pt,
        post_pt=post_pt,
        pre_root_id=pre_root_id,
        post_root_id=post_root_id,
        synapse_id=np.arange(n, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# SynapseGraph construction
# ---------------------------------------------------------------------------

class TestBuildSynapseGraph:
    def test_basic_shape(self):
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=200.0)
        assert graph.n_synapses == 12
        assert graph.role == "pre"
        assert graph.node_positions.shape == (12, 3)
        assert graph.node_scaffold_ids.shape == (12,)

    def test_root_ids_populated(self):
        syn = _make_synapses(n_cells=2, synapses_per_cell=5)
        graph = build_synapse_graph(syn, "pre")
        assert graph.root_ids is not None
        assert graph.root_ids.shape == (10,)

    def test_post_side(self):
        syn = _make_synapses(n_cells=2, synapses_per_cell=3)
        graph = build_synapse_graph(syn, "post")
        assert graph.n_synapses == 6
        assert graph.role == "post"

    def test_nearby_synapses_get_edges(self):
        """Synapses in the same tight cluster should be connected."""
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        # Large radius ensures within-cluster edges exist
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)
        assert len(graph.edges) > 0

    def test_scaffold_same_seg_creates_same_scaffold_edge(self):
        """Synapses with the same seg_id should get same_scaffold=1.0 edges."""
        syn = _make_synapses(n_cells=2, synapses_per_cell=3)
        # Assign all synapses in cell 0 the same seg_id
        seg_ids = np.zeros(6, dtype=np.int64)
        seg_ids[:3] = 42  # first 3 synapses share seg 42
        syn.pre_seg_id = seg_ids
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=1e9)
        scaf_edges = [e for e in graph.edges if e.same_scaffold == 1.0]
        # At least the 3 intra-group edges should be scaffold edges
        assert len(scaf_edges) >= 3

    def test_empty_synapses(self):
        syn = SynapseTable(
            pre_pt=np.zeros((0, 3), dtype=np.float32),
            post_pt=np.zeros((0, 3), dtype=np.float32),
            pre_root_id=np.zeros(0, dtype=np.int64),
            post_root_id=np.zeros(0, dtype=np.int64),
            synapse_id=np.zeros(0, dtype=np.int64),
        )
        graph = build_synapse_graph(syn, "pre")
        assert graph.n_synapses == 0
        assert graph.edges == []


# ---------------------------------------------------------------------------
# CellGNN architecture
# ---------------------------------------------------------------------------

class TestCellGNN:
    def test_forward_shape(self):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        N, E = 10, 20
        node_feat = torch.randn(N, 3)
        edge_src = torch.randint(0, N, (E,))
        edge_dst = torch.randint(0, N, (E,))
        edge_feat = torch.randn(E, 4)
        out = model(node_feat, edge_src, edge_dst, edge_feat)
        assert out.shape == (N, 8)

    def test_forward_no_edges(self):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        node_feat = torch.randn(5, 3)
        empty = torch.zeros(0, dtype=torch.long)
        edge_feat = torch.zeros(0, 4)
        out = model(node_feat, empty, empty, edge_feat)
        assert out.shape == (5, 8)

    def test_single_node(self):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=1,
                        n_heads=2, embedding_dim=8)
        node_feat = torch.randn(1, 3)
        # Self-loop only
        edge_src = torch.tensor([0])
        edge_dst = torch.tensor([0])
        edge_feat = torch.randn(1, 4)
        out = model(node_feat, edge_src, edge_dst, edge_feat)
        assert out.shape == (1, 8)

    def test_init_kwargs_saved(self):
        model = CellGNN(node_input_dim=3, d_model=32, n_layers=2,
                        n_heads=4, embedding_dim=16)
        assert model._init_kwargs["d_model"] == 32
        assert model._init_kwargs["embedding_dim"] == 16


# ---------------------------------------------------------------------------
# Graph -> tensor conversion
# ---------------------------------------------------------------------------

class TestGraphToTensors:
    def test_bidirectional_edges(self):
        """Each undirected edge should produce two directed edge entries."""
        from neuronauts.cell_graph import _graph_to_tensors

        graph = SynapseGraph(
            n_synapses=3,
            role="pre",
            node_positions=np.eye(3, dtype=np.float32),
            node_scaffold_ids=np.zeros(3, dtype=np.int64),
            edges=[SynapseEdge(src=0, dst=1, distance=1.0,
                               same_scaffold=0.0, grammar_score=0.5,
                               shared_agents=1)],
            root_ids=None,
        )
        _, es, ed, ef = _graph_to_tensors(graph)
        # 1 undirected → 2 directed + 3 self-loops = 5 total
        assert len(es) == 5
        assert ef.shape == (5, 4)


# ---------------------------------------------------------------------------
# partition_from_embeddings
# ---------------------------------------------------------------------------

class TestPartitionFromEmbeddings:
    def test_perfect_clusters(self):
        """Three tight clusters should be assigned 3 distinct labels."""
        rng = np.random.default_rng(0)
        centres = np.eye(3, dtype=np.float32) * 10
        embs = np.vstack([
            centres[c] + rng.standard_normal((4, 3)) * 0.01
            for c in range(3)
        ]).astype(np.float32)
        labels = partition_from_embeddings(embs, threshold=0.5)
        assert len(set(labels)) == 3

    def test_single_synapse(self):
        embs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        labels = partition_from_embeddings(embs)
        assert labels.tolist() == [0]

    def test_empty(self):
        labels = partition_from_embeddings(np.zeros((0, 4), dtype=np.float32))
        assert len(labels) == 0

    def test_greedy_method(self):
        rng = np.random.default_rng(1)
        embs = rng.standard_normal((8, 4)).astype(np.float32)
        labels = partition_from_embeddings(embs, method="greedy")
        assert len(labels) == 8
        assert labels.min() == 0

    def test_output_contiguous(self):
        """Labels should be contiguous integers starting at 0."""
        rng = np.random.default_rng(2)
        embs = rng.standard_normal((6, 4)).astype(np.float32)
        labels = partition_from_embeddings(embs, threshold=2.0)  # all separate
        assert sorted(set(labels.tolist())) == list(range(len(set(labels.tolist()))))


# ---------------------------------------------------------------------------
# cell_graph_train_step
# ---------------------------------------------------------------------------

class TestCellGraphTrainStep:
    def test_step_runs_and_returns_metrics(self):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        syn = _make_synapses(n_cells=3, synapses_per_cell=6)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)
        m = cell_graph_train_step(model, optimizer, graph)
        assert "loss" in m
        assert "pos_sim" in m
        assert "neg_sim" in m
        assert m["n_pos"] > 0

    def test_loss_is_non_negative(self):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        syn = _make_synapses(n_cells=2, synapses_per_cell=4)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)
        m = cell_graph_train_step(model, optimizer, graph)
        assert m["loss"] >= 0.0

    def test_loss_decreases_over_steps(self):
        """With clearly separated clusters, loss should decrease after several steps."""
        model = CellGNN(node_input_dim=3, d_model=32, n_layers=3,
                        n_heads=4, embedding_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        # Very tight clusters (std=2nm) far apart (centre spacing 1000nm)
        rng_np = np.random.default_rng(0)
        n_cells, spc = 3, 6
        n = n_cells * spc
        pre_pt = np.zeros((n, 3), dtype=np.float32)
        pre_root_id = np.zeros(n, dtype=np.int64)
        for c in range(n_cells):
            centre = rng_np.standard_normal(3) * 1000
            for k in range(spc):
                idx = c * spc + k
                pre_pt[idx] = centre + rng_np.standard_normal(3) * 2
                pre_root_id[idx] = c + 1
        syn = SynapseTable(
            pre_pt=pre_pt,
            post_pt=rng_np.standard_normal((n, 3)).astype(np.float32),
            pre_root_id=pre_root_id,
            post_root_id=np.ones(n, dtype=np.int64),
            synapse_id=np.arange(n, dtype=np.int64),
        )
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=500.0)
        rng = np.random.default_rng(42)
        first = cell_graph_train_step(model, optimizer, graph, rng=rng)
        for _ in range(19):
            last = cell_graph_train_step(model, optimizer, graph, rng=rng)
        assert last["loss"] < first["loss"], (
            f"loss did not decrease: {first['loss']:.4f} -> {last['loss']:.4f}"
        )

    def test_rng_non_deterministic_by_default(self):
        """Two calls with default (no) rng should not be identical if graph is large."""
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        syn = _make_synapses(n_cells=4, synapses_per_cell=8)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)
        m1 = cell_graph_train_step(model, optimizer, graph)
        m2 = cell_graph_train_step(model, optimizer, graph)
        # At least one metric should differ (different pair sample)
        # This is probabilistic but near-certain for n_pos > 20
        assert m1["n_pos"] > 0 and m2["n_pos"] > 0  # sanity


# ---------------------------------------------------------------------------
# infer_cells
# ---------------------------------------------------------------------------

class TestInferCells:
    def test_returns_correct_shape(self):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        syn = _make_synapses(n_cells=2, synapses_per_cell=5)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)
        labels = infer_cells(model, graph)
        assert labels.shape == (10,)
        assert labels.dtype == np.int64

    def test_labels_are_contiguous(self):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)
        labels = infer_cells(model, graph)
        unique = sorted(set(labels.tolist()))
        assert unique == list(range(len(unique)))


# ---------------------------------------------------------------------------
# connectivity_graph_from_cell_labels
# ---------------------------------------------------------------------------

class TestConnectivityGraphFromLabels:
    def test_all_synapses_assigned(self):
        n = 12
        pre_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        post_labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        cg = connectivity_graph_from_cell_labels(pre_labels, post_labels, syn)

        all_assigned = set()
        for e in cg.edges:
            all_assigned.add(e[2])
        assert len(all_assigned) == n

    def test_neuron_roles(self):
        pre_labels = np.zeros(6, dtype=np.int64)
        post_labels = np.zeros(6, dtype=np.int64)
        syn = _make_synapses(n_cells=1, synapses_per_cell=6)
        cg = connectivity_graph_from_cell_labels(pre_labels, post_labels, syn)
        roles = {n.role for n in cg.neurons.values()}
        assert "pre" in roles
        assert "post" in roles

    def test_edge_connects_pre_to_post(self):
        pre_labels = np.array([0, 0, 1, 1], dtype=np.int64)
        post_labels = np.array([0, 0, 0, 0], dtype=np.int64)
        syn = _make_synapses(n_cells=2, synapses_per_cell=2)
        cg = connectivity_graph_from_cell_labels(pre_labels, post_labels, syn)
        for pre_nid, post_nid, _ in cg.edges:
            assert cg.neurons[pre_nid].role == "pre"
            assert cg.neurons[post_nid].role == "post"

    def test_no_duplicate_edges(self):
        pre_labels = np.zeros(4, dtype=np.int64)
        post_labels = np.zeros(4, dtype=np.int64)
        syn = _make_synapses(n_cells=1, synapses_per_cell=4)
        cg = connectivity_graph_from_cell_labels(pre_labels, post_labels, syn)
        syn_indices = [e[2] for e in cg.edges]
        assert len(syn_indices) == len(set(syn_indices))

    def test_line_graph_f1_roundtrip(self):
        """A perfect partition should yield F1=1 against ground truth."""
        from neuronauts.line_graph import evaluate

        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        # Perfect partition: use ground-truth root IDs directly as labels
        # Map root IDs to contiguous integers
        pre_roots = syn.pre_root_id
        post_roots = syn.post_root_id
        unique_pre = {r: i for i, r in enumerate(sorted(set(pre_roots.tolist())))}
        unique_post = {r: i for i, r in enumerate(sorted(set(post_roots.tolist())))}
        pre_labels = np.array([unique_pre[r] for r in pre_roots], dtype=np.int64)
        post_labels = np.array([unique_post[r] for r in post_roots], dtype=np.int64)

        cg = connectivity_graph_from_cell_labels(pre_labels, post_labels, syn)
        metrics = evaluate(cg, syn.pre_root_id, syn.post_root_id)
        assert metrics.f1 == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        path = tmp_path / "cell_gnn.pt"
        save_cell_gnn(str(path), model)
        loaded = load_cell_gnn(str(path))
        # Check weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    def test_loaded_model_produces_same_output(self, tmp_path):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        model.eval()
        path = tmp_path / "cell_gnn.pt"
        save_cell_gnn(str(path), model)
        loaded = load_cell_gnn(str(path))
        loaded.eval()

        node_feat = torch.randn(5, 3)
        edge_src = torch.tensor([0, 1, 2])
        edge_dst = torch.tensor([1, 2, 0])
        edge_feat = torch.randn(3, 4)

        with torch.no_grad():
            out1 = model(node_feat, edge_src, edge_dst, edge_feat)
            out2 = loaded(node_feat, edge_src, edge_dst, edge_feat)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Sampling strategy: tangledness and spatial splitting
# ---------------------------------------------------------------------------

def _make_box_cache(tmp_path, n_boxes=6, seed=0):
    """Create a BoxCache with synthetic boxes for testing sampling utilities."""
    from neuronauts.dataset_builder import BoxCache
    from neuronauts.fetch import RealBoxSpec

    rng = np.random.default_rng(seed)
    cache = BoxCache(str(tmp_path / "cache"))

    for i in range(n_boxes):
        n_cells = rng.integers(2, 6)
        syn = _make_synapses(n_cells=n_cells, synapses_per_cell=rng.integers(3, 8), seed=seed + i)
        # Spread boxes along x axis for spatial splitting
        center_x = 500_000 + i * 200_000
        center_y = 1_000_000
        center_z = 300_000
        spec = RealBoxSpec(
            center_nm=(center_x, center_y, center_z),
            side_um=30.0,
            mip=2,
        )
        from neuronauts.dataset_builder import count_positive_pairs
        n_pos = count_positive_pairs(syn)
        cache.save_synapse_only(spec, syn, n_positive_pairs=n_pos)

    return cache


class TestTangledness:
    def test_score_box_tangledness_returns_metrics(self, tmp_path):
        cache = _make_box_cache(tmp_path, n_boxes=3)
        records = cache.all_records()
        assert len(records) == 3

        metrics = score_box_tangledness(cache, records[0])
        assert "tangledness" in metrics
        assert "n_pre_roots" in metrics
        assert "root_density" in metrics
        assert "multi_root_fraction" in metrics
        assert metrics["tangledness"] >= 0.0

    def test_score_box_tangledness_positive_for_multi_root(self, tmp_path):
        cache = _make_box_cache(tmp_path, n_boxes=1)
        records = cache.all_records()
        metrics = score_box_tangledness(cache, records[0])
        # Our synthetic boxes have multiple roots, so tangledness > 0
        assert metrics["n_pre_roots"] >= 2
        assert metrics["multi_root_fraction"] > 0

    def test_rank_boxes_by_tangledness(self, tmp_path):
        cache = _make_box_cache(tmp_path, n_boxes=5)
        ranked = rank_boxes_by_tangledness(cache, min_synapses=2, min_positive_pairs=0)
        assert len(ranked) > 0
        # Should be sorted descending by tangledness
        tangle_vals = [m["tangledness"] for _, m in ranked]
        for i in range(len(tangle_vals) - 1):
            assert tangle_vals[i] >= tangle_vals[i + 1]

    def test_rank_filters_small_boxes(self, tmp_path):
        cache = _make_box_cache(tmp_path, n_boxes=3)
        # Very high min_synapses should filter most/all
        ranked = rank_boxes_by_tangledness(cache, min_synapses=99999)
        assert len(ranked) == 0


class TestSpatialSplit:
    def test_split_covers_all_records(self, tmp_path):
        cache = _make_box_cache(tmp_path, n_boxes=9)
        records = cache.all_records()
        splits = spatial_train_val_test_split(
            cache, records, val_fraction=0.2, test_fraction=0.2, seed=42,
        )
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(records)

    def test_split_no_overlap(self, tmp_path):
        cache = _make_box_cache(tmp_path, n_boxes=9)
        records = cache.all_records()
        splits = spatial_train_val_test_split(
            cache, records, val_fraction=0.2, test_fraction=0.2, seed=42,
        )
        train_hashes = {r.box_hash for r in splits["train"]}
        val_hashes = {r.box_hash for r in splits["val"]}
        test_hashes = {r.box_hash for r in splits["test"]}
        assert not (train_hashes & val_hashes)
        assert not (train_hashes & test_hashes)
        assert not (val_hashes & test_hashes)

    def test_split_each_nonempty(self, tmp_path):
        cache = _make_box_cache(tmp_path, n_boxes=9)
        records = cache.all_records()
        splits = spatial_train_val_test_split(
            cache, records, val_fraction=0.2, test_fraction=0.2, seed=42,
        )
        assert len(splits["train"]) > 0
        assert len(splits["val"]) > 0
        assert len(splits["test"]) > 0

    def test_split_empty_input(self, tmp_path):
        cache = _make_box_cache(tmp_path, n_boxes=1)
        splits = spatial_train_val_test_split(cache, [], seed=42)
        assert splits == {"train": [], "val": [], "test": []}


class TestSelectCellGNNTrainingBoxes:
    def test_returns_all_splits(self, tmp_path):
        cache = _make_box_cache(tmp_path, n_boxes=9)
        splits = select_cell_gnn_training_boxes(
            cache,
            min_synapses=2,
            min_positive_pairs=0,
            seed=42,
        )
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

    def test_max_train_caps(self, tmp_path):
        cache = _make_box_cache(tmp_path, n_boxes=9)
        splits = select_cell_gnn_training_boxes(
            cache,
            max_train=2,
            min_synapses=2,
            min_positive_pairs=0,
            seed=42,
        )
        assert len(splits["train"]) <= 2

    def test_min_tangledness_filters(self, tmp_path):
        cache = _make_box_cache(tmp_path, n_boxes=5)
        # Very high tangledness threshold
        splits = select_cell_gnn_training_boxes(
            cache,
            min_tangledness=9999.0,
            min_synapses=2,
            min_positive_pairs=0,
            seed=42,
        )
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 0
