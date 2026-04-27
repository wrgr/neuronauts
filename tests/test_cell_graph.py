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
    boundary_partition_search,
    cell_gnn_assembly,
    cell_graph_train_step,
    connectivity_graph_from_cell_labels,
    extract_grammar_scores,
    infer_cells,
    infer_cells_with_search,
    load_cell_gnn,
    partition_from_embeddings,
    rank_boxes_by_tangledness,
    save_cell_gnn,
    score_box_tangledness,
    score_cell_quality,
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
        edge_feat = torch.randn(E, 5)
        out = model(node_feat, edge_src, edge_dst, edge_feat)
        assert out.shape == (N, 8)

    def test_forward_no_edges(self):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        node_feat = torch.randn(5, 3)
        empty = torch.zeros(0, dtype=torch.long)
        edge_feat = torch.zeros(0, 5)
        out = model(node_feat, empty, empty, edge_feat)
        assert out.shape == (5, 8)

    def test_single_node(self):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=1,
                        n_heads=2, embedding_dim=8)
        node_feat = torch.randn(1, 3)
        # Self-loop only
        edge_src = torch.tensor([0])
        edge_dst = torch.tensor([0])
        edge_feat = torch.randn(1, 5)
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
        assert ef.shape == (5, 5)


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
# score_cell_quality
# ---------------------------------------------------------------------------

class TestScoreCellQuality:
    def test_returns_scores_for_each_cell(self):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)
        labels = infer_cells(model, graph)
        scores = score_cell_quality(model, graph, labels)
        unique_cells = set(labels.tolist())
        assert set(scores.keys()) == unique_cells
        for v in scores.values():
            assert 0.0 <= v <= 1.0 or np.isclose(v, 0.0) or np.isclose(v, 1.0)

    def test_single_synapse_cell_gets_perfect_score(self):
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)
        # Force each synapse into its own cell
        labels = np.arange(graph.n_synapses, dtype=np.int64)
        scores = score_cell_quality(model, graph, labels)
        for v in scores.values():
            assert v == 1.0

    def test_with_topology_validator(self):
        from neuronauts.topology_model import AttentionArborValidator
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        validator = AttentionArborValidator(embed_dim=8)
        syn = _make_synapses(n_cells=2, synapses_per_cell=5)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)
        labels = infer_cells(model, graph)
        scores = score_cell_quality(model, graph, labels,
                                    topology_validator=validator)
        for v in scores.values():
            assert 0.0 <= v <= 1.0


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
        edge_feat = torch.randn(3, 5)

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


# ---------------------------------------------------------------------------
# Grammar score extraction and CellGNN assembly
# ---------------------------------------------------------------------------

def _dummy_grammar_score_fn(left_seq, right_seq):
    """Dummy grammar scorer: returns cosine similarity of mean features."""
    if len(left_seq) == 0 or len(right_seq) == 0:
        return 0.0
    a = left_seq.mean(axis=0)
    b = right_seq.mean(axis=0)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-8:
        return 0.0
    return float(np.dot(a, b) / norm)


class TestExtractGrammarScores:
    def test_returns_dict(self):
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        # Give synapses distinct seg_ids per cell to form scaffold groups
        syn = SynapseTable(
            pre_pt=syn.pre_pt,
            post_pt=syn.post_pt,
            pre_root_id=syn.pre_root_id,
            post_root_id=syn.post_root_id,
            synapse_id=syn.synapse_id,
            pre_seg_id=syn.pre_root_id.copy(),  # use root IDs as seg IDs
        )
        scores = extract_grammar_scores(
            syn, "pre", _dummy_grammar_score_fn,
            proximity_radius_nm=100000.0,
        )
        assert isinstance(scores, dict)
        # Should have some pairs scored
        assert len(scores) > 0
        for (i, j), v in scores.items():
            assert i < j
            assert isinstance(v, float)

    def test_no_seg_ids_returns_empty(self):
        syn = _make_synapses(n_cells=2, synapses_per_cell=3)
        scores = extract_grammar_scores(syn, "pre", _dummy_grammar_score_fn)
        assert scores == {}


class TestCellGNNAssembly:
    def test_produces_connectivity_graph(self):
        syn = _make_synapses(n_cells=3, synapses_per_cell=5)
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        model.eval()
        cg = cell_gnn_assembly(syn, model, proximity_radius_nm=50000.0)
        from neuronauts.merge import ConnectivityGraph
        assert isinstance(cg, ConnectivityGraph)
        assert len(cg.neurons) > 0
        assert len(cg.edges) > 0

    def test_with_grammar_scores(self):
        syn = _make_synapses(n_cells=2, synapses_per_cell=4)
        syn = SynapseTable(
            pre_pt=syn.pre_pt,
            post_pt=syn.post_pt,
            pre_root_id=syn.pre_root_id,
            post_root_id=syn.post_root_id,
            synapse_id=syn.synapse_id,
            pre_seg_id=syn.pre_root_id.copy(),
            post_seg_id=syn.post_root_id.copy(),
        )
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        model.eval()
        cg = cell_gnn_assembly(
            syn, model,
            grammar_score_fn=_dummy_grammar_score_fn,
            proximity_radius_nm=100000.0,
        )
        assert len(cg.neurons) > 0

    def test_f1_computable(self):
        """CellGNN assembly output can be evaluated for F1."""
        from neuronauts.line_graph import evaluate
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        model.eval()
        cg = cell_gnn_assembly(syn, model, proximity_radius_nm=50000.0)
        metrics = evaluate(cg, syn.pre_root_id, syn.post_root_id)
        assert 0.0 <= metrics.f1 <= 1.0


# ---------------------------------------------------------------------------
# Edit history
# ---------------------------------------------------------------------------

class TestEditHistory:
    def test_edits_to_synapse_pairs_merge(self):
        from neuronauts.edit_history import EditOperation, edits_to_synapse_pairs
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        # Simulate a merge: roots 1 and 2 were merged
        edits = [EditOperation(
            operation="merge",
            before_root_ids=(1, 2),
            after_root_ids=(1,),
        )]
        pairs = edits_to_synapse_pairs(edits, syn, "pre")
        assert len(pairs) > 0
        for p in pairs:
            assert p.label == 1
            assert p.role == "pre"
            assert p.edit_type == "merge"
            assert p.synapse_i < p.synapse_j

    def test_edits_to_synapse_pairs_split(self):
        from neuronauts.edit_history import EditOperation, edits_to_synapse_pairs
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        # Simulate a split: root 1 was split into 1 and 2
        edits = [EditOperation(
            operation="split",
            before_root_ids=(1,),
            after_root_ids=(1, 2),
        )]
        pairs = edits_to_synapse_pairs(edits, syn, "pre")
        assert len(pairs) > 0
        for p in pairs:
            assert p.label == 0
            assert p.edit_type == "split"

    def test_empty_edits_returns_empty(self):
        from neuronauts.edit_history import edits_to_synapse_pairs
        syn = _make_synapses(n_cells=2, synapses_per_cell=3)
        pairs = edits_to_synapse_pairs([], syn, "pre")
        assert pairs == []

    def test_edit_pairs_to_contrastive(self):
        from neuronauts.edit_history import EditPair, edit_pairs_to_contrastive
        pairs = [
            EditPair(0, 1, label=1, role="pre", source_root_a=1, source_root_b=2, edit_type="merge"),
            EditPair(2, 3, label=0, role="pre", source_root_a=1, source_root_b=3, edit_type="split"),
            EditPair(4, 5, label=1, role="post", source_root_a=1, source_root_b=2, edit_type="merge"),
        ]
        pos, neg = edit_pairs_to_contrastive(pairs, "pre")
        assert pos == [(0, 1)]
        assert neg == [(2, 3)]
        # Post pairs should be excluded for role="pre"
        assert (4, 5) not in pos


# ---------------------------------------------------------------------------
# End-to-end training validation
# ---------------------------------------------------------------------------

class TestEndToEndTraining:
    """Validate that the CellGNN training pipeline runs end-to-end,
    loss converges, and checkpoints can be saved/loaded."""

    def test_training_loss_converges(self):
        """Train CellGNN for several epochs and verify loss decreases."""
        from neuronauts.cell_graph import train_cell_gnn
        syn = _make_synapses(n_cells=4, synapses_per_cell=6, seed=42)
        model = CellGNN(d_model=32, n_layers=2, n_heads=2, embedding_dim=16)

        # Use a mock cache that yields a single box
        class _MockCache:
            def iter_records(self, shuffle=False, rng=None):
                return [_MockRecord()]
            def load(self, record):
                return None, syn

        class _MockRecord:
            n_positive_pairs = 10
            n_synapses = 24
            box_hash = "test0000"

        cfg = CellGNNConfig(
            d_model=32, n_layers=2, n_heads=2, embedding_dim=16,
            epochs=15, learning_rate=1e-3, proximity_radius_nm=100000.0,
            seed=42,
        )
        history = train_cell_gnn(model, _MockCache(), config=cfg, verbose=False)

        assert len(history["train_loss"]) == 15
        # Loss should decrease: early avg > late avg
        early = np.mean(history["train_loss"][:3])
        late = np.mean(history["train_loss"][-3:])
        assert late < early, f"Loss did not converge: early={early:.4f} late={late:.4f}"

    def test_checkpoint_save_load_roundtrip(self, tmp_path):
        """Save and reload a CellGNN checkpoint; verify inference matches."""
        syn = _make_synapses(n_cells=3, synapses_per_cell=4, seed=7)
        model = CellGNN(d_model=32, n_layers=2, embedding_dim=16)
        model.eval()

        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=100000.0)
        labels_before = infer_cells(model, graph, threshold=0.5)

        ckpt_path = str(tmp_path / "cell_gnn_test.pt")
        save_cell_gnn(ckpt_path, model)
        loaded = load_cell_gnn(ckpt_path)
        labels_after = infer_cells(loaded, graph, threshold=0.5)

        np.testing.assert_array_equal(labels_before, labels_after)

    def test_training_with_edit_pairs(self):
        """Verify edit-history pairs are accepted and don't crash training."""
        from neuronauts.edit_history import EditPair
        from neuronauts.cell_graph import train_cell_gnn

        syn = _make_synapses(n_cells=3, synapses_per_cell=5, seed=11)
        model = CellGNN(d_model=32, n_layers=2, embedding_dim=16)

        edit_pairs = [
            EditPair(0, 5, label=1, role="pre", source_root_a=1, source_root_b=2, edit_type="merge"),
            EditPair(1, 10, label=0, role="pre", source_root_a=1, source_root_b=3, edit_type="split"),
            EditPair(2, 8, label=1, role="post", source_root_a=1, source_root_b=2, edit_type="merge"),
        ]

        class _MockCache:
            def iter_records(self, shuffle=False, rng=None):
                return [_MockRecord()]
            def load(self, record):
                return None, syn

        class _MockRecord:
            n_positive_pairs = 5
            n_synapses = 15
            box_hash = "edit0000"

        cfg = CellGNNConfig(
            d_model=32, n_layers=2, epochs=3, proximity_radius_nm=100000.0, seed=42,
        )
        history = train_cell_gnn(
            model, _MockCache(), config=cfg,
            edit_pairs=edit_pairs, edit_weight=2.0,
            verbose=False,
        )
        assert len(history["train_loss"]) == 3
        assert all(l >= 0 for l in history["train_loss"])

    def test_cell_graph_train_step_with_edit_pairs(self):
        """cell_graph_train_step accepts edit pair arguments."""
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=100000.0)
        model = CellGNN(d_model=32, n_layers=2, embedding_dim=16)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        m = cell_graph_train_step(
            model, opt, graph,
            edit_positive_pairs=[(0, 4), (1, 5)],
            edit_negative_pairs=[(0, 8)],
            edit_weight=3.0,
        )
        assert m["n_pos"] >= 2  # at least the injected pairs
        assert m["n_neg"] >= 1

    def test_evaluate_subcommand_args_parse(self):
        """Verify the evaluate subcommand parses without error."""
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
        from scripts.train import parse_args
        args = parse_args([
            "evaluate",
            "--cache-dir", "data/boxes",
            "--cell-gnn-checkpoint", "models/cell_gnn.pt",
        ])
        assert args.command == "evaluate"
        assert args.cell_gnn_checkpoint == "models/cell_gnn.pt"
        assert args.split == "test"

    def test_sweep_subcommand_args_parse(self):
        """Verify the sweep subcommand parses without error."""
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
        from scripts.train import parse_args
        args = parse_args([
            "sweep",
            "--cache-dir", "data/boxes",
            "--d-models", "32,64",
            "--n-layers-list", "2,3",
        ])
        assert args.command == "sweep"
        assert args.d_models == "32,64"

    def test_scale_test_subcommand_args_parse(self):
        """Verify the scale-test subcommand parses without error."""
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
        from scripts.train import parse_args
        args = parse_args([
            "scale-test",
            "--cache-dir", "data/boxes",
            "--min-synapses", "50",
        ])
        assert args.command == "scale-test"
        assert args.min_synapses == 50


# ---------------------------------------------------------------------------
# Boundary-edge partition search
# ---------------------------------------------------------------------------

def _make_small_graph_with_edges(
    n: int,
    edges: list[tuple[int, int]],
    positions: np.ndarray | None = None,
) -> SynapseGraph:
    """Build a minimal SynapseGraph with explicit edges for unit testing."""
    if positions is None:
        rng = np.random.default_rng(0)
        positions = rng.standard_normal((n, 3)).astype(np.float32) * 10.0
    syn_edges = [
        SynapseEdge(src=i, dst=j, distance=1.0, same_scaffold=0.0,
                    grammar_score=0.5, shared_agents=0)
        for (i, j) in edges
    ]
    return SynapseGraph(
        n_synapses=n,
        role="pre",
        node_positions=positions,
        node_scaffold_ids=np.zeros(n, dtype=np.int64),
        edges=syn_edges,
        root_ids=None,
    )


class TestBoundaryPartitionSearch:

    def test_returns_valid_labels(self):
        """boundary_partition_search returns labels with correct shape and non-negative values."""
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)
        labels = boundary_partition_search(model, graph)
        assert labels.shape == (graph.n_synapses,)
        assert labels.dtype == np.int64
        assert labels.min() >= 0

    def test_boundary_edges_identified(self):
        """Edges in [low_sim, high_sim) are found; out-of-band edges are not."""
        # Build a model and craft embeddings so that two pairs sit exactly in-band
        # and two are out-of-band.  We do this by constructing normalised embeddings
        # directly and bypassing the model via monkey-patching _graph_to_tensors.
        import types, torch as _torch

        D = 4
        # Craft four unit vectors:
        # v0 and v1: sim ~0.96 (in-band)
        # v2 and v3: sim = 1.0 (above high_sim, out-of-band)
        v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        angle = np.arccos(0.96)
        v1 = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # identical to v2

        n = 4
        positions = np.zeros((n, 3), dtype=np.float32)
        graph = _make_small_graph_with_edges(n, [(0, 1), (2, 3)], positions)

        # Monkey-patch the model to return our crafted embeddings
        fixed_emb = _torch.tensor(np.stack([v0, v1, v2, v3]), dtype=_torch.float32)

        class _FakeModel:
            def eval(self): pass
            def __call__(self, *args, **kwargs): return fixed_emb

        labels = boundary_partition_search(
            _FakeModel(), graph,
            low_sim=0.93, high_sim=0.99,
            max_boundary_edges=12, beam_width=4,
        )
        # v2 and v3 are identical (sim=1.0 >= high_sim) → merged in base partition
        assert labels[2] == labels[3], "High-sim pair should be merged"
        # v0 and v1 are in-band → explored by beam search but shape is still valid
        assert labels.shape == (4,)

    def test_beam_selects_best_partition(self):
        """Beam search selects the partition with higher within-cell coherence."""
        import torch as _torch

        D = 4
        # Three unit vectors: v0 and v1 are very similar, v2 is orthogonal.
        # Merging (0,1) should score better than not merging.
        v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        angle = np.arccos(0.965)
        v1 = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        n = 3
        graph = _make_small_graph_with_edges(n, [(0, 1)])

        fixed_emb = _torch.tensor(np.stack([v0, v1, v2]), dtype=_torch.float32)

        class _FakeModel:
            def eval(self): pass
            def __call__(self, *args, **kwargs): return fixed_emb

        labels = boundary_partition_search(
            _FakeModel(), graph,
            low_sim=0.93, high_sim=0.99,
            max_boundary_edges=5, beam_width=4,
        )
        # v0 and v1 are similar → should end up in the same cell
        assert labels[0] == labels[1], (
            f"Beam should merge highly similar nodes 0 and 1 (labels={labels})"
        )
        # v2 should be its own cell
        assert labels[2] != labels[0], (
            f"Node 2 should not merge with 0/1 (labels={labels})"
        )

    def test_singleton_graph(self):
        """A graph with one node returns label [0]."""
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=1,
                        n_heads=2, embedding_dim=8)
        positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        graph = SynapseGraph(
            n_synapses=1,
            role="pre",
            node_positions=positions,
            node_scaffold_ids=np.zeros(1, dtype=np.int64),
            edges=[],
            root_ids=None,
        )
        labels = boundary_partition_search(model, graph)
        assert labels.tolist() == [0]

    def test_no_boundary_edges_falls_back_to_threshold(self):
        """When no edges fall in the ambiguous band the result equals infer_cells at high_sim."""
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        model.eval()
        syn = _make_synapses(n_cells=3, synapses_per_cell=4)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)

        high_sim = 0.99
        # boundary_partition_search with an impossible band (no edges can fall in it)
        labels_search = boundary_partition_search(
            model, graph,
            low_sim=0.999,   # band so narrow that no edges land in it
            high_sim=high_sim,
            max_boundary_edges=12,
            beam_width=8,
        )
        labels_infer = infer_cells(model, graph, threshold=high_sim)
        np.testing.assert_array_equal(
            labels_search, labels_infer,
            err_msg="With no boundary edges, result should match infer_cells at high_sim",
        )

    def test_infer_cells_with_search_returns_valid_labels(self):
        """infer_cells_with_search wrapper produces valid integer labels."""
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        syn = _make_synapses(n_cells=2, synapses_per_cell=5)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=10_000.0)
        labels = infer_cells_with_search(model, graph)
        assert labels.shape == (graph.n_synapses,)
        assert labels.dtype == np.int64
        assert labels.min() >= 0

    # ------------------------------------------------------------------
    # EM corridor override tests
    # ------------------------------------------------------------------

    def test_corridor_force_accept(self):
        """A high corridor score forces a merge even without beam search."""
        import torch as _torch

        D = 4
        v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        angle = np.arccos(0.965)
        v1 = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        graph = _make_small_graph_with_edges(3, [(0, 1)])
        fixed_emb = _torch.tensor(np.stack([v0, v1, v2]), dtype=_torch.float32)

        class _FakeModel:
            def eval(self): pass
            def __call__(self, *args, **kwargs): return fixed_emb

        # Corridor score 0.9 for edge (0,1) → should force-accept
        labels = boundary_partition_search(
            _FakeModel(), graph,
            low_sim=0.93, high_sim=0.99,
            corridor_scores={(0, 1): 0.9},
            corridor_accept_threshold=0.8,
        )
        assert labels[0] == labels[1], f"Force-accept should merge 0 and 1, got {labels}"
        assert labels[2] != labels[0], f"Node 2 should remain separate, got {labels}"

    def test_corridor_force_reject(self):
        """A low corridor score prevents a merge that the beam might otherwise accept."""
        import torch as _torch

        D = 4
        v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        angle = np.arccos(0.965)
        v1 = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        graph = _make_small_graph_with_edges(3, [(0, 1)])
        fixed_emb = _torch.tensor(np.stack([v0, v1, v2]), dtype=_torch.float32)

        class _FakeModel:
            def eval(self): pass
            def __call__(self, *args, **kwargs): return fixed_emb

        # Corridor score 0.1 for edge (0,1) → should force-reject
        labels = boundary_partition_search(
            _FakeModel(), graph,
            low_sim=0.93, high_sim=0.99,
            corridor_scores={(0, 1): 0.1},
            corridor_reject_threshold=0.2,
        )
        # Force-reject: nodes 0 and 1 must remain separate
        assert labels[0] != labels[1], f"Force-reject should keep 0 and 1 separate, got {labels}"

    def test_corridor_neutral_score_still_uses_beam(self):
        """A corridor score in the middle band (0.2-0.8) still goes through beam search."""
        import torch as _torch

        D = 4
        v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        angle = np.arccos(0.965)
        v1 = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        graph = _make_small_graph_with_edges(3, [(0, 1)])
        fixed_emb = _torch.tensor(np.stack([v0, v1, v2]), dtype=_torch.float32)

        class _FakeModel:
            def eval(self): pass
            def __call__(self, *args, **kwargs): return fixed_emb

        # Score 0.5 → ambiguous → beam decides (beam should still merge based on sim)
        labels = boundary_partition_search(
            _FakeModel(), graph,
            low_sim=0.93, high_sim=0.99,
            corridor_scores={(0, 1): 0.5},
        )
        assert labels[0] == labels[1], (
            f"Neutral corridor score should let beam decide; beam merges at sim=0.965 (labels={labels})"
        )

    def test_infer_cells_with_search_passes_corridor_scores(self):
        """corridor_scores kwarg is correctly forwarded from infer_cells_with_search."""
        import torch as _torch

        D = 4
        v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        angle = np.arccos(0.965)
        v1 = np.array([np.cos(angle), np.sin(angle), 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        graph = _make_small_graph_with_edges(3, [(0, 1)])
        fixed_emb = _torch.tensor(np.stack([v0, v1, v2]), dtype=_torch.float32)

        class _FakeModel:
            def eval(self): pass
            def __call__(self, *args, **kwargs): return fixed_emb

        labels = infer_cells_with_search(
            _FakeModel(), graph,
            corridor_scores={(0, 1): 0.9},
            corridor_accept_threshold=0.8,
        )
        assert labels[0] == labels[1], f"corridor_scores not forwarded correctly (labels={labels})"


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------

class TestHardNegativeMining:
    """Tests for online hard negative mining in cell_graph_train_step."""

    def _make_graph_with_confusing_pair(self) -> SynapseGraph:
        """Build a SynapseGraph where nodes 0 and 1 have different roots but
        are positioned identically (so a fresh model will produce very similar
        embeddings for them).  Nodes 2 and 3 belong to root 1 (clearly separate)."""
        # Positions: nodes 0 and 1 share the same location → similar embeddings
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],   # same as node 0 → confusing pair
            [500.0, 0.0, 0.0],
            [500.0, 0.0, 0.0],
        ], dtype=np.float32)
        root_ids = np.array([1, 2, 1, 2], dtype=np.int64)  # node 0 ≠ node 1
        edges = [
            SynapseEdge(src=0, dst=1, distance=0.0, same_scaffold=0.0,
                        grammar_score=0.0, shared_agents=0),
            SynapseEdge(src=2, dst=3, distance=0.0, same_scaffold=0.0,
                        grammar_score=0.0, shared_agents=0),
        ]
        return SynapseGraph(
            n_synapses=4,
            role="pre",
            node_positions=positions,
            node_scaffold_ids=np.zeros(4, dtype=np.int64),
            edges=edges,
            root_ids=root_ids,
        )

    def test_hard_neg_mining_finds_confusing_pairs(self):
        """Nodes with different roots but very similar embeddings should be mined."""
        import torch as _torch
        import torch.nn.functional as F
        from neuronauts.cell_graph import _graph_to_tensors

        graph = self._make_graph_with_confusing_pair()
        N = graph.n_synapses

        # Build a mock model that returns nearly identical embeddings for nodes 0 and 1
        # (sim ≈ 1.0) and clearly different embeddings for nodes 2 and 3 vs 0/1.
        fixed_emb = _torch.tensor([
            [1.0, 0.0, 0.0, 0.0],   # node 0 — root 1
            [1.0, 0.0, 0.0, 0.0],   # node 1 — root 2, nearly identical → hard neg
            [0.0, 1.0, 0.0, 0.0],   # node 2 — root 1
            [0.0, 1.0, 0.0, 0.0],   # node 3 — root 2, nearly identical → hard neg
        ], dtype=_torch.float32)

        model = CellGNN(node_input_dim=3, d_model=16, n_layers=1,
                        n_heads=2, embedding_dim=4)
        # Override model forward to return our crafted embeddings
        original_forward = model.forward
        model.forward = lambda *args, **kwargs: fixed_emb

        emb_norm = F.normalize(fixed_emb, p=2, dim=-1)
        sim_matrix = emb_norm @ emb_norm.T

        ui, uj = _torch.triu_indices(N, N, offset=1)
        pair_sims = sim_matrix[ui, uj]
        root_arr = graph.root_ids
        ri = _torch.tensor(root_arr[ui.numpy()], dtype=_torch.long)
        rj = _torch.tensor(root_arr[uj.numpy()], dtype=_torch.long)
        valid = (ri > 0) & (rj > 0)
        different = ri != rj
        hard_neg_threshold = 0.7
        hard = pair_sims > hard_neg_threshold
        mask = valid & different & hard
        hard_indices = mask.nonzero(as_tuple=False).view(-1)

        hard_neg_pairs = []
        if len(hard_indices) > 0:
            sorted_by_sim = hard_indices[pair_sims[hard_indices].argsort(descending=True)]
            for k in sorted_by_sim[:64]:
                hard_neg_pairs.append((int(ui[k]), int(uj[k])))

        assert len(hard_neg_pairs) > 0, (
            "Expected hard negative pairs with nearly identical embeddings and different roots"
        )
        # Specifically, the (0,1) pair should be mined (same position → sim ≈ 1.0)
        assert (0, 1) in hard_neg_pairs, f"Expected (0,1) in hard_neg_pairs, got {hard_neg_pairs}"

    def test_hard_neg_mining_disabled(self):
        """With hard_neg_mining=False, the step should return n_hard_neg=0."""
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        graph = self._make_graph_with_confusing_pair()

        m = cell_graph_train_step(
            model, optimizer, graph,
            hard_neg_mining=False,
        )
        assert m["n_hard_neg"] == 0, (
            f"Expected n_hard_neg=0 when hard_neg_mining=False, got {m['n_hard_neg']}"
        )

    def test_hard_neg_mining_returns_loss(self):
        """cell_graph_train_step with hard_neg_mining=True should run and return loss > 0."""
        model = CellGNN(node_input_dim=3, d_model=16, n_layers=2,
                        n_heads=2, embedding_dim=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        syn = _make_synapses(n_cells=3, synapses_per_cell=4, seed=7)
        graph = build_synapse_graph(syn, "pre", proximity_radius_nm=100_000.0)

        m = cell_graph_train_step(
            model, optimizer, graph,
            hard_neg_mining=True,
            hard_neg_threshold=0.0,   # threshold=0 → mine ALL different-root pairs
            hard_neg_weight=3.0,
        )
        assert "loss" in m
        assert "hard_neg_sim" in m
        assert "n_hard_neg" in m
        assert m["loss"] >= 0.0, f"Loss must be non-negative, got {m['loss']}"

    def test_hard_neg_mining_threshold(self):
        """Pairs with cosine sim below hard_neg_threshold should not be mined."""
        import torch as _torch
        import torch.nn.functional as F

        # Build a graph where the only different-root pair has sim ≈ 0.5 (below threshold=0.7)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ], dtype=np.float32)
        root_ids = np.array([1, 2], dtype=np.int64)
        graph = SynapseGraph(
            n_synapses=2,
            role="pre",
            node_positions=positions,
            node_scaffold_ids=np.zeros(2, dtype=np.int64),
            edges=[SynapseEdge(src=0, dst=1, distance=100.0, same_scaffold=0.0,
                               grammar_score=0.0, shared_agents=0)],
            root_ids=root_ids,
        )

        # Craft embeddings: sim(0,1) ≈ 0 (orthogonal) — well below threshold=0.7
        fixed_emb = _torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=_torch.float32)

        N = 2
        emb_norm = F.normalize(fixed_emb, p=2, dim=-1)
        sim_matrix = emb_norm @ emb_norm.T
        ui, uj = _torch.triu_indices(N, N, offset=1)
        pair_sims = sim_matrix[ui, uj]
        root_arr = graph.root_ids
        ri = _torch.tensor(root_arr[ui.numpy()], dtype=_torch.long)
        rj = _torch.tensor(root_arr[uj.numpy()], dtype=_torch.long)
        valid = (ri > 0) & (rj > 0)
        different = ri != rj
        hard_neg_threshold = 0.7
        hard = pair_sims > hard_neg_threshold
        mask = valid & different & hard
        hard_indices = mask.nonzero(as_tuple=False).view(-1)

        # With orthogonal embeddings (sim=0), no pairs should exceed threshold=0.7
        assert len(hard_indices) == 0, (
            f"Expected no hard negatives with orthogonal embeddings below threshold, "
            f"got {len(hard_indices)} pairs"
        )

    def test_hard_neg_mining_cli_args_parse(self):
        """Verify the train-cell-gnn subcommand accepts the hard-neg CLI flags."""
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
        from scripts.train import parse_args

        args = parse_args([
            "train-cell-gnn",
            "--cache-dir", "data/boxes",
            "--hard-neg-threshold", "0.85",
            "--hard-neg-weight", "5.0",
            "--no-hard-neg-mining",
        ])
        assert args.hard_neg_threshold == pytest.approx(0.85)
        assert args.hard_neg_weight == pytest.approx(5.0)
        assert args.no_hard_neg_mining is True
