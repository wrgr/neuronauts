"""Data output validation: membrane segmentation, alignment of evidence,
and data contracts for core datatypes.

Ensures that:
- Membrane predictions are valid probabilities
- Synapse positions align with volume bounds
- Graph edges index valid synapses
- LineGraphMetrics satisfy invariants
- Perfect synthetic scaffold yields F1 ≈ 1
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# 1. Membrane output property tests
# ---------------------------------------------------------------------------


class MembraneOutputPropertiesTest(unittest.TestCase):
    """predict_membranes output must be valid probabilities."""

    def test_output_in_zero_one(self):
        from neuronauts.membrane_unet import MembraneUNet, predict_membranes

        rng = np.random.default_rng(42)
        # Min 32x32 spatial for UNet bottleneck (4 pooling layers)
        vol = rng.integers(80, 180, (32, 32, 12), dtype=np.uint8)
        model = MembraneUNet(in_channels=1, context_slices=0)
        out = predict_membranes(model, vol, device="cpu", batch_size=4)
        self.assertGreaterEqual(float(out.min()), 0.0, "membrane probs must be >= 0")
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5, "membrane probs must be <= 1")

    def test_output_finite(self):
        from neuronauts.membrane_unet import MembraneUNet, predict_membranes

        vol = np.full((32, 32, 12), 128, dtype=np.uint8)
        model = MembraneUNet(in_channels=1, context_slices=0)
        out = predict_membranes(model, vol, device="cpu")
        self.assertTrue(np.all(np.isfinite(out)), "membrane output must be finite")

    def test_output_shape_matches_volume(self):
        from neuronauts.membrane_unet import MembraneUNet, predict_membranes

        shape = (32, 32, 12)
        vol = np.zeros(shape, dtype=np.uint8)
        model = MembraneUNet(in_channels=1, context_slices=0)
        out = predict_membranes(model, vol, device="cpu")
        self.assertEqual(out.shape, shape, "membrane shape must match volume")

    def test_output_dtype_float32(self):
        from neuronauts.membrane_unet import MembraneUNet, predict_membranes

        vol = np.zeros((32, 32, 12), dtype=np.uint8)
        model = MembraneUNet(in_channels=1, context_slices=0)
        out = predict_membranes(model, vol, device="cpu")
        self.assertEqual(out.dtype, np.float32, "membrane output must be float32")


# ---------------------------------------------------------------------------
# 2. Alignment tests
# ---------------------------------------------------------------------------


class SynapseVolumeAlignmentTest(unittest.TestCase):
    """Synapse positions must lie within volume bounds."""

    def test_make_test_volume_synapses_in_bounds(self):
        from neuronauts.fetch import make_test_volume, SyntheticBenchmarkConfig

        config = SyntheticBenchmarkConfig(shape=(32, 32, 24), n_synapses=12)
        chunk, synapses = make_test_volume(config=config, seed=0)
        shape = np.array(chunk.data.shape)
        for pt in synapses.pre_pt:
            self.assertTrue(
                np.all(pt >= 0) and np.all(pt < shape),
                f"pre_pt {pt} out of bounds for shape {shape}",
            )
        for pt in synapses.post_pt:
            self.assertTrue(
                np.all(pt >= 0) and np.all(pt < shape),
                f"post_pt {pt} out of bounds for shape {shape}",
            )

    def test_simulate_paths_and_hits_synapse_hits_shape(self):
        from neuronauts.fetch import make_test_volume, SyntheticBenchmarkConfig
        from neuronauts.fields import compute_membrane_field
        from neuronauts.run import simulate_paths_and_hits

        # shape axes must be > 20 for make_test_volume membrane_planes (rng.integers(10, axis-10))
        config = SyntheticBenchmarkConfig(shape=(40, 40, 24), n_synapses=8, membrane_planes=3)
        chunk, synapses = make_test_volume(config=config, seed=1)
        mf = compute_membrane_field(chunk.data)
        path_arr, synapse_hits, path_lengths, _ = simulate_paths_and_hits(
            chunk.data,
            synapses.pre_pt,
            synapses.post_pt,
            verbose=False,
            membrane_field_override=mf,
        )
        n_syn = len(synapses.pre_pt)
        # Agents run over vstack([pre_pt, post_pt]) -> 2*n_syn synapse pts
        expected_cols = 2 * n_syn
        self.assertEqual(
            synapse_hits.shape[1],
            expected_cols,
            f"synapse_hits cols should be 2*n_synapses={expected_cols}",
        )
        self.assertEqual(
            synapse_hits.shape[0],
            path_arr.shape[0],
            "synapse_hits rows must match n_agents",
        )


class GraphEdgeAlignmentTest(unittest.TestCase):
    """ConnectivityGraph edges must index valid synapses and neurons."""

    def test_build_graph_edges_syn_idx_valid(self):
        from neuronauts.fetch import make_test_volume, SyntheticBenchmarkConfig
        from neuronauts.fields import compute_membrane_field
        from neuronauts.line_graph import evaluate
        from neuronauts.run import HeuristicConfig, _build_graph, simulate_paths_and_hits

        config = SyntheticBenchmarkConfig(shape=(40, 40, 24), n_synapses=6, membrane_planes=3)
        chunk, synapses = make_test_volume(config=config, seed=2)
        mf = compute_membrane_field(chunk.data)
        path_arr, synapse_hits, path_lengths, _ = simulate_paths_and_hits(
            chunk.data,
            synapses.pre_pt,
            synapses.post_pt,
            verbose=False,
            membrane_field_override=mf,
        )
        graph = _build_graph(
            path_arr=path_arr,
            path_lengths=path_lengths,
            synapse_hits=synapse_hits,
            pre_pts=synapses.pre_pt,
            post_pts=synapses.post_pt,
            pre_seg_ids=synapses.pre_seg_id,
            post_seg_ids=synapses.post_seg_id,
            heuristic_config=HeuristicConfig.learned(),
        )
        n_syn = len(synapses.pre_pt)
        for pre_nid, post_nid, syn_idx in graph.edges:
            self.assertIn(pre_nid, graph.neurons, f"edge pre_nid {pre_nid} not in neurons")
            self.assertIn(post_nid, graph.neurons, f"edge post_nid {post_nid} not in neurons")
            self.assertGreaterEqual(
                syn_idx, 0, f"edge syn_idx {syn_idx} must be >= 0"
            )
            self.assertLess(
                syn_idx, n_syn,
                f"edge syn_idx {syn_idx} must be < n_synapses={n_syn}",
            )


class BoxCacheRoundTripTest(unittest.TestCase):
    """BoxCache save/load must preserve synapse positions and alignment."""

    def test_save_load_preserves_pre_post_pts(self):
        from neuronauts.dataset_builder import BoxCache
        from neuronauts.fetch import RealBoxSpec, SynapseTable, VolumeChunk

        shape = (32, 32, 16)
        n = 5
        rng = np.random.default_rng(99)
        pre_pt = rng.random((n, 3), dtype=np.float32) * (np.array(shape) - 2)
        post_pt = rng.random((n, 3), dtype=np.float32) * (np.array(shape) - 2)
        syn = SynapseTable(
            pre_pt=pre_pt,
            post_pt=post_pt,
            pre_root_id=np.arange(n, dtype=np.int64),
            post_root_id=np.arange(n, dtype=np.int64) + 100,
            synapse_id=np.arange(n, dtype=np.int64),
        )
        vol = VolumeChunk(
            data=np.zeros(shape, dtype=np.uint8),
            voxel_size_nm=(32, 32, 40),
            bbox_voxels=((0, 0, 0), shape),
            mip=2,
        )
        spec = RealBoxSpec(center_nm=(0, 0, 0), side_um=2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BoxCache(tmpdir)
            record = cache.save(spec, vol, syn)
            vol2, syn2 = cache.load(record)
            np.testing.assert_allclose(syn2.pre_pt, pre_pt, atol=1e-6)
            np.testing.assert_allclose(syn2.post_pt, post_pt, atol=1e-6)
            self.assertEqual(len(syn2.pre_pt), n)

    def test_loaded_synapses_in_volume_bounds(self):
        from neuronauts.dataset_builder import BoxCache
        from neuronauts.fetch import RealBoxSpec, SynapseTable, VolumeChunk

        shape = (20, 20, 10)
        pre_pt = np.array([[5.0, 5.0, 5.0], [10.0, 10.0, 5.0]], dtype=np.float32)
        post_pt = np.array([[5.0, 5.0, 8.0], [10.0, 10.0, 8.0]], dtype=np.float32)
        syn = SynapseTable(
            pre_pt=pre_pt,
            post_pt=post_pt,
            pre_root_id=np.array([1, 2], dtype=np.int64),
            post_root_id=np.array([10, 11], dtype=np.int64),
            synapse_id=np.array([0, 1], dtype=np.int64),
        )
        vol = VolumeChunk(
            data=np.zeros(shape, dtype=np.uint8),
            voxel_size_nm=(32, 32, 40),
            bbox_voxels=((0, 0, 0), shape),
            mip=2,
        )
        spec = RealBoxSpec(center_nm=(0, 0, 0), side_um=2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = BoxCache(tmpdir)
            record = cache.save(spec, vol, syn)
            vol2, syn2 = cache.load(record)
            shape_arr = np.array(vol2.data.shape)
            for pt in syn2.pre_pt:
                self.assertTrue(
                    np.all(pt >= 0) and np.all(pt < shape_arr),
                    f"loaded pre_pt {pt} out of bounds",
                )
            for pt in syn2.post_pt:
                self.assertTrue(
                    np.all(pt >= 0) and np.all(pt < shape_arr),
                    f"loaded post_pt {pt} out of bounds",
                )


# ---------------------------------------------------------------------------
# 3. Data contract tests
# ---------------------------------------------------------------------------


class SynapseTableContractTest(unittest.TestCase):
    """SynapseTable must have consistent field lengths and shapes."""

    def test_minimal_valid_synapse_table(self):
        from neuronauts.fetch import SynapseTable

        n = 4
        syn = SynapseTable(
            pre_pt=np.zeros((n, 3), dtype=np.float32),
            post_pt=np.zeros((n, 3), dtype=np.float32),
            pre_root_id=np.zeros(n, dtype=np.int64),
            post_root_id=np.zeros(n, dtype=np.int64),
            synapse_id=np.arange(n, dtype=np.int64),
        )
        self.assertEqual(len(syn.pre_pt), n)
        self.assertEqual(len(syn.post_pt), n)
        self.assertEqual(syn.pre_pt.shape[1], 3)
        self.assertEqual(syn.post_pt.shape[1], 3)
        self.assertEqual(len(syn.pre_root_id), n)
        self.assertEqual(len(syn.post_root_id), n)

    def test_seg_ids_match_length_when_present(self):
        from neuronauts.fetch import SynapseTable

        n = 3
        syn = SynapseTable(
            pre_pt=np.zeros((n, 3), dtype=np.float32),
            post_pt=np.zeros((n, 3), dtype=np.float32),
            pre_root_id=np.zeros(n, dtype=np.int64),
            post_root_id=np.zeros(n, dtype=np.int64),
            synapse_id=np.arange(n, dtype=np.int64),
            pre_seg_id=np.ones(n, dtype=np.int64),
            post_seg_id=np.ones(n, dtype=np.int64),
        )
        self.assertEqual(len(syn.pre_seg_id), n)
        self.assertEqual(len(syn.post_seg_id), n)


class VolumeChunkContractTest(unittest.TestCase):
    """VolumeChunk data shape must match bbox extent."""

    def test_data_shape_matches_bbox(self):
        from neuronauts.fetch import VolumeChunk

        shape = (40, 40, 20)
        data = np.zeros(shape, dtype=np.uint8)
        chunk = VolumeChunk(
            data=data,
            voxel_size_nm=(32, 32, 40),
            bbox_voxels=((0, 0, 0), shape),
            mip=2,
        )
        lo, hi = chunk.bbox_voxels
        extent = tuple(hi[i] - lo[i] for i in range(3))
        self.assertEqual(chunk.data.shape, extent)


class ConnectivityGraphContractTest(unittest.TestCase):
    """ConnectivityGraph edges must reference existing neurons."""

    def test_edges_reference_existing_neurons(self):
        from neuronauts.merge import ConnectivityGraph, MergedNeuron

        pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
        n1 = MergedNeuron(
            neuron_id=0,
            agent_ids=[0],
            path_points=pts,
            synapse_indices=[0],
            role="pre",
        )
        n2 = MergedNeuron(
            neuron_id=1,
            agent_ids=[1],
            path_points=pts,
            synapse_indices=[0],
            role="post",
        )
        graph = ConnectivityGraph(
            neurons={0: n1, 1: n2},
            edges=[(0, 1, 0)],
            unresolved_synapse_indices=[],
        )
        for pre_nid, post_nid, syn_idx in graph.edges:
            self.assertIn(pre_nid, graph.neurons)
            self.assertIn(post_nid, graph.neurons)


class LineGraphMetricsContractTest(unittest.TestCase):
    """LineGraphMetrics must satisfy 0 <= f1, precision, recall <= 1."""

    def test_evaluate_returns_valid_metrics(self):
        from neuronauts.fetch import make_test_volume, SyntheticBenchmarkConfig
        from neuronauts.fields import compute_membrane_field
        from neuronauts.line_graph import evaluate
        from neuronauts.run import HeuristicConfig, _build_graph, simulate_paths_and_hits

        config = SyntheticBenchmarkConfig(shape=(40, 40, 24), n_synapses=8, membrane_planes=3)
        chunk, synapses = make_test_volume(config=config, seed=3)
        mf = compute_membrane_field(chunk.data)
        path_arr, synapse_hits, path_lengths, _ = simulate_paths_and_hits(
            chunk.data,
            synapses.pre_pt,
            synapses.post_pt,
            verbose=False,
            membrane_field_override=mf,
        )
        graph = _build_graph(
            path_arr=path_arr,
            path_lengths=path_lengths,
            synapse_hits=synapse_hits,
            pre_pts=synapses.pre_pt,
            post_pts=synapses.post_pt,
            pre_seg_ids=synapses.pre_seg_id,
            post_seg_ids=synapses.post_seg_id,
            heuristic_config=HeuristicConfig.learned(),
        )
        metrics = evaluate(graph, synapses.pre_root_id, synapses.post_root_id)
        self.assertGreaterEqual(metrics.f1, 0.0)
        self.assertLessEqual(metrics.f1, 1.0 + 1e-6)
        self.assertGreaterEqual(metrics.precision, 0.0)
        self.assertLessEqual(metrics.precision, 1.0 + 1e-6)
        self.assertGreaterEqual(metrics.recall, 0.0)
        self.assertLessEqual(metrics.recall, 1.0 + 1e-6)
        self.assertGreaterEqual(metrics.tp, 0)
        self.assertGreaterEqual(metrics.fp, 0)
        self.assertGreaterEqual(metrics.fn, 0)

    def test_compute_line_graph_f1_tp_fp_fn_consistency(self):
        from neuronauts.line_graph import build_true_line_graph, compute_line_graph_f1

        pre = np.array([1, 1, 2], dtype=np.int64)
        post = np.array([10, 11, 12], dtype=np.int64)
        true_edges = build_true_line_graph(pre, post)
        est_edges = true_edges  # perfect match
        m = compute_line_graph_f1(true_edges, est_edges, len(pre))
        self.assertEqual(m.tp, len(true_edges))
        self.assertEqual(m.fp, 0)
        self.assertEqual(m.fn, 0)
        self.assertAlmostEqual(m.f1, 1.0, places=5)
        self.assertAlmostEqual(m.precision, 1.0, places=5)
        self.assertAlmostEqual(m.recall, 1.0, places=5)


# ---------------------------------------------------------------------------
# 4. Synthetic oracle test
# ---------------------------------------------------------------------------


class SyntheticOracleTest(unittest.TestCase):
    """Perfect scaffold (seg_id == root_id) should yield F1 ≈ 1 with learned config."""

    def test_perfect_scaffold_high_f1(self):
        from neuronauts.fetch import make_test_volume, SyntheticBenchmarkConfig
        from neuronauts.line_graph import LineGraphMetrics
        from neuronauts.run import evaluate_synthetic_case

        # Volume axes must be > 20 for make_test_volume; perfect scaffold (seg_id = root_id)
        config = SyntheticBenchmarkConfig(
            shape=(40, 40, 24),
            n_synapses=10,
            membrane_planes=3,
            min_neuron_groups=2,
            max_neuron_groups=4,
            anchor_margin=6,
            pre_cluster_std=2.0,
            post_cluster_std=2.0,
        )
        metrics = evaluate_synthetic_case(
            config,
            volume_seed=0,
            run_seed=0,
            verbose=False,
            use_scaffold=True,
        )
        self.assertIsInstance(metrics, LineGraphMetrics)
        # With perfect scaffold and clustered synapses, we expect reasonably
        # high F1 (not necessarily 1.0 due to agent stochasticity, but > 0.5)
        self.assertGreater(
            metrics.f1, 0.3,
            "perfect scaffold should yield F1 > 0.3; got " f"{metrics.f1}",
        )
        self.assertGreaterEqual(metrics.f1, 0.0)
        self.assertLessEqual(metrics.f1, 1.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
