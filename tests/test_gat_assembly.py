"""Tests for PR 4: GlobalAssemblyGAT and gat_refine_connectivity."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from neuronauts.shared_grammar_model import (
    GlobalAssemblyGAT,
    SharedGrammarModel,
    load_global_assembly_gat,
    save_global_assembly_gat,
)
from neuronauts.assembly import (
    _build_gat_edges,
    _encode_neurons,
    _path_seq_from_pts,
    gat_refine_connectivity,
)
from neuronauts.merge import ConnectivityGraph, MergedNeuron


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng_pts(n=20, seed=0):
    return np.random.default_rng(seed).random((n, 3)).astype(np.float32) * 30


def _make_neuron(nid, pts, role="pre"):
    return MergedNeuron(
        neuron_id=nid,
        agent_ids=[nid],
        path_points=pts,
        synapse_indices=[],
        role=role,
    )


def _make_graph(n_pre=3, n_post=3, seed=0):
    """Build a small ConnectivityGraph with n_pre + n_post neurons and a few edges."""
    rng = np.random.default_rng(seed)
    neurons = {}
    for i in range(n_pre):
        neurons[i] = _make_neuron(i, rng.random((15, 3)).astype(np.float32), "pre")
    for i in range(n_post):
        neurons[n_pre + i] = _make_neuron(n_pre + i, rng.random((15, 3)).astype(np.float32), "post")
    # One edge per pre-post pair (synapse index = 0..n_pre-1).
    edges = [(i, n_pre + i, i) for i in range(min(n_pre, n_post))]
    return ConnectivityGraph(neurons=neurons, edges=edges, unresolved_synapse_indices=[])


# ---------------------------------------------------------------------------
# _path_seq_from_pts
# ---------------------------------------------------------------------------

class PathSeqFromPtsTest(unittest.TestCase):
    def test_output_shape(self):
        pts = _rng_pts(10)
        seq = _path_seq_from_pts(pts)
        self.assertEqual(seq.shape, (9, 6))

    def test_single_point_returns_empty(self):
        seq = _path_seq_from_pts(np.zeros((1, 3), dtype=np.float32))
        self.assertEqual(seq.shape[0], 0)

    def test_two_points(self):
        # Points are in MIP-2 voxel coords. Features are isotropic raw deltas:
        # a 1-voxel step in X = (1, 0, 0), a 1-voxel step in Z = (0, 0, 1.25).
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        seq = _path_seq_from_pts(pts)
        self.assertEqual(seq.shape, (1, 6))
        np.testing.assert_allclose(seq[0, :3], [1.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(seq[0, 3:], [1.0, 0.0, 0.0], atol=1e-4)

    def test_z_step_is_longer_than_xy(self):
        # A 1-voxel Z step should produce dz=1.25 units (40/32), not 1.0.
        pts_x = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        pts_z = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float32)
        delta_x = _path_seq_from_pts(pts_x)[0]
        delta_z = _path_seq_from_pts(pts_z)[0]
        np.testing.assert_allclose(delta_x[:3], [1.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(delta_z[:3], [0.0, 0.0, 40.0 / 32.0], atol=1e-4)

    def test_dtype_float32(self):
        seq = _path_seq_from_pts(_rng_pts(5))
        self.assertEqual(seq.dtype, np.float32)


# ---------------------------------------------------------------------------
# _SparseGATLayer (through GlobalAssemblyGAT internals)
# ---------------------------------------------------------------------------

class SparseGATLayerTest(unittest.TestCase):
    def _layer(self, in_dim=16, out_dim=16, n_heads=2):
        from neuronauts.shared_grammar_model import _SparseGATLayer
        return _SparseGATLayer(in_dim, out_dim, n_heads)

    def test_output_shape(self):
        layer = self._layer()
        h = torch.randn(5, 16)
        src = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        dst = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        out = layer(h, src, dst)
        self.assertEqual(tuple(out.shape), (5, 16))

    def test_output_finite(self):
        layer = self._layer()
        h = torch.randn(8, 16)
        src = torch.arange(7, dtype=torch.long)
        dst = torch.arange(1, 8, dtype=torch.long)
        out = layer(h, src, dst)
        self.assertTrue(torch.isfinite(out).all())

    def test_self_loop_only(self):
        # When only self-loops exist, each node attends only to itself.
        layer = self._layer(in_dim=8, out_dim=8, n_heads=2).eval()
        N = 4
        h = torch.randn(N, 8)
        src = dst = torch.arange(N, dtype=torch.long)
        with torch.no_grad():
            out = layer(h, src, dst)
        self.assertEqual(tuple(out.shape), (N, 8))


# ---------------------------------------------------------------------------
# GlobalAssemblyGAT
# ---------------------------------------------------------------------------

class GlobalAssemblyGATTest(unittest.TestCase):
    def _model(self, **kw):
        defaults = dict(node_dim=16, gat_dim=16, n_heads=2, n_layers=2, dropout=0.0)
        defaults.update(kw)
        return GlobalAssemblyGAT(**defaults).eval()

    def test_forward_output_shape(self):
        model = self._model()
        x = torch.randn(6, 16)
        src = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        dst = torch.tensor([1, 2, 3, 4, 5, 0], dtype=torch.long)
        with torch.no_grad():
            h = model(x, src, dst)
        self.assertEqual(tuple(h.shape), (6, 16))

    def test_score_edges_output_shape(self):
        model = self._model()
        h = torch.randn(6, 16)
        src = torch.tensor([0, 1, 2], dtype=torch.long)
        dst = torch.tensor([3, 4, 5], dtype=torch.long)
        with torch.no_grad():
            logits = model.score_edges(h, src, dst)
        self.assertEqual(tuple(logits.shape), (3,))

    def test_forward_is_finite(self):
        model = self._model()
        x = torch.randn(4, 16)
        src = dst = torch.arange(4, dtype=torch.long)
        with torch.no_grad():
            h = model(x, src, dst)
        self.assertTrue(torch.isfinite(h).all())

    def test_single_node_graph(self):
        model = self._model()
        x = torch.randn(1, 16)
        src = dst = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            h = model(x, src, dst)
        self.assertEqual(tuple(h.shape), (1, 16))

    def test_init_kwargs_stored(self):
        model = self._model(node_dim=8, gat_dim=8)
        self.assertEqual(model._init_kwargs["node_dim"], 8)
        self.assertEqual(model._init_kwargs["gat_dim"], 8)

    def test_deterministic_in_eval(self):
        model = self._model()
        x = torch.randn(5, 16)
        src = torch.tensor([0, 1, 2], dtype=torch.long)
        dst = torch.tensor([1, 2, 3], dtype=torch.long)
        with torch.no_grad():
            h1 = model(x, src, dst)
            h2 = model(x, src, dst)
        torch.testing.assert_close(h1, h2)


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------

class GATCheckpointTest(unittest.TestCase):
    def test_round_trip_predictions(self):
        model = GlobalAssemblyGAT(node_dim=8, gat_dim=8, n_heads=2, n_layers=1, dropout=0.0).eval()
        x = torch.randn(4, 8)
        src = torch.tensor([0, 1, 2], dtype=torch.long)
        dst = torch.tensor([1, 2, 3], dtype=torch.long)

        with torch.no_grad():
            h_orig = model(x, src, dst)
            logits_orig = model.score_edges(h_orig, src, dst)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "gat.pt"
            save_global_assembly_gat(ckpt, model)
            loaded = load_global_assembly_gat(ckpt)

        with torch.no_grad():
            h_load = loaded(x, src, dst)
            logits_load = loaded.score_edges(h_load, src, dst)

        torch.testing.assert_close(h_orig, h_load)
        torch.testing.assert_close(logits_orig, logits_load)

    def test_loaded_model_in_eval_mode(self):
        model = GlobalAssemblyGAT(node_dim=8, gat_dim=8, n_heads=2).eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "gat.pt"
            save_global_assembly_gat(ckpt, model)
            loaded = load_global_assembly_gat(ckpt)
        self.assertFalse(loaded.training)


# ---------------------------------------------------------------------------
# _encode_neurons
# ---------------------------------------------------------------------------

class EncodeNeuronsTest(unittest.TestCase):
    def _encoder(self):
        return SharedGrammarModel(input_dim=6, path_d_model=16, embedding_dim=16,
                                  path_n_heads=2).path_encoder.eval()

    def test_returns_correct_node_count(self):
        enc = self._encoder()
        neurons = {i: _make_neuron(i, _rng_pts(12, seed=i)) for i in range(4)}
        node_ids, h = _encode_neurons(neurons, enc)
        self.assertEqual(len(node_ids), 4)
        self.assertEqual(h.shape[0], 4)

    def test_embedding_is_finite(self):
        enc = self._encoder()
        neurons = {0: _make_neuron(0, _rng_pts(10))}
        _, h = _encode_neurons(neurons, enc)
        self.assertTrue(torch.isfinite(h).all())

    def test_empty_neurons(self):
        enc = self._encoder()
        node_ids, h = _encode_neurons({}, enc)
        self.assertEqual(len(node_ids), 0)


# ---------------------------------------------------------------------------
# _build_gat_edges
# ---------------------------------------------------------------------------

class BuildGATEdgesTest(unittest.TestCase):
    def test_self_loops_always_present(self):
        graph = _make_graph(n_pre=2, n_post=2)
        node_ids = sorted(graph.neurons.keys())
        src, dst, pairs = _build_gat_edges(node_ids, graph)
        self_loops = [(i, i) for i in range(len(node_ids))]
        for sl in self_loops:
            self.assertIn(sl, pairs)

    def test_edges_are_bidirectional(self):
        graph = _make_graph(n_pre=2, n_post=2)
        node_ids = sorted(graph.neurons.keys())
        src, dst, pairs = _build_gat_edges(node_ids, graph)
        pair_set = set(pairs)
        for s, d in list(pair_set):
            if s != d:
                self.assertIn((d, s), pair_set)

    def test_empty_graph_returns_empty_tensors(self):
        graph = ConnectivityGraph(neurons={}, edges=[], unresolved_synapse_indices=[])
        src, dst, pairs = _build_gat_edges([], graph)
        self.assertEqual(len(pairs), 0)


# ---------------------------------------------------------------------------
# gat_refine_connectivity
# ---------------------------------------------------------------------------

class GatRefineConnectivityTest(unittest.TestCase):
    def _enc_and_gat(self, node_dim=16):
        enc = SharedGrammarModel(input_dim=6, path_d_model=16, embedding_dim=node_dim,
                                 path_n_heads=2).path_encoder.eval()
        gat = GlobalAssemblyGAT(node_dim=node_dim, gat_dim=16, n_heads=2, n_layers=1,
                                 dropout=0.0).eval()
        return enc, gat

    def test_refined_graph_has_same_neurons(self):
        graph = _make_graph()
        enc, gat = self._enc_and_gat()
        refined = gat_refine_connectivity(graph, enc, gat, threshold=0.5)
        self.assertEqual(set(refined.neurons.keys()), set(graph.neurons.keys()))

    def test_refined_edges_subset_of_original(self):
        graph = _make_graph()
        enc, gat = self._enc_and_gat()
        refined = gat_refine_connectivity(graph, enc, gat, threshold=0.5)
        original_set = {(p, q, s) for p, q, s in graph.edges}
        refined_set = {(p, q, s) for p, q, s in refined.edges}
        self.assertTrue(refined_set.issubset(original_set))

    def test_threshold_0_keeps_all_edges(self):
        graph = _make_graph()
        enc, gat = self._enc_and_gat()
        refined = gat_refine_connectivity(graph, enc, gat, threshold=0.0)
        self.assertEqual(len(refined.edges), len(graph.edges))

    def test_threshold_1_drops_all_edges(self):
        graph = _make_graph()
        enc, gat = self._enc_and_gat()
        refined = gat_refine_connectivity(graph, enc, gat, threshold=1.0)
        self.assertEqual(len(refined.edges), 0)
        # All synapse indices should move to unresolved.
        dropped = {s for _, _, s in graph.edges}
        self.assertTrue(dropped.issubset(set(refined.unresolved_synapse_indices)))

    def test_empty_graph_returns_unchanged(self):
        empty = ConnectivityGraph(neurons={}, edges=[], unresolved_synapse_indices=[])
        enc, gat = self._enc_and_gat()
        result = gat_refine_connectivity(empty, enc, gat)
        self.assertEqual(result.edges, [])

    def test_no_edges_graph_returns_unchanged(self):
        neurons = {0: _make_neuron(0, _rng_pts(10)), 1: _make_neuron(1, _rng_pts(10, 1))}
        graph = ConnectivityGraph(neurons=neurons, edges=[], unresolved_synapse_indices=[3, 4])
        enc, gat = self._enc_and_gat()
        result = gat_refine_connectivity(graph, enc, gat)
        self.assertEqual(result.edges, [])
        self.assertIn(3, result.unresolved_synapse_indices)


# ---------------------------------------------------------------------------
# run() integration: gat_assembly_checkpoint kwarg
# ---------------------------------------------------------------------------

class RunGATIntegrationTest(unittest.TestCase):
    def test_run_with_gat_checkpoint(self):
        from neuronauts.fetch import SyntheticBenchmarkConfig, make_test_volume
        from neuronauts.legacy.run import run

        # Default SharedGrammarModel uses embedding_dim=32; GAT node_dim must match.
        enc = SharedGrammarModel(input_dim=6, path_d_model=32, embedding_dim=32,
                                 path_n_heads=2).path_encoder
        gat = GlobalAssemblyGAT(node_dim=32, gat_dim=32, n_heads=2, n_layers=1, dropout=0.0)

        chunk, synapses = make_test_volume(
            config=SyntheticBenchmarkConfig(n_synapses=8, shape=(32, 32, 32)),
            seed=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "gat.pt"
            save_global_assembly_gat(ckpt, gat)
            metrics = run(
                volume=chunk.data,
                pre_pts=synapses.pre_pt,
                post_pts=synapses.post_pt,
                pre_root_ids=synapses.pre_root_id,
                post_root_ids=synapses.post_root_id,
                verbose=False,
                gat_assembly_checkpoint=str(ckpt),
                gat_edge_threshold=0.5,
            )

        self.assertGreaterEqual(metrics.f1, 0.0)
        self.assertLessEqual(metrics.f1, 1.0)


if __name__ == "__main__":
    unittest.main()
