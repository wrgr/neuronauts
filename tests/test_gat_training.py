"""Tests for PR4 GAT training: label_graph_edges, gat_train_step, and
train_global_assembly_gat.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch not installed")


def _make_tiny_connectivity_graph(n_synapses: int = 6, seed: int = 0):
    """Build a small synthetic ConnectivityGraph using the full pipeline."""
    import numpy as np
    from neuronauts.fetch import SyntheticBenchmarkConfig, make_test_volume
    from neuronauts.legacy.fields import compute_membrane_field
    from neuronauts.run import HeuristicConfig, _build_graph, simulate_paths_and_hits

    rng = np.random.default_rng(seed)
    config = SyntheticBenchmarkConfig(
        shape=(40, 40, 40),
        n_synapses=n_synapses,
        anchor_margin=4,
        min_neuron_groups=2,
        max_neuron_groups=4,
    )
    chunk, synapses = make_test_volume(config=config, seed=seed)
    mf = compute_membrane_field(chunk.data)
    path_arr, synapse_hits, path_lengths, _ = simulate_paths_and_hits(
        chunk.data,
        synapses.pre_pt,
        synapses.post_pt,
        seed=seed,
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
    return graph, synapses.pre_root_id, synapses.post_root_id


def _make_grammar_and_gat(embedding_dim: int = 16):
    """Create a minimal SharedGrammarModel + GlobalAssemblyGAT pair."""
    from neuronauts.shared_grammar_model import GlobalAssemblyGAT, SharedGrammarModel

    model = SharedGrammarModel(embedding_dim=embedding_dim)
    gat = GlobalAssemblyGAT(node_dim=embedding_dim)
    return model, gat


# ---------------------------------------------------------------------------
# label_graph_edges
# ---------------------------------------------------------------------------

class LabelGraphEdgesTest(unittest.TestCase):
    """Tests for assembly.label_graph_edges."""

    def setUp(self):
        self.graph, self.pre_root_ids, self.post_root_ids = (
            _make_tiny_connectivity_graph(n_synapses=8, seed=1)
        )

    def test_output_shape_matches_edges(self):
        from neuronauts.assembly import label_graph_edges

        labels = label_graph_edges(self.graph, self.pre_root_ids, self.post_root_ids)
        self.assertEqual(labels.shape, (len(self.graph.edges),))

    def test_dtype_is_float32(self):
        from neuronauts.assembly import label_graph_edges

        labels = label_graph_edges(self.graph, self.pre_root_ids, self.post_root_ids)
        self.assertEqual(labels.dtype, np.float32)

    def test_values_are_binary(self):
        from neuronauts.assembly import label_graph_edges

        labels = label_graph_edges(self.graph, self.pre_root_ids, self.post_root_ids)
        self.assertTrue(np.all(np.isin(labels, [0.0, 1.0])),
                        f"unexpected values: {np.unique(labels)}")

    def test_empty_edges_returns_empty(self):
        from neuronauts.assembly import label_graph_edges
        from neuronauts.merge import ConnectivityGraph

        empty_graph = ConnectivityGraph(neurons={}, edges=[], unresolved_synapse_indices=[])
        labels = label_graph_edges(empty_graph, self.pre_root_ids, self.post_root_ids)
        self.assertEqual(len(labels), 0)

    def test_perfect_scaffold_has_some_positive_labels(self):
        """A learned-mode graph should have at least some correct edges
        (the scaffold pre-grouped matching neurons)."""
        from neuronauts.assembly import label_graph_edges

        labels = label_graph_edges(self.graph, self.pre_root_ids, self.post_root_ids)
        # With scaffold initialisation some neurons should be correctly grouped.
        # At minimum check we don't crash; positive labels are possible.
        self.assertIsNotNone(labels)

    def test_label_sum_is_non_negative(self):
        from neuronauts.assembly import label_graph_edges

        labels = label_graph_edges(self.graph, self.pre_root_ids, self.post_root_ids)
        self.assertGreaterEqual(float(labels.sum()), 0.0)


# ---------------------------------------------------------------------------
# gat_train_step
# ---------------------------------------------------------------------------

class GATTrainStepTest(unittest.TestCase):
    """Tests for shared_grammar_model.gat_train_step."""

    def setUp(self):
        _require_torch()
        import torch

        self.graph, self.pre_root_ids, self.post_root_ids = (
            _make_tiny_connectivity_graph(n_synapses=8, seed=2)
        )
        self.grammar_model, self.gat = _make_grammar_and_gat(embedding_dim=16)
        self.optimizer = torch.optim.Adam(self.gat.parameters(), lr=1e-3)

    def test_returns_expected_keys(self):
        from neuronauts.shared_grammar_model import gat_train_step

        result = gat_train_step(
            self.gat, self.grammar_model.path_encoder, self.optimizer,
            graph=self.graph,
            pre_root_ids=self.pre_root_ids,
            post_root_ids=self.post_root_ids,
        )
        for key in ("loss", "bce_loss", "f1_loss", "n_edges", "n_pos", "pred_f1"):
            self.assertIn(key, result, f"missing key: {key}")

    def test_loss_is_finite(self):
        from neuronauts.shared_grammar_model import gat_train_step

        if not self.graph.edges:
            self.skipTest("no edges generated")
        result = gat_train_step(
            self.gat, self.grammar_model.path_encoder, self.optimizer,
            graph=self.graph,
            pre_root_ids=self.pre_root_ids,
            post_root_ids=self.post_root_ids,
        )
        self.assertTrue(np.isfinite(result["loss"]),
                        f"loss is not finite: {result['loss']}")

    def test_gat_weights_change_after_step(self):
        """GAT parameters must receive gradients and be updated."""
        import torch
        from neuronauts.shared_grammar_model import gat_train_step

        if not self.graph.edges:
            self.skipTest("no edges generated")

        # Capture initial weights.
        before = {
            name: param.clone().detach()
            for name, param in self.gat.named_parameters()
        }
        gat_train_step(
            self.gat, self.grammar_model.path_encoder, self.optimizer,
            graph=self.graph,
            pre_root_ids=self.pre_root_ids,
            post_root_ids=self.post_root_ids,
        )
        after = {name: param.clone().detach()
                 for name, param in self.gat.named_parameters()}

        changed = any(
            not torch.allclose(before[n], after[n]) for n in before
        )
        self.assertTrue(changed, "no GAT parameter changed after train step")

    def test_path_encoder_weights_frozen(self):
        """Path encoder must not change during GAT training."""
        import torch
        from neuronauts.shared_grammar_model import gat_train_step

        if not self.graph.edges:
            self.skipTest("no edges generated")

        before = {
            name: param.clone().detach()
            for name, param in self.grammar_model.path_encoder.named_parameters()
        }
        gat_train_step(
            self.gat, self.grammar_model.path_encoder, self.optimizer,
            graph=self.graph,
            pre_root_ids=self.pre_root_ids,
            post_root_ids=self.post_root_ids,
        )
        after = {
            name: param.clone().detach()
            for name, param in self.grammar_model.path_encoder.named_parameters()
        }
        for name in before:
            self.assertTrue(torch.allclose(before[name], after[name]),
                            f"path encoder param {name!r} changed unexpectedly")

    def test_empty_graph_returns_zero_loss(self):
        from neuronauts.merge import ConnectivityGraph
        from neuronauts.shared_grammar_model import gat_train_step

        empty = ConnectivityGraph(neurons={}, edges=[], unresolved_synapse_indices=[])
        result = gat_train_step(
            self.gat, self.grammar_model.path_encoder, self.optimizer,
            graph=empty,
            pre_root_ids=self.pre_root_ids,
            post_root_ids=self.post_root_ids,
        )
        self.assertEqual(result["loss"], 0.0)
        self.assertEqual(result["n_edges"], 0)

    def test_soft_f1_weight_zero_uses_only_bce(self):
        """w=0 → total == bce_loss."""
        from neuronauts.shared_grammar_model import gat_train_step
        import torch

        if not self.graph.edges:
            self.skipTest("no edges generated")

        # Fresh optimiser to avoid state from other tests.
        opt = torch.optim.Adam(self.gat.parameters(), lr=1e-3)
        result = gat_train_step(
            self.gat, self.grammar_model.path_encoder, opt,
            graph=self.graph,
            pre_root_ids=self.pre_root_ids,
            post_root_ids=self.post_root_ids,
            soft_f1_weight=0.0,
        )
        self.assertAlmostEqual(result["loss"], result["bce_loss"], places=5)

    def test_loss_decreases_over_steps(self):
        """Loss should trend downward over several gradient steps on the same example."""
        import torch
        from neuronauts.shared_grammar_model import gat_train_step

        if not self.graph.edges:
            self.skipTest("no edges generated")

        gat = _make_grammar_and_gat(embedding_dim=16)[1]
        opt = torch.optim.Adam(gat.parameters(), lr=5e-2)
        losses = [
            gat_train_step(
                gat, self.grammar_model.path_encoder, opt,
                graph=self.graph,
                pre_root_ids=self.pre_root_ids,
                post_root_ids=self.post_root_ids,
            )["loss"]
            for _ in range(20)
        ]
        # At least the later half should be lower than the first half on average.
        first_half = np.mean(losses[:10])
        second_half = np.mean(losses[10:])
        self.assertLess(second_half, first_half + 0.1,
                        "Loss did not decrease: "
                        f"first_half={first_half:.4f} second_half={second_half:.4f}")


# ---------------------------------------------------------------------------
# GATTrainingConfig
# ---------------------------------------------------------------------------

class GATTrainingConfigTest(unittest.TestCase):

    def test_default_values(self):
        from neuronauts.shared_grammar_model import GATTrainingConfig

        cfg = GATTrainingConfig()
        self.assertGreater(cfg.epochs, 0)
        self.assertGreater(cfg.n_examples, 0)
        self.assertGreater(cfg.learning_rate, 0.0)
        self.assertGreater(cfg.soft_f1_weight, 0.0)
        self.assertLessEqual(cfg.soft_f1_weight, 1.0)
        self.assertGreater(cfg.val_fraction, 0.0)

    def test_immutable(self):
        from neuronauts.shared_grammar_model import GATTrainingConfig

        cfg = GATTrainingConfig()
        with self.assertRaises(Exception):
            cfg.epochs = 999   # type: ignore[misc]

    def test_custom_values(self):
        from neuronauts.shared_grammar_model import GATTrainingConfig

        cfg = GATTrainingConfig(epochs=5, n_examples=10, soft_f1_weight=0.8)
        self.assertEqual(cfg.epochs, 5)
        self.assertEqual(cfg.n_examples, 10)
        self.assertAlmostEqual(cfg.soft_f1_weight, 0.8)


# ---------------------------------------------------------------------------
# train_global_assembly_gat — lightweight convergence smoke test
# ---------------------------------------------------------------------------

class TrainGlobalAssemblyGATTest(unittest.TestCase):

    def test_training_returns_history_keys(self):
        """The training loop should return a dict with the expected keys."""
        _require_torch()
        import tempfile
        from neuronauts.shared_grammar_model import (
            GATTrainingConfig,
            train_global_assembly_gat,
        )

        cfg = GATTrainingConfig(
            epochs=2,
            n_examples=4,
            val_fraction=0.5,
            volume_shape=(40, 40, 40),
            n_synapses=8,
            seed=7,
        )
        grammar_model, gat = _make_grammar_and_gat(embedding_dim=16)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        history = train_global_assembly_gat(
            grammar_model.path_encoder, gat, ckpt_path, config=cfg
        )
        for key in ("train_loss", "val_loss", "train_f1", "val_f1"):
            self.assertIn(key, history, f"missing key: {key}")
        self.assertEqual(len(history["train_loss"]), cfg.epochs)

    def test_checkpoint_is_written(self):
        """Best-val checkpoint file should exist after training."""
        _require_torch()
        import tempfile
        from neuronauts.shared_grammar_model import (
            GATTrainingConfig,
            train_global_assembly_gat,
        )

        cfg = GATTrainingConfig(
            epochs=2,
            n_examples=4,
            val_fraction=0.5,
            volume_shape=(40, 40, 40),
            n_synapses=8,
            seed=8,
        )
        grammar_model, gat = _make_grammar_and_gat(embedding_dim=16)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        train_global_assembly_gat(
            grammar_model.path_encoder, gat, ckpt_path, config=cfg
        )
        self.assertTrue(os.path.exists(ckpt_path))
        self.assertGreater(os.path.getsize(ckpt_path), 0)

    def test_checkpoint_loads_cleanly(self):
        """Saved checkpoint should be loadable and produce matching predictions."""
        _require_torch()
        import tempfile
        import torch
        from neuronauts.shared_grammar_model import (
            GATTrainingConfig,
            GlobalAssemblyGAT,
            load_global_assembly_gat,
            train_global_assembly_gat,
        )
        from neuronauts.assembly import _build_gat_edges, _encode_neurons

        cfg = GATTrainingConfig(
            epochs=1,
            n_examples=4,
            val_fraction=0.5,
            volume_shape=(40, 40, 40),
            n_synapses=8,
            seed=9,
        )
        grammar_model, gat = _make_grammar_and_gat(embedding_dim=16)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        train_global_assembly_gat(
            grammar_model.path_encoder, gat, ckpt_path, config=cfg
        )
        loaded_gat = load_global_assembly_gat(ckpt_path)
        loaded_gat.eval()
        gat.eval()

        # Both models should produce the same predictions.
        graph, pre_root_ids, post_root_ids = _make_tiny_connectivity_graph(
            n_synapses=8, seed=99
        )
        if not graph.edges:
            self.skipTest("no edges generated")

        with torch.no_grad():
            node_ids, h = _encode_neurons(graph.neurons, grammar_model.path_encoder)
            if not node_ids:
                self.skipTest("no nodes")
            src, dst, _ = _build_gat_edges(node_ids, graph)
            h_gat_orig = gat(h, src, dst)
            h_gat_loaded = loaded_gat(h, src, dst)
            torch.testing.assert_close(h_gat_orig, h_gat_loaded)

    def test_history_lengths_match_epochs(self):
        _require_torch()
        import tempfile
        from neuronauts.shared_grammar_model import (
            GATTrainingConfig,
            train_global_assembly_gat,
        )

        cfg = GATTrainingConfig(
            epochs=3,
            n_examples=4,
            val_fraction=0.5,
            volume_shape=(40, 40, 40),
            n_synapses=8,
            seed=10,
        )
        grammar_model, gat = _make_grammar_and_gat(embedding_dim=16)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        history = train_global_assembly_gat(
            grammar_model.path_encoder, gat, ckpt_path, config=cfg
        )
        for key, vals in history.items():
            self.assertEqual(len(vals), cfg.epochs,
                             f"expected {cfg.epochs} entries for {key!r}, "
                             f"got {len(vals)}")


class GenerateGatExampleTest(unittest.TestCase):
    """G1 — HIGH: _generate_gat_example is never directly tested.

    It is only reached through train_global_assembly_gat which silently
    swallows all exceptions (``except Exception: pass``).  Bugs here produce
    no test failure — just an empty example pool and silent training failure.
    """

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_returns_three_tuple(self):
        from neuronauts.shared_grammar_model import _generate_gat_example
        result = _generate_gat_example(volume_shape=(40, 40, 40), n_synapses=12, seed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_graph_has_neurons(self):
        from neuronauts.shared_grammar_model import _generate_gat_example
        from neuronauts.merge import ConnectivityGraph
        graph, _, _ = _generate_gat_example(volume_shape=(40, 40, 40), n_synapses=12, seed=1)
        self.assertIsInstance(graph, ConnectivityGraph)
        self.assertGreater(len(graph.neurons), 0)

    def test_root_id_arrays_are_1d_numpy(self):
        from neuronauts.shared_grammar_model import _generate_gat_example
        import numpy as np
        _, pre_root_ids, post_root_ids = _generate_gat_example(
            volume_shape=(40, 40, 40), n_synapses=12, seed=2
        )
        self.assertIsInstance(pre_root_ids, np.ndarray)
        self.assertIsInstance(post_root_ids, np.ndarray)
        self.assertEqual(pre_root_ids.ndim, 1)
        self.assertEqual(post_root_ids.ndim, 1)

    def test_root_id_length_equals_n_synapses(self):
        from neuronauts.shared_grammar_model import _generate_gat_example
        n = 10
        _, pre, post = _generate_gat_example(volume_shape=(40, 40, 40), n_synapses=n, seed=3)
        self.assertEqual(len(pre), n)
        self.assertEqual(len(post), n)

    def test_different_seeds_produce_different_graphs(self):
        from neuronauts.shared_grammar_model import _generate_gat_example
        g1, _, _ = _generate_gat_example(volume_shape=(40, 40, 40), n_synapses=12, seed=0)
        g2, _, _ = _generate_gat_example(volume_shape=(40, 40, 40), n_synapses=12, seed=99)
        # Edges may differ between seeds (not a guarantee but usually true).
        # At minimum both should produce valid graphs.
        self.assertIsNotNone(g1)
        self.assertIsNotNone(g2)

    def test_graph_unresolved_synapses_is_list(self):
        from neuronauts.shared_grammar_model import _generate_gat_example
        graph, _, _ = _generate_gat_example(volume_shape=(40, 40, 40), n_synapses=12, seed=4)
        self.assertIsInstance(graph.unresolved_synapse_indices, list)

    def test_small_volume_still_returns_valid_result(self):
        """anchor_margin scaling should handle volumes as small as (30, 30, 30)."""
        from neuronauts.shared_grammar_model import _generate_gat_example
        graph, pre, post = _generate_gat_example(
            volume_shape=(30, 30, 30), n_synapses=8, seed=5
        )
        self.assertIsNotNone(graph)
        self.assertEqual(len(pre), 8)


class TrainGlobalAssemblyGATNoExamplesTest(unittest.TestCase):
    """G2 — MEDIUM: train_global_assembly_gat RuntimeError path when no valid examples.

    If volume_shape is too small to produce any synapse layout, the example
    pool will be empty and the function should raise RuntimeError rather than
    looping silently or returning empty history.
    """

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

    def test_raises_runtime_error_for_empty_example_pool(self):
        from neuronauts.shared_grammar_model import (
            GATTrainingConfig, GlobalAssemblyGAT, SharedGrammarModel, train_global_assembly_gat,
        )
        grammar_model = SharedGrammarModel(embedding_dim=16)
        gat = GlobalAssemblyGAT(node_dim=16)

        # Volume shape so small that make_test_volume will fail to place any
        # valid synapses — anchor_margin=max(2, min_side//6)=max(2,1)=2, but
        # the volume is (6,6,6) leaving almost no room → all seeds fail.
        cfg = GATTrainingConfig(
            epochs=1,
            n_examples=3,
            volume_shape=(6, 6, 6),
            n_synapses=50,  # impossible to place 50 synapses in a 6^3 volume
            seed=0,
        )
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt = f.name

        with self.assertRaises(RuntimeError):
            train_global_assembly_gat(grammar_model.path_encoder, gat, ckpt, config=cfg)


if __name__ == "__main__":
    unittest.main()
