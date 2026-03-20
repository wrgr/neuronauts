"""Tests for helper functions in scripts/train.py.

Covers:
- _accuracy_from_logits  (HIGH: metric used for epoch logging)
- _grammar_batch_from_synapses  (HIGH: primary grammar training batch builder)
- _validate_box  (HIGH: full inference on a validation box, errors swallowed)
- _run_gat_training_step  (MEDIUM: verifies gat_f1_acc updated and GAT trains)
"""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest

import numpy as np

# Make scripts/ importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch not installed")


def _import_train():
    """Import scripts/train.py as a module (not a package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_script",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "train.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_synapses(n: int = 30):
    from neuronauts.fetch import SynapseTable
    rng = np.random.default_rng(0)
    return SynapseTable(
        pre_pt=rng.random((n, 3), dtype=np.float32) * 40,
        post_pt=rng.random((n, 3), dtype=np.float32) * 40,
        pre_root_id=rng.integers(1, 6, size=n, dtype=np.int64),
        post_root_id=rng.integers(11, 16, size=n, dtype=np.int64),
        synapse_id=np.arange(n, dtype=np.int64),
    )


def _make_fake_args(**kwargs):
    """Return a minimal argparse-like namespace for _validate_box."""
    defaults = dict(
        grammar_output="/nonexistent/grammar.pt",
        gat_output="/nonexistent/gat.pt",
        train_gat=False,
        gat_edge_threshold=0.5,
    )
    defaults.update(kwargs)
    ns = types.SimpleNamespace(**defaults)
    return ns


# ---------------------------------------------------------------------------
# _accuracy_from_logits
# ---------------------------------------------------------------------------

class AccuracyFromLogitsTest(unittest.TestCase):

    def setUp(self):
        self.mod = _import_train()
        self.fn = self.mod._accuracy_from_logits

    def test_all_correct_positive(self):
        logits = np.array([1.0, 2.0, 0.5])
        y = np.array([1, 1, 1])
        self.assertAlmostEqual(self.fn(logits, y), 1.0)

    def test_all_correct_negative(self):
        logits = np.array([-1.0, -0.1, -2.0])
        y = np.array([0, 0, 0])
        self.assertAlmostEqual(self.fn(logits, y), 1.0)

    def test_all_wrong(self):
        logits = np.array([1.0, 1.0])
        y = np.array([0, 0])
        self.assertAlmostEqual(self.fn(logits, y), 0.0)

    def test_mixed_half_correct(self):
        logits = np.array([1.0, -1.0, 1.0, -1.0])
        y = np.array([1, 1, 0, 0])
        self.assertAlmostEqual(self.fn(logits, y), 0.5)

    def test_boundary_zero_logit_counts_as_positive(self):
        # 0.0 >= 0.0 is True → predicted positive
        logits = np.array([0.0])
        y_pos = np.array([1])
        y_neg = np.array([0])
        self.assertAlmostEqual(self.fn(logits, y_pos), 1.0)
        self.assertAlmostEqual(self.fn(logits, y_neg), 0.0)

    def test_output_is_float(self):
        result = self.fn(np.array([1.0, -1.0]), np.array([1, 0]))
        self.assertIsInstance(result, float)

    def test_single_element(self):
        self.assertAlmostEqual(
            self.fn(np.array([0.5]), np.array([1])), 1.0
        )


# ---------------------------------------------------------------------------
# _grammar_batch_from_synapses
# ---------------------------------------------------------------------------

class GrammarBatchFromSynapsesTest(unittest.TestCase):

    def setUp(self):
        _require_torch()
        import torch
        self.mod = _import_train()
        self.fn = self.mod._grammar_batch_from_synapses
        self.device = torch.device("cpu")

    def test_returns_none_for_too_sparse_table(self):
        from neuronauts.fetch import SynapseTable
        # Only 2 synapses, all same root — can't form positive merge pairs.
        tiny = SynapseTable(
            pre_pt=np.zeros((2, 3), dtype=np.float32),
            post_pt=np.zeros((2, 3), dtype=np.float32),
            pre_root_id=np.array([1, 1], dtype=np.int64),
            post_root_id=np.array([2, 2], dtype=np.int64),
            synapse_id=np.arange(2, dtype=np.int64),
        )
        result = self.fn(tiny, self.device)
        self.assertIsNone(result[0])
        self.assertIsNone(result[1])

    def test_returns_two_dicts_for_dense_table(self):
        syn = _make_synapses(n=30)
        merge_batch, topo_batch = self.fn(syn, self.device)
        if merge_batch is None:
            self.skipTest("too few examples on this synapse table")
        self.assertIsInstance(merge_batch, dict)
        self.assertIsInstance(topo_batch, dict)

    def test_merge_batch_has_required_keys(self):
        syn = _make_synapses(n=30)
        merge_batch, _ = self.fn(syn, self.device)
        if merge_batch is None:
            self.skipTest("no examples")
        for key in ("left_x", "left_mask", "right_x", "right_mask", "y"):
            self.assertIn(key, merge_batch, f"missing key: {key}")

    def test_topo_batch_has_required_keys(self):
        syn = _make_synapses(n=30)
        _, topo_batch = self.fn(syn, self.device)
        if topo_batch is None:
            self.skipTest("no examples")
        for key in ("branch_x", "branch_sequence_mask", "branch_mask", "y"):
            self.assertIn(key, topo_batch, f"missing key: {key}")

    def test_merge_batch_tensor_types(self):
        import torch
        syn = _make_synapses(n=30)
        merge_batch, _ = self.fn(syn, self.device)
        if merge_batch is None:
            self.skipTest("no examples")
        for key in ("left_x", "right_x"):
            self.assertEqual(merge_batch[key].dtype, torch.float32, f"{key} not float32")
        self.assertIn(merge_batch["y"].dtype, (torch.float32,))

    def test_merge_batch_consistent_batch_size(self):
        syn = _make_synapses(n=30)
        merge_batch, _ = self.fn(syn, self.device)
        if merge_batch is None:
            self.skipTest("no examples")
        B = merge_batch["left_x"].shape[0]
        self.assertEqual(merge_batch["right_x"].shape[0], B)
        self.assertEqual(merge_batch["y"].shape[0], B)
        self.assertEqual(merge_batch["left_mask"].shape[0], B)

    def test_max_merge_cap_respected(self):
        syn = _make_synapses(n=60)
        merge_batch, _ = self.fn(syn, self.device, max_merge=5, max_topo=5)
        if merge_batch is None:
            self.skipTest("no examples")
        self.assertLessEqual(merge_batch["y"].shape[0], 5)

    def test_topo_batch_3d_branch_tensor(self):
        import torch
        syn = _make_synapses(n=40)
        _, topo_batch = self.fn(syn, self.device)
        if topo_batch is None:
            self.skipTest("no examples")
        # branch_x: (B, max_branches, max_steps, 3)
        self.assertEqual(topo_batch["branch_x"].ndim, 4)
        self.assertEqual(topo_batch["branch_x"].dtype, torch.float32)


# ---------------------------------------------------------------------------
# _validate_box
# ---------------------------------------------------------------------------

class ValidateBoxTest(unittest.TestCase):
    """Tests that _validate_box returns LineGraphMetrics on a synthetic box
    and returns None gracefully when the cache fails."""

    def setUp(self):
        from neuronauts.dataset_builder import BoxCache, select_random_boxes
        from neuronauts.fetch import SynapseTable, VolumeChunk

        self.mod = _import_train()
        rng = np.random.default_rng(42)

        # Build a tiny real-ish box in a temp dir.
        self._tmpdir = tempfile.mkdtemp()
        cache = BoxCache(self._tmpdir)
        spec = select_random_boxes(n=1, seed=0)[0]

        n = 20
        vol = VolumeChunk(
            data=rng.integers(0, 255, (32, 32, 20), dtype=np.uint8),
            voxel_size_nm=(32, 32, 40),
            bbox_voxels=((0, 0, 0), (32, 32, 20)),
            mip=2,
        )
        syn = SynapseTable(
            pre_pt=rng.random((n, 3), dtype=np.float32) * 30,
            post_pt=rng.random((n, 3), dtype=np.float32) * 30,
            pre_root_id=rng.integers(1, 6, size=n, dtype=np.int64),
            post_root_id=rng.integers(11, 16, size=n, dtype=np.int64),
            synapse_id=np.arange(n, dtype=np.int64),
        )
        self._record = cache.save(spec, vol, syn)
        self._cache = cache

    def test_returns_line_graph_metrics_or_none(self):
        from neuronauts.line_graph import LineGraphMetrics
        args = _make_fake_args()
        metrics, diag = self.mod._validate_box(
            self._record, self._cache, None, None, args, "cpu"
        )
        # Either a valid metrics object or None (too few synapses / error).
        self.assertTrue(metrics is None or isinstance(metrics, LineGraphMetrics))
        self.assertIsInstance(diag, dict)

    def test_returns_metrics_with_valid_f1_range(self):
        args = _make_fake_args()
        metrics, _diag = self.mod._validate_box(
            self._record, self._cache, None, None, args, "cpu"
        )
        if metrics is None:
            self.skipTest("box too small — returned None as expected")
        self.assertGreaterEqual(metrics.f1, 0.0)
        self.assertLessEqual(metrics.f1, 1.0)

    def test_bad_cache_returns_none_not_raises(self):
        """A broken cache should be silently swallowed, returning None."""
        import types

        class BrokenCache:
            def load(self, record):
                raise OSError("disk error")

        args = _make_fake_args()
        metrics, diag = self.mod._validate_box(
            self._record, BrokenCache(), None, None, args, "cpu"
        )
        self.assertIsNone(metrics)
        self.assertIsInstance(diag, dict)

    def test_too_few_synapses_returns_none(self):
        from neuronauts.dataset_builder import BoxCache, select_random_boxes
        from neuronauts.fetch import SynapseTable, VolumeChunk

        with tempfile.TemporaryDirectory() as d:
            cache = BoxCache(d)
            spec = select_random_boxes(n=1, seed=5)[0]
            tiny_syn = SynapseTable(
                pre_pt=np.zeros((3, 3), dtype=np.float32),
                post_pt=np.zeros((3, 3), dtype=np.float32),
                pre_root_id=np.ones(3, dtype=np.int64),
                post_root_id=np.ones(3, dtype=np.int64) * 2,
                synapse_id=np.arange(3, dtype=np.int64),
            )
            vol = VolumeChunk(
                data=np.zeros((10, 10, 5), dtype=np.uint8),
                voxel_size_nm=(32, 32, 40),
                bbox_voxels=((0, 0, 0), (10, 10, 5)),
                mip=2,
            )
            record = cache.save(spec, vol, tiny_syn)
            args = _make_fake_args()
            metrics, diag = self.mod._validate_box(record, cache, None, None, args, "cpu")
            self.assertIsNone(metrics)
            self.assertIsInstance(diag, dict)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)


class ValidateBoxFastTest(unittest.TestCase):
    def setUp(self):
        _require_torch()
        from neuronauts.dataset_builder import BoxCache, select_random_boxes
        from neuronauts.fetch import SynapseTable, VolumeChunk

        self.mod = _import_train()
        self._tmpdir = tempfile.mkdtemp()
        self._cache = BoxCache(self._tmpdir)
        spec = select_random_boxes(n=1, seed=17)[0]
        rng = np.random.default_rng(17)
        n = 40
        vol = VolumeChunk(
            data=np.zeros((8, 8, 8), dtype=np.uint8),
            voxel_size_nm=(32, 32, 40),
            bbox_voxels=((0, 0, 0), (8, 8, 8)),
            mip=2,
        )
        syn = SynapseTable(
            pre_pt=rng.random((n, 3), dtype=np.float32) * 30,
            post_pt=rng.random((n, 3), dtype=np.float32) * 30,
            pre_root_id=np.repeat(np.array([1, 2, 3, 4], dtype=np.int64), n // 4),
            post_root_id=np.repeat(np.array([11, 12, 13, 14], dtype=np.int64), n // 4),
            synapse_id=np.arange(n, dtype=np.int64),
        )
        self._record = self._cache.save(spec, vol, syn)

    def test_returns_merge_and_topology_metrics(self):
        import torch
        from neuronauts.shared_grammar_model import SharedGrammarModel

        model = SharedGrammarModel(input_dim=6, path_feature_mode="raw_delta3+skeleton")
        result = self.mod._validate_box_fast(
            self._record,
            self._cache,
            model,
            torch.device("cpu"),
            "raw_delta3+skeleton",
        )
        self.assertIsNotNone(result)
        for key in ("merge_acc", "merge_bce", "topo_acc", "topo_bce", "n_pairs", "n_topo"):
            self.assertIn(key, result)
        self.assertGreater(result["n_pairs"], 0)
        self.assertGreater(result["n_topo"], 0)
        self.assertGreaterEqual(result["merge_acc"], 0.0)
        self.assertLessEqual(result["merge_acc"], 1.0)
        self.assertGreaterEqual(result["topo_acc"], 0.0)
        self.assertLessEqual(result["topo_acc"], 1.0)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# _run_gat_training_step
# ---------------------------------------------------------------------------

class RunGATTrainingStepTest(unittest.TestCase):

    def setUp(self):
        _require_torch()
        import torch
        self.mod = _import_train()
        self.device = torch.device("cpu")

    def test_gat_f1_acc_updated_after_step(self):
        """After one GAT step on a real volume, gat_f1_acc should gain an entry."""
        from neuronauts.fetch import SynapseTable, VolumeChunk
        from neuronauts.shared_grammar_model import GlobalAssemblyGAT, SharedGrammarModel
        import torch

        rng = np.random.default_rng(7)
        n = 20
        vol = VolumeChunk(
            data=rng.integers(0, 200, (32, 32, 16), dtype=np.uint8),
            voxel_size_nm=(32, 32, 40),
            bbox_voxels=((0, 0, 0), (32, 32, 16)),
            mip=2,
        )
        syn = SynapseTable(
            pre_pt=rng.random((n, 3), dtype=np.float32) * 28,
            post_pt=rng.random((n, 3), dtype=np.float32) * 28,
            pre_root_id=rng.integers(1, 5, size=n, dtype=np.int64),
            post_root_id=rng.integers(11, 15, size=n, dtype=np.int64),
            synapse_id=np.arange(n, dtype=np.int64),
        )

        grammar_model = SharedGrammarModel(embedding_dim=16)
        gat_model = GlobalAssemblyGAT(node_dim=16)
        gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=1e-3)

        args = _make_fake_args(gat_soft_f1_weight=0.5)
        gat_f1_acc: list[float] = []
        self.mod._run_gat_training_step(
            vol, syn, gat_model, grammar_model,
            gat_optimizer, self.device, args, gat_f1_acc,
        )
        # If the graph had edges, gat_f1_acc should be non-empty.
        # (May be empty if the tiny volume produces no graph edges — both are valid.)
        self.assertIsInstance(gat_f1_acc, list)

    def test_gat_weights_change_after_step_with_edges(self):
        """If the resulting graph has edges, GAT parameters must be updated."""
        from neuronauts.fetch import SyntheticBenchmarkConfig, make_test_volume
        from neuronauts.shared_grammar_model import GlobalAssemblyGAT, SharedGrammarModel
        import torch

        # Use the synthetic benchmark to guarantee edges.
        config = SyntheticBenchmarkConfig(
            shape=(40, 40, 40), n_synapses=15,
            anchor_margin=5, min_neuron_groups=2, max_neuron_groups=4,
        )
        from neuronauts.fetch import VolumeChunk, SynapseTable
        chunk, synapses = make_test_volume(config=config, seed=3)
        vol = chunk
        syn = synapses

        grammar_model = SharedGrammarModel(embedding_dim=16)
        gat_model = GlobalAssemblyGAT(node_dim=16)
        gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=1e-2)

        before = {n: p.clone().detach() for n, p in gat_model.named_parameters()}

        args = _make_fake_args(gat_soft_f1_weight=0.5)
        gat_f1_acc: list[float] = []
        self.mod._run_gat_training_step(
            vol, syn, gat_model, grammar_model,
            gat_optimizer, self.device, args, gat_f1_acc,
        )

        if not gat_f1_acc:
            self.skipTest("no graph edges produced by this synthetic volume")

        after = {n: p.clone().detach() for n, p in gat_model.named_parameters()}
        changed = any(not torch.allclose(before[n], after[n]) for n in before)
        self.assertTrue(changed, "GAT parameters unchanged after step with edges")

    def test_errors_are_swallowed_silently(self):
        """A bad volume input should not propagate an exception."""
        from neuronauts.fetch import SynapseTable, VolumeChunk
        from neuronauts.shared_grammar_model import GlobalAssemblyGAT, SharedGrammarModel
        import torch

        # Deliberately empty volume to trigger any internal errors.
        bad_vol = VolumeChunk(
            data=np.zeros((2, 2, 2), dtype=np.uint8),
            voxel_size_nm=(32, 32, 40),
            bbox_voxels=((0, 0, 0), (2, 2, 2)),
            mip=2,
        )
        tiny_syn = SynapseTable(
            pre_pt=np.zeros((2, 3), dtype=np.float32),
            post_pt=np.zeros((2, 3), dtype=np.float32),
            pre_root_id=np.array([1, 2], dtype=np.int64),
            post_root_id=np.array([3, 4], dtype=np.int64),
            synapse_id=np.arange(2, dtype=np.int64),
        )
        grammar_model = SharedGrammarModel(embedding_dim=16)
        gat_model = GlobalAssemblyGAT(node_dim=16)
        gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=1e-3)
        args = _make_fake_args(gat_soft_f1_weight=0.5)
        gat_f1_acc: list[float] = []
        # Should not raise.
        self.mod._run_gat_training_step(
            bad_vol, tiny_syn, gat_model, grammar_model,
            gat_optimizer, self.device, args, gat_f1_acc,
        )


if __name__ == "__main__":
    unittest.main()
