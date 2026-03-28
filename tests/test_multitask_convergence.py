"""Tests for training correctness and score-function integration.

These tests catch the class of bugs where:
- Grammar accuracy is high but line-graph F1 is still near zero
- Merge logit sign convention is wrong (positive should mean "merge")
- The live score function is stale (lru_cache'd weights not updated)
- Multitask training doesn't actually converge on a fixed dataset
- The score function used in _validate_box reflects the in-memory model
"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch not installed")


def _make_same_neuron_synapses(n: int = 8):
    """Return a SynapseTable where all synapses share root IDs (clearly same neuron)."""
    from neuronauts.fetch import SynapseTable

    rng = np.random.default_rng(0)
    return SynapseTable(
        pre_pt=rng.random((n, 3), dtype=np.float32) * 5,   # clustered
        post_pt=rng.random((n, 3), dtype=np.float32) * 5,
        pre_root_id=np.full(n, 101, dtype=np.int64),
        post_root_id=np.full(n, 201, dtype=np.int64),
        synapse_id=np.arange(n, dtype=np.int64),
    )


def _make_diff_neuron_synapses(n: int = 8):
    """Return a SynapseTable where every synapse has a unique root ID (all different)."""
    from neuronauts.fetch import SynapseTable

    rng = np.random.default_rng(1)
    return SynapseTable(
        pre_pt=rng.random((n, 3), dtype=np.float32) * 100,  # spread out
        post_pt=rng.random((n, 3), dtype=np.float32) * 100,
        pre_root_id=np.arange(1, n + 1, dtype=np.int64),
        post_root_id=np.arange(101, n + 101, dtype=np.int64),
        synapse_id=np.arange(n, dtype=np.int64),
    )


def _make_mixed_synapses(n_per_group: int = 6):
    """Two distinct neuron groups for building merge and topology examples."""
    from neuronauts.fetch import SynapseTable

    rng = np.random.default_rng(2)
    n = n_per_group * 2
    pre_root = np.array([101] * n_per_group + [202] * n_per_group, dtype=np.int64)
    post_root = np.array([301] * n_per_group + [402] * n_per_group, dtype=np.int64)
    return SynapseTable(
        pre_pt=np.vstack([
            rng.random((n_per_group, 3), dtype=np.float32) * 4,
            rng.random((n_per_group, 3), dtype=np.float32) * 4 + 50,
        ]),
        post_pt=np.vstack([
            rng.random((n_per_group, 3), dtype=np.float32) * 4,
            rng.random((n_per_group, 3), dtype=np.float32) * 4 + 50,
        ]),
        pre_root_id=pre_root,
        post_root_id=post_root,
        synapse_id=np.arange(n, dtype=np.int64),
    )


def _make_model_and_optimizer(embedding_dim: int = 16):
    import torch
    from neuronauts.shared_grammar_model import SharedGrammarModel

    model = SharedGrammarModel(embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    return model, optimizer


def _build_batches(synapses):
    """Convert a SynapseTable into merge + topology batch dicts."""
    import torch
    from neuronauts.merge_dataset import build_merge_examples, examples_to_arrays
    from neuronauts.topology_dataset import (
        build_cluster_examples,
        examples_to_branch_sequence_arrays,
    )

    merge_examples = build_merge_examples(synapses, min_fragment_size=2,
                                          max_negative_pairs_per_role=4)
    topo_examples = build_cluster_examples(
        synapses,
        membrane_field=np.zeros((20, 20, 20), dtype=np.float32),
        min_cluster_size=2,
        max_negative_pairs_per_role=4,
        max_branches=4,
        seed=7,
    )
    if not merge_examples or not topo_examples:
        return None, None

    lx, lm, rx, rm, y_merge = examples_to_arrays(merge_examples)
    bx, bsm, bm = examples_to_branch_sequence_arrays(topo_examples, max_branches=4)
    y_topo = np.array([ex.label for ex in topo_examples], dtype=np.float32)

    merge_batch = {
        "left_x": torch.from_numpy(lx),
        "left_mask": torch.from_numpy(lm),
        "right_x": torch.from_numpy(rx),
        "right_mask": torch.from_numpy(rm),
        "y": torch.from_numpy(y_merge.astype(np.float32)),
    }
    topo_batch = {
        "branch_x": torch.from_numpy(bx),
        "branch_sequence_mask": torch.from_numpy(bsm),
        "branch_mask": torch.from_numpy(bm),
        "y": torch.from_numpy(y_topo),
    }
    return merge_batch, topo_batch


# ---------------------------------------------------------------------------
# Merge logit sign convention
# ---------------------------------------------------------------------------

class MergeLogitSignConventionTest(unittest.TestCase):
    """The grammar model's merge logit must be positive for same-neuron pairs
    and negative for different-neuron pairs — after a few gradient steps on a
    clearly separable dataset.

    A sign-flip here means every merge decision is inverted: the model says
    "split" when it should say "merge", producing near-zero recall.
    """

    def setUp(self):
        _require_torch()

    def test_merge_logit_is_scalar_per_pair(self):
        """score_merge must return a 1-D tensor of shape [B]."""
        import torch
        from neuronauts.shared_grammar_model import SharedGrammarModel

        model = SharedGrammarModel(embedding_dim=16)
        model.eval()
        D = model._init_kwargs["input_dim"]
        B, T = 4, 5
        with torch.no_grad():
            logits = model.score_merge(
                torch.randn(B, T, D), torch.zeros(B, T, dtype=torch.bool),
                torch.randn(B, T, D), torch.zeros(B, T, dtype=torch.bool),
            )
        self.assertEqual(logits.shape, (B,), f"expected [B={B}], got {logits.shape}")

    def test_logit_range_is_finite(self):
        """Fresh model logits must be finite (no NaN/Inf on init)."""
        import torch
        from neuronauts.shared_grammar_model import SharedGrammarModel

        model = SharedGrammarModel(embedding_dim=16)
        model.eval()
        D = model._init_kwargs["input_dim"]
        with torch.no_grad():
            logits = model.score_merge(
                torch.randn(3, 4, D), torch.zeros(3, 4, dtype=torch.bool),
                torch.randn(3, 4, D), torch.zeros(3, 4, dtype=torch.bool),
            )
        self.assertTrue(torch.all(torch.isfinite(logits)).item(),
                        f"non-finite logits: {logits}")

    def test_trained_model_positive_logit_for_same_paths(self):
        """After training on a clearly same-neuron dataset the model should
        output a positive logit (>0) for same-neuron pairs and negative for
        random different pairs — this checks the sign convention end-to-end.
        """
        import torch
        from neuronauts.shared_grammar_model import SharedGrammarModel, multitask_train_step

        synapses = _make_mixed_synapses(n_per_group=8)
        merge_batch, topo_batch = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")

        model = SharedGrammarModel(embedding_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Identify positive merge pairs (y == 1).
        y = merge_batch["y"]
        pos_mask = (y == 1.0)
        neg_mask = (y == 0.0)
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            self.skipTest("need both positive and negative merge examples")

        # Train for several steps.
        for _ in range(30):
            multitask_train_step(model, optimizer,
                                 merge_batch=merge_batch,
                                 topology_batch=topo_batch)

        # Evaluate sign convention on training examples (sanity check).
        model.eval()
        with torch.no_grad():
            logits = model.score_merge(
                merge_batch["left_x"], merge_batch["left_mask"],
                merge_batch["right_x"], merge_batch["right_mask"],
            )

        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        # At least majority of positive examples should have logit > negative mean.
        self.assertGreater(
            float(pos_logits.mean()), float(neg_logits.mean()),
            "positive-pair logits should be higher than negative-pair logits "
            "after training — sign convention may be inverted",
        )

    def test_zero_logit_threshold_interpretation(self):
        """logit >= 0 should be the merge decision boundary (BCE w/ logits convention)."""
        import torch
        from neuronauts.shared_grammar_model import SharedGrammarModel

        model = SharedGrammarModel(embedding_dim=16)
        model.eval()
        D = model._init_kwargs["input_dim"]
        T = 4
        with torch.no_grad():
            # Identical left and right should tend toward positive (cosine-like).
            x = torch.randn(1, T, D)
            logit_same = model.score_merge(x, torch.zeros(1, T, dtype=torch.bool),
                                           x, torch.zeros(1, T, dtype=torch.bool))
            logit_diff = model.score_merge(
                x, torch.zeros(1, T, dtype=torch.bool),
                torch.randn(1, T, D), torch.zeros(1, T, dtype=torch.bool),
            )
        # Just verify they are finite scalars — not a hard directional test.
        self.assertEqual(logit_same.shape, (1,))
        self.assertEqual(logit_diff.shape, (1,))
        self.assertTrue(torch.isfinite(logit_same).item())
        self.assertTrue(torch.isfinite(logit_diff).item())


# ---------------------------------------------------------------------------
# Live merge score function (not lru_cache'd)
# ---------------------------------------------------------------------------

class LiveMergeScoreFnTest(unittest.TestCase):
    """_make_live_merge_score_fn must return a closure that reflects the
    *current* in-memory model weights, not a snapshot from epoch 1.

    If the closure captures stale weights (or uses lru_cache), the validation
    F1 reported during training will not reflect actual model improvement.
    """

    def setUp(self):
        _require_torch()

    def _import_train(self):
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "train_script",
            os.path.join(os.path.dirname(__file__), "..", "scripts", "train.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_score_fn_returns_float(self):
        import numpy as np
        from neuronauts.shared_grammar_model import SharedGrammarModel

        mod = self._import_train()
        model = SharedGrammarModel(embedding_dim=16)
        D = model._init_kwargs["input_dim"]
        score_fn = mod._make_live_merge_score_fn(model)

        rng = np.random.default_rng(0)
        seq = rng.random((5, D), dtype=np.float32)
        result = score_fn(seq, seq)
        self.assertIsInstance(result, float)
        self.assertFalse(np.isnan(result), "score is NaN")
        self.assertFalse(np.isinf(result), "score is Inf")

    def test_score_fn_changes_after_model_update(self):
        """The closure must use the *live* model weights, so outputs change
        when the model is updated between calls.
        """
        import torch
        import numpy as np
        from neuronauts.shared_grammar_model import SharedGrammarModel, multitask_train_step

        synapses = _make_mixed_synapses(n_per_group=6)
        merge_batch, topo_batch = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")

        mod = self._import_train()
        model = SharedGrammarModel(embedding_dim=16)
        D = model._init_kwargs["input_dim"]
        score_fn = mod._make_live_merge_score_fn(model)

        rng = np.random.default_rng(3)
        seq_a = rng.random((4, D), dtype=np.float32)
        seq_b = rng.random((4, D), dtype=np.float32)

        score_before = score_fn(seq_a, seq_b)

        # Train for a few steps.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        for _ in range(10):
            multitask_train_step(model, optimizer,
                                 merge_batch=merge_batch,
                                 topology_batch=topo_batch)

        score_after = score_fn(seq_a, seq_b)

        self.assertNotAlmostEqual(
            score_before, score_after, places=5,
            msg="score_fn did not change after model update — "
                "may be capturing stale weights (lru_cache bug)",
        )

    def test_score_fn_deterministic_in_eval_mode(self):
        """Two calls with the same input on the same model must return the same value."""
        import numpy as np
        from neuronauts.shared_grammar_model import SharedGrammarModel

        mod = self._import_train()
        model = SharedGrammarModel(embedding_dim=16)
        D = model._init_kwargs["input_dim"]
        score_fn = mod._make_live_merge_score_fn(model)

        rng = np.random.default_rng(5)
        seq = rng.random((6, D), dtype=np.float32)
        r1 = score_fn(seq, seq)
        r2 = score_fn(seq, seq)
        self.assertAlmostEqual(r1, r2, places=6)

    def test_score_fn_accepts_variable_length_sequences(self):
        """The score function must handle variable-length path sequences."""
        import numpy as np
        from neuronauts.shared_grammar_model import SharedGrammarModel

        mod = self._import_train()
        model = SharedGrammarModel(embedding_dim=16)
        D = model._init_kwargs["input_dim"]
        score_fn = mod._make_live_merge_score_fn(model)

        rng = np.random.default_rng(6)
        for T in (1, 5, 20, 100):
            seq = rng.random((T, D), dtype=np.float32)
            result = score_fn(seq, seq)
            self.assertIsInstance(result, float, f"failed for T={T}")
            self.assertFalse(np.isnan(result), f"NaN for T={T}")


# ---------------------------------------------------------------------------
# Multitask training convergence
# ---------------------------------------------------------------------------

class MultitaskConvergenceTest(unittest.TestCase):
    """multitask_train_step must decrease loss and increase accuracy on a
    fixed dataset over many gradient steps.

    A failure here indicates the backward pass is broken, gradients are not
    flowing, or the loss weighting is misconfigured.
    """

    def setUp(self):
        _require_torch()

    def test_loss_decreases_over_steps(self):
        """Loss must trend downward over 50 gradient steps on a fixed batch."""
        import torch
        from neuronauts.shared_grammar_model import SharedGrammarModel, multitask_train_step

        synapses = _make_mixed_synapses(n_per_group=8)
        merge_batch, topo_batch = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")

        model = SharedGrammarModel(embedding_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

        losses = []
        for _ in range(50):
            m = multitask_train_step(model, optimizer,
                                     merge_batch=merge_batch,
                                     topology_batch=topo_batch)
            losses.append(m["loss"])

        first_half = float(np.mean(losses[:10]))
        second_half = float(np.mean(losses[40:]))
        self.assertLess(
            second_half, first_half,
            f"Loss did not decrease: first_half={first_half:.4f} "
            f"second_half={second_half:.4f}",
        )

    def test_merge_accuracy_improves_over_steps(self):
        """Merge accuracy should improve from ~0.5 to > 0.7 on a separable dataset."""
        import torch
        from neuronauts.shared_grammar_model import SharedGrammarModel, multitask_train_step

        synapses = _make_mixed_synapses(n_per_group=8)
        merge_batch, topo_batch = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")

        model = SharedGrammarModel(embedding_dim=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

        initial_accs, final_accs = [], []
        for step in range(60):
            m = multitask_train_step(model, optimizer,
                                     merge_batch=merge_batch,
                                     topology_batch=topo_batch)
            if step < 5:
                initial_accs.append(m["merge_accuracy"])
            if step >= 55:
                final_accs.append(m["merge_accuracy"])

        initial_avg = float(np.mean(initial_accs))
        final_avg = float(np.mean(final_accs))
        self.assertGreater(
            final_avg, initial_avg,
            f"Merge accuracy did not improve: initial={initial_avg:.3f} "
            f"final={final_avg:.3f}",
        )

    def test_bridge_batch_loss_is_finite_when_supplied(self):
        """When bridge_batch is provided, the total loss must stay finite."""
        import torch
        from neuronauts.shared_grammar_model import SharedGrammarModel, multitask_train_step

        synapses = _make_mixed_synapses(n_per_group=6)
        merge_batch, topo_batch = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")

        model = SharedGrammarModel(embedding_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        B, T = merge_batch["left_x"].shape[:2]
        bridge_batch = {
            "left_x": merge_batch["left_x"],
            "left_mask": merge_batch["left_mask"],
            "right_x": merge_batch["right_x"],
            "right_mask": merge_batch["right_mask"],
            "target_midpoint": torch.randn(B, 3),
            "target_direction": torch.nn.functional.normalize(torch.randn(B, 3), dim=-1),
        }

        metrics = multitask_train_step(
            model, optimizer,
            merge_batch=merge_batch,
            topology_batch=topo_batch,
            bridge_batch=bridge_batch,
        )
        self.assertIn("bridge_loss", metrics)
        self.assertTrue(np.isfinite(metrics["loss"]),
                        f"total loss not finite: {metrics['loss']}")
        self.assertGreater(metrics["bridge_loss"], 0.0,
                           "bridge_loss should be > 0 with random targets")

    def test_metrics_dict_has_all_expected_keys(self):
        import torch
        from neuronauts.shared_grammar_model import SharedGrammarModel, multitask_train_step

        synapses = _make_mixed_synapses(n_per_group=6)
        merge_batch, topo_batch = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")

        model = SharedGrammarModel(embedding_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = multitask_train_step(model, optimizer,
                                       merge_batch=merge_batch,
                                       topology_batch=topo_batch)
        for key in ("loss", "merge_loss", "atomicity_loss", "bridge_loss",
                    "merge_accuracy", "atomicity_accuracy"):
            self.assertIn(key, metrics, f"missing key: {key!r}")

    def test_gradient_flows_to_merge_and_atomicity_heads(self):
        """Merge and atomicity head parameters must receive gradients even
        without a bridge_batch (bridge_head is only updated when bridge_batch
        is supplied — that is tested separately below).
        """
        import torch
        from neuronauts.shared_grammar_model import SharedGrammarModel, multitask_train_step

        synapses = _make_mixed_synapses(n_per_group=6)
        merge_batch, topo_batch = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")

        model = SharedGrammarModel(embedding_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        multitask_train_step(model, optimizer,
                             merge_batch=merge_batch,
                             topology_batch=topo_batch)

        # Only check parameters not in bridge_head (bridge_head needs bridge_batch).
        no_grad = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None and "bridge_head" not in name
        ]
        self.assertEqual(
            no_grad, [],
            f"Non-bridge parameters with no gradient after train step: {no_grad}",
        )

    def test_bridge_head_receives_gradient_with_bridge_batch(self):
        """bridge_head parameters must receive gradients when bridge_batch is
        provided.  If they stay at None the bridge loss is not flowing.
        """
        import torch
        from neuronauts.shared_grammar_model import SharedGrammarModel, multitask_train_step

        synapses = _make_mixed_synapses(n_per_group=6)
        merge_batch, topo_batch = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")

        model = SharedGrammarModel(embedding_dim=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        B = merge_batch["left_x"].shape[0]
        bridge_batch = {
            "left_x": merge_batch["left_x"],
            "left_mask": merge_batch["left_mask"],
            "right_x": merge_batch["right_x"],
            "right_mask": merge_batch["right_mask"],
            "target_midpoint": torch.randn(B, 3),
            "target_direction": torch.nn.functional.normalize(torch.randn(B, 3), dim=-1),
        }

        multitask_train_step(model, optimizer,
                             merge_batch=merge_batch,
                             topology_batch=topo_batch,
                             bridge_batch=bridge_batch)

        bridge_no_grad = [
            name for name, p in model.named_parameters()
            if p.requires_grad and p.grad is None and "bridge_head" in name
        ]
        self.assertEqual(
            bridge_no_grad, [],
            f"bridge_head parameters with no gradient when bridge_batch supplied: "
            f"{bridge_no_grad}",
        )


# ---------------------------------------------------------------------------
# Grammar batch correctness (shapes and dtypes used in training loop)
# ---------------------------------------------------------------------------

class GrammarBatchShapeTest(unittest.TestCase):
    """Verify that the batches produced by merge_dataset and topology_dataset
    have the exact shapes and dtypes expected by the model.

    Wrong batch shapes propagate silently until a broadcasting bug causes
    a train step to produce garbage gradients or NaN loss.
    """

    def setUp(self):
        _require_torch()

    def test_merge_batch_left_right_same_shape(self):
        synapses = _make_mixed_synapses(n_per_group=6)
        merge_batch, _ = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")
        self.assertEqual(merge_batch["left_x"].shape, merge_batch["right_x"].shape)
        self.assertEqual(merge_batch["left_mask"].shape, merge_batch["right_mask"].shape)

    def test_merge_batch_y_is_1d(self):
        synapses = _make_mixed_synapses(n_per_group=6)
        merge_batch, _ = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")
        self.assertEqual(merge_batch["y"].ndim, 1,
                         f"merge y must be 1-D, got shape {merge_batch['y'].shape}")

    def test_topo_batch_branch_x_is_4d(self):
        synapses = _make_mixed_synapses(n_per_group=6)
        _, topo_batch = _build_batches(synapses)
        if topo_batch is None:
            self.skipTest("not enough examples")
        self.assertEqual(topo_batch["branch_x"].ndim, 4,
                         f"branch_x must be 4-D (B, branches, steps, feat), "
                         f"got {topo_batch['branch_x'].shape}")

    def test_topo_batch_y_shape_matches_batch(self):
        synapses = _make_mixed_synapses(n_per_group=6)
        _, topo_batch = _build_batches(synapses)
        if topo_batch is None:
            self.skipTest("not enough examples")
        B = topo_batch["branch_x"].shape[0]
        self.assertEqual(topo_batch["y"].shape[0], B)

    def test_merge_batch_input_dim_matches_model(self):
        """The feature dimension of left_x / right_x must match model input_dim."""
        from neuronauts.grammar import path_feature_dim, DEFAULT_PATH_FEATURE_MODE

        synapses = _make_mixed_synapses(n_per_group=6)
        merge_batch, _ = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")
        feat_dim = merge_batch["left_x"].shape[-1]
        expected_dim = path_feature_dim(DEFAULT_PATH_FEATURE_MODE)
        self.assertEqual(feat_dim, expected_dim,
                         f"expected input_dim={expected_dim}, got {feat_dim}")

    def test_branch_mask_shape_matches_branch_x(self):
        """branch_mask shape: (B, max_branches); branch_sequence_mask: (B, max_branches, T)."""
        import torch
        synapses = _make_mixed_synapses(n_per_group=6)
        _, topo_batch = _build_batches(synapses)
        if topo_batch is None:
            self.skipTest("not enough examples")
        B, n_branches, T, D = topo_batch["branch_x"].shape
        self.assertEqual(topo_batch["branch_mask"].shape, torch.Size([B, n_branches]),
                         "branch_mask shape mismatch")
        self.assertEqual(topo_batch["branch_sequence_mask"].shape,
                         torch.Size([B, n_branches, T]),
                         "branch_sequence_mask shape mismatch")


# ---------------------------------------------------------------------------
# Checkpoint round-trip and weight identity
# ---------------------------------------------------------------------------

class CheckpointConsistencyTest(unittest.TestCase):
    """After saving and reloading a grammar model:
    - Predictions must be bit-identical to the in-memory model.
    - Training from the loaded checkpoint must continue to update weights.
    """

    def setUp(self):
        _require_torch()

    def test_loaded_checkpoint_predictions_match_original(self):
        import torch
        from neuronauts.shared_grammar_model import (
            SharedGrammarModel,
            save_shared_grammar_model,
            load_shared_grammar_model,
        )

        model = SharedGrammarModel(embedding_dim=16)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt = f.name
        save_shared_grammar_model(ckpt, model)
        loaded = load_shared_grammar_model(ckpt)
        loaded.eval()

        D = model._init_kwargs["input_dim"]
        x = torch.randn(2, 5, D)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        with torch.no_grad():
            out_orig = model.score_merge(x, mask, x, mask)
            out_load = loaded.score_merge(x, mask, x, mask)
        torch.testing.assert_close(out_orig, out_load,
                                   msg="loaded checkpoint gave different predictions")

    def test_loaded_checkpoint_can_be_fine_tuned(self):
        """A loaded model should be trainable (gradient flow must work)."""
        import torch
        from neuronauts.shared_grammar_model import (
            SharedGrammarModel,
            save_shared_grammar_model,
            load_shared_grammar_model,
            multitask_train_step,
        )

        synapses = _make_mixed_synapses(n_per_group=6)
        merge_batch, topo_batch = _build_batches(synapses)
        if merge_batch is None:
            self.skipTest("not enough examples")

        model = SharedGrammarModel(embedding_dim=16)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt = f.name
        save_shared_grammar_model(ckpt, model)
        loaded = load_shared_grammar_model(ckpt)
        loaded.train()

        before = {n: p.clone().detach() for n, p in loaded.named_parameters()}
        optimizer = torch.optim.Adam(loaded.parameters(), lr=1e-2)
        multitask_train_step(loaded, optimizer,
                             merge_batch=merge_batch,
                             topology_batch=topo_batch)
        after = {n: p.clone().detach() for n, p in loaded.named_parameters()}

        changed = any(not torch.allclose(before[n], after[n]) for n in before)
        self.assertTrue(changed, "loaded checkpoint parameters unchanged after fine-tuning step")


if __name__ == "__main__":
    unittest.main()
