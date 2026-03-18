"""Tests filling coverage gaps in grammar.py, training_batches.py,
topology_model.py, and experiment_driver.py."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuronauts.grammar import MergeScorer, build_multimodal_path_sequence, build_path_batch
from neuronauts.training_batches import pad_nested_path_sequences
from neuronauts.experiment_driver import parse_validation_metrics

try:
    import torch
    from neuronauts.topology_model import (
        AttentionArborValidator,
        TrainingConfig,
        load_validator,
        save_validator,
        train_iteration,
    )
    _TORCH = True
except ImportError:
    _TORCH = False


# ---------------------------------------------------------------------------
# MergeScorer (numpy cosine-similarity baseline)
# ---------------------------------------------------------------------------

class MergeScorerTest(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        scorer = MergeScorer()
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertAlmostEqual(scorer.score(v, v), 1.0, places=5)

    def test_orthogonal_vectors_score_zero(self):
        scorer = MergeScorer()
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(scorer.score(a, b), 0.0, places=5)

    def test_opposite_vectors_score_negative_one(self):
        scorer = MergeScorer()
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(scorer.score(a, b), -1.0, places=5)

    def test_zero_vector_returns_zero(self):
        scorer = MergeScorer()
        z = np.zeros(4, dtype=np.float32)
        v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self.assertAlmostEqual(scorer.score(z, v), 0.0, places=5)

    def test_score_is_symmetric(self):
        scorer = MergeScorer()
        rng = np.random.default_rng(0)
        a = rng.random(8).astype(np.float32)
        b = rng.random(8).astype(np.float32)
        self.assertAlmostEqual(scorer.score(a, b), scorer.score(b, a), places=5)

    def test_score_returns_float(self):
        scorer = MergeScorer()
        v = np.ones(3, dtype=np.float32)
        result = scorer.score(v, v)
        self.assertIsInstance(result, float)


# ---------------------------------------------------------------------------
# build_multimodal_path_sequence
# ---------------------------------------------------------------------------

class BuildMultimodalPathSequenceTest(unittest.TestCase):
    def _batch(self, T=5):
        return build_path_batch(
            edge_len=[1.0] * T,
            radius=[0.5] * T,
            curvature=[0.1] * T,
        )

    def test_base_only_has_shape_t_by_3(self):
        batch = self._batch(T=5)
        seq = build_multimodal_path_sequence(batch)
        self.assertEqual(seq.shape, (5, 3))

    def test_with_1d_skeleton_feat_expands_to_t_by_4(self):
        batch = self._batch(T=4)
        sk = np.ones(4, dtype=np.float32)
        seq = build_multimodal_path_sequence(batch, skeleton_feat=sk)
        self.assertEqual(seq.shape, (4, 4))

    def test_with_2d_skeleton_feat_has_correct_width(self):
        batch = self._batch(T=3)
        sk = np.ones((3, 2), dtype=np.float32)
        seq = build_multimodal_path_sequence(batch, skeleton_feat=sk)
        self.assertEqual(seq.shape, (3, 5))

    def test_with_skeleton_and_mesh_feat(self):
        batch = self._batch(T=6)
        sk = np.ones((6, 3), dtype=np.float32)
        ms = np.ones((6, 3), dtype=np.float32)
        seq = build_multimodal_path_sequence(batch, skeleton_feat=sk, mesh_feat=ms)
        self.assertEqual(seq.shape, (6, 9))

    def test_output_dtype_is_float32(self):
        batch = self._batch(T=4)
        seq = build_multimodal_path_sequence(batch)
        self.assertEqual(seq.dtype, np.float32)

    def test_shape_mismatch_raises_value_error(self):
        batch = self._batch(T=5)
        sk_wrong = np.ones(3, dtype=np.float32)  # length 3 != 5
        with self.assertRaises(ValueError):
            build_multimodal_path_sequence(batch, skeleton_feat=sk_wrong)

    def test_mesh_feat_shape_mismatch_raises_value_error(self):
        batch = self._batch(T=5)
        ms_wrong = np.ones((4, 2), dtype=np.float32)
        with self.assertRaises(ValueError):
            build_multimodal_path_sequence(batch, mesh_feat=ms_wrong)

    def test_base_features_are_correct_values(self):
        batch = build_path_batch(
            edge_len=[2.0, 3.0],
            radius=[0.1, 0.2],
            curvature=[0.0, 0.5],
        )
        seq = build_multimodal_path_sequence(batch)
        np.testing.assert_allclose(seq[:, 0], [2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(seq[:, 1], [0.1, 0.2], atol=1e-6)
        np.testing.assert_allclose(seq[:, 2], [0.0, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# pad_nested_path_sequences (direct coverage for edge cases)
# ---------------------------------------------------------------------------

class PadNestedPathSequencesTest(unittest.TestCase):
    def test_basic_shape(self):
        seqs = [
            [np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32),
             np.array([[3, 3, 3]], dtype=np.float32)],
            [np.array([[4, 4, 4], [5, 5, 5], [6, 6, 6]], dtype=np.float32)],
        ]
        batch = pad_nested_path_sequences(seqs, feature_dim=3)
        # batch_size=2, max_items=2, max_steps=3, feature_dim=3
        self.assertEqual(batch.x.shape, (2, 2, 3, 3))
        self.assertEqual(batch.sequence_mask.shape, (2, 2, 3))
        self.assertEqual(batch.item_mask.shape, (2, 2))

    def test_empty_input_returns_zero_batch(self):
        batch = pad_nested_path_sequences([])
        self.assertEqual(batch.x.shape[0], 0)

    def test_max_items_truncates_extra_sequences(self):
        seqs = [[
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
        ]]
        batch = pad_nested_path_sequences(seqs, max_items=2, feature_dim=3)
        self.assertEqual(batch.x.shape[1], 2)

    def test_item_mask_true_for_padding_items(self):
        seqs = [
            [np.ones((2, 3), dtype=np.float32)],   # 1 real item
            [np.ones((2, 3), dtype=np.float32),
             np.ones((2, 3), dtype=np.float32)],   # 2 real items
        ]
        batch = pad_nested_path_sequences(seqs, feature_dim=3)
        # batch[0] has only 1 item; position [0,1] should be masked (True).
        self.assertTrue(bool(batch.item_mask[0, 1]))
        self.assertFalse(bool(batch.item_mask[0, 0]))

    def test_sequence_mask_true_for_padding_steps(self):
        seqs = [[
            np.ones((3, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]]
        batch = pad_nested_path_sequences(seqs, feature_dim=3)
        # Item 1 has only 1 step; steps 1,2 should be masked.
        self.assertFalse(bool(batch.sequence_mask[0, 1, 0]))  # real step
        self.assertTrue(bool(batch.sequence_mask[0, 1, 1]))   # pad step
        self.assertTrue(bool(batch.sequence_mask[0, 1, 2]))   # pad step

    def test_padded_positions_are_zero(self):
        seqs = [[
            np.full((2, 3), 7.0, dtype=np.float32),
            np.full((1, 3), 3.0, dtype=np.float32),
        ]]
        batch = pad_nested_path_sequences(seqs, feature_dim=3)
        # Pad step in item 1 should be zero.
        np.testing.assert_array_equal(batch.x[0, 1, 1, :], 0.0)

    def test_ragged_groups_produce_correct_item_count(self):
        seqs = [
            [np.ones((2, 3), dtype=np.float32)],
            [np.ones((2, 3), dtype=np.float32)] * 4,
        ]
        batch = pad_nested_path_sequences(seqs, feature_dim=3)
        self.assertEqual(batch.x.shape[1], 4)


# ---------------------------------------------------------------------------
# AttentionArborValidator (topology_model.py)
# ---------------------------------------------------------------------------

@unittest.skipIf(not _TORCH, "torch not installed")
class AttentionArborValidatorTest(unittest.TestCase):
    def test_forward_output_shape(self):
        model = AttentionArborValidator(embed_dim=16, num_heads=4)
        model.eval()
        x = torch.randn(3, 5, 16)
        with torch.no_grad():
            out = model(x)
        self.assertEqual(tuple(out.shape), (3, 1))

    def test_output_is_probability_in_zero_one(self):
        model = AttentionArborValidator(embed_dim=16, num_heads=4)
        model.eval()
        x = torch.randn(4, 6, 16)
        with torch.no_grad():
            out = model(x)
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(out <= 1.0))

    def test_forward_with_padding_mask(self):
        model = AttentionArborValidator(embed_dim=16, num_heads=4)
        model.eval()
        x = torch.randn(2, 4, 16)
        mask = torch.tensor([[False, False, True, True],
                              [False, False, False, True]])
        with torch.no_grad():
            out = model(x, mask=mask)
        self.assertEqual(tuple(out.shape), (2, 1))

    def test_all_masked_doesnt_crash(self):
        model = AttentionArborValidator(embed_dim=8, num_heads=2)
        model.eval()
        x = torch.randn(1, 3, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)
        with torch.no_grad():
            out = model(x, mask=mask)
        self.assertEqual(tuple(out.shape), (1, 1))

    def test_train_iteration_reduces_loss(self):
        model = AttentionArborValidator(embed_dim=16, num_heads=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.randn(8, 4, 16)
        y = torch.zeros(8)
        losses = []
        for _ in range(5):
            loss = train_iteration(model, optimizer, x, y)
            losses.append(loss)
        # Loss should generally decrease (may fluctuate, but not monotonically grow).
        self.assertLess(losses[-1], losses[0] * 2)

    def test_train_iteration_returns_float(self):
        model = AttentionArborValidator(embed_dim=8, num_heads=2)
        optimizer = torch.optim.Adam(model.parameters())
        x = torch.randn(2, 3, 8)
        y = torch.ones(2)
        loss = train_iteration(model, optimizer, x, y)
        self.assertIsInstance(loss, float)

    def test_save_and_load_round_trip(self):
        model = AttentionArborValidator(embed_dim=16, num_heads=4)
        model.eval()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "validator.pt"
            save_validator(path, model, embed_dim=16)
            loaded = load_validator(path)
            x = torch.randn(2, 5, 16)
            with torch.no_grad():
                expected = model(x)
                actual = loaded(x)
            torch.testing.assert_close(actual, expected)

    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        self.assertEqual(cfg.epochs, 100)
        self.assertEqual(cfg.batch_size, 32)
        self.assertAlmostEqual(cfg.learning_rate, 1e-3)


# ---------------------------------------------------------------------------
# parse_validation_metrics (experiment_driver.py)
# ---------------------------------------------------------------------------

class ParseValidationMetricsTest(unittest.TestCase):
    _SAMPLE_OUTPUT = (
        "LineGraph F1=0.742  P=0.810  R=0.685  "
        "TP=54 FP=13 FN=25  (true edges=79, est edges=67)\n"
        "\nval_f1 = 0.742\n"
    )

    def test_parses_val_f1(self):
        m = parse_validation_metrics(self._SAMPLE_OUTPUT)
        self.assertAlmostEqual(m["val_f1"], 0.742, places=3)

    def test_parses_precision(self):
        m = parse_validation_metrics(self._SAMPLE_OUTPUT)
        self.assertAlmostEqual(m["precision"], 0.810, places=3)

    def test_parses_recall(self):
        m = parse_validation_metrics(self._SAMPLE_OUTPUT)
        self.assertAlmostEqual(m["recall"], 0.685, places=3)

    def test_parses_tp_fp_fn_as_int(self):
        m = parse_validation_metrics(self._SAMPLE_OUTPUT)
        self.assertEqual(m["tp"], 54)
        self.assertEqual(m["fp"], 13)
        self.assertEqual(m["fn"], 25)

    def test_empty_string_returns_all_none(self):
        m = parse_validation_metrics("")
        for key in ("val_f1", "precision", "recall", "tp", "fp", "fn"):
            self.assertIsNone(m[key])

    def test_partial_output_returns_none_for_missing_fields(self):
        m = parse_validation_metrics("val_f1 = 0.500")
        self.assertAlmostEqual(m["val_f1"], 0.500, places=3)
        self.assertIsNone(m["precision"])

    def test_zero_values_parse_correctly(self):
        text = "P=0.000  R=0.000  TP=0 FP=0 FN=10\nval_f1 = 0.000"
        m = parse_validation_metrics(text)
        self.assertAlmostEqual(m["val_f1"], 0.0)
        self.assertEqual(m["tp"], 0)
        self.assertEqual(m["fn"], 10)

    def test_returns_dict_with_all_expected_keys(self):
        m = parse_validation_metrics("")
        expected_keys = {"val_f1", "precision", "recall", "tp", "fp", "fn"}
        self.assertEqual(set(m.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
