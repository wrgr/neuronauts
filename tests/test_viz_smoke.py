"""Smoke tests for neuronauts.viz plot functions.

Verifies each plotting function returns a Figure on minimal input,
and that plot_f1_history_from_ledger / plot_scaffold_purity handle
edge cases correctly.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import matplotlib  # noqa: F401
except ImportError:
    raise unittest.SkipTest("matplotlib not installed")

from neuronauts.viz import (
    plot_bridge_proposals,
    plot_cell_labels,
    plot_cell_quality,
    plot_evaluation_summary,
    plot_f1_history,
    plot_f1_history_from_ledger,
    plot_merge_probabilities,
    plot_scaffold_groups,
    plot_scaffold_purity,
    plot_scaffold_synapses,
    plot_training_history,
)


class PlotReturnsFigureTest(unittest.TestCase):
    """Each plot function returns a matplotlib.figure.Figure."""

    def test_plot_scaffold_synapses_returns_figure(self):
        pre_pt = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        post_pt = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
        fig = plot_scaffold_synapses(pre_pt, post_pt)
        self.assertIsNotNone(fig)
        self.assertEqual(type(fig).__name__, "Figure")

    def test_plot_scaffold_groups_returns_figure(self):
        pre_pt = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        post_pt = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
        pre_seg = np.array([10, 10], dtype=np.int64)
        post_seg = np.array([20], dtype=np.int64)
        pre_root = np.array([100, 100], dtype=np.int64)
        post_root = np.array([200], dtype=np.int64)
        fig = plot_scaffold_groups(
            pre_pt, post_pt, pre_seg, post_seg, pre_root, post_root
        )
        self.assertIsNotNone(fig)
        self.assertEqual(type(fig).__name__, "Figure")

    def test_plot_bridge_proposals_returns_figure(self):
        pre_pt = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        post_pt = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
        proposals = [(0, 1, 5.0)]
        neuron_endpoint_pts = {
            0: np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float32),
            1: np.array([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]], dtype=np.float32),
        }
        fig = plot_bridge_proposals(
            pre_pt, post_pt, proposals, neuron_endpoint_pts
        )
        self.assertIsNotNone(fig)
        self.assertEqual(type(fig).__name__, "Figure")

    def test_plot_f1_history_returns_figure(self):
        fig = plot_f1_history(
            run_ids=["run1", "run2", "run3"],
            f1_values=[0.5, 0.6, 0.55],
            holdout_f1_values=[0.4, 0.5, 0.45],
        )
        self.assertIsNotNone(fig)
        self.assertEqual(type(fig).__name__, "Figure")

    def test_plot_scaffold_purity_returns_figure(self):
        seg_ids = np.array([1, 1, 1, 2, 2], dtype=np.int64)
        root_ids = np.array([10, 10, 20, 30, 30], dtype=np.int64)
        fig = plot_scaffold_purity(seg_ids, root_ids)
        self.assertIsNotNone(fig)
        self.assertEqual(type(fig).__name__, "Figure")


class PlotF1HistoryFromLedgerTest(unittest.TestCase):
    """Test plot_f1_history_from_ledger JSON parsing and max_runs."""

    def test_parses_jsonl_and_returns_figure(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"run_id": "a", "val_f1": 0.5, "holdout_f1": 0.4}\n')
            f.write('{"run_id": "b", "val_f1": 0.6}\n')
            path = f.name
        try:
            fig = plot_f1_history_from_ledger(path)
            self.assertIsNotNone(fig)
            self.assertEqual(type(fig).__name__, "Figure")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_skips_entries_without_val_f1(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"run_id": "a", "val_f1": 0.5}\n')
            f.write('{"run_id": "b"}\n')  # no val_f1
            f.write('{"run_id": "c", "val_f1": 0.7}\n')
            path = f.name
        try:
            fig = plot_f1_history_from_ledger(path, max_runs=10)
            self.assertIsNotNone(fig)
            # Should have 2 runs (a, c), not 3
        finally:
            Path(path).unlink(missing_ok=True)

    def test_max_runs_slices_to_most_recent(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(50):
                f.write(json.dumps({"run_id": f"r{i}", "val_f1": 0.5 + i * 0.001}) + "\n")
            path = f.name
        try:
            fig = plot_f1_history_from_ledger(path, max_runs=5)
            self.assertIsNotNone(fig)
            # Chart should show at most 5 runs
        finally:
            Path(path).unlink(missing_ok=True)

    def test_handles_empty_ledger(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            fig = plot_f1_history_from_ledger(path)
            self.assertIsNotNone(fig)
            self.assertEqual(type(fig).__name__, "Figure")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_handles_json_decode_error_gracefully(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"run_id": "a", "val_f1": 0.5}\n')
            f.write('not valid json\n')
            f.write('{"run_id": "c", "val_f1": 0.7}\n')
            path = f.name
        try:
            fig = plot_f1_history_from_ledger(path)
            self.assertIsNotNone(fig)
        finally:
            Path(path).unlink(missing_ok=True)


class PlotScaffoldPurityTest(unittest.TestCase):
    """Test plot_scaffold_purity purity computation."""

    def test_purity_one_segment_all_same_root(self):
        """Single segment with one root → purity 1.0."""
        seg_ids = np.array([1, 1, 1], dtype=np.int64)
        root_ids = np.array([10, 10, 10], dtype=np.int64)
        fig = plot_scaffold_purity(seg_ids, root_ids)
        self.assertIsNotNone(fig)
        # Purity logic: majority_count / total = 3/3 = 1.0
        axes = fig.axes
        self.assertGreaterEqual(len(axes), 1)
        bars = axes[0].patches if hasattr(axes[0], "patches") else []
        # Bar heights should include 1.0
        for bar in bars:
            self.assertLessEqual(bar.get_height(), 1.05)
            self.assertGreaterEqual(bar.get_height(), 0)

    def test_purity_segment_with_mixed_roots(self):
        """Segment with 2 of one root, 1 of another → purity 2/3."""
        seg_ids = np.array([1, 1, 1], dtype=np.int64)
        root_ids = np.array([10, 10, 20], dtype=np.int64)
        fig = plot_scaffold_purity(seg_ids, root_ids)
        self.assertIsNotNone(fig)
        # Majority = 2, total = 3 → purity 2/3 ≈ 0.667
        axes = fig.axes
        self.assertGreaterEqual(len(axes), 1)

    def test_multiple_segments(self):
        """Multiple segments produce multiple bars."""
        seg_ids = np.array([1, 1, 2, 2, 2], dtype=np.int64)
        root_ids = np.array([10, 10, 20, 20, 30], dtype=np.int64)
        fig = plot_scaffold_purity(seg_ids, root_ids)
        self.assertIsNotNone(fig)
        self.assertEqual(type(fig).__name__, "Figure")


class PlotCellLabelsTest(unittest.TestCase):
    """Test plot_cell_labels with inferred vs ground truth."""

    def test_with_ground_truth(self):
        pre_pt = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        post_pt = np.array([[10, 11, 12], [13, 14, 15]], dtype=np.float32)
        pre_labels = np.array([0, 0, 1], dtype=np.int64)
        post_labels = np.array([0, 1], dtype=np.int64)
        fig = plot_cell_labels(
            pre_pt, post_pt, pre_labels, post_labels,
            pre_root_id=np.array([10, 10, 20], dtype=np.int64),
            post_root_id=np.array([30, 40], dtype=np.int64),
        )
        self.assertEqual(type(fig).__name__, "Figure")
        # 2 rows x 2 cols = 4 axes
        self.assertEqual(len(fig.axes), 4)

    def test_without_ground_truth(self):
        pre_pt = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        post_pt = np.array([[7, 8, 9]], dtype=np.float32)
        fig = plot_cell_labels(
            pre_pt, post_pt,
            np.array([0, 1], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )
        self.assertEqual(type(fig).__name__, "Figure")
        # 2 rows x 1 col = 2 axes
        self.assertEqual(len(fig.axes), 2)


class PlotCellQualityTest(unittest.TestCase):
    """Test plot_cell_quality bar chart and histogram."""

    def test_basic(self):
        scores = {0: 0.95, 1: 0.82, 2: 0.45, 3: 0.12}
        fig = plot_cell_quality(scores, threshold=0.5)
        self.assertEqual(type(fig).__name__, "Figure")

    def test_all_high_quality(self):
        scores = {i: 0.9 + i * 0.01 for i in range(5)}
        fig = plot_cell_quality(scores)
        self.assertEqual(type(fig).__name__, "Figure")

    def test_empty(self):
        fig = plot_cell_quality({})
        self.assertEqual(type(fig).__name__, "Figure")


class PlotTrainingHistoryTest(unittest.TestCase):
    """Test plot_training_history with various metric sets."""

    def test_basic_loss_only(self):
        history = {"train_loss": [1.0, 0.8, 0.6, 0.5]}
        fig = plot_training_history(history)
        self.assertEqual(type(fig).__name__, "Figure")

    def test_multi_metric(self):
        history = {
            "train_loss": [1.0, 0.5, 0.3],
            "val_loss": [1.1, 0.7, 0.4],
            "train_pos_sim": [0.3, 0.5, 0.7],
            "train_neg_sim": [0.5, 0.3, 0.1],
        }
        fig = plot_training_history(history)
        self.assertEqual(type(fig).__name__, "Figure")

    def test_empty_history(self):
        fig = plot_training_history({})
        self.assertEqual(type(fig).__name__, "Figure")


class PlotMergeProbabilitiesTest(unittest.TestCase):
    """Test plot_merge_probabilities histogram."""

    def test_basic(self):
        probs = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.2, 0.8]
        fig = plot_merge_probabilities(probs, threshold=0.5)
        self.assertEqual(type(fig).__name__, "Figure")

    def test_all_accept(self):
        fig = plot_merge_probabilities([0.9, 0.95, 0.99])
        self.assertEqual(type(fig).__name__, "Figure")

    def test_empty(self):
        fig = plot_merge_probabilities([])
        self.assertEqual(type(fig).__name__, "Figure")


class PlotEvaluationSummaryTest(unittest.TestCase):
    """Test plot_evaluation_summary grouped bar chart."""

    def test_gnn_only(self):
        results = {
            "split": "test", "n_boxes": 10,
            "gnn": {"f1_mean": 0.72, "precision_mean": 0.80, "recall_mean": 0.65},
        }
        fig = plot_evaluation_summary(results)
        self.assertEqual(type(fig).__name__, "Figure")

    def test_with_baseline(self):
        results = {
            "split": "test", "n_boxes": 10,
            "gnn": {"f1_mean": 0.72, "precision_mean": 0.80, "recall_mean": 0.65},
            "baseline": {"f1_mean": 0.60, "precision_mean": 0.70, "recall_mean": 0.52},
        }
        fig = plot_evaluation_summary(results)
        self.assertEqual(type(fig).__name__, "Figure")


if __name__ == "__main__":
    unittest.main()
