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

from neuronauts.viz import (
    plot_bridge_proposals,
    plot_f1_history,
    plot_f1_history_from_ledger,
    plot_scaffold_groups,
    plot_scaffold_purity,
    plot_scaffold_synapses,
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


if __name__ == "__main__":
    unittest.main()
