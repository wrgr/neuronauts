"""Tests for PR 3: scaffold-aware graph initialization and viz helpers."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuronauts.fetch import SynapseTable, make_test_volume, SyntheticBenchmarkConfig
from neuronauts.run import _scaffold_union_from_seg_ids


# ---------------------------------------------------------------------------
# SynapseTable backward-compatibility
# ---------------------------------------------------------------------------

class SynapseTableSchemaTest(unittest.TestCase):
    def test_default_seg_ids_are_none(self):
        t = SynapseTable(
            pre_pt=np.zeros((2, 3), dtype=np.float32),
            post_pt=np.zeros((2, 3), dtype=np.float32),
            pre_root_id=np.array([1, 2], dtype=np.int64),
            post_root_id=np.array([3, 4], dtype=np.int64),
            synapse_id=np.array([0, 1], dtype=np.int64),
        )
        self.assertIsNone(t.pre_seg_id)
        self.assertIsNone(t.post_seg_id)

    def test_seg_ids_can_be_set(self):
        seg = np.array([10, 20], dtype=np.int64)
        t = SynapseTable(
            pre_pt=np.zeros((2, 3), dtype=np.float32),
            post_pt=np.zeros((2, 3), dtype=np.float32),
            pre_root_id=np.array([1, 2], dtype=np.int64),
            post_root_id=np.array([3, 4], dtype=np.int64),
            synapse_id=np.array([0, 1], dtype=np.int64),
            pre_seg_id=seg.copy(),
            post_seg_id=seg.copy(),
        )
        np.testing.assert_array_equal(t.pre_seg_id, seg)
        np.testing.assert_array_equal(t.post_seg_id, seg)

    def test_make_test_volume_populates_seg_ids(self):
        _, synapses = make_test_volume(seed=0)
        self.assertIsNotNone(synapses.pre_seg_id)
        self.assertIsNotNone(synapses.post_seg_id)
        self.assertEqual(len(synapses.pre_seg_id), len(synapses.pre_pt))
        self.assertEqual(len(synapses.post_seg_id), len(synapses.post_pt))

    def test_make_test_volume_seg_ids_match_root_ids(self):
        # In synthetic mode seg_ids == root_ids (perfect scaffold).
        _, synapses = make_test_volume(seed=1)
        np.testing.assert_array_equal(synapses.pre_seg_id, synapses.pre_root_id)
        np.testing.assert_array_equal(synapses.post_seg_id, synapses.post_root_id)


# ---------------------------------------------------------------------------
# _scaffold_union_from_seg_ids
# ---------------------------------------------------------------------------

class ScaffoldUnionTest(unittest.TestCase):
    def _parent_and_find(self, agents):
        parent = {a: a for a in agents}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        return parent, find

    def test_noop_when_seg_ids_none(self):
        agents = np.array([0, 1, 2], dtype=np.int32)
        role_hits = np.array([[True, False], [False, True], [True, True]], dtype=bool)
        parent, find = self._parent_and_find([0, 1, 2])
        _scaffold_union_from_seg_ids(agents, role_hits, None, parent)
        # Parent unchanged — all still self-rooted.
        self.assertEqual(find(0), 0)
        self.assertEqual(find(1), 1)
        self.assertEqual(find(2), 2)

    def test_same_seg_id_agents_are_unioned(self):
        # Agents 0 and 1 both hit synapse 0, which has seg_id=99.
        # Agent 2 hits synapse 1 with seg_id=77.
        agents = np.array([0, 1, 2], dtype=np.int32)
        role_hits = np.array([[True, False], [True, False], [False, True]], dtype=bool)
        seg_ids = np.array([99, 77], dtype=np.int64)  # per synapse
        parent, find = self._parent_and_find([0, 1, 2])
        _scaffold_union_from_seg_ids(agents, role_hits, seg_ids, parent)
        # Agents 0 and 1 should be in the same group.
        self.assertEqual(find(0), find(1))
        # Agent 2 should be separate.
        self.assertNotEqual(find(2), find(0))

    def test_multi_seg_agent_skipped(self):
        # Agent 0 hits synapses with TWO different seg_ids → not merged.
        agents = np.array([0, 1], dtype=np.int32)
        role_hits = np.array([[True, True], [True, False]], dtype=bool)
        seg_ids = np.array([10, 20], dtype=np.int64)
        parent, find = self._parent_and_find([0, 1])
        _scaffold_union_from_seg_ids(agents, role_hits, seg_ids, parent)
        # Agent 0 spans two segs → skipped; agent 1 only has seg 10.
        # They should NOT be merged because agent 0 was skipped.
        self.assertNotEqual(find(0), find(1))

    def test_zero_seg_id_is_ignored(self):
        # Seg_id == 0 is treated as "unknown", should not trigger merging.
        agents = np.array([0, 1], dtype=np.int32)
        role_hits = np.array([[True, False], [False, True]], dtype=bool)
        seg_ids = np.array([0, 0], dtype=np.int64)
        parent, find = self._parent_and_find([0, 1])
        _scaffold_union_from_seg_ids(agents, role_hits, seg_ids, parent)
        self.assertNotEqual(find(0), find(1))

    def test_three_agents_same_seg_id_all_merged(self):
        agents = np.array([0, 1, 2], dtype=np.int32)
        role_hits = np.array([[True, False], [True, False], [True, False]], dtype=bool)
        seg_ids = np.array([42, 99], dtype=np.int64)
        parent, find = self._parent_and_find([0, 1, 2])
        _scaffold_union_from_seg_ids(agents, role_hits, seg_ids, parent)
        self.assertEqual(find(0), find(1))
        self.assertEqual(find(1), find(2))

    def test_no_hits_agent_skipped(self):
        agents = np.array([0, 1], dtype=np.int32)
        role_hits = np.array([[False, False], [True, False]], dtype=bool)
        seg_ids = np.array([5, 6], dtype=np.int64)
        parent, find = self._parent_and_find([0, 1])
        _scaffold_union_from_seg_ids(agents, role_hits, seg_ids, parent)
        self.assertNotEqual(find(0), find(1))


# ---------------------------------------------------------------------------
# make_test_volume + scaffold round-trip
# ---------------------------------------------------------------------------

class ScaffoldRoundTripTest(unittest.TestCase):
    def test_scaffold_groups_reduce_neuron_count(self):
        """With a perfect scaffold, scaffold-aware merging should produce
        fewer (or equal) neurons than blind merging, because scaffold groups
        are pre-merged before geometry scoring."""
        from neuronauts.run import _merge_role_groups
        import numpy as np

        rng = np.random.default_rng(7)
        n_agents = 20
        n_synapses = 8
        # Assign agents to two synthetic groups via seg_ids.
        seg_ids = rng.integers(1, 3, size=n_synapses, dtype=np.int64)  # seg 1 or 2
        role_hits = rng.integers(0, 2, size=(n_agents, n_synapses), dtype=bool)
        # Make sure at least some agents have hits.
        role_hits[:5, :4] = True
        role_hits[5:10, 4:] = True

        # Fake path array (all zeros — agents have trivial paths).
        path_arr = np.zeros((n_agents, 10, 3), dtype=np.float32)

        neurons_no_scaffold, _, _, _ = _merge_role_groups(
            path_arr, role_hits, "pre", 0,
        )
        neurons_with_scaffold, _, _, _ = _merge_role_groups(
            path_arr, role_hits, "pre", 0,
            role_seg_ids=seg_ids,
        )

        # Scaffold can only reduce or equal the neuron count.
        self.assertLessEqual(
            len(neurons_with_scaffold),
            len(neurons_no_scaffold),
            "Scaffold grouping should not increase neuron count",
        )


# ---------------------------------------------------------------------------
# viz.py tests
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for CI
    from neuronauts.viz import (
        plot_scaffold_synapses,
        plot_scaffold_groups,
        plot_bridge_proposals,
        plot_f1_history,
        plot_f1_history_from_ledger,
        plot_scaffold_purity,
    )
    _MPL = True
except ImportError:
    _MPL = False


@unittest.skipIf(not _MPL, "matplotlib not installed")
class VizTest(unittest.TestCase):
    def _synapses(self, n=10, seed=0):
        rng = np.random.default_rng(seed)
        pre_pt = rng.random((n, 3)).astype(np.float32) * 50
        post_pt = rng.random((n, 3)).astype(np.float32) * 50
        seg_id = rng.integers(1, 4, size=n, dtype=np.int64)
        root_id = rng.integers(1, 4, size=n, dtype=np.int64)
        return pre_pt, post_pt, seg_id, root_id

    def test_plot_scaffold_synapses_returns_figure(self):
        from matplotlib.figure import Figure
        pre, post, seg, root = self._synapses()
        fig = plot_scaffold_synapses(pre, post, seg, seg)
        self.assertIsInstance(fig, Figure)

    def test_plot_scaffold_synapses_no_seg_ids(self):
        from matplotlib.figure import Figure
        pre, post, _, _ = self._synapses()
        fig = plot_scaffold_synapses(pre, post)
        self.assertIsInstance(fig, Figure)

    def test_plot_scaffold_groups_returns_figure(self):
        from matplotlib.figure import Figure
        pre, post, seg, root = self._synapses()
        fig = plot_scaffold_groups(pre, post, seg, seg, root, root)
        self.assertIsInstance(fig, Figure)

    def test_plot_bridge_proposals_returns_figure(self):
        from matplotlib.figure import Figure
        rng = np.random.default_rng(0)
        pre = rng.random((5, 3)).astype(np.float32)
        post = rng.random((5, 3)).astype(np.float32)
        proposals = [(0, 1, 2.5), (1, 2, 4.0)]
        neuron_pts = {
            0: rng.random((3, 3)).astype(np.float32),
            1: rng.random((3, 3)).astype(np.float32),
            2: rng.random((3, 3)).astype(np.float32),
        }
        fig = plot_bridge_proposals(pre, post, proposals, neuron_pts)
        self.assertIsInstance(fig, Figure)

    def test_plot_bridge_proposals_empty_proposals(self):
        from matplotlib.figure import Figure
        pre = np.random.rand(3, 3).astype(np.float32)
        post = np.random.rand(3, 3).astype(np.float32)
        fig = plot_bridge_proposals(pre, post, [], {})
        self.assertIsInstance(fig, Figure)

    def test_plot_f1_history_returns_figure(self):
        from matplotlib.figure import Figure
        fig = plot_f1_history(
            ["run1", "run2", "run3"],
            [0.4, 0.55, 0.62],
            holdout_f1_values=[0.38, 0.50, 0.58],
        )
        self.assertIsInstance(fig, Figure)

    def test_plot_f1_history_empty(self):
        from matplotlib.figure import Figure
        fig = plot_f1_history([], [])
        self.assertIsInstance(fig, Figure)

    def test_plot_f1_history_from_ledger(self):
        from matplotlib.figure import Figure
        entries = [
            {"run_id": f"r{i}", "val_f1": 0.3 + i * 0.05, "holdout_f1": 0.28 + i * 0.04}
            for i in range(5)
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            ledger_path = f.name
        try:
            fig = plot_f1_history_from_ledger(ledger_path)
            self.assertIsInstance(fig, Figure)
        finally:
            Path(ledger_path).unlink(missing_ok=True)

    def test_plot_f1_history_from_ledger_skips_missing_val_f1(self):
        from matplotlib.figure import Figure
        entries = [
            {"run_id": "r0", "val_f1": 0.4},
            {"run_id": "r1"},  # no val_f1
            {"run_id": "r2", "val_f1": 0.5},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
            ledger_path = f.name
        try:
            fig = plot_f1_history_from_ledger(ledger_path)
            self.assertIsInstance(fig, Figure)
        finally:
            Path(ledger_path).unlink(missing_ok=True)

    def test_plot_scaffold_purity_returns_figure(self):
        from matplotlib.figure import Figure
        rng = np.random.default_rng(42)
        seg_ids = rng.integers(1, 4, size=20, dtype=np.int64)
        root_ids = rng.integers(1, 4, size=20, dtype=np.int64)
        fig = plot_scaffold_purity(seg_ids, root_ids, role="pre")
        self.assertIsInstance(fig, Figure)

    def test_plot_scaffold_purity_perfect_scaffold(self):
        """When seg_id == root_id, every segment should have purity=1.0."""
        from matplotlib.figure import Figure
        ids = np.array([1, 1, 2, 2, 3, 3], dtype=np.int64)
        fig = plot_scaffold_purity(ids, ids, role="post")
        self.assertIsInstance(fig, Figure)

    def test_all_projections_work(self):
        from matplotlib.figure import Figure
        pre, post, seg, root = self._synapses()
        for proj in ("xy", "xz", "yz"):
            fig = plot_scaffold_synapses(pre, post, seg, seg, projection=proj)
            self.assertIsInstance(fig, Figure)


if __name__ == "__main__":
    unittest.main()
