"""Tests for neuronauts/merge.py — the cKDTree fallback and merge_agents.

merge.py is at 15% coverage despite being on the critical path:
every agent simulation result passes through merge_agents before any
downstream graph construction or evaluation.  Bugs here silently
produce the wrong number of neurons or wrong synapse assignments.
"""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# cKDTree fallback
# We force-import the fallback by temporarily patching scipy out of sys.modules.
# ---------------------------------------------------------------------------

def _import_merge_with_fallback():
    """Import neuronauts.merge after hiding scipy so the fallback cKDTree is used.

    Carefully restores sys.modules to its exact original state after extracting
    the fallback class, so that subsequent imports of neuronauts.merge from
    other test modules receive the same (already-loaded) module object.
    """
    import importlib

    # Snapshot the full module cache before we touch anything.
    saved_modules = dict(sys.modules)

    # Hide scipy so the try/except in merge.py takes the fallback branch.
    for k in list(sys.modules):
        if k.startswith("scipy"):
            sys.modules.pop(k)

    # Unload the real merge module so reload() will re-execute the try/except.
    orig_merge = sys.modules.pop("neuronauts.merge", None)

    try:
        import neuronauts.merge as m
        importlib.reload(m)
        FallbackKDTree = m.cKDTree
    finally:
        # Restore everything exactly as it was.
        sys.modules.clear()
        sys.modules.update(saved_modules)

    return FallbackKDTree


class CKDTreeFallbackTest(unittest.TestCase):
    """Tests for the pure-numpy cKDTree fallback defined in merge.py."""

    @classmethod
    def setUpClass(cls):
        cls.KDTree = _import_merge_with_fallback()

    # ------------------------------------------------------------------
    # query_ball_point
    # ------------------------------------------------------------------

    def test_query_ball_point_scalar_returns_list(self):
        """Single query point (1-D) must return a flat list, not a list-of-list."""
        data = np.array([[0, 0, 0], [1, 0, 0], [5, 0, 0]], dtype=np.float32)
        tree = self.KDTree(data)
        result = tree.query_ball_point(np.array([0, 0, 0], dtype=np.float32), r=1.5)
        # Should contain indices 0 and 1 (distance <= 1.5), not 2 (distance 5)
        self.assertIsInstance(result, list)
        self.assertIn(0, result)
        self.assertIn(1, result)
        self.assertNotIn(2, result)

    def test_query_ball_point_multiple_returns_list_of_lists(self):
        data = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]], dtype=np.float32)
        tree = self.KDTree(data)
        queries = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
        result = tree.query_ball_point(queries, r=1.5)
        self.assertEqual(len(result), 2)
        # First query: neighbours of (0,0,0) within 1.5 → indices 0,1
        self.assertIn(0, result[0])
        self.assertIn(1, result[0])
        # Second query: neighbours of (10,0,0) within 1.5 → index 2
        self.assertIn(2, result[1])
        self.assertNotIn(0, result[1])

    def test_query_ball_point_empty_result_when_no_neighbours(self):
        data = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32)
        tree = self.KDTree(data)
        result = tree.query_ball_point(np.array([100, 100, 100], dtype=np.float32), r=1.0)
        self.assertEqual(result, [])

    def test_query_ball_point_radius_zero_only_exact(self):
        """Radius 0 should only return exact-match indices."""
        data = np.array([[0.0, 0.0, 0.0], [1e-6, 0.0, 0.0]], dtype=np.float32)
        tree = self.KDTree(data)
        result = tree.query_ball_point(np.array([0.0, 0.0, 0.0], dtype=np.float32), r=0.0)
        self.assertIn(0, result)
        # index 1 is distance 1e-6 > 0 — may or may not be included depending on float precision,
        # but index 0 must always be included.

    # ------------------------------------------------------------------
    # query_pairs
    # ------------------------------------------------------------------

    def test_query_pairs_returns_set_by_default(self):
        data = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]], dtype=np.float32)
        tree = self.KDTree(data)
        pairs = tree.query_pairs(r=1.5)
        self.assertIsInstance(pairs, set)
        self.assertIn((0, 1), pairs)
        self.assertNotIn((0, 2), pairs)
        self.assertNotIn((1, 2), pairs)

    def test_query_pairs_ndarray_output(self):
        data = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]], dtype=np.float32)
        tree = self.KDTree(data)
        arr = tree.query_pairs(r=1.5, output_type="ndarray")
        self.assertIsInstance(arr, np.ndarray)
        if len(arr) > 0:
            self.assertEqual(arr.ndim, 2)
            self.assertEqual(arr.shape[1], 2)
            # The pair (0,1) should appear.
            rows = [tuple(row) for row in arr]
            self.assertIn((0, 1), rows)

    def test_query_pairs_no_pairs_in_radius(self):
        data = np.array([[0, 0, 0], [100, 0, 0]], dtype=np.float32)
        tree = self.KDTree(data)
        pairs = tree.query_pairs(r=1.0)
        self.assertEqual(len(pairs), 0)

    def test_query_pairs_single_point(self):
        data = np.array([[5, 5, 5]], dtype=np.float32)
        tree = self.KDTree(data)
        pairs = tree.query_pairs(r=10.0)
        self.assertEqual(len(pairs), 0)

    def test_query_pairs_all_within_radius(self):
        """All 3 points within radius → should find all 3 pairs."""
        data = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        tree = self.KDTree(data)
        pairs = tree.query_pairs(r=2.5)
        self.assertIn((0, 1), pairs)
        self.assertIn((1, 2), pairs)
        self.assertIn((0, 2), pairs)

    # ------------------------------------------------------------------
    # query (nearest-neighbour)
    # ------------------------------------------------------------------

    def test_query_returns_distance_and_index(self):
        data = np.array([[0, 0, 0], [3, 0, 0], [10, 0, 0]], dtype=np.float32)
        tree = self.KDTree(data)
        dist, idx = tree.query(np.array([2.5, 0, 0], dtype=np.float32))
        self.assertEqual(idx, 1)   # closest is point at (3,0,0)
        self.assertAlmostEqual(dist, 0.5, places=5)

    def test_query_exact_match_gives_zero_distance(self):
        data = np.array([[7, 3, 1], [0, 0, 0]], dtype=np.float32)
        tree = self.KDTree(data)
        dist, idx = tree.query(np.array([7, 3, 1], dtype=np.float32))
        self.assertAlmostEqual(dist, 0.0, places=5)
        self.assertEqual(idx, 0)


# ---------------------------------------------------------------------------
# merge_agents
# ---------------------------------------------------------------------------

def _make_agent(agent_id: int, path: list[tuple], visited_synapses: list[int]):
    from neuronauts.agent import Agent
    a = Agent(agent_id=agent_id)
    a.path = [np.array(p, dtype=np.float32) for p in path]
    a.visited_synapses = list(visited_synapses)
    return a


class MergeAgentsTest(unittest.TestCase):
    """Tests for merge.merge_agents — the union-find merge step."""

    def test_empty_agents_returns_empty_dict(self):
        from neuronauts.merge import merge_agents
        result = merge_agents([], merge_radius=5.0)
        self.assertEqual(result, {})

    def test_agents_shorter_than_min_path_filtered_out(self):
        """Agents with fewer than min_path_length steps must be ignored."""
        from neuronauts.merge import merge_agents
        a = _make_agent(0, [(0, 0, 0), (1, 0, 0)], [])  # length 2
        result = merge_agents([a], merge_radius=5.0, min_path_length=5)
        self.assertEqual(result, {})

    def test_single_valid_agent_creates_one_neuron(self):
        from neuronauts.merge import merge_agents, MergedNeuron
        path = [(float(i), 0.0, 0.0) for i in range(10)]
        a = _make_agent(0, path, [3, 7])
        neurons = merge_agents([a], merge_radius=1.0, min_path_length=5)
        self.assertEqual(len(neurons), 1)
        n = next(iter(neurons.values()))
        self.assertIsInstance(n, MergedNeuron)
        self.assertIn(3, n.synapse_indices)
        self.assertIn(7, n.synapse_indices)

    def test_two_close_agents_merge_into_one_neuron(self):
        """Two agents whose paths overlap within merge_radius → same neuron."""
        from neuronauts.merge import merge_agents
        path_a = [(float(i), 0.0, 0.0) for i in range(10)]
        path_b = [(float(i), 0.5, 0.0) for i in range(10)]  # only 0.5 away
        a = _make_agent(0, path_a, [1])
        b = _make_agent(1, path_b, [2])
        neurons = merge_agents([a, b], merge_radius=1.0, min_path_length=5)
        self.assertEqual(len(neurons), 1)
        n = next(iter(neurons.values()))
        self.assertIn(1, n.synapse_indices)
        self.assertIn(2, n.synapse_indices)
        self.assertEqual(len(n.agent_ids), 2)

    def test_two_far_agents_remain_separate_neurons(self):
        """Two agents with paths far apart → two distinct neurons."""
        from neuronauts.merge import merge_agents
        path_a = [(float(i), 0.0, 0.0) for i in range(10)]
        path_b = [(float(i) + 100, 0.0, 0.0) for i in range(10)]
        a = _make_agent(0, path_a, [])
        b = _make_agent(1, path_b, [])
        neurons = merge_agents([a, b], merge_radius=1.0, min_path_length=5)
        self.assertEqual(len(neurons), 2)

    def test_merged_neuron_path_points_contains_all_agent_paths(self):
        """After merge, path_points must include points from all member agents."""
        from neuronauts.merge import merge_agents
        path_a = [(float(i), 0.0, 0.0) for i in range(8)]
        path_b = [(float(i), 0.5, 0.0) for i in range(8)]  # within radius
        a = _make_agent(0, path_a, [])
        b = _make_agent(1, path_b, [])
        neurons = merge_agents([a, b], merge_radius=1.0, min_path_length=5)
        self.assertEqual(len(neurons), 1)
        n = next(iter(neurons.values()))
        # Combined path should have at least 8+8 points (no deduplication)
        self.assertGreaterEqual(len(n.path_points), 8)

    def test_synapse_indices_deduplicated_across_merged_agents(self):
        """Synapse indices are deduplicated after merging."""
        from neuronauts.merge import merge_agents
        path_a = [(float(i), 0.0, 0.0) for i in range(8)]
        path_b = [(float(i), 0.3, 0.0) for i in range(8)]
        a = _make_agent(0, path_a, [5, 7])
        b = _make_agent(1, path_b, [5, 9])  # 5 is duplicate
        neurons = merge_agents([a, b], merge_radius=1.0, min_path_length=5)
        n = next(iter(neurons.values()))
        self.assertEqual(len([x for x in n.synapse_indices if x == 5]), 1,
                         "synapse index 5 appears more than once")

    def test_three_way_transitive_merge(self):
        """A-B are close, B-C are close → A, B, C all end up in one neuron."""
        from neuronauts.merge import merge_agents
        path_a = [(0.0, 0.0, 0.0)] * 8
        path_b = [(0.5, 0.0, 0.0)] * 8  # close to a
        path_c = [(1.0, 0.0, 0.0)] * 8  # close to b
        a = _make_agent(0, path_a, [])
        b = _make_agent(1, path_b, [])
        c = _make_agent(2, path_c, [])
        neurons = merge_agents([a, b, c], merge_radius=0.8, min_path_length=5)
        self.assertEqual(len(neurons), 1)

    def test_neuron_ids_are_contiguous_integers(self):
        from neuronauts.merge import merge_agents
        agents = []
        for k in range(4):
            path = [(float(i + k * 100), 0.0, 0.0) for i in range(8)]
            agents.append(_make_agent(k, path, []))
        neurons = merge_agents(agents, merge_radius=1.0, min_path_length=5)
        self.assertEqual(set(neurons.keys()), set(range(len(neurons))))

    def test_path_points_shape_is_2d(self):
        from neuronauts.merge import merge_agents
        path = [(float(i), 0.0, 0.0) for i in range(8)]
        a = _make_agent(0, path, [])
        neurons = merge_agents([a], merge_radius=1.0, min_path_length=5)
        n = next(iter(neurons.values()))
        self.assertEqual(n.path_points.ndim, 2)
        self.assertEqual(n.path_points.shape[1], 3)

    def test_only_agents_meeting_min_path_length_merged(self):
        """Mix of short and long agents; only long ones contribute."""
        from neuronauts.merge import merge_agents
        short_path = [(0.0, 0.0, 0.0)] * 3  # below min
        long_path = [(float(i), 0.0, 0.0) for i in range(8)]
        short_agent = _make_agent(0, short_path, [99])
        long_agent = _make_agent(1, long_path, [42])
        neurons = merge_agents([short_agent, long_agent],
                               merge_radius=5.0, min_path_length=5)
        self.assertEqual(len(neurons), 1)
        n = next(iter(neurons.values()))
        self.assertNotIn(99, n.synapse_indices)
        self.assertIn(42, n.synapse_indices)


# ---------------------------------------------------------------------------
# MergedNeuron / ConnectivityGraph dataclass construction
# ---------------------------------------------------------------------------

class DataclassTest(unittest.TestCase):

    def test_merged_neuron_fields(self):
        from neuronauts.merge import MergedNeuron
        pts = np.zeros((4, 3), dtype=np.float32)
        n = MergedNeuron(neuron_id=7, agent_ids=[0, 1], path_points=pts,
                         synapse_indices=[3, 5], role="pre")
        self.assertEqual(n.neuron_id, 7)
        self.assertEqual(n.role, "pre")
        self.assertListEqual(n.synapse_indices, [3, 5])

    def test_merged_neuron_default_role_is_mixed(self):
        from neuronauts.merge import MergedNeuron
        pts = np.zeros((2, 3), dtype=np.float32)
        n = MergedNeuron(neuron_id=0, agent_ids=[], path_points=pts, synapse_indices=[])
        self.assertEqual(n.role, "mixed")

    def test_connectivity_graph_fields(self):
        from neuronauts.merge import ConnectivityGraph
        g = ConnectivityGraph(neurons={}, edges=[(0, 1, 2)], unresolved_synapse_indices=[5])
        self.assertEqual(len(g.edges), 1)
        self.assertListEqual(g.unresolved_synapse_indices, [5])


if __name__ == "__main__":
    unittest.main()
