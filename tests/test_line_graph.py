"""Tests for the line-graph F1 metric — the primary evaluation scalar.

These cover build_true_line_graph, compute_line_graph_f1, evaluate, and
evaluate_from_root_ids.  Because the keep/revert logic in the outer loop
is anchored on these numbers, correctness here is critical.
"""

import unittest

import numpy as np

from neuronauts.line_graph import (
    LineGraphMetrics,
    build_estimated_line_graph,
    build_true_line_graph,
    compute_line_graph_f1,
    evaluate,
    evaluate_from_root_ids,
)
from neuronauts.merge import ConnectivityGraph, MergedNeuron


# ---------------------------------------------------------------------------
# build_true_line_graph
# ---------------------------------------------------------------------------

class BuildTrueLineGraphTest(unittest.TestCase):
    def test_two_synapses_same_pre_root_produce_one_edge(self):
        pre = np.array([1, 1], dtype=np.int64)
        post = np.array([10, 11], dtype=np.int64)
        edges = build_true_line_graph(pre, post)
        self.assertEqual(edges, {(0, 1)})

    def test_two_synapses_same_post_root_produce_one_edge(self):
        pre = np.array([1, 2], dtype=np.int64)
        post = np.array([10, 10], dtype=np.int64)
        edges = build_true_line_graph(pre, post)
        self.assertEqual(edges, {(0, 1)})

    def test_four_synapses_two_pre_groups_produce_two_edges(self):
        # synapses 0,1 share pre-root A; synapses 2,3 share pre-root B
        pre = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([10, 11, 12, 13], dtype=np.int64)
        edges = build_true_line_graph(pre, post)
        self.assertEqual(edges, {(0, 1), (2, 3)})

    def test_edges_are_canonical_min_max_pairs(self):
        pre = np.array([7, 7], dtype=np.int64)
        post = np.array([99, 99], dtype=np.int64)
        edges = build_true_line_graph(pre, post)
        for a, b in edges:
            self.assertLessEqual(a, b)

    def test_no_cross_root_edges_generated(self):
        # All pre-roots and post-roots are unique — no synapse shares a root
        pre = np.array([1, 2, 3], dtype=np.int64)
        post = np.array([4, 5, 6], dtype=np.int64)
        edges = build_true_line_graph(pre, post)
        self.assertEqual(len(edges), 0)

    def test_empty_input_returns_empty_set(self):
        edges = build_true_line_graph(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        )
        self.assertEqual(edges, set())

    def test_single_synapse_returns_empty_set(self):
        edges = build_true_line_graph(
            np.array([1], dtype=np.int64), np.array([2], dtype=np.int64)
        )
        self.assertEqual(edges, set())

    def test_three_synapses_same_root_produce_three_edges(self):
        pre = np.array([5, 5, 5], dtype=np.int64)
        post = np.array([10, 11, 12], dtype=np.int64)
        edges = build_true_line_graph(pre, post)
        self.assertEqual(edges, {(0, 1), (0, 2), (1, 2)})

    def test_pre_and_post_groups_union(self):
        # synapses 0,1 share pre-root; synapses 1,2 share post-root
        # → two edges: (0,1) from pre and (1,2) from post
        pre = np.array([1, 1, 2], dtype=np.int64)
        post = np.array([10, 11, 11], dtype=np.int64)
        edges = build_true_line_graph(pre, post)
        self.assertIn((0, 1), edges)
        self.assertIn((1, 2), edges)


# ---------------------------------------------------------------------------
# compute_line_graph_f1
# ---------------------------------------------------------------------------

class ComputeLineGraphF1Test(unittest.TestCase):
    def test_perfect_match_gives_f1_one(self):
        edges = {(0, 1), (1, 2)}
        m = compute_line_graph_f1(edges, edges, n_synapses=3)
        self.assertAlmostEqual(m.f1, 1.0)
        self.assertAlmostEqual(m.precision, 1.0)
        self.assertAlmostEqual(m.recall, 1.0)
        self.assertEqual(m.tp, 2)
        self.assertEqual(m.fp, 0)
        self.assertEqual(m.fn, 0)

    def test_empty_estimated_gives_zero_f1(self):
        true_edges = {(0, 1)}
        m = compute_line_graph_f1(true_edges, set(), n_synapses=2)
        self.assertAlmostEqual(m.f1, 0.0)
        self.assertAlmostEqual(m.precision, 0.0)
        self.assertAlmostEqual(m.recall, 0.0)
        self.assertEqual(m.fn, 1)

    def test_empty_true_and_estimated_gives_zero_f1(self):
        m = compute_line_graph_f1(set(), set(), n_synapses=0)
        self.assertAlmostEqual(m.f1, 0.0)

    def test_all_false_positives(self):
        m = compute_line_graph_f1(set(), {(0, 1)}, n_synapses=2)
        self.assertAlmostEqual(m.precision, 0.0)
        self.assertEqual(m.fp, 1)
        self.assertEqual(m.tp, 0)

    def test_partial_overlap(self):
        true = {(0, 1), (1, 2), (0, 2)}
        est = {(0, 1), (1, 2), (2, 3)}
        m = compute_line_graph_f1(true, est, n_synapses=4)
        self.assertEqual(m.tp, 2)
        self.assertEqual(m.fp, 1)
        self.assertEqual(m.fn, 1)
        self.assertAlmostEqual(m.precision, 2 / 3)
        self.assertAlmostEqual(m.recall, 2 / 3)

    def test_n_true_and_estimated_edges_are_set_sizes(self):
        true = {(0, 1), (1, 2)}
        est = {(0, 1), (2, 3)}
        m = compute_line_graph_f1(true, est, n_synapses=4)
        self.assertEqual(m.n_true_edges, 2)
        self.assertEqual(m.n_estimated_edges, 2)

    def test_linegraph_metrics_str_contains_f1(self):
        m = LineGraphMetrics(tp=1, fp=0, fn=0, precision=1.0, recall=1.0, f1=1.0,
                             n_true_edges=1, n_estimated_edges=1, n_synapses=2)
        self.assertIn("F1=1.000", str(m))


# ---------------------------------------------------------------------------
# evaluate (integration: build_true + build_estimated + F1)
# ---------------------------------------------------------------------------

class EvaluateTest(unittest.TestCase):
    def _make_perfect_graph(self):
        # Two neurons, each owning two synapses, matching the ground truth.
        neurons = {
            0: MergedNeuron(neuron_id=0, agent_ids=[0], path_points=np.zeros((1, 3)),
                            synapse_indices=[0, 1], role="pre"),
            1: MergedNeuron(neuron_id=1, agent_ids=[1], path_points=np.zeros((1, 3)),
                            synapse_indices=[2, 3], role="pre"),
        }
        return ConnectivityGraph(neurons=neurons, edges=[], unresolved_synapse_indices=[])

    def test_perfect_reconstruction_gives_f1_one(self):
        graph = self._make_perfect_graph()
        pre = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([10, 11, 12, 13], dtype=np.int64)
        m = evaluate(graph, pre, post)
        self.assertAlmostEqual(m.f1, 1.0)

    def test_empty_graph_gives_zero_f1_when_true_edges_exist(self):
        graph = ConnectivityGraph(neurons={}, edges=[], unresolved_synapse_indices=[0, 1])
        pre = np.array([1, 1], dtype=np.int64)
        post = np.array([2, 3], dtype=np.int64)
        m = evaluate(graph, pre, post)
        self.assertAlmostEqual(m.f1, 0.0)
        self.assertEqual(m.fn, 1)

    def test_over_merged_neuron_produces_false_positives(self):
        # One neuron claims synapses from two different true neurons.
        neurons = {
            0: MergedNeuron(neuron_id=0, agent_ids=[0], path_points=np.zeros((1, 3)),
                            synapse_indices=[0, 1, 2, 3], role="pre"),
        }
        graph = ConnectivityGraph(neurons=neurons, edges=[], unresolved_synapse_indices=[])
        pre = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([10, 11, 12, 13], dtype=np.int64)
        m = evaluate(graph, pre, post)
        # Extra cross-root edges are false positives.
        self.assertGreater(m.fp, 0)
        self.assertLess(m.f1, 1.0)


# ---------------------------------------------------------------------------
# evaluate_from_root_ids
# ---------------------------------------------------------------------------

class EvaluateFromRootIdsTest(unittest.TestCase):
    def test_identical_root_ids_give_perfect_f1(self):
        pre = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([3, 3, 4, 4], dtype=np.int64)
        m = evaluate_from_root_ids(pre, post, pre, post)
        self.assertAlmostEqual(m.f1, 1.0)

    def test_swapped_root_ids_give_zero_f1(self):
        true_pre = np.array([1, 1], dtype=np.int64)
        true_post = np.array([10, 11], dtype=np.int64)
        # Estimated: no synapses share a root at all.
        est_pre = np.array([1, 2], dtype=np.int64)
        est_post = np.array([10, 11], dtype=np.int64)
        m = evaluate_from_root_ids(est_pre, est_post, true_pre, true_post)
        self.assertAlmostEqual(m.f1, 0.0)
        self.assertEqual(m.fn, 1)


if __name__ == "__main__":
    unittest.main()
