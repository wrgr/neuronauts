"""Tests for the line-graph F1 metric — the primary evaluation scalar.

These cover build_true_line_graph, compute_line_graph_f1, evaluate,
evaluate_from_root_ids, and the full evaluate_suite with all four variants.
"""

import unittest

import numpy as np

from neuronauts.line_graph import (
    LineGraphMetrics,
    LineGraphSuite,
    build_estimated_line_graph,
    build_true_line_graph,
    build_true_pairs_and,
    build_true_pairs_post,
    build_true_pairs_pre,
    compute_sampled_line_graph_f1,
    compute_line_graph_f1,
    evaluate,
    evaluate_sampled,
    evaluate_from_root_ids,
    evaluate_suite,
    sample_synapse_pairs,
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


class SampledLineGraphMetricsTest(unittest.TestCase):
    def test_sample_synapse_pairs_caps_pair_count(self):
        pairs = sample_synapse_pairs(10, max_pairs=7, seed=0)
        self.assertEqual(len(pairs), 7)
        self.assertTrue(all(i < j for i, j in pairs))

    def test_compute_sampled_line_graph_f1_matches_full_when_sampling_all_pairs(self):
        true = {(0, 1), (1, 2)}
        est = {(0, 1), (2, 3)}
        full = compute_line_graph_f1(true, est, n_synapses=4)
        sampled = compute_sampled_line_graph_f1(true, est, n_synapses=4, max_pairs=100)
        self.assertEqual(sampled.tp, full.tp)
        self.assertEqual(sampled.fp, full.fp)
        self.assertEqual(sampled.fn, full.fn)
        self.assertAlmostEqual(sampled.f1, full.f1)

    def test_evaluate_sampled_returns_metrics(self):
        neurons = {
            0: MergedNeuron(neuron_id=0, agent_ids=[0], path_points=np.zeros((1, 3)),
                            synapse_indices=[0, 1], role="pre"),
        }
        graph = ConnectivityGraph(neurons=neurons, edges=[], unresolved_synapse_indices=[])
        pre = np.array([1, 1, 2], dtype=np.int64)
        post = np.array([3, 4, 5], dtype=np.int64)
        m = evaluate_sampled(graph, pre, post, max_pairs=3, seed=1)
        self.assertIsInstance(m, LineGraphMetrics)


# ---------------------------------------------------------------------------
# build_true_pairs_pre / post / and
# ---------------------------------------------------------------------------

class BuildTruePairsVariantsTest(unittest.TestCase):
    def test_build_true_pairs_pre_matches_same_pre_root_only(self):
        pre = np.array([1, 1, 2], dtype=np.int64)
        edges = build_true_pairs_pre(pre)
        self.assertEqual(edges, {(0, 1)})

    def test_build_true_pairs_post_matches_same_post_root_only(self):
        post = np.array([10, 11, 11], dtype=np.int64)
        edges = build_true_pairs_post(post)
        self.assertEqual(edges, {(1, 2)})

    def test_build_true_pairs_and_requires_both_match(self):
        # synapses 0,1: same pre (root 1) but different post → no AND edge
        # synapses 1,2: same post (root 11) but different pre → no AND edge
        # synapses 0,1 both: same pre AND same post only if both match
        pre  = np.array([1, 1, 2], dtype=np.int64)
        post = np.array([10, 11, 11], dtype=np.int64)
        edges = build_true_pairs_and(pre, post)
        self.assertEqual(edges, set())  # no pair shares both pre AND post

    def test_build_true_pairs_and_same_circuit_edge(self):
        # synapses 0 and 2 both have pre=1, post=10 → one AND edge
        pre  = np.array([1, 2, 1], dtype=np.int64)
        post = np.array([10, 10, 10], dtype=np.int64)
        edges = build_true_pairs_and(pre, post)
        self.assertEqual(edges, {(0, 2)})

    def test_build_true_pairs_and_subset_of_or(self):
        # OR includes pre-only or post-only matches; AND is the intersection
        pre  = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([10, 11, 10, 11], dtype=np.int64)
        or_edges  = build_true_line_graph(pre, post)
        and_edges = build_true_pairs_and(pre, post)
        self.assertTrue(and_edges.issubset(or_edges))

    def test_build_true_pairs_and_empty_when_all_unique_pairs(self):
        pre  = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([10, 11, 10, 11], dtype=np.int64)
        # Each (pre, post) pair appears exactly once → no AND edge
        edges = build_true_pairs_and(pre, post)
        self.assertEqual(edges, set())


# ---------------------------------------------------------------------------
# evaluate_suite — single-side (no pred_post)
# ---------------------------------------------------------------------------

class EvaluateSuiteSingleSideTest(unittest.TestCase):
    def _make_pre_post(self):
        pre  = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([10, 11, 12, 13], dtype=np.int64)
        return pre, post

    def test_single_side_returns_none_for_post_and_and(self):
        pred = np.array([0, 0, 1, 1])
        pre, post = self._make_pre_post()
        suite = evaluate_suite(pred, pre, post)
        self.assertIsInstance(suite, LineGraphSuite)
        self.assertIsNone(suite.post_only)
        self.assertIsNone(suite.and_metric)

    def test_single_side_perfect_pre_partition_gives_pre_only_f1_one(self):
        pred = np.array([0, 0, 1, 1])
        pre, post = self._make_pre_post()
        suite = evaluate_suite(pred, pre, post)
        self.assertAlmostEqual(suite.pre_only.f1, 1.0)

    def test_single_side_or_metric_includes_post_true_edges(self):
        # pre: synapses 0,1 share root; post: all unique → OR same as pre here
        pred = np.array([0, 0, 1, 1])
        pre  = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([10, 11, 12, 13], dtype=np.int64)
        suite = evaluate_suite(pred, pre, post)
        # OR true-edges = pre-edges (no post-sharing) → same as pre_only here
        self.assertAlmostEqual(suite.or_metric.f1, suite.pre_only.f1)

    def test_single_side_over_fragmented_pred_lowers_pre_only_recall(self):
        # Pred never-merges: each synapse is its own cluster → recall=0
        pred = np.array([0, 1, 2, 3])
        pre  = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([10, 11, 12, 13], dtype=np.int64)
        suite = evaluate_suite(pred, pre, post)
        self.assertAlmostEqual(suite.pre_only.recall, 0.0)

    def test_or_metric_insensitive_to_overfragmentation_when_post_matches(self):
        # Pre: all different; Post: synapses 0,1 share post-root 99.
        # OR true-edges = {(0,1)} from post.
        # Pred merges 0+1 → OR recall=1; pre_only has no true edges → f1=0.
        pred = np.array([0, 0, 1, 2])
        pre  = np.array([1, 2, 3, 4], dtype=np.int64)
        post = np.array([99, 99, 5, 6], dtype=np.int64)
        suite = evaluate_suite(pred, pre, post)
        # pre_only: no true pre-edges → no tp, no fn; precision=0 (fp exists), recall=0, f1=0
        self.assertAlmostEqual(suite.pre_only.f1, 0.0)
        # or_metric should see the post-side true edge (0,1) as matched
        self.assertAlmostEqual(suite.or_metric.recall, 1.0)


# ---------------------------------------------------------------------------
# evaluate_suite — dual-side (pred_post provided)
# ---------------------------------------------------------------------------

class EvaluateSuiteDualSideTest(unittest.TestCase):
    def _perfect_setup(self):
        # 4 synapses: pre neurons A(1) and B(2); post neurons X(10) and Y(11).
        # Synapses 0,1 share pre=1 and post=10; synapses 2,3 share pre=2 and post=11.
        # Perfect partitions mirror the true grouping.
        pre  = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([10, 10, 11, 11], dtype=np.int64)
        pred_pre  = np.array([0, 0, 1, 1])
        pred_post = np.array([0, 0, 1, 1])
        return pred_pre, pred_post, pre, post

    def test_perfect_dual_partition_all_f1_one(self):
        pred_pre, pred_post, pre, post = self._perfect_setup()
        suite = evaluate_suite(pred_pre, pre, post, pred_post=pred_post)
        self.assertAlmostEqual(suite.pre_only.f1, 1.0)
        self.assertAlmostEqual(suite.or_metric.f1, 1.0)
        self.assertAlmostEqual(suite.post_only.f1, 1.0)
        self.assertAlmostEqual(suite.and_metric.f1, 1.0)

    def test_dual_side_all_four_metrics_present(self):
        pred_pre, pred_post, pre, post = self._perfect_setup()
        suite = evaluate_suite(pred_pre, pre, post, pred_post=pred_post)
        self.assertIsNotNone(suite.post_only)
        self.assertIsNotNone(suite.and_metric)

    def test_and_metric_never_merge_gives_recall_zero(self):
        # Never-merge: every synapse in its own cluster on BOTH sides.
        # No two synapses share (pred_pre, pred_post) → no est AND edges → recall=0.
        n = 4
        pred_pre  = np.arange(n)
        pred_post = np.arange(n)
        pre  = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([10, 11, 12, 13], dtype=np.int64)
        suite = evaluate_suite(pred_pre, pre, post, pred_post=pred_post)
        self.assertAlmostEqual(suite.and_metric.recall, 0.0)

    def test_or_metric_not_penalised_by_never_merge_when_post_compensates(self):
        # synapses 0,1 share post-root 99; pred_pre never-merges them but
        # the OR true-edge (0,1) is still a false-negative → or_metric recall < 1.
        # Contrast: if pred_pre DID merge them, or_metric recall=1.
        pred_pre  = np.array([0, 1, 2, 3])  # never-merge
        pred_post = np.array([0, 1, 2, 3])  # never-merge
        pre  = np.array([1, 2, 3, 4], dtype=np.int64)  # all unique pre
        post = np.array([99, 99, 5, 6], dtype=np.int64)
        suite_nm = evaluate_suite(pred_pre, pre, post, pred_post=pred_post)
        # True OR edge is (0,1) from post; est_pre has no (0,1) → fn=1
        self.assertEqual(suite_nm.or_metric.fn, 1)

    def test_and_metric_penalises_overmerge_on_pre_side(self):
        # Pre: true neurons A (0,1) and B (2,3); pred merges everything into one.
        # Post: two true groups (10,10,11,11); pred post is correct (0,0,1,1).
        # True AND-edges: {(0,1)} (pre=1,post=10) and {(2,3)} (pre=2,post=11).
        # Est AND-edges: (0,0)→{0,1}→edge(0,1); (0,1)→{2,3}→edge(2,3)
        #                but also cross-group false merges introduced by pred_pre=0 for all.
        # Actually: est_and = _pairs_from_joint([0,0,0,0],[0,0,1,1])
        #   key(0,0) = [0,1] → (0,1); key(0,1) = [2,3] → (2,3)
        #   Those happen to equal the true AND-edges, so f1=1 here.
        # Instead: pred merges groups that belong to different true circuit edges.
        # Use 4 synapses: true circuit edges A→X (idx 0,2) and B→Y (idx 1,3).
        # pred_pre merges A+B; pred_post is correct: group X (idx 0,2) and Y (idx 1,3).
        # Est AND-edges: key(0,0)=[0,2]→(0,2); key(0,1)=[1,3]→(1,3)
        # True AND-edges from _pairs_from_joint(pre,post):
        #   (pre=1,post=10)=[0,2]→(0,2); (pre=2,post=11)=[1,3]→(1,3) → same!
        # That's still perfect. Let's use a simpler failure mode:
        # pred_pre merges A+B (all 0); pred_post also all 0 (merges X+Y).
        # Est AND-edges: key(0,0)=[0,1,2,3] → 6 pairs, all FP (true AND has 0 pairs
        # if all post are unique, or 2 if we use the _perfect_setup arrangement).
        # Use unique post-roots so true AND = empty → all est pairs are FP.
        pred_pre  = np.array([0, 0, 0, 0])   # merges everyone on pre
        pred_post = np.array([0, 0, 0, 0])   # merges everyone on post
        pre  = np.array([1, 1, 2, 2], dtype=np.int64)
        post = np.array([10, 11, 12, 13], dtype=np.int64)  # all unique → no true AND edges
        suite = evaluate_suite(pred_pre, pre, post, pred_post=pred_post)
        # est_and: key(0,0)=[0,1,2,3] → 6 pairs; true_and={}
        self.assertAlmostEqual(suite.and_metric.precision, 0.0)
        self.assertEqual(suite.and_metric.fp, 6)
        self.assertEqual(suite.and_metric.fn, 0)  # no true AND edges to miss

    def test_and_metric_with_shared_circuit_edge(self):
        # 3 synapses: (pre=1,post=10), (pre=1,post=10), (pre=2,post=11)
        # True AND-edges: {(0,1)} (shared circuit edge 1→10)
        # Perfect pred → AND F1 = 1.0
        pre  = np.array([1, 1, 2], dtype=np.int64)
        post = np.array([10, 10, 11], dtype=np.int64)
        pred_pre  = np.array([0, 0, 1])
        pred_post = np.array([0, 0, 1])
        suite = evaluate_suite(pred_pre, pre, post, pred_post=pred_post)
        self.assertAlmostEqual(suite.and_metric.f1, 1.0)
        self.assertEqual(suite.and_metric.tp, 1)

    def test_post_only_uses_post_partition_not_pre(self):
        # Post: synapses 0,1 share post-root 10; pred_post merges them correctly.
        # pred_pre incorrectly merges 0+2 (different post-roots).
        # post_only should reflect only pred_post quality.
        pre  = np.array([1, 2, 3], dtype=np.int64)
        post = np.array([10, 10, 11], dtype=np.int64)
        pred_pre  = np.array([0, 1, 0])  # incorrect grouping for pre
        pred_post = np.array([0, 0, 1])  # correct post grouping
        suite = evaluate_suite(pred_pre, pre, post, pred_post=pred_post)
        self.assertAlmostEqual(suite.post_only.f1, 1.0)

    def test_linegraph_suite_is_dataclass(self):
        pred_pre, pred_post, pre, post = _make_perfect_setup()
        suite = evaluate_suite(pred_pre, pre, post, pred_post=pred_post)
        self.assertIsInstance(suite, LineGraphSuite)
        self.assertIsInstance(suite.pre_only, LineGraphMetrics)
        self.assertIsInstance(suite.or_metric, LineGraphMetrics)
        self.assertIsInstance(suite.post_only, LineGraphMetrics)
        self.assertIsInstance(suite.and_metric, LineGraphMetrics)


def _make_perfect_setup():
    pre  = np.array([1, 1, 2, 2], dtype=np.int64)
    post = np.array([10, 10, 11, 11], dtype=np.int64)
    pred_pre  = np.array([0, 0, 1, 1])
    pred_post = np.array([0, 0, 1, 1])
    return pred_pre, pred_post, pre, post


if __name__ == "__main__":
    unittest.main()
