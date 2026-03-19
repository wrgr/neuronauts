"""Tests for experiments/soma_graph — soma-level neuron × neuron graph."""

import unittest

import numpy as np

from experiments.soma_graph.build_graph import SomaGraph, build_soma_graph_from_synapses

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BuildSomaGraphTest(unittest.TestCase):
    def test_empty_synapses_raises_or_returns_minimal(self):
        pre = np.array([], dtype=np.int64)
        post = np.array([], dtype=np.int64)
        graph = build_soma_graph_from_synapses(pre, post)
        self.assertEqual(graph.n_nodes, 0)
        self.assertEqual(graph.n_edges, 0)

    def test_single_synapse_pair(self):
        pre = np.array([100], dtype=np.int64)
        post = np.array([200], dtype=np.int64)
        graph = build_soma_graph_from_synapses(pre, post)
        self.assertEqual(graph.n_nodes, 2)
        self.assertEqual(graph.n_edges, 1)
        self.assertIn(100, graph.node_ids)
        self.assertIn(200, graph.node_ids)
        self.assertEqual(graph.edge_synapse_count[0], 1)

    def test_aggregates_duplicate_edges(self):
        pre = np.array([100, 100, 100], dtype=np.int64)
        post = np.array([200, 200, 200], dtype=np.int64)
        graph = build_soma_graph_from_synapses(pre, post)
        self.assertEqual(graph.n_nodes, 2)
        self.assertEqual(graph.n_edges, 1)
        self.assertEqual(graph.edge_synapse_count[0], 3)

    def test_filters_invalid_roots(self):
        pre = np.array([0, 100, 100], dtype=np.int64)
        post = np.array([200, 0, 200], dtype=np.int64)
        graph = build_soma_graph_from_synapses(pre, post)
        # Only (100, 200) is valid
        self.assertEqual(graph.n_nodes, 2)
        self.assertEqual(graph.n_edges, 1)

    def test_filters_self_loops(self):
        pre = np.array([100], dtype=np.int64)
        post = np.array([100], dtype=np.int64)
        graph = build_soma_graph_from_synapses(pre, post)
        self.assertEqual(graph.n_nodes, 0)
        self.assertEqual(graph.n_edges, 0)

    def test_node_features_shape(self):
        pre = np.array([1, 2, 2, 3], dtype=np.int64)
        post = np.array([2, 1, 3, 1], dtype=np.int64)
        graph = build_soma_graph_from_synapses(pre, post, node_feat_dim=16)
        self.assertEqual(graph.node_features.shape, (3, 16))

    def test_zero_features_when_seed_none(self):
        pre = np.array([1, 2], dtype=np.int64)
        post = np.array([2, 1], dtype=np.int64)
        graph = build_soma_graph_from_synapses(pre, post, feature_seed=None)
        np.testing.assert_array_equal(graph.node_features, 0.0)


@unittest.skipIf(not HAS_TORCH, "torch not installed (need pip install -e .[topology])")
class SmokeTest(unittest.TestCase):
    def test_smoke_run(self):
        from experiments.soma_graph.smoke_test import run_smoke_test

        self.assertTrue(run_smoke_test())
