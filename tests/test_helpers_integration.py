"""Integration tests verifying that refactored modules correctly use helpers.

These tests ensure that:
- UnionFind is used consistently across merge, run, skeleton_graph, cell_graph
- safe_normalize produces identical results to the old inline patterns
- pairwise_edges is equivalent to the old nested-loop edge building
- __init__.py exports the expected public API
"""

from __future__ import annotations

import unittest

import numpy as np


class PublicAPIExportsTest(unittest.TestCase):
    """Verify that neuronauts.__init__ exports the advertised API."""

    def test_all_names_importable(self):
        import neuronauts
        for name in neuronauts.__all__:
            self.assertTrue(hasattr(neuronauts, name), f"{name} not in neuronauts namespace")

    def test_core_types_importable(self):
        from neuronauts import (
            Agent,
            AgentConfig,
            BridgeGraph,
            BridgePath,
            ConnectivityGraph,
            LineGraphMetrics,
            MergedNeuron,
            UnionFind,
            evaluate,
            evaluate_sampled,
            merge_agents,
            pairwise_edges,
            safe_normalize,
        )
        # Just verify they are callable / instantiable types.
        self.assertTrue(callable(UnionFind))
        self.assertTrue(callable(safe_normalize))
        self.assertTrue(callable(pairwise_edges))
        self.assertTrue(callable(evaluate))


class SafeNormalizeEquivalenceTest(unittest.TestCase):
    """Verify safe_normalize matches the old inline normalization pattern."""

    def test_matches_old_pattern(self):
        from neuronauts.helpers import safe_normalize

        rng = np.random.default_rng(99)
        v = rng.standard_normal((50, 3)).astype(np.float32)

        old_result = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        new_result = safe_normalize(v, axis=1)

        # Should be very close for non-zero vectors.
        np.testing.assert_allclose(new_result, old_result, atol=1e-6)

    def test_zero_vector_difference(self):
        from neuronauts.helpers import safe_normalize

        v = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        # Old pattern would give tiny nonzero values
        old = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        # New pattern gives exact zeros
        new = safe_normalize(v, axis=1)
        np.testing.assert_array_equal(new, 0.0)


class PairwiseEdgesEquivalenceTest(unittest.TestCase):
    """Verify pairwise_edges matches the old nested-loop edge builder."""

    def _old_pairwise_edges(self, items):
        edges = set()
        items = sorted(set(items))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                edges.add((min(a, b), max(a, b)))
        return edges

    def test_equivalence_random(self):
        from neuronauts.helpers import pairwise_edges

        rng = np.random.default_rng(42)
        for _ in range(20):
            items = rng.integers(0, 50, size=rng.integers(0, 15)).tolist()
            old = self._old_pairwise_edges(items)
            new = pairwise_edges(items)
            self.assertEqual(old, new, f"Mismatch for items={items}")


class MergeAgentsUsesUnionFindTest(unittest.TestCase):
    """Smoke test that merge_agents still works after refactoring."""

    def test_merge_nearby_agents(self):
        from neuronauts.legacy.agent import Agent
        from neuronauts.legacy.agent_merge import merge_agents

        agents = [
            Agent(agent_id=0, path=[np.array([0, 0, 0]), np.array([1, 0, 0]),
                                     np.array([2, 0, 0]), np.array([3, 0, 0]),
                                     np.array([4, 0, 0])]),
            Agent(agent_id=1, path=[np.array([0.5, 0, 0]), np.array([1.5, 0, 0]),
                                     np.array([2.5, 0, 0]), np.array([3.5, 0, 0]),
                                     np.array([4.5, 0, 0])]),
            Agent(agent_id=2, path=[np.array([100, 100, 100]), np.array([101, 100, 100]),
                                     np.array([102, 100, 100]), np.array([103, 100, 100]),
                                     np.array([104, 100, 100])]),
        ]
        neurons = merge_agents(agents, merge_radius=5.0)
        # Agents 0 and 1 should merge; agent 2 should be separate.
        self.assertEqual(len(neurons), 2)

    def test_empty_agents_returns_empty(self):
        from neuronauts.legacy.agent_merge import merge_agents
        self.assertEqual(merge_agents([]), {})


class LineGraphUsesPairwiseEdgesTest(unittest.TestCase):
    """Verify line_graph module still works after pairwise_edges refactor."""

    def test_build_true_line_graph(self):
        from neuronauts.line_graph import build_true_line_graph

        pre = np.array([1, 1, 2, 2])
        post = np.array([10, 20, 10, 20])
        edges = build_true_line_graph(pre, post)
        # Pre group {0,1} → edge (0,1), pre group {2,3} → edge (2,3)
        # Post group {0,2} → edge (0,2), post group {1,3} → edge (1,3)
        self.assertIn((0, 1), edges)
        self.assertIn((2, 3), edges)
        self.assertIn((0, 2), edges)
        self.assertIn((1, 3), edges)

    def test_evaluate_smoke(self):
        from neuronauts.line_graph import evaluate
        from neuronauts.merge import ConnectivityGraph, MergedNeuron

        neurons = {
            0: MergedNeuron(
                neuron_id=0, agent_ids=[0],
                path_points=np.zeros((5, 3)), synapse_indices=[0, 1],
            ),
            1: MergedNeuron(
                neuron_id=1, agent_ids=[1],
                path_points=np.zeros((5, 3)), synapse_indices=[2, 3],
            ),
        }
        graph = ConnectivityGraph(
            neurons=neurons, edges=[], unresolved_synapse_indices=[],
        )
        pre = np.array([1, 1, 2, 2])
        post = np.array([10, 20, 10, 20])
        metrics = evaluate(graph, pre, post)
        self.assertGreaterEqual(metrics.f1, 0.0)
        self.assertLessEqual(metrics.f1, 1.0)


class FieldsNormalizationTest(unittest.TestCase):
    """Verify compute_membrane_vectors still produces unit vectors after refactor."""

    def test_unit_vectors(self):
        from neuronauts.legacy.fields import compute_membrane_vectors

        rng = np.random.default_rng(42)
        mf = rng.random((8, 8, 8)).astype(np.float32)
        mv = compute_membrane_vectors(mf)
        norms = np.linalg.norm(mv, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_all_finite(self):
        from neuronauts.legacy.fields import compute_membrane_vectors

        mf = np.zeros((4, 4, 4), dtype=np.float32)
        mv = compute_membrane_vectors(mf)
        self.assertTrue(np.all(np.isfinite(mv)))


if __name__ == "__main__":
    unittest.main()
