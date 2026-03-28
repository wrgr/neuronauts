"""Tests for neuronauts/helpers.py — UnionFind, safe_normalize, pairwise_edges.

These helpers underpin merge, assembly, and evaluation code, so correctness
is critical.
"""

from __future__ import annotations

import unittest

import numpy as np

from neuronauts.helpers import UnionFind, pairwise_edges, safe_normalize


# ---------------------------------------------------------------------------
# UnionFind (integer-indexed)
# ---------------------------------------------------------------------------

class UnionFindIntTest(unittest.TestCase):
    """Tests for the list-backed (0..n-1) UnionFind."""

    def test_singletons(self):
        uf = UnionFind(5)
        for i in range(5):
            self.assertEqual(uf.find(i), i)
        self.assertEqual(len(uf.groups()), 5)

    def test_basic_union(self):
        uf = UnionFind(4)
        self.assertTrue(uf.union(0, 1))
        self.assertTrue(uf.connected(0, 1))
        self.assertFalse(uf.connected(0, 2))

    def test_redundant_union_returns_false(self):
        uf = UnionFind(3)
        uf.union(0, 1)
        self.assertFalse(uf.union(0, 1))

    def test_transitive_union(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(1, 2)
        self.assertTrue(uf.connected(0, 2))
        self.assertFalse(uf.connected(0, 3))

    def test_groups(self):
        uf = UnionFind(6)
        uf.union(0, 1)
        uf.union(2, 3)
        uf.union(4, 5)
        groups = uf.groups()
        self.assertEqual(len(groups), 3)
        group_sets = [frozenset(g) for g in groups]
        self.assertIn(frozenset([0, 1]), group_sets)
        self.assertIn(frozenset([2, 3]), group_sets)
        self.assertIn(frozenset([4, 5]), group_sets)

    def test_group_dict(self):
        uf = UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        gd = uf.group_dict()
        self.assertEqual(len(gd), 2)
        members = [sorted(v) for v in gd.values()]
        self.assertIn([0, 1], members)
        self.assertIn([2, 3], members)

    def test_len(self):
        uf = UnionFind(7)
        self.assertEqual(len(uf), 7)

    def test_contains(self):
        uf = UnionFind(3)
        self.assertIn(0, uf)
        self.assertIn(2, uf)
        self.assertNotIn(3, uf)
        self.assertNotIn(-1, uf)

    def test_large_chain(self):
        n = 1000
        uf = UnionFind(n)
        for i in range(n - 1):
            uf.union(i, i + 1)
        self.assertEqual(len(uf.groups()), 1)
        self.assertTrue(uf.connected(0, n - 1))

    def test_all_singletons_groups(self):
        uf = UnionFind(3)
        groups = sorted(uf.groups())
        self.assertEqual(groups, [[0], [1], [2]])


# ---------------------------------------------------------------------------
# UnionFind (key-backed via from_keys)
# ---------------------------------------------------------------------------

class UnionFindKeyTest(unittest.TestCase):
    """Tests for the dict-backed (arbitrary keys) UnionFind."""

    def test_from_keys_basic(self):
        uf = UnionFind.from_keys([10, 20, 30])
        self.assertEqual(len(uf), 3)
        self.assertIn(10, uf)
        self.assertNotIn(5, uf)

    def test_from_keys_union(self):
        uf = UnionFind.from_keys([100, 200, 300])
        uf.union(100, 300)
        self.assertTrue(uf.connected(100, 300))
        self.assertFalse(uf.connected(100, 200))

    def test_from_keys_groups(self):
        uf = UnionFind.from_keys([5, 10, 15, 20])
        uf.union(5, 15)
        uf.union(10, 20)
        groups = uf.groups()
        self.assertEqual(len(groups), 2)
        group_sets = [frozenset(g) for g in groups]
        self.assertIn(frozenset([5, 15]), group_sets)
        self.assertIn(frozenset([10, 20]), group_sets)

    def test_from_keys_group_dict(self):
        uf = UnionFind.from_keys([1, 2, 3])
        uf.union(1, 3)
        gd = uf.group_dict()
        self.assertEqual(len(gd), 2)

    def test_from_generator(self):
        uf = UnionFind.from_keys(x * 10 for x in range(4))
        self.assertEqual(len(uf), 4)
        uf.union(0, 30)
        self.assertTrue(uf.connected(0, 30))


# ---------------------------------------------------------------------------
# safe_normalize
# ---------------------------------------------------------------------------

class SafeNormalizeTest(unittest.TestCase):
    def test_unit_vectors_unchanged(self):
        v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        normed = safe_normalize(v, axis=1)
        np.testing.assert_allclose(normed, v, atol=1e-7)

    def test_zero_vectors_stay_zero(self):
        v = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        normed = safe_normalize(v, axis=1)
        np.testing.assert_allclose(normed[0], [0, 0, 0])
        np.testing.assert_allclose(np.linalg.norm(normed[1]), 1.0, atol=1e-7)

    def test_batch_normalization(self):
        rng = np.random.default_rng(42)
        v = rng.standard_normal((100, 3)).astype(np.float32)
        normed = safe_normalize(v, axis=1)
        norms = np.linalg.norm(normed, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_1d_vector(self):
        v = np.array([3.0, 4.0, 0.0])
        normed = safe_normalize(v, axis=-1)
        np.testing.assert_allclose(np.linalg.norm(normed), 1.0, atol=1e-7)

    def test_3d_field_normalization(self):
        # Simulate a (2, 2, 2, 3) vector field
        rng = np.random.default_rng(7)
        field = rng.standard_normal((2, 2, 2, 3))
        normed = safe_normalize(field, axis=-1)
        norms = np.linalg.norm(normed, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-7)

    def test_custom_eps(self):
        v = np.array([[1e-6, 0, 0]], dtype=np.float64)
        # With default eps=1e-8, this should normalize
        normed_default = safe_normalize(v, axis=1)
        self.assertGreater(np.linalg.norm(normed_default), 0.5)
        # With eps=1e-4, this should become zero
        normed_large_eps = safe_normalize(v, axis=1, eps=1e-4)
        np.testing.assert_allclose(normed_large_eps, 0.0, atol=1e-10)

    def test_preserves_direction(self):
        v = np.array([[3.0, 4.0, 0.0]])
        normed = safe_normalize(v, axis=1)
        expected = np.array([[0.6, 0.8, 0.0]])
        np.testing.assert_allclose(normed, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# pairwise_edges
# ---------------------------------------------------------------------------

class PairwiseEdgesTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(pairwise_edges([]), set())

    def test_single(self):
        self.assertEqual(pairwise_edges([5]), set())

    def test_two(self):
        self.assertEqual(pairwise_edges([3, 1]), {(1, 3)})

    def test_three(self):
        result = pairwise_edges([3, 1, 2])
        self.assertEqual(result, {(1, 2), (1, 3), (2, 3)})

    def test_duplicates_ignored(self):
        result = pairwise_edges([1, 1, 2, 2, 3])
        self.assertEqual(result, {(1, 2), (1, 3), (2, 3)})

    def test_canonical_ordering(self):
        result = pairwise_edges([10, 5])
        self.assertEqual(result, {(5, 10)})

    def test_count_for_n(self):
        # n choose 2
        for n in range(2, 8):
            result = pairwise_edges(range(n))
            self.assertEqual(len(result), n * (n - 1) // 2)


if __name__ == "__main__":
    unittest.main()
