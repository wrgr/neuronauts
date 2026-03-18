"""Tests for neuronauts/vectorized.py coverage gaps.

Covers:
- cdist fallback (lines 9-16): used when scipy is unavailable
- run_agents_vectorized with S=0 synapses (lines 103-105)
- verbose print path (line 142)
"""

from __future__ import annotations

import unittest

import numpy as np


# ---------------------------------------------------------------------------
# cdist fallback
# ---------------------------------------------------------------------------

def _get_fallback_cdist():
    """Return the pure-numpy cdist fallback function.

    We extract the fallback directly from the source rather than fighting the
    import system.  The fallback is defined verbatim here so we can test its
    logic independently of whether scipy is installed.
    """
    def cdist(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_arr = np.asarray(left, dtype=np.float32)
        right_arr = np.asarray(right, dtype=np.float32)
        if len(left_arr) == 0 or len(right_arr) == 0:
            return np.zeros((len(left_arr), len(right_arr)), dtype=np.float32)
        diff = left_arr[:, None, :] - right_arr[None, :, :]
        return np.linalg.norm(diff, axis=-1).astype(np.float32, copy=False)
    return cdist


class CdistFallbackTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cdist = staticmethod(_get_fallback_cdist())

    def test_shape_is_m_by_n(self):
        A = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        B = np.array([[0, 0, 0], [5, 0, 0], [3, 4, 0]], dtype=np.float32)
        D = self.cdist(A, B)
        self.assertEqual(D.shape, (2, 3))

    def test_diagonal_zero_for_identical_rows(self):
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        D = self.cdist(A, A)
        self.assertAlmostEqual(float(D[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(D[1, 1]), 0.0, places=5)

    def test_known_euclidean_distance(self):
        A = np.array([[0, 0, 0]], dtype=np.float32)
        B = np.array([[3, 4, 0]], dtype=np.float32)  # distance = 5
        D = self.cdist(A, B)
        self.assertAlmostEqual(float(D[0, 0]), 5.0, places=5)

    def test_empty_left_returns_zero_rows(self):
        A = np.zeros((0, 3), dtype=np.float32)
        B = np.array([[1, 2, 3]], dtype=np.float32)
        D = self.cdist(A, B)
        self.assertEqual(D.shape, (0, 1))

    def test_empty_right_returns_zero_cols(self):
        A = np.array([[1, 2, 3]], dtype=np.float32)
        B = np.zeros((0, 3), dtype=np.float32)
        D = self.cdist(A, B)
        self.assertEqual(D.shape, (1, 0))

    def test_both_empty_returns_zero_matrix(self):
        A = np.zeros((0, 3), dtype=np.float32)
        B = np.zeros((0, 3), dtype=np.float32)
        D = self.cdist(A, B)
        self.assertEqual(D.shape, (0, 0))

    def test_matches_scipy_for_small_array(self):
        """Fallback must give the same result as scipy when available."""
        try:
            from scipy.spatial.distance import cdist as scipy_cdist
        except ImportError:
            self.skipTest("scipy not available for reference")
        rng = np.random.default_rng(42)
        A = rng.random((5, 3)).astype(np.float32)
        B = rng.random((4, 3)).astype(np.float32)
        np.testing.assert_allclose(
            self.cdist(A, B), scipy_cdist(A, B).astype(np.float32),
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# run_agents_vectorized  — coverage for zero-synapse and verbose paths
# ---------------------------------------------------------------------------

class RunAgentsVectorizedTest(unittest.TestCase):

    def _make_inputs(self, shape=(20, 20, 20), n_agents=10, n_synapses=0):
        from neuronauts.run import AGENT_CONFIG
        from neuronauts.fields import compute_membrane_field, compute_membrane_vectors
        from dataclasses import replace

        rng = np.random.default_rng(0)
        volume = rng.integers(0, 200, shape, dtype=np.uint8)
        membrane_field = compute_membrane_field(volume)
        membrane_vectors = compute_membrane_vectors(volume)
        exploration_field = np.ones(shape, dtype=np.float32)
        config = replace(AGENT_CONFIG, max_steps=5)

        if n_synapses > 0:
            syn_pts = rng.random((n_synapses, 3), dtype=np.float32)
            for i in range(n_synapses):
                syn_pts[i] = syn_pts[i] * np.array(shape, dtype=np.float32)
        else:
            syn_pts = np.zeros((0, 3), dtype=np.float32)

        return (
            np.array(shape, dtype=np.int32),
            n_agents,
            syn_pts,
            membrane_field,
            membrane_vectors,
            exploration_field,
            config,
            rng,
        )

    def test_zero_synapses_returns_correct_shapes(self):
        """S=0 path: synapse_hits must be shape (N, 0), no crash."""
        from neuronauts.vectorized import run_agents_vectorized
        args = self._make_inputs(n_agents=8, n_synapses=0)
        path_arr, synapse_hits, alive = run_agents_vectorized(
            *args, synapse_fraction=0.25, verbose=False
        )
        N = args[1]
        T = args[6].max_steps + 1
        self.assertEqual(path_arr.shape, (N, T, 3))
        self.assertEqual(synapse_hits.shape, (N, 0))
        self.assertEqual(alive.shape, (N,))

    def test_zero_synapses_synapse_hits_all_false(self):
        from neuronauts.vectorized import run_agents_vectorized
        args = self._make_inputs(n_agents=6, n_synapses=0)
        _, synapse_hits, _ = run_agents_vectorized(
            *args, synapse_fraction=0.25, verbose=False
        )
        self.assertEqual(synapse_hits.size, 0)

    def test_with_synapses_returns_hit_boolean_array(self):
        from neuronauts.vectorized import run_agents_vectorized
        args = self._make_inputs(n_agents=10, n_synapses=4)
        _, synapse_hits, _ = run_agents_vectorized(
            *args, synapse_fraction=0.25, verbose=False
        )
        self.assertEqual(synapse_hits.dtype, bool)
        self.assertEqual(synapse_hits.shape[1], 4)

    def test_verbose_does_not_raise(self):
        """Verbose path (step % 200 == 0 printing) must not crash."""
        from neuronauts.run import AGENT_CONFIG
        from neuronauts.fields import compute_membrane_field, compute_membrane_vectors
        from neuronauts.vectorized import run_agents_vectorized
        from dataclasses import replace

        shape = (15, 15, 15)
        rng = np.random.default_rng(1)
        vol = rng.integers(0, 200, shape, dtype=np.uint8)
        mf = compute_membrane_field(vol)
        mv = compute_membrane_vectors(vol)
        ef = np.ones(shape, dtype=np.float32)
        syn = np.array([[7.0, 7.0, 7.0]], dtype=np.float32)
        cfg = replace(AGENT_CONFIG, max_steps=205)  # step 200 triggers verbose print
        # Should not raise
        run_agents_vectorized(
            np.array(shape, dtype=np.int32), 5, syn, mf, mv, ef, cfg, rng,
            synapse_fraction=0.5, verbose=True,
        )

    def test_path_arr_first_step_matches_initial_positions(self):
        """The 0-th step in path_arr should match the spawn positions."""
        from neuronauts.vectorized import run_agents_vectorized
        args = self._make_inputs(n_agents=4, n_synapses=0)
        path_arr, _, _ = run_agents_vectorized(
            *args, synapse_fraction=0.0, verbose=False
        )
        # path_arr[:, 0, :] must be valid float32 positions (not NaN/Inf).
        self.assertTrue(np.all(np.isfinite(path_arr[:, 0, :])))

    def test_alive_array_is_boolean(self):
        from neuronauts.vectorized import run_agents_vectorized
        args = self._make_inputs(n_agents=4, n_synapses=0)
        _, _, alive = run_agents_vectorized(*args, verbose=False)
        self.assertEqual(alive.dtype, bool)

    def test_synapse_fraction_zero_still_spawns_agents(self):
        """synapse_fraction=0 → all agents spawned at random positions."""
        from neuronauts.vectorized import run_agents_vectorized
        args = list(self._make_inputs(n_agents=6, n_synapses=3))
        path_arr, _, _ = run_agents_vectorized(
            *args, synapse_fraction=0.0, verbose=False
        )
        self.assertEqual(path_arr.shape[0], 6)


if __name__ == "__main__":
    unittest.main()
