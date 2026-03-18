"""Tests for neuronauts/fields.py.

compute_membrane_field and compute_membrane_vectors are in the hot path of
every benchmark run — a normalisation or sign error would silently degrade
all reported F1 scores.
"""

import unittest

import numpy as np

from neuronauts.fields import (
    compute_exploration_field,
    compute_membrane_field,
    compute_membrane_vectors,
    compute_synapse_attraction_field,
    sample_field_trilinear,
    sample_vector_field_trilinear,
)


class ComputeMembraneFieldTest(unittest.TestCase):
    def test_output_shape_matches_input(self):
        vol = np.random.default_rng(0).integers(0, 255, (16, 16, 16), dtype=np.uint8)
        mf = compute_membrane_field(vol)
        self.assertEqual(mf.shape, vol.shape)

    def test_output_is_float32(self):
        vol = np.ones((8, 8, 8), dtype=np.uint8)
        mf = compute_membrane_field(vol)
        self.assertEqual(mf.dtype, np.float32)

    def test_normalized_output_in_zero_one(self):
        rng = np.random.default_rng(1)
        vol = rng.integers(0, 255, (12, 12, 12), dtype=np.uint8)
        mf = compute_membrane_field(vol, normalize=True)
        self.assertGreaterEqual(float(mf.min()), 0.0)
        self.assertLessEqual(float(mf.max()), 1.0 + 1e-6)

    def test_max_is_one_when_normalized(self):
        rng = np.random.default_rng(2)
        vol = rng.integers(10, 245, (10, 10, 10), dtype=np.uint8)
        mf = compute_membrane_field(vol, normalize=True)
        self.assertAlmostEqual(float(mf.max()), 1.0, places=5)

    def test_uniform_volume_gives_zero_field(self):
        # A completely flat volume has zero gradient everywhere.
        vol = np.full((8, 8, 8), 128, dtype=np.uint8)
        mf = compute_membrane_field(vol, sigma=0)
        np.testing.assert_allclose(mf, 0.0, atol=1e-6)

    def test_sharp_plane_produces_high_response_at_boundary(self):
        # Create a volume with a sharp boundary in the middle of axis 0.
        vol = np.zeros((16, 8, 8), dtype=np.uint8)
        vol[8:, :, :] = 200
        mf = compute_membrane_field(vol, sigma=0, normalize=True)
        # The gradient magnitude should peak near the step boundary.
        boundary_response = float(mf[7:9, :, :].mean())
        interior_response = float(mf[:6, :, :].mean())
        self.assertGreater(boundary_response, interior_response)

    def test_no_normalization_returns_raw_magnitude(self):
        rng = np.random.default_rng(3)
        vol = rng.integers(0, 255, (8, 8, 8), dtype=np.uint8)
        mf_raw = compute_membrane_field(vol, normalize=False)
        mf_norm = compute_membrane_field(vol, normalize=True)
        # Normalized max should be 1; unnormalized max should be larger.
        self.assertAlmostEqual(float(mf_norm.max()), 1.0, places=5)
        self.assertGreater(float(mf_raw.max()), 1.0)

    def test_all_values_finite(self):
        rng = np.random.default_rng(4)
        vol = rng.integers(0, 255, (12, 12, 12), dtype=np.uint8)
        mf = compute_membrane_field(vol)
        self.assertTrue(np.all(np.isfinite(mf)))

    def test_sigma_zero_skips_smoothing(self):
        vol = np.zeros((8, 8, 8), dtype=np.uint8)
        vol[4, :, :] = 200
        mf_s0 = compute_membrane_field(vol, sigma=0)
        mf_s1 = compute_membrane_field(vol, sigma=1)
        # With sigma=0 the response is sharper (more concentrated).
        # Check they are not identical.
        self.assertFalse(np.allclose(mf_s0, mf_s1))


class ComputeMembraneVectorsTest(unittest.TestCase):
    def test_output_shape_is_volume_plus_three_channel(self):
        mf = np.random.default_rng(5).random((10, 10, 10)).astype(np.float32)
        mv = compute_membrane_vectors(mf)
        self.assertEqual(mv.shape, (10, 10, 10, 3))

    def test_output_is_float32(self):
        mf = np.ones((6, 6, 6), dtype=np.float32)
        mv = compute_membrane_vectors(mf)
        self.assertEqual(mv.dtype, np.float32)

    def test_vectors_are_unit_length(self):
        rng = np.random.default_rng(6)
        mf = rng.random((8, 8, 8)).astype(np.float32)
        mv = compute_membrane_vectors(mf)
        norms = np.linalg.norm(mv, axis=-1)
        # Every voxel should have a unit-length vector.
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_all_values_finite(self):
        rng = np.random.default_rng(7)
        mf = rng.random((8, 8, 8)).astype(np.float32)
        mv = compute_membrane_vectors(mf)
        self.assertTrue(np.all(np.isfinite(mv)))

    def test_vectors_point_away_from_gradient_direction(self):
        # A membrane field that increases along axis 0 should produce
        # vectors pointing in the negative x direction.
        mf = np.zeros((10, 4, 4), dtype=np.float32)
        for i in range(10):
            mf[i, :, :] = float(i)
        mv = compute_membrane_vectors(mf, sigma=0)
        # The x-component should be negative in the interior.
        self.assertLess(float(mv[5, 2, 2, 0]), 0.0)


class ComputeExplorationFieldTest(unittest.TestCase):
    def test_returns_ones_array(self):
        ef = compute_exploration_field((4, 5, 6))
        np.testing.assert_array_equal(ef, np.ones((4, 5, 6), dtype=np.float32))

    def test_shape_matches_argument(self):
        ef = compute_exploration_field((3, 7, 2))
        self.assertEqual(ef.shape, (3, 7, 2))


class ComputeSynapseAttractionFieldTest(unittest.TestCase):
    def test_output_shape_matches_shape_arg(self):
        pts = np.array([[4, 4, 4]], dtype=np.float32)
        field = compute_synapse_attraction_field((8, 8, 8), pts)
        self.assertEqual(field.shape, (8, 8, 8))

    def test_peak_at_synapse_location(self):
        pts = np.array([[4, 4, 4]], dtype=np.float32)
        field = compute_synapse_attraction_field((9, 9, 9), pts, radius=3.0)
        self.assertAlmostEqual(float(field[4, 4, 4]), 1.0, places=4)

    def test_empty_synapses_gives_zero_field(self):
        field = compute_synapse_attraction_field((6, 6, 6), np.zeros((0, 3)))
        np.testing.assert_array_equal(field, 0.0)

    def test_linear_falloff_mode(self):
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        field = compute_synapse_attraction_field((5, 1, 1), pts, radius=4.0, falloff="linear")
        self.assertAlmostEqual(float(field[0, 0, 0]), 1.0, places=5)
        self.assertLess(float(field[2, 0, 0]), 1.0)


class SampleFieldTrilinearTest(unittest.TestCase):
    def test_exact_grid_point_returns_value(self):
        field = np.zeros((4, 4, 4), dtype=np.float32)
        field[2, 2, 2] = 1.0
        val = sample_field_trilinear(field, np.array([2.0, 2.0, 2.0]))
        self.assertAlmostEqual(val, 1.0, places=5)

    def test_midpoint_interpolates(self):
        field = np.zeros((4, 4, 4), dtype=np.float32)
        field[1, 0, 0] = 1.0
        # At x=0.5 the result should be 0.5.
        val = sample_field_trilinear(field, np.array([0.5, 0.0, 0.0]))
        self.assertAlmostEqual(val, 0.5, places=5)

    def test_out_of_bounds_point_is_clamped(self):
        field = np.ones((4, 4, 4), dtype=np.float32)
        # Should not raise; clamped to valid range.
        val = sample_field_trilinear(field, np.array([10.0, 10.0, 10.0]))
        self.assertAlmostEqual(val, 1.0, places=5)

    def test_vector_field_returns_three_components(self):
        field = np.zeros((4, 4, 4, 3), dtype=np.float32)
        field[:, :, :, 0] = 1.0
        vec = sample_vector_field_trilinear(field, np.array([1.0, 1.0, 1.0]))
        self.assertEqual(vec.shape, (3,))
        self.assertAlmostEqual(float(vec[0]), 1.0, places=5)
        self.assertAlmostEqual(float(vec[1]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
