"""Tests for geometry helpers in neuronauts/fetch.py.

skeleton_tortuosity, skeleton_stepwise_features, mesh_volume_surface_ratio,
and mesh_stepwise_features are the feed-in functions for the multi-modal
path encoder. Degenerate-input handling must be robust.
"""

import unittest

import numpy as np

from neuronauts.fetch import (
    make_cube_bbox_nm,
    mesh_stepwise_features,
    mesh_volume_surface_ratio,
    skeleton_stepwise_features,
    skeleton_tortuosity,
)


# ---------------------------------------------------------------------------
# make_cube_bbox_nm
# ---------------------------------------------------------------------------

class MakeCubeBboxNmTest(unittest.TestCase):
    def test_6um_cube_spans_6000nm_per_side(self):
        center = (10_000, 20_000, 30_000)
        lo, hi = make_cube_bbox_nm(center, side_um=6.0)
        for axis in range(3):
            self.assertEqual(hi[axis] - lo[axis], 6000)

    def test_center_is_midpoint(self):
        center = (0, 0, 0)
        lo, hi = make_cube_bbox_nm(center, side_um=4.0)
        for axis in range(3):
            mid = (lo[axis] + hi[axis]) / 2
            self.assertAlmostEqual(mid, 0.0, places=1)

    def test_returns_two_length_three_tuples(self):
        lo, hi = make_cube_bbox_nm((1, 2, 3))
        self.assertEqual(len(lo), 3)
        self.assertEqual(len(hi), 3)


# ---------------------------------------------------------------------------
# skeleton_tortuosity
# ---------------------------------------------------------------------------

class SkeletonTortuosityTest(unittest.TestCase):
    def test_straight_line_is_one(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
        t = skeleton_tortuosity(pts)
        self.assertAlmostEqual(t, 1.0, places=5)

    def test_wound_path_exceeds_one(self):
        # Zigzag path: arc length > end-to-end
        pts = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0], [4, 0, 0]], dtype=np.float32)
        t = skeleton_tortuosity(pts)
        self.assertGreater(t, 1.0)

    def test_single_point_returns_one(self):
        pts = np.array([[5, 5, 5]], dtype=np.float32)
        t = skeleton_tortuosity(pts)
        self.assertAlmostEqual(t, 1.0, places=5)

    def test_empty_returns_one(self):
        pts = np.zeros((0, 3), dtype=np.float32)
        t = skeleton_tortuosity(pts)
        self.assertAlmostEqual(t, 1.0, places=5)

    def test_degenerate_zero_length_path_returns_finite(self):
        # All points at the same location: arc_length = 0, end_to_end = 0.
        pts = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)
        t = skeleton_tortuosity(pts)
        self.assertTrue(np.isfinite(t))

    def test_result_is_always_at_least_one(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            pts = rng.random((10, 3)).astype(np.float32)
            t = skeleton_tortuosity(pts)
            self.assertGreaterEqual(t, 1.0 - 1e-6)


# ---------------------------------------------------------------------------
# skeleton_stepwise_features
# ---------------------------------------------------------------------------

class SkeletonStepwiseFeaturesTest(unittest.TestCase):
    def test_output_shape_is_t_by_3(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
        feats = skeleton_stepwise_features(pts)
        self.assertEqual(feats.shape, (3, 3))

    def test_output_dtype_is_float32(self):
        pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        feats = skeleton_stepwise_features(pts)
        self.assertEqual(feats.dtype, np.float32)

    def test_step_distances_are_non_negative(self):
        rng = np.random.default_rng(7)
        pts = rng.random((8, 3)).astype(np.float32)
        feats = skeleton_stepwise_features(pts)
        self.assertTrue(np.all(feats[:, 0] >= 0.0))

    def test_normalised_cumulative_arc_in_zero_one(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
        feats = skeleton_stepwise_features(pts)
        self.assertGreaterEqual(float(feats[:, 1].min()), 0.0)
        self.assertLessEqual(float(feats[:, 1].max()), 1.0 + 1e-6)

    def test_turning_angles_are_non_negative(self):
        rng = np.random.default_rng(8)
        pts = rng.random((6, 3)).astype(np.float32)
        feats = skeleton_stepwise_features(pts)
        self.assertTrue(np.all(feats[:, 2] >= 0.0))

    def test_straight_path_has_zero_turning_angles(self):
        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
        feats = skeleton_stepwise_features(pts)
        np.testing.assert_allclose(feats[:, 2], 0.0, atol=1e-5)

    def test_single_point_returns_empty(self):
        pts = np.array([[0, 0, 0]], dtype=np.float32)
        feats = skeleton_stepwise_features(pts)
        self.assertEqual(feats.shape, (0, 3))

    def test_empty_input_returns_empty(self):
        feats = skeleton_stepwise_features(np.zeros((0, 3), dtype=np.float32))
        self.assertEqual(feats.shape, (0, 3))

    def test_all_values_finite(self):
        rng = np.random.default_rng(9)
        pts = rng.random((10, 3)).astype(np.float32)
        feats = skeleton_stepwise_features(pts)
        self.assertTrue(np.all(np.isfinite(feats)))


# ---------------------------------------------------------------------------
# mesh_volume_surface_ratio
# ---------------------------------------------------------------------------

class MeshVolumeSurfaceRatioTest(unittest.TestCase):
    def _unit_tetrahedron(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
        return vertices, faces

    def test_unit_tetrahedron_is_positive(self):
        v, f = self._unit_tetrahedron()
        ratio = mesh_volume_surface_ratio(v, f)
        self.assertGreater(ratio, 0.0)

    def test_ratio_is_finite(self):
        v, f = self._unit_tetrahedron()
        ratio = mesh_volume_surface_ratio(v, f)
        self.assertTrue(np.isfinite(ratio))

    def test_empty_faces_returns_zero(self):
        v = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float64)
        f = np.zeros((0, 3), dtype=np.int64)
        self.assertAlmostEqual(mesh_volume_surface_ratio(v, f), 0.0)

    def test_wrong_shape_returns_zero(self):
        self.assertAlmostEqual(
            mesh_volume_surface_ratio(
                np.array([[0, 0]], dtype=np.float64),  # 2D, not 3D
                np.array([[0, 0, 0]], dtype=np.int64),
            ),
            0.0,
        )

    def test_larger_sphere_like_mesh_has_smaller_ratio_than_cube(self):
        # A sphere has a higher volume-to-surface ratio than a thin flat mesh.
        # Approximate a flat quad (high surface, low volume): vertices near z=0.
        flat_v = np.array([
            [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
            [0, 0, 0.01], [10, 0, 0.01], [10, 10, 0.01], [0, 10, 0.01],
        ], dtype=np.float64)
        flat_f = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ], dtype=np.int64)
        flat_ratio = mesh_volume_surface_ratio(flat_v, flat_f)

        # A more cubic-ish mesh:
        cube_v = np.array([
            [0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0],
            [0, 0, 5], [5, 0, 5], [5, 5, 5], [0, 5, 5],
        ], dtype=np.float64)
        cube_f = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ], dtype=np.int64)
        cube_ratio = mesh_volume_surface_ratio(cube_v, cube_f)
        # Cube has higher V/A than flat slab.
        self.assertGreater(cube_ratio, flat_ratio)


# ---------------------------------------------------------------------------
# mesh_stepwise_features
# ---------------------------------------------------------------------------

class MeshStepwiseFeaturesTest(unittest.TestCase):
    def _simple_mesh(self):
        v = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        ], dtype=np.float32)
        f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
        return v, f

    def test_output_shape_is_t_by_3(self):
        v, f = self._simple_mesh()
        waypoints = np.array([[0.2, 0.2, 0.2], [0.5, 0.1, 0.1]], dtype=np.float32)
        feats = mesh_stepwise_features(v, f, waypoints)
        self.assertEqual(feats.shape, (2, 3))

    def test_output_dtype_is_float32(self):
        v, f = self._simple_mesh()
        waypoints = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        feats = mesh_stepwise_features(v, f, waypoints)
        self.assertEqual(feats.dtype, np.float32)

    def test_all_values_finite(self):
        v, f = self._simple_mesh()
        rng = np.random.default_rng(10)
        waypoints = rng.random((5, 3)).astype(np.float32)
        feats = mesh_stepwise_features(v, f, waypoints)
        self.assertTrue(np.all(np.isfinite(feats)))

    def test_mean_distance_col_is_non_negative(self):
        v, f = self._simple_mesh()
        waypoints = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]], dtype=np.float32)
        feats = mesh_stepwise_features(v, f, waypoints)
        self.assertTrue(np.all(feats[:, 0] >= 0.0))

    def test_empty_waypoints_returns_empty(self):
        v, f = self._simple_mesh()
        feats = mesh_stepwise_features(v, f, np.zeros((0, 3), dtype=np.float32))
        self.assertEqual(feats.shape, (0, 3))

    def test_empty_mesh_returns_zeros(self):
        waypoints = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        feats = mesh_stepwise_features(
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int64),
            waypoints,
        )
        np.testing.assert_array_equal(feats, 0.0)

    def test_waypoint_at_vertex_has_zero_mean_distance(self):
        v, f = self._simple_mesh()
        # A waypoint exactly on vertex 0.
        waypoints = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        feats = mesh_stepwise_features(v, f, waypoints, k_nearest=1)
        self.assertAlmostEqual(float(feats[0, 0]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
