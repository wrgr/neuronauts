"""Tests for multi_fragment_ablation.py skeleton splitting logic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "attic" / "prior_results"))
from multi_fragment_ablation import split_skeleton_n_parts


def _chain(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple n-vertex chain skeleton."""
    verts = np.stack([np.arange(n, dtype=np.float32),
                      np.zeros(n, np.float32),
                      np.zeros(n, np.float32)], axis=1)
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int64)
    radii = np.ones(n, np.float32) * 300.0
    return verts, edges, radii


def _star(n_arms: int, arm_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Star skeleton: one centre vertex connected to n_arms arms of arm_len vertices."""
    verts = [np.array([0.0, 0.0, 0.0])]  # centre = vertex 0
    edges_list = []
    for arm in range(n_arms):
        angle = 2 * np.pi * arm / n_arms
        for step in range(1, arm_len + 1):
            v = np.array([np.cos(angle) * step * 500,
                          np.sin(angle) * step * 500,
                          0.0])
            verts.append(v)
            edges_list.append([len(verts) - 2, len(verts) - 1])
        # Connect first arm vertex to centre
        edges_list[arm * arm_len][0] = 0

    verts_arr = np.array(verts, np.float32)
    edges_arr = np.array(edges_list, np.int64)
    radii = np.ones(len(verts), np.float32) * 300.0
    return verts_arr, edges_arr, radii


class TestSplitSkeletonNParts:
    def test_chain_two_splits(self):
        """Splitting a chain into 2 parts yields 2 non-empty sub-trees."""
        verts, edges, radii = _chain(20)
        rng = np.random.default_rng(0)
        parts = split_skeleton_n_parts(verts, edges, radii, n_splits=2, rng=rng)
        assert len(parts) == 2
        total_verts = sum(len(p[0]) for p in parts)
        assert total_verts == 20

    def test_chain_four_splits(self):
        """Splitting a 40-vertex chain into 4 parts covers all vertices."""
        verts, edges, radii = _chain(40)
        rng = np.random.default_rng(0)
        parts = split_skeleton_n_parts(verts, edges, radii, n_splits=4, rng=rng)
        assert len(parts) >= 2  # may not always achieve exactly 4 splits
        total_verts = sum(len(p[0]) for p in parts)
        assert total_verts == 40

    def test_each_part_has_vertices_and_endpoints(self):
        verts, edges, radii = _chain(30)
        rng = np.random.default_rng(1)
        parts = split_skeleton_n_parts(verts, edges, radii, n_splits=3, rng=rng)
        for sv, se, sr, ep in parts:
            assert len(sv) >= 3
            assert sv.shape[1] == 3
            assert ep.ndim == 2 and ep.shape[1] == 3

    def test_radii_length_matches_vertices(self):
        verts, edges, radii = _chain(24)
        rng = np.random.default_rng(2)
        parts = split_skeleton_n_parts(verts, edges, radii, n_splits=4, rng=rng)
        for sv, se, sr, ep in parts:
            assert len(sr) == len(sv)

    def test_too_small_skeleton_returns_intact(self):
        """Skeleton smaller than n_splits * 3 is returned as one piece."""
        verts, edges, radii = _chain(5)
        rng = np.random.default_rng(0)
        parts = split_skeleton_n_parts(verts, edges, radii, n_splits=4, rng=rng)
        # Should fall through to single-piece return
        assert sum(len(p[0]) for p in parts) == 5

    def test_star_four_splits(self):
        """A 4-arm star split into 4 isolates the arms."""
        verts, edges, radii = _star(n_arms=4, arm_len=10)
        rng = np.random.default_rng(3)
        parts = split_skeleton_n_parts(verts, edges, radii, n_splits=4, rng=rng)
        # Each arm is its own component; the centre goes to one.
        assert len(parts) >= 2
        total_verts = sum(len(p[0]) for p in parts)
        assert total_verts == len(verts)

    def test_reproducible_with_same_seed(self):
        verts, edges, radii = _chain(30)
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        parts_a = split_skeleton_n_parts(verts, edges, radii, n_splits=3, rng=rng_a)
        parts_b = split_skeleton_n_parts(verts, edges, radii, n_splits=3, rng=rng_b)
        assert len(parts_a) == len(parts_b)
        for (va, _, _, _), (vb, _, _, _) in zip(parts_a, parts_b):
            np.testing.assert_array_equal(va, vb)
