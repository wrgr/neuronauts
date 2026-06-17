"""Tests for detect_soma in treestitch.assemble."""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.schemas import Fragment
from treestitch.assemble import detect_soma


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_neuron(center_nm: np.ndarray, half_extent_nm: float = 5_000.0) -> Fragment:
    """Minimal Fragment with a small cube of skeleton vertices around center_nm."""
    offsets = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [-1, 1,  1], [1, 1,  1],
    ], dtype=np.float32) * half_extent_nm
    verts = (center_nm[None, :].astype(np.float32) + offsets)
    edges = np.zeros((0, 2), dtype=np.int64)
    return Fragment(
        fragment_id=0,
        region_id="test",
        base_root_id=0,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=np.zeros((0, 3), dtype=np.float32),
        radius_nm=np.ones(len(verts), dtype=np.float32),
        synapse_indices=np.zeros(0, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectSoma:
    def test_nucleus_inside_bbox_returns_true(self):
        center = np.array([100_000.0, 200_000.0, 300_000.0])
        neuron = _make_neuron(center, half_extent_nm=5_000.0)
        # Nucleus exactly at center — well inside even without margin.
        nuclei = np.array([[100_000.0, 200_000.0, 300_000.0]], dtype=np.float64)
        has_soma, pos = detect_soma(neuron, nuclei, margin_nm=0.0)
        assert has_soma is True
        assert pos is not None
        np.testing.assert_allclose(pos, center, atol=1.0)

    def test_nucleus_outside_bbox_returns_false(self):
        center = np.array([0.0, 0.0, 0.0])
        neuron = _make_neuron(center, half_extent_nm=5_000.0)
        # Nucleus far away — beyond even a 10 µm margin.
        nuclei = np.array([[100_000.0, 0.0, 0.0]], dtype=np.float64)
        has_soma, pos = detect_soma(neuron, nuclei, margin_nm=0.0)
        assert has_soma is False
        assert pos is None

    def test_nucleus_inside_margin_but_outside_bbox_returns_true(self):
        center = np.array([0.0, 0.0, 0.0])
        neuron = _make_neuron(center, half_extent_nm=5_000.0)
        # Nucleus is 8 µm from the bbox face — inside the 10 µm default margin.
        nuclei = np.array([[13_000.0, 0.0, 0.0]], dtype=np.float64)
        has_soma, pos = detect_soma(neuron, nuclei, margin_nm=10_000.0)
        assert has_soma is True

    def test_empty_nucleus_array_returns_false(self):
        center = np.array([0.0, 0.0, 0.0])
        neuron = _make_neuron(center)
        nuclei = np.zeros((0, 3), dtype=np.float64)
        has_soma, pos = detect_soma(neuron, nuclei)
        assert has_soma is False
        assert pos is None

    def test_none_nucleus_array_returns_false(self):
        center = np.array([0.0, 0.0, 0.0])
        neuron = _make_neuron(center)
        has_soma, pos = detect_soma(neuron, None)
        assert has_soma is False
        assert pos is None

    def test_closest_nucleus_returned_among_multiple_candidates(self):
        center = np.array([0.0, 0.0, 0.0])
        neuron = _make_neuron(center, half_extent_nm=5_000.0)
        # Two nuclei in bbox; (0,0,0) is closer to centroid (also 0,0,0).
        nuclei = np.array([
            [0.0,      0.0, 0.0],      # distance 0 to centroid
            [4_000.0,  0.0, 0.0],      # distance 4000
        ], dtype=np.float64)
        has_soma, pos = detect_soma(neuron, nuclei, margin_nm=0.0)
        assert has_soma is True
        np.testing.assert_allclose(pos, [0.0, 0.0, 0.0], atol=1.0)

    def test_neuron_with_no_vertices_returns_false(self):
        neuron = Fragment(
            fragment_id=0,
            region_id="test",
            base_root_id=0,
            vertices_nm=np.zeros((0, 3), dtype=np.float32),
            edges=np.zeros((0, 2), dtype=np.int64),
            endpoints_nm=np.zeros((0, 3), dtype=np.float32),
            radius_nm=np.zeros(0, dtype=np.float32),
            synapse_indices=np.zeros(0, dtype=np.int64),
        )
        nuclei = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        has_soma, pos = detect_soma(neuron, nuclei)
        assert has_soma is False
        assert pos is None

    def test_returned_position_is_a_copy(self):
        center = np.array([0.0, 0.0, 0.0])
        neuron = _make_neuron(center, half_extent_nm=5_000.0)
        nuclei = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        _, pos = detect_soma(neuron, nuclei, margin_nm=0.0)
        assert pos is not None
        # Mutating pos should not affect the original nuclei array.
        pos[:] = 999.0
        assert nuclei[0, 0] != 999.0
