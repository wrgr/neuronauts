"""Tests for treestitch.assemble: skeleton merger, shape assembly, metrics."""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.schemas import Fragment
from treestitch.assemble import (
    assemble_partition_shapes,
    merge_fragment_skeletons,
    neuron_shape_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _line_fragment(
    fragment_id: int,
    start: np.ndarray,
    end: np.ndarray,
    n_verts: int = 4,
    seg_root: int | None = None,
) -> Fragment:
    """A straight-line skeleton from start→end with n_verts evenly spaced."""
    t = np.linspace(0, 1, n_verts, dtype=np.float32)
    verts = (start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]).astype(np.float32)
    edges = np.array([[i, i + 1] for i in range(n_verts - 1)], dtype=np.int64)
    endpoints = verts[[0, -1]]
    return Fragment(
        fragment_id=fragment_id,
        region_id="test",
        base_root_id=seg_root if seg_root is not None else fragment_id,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=endpoints,
        radius_nm=np.full(n_verts, 200.0, dtype=np.float32),
        synapse_indices=np.empty(0, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# merge_fragment_skeletons
# ---------------------------------------------------------------------------

def test_merge_single_fragment_is_identity():
    f = _line_fragment(0, np.array([0, 0, 0], np.float32), np.array([5000, 0, 0], np.float32))
    merged = merge_fragment_skeletons([f])
    assert len(merged.vertices_nm) == len(f.vertices_nm)
    assert len(merged.edges) == len(f.edges)
    np.testing.assert_array_equal(merged.vertices_nm, f.vertices_nm)


def test_merge_two_far_fragments_stays_forest():
    """Fragments > stitch_radius apart → no bridge → 2 components."""
    f0 = _line_fragment(0, np.array([0, 0, 0], np.float32), np.array([4000, 0, 0], np.float32))
    f1 = _line_fragment(1, np.array([20000, 0, 0], np.float32), np.array([24000, 0, 0], np.float32))
    merged = merge_fragment_skeletons([f0, f1], stitch_radius_nm=5_000.0)
    m = neuron_shape_metrics(merged)
    assert m["n_connected_components"] == 2
    assert m["is_tree"], "Forest (2 components, no cycles) must still satisfy is_tree"
    # Total vertices = sum of both fragments
    assert len(merged.vertices_nm) == len(f0.vertices_nm) + len(f1.vertices_nm)


def test_merge_two_close_fragments_bridges_and_remains_tree():
    """Fragments within stitch_radius → one bridge edge → 1 component, is_tree."""
    f0 = _line_fragment(0, np.array([0, 0, 0], np.float32), np.array([4000, 0, 0], np.float32))
    # f1 starts at 4200 — just 200 nm from f0's right endpoint
    f1 = _line_fragment(1, np.array([4200, 0, 0], np.float32), np.array([8200, 0, 0], np.float32))
    merged = merge_fragment_skeletons([f0, f1], stitch_radius_nm=5_000.0)
    m = neuron_shape_metrics(merged)
    assert m["n_connected_components"] == 1
    assert m["is_tree"]
    # E = V - 1 for a single-component tree
    V = len(merged.vertices_nm)
    E = len(merged.edges)
    assert E == V - 1


def test_merge_three_fragments_linear_chain():
    """Three collinear fragments stitched into one chain."""
    f0 = _line_fragment(0, np.array([0,    0, 0], np.float32), np.array([4000, 0, 0], np.float32))
    f1 = _line_fragment(1, np.array([4200, 0, 0], np.float32), np.array([8200, 0, 0], np.float32))
    f2 = _line_fragment(2, np.array([8400, 0, 0], np.float32), np.array([12400, 0, 0], np.float32))
    merged = merge_fragment_skeletons([f0, f1, f2], stitch_radius_nm=5_000.0)
    m = neuron_shape_metrics(merged)
    assert m["n_connected_components"] == 1
    assert m["is_tree"]
    assert m["cable_length_um"] == pytest.approx(12.4, abs=0.1)


def test_merge_does_not_create_cycles():
    """Three fragments in a triangle arrangement — only 2 bridges, no cycle."""
    # Place three fragments at corners of a small triangle
    f0 = _line_fragment(0, np.array([0,    0,    0], np.float32), np.array([1000, 0, 0], np.float32), n_verts=2)
    f1 = _line_fragment(1, np.array([2000, 0,    0], np.float32), np.array([3000, 0, 0], np.float32), n_verts=2)
    f2 = _line_fragment(2, np.array([1500, 2000, 0], np.float32), np.array([1500, 3000, 0], np.float32), n_verts=2)
    merged = merge_fragment_skeletons([f0, f1, f2], stitch_radius_nm=5_000.0)
    m = neuron_shape_metrics(merged)
    assert m["is_tree"], "Kruskal must prevent cycles; is_tree must be True"
    assert m["n_connected_components"] == 1


def test_merge_preserves_total_vertex_count():
    f0 = _line_fragment(0, np.array([0, 0, 0], np.float32), np.array([4000, 0, 0], np.float32), n_verts=5)
    f1 = _line_fragment(1, np.array([4200, 0, 0], np.float32), np.array([8200, 0, 0], np.float32), n_verts=6)
    merged = merge_fragment_skeletons([f0, f1], stitch_radius_nm=5_000.0)
    assert len(merged.vertices_nm) == 5 + 6


# ---------------------------------------------------------------------------
# neuron_shape_metrics
# ---------------------------------------------------------------------------

def test_shape_metrics_single_line():
    f = _line_fragment(0, np.array([0, 0, 0], np.float32), np.array([10_000, 0, 0], np.float32), n_verts=5)
    m = neuron_shape_metrics(f)
    assert m["cable_length_um"] == pytest.approx(10.0, abs=0.1)
    assert m["n_endpoints"] == 2
    assert m["n_branch_points"] == 0
    assert m["n_connected_components"] == 1
    assert m["is_tree"]
    assert m["bbox_volume_um3"] == pytest.approx(0.0, abs=1e-6)  # flat line


def test_shape_metrics_empty_fragment():
    f = Fragment(
        fragment_id=0, region_id="t", base_root_id=0,
        vertices_nm=np.empty((0, 3), np.float32),
        edges=np.empty((0, 2), np.int64),
        endpoints_nm=np.empty((0, 3), np.float32),
        radius_nm=np.empty(0, np.float32),
        synapse_indices=np.empty(0, np.int64),
    )
    m = neuron_shape_metrics(f)
    assert m["cable_length_um"] == 0.0
    assert m["n_connected_components"] == 0
    assert m["is_tree"]


def test_shape_metrics_branching():
    """Y-shaped skeleton: one branch point, three endpoints."""
    # 0-1 (stem), 1-2 (branch A), 1-3 (branch B)
    verts = np.array([[0, 0, 0], [1000, 0, 0], [2000, 1000, 0], [2000, -1000, 0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
    f = Fragment(
        fragment_id=0, region_id="t", base_root_id=0,
        vertices_nm=verts, edges=edges,
        endpoints_nm=verts[[0, 2, 3]],
        radius_nm=np.full(4, 200.0, np.float32),
        synapse_indices=np.empty(0, np.int64),
    )
    m = neuron_shape_metrics(f)
    assert m["n_branch_points"] == 1
    assert m["n_endpoints"] == 3
    assert m["is_tree"]
    assert m["n_connected_components"] == 1


def test_shape_metrics_cycle_detected():
    """A triangle (cycle) must produce is_tree=False."""
    verts = np.array([[0, 0, 0], [1000, 0, 0], [500, 1000, 0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    f = Fragment(
        fragment_id=0, region_id="t", base_root_id=0,
        vertices_nm=verts, edges=edges,
        endpoints_nm=verts,
        radius_nm=np.full(3, 200.0, np.float32),
        synapse_indices=np.empty(0, np.int64),
    )
    m = neuron_shape_metrics(f)
    # 3 vertices, 3 edges, 1 component → 3 ≠ 3-1=2 → not a tree
    assert not m["is_tree"]


# ---------------------------------------------------------------------------
# assemble_partition_shapes
# ---------------------------------------------------------------------------

def test_assemble_groups_fragments_by_label():
    """Two neurons, each one fragment — shapes should be separate."""
    f0 = _line_fragment(0, np.array([0, 0, 0], np.float32), np.array([5000, 0, 0], np.float32), seg_root=100)
    f1 = _line_fragment(1, np.array([50000, 0, 0], np.float32), np.array([55000, 0, 0], np.float32), seg_root=200)

    # 4 observations per fragment
    seg_ids    = np.array([100, 100, 100, 100, 200, 200, 200, 200], dtype=np.int64)
    pred_labels= np.array([  0,   0,   0,   0,   1,   1,   1,   1], dtype=np.int64)

    shapes = assemble_partition_shapes([f0, f1], pred_labels, seg_ids)
    assert set(shapes.keys()) == {0, 1}
    m0 = neuron_shape_metrics(shapes[0])
    m1 = neuron_shape_metrics(shapes[1])
    assert m0["n_connected_components"] == 1
    assert m1["n_connected_components"] == 1


def test_assemble_ignores_abstained_observations():
    f0 = _line_fragment(0, np.array([0, 0, 0], np.float32), np.array([5000, 0, 0], np.float32), seg_root=100)
    seg_ids    = np.array([100, 100, 100, 100], dtype=np.int64)
    pred_labels= np.array([ -1,   0,   0,   0], dtype=np.int64)  # first obs abstained

    shapes = assemble_partition_shapes([f0], pred_labels, seg_ids)
    assert 0 in shapes
    assert -1 not in shapes


def test_assemble_empty_pred_returns_empty():
    f0 = _line_fragment(0, np.array([0, 0, 0], np.float32), np.array([5000, 0, 0], np.float32), seg_root=100)
    shapes = assemble_partition_shapes(
        [f0],
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
    )
    assert shapes == {}


def test_assemble_two_fragments_same_neuron_stitched():
    """Two close fragments assigned the same label → merged and stitched."""
    f0 = _line_fragment(0, np.array([0,    0, 0], np.float32), np.array([4000, 0, 0], np.float32), seg_root=10)
    f1 = _line_fragment(1, np.array([4200, 0, 0], np.float32), np.array([8200, 0, 0], np.float32), seg_root=20)

    seg_ids    = np.array([10, 10, 20, 20], dtype=np.int64)
    pred_labels= np.array([ 0,  0,  0,  0], dtype=np.int64)  # same neuron

    shapes = assemble_partition_shapes([f0, f1], pred_labels, seg_ids, stitch_radius_nm=5_000.0)
    assert 0 in shapes
    m = neuron_shape_metrics(shapes[0])
    assert m["n_connected_components"] == 1
    assert m["is_tree"]


# ---------------------------------------------------------------------------
# treestitch.partition wrapper
# ---------------------------------------------------------------------------

def test_partition_wrappers_assemble_and_metrics():
    from treestitch.partition import assemble_partition_shapes as aps_w
    from treestitch.partition import neuron_shape_metrics as nsm_w

    f = _line_fragment(0, np.array([0, 0, 0], np.float32), np.array([8000, 0, 0], np.float32), seg_root=99)
    pred = np.array([0, 0, 0], dtype=np.int64)
    segs = np.array([99, 99, 99], dtype=np.int64)
    shapes = aps_w([f], pred, segs)
    assert 0 in shapes
    m = nsm_w(shapes[0])
    assert m["is_tree"]
    assert m["cable_length_um"] == pytest.approx(8.0, abs=0.1)
