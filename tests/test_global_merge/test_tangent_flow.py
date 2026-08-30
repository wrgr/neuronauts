"""
Tests for Tangent-Flow and Ray-Casting Collinearity.
"""

import pytest
import numpy as np
from neuronauts.global_merge.schemas import EndpointTangent, SegmentFragment
from neuronauts.global_merge.represent.tangent_flow import (
    extract_endpoints_from_skeleton,
    compute_collinearity,
    find_tangent_flow_bridges
)


def test_extract_endpoints_linear_skeleton():
    # Line along X axis: (0,0,0) -> (100,0,0) -> (200,0,0)
    verts = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [200.0, 0.0, 0.0],
    ], dtype=np.float32)
    radii = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)

    eps = extract_endpoints_from_skeleton("frag_01", verts, radii, edges)
    assert len(eps) == 2

    # ep at vertex 0 should point in -X direction (-1, 0, 0)
    ep0 = [e for e in eps if e.vertex_idx == 0][0]
    assert np.allclose(ep0.tangent, [-1.0, 0.0, 0.0], atol=1e-2)

    # ep at vertex 2 should point in +X direction (+1, 0, 0)
    ep2 = [e for e in eps if e.vertex_idx == 2][0]
    assert np.allclose(ep2.tangent, [1.0, 0.0, 0.0], atol=1e-2)


def test_compute_collinearity_cases():
    # Case 1: Perfectly aligned pointing at each other
    # ep1 at (0,0,0) pointing +X [1,0,0]
    ep1 = EndpointTangent("f1", 0, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 50.0)
    # ep2 at (1000,0,0) pointing -X [-1,0,0]
    ep2 = EndpointTangent("f2", 0, np.array([1000.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 50.0)

    score_aligned = compute_collinearity(ep1, ep2, sigma_dist_nm=15000.0)
    assert score_aligned > 0.85

    # Case 2: Orthogonal (ep1 pointing +X, ep2 pointing +Y)
    ep_orth = EndpointTangent("f2", 0, np.array([1000.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 50.0)
    score_orth = compute_collinearity(ep1, ep_orth, sigma_dist_nm=15000.0)
    assert score_orth == 0.0

    # Case 3: Pointing away from each other
    ep_away = EndpointTangent("f2", 0, np.array([1000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 50.0)
    score_away = compute_collinearity(ep1, ep_away, sigma_dist_nm=15000.0)
    assert score_away == 0.0


def test_find_tangent_flow_bridges():
    # Fragment 1: line from (0,0,0) to (5000,0,0)
    f1 = SegmentFragment(
        fragment_id="f1", segment_id=1,
        vertices_nm=np.array([[0.0, 0.0, 0.0], [5000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([50.0, 50.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("f1", 0, np.array([0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 50.0),
            EndpointTangent("f1", 1, np.array([5000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 50.0)
        ]
    )

    # Fragment 2: line from (7000,0,0) to (12000,0,0) -> 2000 nm gap along X axis
    f2 = SegmentFragment(
        fragment_id="f2", segment_id=2,
        vertices_nm=np.array([[7000.0, 0.0, 0.0], [12000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([50.0, 50.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("f2", 0, np.array([7000.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 50.0),
            EndpointTangent("f2", 1, np.array([12000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 50.0)
        ]
    )

    bridges = find_tangent_flow_bridges([f1, f2], max_distance_nm=10000.0, min_collinearity=0.3)
    assert len(bridges) == 1
    assert bridges[0].src_id == "f1"
    assert bridges[0].dst_id == "f2"
    assert bridges[0].collinearity_score > 0.70
