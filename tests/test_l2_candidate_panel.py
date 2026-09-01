import numpy as np

from neuronauts.l2_candidate_panel import (
    candidate_endpoint_pairs, endpoint_records, filter_candidate_pairs,
    panel_sizes,
)
from neuronauts.real_dense_soma import Fragment


def line(root, points):
    points = np.asarray(points, dtype=np.float32)
    edges = np.column_stack([np.arange(len(points) - 1),
                             np.arange(1, len(points))])
    return Fragment(root, points, edges)


def test_facing_endpoints_survive_tight_cone():
    fragments = [
        line(1, [[0, 0, 0], [-1, 0, 0], [-2, 0, 0]]),
        line(2, [[2, 0, 0], [3, 0, 0], [4, 0, 0]]),
    ]
    pairs = candidate_endpoint_pairs(endpoint_records(fragments),
                                     max_distance_nm=2.1)
    kept = filter_candidate_pairs(pairs, max_distance_nm=2.1,
                                  cone_degrees=30)
    assert len(kept) == 1
    assert kept[0].facing == 1.0
    assert panel_sizes([1, 2], kept).tolist() == [1, 1]


def test_parallel_endpoints_fail_symmetric_facing_cone():
    fragments = [
        line(1, [[0, 0, 0], [-1, 0, 0], [-2, 0, 0]]),
        line(2, [[2, 0, 0], [1, 0, 0], [-5, 0, 0]]),
    ]
    pairs = candidate_endpoint_pairs(endpoint_records(fragments),
                                     max_distance_nm=2.1)
    assert filter_candidate_pairs(
        pairs, max_distance_nm=2.1, cone_degrees=60) == []
