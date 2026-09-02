"""Contraction of an atom's L2 adjacency into junctions and segments.

The invariants matter more than the counts: if segments do not partition the
edges, every cable length and every endpoint downstream is quietly wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.harness.topology import (
    build_adjacency, connected_components, contract, endpoint_tangents,
)


def run(ids, edges):
    ids = np.asarray(ids, np.uint64)
    edges = np.asarray(edges, np.uint64).reshape(-1, 2)
    indptr, indices, deg = build_adjacency(ids, edges)
    return contract(indptr, indices, deg), deg


# ---------------------------------------------------------------------------
# known-answer shapes
# ---------------------------------------------------------------------------

def test_path_is_one_segment():
    t, deg = run([10, 11, 12, 13, 14], [[10, 11], [11, 12], [12, 13], [13, 14]])
    assert len(t.seg_ends) == 1
    assert int(t.comp.max()) + 1 == 1
    assert t.cycles == 0
    assert t.seg_is_leaf.all()
    assert sorted(deg.tolist()) == [1, 1, 2, 2, 2]


def test_fork_gives_three_leaf_segments():
    t, deg = run([10, 11, 12, 13], [[10, 11], [11, 12], [11, 13]])
    assert len(t.seg_ends) == 3
    assert int((deg == 1).sum()) == 3
    assert int((deg >= 3).sum()) == 1
    assert t.seg_is_leaf.all()


def test_pure_cycle_has_no_segment():
    t, _ = run([10, 11, 12], [[10, 11], [11, 12], [12, 10]])
    assert len(t.seg_ends) == 0
    assert t.cycles == 1


def test_isolated_node_is_not_a_cycle():
    """A degree-0 node is a junction, so it must not be counted as a loop."""
    t, deg = run([10], [])
    assert deg.tolist() == [0]
    assert len(t.seg_ends) == 0
    assert t.cycles == 0


def test_duplicate_and_self_edges_do_not_invent_branches():
    t, deg = run([10, 11, 12], [[10, 11], [11, 12], [10, 11], [11, 10], [12, 12]])
    assert deg.max() == 2, "a repeated pair must not raise degree"
    assert len(t.seg_ends) == 1


def test_two_components_are_counted_separately():
    t, _ = run([10, 11, 20, 21], [[10, 11], [20, 21]])
    assert int(t.comp.max()) + 1 == 2
    assert len(t.seg_ends) == 2


def test_edges_referencing_unknown_nodes_are_dropped():
    t, deg = run([10, 11], [[10, 11], [10, 999]])
    assert deg.tolist() == [1, 1]
    assert len(t.seg_ends) == 1


# ---------------------------------------------------------------------------
# invariants on a random graph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_segments_partition_the_edges(seed):
    """Every edge lies in exactly one segment, except edges of pure cycles."""
    rng = np.random.default_rng(seed)
    n = 60
    ids = np.arange(100, 100 + n, dtype=np.uint64)
    e = rng.integers(0, n, size=(90, 2))
    e = e[e[:, 0] != e[:, 1]]
    edges = ids[e]

    indptr, indices, deg = build_adjacency(ids, edges)
    t = contract(indptr, indices, deg)

    n_edges = int(indptr[-1] // 2)
    juncs = set(t.junctions.tolist())
    n_comp = int(t.comp.max()) + 1
    cyc_edges = sum(
        int((t.comp == c).sum())
        for c in range(n_comp)
        if not (set(np.flatnonzero(t.comp == c).tolist()) & juncs)
    )
    seg_edges = sum(len(s) + 1 for s in t.seg_nodes)
    assert seg_edges + cyc_edges == n_edges

    # each degree-2 node is interior to exactly one segment
    interior = (np.concatenate(t.seg_nodes) if t.seg_nodes
                else np.zeros(0, np.int32))
    _, counts = np.unique(interior, return_counts=True)
    assert not len(counts) or counts.max() == 1
    assert (deg[interior] == 2).all()


@pytest.mark.parametrize("seed", range(4))
def test_components_match_a_union_find_reference(seed):
    rng = np.random.default_rng(seed)
    n = 50
    ids = np.arange(n, dtype=np.uint64)
    e = rng.integers(0, n, size=(60, 2))
    e = e[e[:, 0] != e[:, 1]]
    indptr, indices, deg = build_adjacency(ids, ids[e])

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in e.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    ref = len({find(i) for i in range(n)})

    comp = connected_components(indptr, indices, n)
    assert int(comp.max()) + 1 == ref


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_segment_length_sums_the_path():
    ids = [10, 11, 12]
    pos = np.array([[0, 0, 0], [300, 0, 0], [300, 400, 0]], np.float32)
    indptr, indices, deg = build_adjacency(np.asarray(ids, np.uint64),
                                           np.asarray([[10, 11], [11, 12]], np.uint64))
    t = contract(indptr, indices, deg, pos=pos)
    assert t.seg_len_nm[0] == pytest.approx(700.0)


def test_missing_coordinates_do_not_fabricate_length():
    """A gap makes the length unknown, not shorter.

    Bridging across the missing node would silently under-measure the cable, so
    the segment reports NaN and the caller counts it.
    """
    ids = [10, 11, 12]
    pos = np.array([[0, 0, 0], [np.nan] * 3, [300, 0, 0]], np.float32)
    indptr, indices, deg = build_adjacency(np.asarray(ids, np.uint64),
                                           np.asarray([[10, 11], [11, 12]], np.uint64))
    t = contract(indptr, indices, deg, pos=pos)
    assert np.isnan(t.seg_len_nm[0])


def test_endpoint_tangent_points_out_of_the_atom():
    ids = [10, 11, 12]
    pos = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0]], np.float32)
    indptr, indices, deg = build_adjacency(np.asarray(ids, np.uint64),
                                           np.asarray([[10, 11], [11, 12]], np.uint64))
    t = contract(indptr, indices, deg, pos=pos)
    idx, tan = endpoint_tangents(t, pos)
    assert len(idx) == 2
    # the two tips of a straight run must face opposite ways
    assert np.dot(tan[0], tan[1]) == pytest.approx(-1.0, abs=1e-5)


def test_endpoint_tangent_walks_the_segment_from_its_own_end():
    """A hooked leaf: five nodes along +x then four along +y, one segment.

    A straight run cannot catch a path that starts from the wrong end -- every
    node is on the same line. This one can: the tangent at the x-end, taken
    over ``span`` nodes back along its own segment, must lean into the hook,
    and must agree with the vectorised ``segment_tip_tangents`` that the
    topology build uses. A first version reversed the interior for BOTH ends,
    so the x-end's path ran from the far end of the segment (QA found 715 such
    tips on one shard, all at the 'a' end).
    """
    from neuronauts.harness.topology import segment_paths, segment_tip_tangents

    xs = [[100 * i, 0, 0] for i in range(5)]
    ys = [[400, 100 * j, 0] for j in range(1, 5)]
    pos = np.array(xs + ys, np.float32)
    n = len(pos)
    ids = np.arange(100, 100 + n, dtype=np.uint64)
    edges = np.array([[ids[i], ids[i + 1]] for i in range(n - 1)], np.uint64)
    indptr, indices, deg = build_adjacency(ids, edges)
    t = contract(indptr, indices, deg, pos=pos)

    idx, tan = endpoint_tangents(t, pos, span=5)
    flat, ptr = segment_paths(t)
    tip_v, tan_v = segment_tip_tangents(t, flat, ptr, pos, span=5)

    by_loop = {int(i): tan[k] for k, i in enumerate(idx.tolist())}
    by_vec = {int(i): tan_v[k] for k, i in enumerate(tip_v.tolist())}
    assert set(by_loop) == set(by_vec) == {0, n - 1}
    for tip in (0, n - 1):
        np.testing.assert_allclose(by_loop[tip], by_vec[tip], atol=1e-5)
    # and the x-end tangent really does lean into the hook, not straight -x
    assert by_loop[0][0] < 0 and by_loop[0][1] < 0


# ---------------------------------------------------------------------------
# vectorized helpers must agree with the per-segment definition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_vectorized_lengths_match_a_naive_per_segment_sum(seed):
    from neuronauts.harness.topology import segment_lengths, segment_paths

    rng = np.random.default_rng(seed)
    n = 40
    ids = np.arange(n, dtype=np.uint64)
    e = rng.integers(0, n, size=(55, 2))
    e = e[e[:, 0] != e[:, 1]]
    pos = rng.normal(0, 1000, size=(n, 3)).astype(np.float32)

    indptr, indices, deg = build_adjacency(ids, ids[e])
    t = contract(indptr, indices, deg, pos=pos)
    flat, ptr = segment_paths(t)
    fast = segment_lengths(flat, ptr, pos)

    for i in range(len(t.seg_ends)):
        path = np.concatenate([[t.seg_ends[i, 0]], t.seg_nodes[i],
                               [t.seg_ends[i, 1]]]).astype(np.int64)
        naive = np.linalg.norm(np.diff(pos[path], axis=0), axis=1).sum()
        assert fast[i] == pytest.approx(naive, rel=1e-4)
        # and the flat layout really is that path
        assert flat[ptr[i]:ptr[i + 1]].tolist() == path.tolist()


def test_vectorized_tangents_match_the_reference_at_tips():
    from neuronauts.harness.topology import segment_paths, segment_tip_tangents

    ids = np.asarray([10, 11, 12, 13], np.uint64)
    edges = np.asarray([[10, 11], [11, 12], [11, 13]], np.uint64)
    pos = np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0], [100, 500, 0]],
                   np.float32)
    indptr, indices, deg = build_adjacency(ids, edges)
    t = contract(indptr, indices, deg, pos=pos)
    flat, ptr = segment_paths(t)
    tip, tan = segment_tip_tangents(t, flat, ptr, pos)

    assert len(tip) == 3                      # three degree-1 tips
    got = {int(a): v for a, v in zip(tip.tolist(), tan)}
    assert np.allclose(got[0], [-1, 0, 0], atol=1e-5)   # node 10 points -x
    assert np.allclose(got[2], [1, 0, 0], atol=1e-5)    # node 12 points +x
    assert np.allclose(got[3], [0, 1, 0], atol=1e-5)    # node 13 points +y
