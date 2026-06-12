"""Tests for the shared world-assembly helpers (treestitch.worldbuild)."""

from __future__ import annotations

import numpy as np

from treestitch.worldbuild import (
    build_world_from_pieces,
    frankenmerge_adjacent,
    frankenmerge_random,
)


def _line_piece(obj_id, x0, n=10, step=1000.0):
    """A short straight skeleton piece starting at x=x0 with a few observations."""
    verts = np.stack([
        np.arange(n) * step + x0,
        np.zeros(n),
        np.zeros(n),
    ], axis=1).astype(np.float32)
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int64)
    radii = np.full(n, 200.0, dtype=np.float32)
    obs_pts = verts[:: max(1, n // 4)].copy()
    return {"obj_id": obj_id, "verts": verts, "edges": edges,
            "radii": radii, "obs_pts": obs_pts}


def test_build_world_default_one_seg_per_piece():
    pieces = [_line_piece(1, 0), _line_piece(1, 20_000), _line_piece(2, 100_000)]
    frags, region, lm = build_world_from_pieces(pieces)
    assert len(frags) == 3
    # observations: each piece contributes len(obs_pts)
    assert region.pre_pt_nm.shape[0] == sum(len(p["obs_pts"]) for p in pieces)
    # every fragment maps to exactly one object (no franken)
    assert all(len(v) == 1 for v in lm.values())


def test_build_world_franken_seg_owns_two_objects():
    pieces = [_line_piece(1, 0), _line_piece(2, 1_000)]
    # Force both pieces into one segment group.
    frags, region, lm = build_world_from_pieces(pieces, seg_of_piece=[0, 0])
    assert len(frags) == 1
    # The single fragment's seg id owns observations from both objects.
    franken = frags[0]
    objs = lm[franken.base_root_id]
    assert objs == {1, 2}
    # region pre_seg_id is constant (one seg), pre_root_id has both objects
    assert len(np.unique(region.pre_seg_id)) == 1
    assert set(np.unique(region.pre_root_id).tolist()) == {1, 2}


def test_frankenmerge_random_fuses_cross_object():
    pieces = [_line_piece(1, 0), _line_piece(1, 10_000),
              _line_piece(2, 500_000), _line_piece(2, 510_000)]
    seg, n = frankenmerge_random(pieces, 0.5, np.random.default_rng(0))
    assert n >= 1
    # at least one seg group spans two objects
    groups: dict[int, set[int]] = {}
    for pi, sid in enumerate(seg):
        groups.setdefault(sid, set()).add(pieces[pi]["obj_id"])
    assert any(len(o) == 2 for o in groups.values())


def test_frankenmerge_adjacent_only_fuses_nearby():
    # obj1 at x≈0, obj2 overlapping at x≈500 (adjacent), obj3 far away at x≈1e6.
    pieces = [_line_piece(1, 0), _line_piece(2, 500), _line_piece(3, 1_000_000)]
    seg, n = frankenmerge_adjacent(pieces, 1.0, np.random.default_rng(0), radius_nm=3000.0)
    assert n == 1
    # the fused pair must be the adjacent objects (1 and 2), never the far one (3)
    groups: dict[int, set[int]] = {}
    for pi, sid in enumerate(seg):
        groups.setdefault(sid, set()).add(pieces[pi]["obj_id"])
    fused = [o for o in groups.values() if len(o) == 2][0]
    assert fused == {1, 2}
    assert 3 not in fused


def test_frankenmerge_adjacent_none_when_all_distant():
    pieces = [_line_piece(1, 0), _line_piece(2, 1_000_000)]
    seg, n = frankenmerge_adjacent(pieces, 1.0, np.random.default_rng(0), radius_nm=3000.0)
    assert n == 0
    assert seg == [0, 1]


def test_frankenmerge_frac_zero_is_identity():
    pieces = [_line_piece(1, 0), _line_piece(2, 500)]
    seg_r, n_r = frankenmerge_random(pieces, 0.0, np.random.default_rng(0))
    seg_a, n_a = frankenmerge_adjacent(pieces, 0.0, np.random.default_rng(0))
    assert n_r == 0 and n_a == 0
    assert seg_r == [0, 1] and seg_a == [0, 1]
