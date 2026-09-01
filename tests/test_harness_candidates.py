"""Candidate panel: endpoint-pair features and atom-pair reduction."""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.harness.candidates import (
    build_candidate_panel, endpoint_pair_features,
)


def test_facing_tips_score_high():
    # two endpoints 1 um apart, tangents pointing straight at each other
    pos_a = np.array([[0.0, 0.0, 0.0]])
    pos_b = np.array([[1000.0, 0.0, 0.0]])
    tan_a = np.array([[1.0, 0.0, 0.0]])   # points toward b
    tan_b = np.array([[-1.0, 0.0, 0.0]])  # points toward a
    gap, facing, al_a, al_b = endpoint_pair_features(pos_a, tan_a, pos_b, tan_b)
    assert gap[0] == pytest.approx(1000.0)
    assert facing[0] == pytest.approx(1.0)
    assert al_a[0] == pytest.approx(1.0)
    assert al_b[0] == pytest.approx(1.0)


def test_parallel_tangents_are_not_facing():
    # tangents pointing the same absolute direction (e.g. two unrelated
    # parallel neurites) are neither a continuation nor an antiparallel pair.
    pos_a = np.array([[0.0, 0.0, 0.0]])
    pos_b = np.array([[1000.0, 0.0, 0.0]])
    tan_a = np.array([[1.0, 0.0, 0.0]])
    tan_b = np.array([[1.0, 0.0, 0.0]])
    gap, facing, al_a, al_b = endpoint_pair_features(pos_a, tan_a, pos_b, tan_b)
    assert facing[0] == pytest.approx(-1.0)


def test_facing_alone_cannot_see_direction_along_the_gap():
    # Both tips retreat from the gap (a grows away from b, b grows away from
    # a): this is not a plausible continuation, but the tangents are still
    # antiparallel, so `facing` alone scores it identically to a genuine
    # continuation. `align_a`/`align_b` are what catch this case -- they
    # score -1 here because neither tip points along the connecting axis
    # toward its partner. This is by design: `facing` measures only whether
    # the two tangent directions oppose each other, not their sense relative
    # to the endpoint positions.
    pos_a = np.array([[0.0, 0.0, 0.0]])
    pos_b = np.array([[1000.0, 0.0, 0.0]])
    tan_a = np.array([[-1.0, 0.0, 0.0]])  # retreats from b
    tan_b = np.array([[1.0, 0.0, 0.0]])   # retreats from a
    gap, facing, al_a, al_b = endpoint_pair_features(pos_a, tan_a, pos_b, tan_b)
    assert facing[0] == pytest.approx(1.0)      # antiparallel tangents
    assert al_a[0] == pytest.approx(-1.0)       # but neither points at the other
    assert al_b[0] == pytest.approx(-1.0)


def _two_atom_fixture():
    # atom 1: two endpoints; atom 2: two endpoints. One pair is close and
    # facing (the real candidate), the rest are far or averted.
    ep_atom = np.array([1, 1, 2, 2], np.uint64)
    ep_pos = np.array([[0, 0, 0], [0, 0, 20000],
                       [1000, 0, 0], [1000, 0, 30000]], np.float32)
    ep_tan = np.array([[1, 0, 0], [0, 0, 1],
                       [-1, 0, 0], [0, 0, 1]], np.float32)
    ep_leaf = np.array([2000, 2000, 2000, 2000], np.float32)
    ep_cal = np.array([50, 50, 50, 50], np.float32)
    return ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal


def test_panel_keeps_closest_endpoint_pair_per_atom_pair():
    ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal = _two_atom_fixture()
    panel = build_candidate_panel(ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal,
                                  min_leaf_nm=100, min_caliber_nm=10,
                                  radius_nm=50000, k=8)
    assert len(panel) == 1                       # one atom pair (1, 2)
    assert int(panel.atom_a[0]) == 1 and int(panel.atom_b[0]) == 2
    assert panel.col("gap_nm")[0] == pytest.approx(1000.0)
    assert panel.col("facing")[0] == pytest.approx(1.0)


def test_panel_respects_leaf_and_caliber_filters():
    ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal = _two_atom_fixture()
    panel = build_candidate_panel(ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal,
                                  min_leaf_nm=5000, min_caliber_nm=10,
                                  radius_nm=50000, k=8)
    assert len(panel) == 0


def test_panel_respects_radius():
    ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal = _two_atom_fixture()
    panel = build_candidate_panel(ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal,
                                  min_leaf_nm=100, min_caliber_nm=10,
                                  radius_nm=500, k=8)
    assert len(panel) == 0


def test_panel_never_pairs_same_atom():
    ep_atom = np.array([1, 1], np.uint64)
    ep_pos = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    ep_tan = np.array([[1, 0, 0], [-1, 0, 0]], np.float32)
    ep_leaf = np.array([2000.0, 2000.0], np.float32)
    ep_cal = np.array([50.0, 50.0], np.float32)
    panel = build_candidate_panel(ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal,
                                  min_leaf_nm=100, min_caliber_nm=10,
                                  radius_nm=50000, k=8)
    assert len(panel) == 0


def test_panel_atom_order_is_canonical():
    ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal = _two_atom_fixture()
    panel = build_candidate_panel(ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal,
                                  min_leaf_nm=100, min_caliber_nm=10,
                                  radius_nm=50000, k=8)
    assert np.all(panel.atom_a < panel.atom_b)


def test_empty_input_yields_empty_panel():
    panel = build_candidate_panel(np.zeros(0, np.uint64), np.zeros((0, 3), np.float32),
                                  np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
                                  np.zeros(0, np.float32))
    assert len(panel) == 0
    assert panel.meta["n_endpoints"] == 0


def test_atom_subset_restricts_endpoints():
    ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal = _two_atom_fixture()
    panel = build_candidate_panel(ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal,
                                  min_leaf_nm=100, min_caliber_nm=10,
                                  radius_nm=50000, k=8,
                                  atom_subset=np.array([1], np.uint64))
    assert len(panel) == 0     # only atom 1 survives -> no cross-atom pair


def test_save_load_roundtrip(tmp_path):
    ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal = _two_atom_fixture()
    panel = build_candidate_panel(ep_atom, ep_pos, ep_tan, ep_leaf, ep_cal,
                                  min_leaf_nm=100, min_caliber_nm=10,
                                  radius_nm=50000, k=8)
    from neuronauts.harness.candidates import load_panel
    p = tmp_path / "panel.npz"
    panel.save(p)
    back = load_panel(p)
    assert back.atom_a.tolist() == panel.atom_a.tolist()
    assert np.allclose(back.feat, panel.feat)
    assert back.meta["radius_nm"] == pytest.approx(50000.0)
