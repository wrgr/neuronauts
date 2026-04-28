"""Tests for neuronauts/em_corridor.py.

All tests use synthetic data and do not require network access.
"""

import numpy as np
import pytest

from neuronauts.em_corridor import (
    CorridorSpec,
    corridor_intensity_stats,
    corridor_mask,
    corridor_seg_connectivity_score,
    corridors_from_boundary_edges,
    score_corridor_connectivity,
)
from neuronauts.fetch import VolumeChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_volume(
    shape=(20, 20, 10),
    voxel_size_nm=(32, 32, 40),
    bbox_origin=(0, 0, 0),
    seed=42,
) -> VolumeChunk:
    """Return a synthetic VolumeChunk filled with random uint8 values."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=shape, dtype=np.uint8)
    x0, y0, z0 = bbox_origin
    x1, y1, z1 = x0 + shape[0], y0 + shape[1], z0 + shape[2]
    return VolumeChunk(
        data=data,
        voxel_size_nm=voxel_size_nm,
        bbox_voxels=((x0, y0, z0), (x1, y1, z1)),
        mip=2,
    )


# ---------------------------------------------------------------------------
# CorridorSpec.bbox_nm
# ---------------------------------------------------------------------------

def test_corridor_spec_bbox():
    """bbox_nm should expand the tight bounding box by radius_nm on all sides."""
    pos_a = np.array([1000.0, 2000.0, 3000.0])
    pos_b = np.array([4000.0, 2000.0, 3000.0])
    radius = 500.0
    spec = CorridorSpec(pos_a_nm=pos_a, pos_b_nm=pos_b, radius_nm=radius)

    (x0, y0, z0), (x1, y1, z1) = spec.bbox_nm

    # x: min of endpoints expanded by radius
    assert x0 == pytest.approx(1000.0 - radius)
    assert x1 == pytest.approx(4000.0 + radius)
    # y: both endpoints equal, so tight bbox is [2000, 2000]; expanded by radius
    assert y0 == pytest.approx(2000.0 - radius)
    assert y1 == pytest.approx(2000.0 + radius)
    # z: similarly
    assert z0 == pytest.approx(3000.0 - radius)
    assert z1 == pytest.approx(3000.0 + radius)


def test_corridor_spec_bbox_asymmetric():
    """bbox_nm covers both endpoints regardless of which is larger per axis."""
    pos_a = np.array([5000.0, 1000.0, 8000.0])
    pos_b = np.array([2000.0, 6000.0, 3000.0])
    radius = 200.0
    spec = CorridorSpec(pos_a_nm=pos_a, pos_b_nm=pos_b, radius_nm=radius)

    (x0, y0, z0), (x1, y1, z1) = spec.bbox_nm

    assert x0 == pytest.approx(min(5000.0, 2000.0) - radius)
    assert x1 == pytest.approx(max(5000.0, 2000.0) + radius)
    assert y0 == pytest.approx(min(1000.0, 6000.0) - radius)
    assert y1 == pytest.approx(max(1000.0, 6000.0) + radius)
    assert z0 == pytest.approx(min(8000.0, 3000.0) - radius)
    assert z1 == pytest.approx(max(8000.0, 3000.0) + radius)


# ---------------------------------------------------------------------------
# CorridorSpec.length_nm
# ---------------------------------------------------------------------------

def test_corridor_spec_length():
    """length_nm should equal the Euclidean distance between the two endpoints."""
    pos_a = np.array([0.0, 0.0, 0.0])
    pos_b = np.array([3000.0, 4000.0, 0.0])
    spec = CorridorSpec(pos_a_nm=pos_a, pos_b_nm=pos_b)
    assert spec.length_nm == pytest.approx(5000.0)


def test_corridor_spec_length_3d():
    """length_nm in three dimensions."""
    pos_a = np.array([1000.0, 2000.0, 3000.0])
    pos_b = np.array([4000.0, 6000.0, 3000.0])
    expected = float(np.linalg.norm(pos_b - pos_a))
    spec = CorridorSpec(pos_a_nm=pos_a, pos_b_nm=pos_b)
    assert spec.length_nm == pytest.approx(expected)


# ---------------------------------------------------------------------------
# corridor_mask — shape
# ---------------------------------------------------------------------------

def test_corridor_mask_shape():
    """corridor_mask should return a boolean array with the same shape as volume.data."""
    volume = _make_fake_volume(shape=(15, 12, 8))
    # Place the corridor entirely inside the volume (in nm).
    vox = volume.voxel_size_nm  # (32, 32, 40)
    # Volume spans voxels 0..15 in x → 0..480 nm, etc.
    pos_a = np.array([100.0, 100.0, 60.0])
    pos_b = np.array([380.0, 280.0, 260.0])
    spec = CorridorSpec(pos_a_nm=pos_a, pos_b_nm=pos_b, radius_nm=200.0)
    mask = corridor_mask(spec, volume)

    assert mask.shape == volume.data.shape
    assert mask.dtype == bool


# ---------------------------------------------------------------------------
# corridor_mask — endpoints included
# ---------------------------------------------------------------------------

def test_corridor_mask_contains_endpoints():
    """Voxels nearest to pos_a and pos_b must be True in the mask."""
    # Use a large volume so both endpoints are well inside.
    shape = (50, 50, 20)
    vox_size = (32, 32, 40)
    volume = _make_fake_volume(shape=shape, voxel_size_nm=vox_size, bbox_origin=(0, 0, 0))

    # Place endpoints at known voxel centres.
    # Voxel (10, 10, 5) has centre at nm: (10 + 0.5)*32, (10 + 0.5)*32, (5 + 0.5)*40
    #   = (336, 336, 220)
    pos_a = np.array([336.0, 336.0, 220.0])
    # Voxel (40, 40, 15) → (40.5*32, 40.5*32, 15.5*40) = (1296, 1296, 620)
    pos_b = np.array([1296.0, 1296.0, 620.0])

    spec = CorridorSpec(pos_a_nm=pos_a, pos_b_nm=pos_b, radius_nm=200.0)
    mask = corridor_mask(spec, volume)

    # The voxel that contains pos_a must be True.
    assert mask[10, 10, 5], "Voxel at pos_a should be inside the corridor mask"
    # The voxel that contains pos_b must be True.
    assert mask[40, 40, 15], "Voxel at pos_b should be inside the corridor mask"


# ---------------------------------------------------------------------------
# corridors_from_boundary_edges — filtering long edges
# ---------------------------------------------------------------------------

def test_corridors_from_boundary_edges_filters_long():
    """Edges longer than max_length_nm must be dropped."""
    # Two synapses 20 000 nm apart → longer than default 15 000 nm limit.
    syn_positions = np.array([
        [0.0, 0.0, 0.0],
        [20_000.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],  # close to synapse 0
    ])
    boundary_edges = [(0, 1), (0, 2)]
    specs = corridors_from_boundary_edges(
        syn_positions,
        boundary_edges,
        max_length_nm=15_000.0,
    )
    # Edge (0, 1) is 20 000 nm — should be filtered out.
    assert len(specs) == 1
    assert specs[0].edge_key == (0, 2)


def test_corridors_from_boundary_edges_all_short():
    """All short edges should produce one CorridorSpec per edge."""
    syn_positions = np.array([
        [0.0, 0.0, 0.0],
        [1000.0, 0.0, 0.0],
        [2000.0, 0.0, 0.0],
        [3000.0, 0.0, 0.0],
    ])
    boundary_edges = [(0, 1), (1, 2), (2, 3)]
    specs = corridors_from_boundary_edges(
        syn_positions,
        boundary_edges,
        max_length_nm=15_000.0,
    )
    assert len(specs) == 3
    for idx, (i, j) in enumerate(boundary_edges):
        assert specs[idx].edge_key == (i, j)


def test_corridors_from_boundary_edges_edge_key_matches():
    """Each spec's edge_key must match the originating (i, j) pair."""
    rng = np.random.default_rng(0)
    syn_positions = rng.uniform(0, 5000, size=(10, 3))
    boundary_edges = [(0, 3), (2, 7), (4, 9)]
    specs = corridors_from_boundary_edges(syn_positions, boundary_edges)
    keys = [s.edge_key for s in specs]
    # All edges are well within 15 000 nm for random positions in [0, 5000].
    for edge in boundary_edges:
        assert edge in keys


# ---------------------------------------------------------------------------
# corridor_intensity_stats — keys present
# ---------------------------------------------------------------------------

def test_intensity_stats_keys():
    """corridor_intensity_stats must return all required keys."""
    volume = _make_fake_volume(shape=(20, 20, 10))
    pos_a = np.array([100.0, 100.0, 60.0])
    pos_b = np.array([500.0, 500.0, 340.0])
    spec = CorridorSpec(pos_a_nm=pos_a, pos_b_nm=pos_b, radius_nm=300.0)

    stats = corridor_intensity_stats(spec, volume)

    required_keys = {"mean", "std", "n_voxels", "fraction_bright", "length_nm"}
    assert required_keys == set(stats.keys()), (
        f"Missing keys: {required_keys - set(stats.keys())}"
    )


def test_intensity_stats_ranges():
    """Stats should be within plausible ranges for uint8 data."""
    volume = _make_fake_volume(shape=(20, 20, 10))
    pos_a = np.array([100.0, 100.0, 60.0])
    pos_b = np.array([500.0, 500.0, 340.0])
    spec = CorridorSpec(pos_a_nm=pos_a, pos_b_nm=pos_b, radius_nm=300.0)

    stats = corridor_intensity_stats(spec, volume)

    assert 0.0 <= stats["mean"] <= 255.0
    assert stats["std"] >= 0.0
    assert stats["n_voxels"] >= 0
    assert 0.0 <= stats["fraction_bright"] <= 1.0
    assert stats["length_nm"] > 0.0


# ---------------------------------------------------------------------------
# score_corridor_connectivity — always in [0, 1]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fraction_bright", [0.0, 0.1, 0.5, 0.9, 1.0, 2.0])
def test_score_corridor_connectivity_range(fraction_bright):
    """score_corridor_connectivity must always return a value in [0, 1]."""
    stats = {
        "mean": 128.0,
        "std": 30.0,
        "n_voxels": 500,
        "fraction_bright": fraction_bright,
        "length_nm": 3000.0,
    }
    score = score_corridor_connectivity(stats)
    assert 0.0 <= score <= 1.0, f"Score {score} out of range for fraction_bright={fraction_bright}"


def test_score_corridor_connectivity_monotone():
    """Higher fraction_bright should yield a higher or equal score."""
    def _score(fb):
        return score_corridor_connectivity({
            "mean": 128.0, "std": 20.0, "n_voxels": 100,
            "fraction_bright": fb, "length_nm": 1000.0,
        })

    assert _score(0.0) <= _score(0.25) <= _score(0.5)


# ---------------------------------------------------------------------------
# corridor_seg_connectivity_score tests
# ---------------------------------------------------------------------------

def _make_seg_volume(
    shape=(20, 20, 10),
    voxel_size_nm=(64, 64, 40),
    bbox_origin_vox=(0, 0, 0),
    fill: int = 0,
) -> VolumeChunk:
    """Return a VolumeChunk filled with a constant uint64 seg ID."""
    data = np.full(shape, fill, dtype=np.uint64)
    return VolumeChunk(
        data=data,
        voxel_size_nm=voxel_size_nm,
        bbox_voxels=(bbox_origin_vox, tuple(bbox_origin_vox[i] + shape[i] for i in range(3))),
        mip=3,
    )


def _make_two_seg_volume(
    shape=(20, 20, 10),
    voxel_size_nm=(64, 64, 40),
    seg_a: int = 111,
    seg_b: int = 222,
) -> VolumeChunk:
    """First half (x < shape[0]//2) filled with seg_a; second half with seg_b."""
    data = np.full(shape, seg_a, dtype=np.uint64)
    data[shape[0] // 2:] = seg_b
    return VolumeChunk(
        data=data,
        voxel_size_nm=voxel_size_nm,
        bbox_voxels=((0, 0, 0), shape),
        mip=3,
    )


def test_seg_score_same_id_returns_one():
    """Both endpoints on the same seg ID → score=1.0."""
    vox = (64, 64, 40)
    spec = CorridorSpec(
        pos_a_nm=np.array([100.0, 100.0, 100.0]),
        pos_b_nm=np.array([500.0, 500.0, 200.0]),
        radius_nm=200.0,
        mip=3,
    )
    vol = _make_seg_volume(shape=(20, 20, 10), voxel_size_nm=vox, fill=42)
    score = corridor_seg_connectivity_score(spec, vol)
    assert score == 1.0


def test_seg_score_different_ids_returns_zero():
    """Endpoints on different non-zero seg IDs → score=0.0."""
    vox = (64, 64, 40)
    # pos_a is in x < 10 (seg=111); pos_b is in x >= 10 (seg=222)
    spec = CorridorSpec(
        pos_a_nm=np.array([100.0, 100.0, 100.0]),   # x=100/64 ≈ vox 1 → seg 111
        pos_b_nm=np.array([900.0, 100.0, 100.0]),   # x=900/64 ≈ vox 14 → seg 222
        radius_nm=200.0,
        mip=3,
    )
    vol = _make_two_seg_volume(shape=(20, 20, 10), voxel_size_nm=vox, seg_a=111, seg_b=222)
    score = corridor_seg_connectivity_score(spec, vol)
    assert score == 0.0


def test_seg_score_background_returns_half():
    """If either endpoint is background (seg_id=0) → score=0.5."""
    vox = (64, 64, 40)
    spec = CorridorSpec(
        pos_a_nm=np.array([100.0, 100.0, 100.0]),
        pos_b_nm=np.array([500.0, 500.0, 200.0]),
        radius_nm=200.0,
        mip=3,
    )
    vol = _make_seg_volume(shape=(20, 20, 10), voxel_size_nm=vox, fill=0)
    score = corridor_seg_connectivity_score(spec, vol)
    assert score == 0.5


def test_seg_score_clamps_to_volume_boundary():
    """A position outside the volume bbox is clamped to the nearest voxel without error."""
    vox = (64, 64, 40)
    # pos_b is well outside the volume bounds
    spec = CorridorSpec(
        pos_a_nm=np.array([100.0, 100.0, 100.0]),
        pos_b_nm=np.array([99_000.0, 99_000.0, 99_000.0]),
        radius_nm=200.0,
        mip=3,
    )
    vol = _make_seg_volume(shape=(20, 20, 10), voxel_size_nm=vox, fill=7)
    # Should not raise; clamped pos_b ends up at last voxel, same seg_id=7
    score = corridor_seg_connectivity_score(spec, vol)
    assert score in (0.0, 0.5, 1.0)
