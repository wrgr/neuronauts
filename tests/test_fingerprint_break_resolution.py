"""Network-free tests for the cut-face fingerprint experiment.

Builds a synthetic EM+seg volume with several neurites whose cross-section
patterns are continuous through z, then checks that the image-patch hash
re-links the two faces of an artificial cut far above chance.
"""

import numpy as np
import pytest

from experiments.fingerprints.fingerprint_break_resolution import (
    Volume,
    FEATURE_NAMES,
    N_FEATURES,
    PATCH,
    face_hash,
    rank_matches,
    run_experiment,
)


def _synthetic_volume(n_neurites=6, side=64, nz=24, seed=0) -> Volume:
    rng = np.random.default_rng(seed)
    em = rng.integers(120, 140, size=(side, side, nz)).astype(np.uint8)  # bg tissue
    seg = np.zeros((side, side, nz), dtype=np.uint64)

    # Each neurite: a disk at a distinct location with a fixed idiosyncratic
    # internal texture broadcast through z (continuous), plus light per-z noise.
    centers = [(12, 12), (12, 40), (40, 12), (40, 40), (26, 26), (50, 50)][:n_neurites]
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    for k, (cx, cy) in enumerate(centers):
        r = 6 + k % 3
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        texture = rng.integers(40, 220, size=(side, side)).astype(np.int32)
        for z in range(nz):
            noise = rng.integers(-8, 9, size=(side, side))
            vals = np.clip(texture + noise, 0, 255).astype(np.uint8)
            em[:, :, z][disk] = vals[disk]
            seg[:, :, z][disk] = np.uint64(1000 + k)
    return Volume(em=em, seg=seg, resolution_nm=(8, 8, 40), origin_vox=(0, 0, 0))


def test_face_hash_shapes():
    vol = _synthetic_volume()
    grad = np.zeros_like(vol.em, dtype=np.float32)
    faces = face_hash(vol.em, vol.seg, grad, 4, 7, dark_thresh=100.0, min_vox=5)
    assert len(faces) >= 3
    for f in faces.values():
        assert f.vec.shape == (N_FEATURES,)
        assert f.patch.shape == (PATCH, PATCH)
    assert len(FEATURE_NAMES) == N_FEATURES


def test_patch_hash_relinks_above_chance():
    vol = _synthetic_volume()
    results = run_experiment(
        vol, slab_width=3, gaps=(2,), per_section_norm_variants=(False,),
    )
    assert results, "experiment produced no comparable cut"
    r = results[0]
    # Distinct, continuous textures -> the image hash should re-link nearly all
    # faces, and far above chance.
    assert r.top1_patch >= 0.8
    assert r.top1_patch > r.top1_chance
    # Scalar-summary hash is weaker by construction; just sanity-check bounds.
    assert 0.0 <= r.top1_hash <= 1.0


def test_rank_matches_needs_enough_faces():
    vol = _synthetic_volume(n_neurites=2)
    grad = np.zeros_like(vol.em, dtype=np.float32)
    faces = face_hash(vol.em, vol.seg, grad, 4, 7, dark_thresh=100.0, min_vox=5)
    # Fewer than 3 shared faces -> rank_matches declines (returns None).
    assert rank_matches(faces, faces) is None
