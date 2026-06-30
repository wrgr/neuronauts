"""Offline tests for the v117 real-error re-linking harness.

The CAVE-dependent parts (finding split neurons, L2 positions) are not tested
here; we exercise the geometry helpers and the end-to-end ranking on a synthetic
volume with a stub encoder, by monkeypatching the box fetch.
"""

import numpy as np
import pytest

from experiments.fingerprints.cutface.fingerprint_break_resolution import Volume
from experiments.fingerprints.cutface import v117_error_relink as mod


def _synthetic_volume(side=64, nz=24, seed=0) -> Volume:
    rng = np.random.default_rng(seed)
    em = rng.integers(115, 140, size=(side, side, nz)).astype(np.uint8)
    seg = np.zeros((side, side, nz), dtype=np.uint64)
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    centers = {100: (16, 16, 6), 101: (16, 46, 5), 102: (46, 16, 5), 103: (46, 46, 6)}
    for sid, (cx, cy, r) in centers.items():
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        texture = rng.integers(40, 220, size=(side, side)).astype(np.int32)
        for z in range(nz):
            noise = rng.integers(-6, 7, size=(side, side))
            em[:, :, z][disk] = np.clip(texture + noise, 0, 255).astype(np.uint8)[disk]
            seg[:, :, z][disk] = np.uint64(sid)
    return Volume(em=em, seg=seg, resolution_nm=(16, 16, 40), origin_vox=(0, 0, 0))


def test_geometry_helpers_roundtrip():
    vol = _synthetic_volume()
    # neurite 100 centroid ~ (16,16) voxels; pick z=6 -> nm
    pos = (16 * 16, 16 * 16, 6 * 40)
    sid, idx = mod._seg_id_at(vol, pos)
    assert sid == 100
    assert mod._z_index(vol, pos[2]) == 6
    v = mod._flatnorm(np.arange(9, dtype=float).reshape(3, 3))
    np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-9)


def _split_volume(side=64, nz=24, seed=1):
    """Query fragment (100) and its true continuation (101) share a texture/shape
    -- they are 'one neuron split in two'; 102/103 are distractors."""
    rng = np.random.default_rng(seed)
    em = rng.integers(115, 140, size=(side, side, nz)).astype(np.uint8)
    seg = np.zeros((side, side, nz), dtype=np.uint64)
    r = 6
    w = 2 * r + 1
    ly, lx = np.meshgrid(np.arange(w), np.arange(w), indexing="ij")
    disk = (lx - r) ** 2 + (ly - r) ** 2 <= r * r
    shared_tmpl = rng.integers(40, 220, size=(w, w)).astype(np.int32)  # 100 & 101 share this
    tmpls = {100: shared_tmpl, 101: shared_tmpl,
             102: rng.integers(40, 220, size=(w, w)).astype(np.int32),
             103: rng.integers(40, 220, size=(w, w)).astype(np.int32)}
    centers = {100: (16, 16), 101: (40, 40), 102: (16, 46), 103: (46, 16)}
    for sid, (cx, cy) in centers.items():
        tmpl = tmpls[sid]
        for z in range(nz):
            noise = rng.integers(-3, 4, size=(w, w))
            block_em = em[cx - r:cx + r + 1, cy - r:cy + r + 1, z]
            block_seg = seg[cx - r:cx + r + 1, cy - r:cy + r + 1, z]
            block_em[disk] = np.clip(tmpl + noise, 0, 255).astype(np.uint8)[disk]
            block_seg[disk] = np.uint64(sid)
    return Volume(em=em, seg=seg, resolution_nm=(16, 16, 40), origin_vox=(0, 0, 0))


def test_evaluate_site_proximity_ranks_true_partner(monkeypatch):
    vol = _split_volume()
    monkeypatch.setattr(mod, "_fetch_box", lambda *a, **k: vol)

    # query on fragment 100, true continuation is the distinct id 101 (same texture).
    site = mod.ErrorSite(
        root=999,
        pos_main_nm=(16 * 16, 16 * 16, 6 * 40),
        pos_frag_nm=(40 * 16, 40 * 16, 6 * 40),
        gap_nm=540.0, frag_l2=5,
    )
    embed_fn = lambda patches: np.asarray(patches).reshape(len(patches), -1)
    res = mod.evaluate_site(site, embed_fn, mip=1, slab=3,
                            candidate_mode="proximity", radius_nm=4000.0)
    assert res is not None
    assert res.n_candidates >= 3          # 101 + distractors, query (100) excluded
    # the shared-texture continuation should rank first
    assert res.top1_raw
    assert res.rank_learned == 0


def test_fetch_box_disk_cache(tmp_path, monkeypatch):
    """_fetch_box caches each fetched box and reuses it (no second network call)."""
    from neuronauts.fetch import VolumeChunk
    calls = {"n": 0}

    def fake_em(bbox, mip=1):
        calls["n"] += 1
        em = np.full((8, 8, 4), 120, np.uint8)
        return VolumeChunk(data=em, voxel_size_nm=(16, 16, 40),
                           bbox_voxels=((0, 0, 0), (8, 8, 4)), mip=mip)

    def fake_seg(bbox, mip=1):
        seg = np.zeros((8, 8, 4), np.uint64)
        seg[2:6, 2:6, :] = 7
        return VolumeChunk(data=seg, voxel_size_nm=(16, 16, 40),
                           bbox_voxels=((0, 0, 0), (8, 8, 4)), mip=mip)

    monkeypatch.setattr(mod, "_fetch_em", fake_em)
    monkeypatch.setattr(mod, "_fetch_seg", fake_seg)
    monkeypatch.setattr(mod, "BOX_CACHE_DIR", str(tmp_path))

    v1 = mod._fetch_box((100.0, 100.0, 80.0), (150.0, 150.0, 120.0), 200.0, 1)
    assert calls["n"] == 1
    v2 = mod._fetch_box((100.0, 100.0, 80.0), (150.0, 150.0, 120.0), 200.0, 1)
    assert calls["n"] == 1                      # served from disk cache, not refetched
    assert np.array_equal(v1.seg, v2.seg)
    assert v2.resolution_nm == (16, 16, 40)
