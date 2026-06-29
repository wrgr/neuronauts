"""Offline tests for the v117 real-error re-linking harness.

The CAVE-dependent parts (finding split neurons, L2 positions) are not tested
here; we exercise the geometry helpers and the end-to-end ranking on a synthetic
volume with a stub encoder, by monkeypatching the box fetch.
"""

import numpy as np
import pytest

from experiments.fingerprints.fingerprint_break_resolution import Volume
from experiments.fingerprints import v117_error_relink as mod


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


def test_evaluate_site_ranks_true_partner(monkeypatch):
    vol = _synthetic_volume()
    monkeypatch.setattr(mod, "_fetch_box", lambda *a, **k: vol)

    # main and fragment points both on neurite 100 at different z -> true_id = 100.
    site = mod.ErrorSite(
        root=999,
        pos_main_nm=(16 * 16, 16 * 16, 6 * 40),
        pos_frag_nm=(16 * 16, 16 * 16, 17 * 40),
        gap_nm=440.0, frag_l2=5,
    )
    embed_fn = lambda patches: np.asarray(patches).reshape(len(patches), -1)
    res = mod.evaluate_site(site, embed_fn, mip=1, slab=3, margin_nm=0.0)
    assert res is not None
    assert res.n_candidates >= 3
    # distinct textures -> the true continuation should rank first
    assert res.top1_raw
    assert res.rank_learned == 0
