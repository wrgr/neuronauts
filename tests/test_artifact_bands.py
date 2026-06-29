"""Offline tests for the artifact/biological band split (no CAVE / no network)."""

import numpy as np
import pytest

from experiments.fingerprints.fingerprint_break_resolution import Volume, PATCH
from experiments.fingerprints import v117_artifact_bands as ab
from experiments.fingerprints import v117_reconstructed as r


def _disk_volume(side=64, nz=4, seed=0):
    rng = np.random.default_rng(seed)
    em = np.full((side, side, nz), 128, np.uint8)
    seg = np.zeros((side, side, nz), np.uint64)
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    disk = (xx - 32) ** 2 + (yy - 32) ** 2 <= 10 ** 2
    for z in range(nz):
        tex = rng.integers(60, 200, size=(side, side)).astype(np.uint8)
        em[:, :, z][disk] = tex[disk]
        seg[:, :, z][disk] = 7
    return em, seg


def test_band_face_shapes_and_footprint():
    em, seg = _disk_volume()
    low, high = ab._band_face(em, seg, 0, 2, 7, sigma=2.0)
    assert low.shape == (PATCH, PATCH) and high.shape == (PATCH, PATCH)
    # both bands zero outside the footprint; same support
    assert np.array_equal(low != 0, high != 0)
    # high-pass carries texture (nonzero variance), low is smoother than the raw
    assert high.std() > 0
    assert low.std() > 0


def test_band_face_empty_returns_none():
    em, seg = _disk_volume()
    assert ab._band_face(em, seg, 0, 2, 999, sigma=2.0) is None  # absent id


def test_site_faces_bands_marks_true(monkeypatch):
    # reuse the reconstructed-v117 synthetic fragment volume (100 & 101 share current root)
    from tests.test_v117_reconstructed import _frag_volume
    vol, frag2cur = _frag_volume()
    monkeypatch.setattr(r, "fetch_v117_box", lambda *a, **k: (vol, frag2cur))
    from experiments.fingerprints import v117_error_relink as v
    site = v.ErrorSite(root=1, pos_main_nm=(16 * 16, 16 * 16, 6 * 40),
                       pos_frag_nm=(40 * 16, 40 * 16, 6 * 40), gap_nm=540.0, frag_l2=5)
    f = ab.site_faces_bands(None, None, site, radius_nm=4000.0, direction_cone_deg=45.0)
    assert f is not None
    assert f["q_low"].shape == (PATCH, PATCH)
    assert f["low"].shape[0] == f["high"].shape[0] == len(f["is_true"])
    assert f["is_true"].sum() == 1
