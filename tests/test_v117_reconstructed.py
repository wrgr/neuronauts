"""Offline tests for reconstructed-v117 re-linking (no CAVE / no network)."""

import numpy as np
import pytest

from experiments.fingerprints.cutface.fingerprint_break_resolution import Volume, PATCH
from experiments.fingerprints.cutface import v117_reconstructed as r


def _frag_volume(side=64, nz=24, seed=2):
    """4 v117 fragments; 100 & 101 share a (true) current root, 102/103 differ."""
    rng = np.random.default_rng(seed)
    em = rng.integers(115, 140, size=(side, side, nz)).astype(np.uint8)
    seg = np.zeros((side, side, nz), dtype=np.uint64)
    r0 = 6
    w = 2 * r0 + 1
    ly, lx = np.meshgrid(np.arange(w), np.arange(w), indexing="ij")
    disk = (lx - r0) ** 2 + (ly - r0) ** 2 <= r0 * r0
    shared = rng.integers(40, 220, size=(w, w)).astype(np.int32)  # 100 & 101 look alike
    tmpls = {100: shared, 101: shared,
             102: rng.integers(40, 220, size=(w, w)).astype(np.int32),
             103: rng.integers(40, 220, size=(w, w)).astype(np.int32)}
    centers = {100: (16, 16), 101: (40, 40), 102: (16, 46), 103: (46, 16)}
    for sid, (cx, cy) in centers.items():
        tmpl = tmpls[sid]
        for z in range(nz):
            noise = rng.integers(-3, 4, size=(w, w))
            seg[cx - r0:cx + r0 + 1, cy - r0:cy + r0 + 1, z][disk] = np.uint64(sid)
            blk = em[cx - r0:cx + r0 + 1, cy - r0:cy + r0 + 1, z]
            blk[disk] = np.clip(tmpl + noise, 0, 255).astype(np.uint8)[disk]
    vol = Volume(em=em, seg=seg, resolution_nm=(16, 16, 40), origin_vox=(0, 0, 0))
    frag2cur = {100: 9, 101: 9, 102: 5, 103: 7}   # 100 & 101 -> same current neuron
    return vol, frag2cur


def test_best_rank_picks_smallest_true():
    d = np.array([0.5, 0.1, 0.9, 0.2])
    is_true = np.array([False, False, True, True])  # true at idx 2 (rank?), 3
    # sorted by d: idx1(0.1)=rank0, idx3(0.2)=rank1, idx0(0.5)=rank2, idx2(0.9)=rank3
    assert r._best_rank(d, is_true) == 1   # best true is idx3 at rank 1


def test_site_faces_v117_marks_true_partner(monkeypatch):
    vol, frag2cur = _frag_volume()
    monkeypatch.setattr(r, "fetch_v117_box", lambda *a, **k: (vol, frag2cur))
    from experiments.fingerprints.cutface import v117_error_relink as v
    site = v.ErrorSite(root=1, pos_main_nm=(16 * 16, 16 * 16, 6 * 40),
                       pos_frag_nm=(40 * 16, 40 * 16, 6 * 40), gap_nm=540.0, frag_l2=5)
    f = r.site_faces_v117(None, None, site, radius_nm=4000.0, direction_cone_deg=45.0)
    assert f is not None
    assert f["query"].shape == (PATCH, PATCH)
    assert f["is_true"].sum() == 1               # exactly fragment 101 shares the current root
    assert f["patches"].shape[0] == len(f["is_true"])

    # with a stub flatten encoder, the shared-texture true partner should rank top-1
    embed = lambda P: np.asarray(P).reshape(len(P), -1)
    qe = embed(f["query"][None])[0]
    ce = embed(f["patches"])
    qe = qe / (np.linalg.norm(qe) + 1e-9)
    ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-9)
    assert r._best_rank(1.0 - ce @ qe, f["is_true"]) == 0
