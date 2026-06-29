"""Offline tests for real-error encoder training plumbing (no CAVE / no network)."""

import numpy as np
import pytest

from experiments.fingerprints.fingerprint_break_resolution import Volume, PATCH
from experiments.fingerprints import v117_error_relink as v
from experiments.fingerprints import train_real_cutface as tr


def _split_volume(side=64, nz=24, seed=1):
    """100 (query) and 101 (true partner) share a stamped template; 102/103 differ."""
    rng = np.random.default_rng(seed)
    em = rng.integers(115, 140, size=(side, side, nz)).astype(np.uint8)
    seg = np.zeros((side, side, nz), dtype=np.uint64)
    r = 6
    w = 2 * r + 1
    ly, lx = np.meshgrid(np.arange(w), np.arange(w), indexing="ij")
    disk = (lx - r) ** 2 + (ly - r) ** 2 <= r * r
    shared = rng.integers(40, 220, size=(w, w)).astype(np.int32)
    tmpls = {100: shared, 101: shared,
             102: rng.integers(40, 220, size=(w, w)).astype(np.int32),
             103: rng.integers(40, 220, size=(w, w)).astype(np.int32)}
    centers = {100: (16, 16), 101: (40, 40), 102: (16, 46), 103: (46, 16)}
    for sid, (cx, cy) in centers.items():
        tmpl = tmpls[sid]
        for z in range(nz):
            noise = rng.integers(-3, 4, size=(w, w))
            seg[cx - r:cx + r + 1, cy - r:cy + r + 1, z][disk] = np.uint64(sid)
            blk = em[cx - r:cx + r + 1, cy - r:cy + r + 1, z]
            blk[disk] = np.clip(tmpl + noise, 0, 255).astype(np.uint8)[disk]
    return Volume(em=em, seg=seg, resolution_nm=(16, 16, 40), origin_vox=(0, 0, 0))


def test_site_faces_returns_query_and_candidates(monkeypatch):
    vol = _split_volume()
    monkeypatch.setattr(v, "_fetch_box", lambda *a, **k: vol)
    site = v.ErrorSite(
        root=1, pos_main_nm=(16 * 16, 16 * 16, 6 * 40),
        pos_frag_nm=(40 * 16, 40 * 16, 6 * 40), gap_nm=540.0, frag_l2=5,
    )
    f = tr.site_faces(site, radius_nm=4000.0, direction_cone_deg=45.0)
    assert f is not None
    assert f["query"].shape == (PATCH, PATCH)
    assert f["patches"].shape[1:] == (PATCH, PATCH)
    assert 0 <= f["true_idx"] < len(f["cand_ids"])
    # the shared-template partner (101) is the true candidate, query id (100) excluded
    assert 100 not in f["cand_ids"]
    assert f["cand_ids"][f["true_idx"]] == 101


def test_finetune_reduces_loss():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    base = rng.normal(size=(48, PATCH, PATCH)).astype(np.float32)
    anchors = base + 0.1 * rng.normal(size=base.shape).astype(np.float32)
    positives = base + 0.1 * rng.normal(size=base.shape).astype(np.float32)
    distractors = rng.normal(size=(48, 6, PATCH, PATCH)).astype(np.float32)
    enc, losses = tr.finetune(
        anchors, positives, distractors, init_ckpt=None, epochs=10, batch=16, verbose=False)
    assert losses[-1] < losses[0]          # InfoNCE should drop when pos shares structure
    from experiments.fingerprints.learned_cutface_encoder import make_embed_fn
    assert make_embed_fn(enc)(anchors[:2]).shape == (2, 32)


def test_summary_shapes():
    s = tr._summary({"real": [0, 1, 0, 2], "raw": [1, 0, 3, 0]}, [10, 10, 10, 10])
    assert s["real"]["n"] == 4
    assert 0.0 <= s["real"]["top1"] <= 1.0
    assert s["chance_top1"] == pytest.approx(0.1)
