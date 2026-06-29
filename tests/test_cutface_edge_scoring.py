"""Offline tests for cut-face edge scoring in neuronauts.em_corridor.

No network and no torch: cross-section extraction is pure numpy, and the edge
scorers take an injected ``embed_fn`` so we can drive them with a stub.
"""

import numpy as np
import pytest

from neuronauts.em_corridor import (
    CUTFACE_PATCH,
    cross_section_patch,
    batch_cutface_similarity,
)
from neuronauts.fetch import VolumeChunk


def _synthetic_volumes(side=64, nz=8):
    """Two neurite columns (ids 100, 200) in a small EM+seg grid."""
    rng = np.random.default_rng(0)
    em = rng.integers(110, 150, size=(side, side, nz)).astype(np.uint8)
    seg = np.zeros((side, side, nz), dtype=np.uint64)
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    for sid, (cx, cy, r) in {100: (20, 20, 6), 200: (44, 44, 5)}.items():
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        for z in range(nz):
            seg[:, :, z][disk] = np.uint64(sid)
    vox = (16, 16, 40)
    em_vol = VolumeChunk(data=em, voxel_size_nm=vox, bbox_voxels=((0, 0, 0), (side, side, nz)), mip=1)
    seg_vol = VolumeChunk(data=seg, voxel_size_nm=vox, bbox_voxels=((0, 0, 0), (side, side, nz)), mip=1)
    return em_vol, seg_vol


def test_cross_section_patch_shape_and_background():
    em_vol, seg_vol = _synthetic_volumes()
    # A point on neurite 100 (voxel ~ (20,20,4) -> nm).
    on = cross_section_patch(em_vol, seg_vol, (20 * 16, 20 * 16, 4 * 40))
    assert on.shape == (CUTFACE_PATCH, CUTFACE_PATCH)
    assert on.sum() > 0  # masked footprint present
    # A background corner point -> all zeros.
    off = cross_section_patch(em_vol, seg_vol, (0, 0, 4 * 40))
    assert off.shape == (CUTFACE_PATCH, CUTFACE_PATCH)
    assert off.sum() == 0


def test_batch_cutface_similarity_with_stub_embed(monkeypatch):
    em_vol, seg_vol = _synthetic_volumes()
    import neuronauts.fetch as fetch_mod

    monkeypatch.setattr(fetch_mod, "fetch_volume", lambda *a, **k: em_vol)
    monkeypatch.setattr(fetch_mod, "fetch_seg_volume", lambda *a, **k: seg_vol)

    # Stub encoder: flatten the patch (so identical faces -> cosine 1).
    embed_fn = lambda patches: patches.reshape(len(patches), -1)

    positions = np.array([
        [20 * 16, 20 * 16, 4 * 40],   # 0: on neurite 100
        [20 * 16, 20 * 16, 5 * 40],   # 1: same neurite, next section -> high sim
        [44 * 16, 44 * 16, 4 * 40],   # 2: neurite 200 -> low sim vs 0
    ], dtype=np.float64)
    edges = [(0, 1), (0, 2)]
    scores = batch_cutface_similarity(positions, edges, embed_fn, mip=1)

    assert set(scores) == {(0, 1), (0, 2)}
    for v in scores.values():
        assert -1.0001 <= v <= 1.0001
    # Same neurite (different section) should look more alike than a different one.
    assert scores[(0, 1)] > scores[(0, 2)]


def test_learned_encoder_roundtrip():
    torch = pytest.importorskip("torch")
    from experiments.fingerprints.learned_cutface_encoder import (
        build_encoder, embed_patches, make_embed_fn,
    )
    enc = build_encoder(embed_dim=16)
    patches = np.random.default_rng(0).normal(size=(5, CUTFACE_PATCH, CUTFACE_PATCH)).astype(np.float32)
    emb = embed_patches(enc, patches)
    assert emb.shape == (5, 16)
    # rows are L2-normalised by the encoder head
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-4)
    fn = make_embed_fn(enc)
    assert fn(patches).shape == (5, 16)
