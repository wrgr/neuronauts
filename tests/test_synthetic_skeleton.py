"""Offline test for synthetic same-fragment pair mining (no CAVE)."""
import numpy as np
from experiments.fingerprints.train_synthetic_skeleton import mine_box, _fragment_z_extents
from experiments.fingerprints.fingerprint_break_resolution import Volume, PATCH


def _multi_frag_box(side=64, nz=12, seed=0):
    rng = np.random.default_rng(seed)
    em = rng.integers(110, 150, size=(side, side, nz)).astype(np.uint8)
    seg = np.zeros((side, side, nz), np.uint64)
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    for k, (cx, cy) in enumerate([(16, 16), (16, 46), (46, 16), (46, 46), (31, 31)]):
        disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= 6 ** 2
        for z in range(nz):
            seg[:, :, z][disk] = 1000 + k
            em[:, :, z][disk] = rng.integers(60, 200, size=(side, side)).astype(np.uint8)[disk]
    return Volume(em=em, seg=seg, resolution_nm=(8, 8, 40), origin_vox=(0, 0, 0))


def test_z_extents_and_mining():
    vol = _multi_frag_box()
    ext = _fragment_z_extents(vol.seg)
    assert len(ext) == 5 and all(len(zs) == 12 for zs in ext.values())
    samples = mine_box(vol, slab=2, gap_sections=2, sigma=2.0, pairs_per_fragment=2)
    assert len(samples) > 0
    la, lp, ld, ha, hp, hd = samples[0]
    assert la.shape == (PATCH, PATCH) and lp.shape == (PATCH, PATCH)
    assert ld.shape[1:] == (PATCH, PATCH) and ld.shape[0] >= 2   # hard negatives present
    assert ha.shape == (PATCH, PATCH)   # high band too


def test_mining_needs_enough_fragments():
    # a box with <4 fragments yields nothing (need negatives)
    vol = _multi_frag_box()
    vol.seg[vol.seg > 1001] = 0   # keep only 2 fragments
    assert mine_box(vol, slab=2, gap_sections=2) == []
