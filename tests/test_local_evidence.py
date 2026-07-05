"""Offline tests for Pillar-2 local-EM evidence (synthetic volumes, no network)."""
import numpy as np

from neuronauts.fetch import VolumeChunk
from experiments.proofread.local_evidence import (
    LocalEvidence, local_evidence, _membrane_barrier)


def _vol(data, vox=(16, 16, 40)):
    return VolumeChunk(data=data, voxel_size_nm=vox,
                       bbox_voxels=((0, 0, 0), data.shape), mip=1)


def _ident_embed(patches):
    """A trivial embed_fn: flatten the patch (so cosine == raw-patch cosine)."""
    p = np.asarray(patches, np.float32).reshape(len(patches), -1)
    return p


def test_barrier_zero_on_uniform_cytoplasm():
    # uniformly dark cytoplasm along the whole axis -> no membrane barrier
    data = np.full((60, 20, 6), 80.0, np.float32)
    v = _vol(data)
    a = np.array([100.0, 160.0, 120.0])       # x=~6vox
    b = np.array([840.0, 160.0, 120.0])       # x=~52vox
    assert _membrane_barrier(v, a, b) < 0.05


def test_barrier_fires_on_dark_plane():
    # a dark membrane plane crossing the middle of the axis -> high barrier
    data = np.full((60, 20, 6), 120.0, np.float32)
    data[28:31, :, :] = 10.0                  # ~48nm dark membrane apposition at mid-x
    v = _vol(data)
    a = np.array([100.0, 160.0, 120.0])
    b = np.array([840.0, 160.0, 120.0])
    assert _membrane_barrier(v, a, b) > 0.5


def test_continuation_high_when_similar_and_no_barrier():
    ev = LocalEvidence(cutface_sim=0.95, barrier=0.02, axis_len_nm=5000.0, ok=True)
    assert ev.continuation > 0.9


def test_continuation_low_when_dissimilar_and_barrier():
    ev = LocalEvidence(cutface_sim=-0.5, barrier=0.9, axis_len_nm=5000.0, ok=True)
    assert ev.continuation < 0.25


def test_local_evidence_same_footprint_high_sim():
    # two points on the same seg id with identical local EM -> cutface_sim == 1
    seg = np.zeros((60, 40, 6), np.uint64)
    seg[:, 15:25, :] = 7                       # one horizontal process
    em = np.where(seg > 0, 70.0, 200.0).astype(np.float32)
    ev = local_evidence(
        np.array([100.0, 320.0, 120.0]), np.array([840.0, 320.0, 120.0]),
        _ident_embed, em_vol=_vol(em), seg_vol=_vol(seg))
    assert ev.ok
    assert ev.cutface_sim > 0.99             # identical footprints
    assert ev.barrier < 0.05                 # continuous cytoplasm
