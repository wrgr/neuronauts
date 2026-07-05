"""Offline test for the raw-EM corridor sampler (synthetic volume, no network)."""
import numpy as np

from neuronauts.fetch import VolumeChunk
from experiments.proofread.follow_em import _sample_line


def _vol(data, vox=(16, 16, 40)):
    return VolumeChunk(data=data, voxel_size_nm=vox,
                       bbox_voxels=((0, 0, 0), data.shape), mip=1)


def test_min_profile_catches_a_dark_membrane():
    # uniform mid-gray cytoplasm with a thin dark membrane plane crossing mid-corridor
    data = np.full((60, 20, 6), 128, np.uint8)
    data[29:31, :, :] = 90                      # ~2-vox dark membrane at mid-x
    v = _vol(data)
    a = np.array([100.0, 160.0, 120.0]); b = np.array([840.0, 160.0, 120.0])
    prof = _sample_line(v, a, b, K=24, rad=1)
    assert prof.shape[0] == 48                  # [means(24), mins(24)]
    mins = prof[24:]
    # the min-profile dips at the membrane crossing; the mean barely moves
    assert mins.min() < (90 + 5) / 255.0
    assert prof[:24].min() > 100 / 255.0        # mean stays near cytoplasm


def test_uniform_corridor_has_no_dip():
    data = np.full((60, 20, 6), 128, np.uint8)
    v = _vol(data)
    prof = _sample_line(v, np.array([100.0, 160, 120]), np.array([840.0, 160, 120]), K=16)
    assert np.allclose(prof, 128 / 255.0, atol=1e-3)
