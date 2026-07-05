"""Offline test for cut-face slice-linking (synthetic seg volume, no network)."""
import numpy as np

from neuronauts.fetch import VolumeChunk
from experiments.proofread.cutface_slices import _footprints, _iou, evaluate_cutfaces


def _seg_vol():
    # object 5: a blob that persists across z at ~(10,10); object 7: a blob at (30,30)
    d = np.zeros((50, 50, 20), np.uint64)
    d[8:13, 8:13, :] = 5
    d[28:33, 28:33, :] = 7
    return VolumeChunk(data=d, voxel_size_nm=(32, 32, 40),
                       bbox_voxels=((0, 0, 0), d.shape), mip=2)


def test_footprints_and_iou():
    v = _seg_vol()
    fp = _footprints(v.data[:, :, 5], min_area=10)
    assert set(fp) == {5, 7}
    # same object across adjacent slices -> IoU 1; different objects -> 0
    a = _footprints(v.data[:, :, 5]); b = _footprints(v.data[:, :, 6])
    same = _iou(a[5][0], a[5][1], b[5][0], b[5][1], v.data.shape[:2])
    diff = _iou(a[5][0], a[5][1], b[7][0], b[7][1], v.data.shape[:2])
    assert same > 0.99 and diff == 0.0


def test_cutface_links_true_continuation():
    v = _seg_vol()
    res = evaluate_cutfaces(v, gaps=(1,), min_area=10, search_nm=3000.0, verbose=False)
    assert res[1]["iou_top1"] == 1.0        # footprint overlap picks the same object
    assert res[1]["mean_candidates"] >= 2


def test_motion_compensation_beats_raw_for_moving_object():
    # object 5 drifts +2 vox/slice in x; a distractor 7 sits where 5 STARTED.
    # raw IoU would prefer the stationary distractor; motion-comp should track 5.
    from experiments.proofread.cutface_slices import evaluate_combined
    d = np.zeros((60, 120, 30), np.uint64)
    for z in range(30):
        x0 = 20 + 2 * z
        d[25:35, x0:x0 + 10, z] = 5          # moving object
        d[25:35, 18:28, z] = 7               # stationary distractor near 5's origin
    v = VolumeChunk(data=d, voxel_size_nm=(32, 32, 40),
                    bbox_voxels=((0, 0, 0), d.shape), mip=2)
    res = evaluate_combined(v, gaps=(5,), traj_k=4, min_area=20,
                            search_nm=5000.0, verbose=False)
    assert 5 in res
    # motion-compensated cut-face should track the moving object at least as well as raw
    assert res[5]["iou_motioncomp_top1"] >= res[5]["iou_raw_top1"]
