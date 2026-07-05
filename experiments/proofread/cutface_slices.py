"""Follow a process by its real CUT FACES on the following slices (not a blind line).

The correction: a fragment doesn't end at an abstract point — it ends in a real 2-D
**cut face** (its segmented cross-section on its terminal slice), and the continuation
is a real object cross-section on the *following* slices.  Following = linking those
real footprints slice-to-slice by their geometry (overlap, centroid, area), which is
what a proofreader (and RoboEM) actually use, and which a straight-corridor projection
threw away.

Honest, non-circular test: take a real seg object, **cut it at a z-slice**, and ask
whether its cut-face geometry re-links to the true continuation on slice ``z0+gap``
against every other object's footprint there.  Matching uses *footprint geometry only*
— the seg id is used solely to define ground truth and to segment the footprints, never
as a matching feature.  So it measures the real cut-face cue, not a seg-id shortcut.
"""
from __future__ import annotations

import numpy as np


def _footprints(seg_slice, min_area=15):
    """{id: (mask_indices, centroid_xy, area)} for objects on one z-slice."""
    ids, inv = np.unique(seg_slice, return_inverse=True)
    out = {}
    for k, i in enumerate(ids):
        if i == 0:
            continue
        ys, xs = np.nonzero(seg_slice == i)
        if len(ys) < min_area:
            continue
        out[int(i)] = (ys, xs, np.array([ys.mean(), xs.mean()]), len(ys))
    return out


def _iou(a_ys, a_xs, b_ys, b_xs, shape):
    A = np.zeros(shape, bool); A[a_ys, a_xs] = True
    B = np.zeros(shape, bool); B[b_ys, b_xs] = True
    inter = (A & B).sum(); union = (A | B).sum()
    return inter / union if union else 0.0


def evaluate_cutfaces(seg, *, gaps=(1, 3, 6), min_area=15, cut_slices=None,
                      conf_nm=600.0, search_nm=2000.0, verbose=True):
    """Cut-face continuity top-1 across z-gaps, footprint geometry only."""
    d = seg.data
    vox = np.asarray(seg.voxel_size_nm, float)
    nz = d.shape[2]
    shape2d = d.shape[:2]
    if cut_slices is None:
        cut_slices = list(range(6, nz - max(gaps) - 1, 4))

    results = {}
    for gap in gaps:
        rows = []  # (hit_iou, hit_centroid, hit_area, confusable, n_cand)
        for z0 in cut_slices:
            if z0 + gap >= nz:
                continue
            face = _footprints(d[:, :, z0], min_area)
            nextf = _footprints(d[:, :, z0 + gap], min_area)
            if not nextf:
                continue
            nx_ids = list(nextf)
            nx_cent = np.array([nextf[j][2] for j in nx_ids]) * vox[:2]
            nx_area = np.array([nextf[j][3] for j in nx_ids], float)
            for oid, (ays, axs, acent, aarea) in face.items():
                if oid not in nextf:      # object must actually continue past the cut
                    continue
                acent_nm = acent * vox[:2]
                dcent = np.linalg.norm(nx_cent - acent_nm, axis=1)
                pool = np.where(dcent <= search_nm)[0]   # real nearby candidates
                if len(pool) < 2:
                    continue
                # footprint-geometry scores vs the cut face (NO seg id used)
                iou = np.array([_iou(ays, axs, nextf[nx_ids[p]][0], nextf[nx_ids[p]][1],
                                     shape2d) for p in pool])
                cen = -dcent[pool]
                area_match = -np.abs(np.log((nx_area[pool] + 1) / (aarea + 1)))
                true_local = np.array([nx_ids[p] == oid for p in pool])
                if not true_local.any():
                    continue
                # a confusable cut has another object very close in xy to the cut face
                confusable = bool(((dcent[pool] <= conf_nm) & (~true_local)).any())
                rows.append((
                    int(true_local[np.argmax(iou)]),
                    int(true_local[np.argmax(cen)]),
                    int(true_local[np.argmax(iou + 0.0 * cen)]),  # iou primary
                    int(true_local[np.argmax(iou + area_match)]),
                    confusable, len(pool)))
        if not rows:
            continue
        R = np.array(rows, float)
        conf = R[:, 4] > 0
        results[gap] = {
            "n": len(R), "mean_candidates": float(R[:, 5].mean()),
            "chance": float((1.0 / R[:, 5]).mean()),
            "iou_top1": float(R[:, 0].mean()),
            "centroid_top1": float(R[:, 1].mean()),
            "iou_area_top1": float(R[:, 3].mean()),
            "n_confusable": int(conf.sum()),
            "iou_top1_confusable": float(R[conf, 0].mean()) if conf.any() else float("nan"),
            "centroid_top1_confusable": float(R[conf, 1].mean()) if conf.any() else float("nan"),
            "iou_area_top1_confusable": float(R[conf, 3].mean()) if conf.any() else float("nan"),
        }
        if verbose:
            r = results[gap]
            print(f"gap={gap} ({gap*int(vox[2])}nm)  n={r['n']}  cand={r['mean_candidates']:.1f}"
                  f"  chance={r['chance']:.3f}")
            print(f"   top-1     iou={r['iou_top1']:.3f}  centroid={r['centroid_top1']:.3f}"
                  f"  iou+area={r['iou_area_top1']:.3f}")
            print(f"   confusable(n={r['n_confusable']})  iou={r['iou_top1_confusable']:.3f}"
                  f"  centroid={r['centroid_top1_confusable']:.3f}"
                  f"  iou+area={r['iou_area_top1_confusable']:.3f}")
    return results
