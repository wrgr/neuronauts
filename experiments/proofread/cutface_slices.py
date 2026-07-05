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
    """IoU of two footprints via raveled linear indices (no full-slice allocation)."""
    a = np.unique(np.asarray(a_ys) * shape[1] + np.asarray(a_xs))
    b = np.unique(np.asarray(b_ys) * shape[1] + np.asarray(b_xs))
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - inter
    return inter / union if union else 0.0


def _centroid_on(d, oid, z, min_area=8):
    """Centroid (row,col) of object ``oid`` on slice z, or None."""
    ys, xs = np.nonzero(d[:, :, z] == oid)
    if len(ys) < min_area:
        return None
    return np.array([ys.mean(), xs.mean()])


def _shift_mask(ys, xs, dyx, shape):
    """Shift footprint indices by integer (dy, dx), clipped to the slice."""
    y2 = np.clip(np.round(ys + dyx[0]).astype(int), 0, shape[0] - 1)
    x2 = np.clip(np.round(xs + dyx[1]).astype(int), 0, shape[1] - 1)
    return y2, x2


def evaluate_combined(seg, *, gaps=(3, 6, 10, 15), traj_k=5, min_area=15,
                      cut_slices=None, search_nm=2500.0, conf_nm=600.0, verbose=True):
    """Combined follower: raw cut-face IoU vs trajectory-position vs MOTION-COMPENSATED
    cut-face IoU (shift the cut face by the extrapolated z-drift, then match shape).

    The fusion is the honest one: the fragment's own centroid track over the slices
    below the cut predicts *where* its cross-section should land ``gap`` slices later;
    motion-compensated IoU then checks the *shape* matches there.  Cut-face alone wins
    small gaps; trajectory-position is coarse; motion-compensated IoU should hold up
    across the gap where raw overlap has drifted away.
    """
    d = seg.data
    vox = np.asarray(seg.voxel_size_nm, float)
    nz = d.shape[2]; shape2d = d.shape[:2]
    if cut_slices is None:
        cut_slices = list(range(traj_k + 1, nz - max(gaps) - 1, 4))

    results = {}
    for gap in gaps:
        rows = []
        for z0 in cut_slices:
            if z0 + gap >= nz or z0 - traj_k < 0:
                continue
            face = _footprints(d[:, :, z0], min_area)
            nextf = _footprints(d[:, :, z0 + gap], min_area)
            if not nextf:
                continue
            nx_ids = list(nextf)
            nx_cent = np.array([nextf[j][2] for j in nx_ids])          # voxels
            nx_cent_nm = nx_cent * vox[:2]
            for oid, (ays, axs, acent, aarea) in face.items():
                if oid not in nextf:
                    continue
                # fragment velocity (vox/slice) from its centroid track below the cut
                c_now = acent
                c_prev = _centroid_on(d, oid, z0 - traj_k)
                vel = (c_now - c_prev) / traj_k if c_prev is not None else np.zeros(2)
                pred = c_now + vel * gap                               # predicted centroid (vox)
                pred_nm = pred * vox[:2]

                acent_nm = acent * vox[:2]
                dcent = np.linalg.norm(nx_cent_nm - acent_nm, axis=1)
                pool = np.where(dcent <= search_nm)[0]
                if len(pool) < 2:
                    continue
                true_local = np.array([nx_ids[p] == oid for p in pool])
                if not true_local.any():
                    continue
                sy, sx = _shift_mask(ays, axs, vel * gap, shape2d)     # motion-comp face
                iou_raw, iou_mc, tdist = [], [], []
                for p in pool:
                    bys, bxs = nextf[nx_ids[p]][0], nextf[nx_ids[p]][1]
                    iou_raw.append(_iou(ays, axs, bys, bxs, shape2d))
                    iou_mc.append(_iou(sy, sx, bys, bxs, shape2d))
                    tdist.append(np.linalg.norm(nx_cent_nm[p] - pred_nm))
                iou_raw = np.array(iou_raw); iou_mc = np.array(iou_mc); tdist = np.array(tdist)
                confusable = bool(((dcent[pool] <= conf_nm) & (~true_local)).any())
                rows.append((
                    int(true_local[np.argmax(iou_raw)]),
                    int(true_local[np.argmin(tdist)]),
                    int(true_local[np.argmax(iou_mc)]),
                    confusable, len(pool)))
        if not rows:
            continue
        R = np.array(rows, float); conf = R[:, 3] > 0
        results[gap] = {
            "n": len(R), "mean_candidates": float(R[:, 4].mean()),
            "iou_raw_top1": float(R[:, 0].mean()),
            "trajectory_top1": float(R[:, 1].mean()),
            "iou_motioncomp_top1": float(R[:, 2].mean()),
            "n_confusable": int(conf.sum()),
            "iou_raw_top1_conf": float(R[conf, 0].mean()) if conf.any() else float("nan"),
            "trajectory_top1_conf": float(R[conf, 1].mean()) if conf.any() else float("nan"),
            "iou_motioncomp_top1_conf": float(R[conf, 2].mean()) if conf.any() else float("nan"),
        }
        if verbose:
            r = results[gap]
            print(f"gap={gap} ({gap*int(vox[2])}nm)  n={r['n']}  cand={r['mean_candidates']:.1f}")
            print(f"   top-1   cutface_raw={r['iou_raw_top1']:.3f}  "
                  f"trajectory={r['trajectory_top1']:.3f}  "
                  f"COMBINED(motion-comp)={r['iou_motioncomp_top1']:.3f}")
            print(f"   confusable(n={r['n_confusable']})  cutface_raw={r['iou_raw_top1_conf']:.3f}"
                  f"  trajectory={r['trajectory_top1_conf']:.3f}"
                  f"  COMBINED={r['iou_motioncomp_top1_conf']:.3f}")
    return results


def evaluate_follow_fused(seg, *, gaps=(3, 6, 10, 15, 20), traj_k=5, min_area=15,
                          cut_slices=None, search_nm=2500.0, seed=0, verbose=True):
    """The combined follower: a LEARNED fusion of cut-face + trajectory per candidate.

    Per candidate we compute four honest geometry scores — raw cut-face IoU,
    motion-compensated IoU (cut face shifted by the extrapolated z-drift), trajectory
    position distance, and area match — plus the gap.  A leakage-safe logistic
    (GroupKFold by cut slice) learns to weight them, so it picks cut-face at short gaps
    and trajectory at long gaps automatically.  Reported against each single cue.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold

    d = seg.data; vox = np.asarray(seg.voxel_size_nm, float)
    nz = d.shape[2]; shape2d = d.shape[:2]
    if cut_slices is None:
        cut_slices = list(range(traj_k + 1, nz - max(gaps) - 1, 3))

    feats, labels, inst_id, grp, inst_gap = [], [], [], [], []
    iid = 0
    for gap in gaps:
        for z0 in cut_slices:
            if z0 + gap >= nz or z0 - traj_k < 0:
                continue
            face = _footprints(d[:, :, z0], min_area)
            nextf = _footprints(d[:, :, z0 + gap], min_area)
            if not nextf:
                continue
            nx_ids = list(nextf)
            nx_cent = np.array([nextf[j][2] for j in nx_ids]); nx_cent_nm = nx_cent * vox[:2]
            for oid, (ays, axs, acent, aarea) in face.items():
                if oid not in nextf:
                    continue
                c_prev = _centroid_on(d, oid, z0 - traj_k)
                vel = (acent - c_prev) / traj_k if c_prev is not None else np.zeros(2)
                pred_nm = (acent + vel * gap) * vox[:2]
                acent_nm = acent * vox[:2]
                dcent = np.linalg.norm(nx_cent_nm - acent_nm, axis=1)
                pool = np.where(dcent <= search_nm)[0]
                if len(pool) < 2:
                    continue
                true_local = np.array([nx_ids[p] == oid for p in pool])
                if not true_local.any():
                    continue
                sy, sx = _shift_mask(ays, axs, vel * gap, shape2d)
                for p in pool:
                    bys, bxs = nextf[nx_ids[p]][0], nextf[nx_ids[p]][1]
                    ir = _iou(ays, axs, bys, bxs, shape2d)
                    im = _iou(sy, sx, bys, bxs, shape2d)
                    td = np.linalg.norm(nx_cent_nm[p] - pred_nm) / 1000.0
                    ar = -abs(np.log((nextf[nx_ids[p]][3] + 1) / (aarea + 1)))
                    feats.append([ir, im, td, ar, gap]); labels.append(int(nx_ids[p] == oid))
                    inst_id.append(iid); grp.append(z0); inst_gap.append(gap)
                iid += 1
    if not feats:
        return {}
    X = np.array(feats, float); y = np.array(labels); inst_id = np.array(inst_id)
    grp = np.array(grp); inst_gap = np.array(inst_gap)

    learned = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, grp):
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(X[tr], y[tr]); learned[te] = m.predict_proba(X[te])[:, 1]

    def per_gap_top1(colscore):
        out = {}
        for gap in gaps:
            hits = n = 0
            for k in np.unique(inst_id[inst_gap == gap]):
                sel = inst_id == k
                sc = colscore(X[sel], learned[sel]); yl = y[sel]
                if np.isnan(sc).all():
                    continue
                hits += int(yl[np.argmax(sc)]); n += 1
            out[gap] = hits / n if n else float("nan")
        return out

    res = {"raw_cutface": per_gap_top1(lambda Xs, lr: Xs[:, 0]),
           "trajectory": per_gap_top1(lambda Xs, lr: -Xs[:, 2]),
           "motioncomp": per_gap_top1(lambda Xs, lr: Xs[:, 1]),
           "learned_fusion": per_gap_top1(lambda Xs, lr: lr)}
    if verbose:
        print(f"{'gap(nm)':>8} {'cutface':>9} {'traject':>9} {'motioncmp':>10} {'FUSED':>9}")
        for gap in gaps:
            print(f"{gap*int(vox[2]):>8} {res['raw_cutface'][gap]:>9.3f} "
                  f"{res['trajectory'][gap]:>9.3f} {res['motioncomp'][gap]:>10.3f} "
                  f"{res['learned_fusion'][gap]:>9.3f}")
    return res


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
