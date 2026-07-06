"""Real proofreader false-splits via CAVE v117 -> later agglomeration (the honest test).

Fetch the graphene segmentation agglomerated at the **v117** timestamp (fragments, pre
much of the proofreading) and at a **later** version (splits fixed).  A voxel's v117 id
is its fragment; its later id is the proofread neuron it belongs to.  A **real false
split** = a later root that gathers >= 2 distinct v117 fragments — two pieces a human
merged.  This is dense, voxel-level ground truth (cut faces + which pieces are one
neuron), the same v117<->later machinery used throughout this repo.

The follower/matcher then re-links v117 fragments across their boundaries; a link is
correct iff the two fragments share a later root.  No appearance, no seg-id shortcut.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AggloVol:
    data: np.ndarray            # (X,Y,Z) int64 root ids at a version
    voxel_size_nm: tuple
    bbox_voxels: tuple


def fetch_agglo_volume(bbox_nm, version, *, token, mip=2,
                       datastack="minnie65_public") -> AggloVol:
    """Dense graphene cutout agglomerated at ``version``'s timestamp."""
    from caveclient import CAVEclient
    from cloudvolume import CloudVolume
    cl = CAVEclient(datastack, auth_token=token)
    seg_src = cl.info.segmentation_source()
    ts = cl.materialize.get_timestamp(version)
    cv = CloudVolume(seg_src, mip=mip, use_https=True, progress=False, fill_missing=True,
                     secrets={"token": token}, agglomerate=True, timestamp=ts)
    vox = tuple(int(x) for x in cv.resolution)
    lo = [int(bbox_nm[0][i] / vox[i]) for i in range(3)]
    hi = [int(bbox_nm[1][i] / vox[i]) for i in range(3)]
    b = cv.bounds
    lo = [max(lo[i], int(b.minpt[i])) for i in range(3)]
    hi = [min(hi[i], int(b.maxpt[i])) for i in range(3)]
    data = np.squeeze(cv[tuple(slice(lo[i], hi[i]) for i in range(3))]).astype(np.int64)
    return AggloVol(data=data, voxel_size_nm=vox, bbox_voxels=(tuple(lo), tuple(hi)))


def split_truth(v_early: np.ndarray, v_late: np.ndarray, *, min_frag_vox=20):
    """Map each early (v117) fragment id -> its majority later root; list real splits.

    Returns ``(later_of, splits)`` where ``later_of[frag]`` is the later root the
    fragment belongs to, and ``splits`` maps ``later_root -> set(frag ids)`` for later
    roots gathering >= 2 fragments (the false splits a proofreader merged).
    """
    from collections import defaultdict, Counter
    mask = (v_early > 0) & (v_late > 0)
    pairs = defaultdict(Counter)
    for a, b in zip(v_early[mask].ravel().tolist(), v_late[mask].ravel().tolist()):
        pairs[a][b] += 1
    later_of, frag_size = {}, {}
    for frag, ctr in pairs.items():
        b, n = ctr.most_common(1)[0]
        frag_size[frag] = sum(ctr.values())
        if frag_size[frag] >= min_frag_vox:
            later_of[frag] = b
    inv = defaultdict(set)
    for frag, later in later_of.items():
        inv[later].add(frag)
    splits = {later: frs for later, frs in inv.items() if len(frs) >= 2}
    return later_of, splits, frag_size


def evaluate_real_split_recovery(v117_data, vox, later_of, *, gaps=(1, 2, 3),
                                 traj_k=4, min_area=15, search_nm=2000.0, verbose=True):
    """Re-link v117 fragments by cut-face geometry; correct iff same LATER root.

    Focus on **split-boundary** cut faces: where a fragment ends and the neuron
    continues in a *different* v117 fragment sharing its later root (the real false
    split).  Global one-to-one matching by motion-compensated IoU; a link is correct iff
    the matched fragment has the same later root.  Reports split-boundary recovery
    (precision = correct links / committed split links; coverage = of split boundaries).
    """
    from experiments.proofread.cutface_slices import (
        _footprints, _iou, _shift_mask, _centroid_on)
    d = v117_data
    vox = np.asarray(vox, float); nz = d.shape[2]; shape2d = d.shape[:2]
    fp = {}
    def foot(z):
        if z not in fp:
            fp[z] = _footprints(d[:, :, z], min_area)
        return fp[z]
    L = lambda oid: later_of.get(int(oid))

    per_gap = {}
    for gap in gaps:
        edges = []          # (weight, a_key, b_key, correct, is_split_boundary)
        n_split_boundary = 0
        for z0 in range(traj_k + 1, nz - gap - 1, 1):
            A = foot(z0); B = foot(z0 + gap)
            if not A or not B:
                continue
            b_ids = list(B); b_cent = np.array([B[j][2] for j in b_ids]) * vox[:2]
            for oid, (ays, axs, acent, aarea) in A.items():
                la = L(oid)
                if la is None:
                    continue
                # velocity from this fragment's centroid track
                c_prev = _centroid_on(d, oid, z0 - traj_k)
                vel = (acent - c_prev) / traj_k if c_prev is not None else np.zeros(2)
                sy, sx = _shift_mask(ays, axs, vel * gap, shape2d)
                acent_nm = acent * vox[:2]
                near = np.where(np.linalg.norm(b_cent - acent_nm, axis=1) <= search_nm)[0]
                if len(near) == 0:
                    continue
                same_here = any(b_ids[bi] == oid for bi in near)
                cross_true = any(b_ids[bi] != oid and L(b_ids[bi]) == la for bi in near)
                is_split = cross_true and not same_here          # fragment ends, neuron continues in sibling
                if is_split:
                    n_split_boundary += 1
                for bi in near:
                    j = b_ids[bi]
                    if L(j) is None:
                        continue
                    w = _iou(sy, sx, B[j][0], B[j][1], shape2d)
                    if w <= 0:
                        continue
                    edges.append((w, (z0, oid), (z0 + gap, j), int(L(j) == la), is_split))
        # global one-to-one matching, then read off split-boundary recoveries
        edges.sort(reverse=True)
        used_a, used_b = set(), set()
        committed = []      # (weight, correct, is_split)
        for w, ak, bk, correct, is_split in edges:
            if ak in used_a or bk in used_b:
                continue
            used_a.add(ak); used_b.add(bk)
            committed.append((w, correct, is_split))
        C = np.array(committed, float) if committed else np.zeros((0, 3))
        def sweep(split_only):
            out = []
            sub = C[C[:, 2] == 1] if split_only else C
            for t in np.linspace(0.02, 0.8, 40):
                sel = sub[:, 0] >= t
                nc = int(sel.sum())
                if nc == 0:
                    continue
                tp = int(sub[sel, 1].sum())
                denom = n_split_boundary if split_only else max(1, len(C))
                out.append((float(t), tp / nc, tp / max(1, denom), nc))
            return out
        per_gap[gap] = {"n_split_boundary": n_split_boundary,
                        "n_committed": len(C), "n_committed_split": int(C[:, 2].sum()) if len(C) else 0,
                        "pc_all": sweep(False), "pc_split": sweep(True)}
        if verbose:
            r = per_gap[gap]
            def best_p(curve):
                return max((x[1] for x in curve), default=float("nan"))
            def cov_at(curve, p):
                g = [x for x in curve if x[1] >= p]
                return max((x[2] for x in g), default=0.0)
            print(f"gap={gap} ({gap*int(vox[2])}nm)  split-boundaries={r['n_split_boundary']}  "
                  f"committed-split-links={r['n_committed_split']}")
            print(f"   split recovery: best precision={best_p(r['pc_split']):.3f}  "
                  f"coverage@P>=0.9={cov_at(r['pc_split'],0.9):.3f}")
    return per_gap

