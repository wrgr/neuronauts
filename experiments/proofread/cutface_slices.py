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


def _interior_corr(em, z0, ays, axs, zc, bys, bxs, shift, shape):
    """Pearson corr of interior EM content over the motion-compensated overlap.

    Compares the *same physical location's* ultrastructure across the gap: shift the
    cut face by the extrapolated drift, intersect with the candidate footprint, and
    correlate A's interior intensities (at their original A position) with B's.  This
    is interior CONTENT (mitochondria, texture), not silhouette — and it is a
    *continuation* match (same process across a small gap), not a type classifier.
    """
    dy, dx = int(round(shift[0])), int(round(shift[1]))
    a_lin = (ays + dy) * shape[1] + (axs + dx)          # A shifted into z0+gap frame
    b_lin = bys * shape[1] + bxs
    _, ai, bi = np.intersect1d(a_lin, b_lin, assume_unique=False, return_indices=True)
    if len(ai) < 12:
        return 0.0
    av = em[ays[ai], axs[ai], z0].astype(np.float32)    # A interior at its own slice
    bv = em[bys[bi], bxs[bi], zc].astype(np.float32)    # B interior at z0+gap
    if av.std() < 1e-3 or bv.std() < 1e-3:
        return 0.0
    return float(np.corrcoef(av, bv)[0, 1])


def _ring_brightness(em, zc, ys, xs, shape, grow=2):
    """Mean EM in a thin ring just outside the footprint (myelin/membrane character)."""
    from scipy.ndimage import binary_dilation
    y0, y1 = ys.min(), ys.max() + 1; x0, x1 = xs.min(), xs.max() + 1
    pad = grow + 1
    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
    y1 = min(shape[0], y1 + pad); x1 = min(shape[1], x1 + pad)
    m = np.zeros((y1 - y0, x1 - x0), bool); m[ys - y0, xs - x0] = True
    ring = binary_dilation(m, iterations=grow) & ~m
    if not ring.any():
        return 128.0
    crop = em[y0:y1, x0:x1, zc]
    return float(crop[ring].mean())


def evaluate_follow_ultra(seg, em, *, gaps=(3, 6, 10), traj_k=5, min_area=25,
                          search_nm=2500.0, max_per_gap=180, seed=0, verbose=True):
    """Add the ULTRASTRUCTURE channel (interior content + myelin-ring) to the fused
    follower, and measure the connect-vs-abstain signal (cost of failing to connect).

    Two comparisons:
    * geometry fusion vs geometry+ultrastructure — does interior content lift the hard
      *confusable* cuts (similar silhouettes, different insides)?
    * abstention: terminal cut faces (object does NOT continue) are included as
      all-negative instances; the fused score's top-candidate margin should be LOW
      there — so the follower knows when *not* to connect (a real neurite tip), which
      under the merge≫split cost asymmetry is what prevents false merges.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    d = seg.data; E = em.data; vox = np.asarray(seg.voxel_size_nm, float)
    nz = d.shape[2]; shape2d = d.shape[:2]
    assert E.shape[:3] == d.shape[:3], "seg and EM must share the grid (same mip/box)"
    rng = np.random.default_rng(seed)
    fp = {}
    def foot(z):
        if z not in fp:
            fp[z] = _footprints(d[:, :, z], min_area)
        return fp[z]

    cut_slices = list(range(traj_k + 1, nz - max(gaps) - 1, 2))
    rows = []          # per-candidate feature row
    inst_meta = []     # (iid, gap, z0, has_true)
    iid = 0
    for gap in gaps:
        cand_insts = []
        for z0 in cut_slices:
            face = foot(z0); nextf = foot(z0 + gap)
            if not nextf:
                continue
            nx_ids = list(nextf)
            nx_cent = np.array([nextf[j][2] for j in nx_ids]); nx_cent_nm = nx_cent * vox[:2]
            for oid, (ays, axs, acent, aarea) in face.items():
                acent_nm = acent * vox[:2]
                dcent = np.linalg.norm(nx_cent_nm - acent_nm, axis=1)
                pool = np.where(dcent <= search_nm)[0]
                if len(pool) < 2:
                    continue
                has_true = oid in nextf
                true_local = np.array([nx_ids[p] == oid for p in pool])
                confusable = has_true and bool(((dcent[pool] <= 600.0) & (~true_local)).any())
                cand_insts.append((z0, oid, ays, axs, acent, aarea, pool, nx_ids,
                                   nextf, has_true, confusable, dcent))
        # enrich: keep all confusable + terminal, sample the rest, cap per gap
        conf_t = [c for c in cand_insts if c[10] or not c[9]]
        rest = [c for c in cand_insts if not (c[10] or not c[9])]
        keep = conf_t + [rest[i] for i in rng.choice(len(rest),
                         min(len(rest), max(0, max_per_gap - len(conf_t))), replace=False)] \
               if rest else conf_t
        keep = keep[:max_per_gap]
        for (z0, oid, ays, axs, acent, aarea, pool, nx_ids, nextf, has_true, conf, dcent) in keep:
            c_prev = _centroid_on(d, oid, z0 - traj_k)
            vel = (acent - c_prev) / traj_k if c_prev is not None else np.zeros(2)
            pred_nm = (acent + vel * gap) * vox[:2]
            sy, sx = _shift_mask(ays, axs, vel * gap, shape2d)
            nx_cent_nm = np.array([nextf[j][2] for j in nx_ids]) * vox[:2]
            ringA = _ring_brightness(E, z0, ays, axs, shape2d)
            for p in pool:
                j = nx_ids[p]; bys, bxs = nextf[j][0], nextf[j][1]
                ir = _iou(ays, axs, bys, bxs, shape2d)
                im = _iou(sy, sx, bys, bxs, shape2d)
                td = np.linalg.norm(nx_cent_nm[p] - pred_nm) / 1000.0
                ar = -abs(np.log((nextf[j][3] + 1) / (aarea + 1)))
                ic = _interior_corr(E, z0, ays, axs, z0 + gap, bys, bxs, vel * gap, shape2d)
                rm = -abs(ringA - _ring_brightness(E, z0 + gap, bys, bxs, shape2d)) / 255.0
                rows.append([ir, im, td, ar, gap, ic, rm, int(j == oid)])
                inst_meta.append((iid, gap, z0, has_true))
            iid += 1
    if not rows:
        return {"error": "no instances"}
    A = np.array(rows, float)
    X = A[:, :7]; y = A[:, 7].astype(int)
    meta = np.array(inst_meta)  # iid, gap, z0, has_true
    grp = meta[:, 2]
    GEOM = [0, 1, 2, 3, 4]; ULTRA = [0, 1, 2, 3, 4, 5, 6]

    n_splits = max(2, min(5, len(np.unique(grp))))

    def oof(cols):
        s = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, grp):
            if len(np.unique(y[tr])) < 2:
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, class_weight="balanced"))
            m.fit(X[tr][:, cols], y[tr]); s[te] = m.predict_proba(X[te][:, cols])[:, 1]
        return s

    s_geom, s_ultra = oof(GEOM), oof(ULTRA)

    def top1(scores, gap, confusable_only=False):
        hit = n = 0
        for k in np.unique(meta[meta[:, 1] == gap, 0]):
            sel = meta[:, 0] == k
            if not meta[sel][0, 3]:      # terminal cut -> no true continuation; skip top-1
                continue
            yl = y[sel]; sc = scores[sel]
            if np.isnan(sc).all():
                continue
            hit += int(yl[np.argmax(sc)]); n += 1
        return hit / n if n else float("nan")

    # abstention: does max fused score separate has-true (connect) from terminal (abstain)?
    def connect_auc(scores):
        mx, lab = [], []
        for k in np.unique(meta[:, 0]):
            sel = meta[:, 0] == k; sc = scores[sel]
            if np.isnan(sc).all():
                continue
            mx.append(np.nanmax(sc)); lab.append(int(meta[sel][0, 3]))
        lab = np.array(lab)
        if len(np.unique(lab)) < 2:
            return float("nan")
        return float(roc_auc_score(lab, mx))

    res = {"gaps": list(gaps), "n_rows": len(y),
           "connect_auc_geom": connect_auc(s_geom),
           "connect_auc_ultra": connect_auc(s_ultra)}
    for gap in gaps:
        res[f"gap{gap}"] = {
            "fused_geom": top1(s_geom, gap), "fused_ultra": top1(s_ultra, gap)}
    if verbose:
        print(f"rows={len(y)}  (interior-content + myelin-ring added to geometry fusion)")
        print(f"{'gap(nm)':>8} {'fused_geom':>11} {'fused+ULTRA':>12}")
        for gap in gaps:
            r = res[f"gap{gap}"]
            print(f"{gap*int(vox[2]):>8} {r['fused_geom']:>11.3f} {r['fused_ultra']:>12.3f}")
        print(f"connect-vs-abstain AUC (know when NOT to connect):"
              f" geom={res['connect_auc_geom']:.3f}  +ultra={res['connect_auc_ultra']:.3f}")
    return res


def evaluate_follow_matching(seg, *, gaps=(3, 6), traj_k=5, min_area=25,
                             search_nm=2500.0, seed=0, verbose=True):
    """Global one-to-one MATCHING vs greedy top-1 — the safety mechanism.

    Between the cut faces on slice ``z0`` and the candidate faces on ``z0+gap`` we build a
    bipartite graph weighted by motion-compensated cut-face IoU.  Greedy lets every cut
    face grab its best candidate (a candidate can be claimed by many → a terminal tip
    whose neighbour overlaps it becomes a false merge).  **Global** assigns confident
    edges first with mutual exclusion, so each face — on *either* side — claims at most one
    partner: the neighbour is taken by its own true continuation and the tip is correctly
    left unmatched.  We sweep the accept threshold and report precision (of committed
    matches, fraction same-object) vs coverage (true continuations recovered).
    """
    d = seg.data; vox = np.asarray(seg.voxel_size_nm, float)
    nz = d.shape[2]; shape2d = d.shape[:2]
    fp = {}
    def foot(z):
        if z not in fp:
            fp[z] = _footprints(d[:, :, z], min_area)
        return fp[z]
    cut_slices = list(range(traj_k + 1, nz - max(gaps) - 1, 2))

    greedy, glob, n_true = [], [], 0    # each edge: (weight, is_true)
    for gap in gaps:
        for z0 in cut_slices:
            A = foot(z0); B = foot(z0 + gap)
            if not A or not B:
                continue
            b_ids = list(B); b_cent = np.array([B[j][2] for j in b_ids]) * vox[:2]
            a_ids = list(A)
            for oid in a_ids:
                if oid in B:
                    n_true += 1
            # weighted edges (near pairs only), motion-compensated IoU
            edges = []       # (w, ai, bi, is_true)
            best_per_a = {}  # ai -> (w, bi, is_true)  for greedy
            for ai, oid in enumerate(a_ids):
                ays, axs, acent, aarea = A[oid]
                c_prev = _centroid_on(d, oid, z0 - traj_k)
                vel = (acent - c_prev) / traj_k if c_prev is not None else np.zeros(2)
                sy, sx = _shift_mask(ays, axs, vel * gap, shape2d)
                acent_nm = acent * vox[:2]
                near = np.where(np.linalg.norm(b_cent - acent_nm, axis=1) <= search_nm)[0]
                for bi in near:
                    j = b_ids[bi]
                    w = _iou(sy, sx, B[j][0], B[j][1], shape2d)
                    if w <= 0:
                        continue
                    it = int(j == oid)
                    edges.append((w, ai, bi, it))
                    if ai not in best_per_a or w > best_per_a[ai][0]:
                        best_per_a[ai] = (w, bi, it)
            # greedy: each a takes its best b (b may be reused)
            for ai, (w, bi, it) in best_per_a.items():
                greedy.append((w, it))
            # global: assign confident edges first, each a and b claimed once
            edges.sort(reverse=True)
            used_a, used_b = set(), set()
            for w, ai, bi, it in edges:
                if ai in used_a or bi in used_b:
                    continue
                used_a.add(ai); used_b.add(bi)
                glob.append((w, it))
    if n_true == 0:
        return {"error": "no true continuations"}
    G = np.array(greedy, float); GL = np.array(glob, float)

    def sweep(M):
        out = []
        for t in np.linspace(0.02, 0.9, 45):
            sel = M[:, 0] >= t
            nc = int(sel.sum())
            if nc == 0:
                continue
            tp = int(M[sel, 1].sum())
            out.append((float(t), tp / nc, tp / n_true, nc))
        return out

    pc_greedy, pc_glob = sweep(G), sweep(GL)

    def at_prec(curve, p):
        good = [r for r in curve if r[1] >= p]
        return max(good, key=lambda r: r[2]) if good else None

    res = {"n_true_continuations": n_true,
           "op_greedy_p95": at_prec(pc_greedy, 0.95), "op_glob_p95": at_prec(pc_glob, 0.95),
           "op_greedy_p99": at_prec(pc_greedy, 0.99), "op_glob_p99": at_prec(pc_glob, 0.99),
           "pc_greedy": pc_greedy, "pc_glob": pc_glob}
    if verbose:
        print(f"true continuations={n_true}")
        for p, kg, kl in (("0.95", "op_greedy_p95", "op_glob_p95"),
                          ("0.99", "op_greedy_p99", "op_glob_p99")):
            g, l = res[kg], res[kl]
            gs = f"cov={g[2]:.3f}@thr{g[0]:.2f}" if g else "unreached"
            ls = f"cov={l[2]:.3f}@thr{l[0]:.2f}" if l else "unreached"
            print(f"  P>={p}:  greedy {gs:22s}  GLOBAL {ls}")
        # precision at matched coverage
        def pat(curve, cov):
            c = [r for r in curve if r[2] >= cov]
            return max(c, key=lambda r: r[1])[1] if c else float("nan")
        print("  precision at matched coverage:")
        for cov in (0.5, 0.7, 0.85):
            print(f"    cov>={cov}: greedy P={pat(pc_greedy,cov):.3f}  GLOBAL P={pat(pc_glob,cov):.3f}")
    return res


def evaluate_follow_pipeline(seg, *, gaps=(3, 6), traj_k=5, min_area=25,
                             search_nm=2500.0, caliber_ratio=2.5, soma_um2=15.0,
                             seed=0, verbose=True):
    """The three-stage follower end-to-end: RANK -> GRAMMAR-VETO -> ABSTAIN/COMMIT.

    1. **Rank** — fused geometry score (cut-face IoU + motion-comp + trajectory) picks
       the top candidate per cut.
    2. **Grammar veto** — reject the top candidate if the join is ungrammatical: a
       caliber jump (cross-section **area ratio** > ``caliber_ratio``) or joining two
       soma-scale objects (both areas > ``soma_um2`` µm²).  This is Pillar-1
       ``grammar_energy``'s caliber/soma terms applied to the local join, computed from
       the real cross-section areas.
    3. **Abstain or commit** — commit only if the fused score ≥ threshold; else abstain.

    Terminal cut faces (the object does *not* continue) are included, so committing to
    one is a **false merge** — the expensive error.  We sweep the threshold and report
    precision (of commits, fraction that are the correct true continuation) vs coverage
    (fraction of real continuations correctly fixed), **with and without the veto**.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold

    d = seg.data; vox = np.asarray(seg.voxel_size_nm, float)
    px_um2 = (vox[0] * vox[1]) / 1e6
    nz = d.shape[2]; shape2d = d.shape[:2]
    fp = {}
    def foot(z):
        if z not in fp:
            fp[z] = _footprints(d[:, :, z], min_area)
        return fp[z]
    cut_slices = list(range(traj_k + 1, nz - max(gaps) - 1, 2))

    rows, meta = [], []   # row: [iou,imc,td,ar,gap, area_a_um2, area_b_um2, is_true]; meta:[iid,has_true]
    iid = 0
    for gap in gaps:
        for z0 in cut_slices:
            face = foot(z0); nextf = foot(z0 + gap)
            if not nextf:
                continue
            nx_ids = list(nextf)
            nx_cent_nm = np.array([nextf[j][2] for j in nx_ids]) * vox[:2]
            for oid, (ays, axs, acent, aarea) in face.items():
                acent_nm = acent * vox[:2]
                dcent = np.linalg.norm(nx_cent_nm - acent_nm, axis=1)
                pool = np.where(dcent <= search_nm)[0]
                if len(pool) < 2:
                    continue
                has_true = oid in nextf
                c_prev = _centroid_on(d, oid, z0 - traj_k)
                vel = (acent - c_prev) / traj_k if c_prev is not None else np.zeros(2)
                pred_nm = (acent + vel * gap) * vox[:2]
                sy, sx = _shift_mask(ays, axs, vel * gap, shape2d)
                for p in pool:
                    j = nx_ids[p]; bys, bxs = nextf[j][0], nextf[j][1]
                    rows.append([
                        _iou(ays, axs, bys, bxs, shape2d),
                        _iou(sy, sx, bys, bxs, shape2d),
                        np.linalg.norm(nx_cent_nm[p] - pred_nm) / 1000.0,
                        -abs(np.log((nextf[j][3] + 1) / (aarea + 1))), gap,
                        aarea * px_um2, nextf[j][3] * px_um2, int(j == oid)])
                    meta.append([iid, int(has_true)])
                iid += 1
    if not rows:
        return {"error": "no instances"}
    A = np.array(rows, float); meta = np.array(meta)
    X = A[:, :5]; y = A[:, 7].astype(int)
    area_a, area_b = A[:, 5], A[:, 6]
    grp = np.array([m for m in meta[:, 0]])         # group by cut-instance for CV safety
    # fused geometry score (leakage-safe by cut slice would need z0; group by iid keeps
    # a cut's candidates together, which is what matters for ranking)
    s = np.full(len(y), np.nan)
    n_splits = max(2, min(5, len(np.unique(grp))))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
        m.fit(X[tr], y[tr]); s[te] = m.predict_proba(X[te])[:, 1]

    # per-cut: top candidate, its score, whether grammar-vetoed, whether it's the truth
    cuts = []
    for k in np.unique(meta[:, 0]):
        sel = np.where(meta[:, 0] == k)[0]
        sc = s[sel]
        if np.isnan(sc).all():
            continue
        top = sel[np.nanargmax(sc)]
        ratio = max(area_a[top], area_b[top]) / (min(area_a[top], area_b[top]) + 1e-9)
        vetoed = (ratio > caliber_ratio) or (area_a[top] > soma_um2 and area_b[top] > soma_um2)
        cuts.append((float(s[top]), int(y[top]), int(meta[sel[0], 1]), bool(vetoed)))
    cuts = np.array([(a, b, c, d_) for a, b, c, d_ in cuts], float)
    score, is_true_top, has_true, vetoed = cuts[:, 0], cuts[:, 1], cuts[:, 2], cuts[:, 3].astype(bool)
    n_realcont = int(has_true.sum())

    def sweep(use_veto):
        out = []
        for t in np.linspace(0.30, 0.95, 40):
            commit = (score >= t) & (~(vetoed & use_veto))
            nc = int(commit.sum())
            if nc == 0:
                continue
            tp = int(((is_true_top == 1) & (has_true == 1) & commit).sum())  # correct fix
            out.append((float(t), tp / nc, tp / max(1, n_realcont), nc))
        return out

    def op_point(curve):
        good = [r for r in curve if r[1] >= 0.95]
        return max(good, key=lambda r: r[2]) if good else None

    pc_noveto, pc_veto = sweep(False), sweep(True)
    res = {"n_cuts": len(cuts), "n_real_continuations": n_realcont,
           "veto_fire_rate": float(vetoed.mean()),
           "veto_kills_true": int((vetoed & (is_true_top == 1) & (has_true == 1)).sum()),
           "veto_kills_false": int((vetoed & ~((is_true_top == 1) & (has_true == 1))).sum()),
           "op_noveto": op_point(pc_noveto), "op_veto": op_point(pc_veto),
           "pc_noveto": pc_noveto, "pc_veto": pc_veto}
    if verbose:
        print(f"cuts={len(cuts)} (real continuations={n_realcont}, terminals={len(cuts)-n_realcont})")
        print(f"grammar veto fires on {vetoed.mean():.1%} of top picks; "
              f"kills {res['veto_kills_false']} false vs {res['veto_kills_true']} true connects")
        for tag, op in (("no-veto", res["op_noveto"]), ("+veto", res["op_veto"])):
            if op:
                print(f"  {tag:8s} P>=0.95 at thr={op[0]:.2f} -> precision={op[1]:.3f} "
                      f"coverage={op[2]:.3f} ({op[3]} commits)")
            else:
                print(f"  {tag:8s} never reaches P>=0.95")
    return res


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
