"""EXP-077 -- rebuild each contact panel with voxel identity from the
chunkedgraph, and measure gap, caliber and local axis exactly on the voxel grid.

Two things about the panels EXP-075 ranked over turn out to be substrate
artifacts rather than tissue.

**Identity.** `objects_v117_mip5_svmap.npz` maps supervoxel -> v117 root and
holds only the supervoxels a mip-5 read of the cube sampled. One mip-5 voxel
covers 8x8x4 mip-2 voxels, and measured here the map knows ~21% of the
supervoxels present in a mip-2 box. `build_contact_panels.py` labels its box
through that map and drops every voxel the map does not know (`ok = rt > 0`), so
each object enters the panel eroded to a fifth of itself. Distances between
eroded point sets are inflated. Where the map does have an entry it agrees with
the chunkedgraph 100% of the time -- it is incomplete, not wrong.

**Sampling.** Both the closest approach and the local axis were computed from
point sets thinned to a fixed budget (20,000 for the seed, 4,000 per candidate)
by taking every k-th voxel in raster order. For a seed with 4.1M voxels in the
box that is a stride of 205, and the "local axis" is then fitted to a handful of
scattered points.

Here the box is read once, every supervoxel in it is resolved at v117 (object
identity) and at v1822 (which proofread cell owns it), and then:

  gap      exact Euclidean distance transform from the seed's voxels, so every
           candidate's closest approach is measured against every seed voxel
  caliber  a single distance transform from the label boundary: the value at a
           voxel is its distance to the nearest voxel of a different label, so
           the max inside an object is its inscribed radius
  along,
  collin   principal axes from the FULL local voxel sets, not a thinned sample
  frac_cell fraction of the object's voxels the proofread cell owns at v1822
  corridor what fills the cylinder joining the seed and its labelled target

The seed-to-target gap is additionally recomputed under the two defects on their
own -- correct identity with the old thinning recipe, and the map's eroded
identity with no thinning -- so the inflation can be attributed.

    python scripts/probe_exp077_true_gap.py
"""
import argparse, glob, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

R = Path("/Users/wgray13/projects/neuronauts")
OUT = R / "data/external/panels_v2"
LOCAL_NM = 1500.0
CYL_R = 250.0


def axis_of(P):
    if len(P) < 3:
        return None
    Q = P - P.mean(0)
    C = (Q.T @ Q) / len(Q)
    w, v = np.linalg.eigh(C)
    return v[:, -1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", default="")
    ap.add_argument("--half-nm", type=float, default=4000.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    z = np.load(R / "data/substrate/c100um/object_clouds_mip5.npz", allow_pickle=False)
    obj, ptr, pos = z["object_id"], z["node_ptr"], z["pos_nm"]
    rowi = {int(a): k for k, a in enumerate(obj.tolist())}

    def pts(a):
        k = rowi.get(int(a))
        return pos[int(ptr[k]):int(ptr[k + 1])] if k is not None else np.empty((0, 3))

    sv = np.load(R / "data/substrate/c100um/objects_v117_mip5_svmap.npz", allow_pickle=False)
    SV, RT = sv["sv"], sv["root"]
    o = np.argsort(SV); SV, RT = SV[o], RT[o]

    def map_root(x):
        i = np.searchsorted(SV, x); i[i >= len(SV)] = 0
        return np.where(SV[i] == x, RT[i], 0)

    import caveclient
    from cloudvolume import CloudVolume
    cl = caveclient.CAVEclient("minnie65_public")
    ts117 = cl.materialize.get_timestamp(117)
    ts1822 = cl.materialize.get_timestamp(1822)
    cv = CloudVolume(cl.chunkedgraph.cloudvolume_path, mip=2, use_https=True,
                     progress=False, fill_missing=True, agglomerate=False)
    res = np.asarray(cv.resolution, float)

    cards = {}
    for f in glob.glob(str(R / "data/external/cell_cards/*.json")):
        if Path(f).name.startswith("_"):
            continue
        c = json.load(open(f)); cards[str(c["cell"])[-8:]] = c

    def roots(u, ts):
        r = np.zeros(len(u), dtype=np.uint64)
        for i in range(0, len(u), 50_000):
            r[i:i + 50_000] = np.asarray(cl.chunkedgraph.get_roots(u[i:i + 50_000], timestamp=ts),
                                         dtype=np.uint64)
        return r

    want = set(args.keys.split(",")) if args.keys else None
    done = 0
    for pf in sorted(glob.glob(str(R / "data/external/panels/*.npz"))):
        key = Path(pf).stem[-8:]
        if want and key not in want:
            continue
        dest = OUT / f"panel_{key}.npz"
        if dest.exists():
            continue
        if args.limit and done >= args.limit:
            break
        p = np.load(pf, allow_pickle=False)
        it = p["in_target"]
        if not it.any() or bool(p["already_whole"]):
            continue
        seed = int(p["seed"]); card = cards[key]; cell = int(card["cell"])
        tgt = int(p["obj"][it][np.argmin(p["gap_nm"][it])])
        tgtset = set(int(x) for x in card["structure"]["seeded_target"]) - {seed}

        Ps, Pt = pts(seed), pts(tgt)
        d, j = cKDTree(Pt).query(Ps, k=1)
        m = int(np.argmin(d)); ctr = (Ps[m] + Pt[int(j[m])]) / 2.0

        t0 = time.time()
        lo = np.floor((ctr - args.half_nm) / res).astype(int)
        hi = np.ceil((ctr + args.half_nm) / res).astype(int)
        vol = np.asarray(cv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])[..., 0]
        shape = vol.shape
        t_dl = time.time() - t0

        usv = np.unique(vol)
        usv = usv[usv != 0]
        t0 = time.time()
        r117 = roots(usv, ts117); r1822 = roots(usv, ts1822)
        t_rt = time.time() - t0
        mapped = map_root(usv); have = mapped > 0
        cov = float(have.mean())
        agree = float((mapped[have] == r117[have]).mean()) if have.any() else float("nan")

        # compact object index per voxel: 0 = background
        uo, inv = np.unique(r117, return_inverse=True)          # object ids present
        sv_to_oi = (inv + 1).astype(np.int32)                   # 1-based
        flat = vol.reshape(-1)
        k = np.searchsorted(usv, flat)
        k[k >= len(usv)] = 0
        oi = np.where((flat != 0) & (usv[k] == flat), sv_to_oi[k], 0).astype(np.int32)
        lab = oi.reshape(shape)
        del flat, k
        seed_oi = int(np.searchsorted(uo, np.uint64(seed))) + 1
        tgt_oi = int(np.searchsorted(uo, np.uint64(tgt))) + 1
        if uo[seed_oi - 1] != np.uint64(seed) or uo[tgt_oi - 1] != np.uint64(tgt):
            print(f"  {key}: seed or target absent from box", flush=True); continue

        t0 = time.time()
        # exact distance from the seed
        Dseed, IDX = distance_transform_edt(lab != seed_oi, sampling=res, return_indices=True)
        # caliber: distance to the nearest voxel carrying a different label
        bnd = np.zeros(shape, bool)
        for ax in range(3):
            a = np.take(lab, np.arange(shape[ax] - 1), axis=ax)
            b = np.take(lab, np.arange(1, shape[ax]), axis=ax)
            neq = a != b
            sl0 = [slice(None)] * 3; sl0[ax] = slice(0, shape[ax] - 1)
            sl1 = [slice(None)] * 3; sl1[ax] = slice(1, shape[ax])
            bnd[tuple(sl0)] |= neq; bnd[tuple(sl1)] |= neq
        Dbnd = distance_transform_edt(~bnd, sampling=res)
        t_edt = time.time() - t0

        # voxels grouped by object
        vox = np.flatnonzero(lab.reshape(-1))
        lv = lab.reshape(-1)[vox]
        order = np.argsort(lv, kind="stable")
        vox, lv = vox[order], lv[order]
        starts = np.searchsorted(lv, np.arange(1, len(uo) + 1))
        ends = np.searchsorted(lv, np.arange(1, len(uo) + 1), side="right")
        Df = Dseed.reshape(-1); Bf = Dbnd.reshape(-1)
        r1822_by_sv = dict(zip(usv.tolist(), r1822.tolist()))
        volf = vol.reshape(-1)

        def xyz(flat_idx):
            return np.stack(np.unravel_index(flat_idx, shape), -1)

        def local_pts(oi_want, cp):
            c = np.round(cp / res).astype(int) - lo
            rad = np.ceil(LOCAL_NM / res).astype(int)
            a0 = np.maximum(c - rad, 0); a1 = np.minimum(c + rad + 1, shape)
            sub = lab[a0[0]:a1[0], a0[1]:a1[1], a0[2]:a1[2]]
            w = np.stack(np.nonzero(sub == oi_want), -1)
            if not len(w):
                return np.empty((0, 3))
            P = (w + a0 + lo) * res
            return P[np.linalg.norm(P - cp, axis=1) < LOCAL_NM]

        rec = []
        for gi in range(len(uo)):
            a_oi = gi + 1
            s, e = int(starts[gi]), int(ends[gi])
            if e <= s:
                continue
            vi = vox[s:e]
            if a_oi == seed_oi:
                continue
            dd = Df[vi]
            kbest = int(np.argmin(dd)); gap = float(dd[kbest])
            cvox = xyz(vi[kbest])
            cpos = (cvox + lo) * res
            sidx = IDX[:, cvox[0], cvox[1], cvox[2]]
            spos = (sidx + lo) * res
            cp = (cpos + spos) / 2.0
            Sl = local_pts(seed_oi, cp); Cl = local_pts(a_oi, cp)
            aS, aC = axis_of(Sl), axis_of(Cl)
            if aS is None or aC is None:
                al = co = 0.0
            else:
                u = Cl.mean(0) - Sl.mean(0); n = np.linalg.norm(u)
                al = abs(float(aS @ (u / n))) if n > 0 else 0.0
                co = abs(float(aS @ aC))
            cal = float(Bf[vi].max())
            svs = volf[vi]
            fcell = float(np.mean([r1822_by_sv[int(x)] == cell for x in np.unique(svs).tolist()]))
            # voxel-weighted membership
            fcell_v = float(np.mean([r1822_by_sv[int(x)] == cell for x in svs.tolist()]))
            rec.append((int(uo[gi]), gap, al, co, e - s, cal, int(uo[gi]) in tgtset,
                        fcell_v, float(np.abs(cpos - ctr).max()), cp))

        # the seed's own numbers, and the corridor to the labelled target
        si, ei = int(starts[seed_oi - 1]), int(ends[seed_oi - 1])
        seed_vox = vox[si:ei]
        cal_seed_global = float(Bf[seed_vox].max())
        trow = [r for r in rec if r[0] == tgt][0]
        cp = trow[9]
        cal_seed_local = float(Bf[seed_vox][
            np.linalg.norm((xyz(seed_vox) + lo) * res - cp, axis=1) < LOCAL_NM].max()) \
            if (np.linalg.norm((xyz(seed_vox) + lo) * res - cp, axis=1) < LOCAL_NM).any() else np.nan

        # attribute the inflation: the two defects, separately
        def raster(P, cap):
            return P if len(P) <= cap else P[:: len(P) // cap][:cap]
        Sfull = (xyz(vox[si:ei]) + lo) * res
        tvi0 = vox[int(starts[tgt_oi - 1]):int(ends[tgt_oi - 1])]
        Tfull = (xyz(tvi0) + lo) * res
        gap_thin = float(cKDTree(raster(Sfull, 20000)).query(raster(Tfull, 4000), k=1)[0].min())
        mapped_ok = set(usv[have].tolist())
        mroot = dict(zip(usv[have].tolist(), mapped[have].tolist()))
        svS = volf[vox[si:ei]]; svT = volf[tvi0]
        mS = np.array([mroot.get(int(x), 0) == seed for x in svS.tolist()])
        mT = np.array([mroot.get(int(x), 0) == tgt for x in svT.tolist()])
        gap_eroded = (float(cKDTree(Sfull[mS]).query(Tfull[mT], k=1)[0].min())
                      if mS.any() and mT.any() else float("nan"))

        tvi = vox[int(starts[tgt_oi - 1]):int(ends[tgt_oi - 1])]
        kb = int(np.argmin(Df[tvi])); tv = xyz(tvi[kb]); sv_ = IDX[:, tv[0], tv[1], tv[2]]
        A = (tv + lo) * res; B = (sv_ + lo) * res
        L = float(np.linalg.norm(A - B))
        blo = np.maximum(np.floor((np.minimum(A, B) - CYL_R) / res).astype(int) - lo, 0)
        bhi = np.minimum(np.ceil((np.maximum(A, B) + CYL_R) / res).astype(int) - lo + 1, shape)
        gg = np.stack(np.meshgrid(*[np.arange(blo[t], bhi[t]) for t in range(3)], indexing="ij"), -1)
        Pg = (gg.reshape(-1, 3) + lo) * res
        u = (A - B) / max(L, 1e-9)
        tpar = np.clip((Pg - B) @ u, 0, L)
        sel = np.linalg.norm(Pg - (B + tpar[:, None] * u), axis=1) <= CYL_R
        gsel = gg.reshape(-1, 3)[sel]
        clab = lab[gsel[:, 0], gsel[:, 1], gsel[:, 2]]
        n_cyl = int(sel.sum()); n_bg = int((clab == 0).sum())
        cyl_obj = {}
        for oo, cc in zip(*np.unique(clab[clab > 0], return_counts=True)):
            cyl_obj[int(uo[oo - 1])] = int(cc)
        cyl_cell = 0
        csv = vol[gsel[:, 0], gsel[:, 1], gsel[:, 2]]
        for s_, c_ in zip(*np.unique(csv[csv > 0], return_counts=True)):
            if r1822_by_sv[int(s_)] == cell:
                cyl_cell += int(c_)

        corridor = {"seg_len_nm": round(L, 1), "radius_nm": CYL_R, "n_voxels": n_cyl,
                    "n_background": n_bg, "frac_background": round(n_bg / max(n_cyl, 1), 3),
                    "n_seed": cyl_obj.get(seed, 0), "n_target": cyl_obj.get(tgt, 0),
                    "n_same_cell_v1822": cyl_cell,
                    "n_other_objects": len([q for q in cyl_obj if q not in (seed, tgt)]),
                    "top_other": sorted([[int(q), int(v)] for q, v in cyl_obj.items()
                                         if q not in (seed, tgt)], key=lambda kv: -kv[1])[:5],
                    "at_target_nm": A.tolist(), "at_seed_nm": B.tolist()}

        np.savez(dest,
                 obj=np.array([r[0] for r in rec], dtype=np.uint64),
                 gap_nm=np.array([r[1] for r in rec], dtype=np.float32),
                 along=np.array([r[2] for r in rec], dtype=np.float32),
                 collin=np.array([r[3] for r in rec], dtype=np.float32),
                 n_vox=np.array([r[4] for r in rec], dtype=np.int64),
                 cal_cand=np.array([r[5] for r in rec], dtype=np.float32),
                 in_target=np.array([r[6] for r in rec], dtype=bool),
                 frac_cell=np.array([r[7] for r in rec], dtype=np.float32),
                 cheb_ctr_nm=np.array([r[8] for r in rec], dtype=np.float32),
                 seed=np.uint64(seed), cell=np.uint64(cell), target=np.uint64(tgt),
                 ctr_nm=ctr.astype(np.float64),
                 cal_seed=np.float32(cal_seed_local),
                 cal_seed_global=np.float32(cal_seed_global),
                 n_vox_seed=np.int64(ei - si), half_nm=np.float32(args.half_nm),
                 svmap_coverage=np.float32(cov), svmap_agree=np.float32(agree),
                 gap_v1_stored=np.float32(float(p["gap_nm"][it][np.argmin(p["gap_nm"][it])])),
                 gap_thin_only=np.float32(gap_thin), gap_eroded_only=np.float32(gap_eroded),
                 frac_vox_mapped=np.float32(float(mS.mean())),
                 corridor=np.frombuffer(json.dumps(corridor).encode(), dtype=np.uint8))
        g_old = float(p["gap_nm"][it][np.argmin(p["gap_nm"][it])])
        n_cellobj = sum(1 for r in rec if r[7] > 0.5 and r[0] != tgt)
        print(f"  {key}: sv {len(usv)} map {cov:.0%}/{agree:.0%}  obj {len(rec)+1}  "
              f"gap {g_old:.0f}->{trow[1]:.0f}nm (thin {gap_thin:.0f} erode {gap_eroded:.0f})  "
              f"cal seed {cal_seed_local:.0f} tgt {trow[5]:.0f}  "
              f"corridor bg {corridor['frac_background']:.0%} other {corridor['n_other_objects']}  "
              f"cell-owned extra {n_cellobj}  [dl {t_dl:.0f} rt {t_rt:.0f} edt {t_edt:.0f}]", flush=True)
        done += 1


if __name__ == "__main__":
    main()
