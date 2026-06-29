"""Artifact vs biological bands: where does the matchable signal live?

The cut-face hash so far uses a smooth, mean-projected patch -- a *biological-
structure* signal (shape + organelle layout).  This experiment splits each face
into two spatial-frequency bands and asks which one (or their fusion) re-links
real v117 split partners, and whether the artifact band is *complementary* to
geometry:

* **bio** (low-pass)  = gaussian-blurred masked patch  -> smooth structure / shape.
* **art** (high-pass) = patch - blur                   -> texture / membrane-edge
  sharpness / imaging-and-processing grain (the "artifact" band).

Training-free first pass: rank candidates by raw cosine in each band and in
fusions (bio+art, geom+bio, geom+art, geom+bio+art), on the reconstructed-v117
sites.  Reuses the cached v117 boxes, so no refetching.  If the art band adds
signal over geom/bio, the next step is to *learn* per-band encoders.

Faces are masked per v117 fragment (disjoint ids), so the two sides of a gap
never share voxels -- no leakage across the break.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from .fingerprint_break_resolution import PATCH
from . import v117_error_relink as v
from . import v117_reconstructed as r


def _band_face(em, seg, z_lo, z_hi, seg_id, sigma=2.0):
    """Return (low, high) PATCH x PATCH band images for one fragment face, or None."""
    from scipy.ndimage import gaussian_filter
    sub = em[:, :, z_lo:z_hi].astype(np.float32)
    mask = seg[:, :, z_lo:z_hi] == seg_id
    c2 = mask.sum(axis=2)
    if c2.sum() == 0:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        proj = np.where(c2 > 0, (sub * mask).sum(axis=2) / c2, 0.0)
    xs, ys = np.nonzero(c2 > 0)
    ci, cj = int(round(xs.mean())), int(round(ys.mean()))
    h = PATCH // 2
    out = np.zeros((PATCH, PATCH), np.float32)
    fp = np.zeros((PATCH, PATCH), bool)
    xi0, xi1 = max(ci - h, 0), min(ci + h, proj.shape[0])
    yi0, yi1 = max(cj - h, 0), min(cj + h, proj.shape[1])
    px0, py0 = xi0 - (ci - h), yi0 - (cj - h)
    out[px0:px0 + (xi1 - xi0), py0:py0 + (yi1 - yi0)] = proj[xi0:xi1, yi0:yi1]
    fp[px0:px0 + (xi1 - xi0), py0:py0 + (yi1 - yi0)] = (c2[xi0:xi1, yi0:yi1] > 0)
    low = gaussian_filter(out, sigma) * fp        # smooth structure inside footprint
    high = (out - gaussian_filter(out, sigma)) * fp  # texture / artifact band
    return low, high


def site_faces_bands(cl, ts, site, *, mip=1, slab=2, radius_nm=2000.0,
                     direction_cone_deg=45.0, min_vox=40, sigma=2.0,
                     require_true=True):
    """Query + candidate band faces on the v117-painted box; true = shared current root.

    ``require_true=True`` (default) drops sites where the true partner is absent
    from the panel -- correct for the correction-given-candidates eval, but it
    conditions on the candidate generator succeeding.  Pass ``require_true=False``
    to KEEP partner-absent sites (all-false labels) for an honest abstention eval
    over the full population."""
    vol, frag2cur = r.fetch_v117_box(cl, ts, site.pos_main_nm, site.pos_frag_nm, radius_nm, mip)
    nz = vol.em.shape[2]
    qa_id, idx_main = v._seg_id_at(vol, site.pos_main_nm)
    if qa_id == 0 or qa_id not in frag2cur:
        return None
    q_cur = frag2cur[qa_id]

    za = max(min(v._z_index(vol, site.pos_main_nm[2]), nz - slab), 0)
    q = _band_face(vol.em, vol.seg, za, za + slab, qa_id, sigma)
    if q is None:
        return None

    prox = v._proximity_candidates(vol, idx_main, radius_nm, qa_id, min_vox)
    if not prox:
        return None
    vox = np.asarray(vol.resolution_nm, float)
    origin = np.asarray(vol.origin_vox, float)
    pmain = np.asarray(site.pos_main_nm, float)
    tangent = np.asarray(site.tangent_nm, float)
    tn = np.linalg.norm(tangent)
    cone_cos = np.cos(np.deg2rad(direction_cone_deg)) if direction_cone_deg else None

    lows, highs, is_true, gdist = [], [], [], []
    for sid, (nv, _) in prox.items():
        same = frag2cur.get(sid) == q_cur
        d = (origin + nv + 0.5) * vox - pmain
        dn = float(np.linalg.norm(d))
        if cone_cos is not None and tn > 1e-6 and not same:
            if dn > 1e-6 and abs(float(d @ tangent) / (dn * tn)) < cone_cos:
                continue
        zc = max(min(int(nv[2]) - slab // 2, nz - slab), 0)
        bf = _band_face(vol.em, vol.seg, zc, zc + slab, sid, sigma)
        if bf is None:
            continue
        lows.append(bf[0]); highs.append(bf[1]); is_true.append(same); gdist.append(dn)
    if len(lows) < 3 or (require_true and not any(is_true)):
        return None
    return {"q_low": q[0], "q_high": q[1],
            "low": np.stack(lows), "high": np.stack(highs),
            "is_true": np.array(is_true), "geom_dist": np.array(gdist)}


def _cos_dist(query, bank):
    qf = v._flatnorm(query)
    bf = np.stack([v._flatnorm(bank[i]) for i in range(len(bank))])
    return 1.0 - bf @ qf


def evaluate_bands(cl, ts, roots, *, mip=1, radius_nm=2000.0, direction_cone_deg=45.0,
                   max_sites=10, sigma=2.0, verbose=True):
    methods = ["geom", "bio", "art", "bio+art", "geom+bio", "geom+art", "geom+bio+art"]
    ranks = {m: [] for m in methods}
    recov = {m: [] for m in ("bio", "art", "geom+bio", "geom+art", "geom+bio+art")}
    ncand, ntrue = [], []
    for n, rt in enumerate(roots):
        try:
            sites = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=radius_nm, max_sites=max_sites)
        except Exception:
            continue
        for s in sites:
            try:
                f = site_faces_bands(cl, ts, s, mip=mip, radius_nm=radius_nm,
                                     direction_cone_deg=direction_cone_deg, sigma=sigma)
            except Exception:
                f = None
            if f is None:
                continue
            it = f["is_true"]
            ncand.append(len(it)); ntrue.append(int(it.sum()))
            d_geom = f["geom_dist"].astype(float)
            d_bio = _cos_dist(f["q_low"], f["low"])
            d_art = _cos_dist(f["q_high"], f["high"])
            zg, zb, za_ = r._z(d_geom), r._z(d_bio), r._z(d_art)
            dd = {"geom": d_geom, "bio": d_bio, "art": d_art,
                  "bio+art": zb + za_, "geom+bio": zg + zb,
                  "geom+art": zg + za_, "geom+bio+art": zg + zb + za_}
            geom_miss = r._best_rank(d_geom, it) != 0
            for m, d in dd.items():
                rk = r._best_rank(d, it)
                ranks[m].append(rk)
                if geom_miss and m in recov:
                    recov[m].append(int(rk == 0))
        if verbose and ranks["geom"]:
            print(f"  eval: {n + 1}/{len(roots)} neurons, {len(ranks['geom'])} sites")
    return ranks, ncand, ntrue, recov


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=200)
    ap.add_argument("--neurons", type=int, default=50)
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--radius-nm", type=float, default=2000.0)
    ap.add_argument("--direction-cone-deg", type=float, default=45.0)
    ap.add_argument("--max-sites", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=2.0, help="gaussian sigma (px) for the band split")
    ap.add_argument("--out", default="experiments/fingerprints/v117_artifact_bands_metrics.json")
    args = ap.parse_args()

    cl = v._client()
    ts = cl.chunkedgraph.get_oldest_timestamp()
    print(f"[cave] scanning {args.n_scan} somas; v117-era ts = {ts}")
    roots, _ = v.find_split_neurons(cl, n_scan=args.n_scan)
    roots = roots[:args.neurons]
    print(f"[cave] {len(roots)} split neurons; band split sigma={args.sigma}px")

    ranks, ncand, ntrue, recov = evaluate_bands(
        cl, ts, roots, mip=args.mip, radius_nm=args.radius_nm,
        direction_cone_deg=args.direction_cone_deg, max_sites=args.max_sites, sigma=args.sigma)
    n = len(ranks["geom"])
    if n == 0:
        print("no scorable sites"); return
    summ = {"n_sites": n, "mean_candidates": float(np.mean(ncand)),
            "mean_true_partners": float(np.mean(ntrue)),
            "chance_top1": float(np.mean(1.0 / np.asarray(ncand)))}
    for m, rs in ranks.items():
        rs = np.asarray(rs)
        summ[m] = {"top1": float((rs == 0).mean()), "mrr": float(np.mean(1.0 / (rs + 1.0)))}
    summ["geom_miss_recovery"] = {m: (float(np.mean(h)) if h else None) for m, h in recov.items()}
    summ["n_geom_misses"] = int(sum(1 for _ in recov["geom+bio"]))

    print(f"\nArtifact-band re-linking: {n} sites, mean {summ['mean_candidates']:.1f} candidates, "
          f"chance top-1 {summ['chance_top1']:.3f}  (raw cosine, no training)")
    for m in ("geom", "bio", "art", "bio+art", "geom+bio", "geom+art", "geom+bio+art"):
        print(f"  {m:14s} top-1 / MRR: {summ[m]['top1']:.3f} / {summ[m]['mrr']:.3f}")
    print(f"  of {summ['n_geom_misses']} geom-misses, recovered top-1 by:")
    for m, val in summ["geom_miss_recovery"].items():
        if val is not None:
            print(f"      {m:14s}: {val:.3f}")

    with open(args.out, "w") as f:
        json.dump({"radius_nm": args.radius_nm, "sigma": args.sigma,
                   "v117_ts": str(ts), "summary": summ}, f, indent=2)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
