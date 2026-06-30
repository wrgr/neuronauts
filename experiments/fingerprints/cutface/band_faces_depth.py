"""8 nm depth-stack cut-face: identify candidates at 16 nm, sample the face at 8 nm.

Probes whether the headline patch is *representation*-limited rather than
information-limited.  Two changes vs ``site_faces_bands``:

1. **Point the 16 nm candidate identifier at the 8 nm data.**  Candidate
   identification -- the proximity panel, direction cone, geometry, and the
   curseg-based ``is_true`` -- is done on the cached 16 nm (mip1) box, which is
   free.  Only the *face* is resampled from a freshly-fetched 8 nm (mip0) box,
   masked by the same historical-root id (chunkedgraph roots are
   resolution-independent, so the id transfers).
2. **3-section depth stack instead of one mean-projected slab.**  For the query
   and every candidate we take 3 z-sections marching *away from the cut* (into
   the process, never crossing the gap), keeping them as separate channels so
   the trajectory / caliber-taper / organelle-in-depth survives instead of being
   flattened.

Training-free first pass: rank candidates by raw cosine in each band (and
fusions) on the depth-stack faces, exactly like ``v117_artifact_bands`` so the
numbers are directly comparable to the 16 nm / 1-slab baseline.  z resolution is
40 nm at both mips, so the depth stack adds genuine z-structure and the mip0
fetch adds genuine in-plane (8 nm) resolution -- the two simplifications the
flat 16 nm patch threw away.
"""

from __future__ import annotations

import json

import numpy as np

from . import v117_error_relink as v
from . import v117_reconstructed as r
from .v117_artifact_bands import _band_face


def _stack(hi_vol, z_center, dz, n_sec, sid, sigma):
    """n_sec (low, high) band faces from z_center stepping dz, masked by ``sid``.

    Returns (low_stack [n,P,P], high_stack [n,P,P]) or None if any section is
    empty for this fragment (the process ended before n_sec sections)."""
    nz = hi_vol.em.shape[2]
    lows, highs = [], []
    for k in range(n_sec):
        z = int(np.clip(z_center + k * dz, 0, nz - 1))
        bf = _band_face(hi_vol.em, hi_vol.seg, z, z + 1, sid, sigma)
        if bf is None:
            return None
        lows.append(bf[0]); highs.append(bf[1])
    return np.stack(lows), np.stack(highs)


def site_faces_bands_depth(cl, ts, site, *, id_mip=1, hi_mip=0, n_sections=3,
                           radius_nm=2000.0, direction_cone_deg=45.0, min_vox=40,
                           sigma=2.0, require_true=True):
    """Identify the panel on the id_mip box; sample depth-stack faces on hi_mip.

    Mirrors ``site_faces_bands`` identification (scanned-root identity, curseg
    is_true, cone, min size, not-a-split drop) but returns 3-section depth-stack
    band faces sampled from the 8 nm box."""
    id_vol, frag2cur = r.fetch_v117_box(cl, ts, site.pos_main_nm, site.pos_frag_nm, radius_nm, id_mip)
    qa_id, idx_main = v._seg_id_at(id_vol, site.pos_main_nm)
    if qa_id == 0 or qa_id not in frag2cur:
        return None
    q_cur = int(site.root) if getattr(site, "root", None) else frag2cur[qa_id]
    fseg, _ = v._seg_id_at(id_vol, site.pos_frag_nm)

    prox = v._proximity_candidates(id_vol, idx_main, radius_nm, qa_id, min_vox)
    if not prox:
        return None
    vox = np.asarray(id_vol.resolution_nm, float)
    origin = np.asarray(id_vol.origin_vox, float)
    pmain = np.asarray(site.pos_main_nm, float)
    pfrag = np.asarray(site.pos_frag_nm, float)
    tangent = np.asarray(site.tangent_nm, float)
    tn = np.linalg.norm(tangent)
    cone_cos = np.cos(np.deg2rad(direction_cone_deg)) if direction_cone_deg else None
    curseg = getattr(id_vol, "curseg", None)

    def _cur_of(sid, nv):
        if curseg is not None:
            c = int(curseg[int(nv[0]), int(nv[1]), int(nv[2])])
            if c != 0:
                return c
        return int(frag2cur.get(int(sid), 0))

    # surviving candidates (sid, current-root match, nm position) after the cone
    keep = []
    for sid, (nv, _) in prox.items():
        same = _cur_of(sid, nv) == q_cur
        d = (origin + nv + 0.5) * vox - pmain
        dn = float(np.linalg.norm(d))
        if cone_cos is not None and tn > 1e-6 and not same:
            if dn > 1e-6 and abs(float(d @ tangent) / (dn * tn)) < cone_cos:
                continue
        cand_z_nm = (origin[2] + nv[2] + 0.5) * vox[2]
        keep.append((int(sid), same, dn, cand_z_nm))
    if len(keep) < 3:
        return None
    if not any(s for _, s, _, _ in keep):
        if int(fseg) == int(qa_id) or require_true:
            return None

    # --- now sample the faces from the 8 nm box ---
    hi_vol, _ = r.fetch_v117_box(cl, ts, site.pos_main_nm, site.pos_frag_nm, radius_nm, hi_mip)
    # query: start at the main endpoint, march away from the frag side
    z_main = v._z_index(hi_vol, float(pmain[2]))
    dz_main = 1 if pmain[2] >= pfrag[2] else -1
    q = _stack(hi_vol, z_main, dz_main, n_sections, qa_id, sigma)
    if q is None:
        return None

    lows, highs, is_true, gdist = [], [], [], []
    for sid, same, dn, cand_z_nm in keep:
        z_cand = v._z_index(hi_vol, float(cand_z_nm))
        dz = 1 if cand_z_nm >= pmain[2] else -1      # march away from the query
        st = _stack(hi_vol, z_cand, dz, n_sections, sid, sigma)
        if st is None:
            continue
        lows.append(st[0]); highs.append(st[1]); is_true.append(same); gdist.append(dn)
    if len(lows) < 3 or (require_true and not any(is_true)):
        return None
    return {"q_low": q[0], "q_high": q[1],
            "low": np.stack(lows), "high": np.stack(highs),
            "is_true": np.array(is_true), "geom_dist": np.array(gdist)}


def _cos_dist_stack(q_stack, bank_stack):
    """Cosine distance between a query depth-stack and each candidate stack.

    The stacks are aligned section-by-section (both march away from the cut), so
    we concatenate the sections into one vector -- preserving depth order -- and
    take 1 - cosine."""
    qf = v._flatnorm(q_stack.reshape(-1))
    out = np.empty(len(bank_stack))
    for i in range(len(bank_stack)):
        out[i] = 1.0 - float(qf @ v._flatnorm(bank_stack[i].reshape(-1)))
    return out


def evaluate_depth(cl, ts, roots, *, id_mip=1, hi_mip=0, n_sections=3, radius_nm=2000.0,
                   direction_cone_deg=45.0, max_sites=10, sigma=2.0, verbose=True):
    methods = ["geom", "bio", "art", "bio+art", "geom+bio", "geom+art", "geom+bio+art"]
    ranks = {m: [] for m in methods}
    recov = {m: [] for m in ("bio", "art", "geom+bio", "geom+art", "geom+bio+art")}
    nsites = 0
    for n, rt in enumerate(roots):
        try:
            sites = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=radius_nm, max_sites=max_sites)
        except Exception:
            continue
        for s in sites:
            try:
                f = site_faces_bands_depth(cl, ts, s, id_mip=id_mip, hi_mip=hi_mip,
                                           n_sections=n_sections, radius_nm=radius_nm,
                                           direction_cone_deg=direction_cone_deg, sigma=sigma)
            except Exception:
                f = None
            if f is None:
                continue
            it = f["is_true"]
            nsites += 1
            d_geom = f["geom_dist"].astype(float)
            d_bio = _cos_dist_stack(f["q_low"], f["low"])
            d_art = _cos_dist_stack(f["q_high"], f["high"])
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
        if verbose and nsites:
            print(f"  eval: {n + 1}/{len(roots)} neurons, {nsites} sites", flush=True)
    return ranks, recov, nsites


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=200)
    ap.add_argument("--neurons", type=int, default=20)
    ap.add_argument("--id-mip", type=int, default=1, help="mip for candidate identification")
    ap.add_argument("--hi-mip", type=int, default=0, help="mip for face sampling (0 = 8 nm)")
    ap.add_argument("--n-sections", type=int, default=3)
    ap.add_argument("--radius-nm", type=float, default=2000.0)
    ap.add_argument("--direction-cone-deg", type=float, default=45.0)
    ap.add_argument("--max-sites", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--out", default="experiments/fingerprints/band_depth_metrics.json")
    args = ap.parse_args()

    cl = v._client()
    ts = cl.chunkedgraph.get_oldest_timestamp()
    roots, _ = v.find_split_neurons(cl, n_scan=args.n_scan)
    roots = roots[:args.neurons]
    print(f"[depth] {len(roots)} neurons, id_mip={args.id_mip} hi_mip={args.hi_mip} "
          f"n_sections={args.n_sections}", flush=True)

    ranks, recov, nsites = evaluate_depth(
        cl, ts, roots, id_mip=args.id_mip, hi_mip=args.hi_mip, n_sections=args.n_sections,
        radius_nm=args.radius_nm, direction_cone_deg=args.direction_cone_deg,
        max_sites=args.max_sites, sigma=args.sigma)

    def top1(m):
        a = np.asarray(ranks[m])
        return float(np.mean(a == 0)) if len(a) else 0.0

    print(f"\n8nm depth-stack ({args.n_sections} sec), {nsites} sites (training-free cosine):")
    out = {"id_mip": args.id_mip, "hi_mip": args.hi_mip, "n_sections": args.n_sections,
           "n_sites": nsites, "top1": {}, "geom_miss_recovery": {}}
    for m in ranks:
        out["top1"][m] = top1(m)
        print(f"  {m:14s} top-1 {top1(m):.3f}")
    for m, lst in recov.items():
        out["geom_miss_recovery"][m] = float(np.mean(lst)) if lst else 0.0
    print("  geom-miss recovery:", {k: round(val, 3) for k, val in out["geom_miss_recovery"].items()})
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
