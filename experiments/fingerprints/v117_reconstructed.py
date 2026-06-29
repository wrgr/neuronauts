"""Disambiguate v117 errors using the *reconstructed v117 segmentation*.

This is the faithful version of the real-error experiment.  Instead of masking
from the flat MICrONS seg (an intermediate snapshot that already has many merges
baked in), we rebuild the segmentation **as it was at the oldest / v117-era
timestamp**, where the false-split fragments are genuinely separate:

1. Fetch the supervoxel (watershed) layer for a box (graphene, ``agglomerate=
   False``) -- this is version-independent.
2. Map each supervoxel to its root **at the v117-era timestamp** -> paint the box
   with v117 fragment ids (the "question": the errored, split state).
3. Map each supervoxel to its **current** root -> the answer key: two v117
   fragments that share a current root were merged by proofreading and *should*
   connect (the v14XX merges).

The test at each interface: query = a v117 fragment's cut-face; candidate panel =
the nearby v117 fragments (the real merge candidates a v117-state tool would
consider); a candidate is a TRUE partner iff it shares the query fragment's
current root.  Rank the (best) true partner by cut-face hash similarity.

Requires a CAVE token in env ``token`` (written to the cloudvolume secret so
graphene reads authenticate).  EM + v117 seg boxes are cached to disk.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from .fingerprint_break_resolution import Volume, PATCH
from . import v117_error_relink as v
from .learned_cutface_encoder import make_embed_fn, load_encoder


_SV_CV = None


def _ensure_secret():
    """Write the CAVE token to the cloudvolume secret so graphene reads auth."""
    tok = os.environ.get("token") or os.environ.get("CAVE_TOKEN")
    path = os.path.expanduser("~/.cloudvolume/secrets/cave-secret.json")
    if tok and not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"token": tok}, f)


def _sv_volume(cl, mip=1):
    global _SV_CV
    if _SV_CV is None:
        from cloudvolume import CloudVolume
        _ensure_secret()
        src = cl.info.segmentation_source()
        _SV_CV = CloudVolume(src, use_https=True, progress=False, fill_missing=True,
                             agglomerate=False, mip=mip)
    return _SV_CV


V117_CACHE_DIR = os.environ.get("V117_BOX_CACHE", "data/v117_box_cache")


def fetch_v117_box(cl, ts, pos_a, pos_b, margin_nm, mip=1):
    """Return (Volume with v117-painted seg, frag2cur dict) for the box.

    ``frag2cur`` maps each v117 fragment id -> its (majority) current root.
    """
    pts = np.asarray([pos_a, pos_b], float)
    lo = pts.min(0) - margin_nm
    hi = pts.max(0) + margin_nm
    bbox = (tuple(lo.tolist()), tuple(hi.tolist()))

    path = None
    if V117_CACHE_DIR:
        path = os.path.join(V117_CACHE_DIR, f"v117_{v._box_key(bbox, mip)}.npz")
        if os.path.exists(path):
            try:
                z = np.load(path)
                vol = Volume(em=z["em"], seg=z["seg"],
                             resolution_nm=tuple(int(x) for x in z["res"]),
                             origin_vox=tuple(int(x) for x in z["origin"]))
                frag2cur = {int(k): int(val) for k, val in zip(z["frag"], z["cur"])}
                return vol, frag2cur
            except Exception:
                pass

    em = v._fetch_em(bbox, mip=mip)
    cv = _sv_volume(cl, mip=mip)
    vox = np.asarray(em.voxel_size_nm, float)
    x0, y0, z0 = (int(bbox[0][d] / vox[d]) for d in range(3))
    x1, y1, z1 = (int(bbox[1][d] / vox[d]) for d in range(3))
    sv = np.squeeze(np.asarray(cv[x0:x1, y0:y1, z0:z1])).astype(np.uint64)

    u, inv = np.unique(sv, return_inverse=True)
    svids = u[u > 0]
    cg = cl.chunkedgraph
    hist = np.asarray(cg.get_roots(svids.tolist(), timestamp=ts)) if len(svids) else np.array([])
    cur = np.asarray(cg.get_roots(svids.tolist())) if len(svids) else np.array([])
    sv2hist = {int(s): int(h) for s, h in zip(svids.tolist(), hist.tolist())}

    umap = np.array([sv2hist.get(int(x), 0) if x != 0 else 0 for x in u], dtype=np.uint64)
    v117seg = umap[inv].reshape(sv.shape)

    # frag (v117 root) -> majority current root
    frag2cur = {}
    if len(svids):
        from collections import Counter, defaultdict
        buckets = defaultdict(list)
        for h, c in zip(hist.tolist(), cur.tolist()):
            if h != 0 and c != 0:
                buckets[int(h)].append(int(c))
        frag2cur = {h: Counter(cs).most_common(1)[0][0] for h, cs in buckets.items()}

    vol = Volume(em=em.data.astype(np.uint8), seg=v117seg,
                 resolution_nm=tuple(int(x) for x in vox), origin_vox=(x0, y0, z0))

    if path:
        os.makedirs(V117_CACHE_DIR, exist_ok=True)
        tmp = path + f".tmp{os.getpid()}.npz"
        try:
            np.savez_compressed(tmp, em=vol.em, seg=vol.seg,
                                res=np.asarray(vol.resolution_nm),
                                origin=np.asarray(vol.origin_vox),
                                frag=np.asarray(list(frag2cur.keys()), dtype=np.uint64),
                                cur=np.asarray(list(frag2cur.values()), dtype=np.uint64))
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
    return vol, frag2cur


def site_faces_v117(cl, ts, site, *, mip=1, slab=3, radius_nm=2000.0,
                    direction_cone_deg=45.0, min_vox=40):
    """Query face + candidate faces on the v117-painted box; true = shared current root."""
    vol, frag2cur = fetch_v117_box(cl, ts, site.pos_main_nm, site.pos_frag_nm, radius_nm, mip)
    nz = vol.em.shape[2]
    qa_id, idx_main = v._seg_id_at(vol, site.pos_main_nm)
    if qa_id == 0 or qa_id not in frag2cur:
        return None
    q_cur = frag2cur[qa_id]

    za = max(min(v._z_index(vol, site.pos_main_nm[2]), nz - slab), 0)
    q = v._patch_from_slab(vol.em, vol.seg, za, za + slab, qa_id)
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

    cand_ids, patches, is_true, geom_dist = [], [], [], []
    for sid, (nv, _) in prox.items():
        same_neuron = frag2cur.get(sid) == q_cur
        d = (origin + nv + 0.5) * vox - pmain
        dn = float(np.linalg.norm(d))
        if cone_cos is not None and tn > 1e-6 and not same_neuron:
            if dn > 1e-6 and abs(float(d @ tangent) / (dn * tn)) < cone_cos:
                continue
        zc = max(min(int(nv[2]) - slab // 2, nz - slab), 0)
        p = v._patch_from_slab(vol.em, vol.seg, zc, zc + slab, sid)
        if p is None:
            continue
        cand_ids.append(sid)
        patches.append(p)
        is_true.append(same_neuron)
        geom_dist.append(dn)   # nm distance from query endpoint -> geometry baseline
    if not any(is_true) or len(cand_ids) < 3:
        return None
    return {"query": q, "patches": np.stack(patches), "is_true": np.array(is_true),
            "geom_dist": np.array(geom_dist)}


def _best_rank(d, is_true):
    order = np.argsort(d)
    ranks = np.empty(len(d), int)
    ranks[order] = np.arange(len(d))
    return int(ranks[is_true].min())   # best (smallest) rank among true partners


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-9)


def evaluate(cl, ts, roots, embedders, *, mip=1, radius_nm=2000.0,
             direction_cone_deg=45.0, max_sites=6, fuse_with="real", verbose=True):
    """Rank true partner by: geometry (distance), raw/learned hash, and a fusion.

    ``fuse_with`` names the embedder whose hash distance is combined (equal
    weight, within-panel z-scored) with the geometric distance for the fused
    ranker -- the deployment-relevant "geometry + hash" question.
    """
    methods = ["geom", "raw", *embedders, "fused"]
    ranks = {m: [] for m in methods}
    geom_hit, fused_hit = [], []   # paired on the geom-fails (hard) subset
    ncand, ntrue = [], []
    for n, rt in enumerate(roots):
        try:
            sites = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=radius_nm, max_sites=max_sites)
        except Exception:
            continue
        for s in sites:
            try:
                f = site_faces_v117(cl, ts, s, mip=mip, radius_nm=radius_nm,
                                    direction_cone_deg=direction_cone_deg)
            except Exception:
                f = None
            if f is None:
                continue
            P, it, gd = f["patches"], f["is_true"], f["geom_dist"]
            ncand.append(len(it))
            ntrue.append(int(it.sum()))

            d_geom = gd.astype(float)
            ranks["geom"].append(_best_rank(d_geom, it))

            d_raw = 1.0 - np.stack([v._flatnorm(P[i]) for i in range(len(P))]) @ v._flatnorm(f["query"])
            ranks["raw"].append(_best_rank(d_raw, it))

            d_emb = {}
            for name, emb in embedders.items():
                qe = np.asarray(emb(f["query"][None]))[0]
                ce = np.asarray(emb(P))
                qe = qe / (np.linalg.norm(qe) + 1e-9)
                ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-9)
                d = 1.0 - ce @ qe
                d_emb[name] = d
                ranks[name].append(_best_rank(d, it))

            d_hash = d_emb.get(fuse_with, d_raw)
            d_fused = _z(d_geom) + _z(d_hash)        # equal-weight geometry + hash
            ranks["fused"].append(_best_rank(d_fused, it))

            # paired hard subset: sites where geometry alone misses top-1
            if _best_rank(d_geom, it) != 0:
                geom_hit.append(0)
                fused_hit.append(int(_best_rank(d_fused, it) == 0))
        if verbose and ranks["geom"]:
            print(f"  eval: {n + 1}/{len(roots)} neurons, {len(ranks['geom'])} sites")
    return ranks, ncand, ntrue, (geom_hit, fused_hit)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=150)
    ap.add_argument("--neurons", type=int, default=40)
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--radius-nm", type=float, default=2000.0)
    ap.add_argument("--direction-cone-deg", type=float, default=45.0)
    ap.add_argument("--max-sites", type=int, default=6)
    ap.add_argument("--planar", default="experiments/fingerprints/cutface_encoder.pt")
    ap.add_argument("--real", default="experiments/fingerprints/cutface_encoder_real.pt")
    ap.add_argument("--out", default="experiments/fingerprints/v117_reconstructed_metrics.json")
    args = ap.parse_args()

    cl = v._client()
    cg = cl.chunkedgraph
    ts = cg.get_oldest_timestamp()
    print(f"[cave] scanning {args.n_scan} somas; v117-era ts = {ts}")
    roots, _ = v.find_split_neurons(cl, n_scan=args.n_scan)
    roots = roots[:args.neurons]
    print(f"[cave] evaluating {len(roots)} split neurons on reconstructed v117 seg")

    embedders = {}
    if os.path.exists(args.planar):
        embedders["planar"] = make_embed_fn(load_encoder(args.planar))
    if os.path.exists(args.real):
        embedders["real"] = make_embed_fn(load_encoder(args.real))

    fuse_with = "real" if "real" in embedders else ("planar" if "planar" in embedders else "raw")
    ranks, ncand, ntrue, (geom_hit, fused_hit) = evaluate(
        cl, ts, roots, embedders, mip=args.mip, radius_nm=args.radius_nm,
        direction_cone_deg=args.direction_cone_deg, max_sites=args.max_sites,
        fuse_with=fuse_with)
    n = len(ranks["geom"])
    if n == 0:
        print("no scorable sites"); return
    summ = {"n_sites": n,
            "mean_candidates": float(np.mean(ncand)),
            "mean_true_partners": float(np.mean(ntrue)),
            "chance_top1": float(np.mean(1.0 / np.asarray(ncand))),
            "fused_uses": fuse_with}
    for name, rs in ranks.items():
        rs = np.asarray(rs)
        summ[name] = {"top1": float((rs == 0).mean()), "mrr": float(np.mean(1.0 / (rs + 1.0)))}
    if geom_hit:
        summ["geom_miss_recovered_by_fusion"] = {
            "n_geom_misses": len(geom_hit),
            "fused_top1_on_those": float(np.mean(fused_hit))}
    print(f"\nReconstructed-v117 re-linking: {n} sites, mean {summ['mean_candidates']:.1f} "
          f"candidates ({summ['mean_true_partners']:.1f} true), chance top-1 {summ['chance_top1']:.3f}")
    for name in ("geom", "raw", "planar", "real", "fused"):
        if name in summ and isinstance(summ[name], dict) and "top1" in summ[name]:
            tag = f" (=geom+{fuse_with})" if name == "fused" else ""
            print(f"  {name:7s} top-1 / MRR: {summ[name]['top1']:.3f} / {summ[name]['mrr']:.3f}{tag}")
    if "geom_miss_recovered_by_fusion" in summ:
        g = summ["geom_miss_recovered_by_fusion"]
        print(f"  fusion recovers {g['fused_top1_on_those']:.3f} of the {g['n_geom_misses']} "
              f"sites geometry-alone gets wrong")

    with open(args.out, "w") as f:
        json.dump({"radius_nm": args.radius_nm, "direction_cone_deg": args.direction_cone_deg,
                   "v117_ts": str(ts), "summary": summ}, f, indent=2)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
