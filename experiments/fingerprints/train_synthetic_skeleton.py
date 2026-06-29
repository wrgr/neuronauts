"""Scale the texture hash with SYNTHETIC same-object pairs mined along skeletons.

Real v117 error sites are scarce (~tens-hundreds), so the learned hash was
undertrained.  But same-neuron cross-section pairs are essentially unlimited:
within any fetched box a fragment appears across several z-sections, so two
z-separated cross-sections of *one* fragment form a synthetic positive (matched
to the gap scale), and the other fragments in the box are hard negatives.  This
mines thousands of such pairs almost for free (reuses the cached boxes), trains
the bio/art band encoders on them, then **fine-tunes on the real v117 error
pairs** and evaluates on held-out real sites (fused with geometry).

Design choices that make synthetic transfer to real breaks:
- positives separated by a *synthetic gap* (skip sections) -> mimic a break, not
  a clean continuation;
- separation matched to the real gap scale (a few sections), not dist~0;
- negatives = the other fragments in the same box (the proximity confusers);
- geometry is NOT used in synthetic training (separation is arbitrary) -- it is
  only fused back in at real-site evaluation.

Train / test neurons are disjoint.  Mining and eval reuse the v117 box cache.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .fingerprint_break_resolution import PATCH
from . import v117_error_relink as v
from . import v117_reconstructed as r
from .v117_artifact_bands import _band_face
from .train_real_cutface import finetune
from .train_band_encoders import evaluate_bands_learned
from .learned_cutface_encoder import make_embed_fn


def _fragment_z_extents(seg, min_vox_per_section=30):
    """Map fragment id -> sorted list of z indices where it has enough voxels."""
    out = {}
    nz = seg.shape[2]
    for z in range(nz):
        ids, counts = np.unique(seg[:, :, z], return_counts=True)
        for i, c in zip(ids.tolist(), counts.tolist()):
            if i != 0 and c >= min_vox_per_section:
                out.setdefault(int(i), []).append(z)
    return out


def mine_box(vol, *, slab=2, gap_sections=2, sigma=2.0, pairs_per_fragment=2,
             n_distractors=8, min_vox_per_section=30, max_frags=25, seed=0):
    """Mine synthetic same-fragment band pairs (+hard negatives) from one box.

    O(F) in fragments: each fragment's band face is computed once (at its mid z)
    for use as a negative; positives add two band faces per sampled pair.
    """
    rng = np.random.default_rng(seed)
    ext = _fragment_z_extents(vol.seg, min_vox_per_section)
    big = [f for f, zs in ext.items() if len(zs) >= 2 * slab + gap_sections + 1]
    if len(big) < 4:
        return []
    if len(big) > max_frags:
        big = [int(x) for x in rng.choice(big, max_frags, replace=False)]

    # precompute one band face per fragment (mid z) -> reusable hard negatives
    face_at = {}
    nz = vol.em.shape[2]
    for f in big:
        zs = ext[f]
        zc = min(zs[len(zs) // 2], nz - slab)
        bf = _band_face(vol.em, vol.seg, zc, zc + slab, f, sigma)
        if bf is not None:
            face_at[f] = bf
    valid = [f for f in big if f in face_at]
    if len(valid) < 4:
        return []

    samples = []
    for f in valid:
        zs = ext[f]
        z0lo, z0hi = min(zs), max(zs)
        for _ in range(pairs_per_fragment):
            hi = max(z0lo + 1, z0hi - slab - gap_sections - slab + 1)
            za = int(rng.integers(z0lo, hi))
            zb = za + slab + gap_sections           # synthetic gap between the two faces
            if zb + slab > nz:
                continue
            fa = _band_face(vol.em, vol.seg, za, za + slab, f, sigma)
            fb = _band_face(vol.em, vol.seg, zb, zb + slab, f, sigma)
            if fa is None or fb is None:
                continue
            negs = [g for g in valid if g != f]
            idx = rng.choice(len(negs), size=n_distractors, replace=len(negs) < n_distractors)
            nl = np.stack([face_at[negs[i]][0] for i in idx])
            nh = np.stack([face_at[negs[i]][1] for i in idx])
            samples.append((fa[0], fb[0], nl, fa[1], fb[1], nh))
    return samples


def mine_synthetic(cl, ts, roots, *, mip=0, radius_nm=2000.0, max_sites=6, slab=2,
                   gap_sections=2, sigma=2.0, target_pairs=3000, verbose=True):
    lo_a, lo_p, lo_d, hi_a, hi_p, hi_d = [], [], [], [], [], []
    for n, rt in enumerate(roots):
        try:
            sites = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=radius_nm, max_sites=max_sites)
        except Exception:
            continue
        for s in sites:
            try:
                vol, _ = r.fetch_v117_box(cl, ts, s.pos_main_nm, s.pos_frag_nm, radius_nm, mip)
                got = mine_box(vol, slab=slab, gap_sections=gap_sections, sigma=sigma,
                               seed=int(rt) % 2**31)
            except Exception:
                got = []
            for (la, lp, ld, ha, hp, hd) in got:
                lo_a.append(la); lo_p.append(lp); lo_d.append(ld)
                hi_a.append(ha); hi_p.append(hp); hi_d.append(hd)
        if verbose and (n % 5 == 0 or n == len(roots) - 1):
            print(f"  mine: {n + 1}/{len(roots)} neurons, {len(lo_a)} synthetic pairs")
        if len(lo_a) >= target_pairs:
            print(f"  reached target {target_pairs} pairs")
            break
    if not lo_a:
        raise RuntimeError("no synthetic pairs mined")
    st = lambda L: np.stack(L).astype(np.float32)
    return {"lo_a": st(lo_a), "lo_p": st(lo_p), "lo_d": st(lo_d),
            "hi_a": st(hi_a), "hi_p": st(hi_p), "hi_d": st(hi_d)}


def mine_from_cache(cache_dir, exclude_keys, *, mip=1, target_pairs=3000, slab=2,
                    gap_sections=2, sigma=2.0, verbose=True):
    """Mine synthetic pairs directly from cached v117 box npz files -- no CAVE.

    ``exclude_keys`` are box keys (from test-neuron sites) to skip, so training
    boxes are disjoint from the evaluation boxes.
    """
    import glob
    from .fingerprint_break_resolution import Volume
    want_res = 8 if mip == 0 else 16
    files = sorted(glob.glob(os.path.join(cache_dir, "v117_*.npz")))
    lo_a, lo_p, lo_d, hi_a, hi_p, hi_d = [], [], [], [], [], []
    used = 0
    for fp in files:
        key = os.path.basename(fp)[len("v117_"):-len(".npz")]
        if key in exclude_keys:
            continue
        try:
            z = np.load(fp)
            res = tuple(int(x) for x in z["res"])
            if res[0] != want_res:
                continue
            vol = Volume(em=z["em"], seg=z["seg"], resolution_nm=res,
                         origin_vox=tuple(int(x) for x in z["origin"]))
        except Exception:
            continue
        got = mine_box(vol, slab=slab, gap_sections=gap_sections, sigma=sigma,
                       seed=int(key, 16) % 2**31)
        for (la, lp, ld, ha, hp, hd) in got:
            lo_a.append(la); lo_p.append(lp); lo_d.append(ld)
            hi_a.append(ha); hi_p.append(hp); hi_d.append(hd)
        used += 1
        if verbose and used % 25 == 0:
            print(f"  mine: {used} boxes, {len(lo_a)} pairs", flush=True)
        if len(lo_a) >= target_pairs:
            break
    if not lo_a:
        raise RuntimeError("no synthetic pairs mined from cache")
    st = lambda L: np.stack(L).astype(np.float32)
    return {"lo_a": st(lo_a), "lo_p": st(lo_p), "lo_d": st(lo_d),
            "hi_a": st(hi_a), "hi_p": st(hi_p), "hi_d": st(hi_d)}


def main():
    import argparse
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=350)
    ap.add_argument("--train-neurons", type=int, default=50)
    ap.add_argument("--test-neurons", type=int, default=30)
    ap.add_argument("--mip", type=int, default=0)
    ap.add_argument("--radius-nm", type=float, default=1500.0)
    ap.add_argument("--max-sites", type=int, default=6)
    ap.add_argument("--target-pairs", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=80, help="epoch CAP (val early-stop selects)")
    ap.add_argument("--resume", action="store_true", help="warm-resume from out-bio/out-art checkpoints")
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--finetune-real", action="store_true",
                    help="after synthetic pretrain, fine-tune on real v117 pairs")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out-bio", default="experiments/fingerprints/cutface_bio_synth.pt")
    ap.add_argument("--out-art", default="experiments/fingerprints/cutface_art_synth.pt")
    ap.add_argument("--metrics", default="experiments/fingerprints/synth_band_metrics.json")
    args = ap.parse_args()

    cl = v._client()
    ts = cl.chunkedgraph.get_oldest_timestamp()
    roots, _ = v.find_split_neurons(cl, n_scan=args.n_scan)
    test_roots = roots[:args.test_neurons]
    print(f"[eval set] {len(test_roots)} test neurons", flush=True)

    # box keys of the test neurons' sites -> excluded from cache mining (no leak)
    box_cache = os.environ.get("V117_BOX_CACHE", "data/v117_box_cache")
    exclude = set()
    for rt in test_roots:
        try:
            for s in v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=args.radius_nm, max_sites=args.max_sites):
                pts = np.asarray([s.pos_main_nm, s.pos_frag_nm], float)
                lo = pts.min(0) - args.radius_nm
                hi = pts.max(0) + args.radius_nm
                exclude.add(v._box_key((tuple(lo.tolist()), tuple(hi.tolist())), args.mip))
        except Exception:
            continue
    print(f"[exclude] {len(exclude)} test box keys held out of mining", flush=True)

    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache); data = {k: z[k] for k in z.files}
        print(f"[cache] loaded {len(data['lo_a'])} synthetic pairs", flush=True)
    else:
        print(f"[mine] from {box_cache} (target {args.target_pairs}, no CAVE) ...", flush=True)
        data = mine_from_cache(box_cache, exclude, mip=args.mip, target_pairs=args.target_pairs,
                               sigma=args.sigma)
        if args.cache:
            np.savez(args.cache, **data); print(f"[cache] wrote {len(data['lo_a'])} pairs", flush=True)

    # epochs is now a CAP; finetune early-stops on val top-1 and writes the best
    # (resumable, with optimizer state) to ckpt_path. Resume later via --resume.
    print(f"[pretrain] bio on {len(data['lo_a'])} synthetic pairs (cap {args.epochs}) ...", flush=True)
    bio, hb = finetune(data["lo_a"], data["lo_p"], data["lo_d"],
                       init_ckpt=(args.out_bio if args.resume else None),
                       epochs=args.epochs, ckpt_path=args.out_bio)
    print(f"[pretrain] art on {len(data['hi_a'])} synthetic pairs (cap {args.epochs}) ...", flush=True)
    art, ha = finetune(data["hi_a"], data["hi_p"], data["hi_d"],
                       init_ckpt=(args.out_art if args.resume else None),
                       epochs=args.epochs, ckpt_path=args.out_art)

    print("[eval] held-out real v117 sites (synthetic-pretrained encoders) ...")
    ranks, ncand, recov = evaluate_bands_learned(
        cl, ts, test_roots, make_embed_fn(bio), make_embed_fn(art),
        mip=args.mip, radius_nm=args.radius_nm, max_sites=args.max_sites, sigma=args.sigma)
    n = len(ranks["geom"])
    if n == 0:
        print("no scorable test sites"); return
    summ = {"n_sites": n, "n_synth_pairs": int(len(data["lo_a"])), "mip": args.mip,
            "mean_candidates": float(np.mean(ncand)),
            "chance_top1": float(np.mean(1.0 / np.asarray(ncand)))}
    for m, rs in ranks.items():
        rs = np.asarray(rs)
        summ[m] = {"top1": float((rs == 0).mean()), "mrr": float(np.mean(1.0 / (rs + 1.0)))}
    summ["geom_miss_recovery"] = {m: (float(np.mean(h)) if h else None) for m, h in recov.items()}
    print(f"\nSynthetic-pretrained bands @ mip{args.mip}: {n} real test sites, "
          f"{summ['n_synth_pairs']} synth pairs, chance top-1 {summ['chance_top1']:.3f}")
    for m in ranks:
        print(f"  {m:24s} top-1 / MRR: {summ[m]['top1']:.3f} / {summ[m]['mrr']:.3f}")
    for m, val in summ["geom_miss_recovery"].items():
        if val is not None:
            print(f"      recover[{m}] = {val:.3f}")
    with open(args.metrics, "w") as f:
        json.dump({"mip": args.mip, "radius_nm": args.radius_nm, "summary": summ}, f, indent=2)
    print(f"[out] wrote {args.metrics}")


if __name__ == "__main__":
    main()
