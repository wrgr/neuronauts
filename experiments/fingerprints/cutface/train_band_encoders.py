"""Fair artifact test: LEARNED per-band encoders at 8 nm (MIP0).

The raw-cosine band test (`v117_artifact_bands.py`) found the high-pass/artifact
band at chance -- but that is the band raw cosine reads worst, at 16 nm.  This
module runs the fair version:

* at **MIP0 (8 nm)** -- the finest reachable resolution, where texture/grain
  survives,
* with a **learned encoder per band** -- a bio encoder trained on the low-pass
  (structure) face and an art encoder trained on the high-pass (texture/artifact)
  face, both contrastive on real v117 split pairs,
* then evaluate geom / bio / art and every fusion, including **geom+bio+art**
  and a gated variant, with geom-miss recovery.

Train/test neurons are disjoint.  Band patches are cached to npz; 8 nm boxes are
cached on disk (separate from the 16 nm cache, keyed by mip).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .fingerprint_break_resolution import PATCH
from . import v117_error_relink as v
from . import v117_reconstructed as r
from .v117_artifact_bands import site_faces_bands
from .train_real_cutface import finetune
from .learned_cutface_encoder import make_embed_fn, load_encoder


def collect_bands(cl, ts, roots, *, mip=0, radius_nm=2000.0, direction_cone_deg=45.0,
                  n_distractors=8, max_sites=8, sigma=2.0, seed=0, verbose=True):
    """Per site: (anchor, positive, distractors) band patches for low and high."""
    rng = np.random.default_rng(seed)
    lo_a, lo_p, lo_d, hi_a, hi_p, hi_d = [], [], [], [], [], []
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
            t = int(np.argmax(it))
            others = [i for i in range(len(it)) if not it[i]]
            if not others:
                continue
            pick = rng.choice(others, size=n_distractors, replace=len(others) < n_distractors)
            lo_a.append(f["q_low"]); lo_p.append(f["low"][t]); lo_d.append(f["low"][pick])
            hi_a.append(f["q_high"]); hi_p.append(f["high"][t]); hi_d.append(f["high"][pick])
        if verbose and (n % 10 == 0 or n == len(roots) - 1):
            print(f"  collect: {n + 1}/{len(roots)} neurons, {len(lo_a)} pairs")
    if not lo_a:
        raise RuntimeError("no band training pairs")
    st = lambda L: np.stack(L).astype(np.float32)
    return {"lo_a": st(lo_a), "lo_p": st(lo_p), "lo_d": st(lo_d),
            "hi_a": st(hi_a), "hi_p": st(hi_p), "hi_d": st(hi_d)}


def _cos(query, bank, emb):
    qe = np.asarray(emb(query[None]))[0]
    ce = np.asarray(emb(bank))
    qe = qe / (np.linalg.norm(qe) + 1e-9)
    ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-9)
    return 1.0 - ce @ qe


def evaluate_bands_learned(cl, ts, roots, bio_emb, art_emb, *, mip=0, radius_nm=2000.0,
                           direction_cone_deg=45.0, max_sites=8, sigma=2.0, verbose=True):
    methods = ["geom", "bio", "art", "bio+art", "geom+bio", "geom+art", "geom+bio+art",
               "gated_geom_top3+bioart"]
    ranks = {m: [] for m in methods}
    recov = {m: [] for m in ("bio", "art", "bio+art", "geom+bio+art", "gated_geom_top3+bioart")}
    ncand = []
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
            ncand.append(len(it))
            d_geom = f["geom_dist"].astype(float)
            d_bio = _cos(f["q_low"], f["low"], bio_emb)
            d_art = _cos(f["q_high"], f["high"], art_emb)
            zg, zb, za_ = r._z(d_geom), r._z(d_bio), r._z(d_art)
            d_bioart = zb + za_
            dd = {"geom": d_geom, "bio": d_bio, "art": d_art, "bio+art": d_bioart,
                  "geom+bio": zg + zb, "geom+art": zg + za_, "geom+bio+art": zg + zb + za_}
            # gated: geometry's top-3 shortlist, re-ranked by the bio+art hash
            og = np.argsort(d_geom)[:min(3, len(d_geom))]
            pick = og[int(np.argmin(d_bioart[og]))]
            gated_hit = int(bool(it[pick]))
            geom_miss = r._best_rank(d_geom, it) != 0
            for m, d in dd.items():
                rk = r._best_rank(d, it)
                ranks[m].append(rk)
                if geom_miss and m in recov:
                    recov[m].append(int(rk == 0))
            ranks["gated_geom_top3+bioart"].append(0 if gated_hit else 1)  # 0 => top-1 hit
            if geom_miss:
                recov["gated_geom_top3+bioart"].append(gated_hit)
        if verbose and ranks["geom"]:
            print(f"  eval: {n + 1}/{len(roots)} neurons, {len(ranks['geom'])} sites")
    return ranks, ncand, recov


def main():
    import argparse
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=300)
    ap.add_argument("--train-neurons", type=int, default=60)
    ap.add_argument("--test-neurons", type=int, default=40)
    ap.add_argument("--mip", type=int, default=0, help="0 = 8 nm (finest reachable)")
    ap.add_argument("--radius-nm", type=float, default=2000.0)
    ap.add_argument("--direction-cone-deg", type=float, default=45.0)
    ap.add_argument("--max-sites", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=80, help="epoch CAP (val early-stop selects)")
    ap.add_argument("--cache", default=None, help="npz cache of collected band patches")
    ap.add_argument("--out-bio", default="experiments/fingerprints/cutface_encoder_bio.pt")
    ap.add_argument("--out-art", default="experiments/fingerprints/cutface_encoder_art.pt")
    ap.add_argument("--metrics", default="experiments/fingerprints/band_learned_metrics.json")
    args = ap.parse_args()

    cl = v._client()
    ts = cl.chunkedgraph.get_oldest_timestamp()
    print(f"[cave] scanning {args.n_scan} somas; mip={args.mip} (8nm if 0); v117 ts={ts}")
    roots, _ = v.find_split_neurons(cl, n_scan=args.n_scan)
    rng = np.random.default_rng(0)
    roots = list(rng.permutation(roots))
    train_roots = roots[:args.train_neurons]
    test_roots = roots[args.train_neurons:args.train_neurons + args.test_neurons]
    print(f"[split] {len(train_roots)} train / {len(test_roots)} test neurons")

    if args.cache and os.path.exists(args.cache):
        z = np.load(args.cache)
        data = {k: z[k] for k in z.files}
        print(f"[cache] loaded {len(data['lo_a'])} band pairs")
    else:
        print("[collect] band patches at 8nm ...")
        data = collect_bands(cl, ts, train_roots, mip=args.mip, radius_nm=args.radius_nm,
                             direction_cone_deg=args.direction_cone_deg,
                             max_sites=args.max_sites, sigma=args.sigma)
        if args.cache:
            np.savez(args.cache, **data)
            print(f"[cache] wrote {len(data['lo_a'])} pairs -> {args.cache}")

    # epochs is a CAP; finetune early-stops on val top-1 and writes the best
    # (resumable) checkpoint to ckpt_path.
    print(f"[train] bio encoder (low-pass), {len(data['lo_a'])} pairs (cap {args.epochs}) ...")
    bio_enc, _ = finetune(data["lo_a"], data["lo_p"], data["lo_d"], init_ckpt=None,
                          epochs=args.epochs, ckpt_path=args.out_bio)
    print(f"[train] art encoder (high-pass), {len(data['hi_a'])} pairs (cap {args.epochs}) ...")
    art_enc, _ = finetune(data["hi_a"], data["hi_p"], data["hi_d"], init_ckpt=None,
                          epochs=args.epochs, ckpt_path=args.out_art)

    print("[eval] held-out test neurons (8 nm bands) ...")
    ranks, ncand, recov = evaluate_bands_learned(
        cl, ts, test_roots, make_embed_fn(bio_enc), make_embed_fn(art_enc),
        mip=args.mip, radius_nm=args.radius_nm, direction_cone_deg=args.direction_cone_deg,
        max_sites=args.max_sites, sigma=args.sigma)
    n = len(ranks["geom"])
    if n == 0:
        print("no scorable test sites"); return
    summ = {"n_sites": n, "mip": args.mip, "sigma": args.sigma,
            "mean_candidates": float(np.mean(ncand)),
            "chance_top1": float(np.mean(1.0 / np.asarray(ncand)))}
    for m, rs in ranks.items():
        rs = np.asarray(rs)
        summ[m] = {"top1": float((rs == 0).mean()), "mrr": float(np.mean(1.0 / (rs + 1.0)))}
    summ["geom_miss_recovery"] = {m: (float(np.mean(h)) if h else None) for m, h in recov.items()}
    summ["n_geom_misses"] = len(recov["bio"])

    print(f"\nLearned bands @ mip{args.mip}: {n} test sites, mean {summ['mean_candidates']:.1f} "
          f"candidates, chance top-1 {summ['chance_top1']:.3f}")
    for m in ranks:
        print(f"  {m:24s} top-1 / MRR: {summ[m]['top1']:.3f} / {summ[m]['mrr']:.3f}")
    print(f"  of {summ['n_geom_misses']} geom-misses, recovered top-1 by:")
    for m, val in summ["geom_miss_recovery"].items():
        if val is not None:
            print(f"      {m:24s}: {val:.3f}")

    with open(args.metrics, "w") as f:
        json.dump({"mip": args.mip, "sigma": args.sigma, "radius_nm": args.radius_nm,
                   "train_neurons": len(train_roots), "test_neurons": len(test_roots),
                   "n_train_pairs": int(len(data["lo_a"])), "summary": summ}, f, indent=2)
    print(f"[out] wrote {args.metrics}")


if __name__ == "__main__":
    main()
