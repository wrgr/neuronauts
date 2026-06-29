"""Honest combiner eval with abstention -- no cherry-picking.

The headline combiner number (0.758) is a *correction-given-candidates* metric:
``site_faces_bands`` drops any site whose true partner isn't in the panel, so it
conditions on the candidate generator already succeeding.  That is fine as a
component metric but overstates deployed yield.

This module evaluates the deployable system honestly.  It collects ALL real
false-split sites (``require_true=False`` keeps the partner-absent ones, which a
real tool can't filter out), trains the combiner including those all-negative
panels (so it learns to score low when nothing matches), then makes a
*label-blind* accept/abstain decision: act only when the top candidate's score
clears a threshold.  Acting on a partner-absent site is a guaranteed wrong
merge, so the model must learn to abstain there.

Two honest numbers fall out, swept over the threshold:
  precision @ coverage  -- of the sites it acts on, how often the merge is right
  recall                -- fraction of ALL true errors it correctly fixes
The abstention is legitimate (precision-critical proofreading; a wrong auto-merge
is worse than no action); the cherry-pick was not (it used the label to drop).
"""

from __future__ import annotations

import json

import numpy as np

from . import v117_error_relink as v
from .v117_artifact_bands import site_faces_bands
from .learned_cutface_encoder import load_encoder, make_embed_fn
from .train_combiner import _z, _sims, train_mlp, _score


def site_features_all(cl, ts, site, bio_emb, art_emb, *, mip=1, radius_nm=2000.0,
                      direction_cone_deg=45.0, sigma=2.0):
    """Per-candidate X [C,F], labels y [C] -- KEEPS partner-absent sites (all-y=0)."""
    f = site_faces_bands(cl, ts, site, mip=mip, radius_nm=radius_nm,
                         direction_cone_deg=direction_cone_deg, sigma=sigma,
                         require_true=False)
    if f is None:
        return None
    it = f["is_true"].astype(np.float32)
    gd = f["geom_dist"].astype(float)
    arts = _sims(f["q_high"], f["high"], art_emb)
    bios = _sims(f["q_low"], f["low"], bio_emb)
    gz = -_z(gd)
    X = np.stack([
        gz, _z(arts), _z(bios), arts, bios,
        (gd == gd.min()).astype(np.float32),
        (arts == arts.max()).astype(np.float32),
    ], axis=1).astype(np.float32)
    return X, it, gd, arts


def collect_all(cl, ts, roots, bio_emb, art_emb, *, mip=1, radius_nm=2000.0,
                direction_cone_deg=45.0, max_sites=10, verbose=True):
    sites = []
    for n, rt in enumerate(roots):
        try:
            ss = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=radius_nm, max_sites=max_sites)
        except Exception:
            continue
        for s in ss:
            try:
                out = site_features_all(cl, ts, s, bio_emb, art_emb, mip=mip,
                                        radius_nm=radius_nm, direction_cone_deg=direction_cone_deg)
            except Exception:
                out = None
            if out is not None:
                sites.append(out)
        if verbose and sites:
            npos = sum(1 for x in sites if x[1].any())
            print(f"  collect: {n + 1}/{len(roots)} neurons, {len(sites)} sites "
                  f"({npos} with partner present)", flush=True)
    return sites


def evaluate_abstain(test_sites, net):
    """Precision/coverage/recall over a label-blind score threshold.

    For each site: confidence = max candidate score; prediction = argmax.
    A site is a true error needing a fix, so the recall denominator is ALL sites.
    On a partner-absent site any action is a wrong merge -> the model should
    abstain there.  Reports the always-act point and a swept curve.
    """
    conf, correct = [], []
    for X, it, gd, arts in test_sites:
        s = _score(net, X)
        j = int(np.argmax(s))
        conf.append(float(s[j]))
        correct.append(int(bool(it[j])))           # right merge iff picked the true partner
    conf = np.asarray(conf); correct = np.asarray(correct)
    n = len(test_sites)
    n_present = int(sum(1 for x in test_sites if x[1].any()))   # ceiling on recall

    # geometry-alone, always-act (honest top-1 over the full population)
    geom_correct = np.array([int(bool(it[int(np.argmin(gd))]))
                             for _, it, gd, _ in test_sites])

    def point(thr):
        act = conf >= thr
        cov = float(act.mean())
        prec = float(correct[act].mean()) if act.any() else 0.0
        rec = float(correct[act].sum() / n)        # of all true errors, fraction fixed
        return {"thr": float(thr), "coverage": cov, "precision": prec, "recall": rec,
                "n_acted": int(act.sum())}

    # sweep thresholds at the observed confidence quantiles
    qs = [0.0, 0.25, 0.5, 0.75, 0.9]
    curve = [point(np.quantile(conf, q)) for q in qs]
    always = point(conf.min() - 1.0)               # act on everything

    return {"n_sites": n, "n_partner_present": n_present,
            "panel_recall_ceiling": float(n_present / n) if n else 0.0,
            "combiner_alwaysact_top1": always["precision"],   # == recall here (cov=1)
            "geom_alwaysact_top1": float(geom_correct.mean()),
            "curve": curve}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=200)
    ap.add_argument("--train-neurons", type=int, default=40)
    ap.add_argument("--test-neurons", type=int, default=20)
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--radius-nm", type=float, default=2000.0)
    ap.add_argument("--max-sites", type=int, default=10)
    ap.add_argument("--bio", default="experiments/fingerprints/cutface_bio_synth_ft.pt")
    ap.add_argument("--art", default="experiments/fingerprints/cutface_art_synth.pt")
    ap.add_argument("--out", default="experiments/fingerprints/combiner_abstain_metrics.json")
    args = ap.parse_args()

    cl = v._client()
    ts = cl.chunkedgraph.get_oldest_timestamp()
    roots, _ = v.find_split_neurons(cl, n_scan=args.n_scan)
    test_roots = roots[:args.test_neurons]
    train_roots = roots[args.test_neurons:args.test_neurons + args.train_neurons]
    print(f"[split] {len(train_roots)} train / {len(test_roots)} test neurons (disjoint)", flush=True)

    bio_emb = make_embed_fn(load_encoder(args.bio))
    art_emb = make_embed_fn(load_encoder(args.art))

    print("[collect] train sites (incl. partner-absent) ...", flush=True)
    train_sites = collect_all(cl, ts, train_roots, bio_emb, art_emb, mip=args.mip,
                              radius_nm=args.radius_nm, max_sites=args.max_sites)
    print("[collect] test sites (incl. partner-absent) ...", flush=True)
    test_sites = collect_all(cl, ts, test_roots, bio_emb, art_emb, mip=args.mip,
                             radius_nm=args.radius_nm, max_sites=args.max_sites)
    if not train_sites or not test_sites:
        print("insufficient sites"); return

    print(f"[train] combiner on {len(train_sites)} sites ...", flush=True)
    net = train_mlp(train_sites)
    res = evaluate_abstain(test_sites, net)

    print(f"\nHonest abstention eval: {res['n_sites']} real sites "
          f"({res['n_partner_present']} with partner present, "
          f"recall ceiling {res['panel_recall_ceiling']:.3f})")
    print(f"  geometry always-act top-1 (full pop): {res['geom_alwaysact_top1']:.3f}")
    print(f"  combiner always-act top-1 (full pop): {res['combiner_alwaysact_top1']:.3f}")
    print(f"  {'thr':>8} {'coverage':>9} {'precision':>10} {'recall':>8} {'n_acted':>8}")
    for p in res["curve"]:
        print(f"  {p['thr']:8.3f} {p['coverage']:9.3f} {p['precision']:10.3f} "
              f"{p['recall']:8.3f} {p['n_acted']:8d}")
    with open(args.out, "w") as f:
        json.dump({"mip": args.mip, "radius_nm": args.radius_nm,
                   "n_train_sites": len(train_sites), **res}, f, indent=2)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
