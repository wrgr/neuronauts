"""Learned confidence combiner: geometry + bio + art per-candidate ranker.

Equal-weight and hand-gated fusion never beat geometry (0.673): the hash breaks
more of geometry's wins than it fixes when allowed to override.  The fix is to
*learn* when to trust each signal.  This trains a small MLP that scores every
candidate at a real v117 error site from per-candidate features

    [ geom z-score, art sim (+z), bio sim (+z), is-geom-nearest, is-art-best ]

and picks the argmax.  Trained on train neurons, evaluated on disjoint test
neurons; compared against geometry-alone and art-alone.  If the combiner beats
geometry, the fingerprint has been turned into a deployable edge decision.

Uses the band encoders from train_synthetic_skeleton / train_band_encoders.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import v117_error_relink as v
from .v117_artifact_bands import site_faces_bands
from .learned_cutface_encoder import load_encoder, make_embed_fn


def _z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / (s + 1e-9)


def _sims(query, bank, emb):
    qe = np.asarray(emb(query[None]))[0]
    ce = np.asarray(emb(bank))
    qe = qe / (np.linalg.norm(qe) + 1e-9)
    ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-9)
    return ce @ qe                              # cosine similarity per candidate


def site_features(cl, ts, site, bio_emb, art_emb, *, mip=1, radius_nm=2000.0,
                  direction_cone_deg=45.0, sigma=2.0):
    """Per-candidate feature matrix X [C, F] and labels y [C] for one site."""
    f = site_faces_bands(cl, ts, site, mip=mip, radius_nm=radius_nm,
                         direction_cone_deg=direction_cone_deg, sigma=sigma)
    if f is None:
        return None
    it = f["is_true"].astype(np.float32)
    gd = f["geom_dist"].astype(float)
    arts = _sims(f["q_high"], f["high"], art_emb)
    bios = _sims(f["q_low"], f["low"], bio_emb)
    gz = -_z(gd)                                 # higher = closer (geometry favourite)
    X = np.stack([
        gz, _z(arts), _z(bios), arts, bios,
        (gd == gd.min()).astype(np.float32),     # is the geometry nearest
        (arts == arts.max()).astype(np.float32),  # is the art-band favourite
    ], axis=1).astype(np.float32)
    return X, it, gd, arts


def collect(cl, ts, roots, bio_emb, art_emb, *, mip=1, radius_nm=2000.0,
            direction_cone_deg=45.0, max_sites=10, verbose=True):
    sites = []
    for n, rt in enumerate(roots):
        try:
            ss = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=radius_nm, max_sites=max_sites)
        except Exception:
            continue
        for s in ss:
            try:
                out = site_features(cl, ts, s, bio_emb, art_emb, mip=mip,
                                    radius_nm=radius_nm, direction_cone_deg=direction_cone_deg)
            except Exception:
                out = None
            if out is not None:
                sites.append(out)
        if verbose and sites:
            print(f"  collect: {n + 1}/{len(roots)} neurons, {len(sites)} sites", flush=True)
    return sites


def train_mlp(train_sites, *, epochs=200, lr=1e-2, seed=0, verbose=True):
    import torch
    torch.manual_seed(seed)
    X = np.concatenate([s[0] for s in train_sites], axis=0)
    y = np.concatenate([s[1] for s in train_sites], axis=0)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    F = X.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(F, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # class weight: positives are rare (~1 true per site)
    pos_w = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1.0)])
    lossfn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(epochs):
        opt.zero_grad()
        logit = net(Xt).squeeze(1)
        loss = lossfn(logit, yt)
        loss.backward(); opt.step()
        if verbose and ep % 50 == 0:
            print(f"  combiner epoch {ep}  bce={float(loss):.4f}", flush=True)
    net.eval()
    return net


def _score(net, X):
    import torch
    with torch.no_grad():
        return net(torch.from_numpy(X)).squeeze(1).numpy()


def evaluate(test_sites, net):
    """Top-1 of combiner vs geometry-alone vs art-alone on held-out sites."""
    hit_comb, hit_geom, hit_art = [], [], []
    for X, it, gd, arts in test_sites:
        hit_comb.append(int(bool(it[int(np.argmax(_score(net, X)))])))
        hit_geom.append(int(bool(it[int(np.argmin(gd))])))
        hit_art.append(int(bool(it[int(np.argmax(arts))])))
    n = len(test_sites)
    return {"n": n,
            "combiner_top1": float(np.mean(hit_comb)),
            "geom_top1": float(np.mean(hit_geom)),
            "art_top1": float(np.mean(hit_art))}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=300)
    ap.add_argument("--train-neurons", type=int, default=40)
    ap.add_argument("--test-neurons", type=int, default=20)
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--radius-nm", type=float, default=2000.0)
    ap.add_argument("--max-sites", type=int, default=10)
    ap.add_argument("--bio", default="experiments/fingerprints/cutface_bio_synth.pt")
    ap.add_argument("--art", default="experiments/fingerprints/cutface_art_synth.pt")
    ap.add_argument("--cache", default=None, help="npz of collected features (train+test)")
    ap.add_argument("--out", default="experiments/fingerprints/combiner_metrics.json")
    args = ap.parse_args()

    cl = v._client()
    ts = cl.chunkedgraph.get_oldest_timestamp()
    roots, _ = v.find_split_neurons(cl, n_scan=args.n_scan)
    test_roots = roots[:args.test_neurons]
    train_roots = roots[args.test_neurons:args.test_neurons + args.train_neurons]
    print(f"[split] {len(train_roots)} train / {len(test_roots)} test neurons (disjoint)", flush=True)

    bio_emb = make_embed_fn(load_encoder(args.bio))
    art_emb = make_embed_fn(load_encoder(args.art))

    print("[collect] train sites ...", flush=True)
    train_sites = collect(cl, ts, train_roots, bio_emb, art_emb, mip=args.mip,
                          radius_nm=args.radius_nm, max_sites=args.max_sites)
    print("[collect] test sites ...", flush=True)
    test_sites = collect(cl, ts, test_roots, bio_emb, art_emb, mip=args.mip,
                         radius_nm=args.radius_nm, max_sites=args.max_sites)
    if not train_sites or not test_sites:
        print("insufficient sites"); return

    print(f"[train] combiner on {len(train_sites)} sites ...", flush=True)
    net = train_mlp(train_sites)
    res = evaluate(test_sites, net)
    print(f"\nLearned combiner: {res['n']} test sites")
    print(f"  geometry alone   top-1: {res['geom_top1']:.3f}")
    print(f"  art-band alone   top-1: {res['art_top1']:.3f}")
    print(f"  COMBINER         top-1: {res['combiner_top1']:.3f}"
          f"{'  <-- beats geometry' if res['combiner_top1'] > res['geom_top1'] else ''}")
    with open(args.out, "w") as f:
        json.dump({"mip": args.mip, "radius_nm": args.radius_nm,
                   "n_train_sites": len(train_sites), **res}, f, indent=2)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
