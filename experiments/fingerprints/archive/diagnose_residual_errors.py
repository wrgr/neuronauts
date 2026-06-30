"""Are the combiner's residual errors confusable pairs or representation misses?

Trains the confidence combiner exactly as `train_combiner`, then on held-out
test sites dumps the cases it gets WRONG: a montage of the query face, the true
partner, and the candidate the combiner wrongly picked (low-pass bio band, the
most interpretable).  Eyeballing these answers the ceiling question:

* if the true partner and the wrong pick look near-identical -> information
  ceiling (cut faces don't carry enough identity; richer features won't help);
* if the true partner is visibly the better continuation but ranked second ->
  the patch representation is the bottleneck (engineering headroom).

Renders both the combiner's misses and, separately, the geometry-only misses the
combiner *fixed* (for contrast).  No new fetches beyond the cached boxes.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..cutface import v117_error_relink as v
from ..cutface.v117_artifact_bands import site_faces_bands
from ..cutface.learned_cutface_encoder import load_encoder, make_embed_fn
from ..cutface.train_combiner import _z, _sims, train_mlp, _score, collect


def _features(f, bio_emb, art_emb):
    it = f["is_true"].astype(np.float32)
    gd = f["geom_dist"].astype(float)
    arts = _sims(f["q_high"], f["high"], art_emb)
    bios = _sims(f["q_low"], f["low"], bio_emb)
    gz = -_z(gd)
    X = np.stack([gz, _z(arts), _z(bios), arts, bios,
                  (gd == gd.min()).astype(np.float32),
                  (arts == arts.max()).astype(np.float32)], axis=1).astype(np.float32)
    return X, it, gd


def collect_faces(cl, ts, roots, bio_emb, art_emb, *, mip=1, radius_nm=2000.0,
                  direction_cone_deg=45.0, max_sites=10):
    """Like train_combiner.collect but keeps the band images for rendering."""
    out = []
    for rt in roots:
        try:
            ss = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=radius_nm, max_sites=max_sites)
        except Exception:
            continue
        for s in ss:
            try:
                f = site_faces_bands(cl, ts, s, mip=mip, radius_nm=radius_nm,
                                     direction_cone_deg=direction_cone_deg)
            except Exception:
                f = None
            if f is None:
                continue
            X, it, gd = _features(f, bio_emb, art_emb)
            out.append({"X": X, "it": it, "gd": gd, "q_low": f["q_low"], "low": f["low"]})
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=200)
    ap.add_argument("--test-neurons", type=int, default=20)
    ap.add_argument("--train-neurons", type=int, default=40)
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--radius-nm", type=float, default=2000.0)
    ap.add_argument("--max-sites", type=int, default=10)
    ap.add_argument("--bio", default="experiments/fingerprints/cutface_bio_synth_ft.pt")
    ap.add_argument("--art", default="experiments/fingerprints/cutface_art_synth.pt")
    ap.add_argument("--max-panels", type=int, default=12)
    ap.add_argument("--out", default="experiments/fingerprints/residual_errors.png")
    args = ap.parse_args()

    cl = v._client()
    ts = cl.chunkedgraph.get_oldest_timestamp()
    roots, _ = v.find_split_neurons(cl, n_scan=args.n_scan)
    test_roots = roots[:args.test_neurons]
    train_roots = roots[args.test_neurons:args.test_neurons + args.train_neurons]
    bio_emb = make_embed_fn(load_encoder(args.bio))
    art_emb = make_embed_fn(load_encoder(args.art))

    print("[collect] train ...", flush=True)
    train_sites = collect(cl, ts, train_roots, bio_emb, art_emb, mip=args.mip,
                          radius_nm=args.radius_nm, max_sites=args.max_sites)
    print("[collect] test (with faces) ...", flush=True)
    test = collect_faces(cl, ts, test_roots, bio_emb, art_emb, mip=args.mip,
                         radius_nm=args.radius_nm, max_sites=args.max_sites)
    if not train_sites or not test:
        print("insufficient sites"); return

    net = train_mlp(train_sites)

    misses, geom_fixed = [], []
    n_comb_correct = n = 0
    for d in test:
        if not d["it"].any():
            continue
        n += 1
        pick = int(np.argmax(_score(net, d["X"])))
        gpick = int(np.argmin(d["gd"]))
        tru = int(np.argmax(d["it"]))
        comb_ok = bool(d["it"][pick]); geom_ok = bool(d["it"][gpick])
        n_comb_correct += int(comb_ok)
        if not comb_ok:
            misses.append((d, tru, pick, gpick))
        elif not geom_ok and comb_ok:
            geom_fixed.append((d, tru, pick, gpick))
    print(f"[test] {n} sites, combiner top-1 {n_comb_correct / max(n,1):.3f}, "
          f"{len(misses)} misses, {len(geom_fixed)} geom-misses-combiner-fixed", flush=True)

    # render: rows = cases, cols = [query, true partner, combiner pick]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def montage(cases, title, path):
        if not cases:
            print(f"[skip] no cases for {title}"); return
        cases = cases[:args.max_panels]
        fig, axes = plt.subplots(len(cases), 3, figsize=(6, 2 * len(cases)))
        if len(cases) == 1:
            axes = axes[None, :]
        for i, (d, tru, pick, gpick) in enumerate(cases):
            imgs = [d["q_low"], d["low"][tru], d["low"][pick]]
            labs = ["query", f"TRUE (geom rank {int(np.argsort(d['gd']).tolist().index(tru))})",
                    f"PICK {'=geom' if pick == gpick else ''}"]
            for j, (im, lab) in enumerate(zip(imgs, labs)):
                axes[i, j].imshow(im, cmap="gray"); axes[i, j].set_title(lab, fontsize=8)
                axes[i, j].axis("off")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print(f"[out] wrote {path}")

    montage(misses, "Combiner residual errors: query | true partner | wrong pick", args.out)
    base, ext = os.path.splitext(args.out)
    montage(geom_fixed, "Geometry misses the combiner fixed: query | true | pick", base + "_fixed" + ext)

    summ = {"n_test": n, "combiner_top1": n_comb_correct / max(n, 1),
            "n_misses": len(misses), "n_geom_fixed": len(geom_fixed)}
    with open(base + "_summary.json", "w") as fh:
        json.dump(summ, fh, indent=2)
    print(f"[out] wrote {base}_summary.json")


if __name__ == "__main__":
    main()
