"""EXP-083 -- absolute vs. within-site scoring of whole-cell shape.

Two questions, and they get very different answers:

  ABSOLUTE   can a threshold on a whole-cell shape score tell a corrupted
             assembly from a correct one?  Measured by ROC area over all
             assemblies, held out by cell.
  WITHIN-SITE  offered the true piece and a wrong piece at the same frontier
             site, does the assembly built with the true piece score better?
             Measured as paired accuracy.  This is the ranking a grower
             actually performs, and it is what a conditional (paired) logistic
             model is fit for.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "scripts"))
from neuronauts.metrics.ranking import roc_auc
import exp083_shape_lib as L

OUT = R / "results/EXP-083"
N_FOLD = 5
SEED = 0
BIN_NAMES = ["1-3", "3-10", "10-30", "30-100", "100-300", "300-1000", ">1000"]

STRICT = [c for c in L.TOPO_COLS + L.EXTENT_COLS]          # no radius, no label
SETS = {
    "size_only":     L.SIZE_COLS,
    "shape_geom":    STRICT,
    "shape+radius":  STRICT + L.RADIUS_COLS,
    "shape+compart": STRICT + L.COMP_COLS,
    "all":           L.ALL_COLS,
}


def cond_logit(D, l2=1e-2, lr=0.5, n_iter=4000):
    """Conditional logistic: maximise P(corrupted scores above its own twin).

    ``D`` is one row per pair, ``x_corrupted - x_correct``.  There is no
    intercept: any per-cell offset cancels inside a pair, which is exactly the
    confound this model is built to ignore.
    """
    w = np.zeros(D.shape[1])
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(D @ w, -30, 30)))
        g = D.T @ (p - 1.0) / len(D) + l2 * w
        w -= lr * g
    return w


def folds(cells, seed=SEED):
    uc = np.unique(cells)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uc))
    a = {int(uc[perm[i]]): i % N_FOLD for i in range(len(uc))}
    return np.array([a[int(c)] for c in cells])


def load(name):
    d = np.load(OUT / name, allow_pickle=True)
    X = d["X"]
    cols = [str(c) for c in d["cols"]]
    meta = json.loads(str(d["meta"]))
    return X, cols, meta


def boot(vals, cells, n=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    uc = np.unique(cells)
    by = {int(c): vals[cells == c] for c in uc}
    o = [np.concatenate([by[int(c)] for c in rng.choice(uc, len(uc), True)]).mean()
         for _ in range(n)]
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def run(name, tag, rep):
    X, cols, meta = load(name)
    ci = {c: i for i, c in enumerate(cols)}
    y = np.array([m["y"] for m in meta])
    cell = np.array([m["cell"] for m in meta])
    pair = np.array([m["pair"] for m in meta])
    added = np.array([m["added_um"] for m in meta])
    total = np.array([m["total_um"] for m in meta])
    fold = folds(cell)
    # pair table
    order = np.argsort(pair * 2 + y)
    assert (y[order][0::2] == 0).all() and (y[order][1::2] == 1).all()
    i0, i1 = order[0::2], order[1::2]
    pcell, padd = cell[i0], added[i1]
    pfrac = padd / total[i0]
    pfold = fold[i0]
    print(f"\n===== {tag}: {len(i0)} pairs, {len(np.unique(pcell))} cells")
    print(f"  wrong cable added: median {np.median(padd):.0f} um "
          f"= {np.median(pfrac)*100:.2f}% of the arbor; "
          f"p90 {np.percentile(padd,90):.0f} um "
          f"({np.percentile(pfrac,90)*100:.1f}%)")
    rep[tag] = dict(n_pairs=int(len(i0)), n_cells=int(len(np.unique(pcell))),
                    median_added_um=float(np.median(padd)),
                    median_added_frac=float(np.median(pfrac)))

    print("  feature set      absolute AUC   within-site paired acc [95% CI]")
    rep[tag]["sets"] = {}
    scores = {}
    for sname, sc in SETS.items():
        idx = [ci[c] for c in sc]
        Xs = X[:, idx]
        sd = Xs.std(0)
        sd[sd < 1e-12] = 1.0
        Z = Xs / sd
        s = np.zeros(len(y))
        for k in range(N_FOLD):
            trp, tep = pfold != k, fold == k
            if trp.sum() < 10 or not tep.any():
                continue
            w = cond_logit(Z[i1][trp] - Z[i0][trp])
            s[tep] = Z[tep] @ w
        auc = roc_auc(y == 1, s)
        v = (s[i1] > s[i0]).astype(float) + 0.5 * (s[i1] == s[i0])
        lo, hi = boot(v, pcell)
        print(f"  {sname:15s} {auc:.3f}         {v.mean():.3f} [{lo:.3f}, {hi:.3f}]")
        rep[tag]["sets"][sname] = dict(abs_auc=float(auc), paired=float(v.mean()),
                                       ci=[lo, hi])
        scores[sname] = (s, v)

    head = "shape_geom"
    s, v = scores[head]
    print(f"\n  {head} (label-free, radius-free) by wrong cable added")
    print("   bin um        pairs  median added   % of arbor  paired acc [95% CI]")
    rep[tag]["by_bin"] = {}
    edges = [1, 3, 10, 30, 100, 300, 1000, 1e9]
    for b in range(7):
        m = (padd >= edges[b]) & (padd < edges[b + 1])
        if m.sum() < 15:
            continue
        lo, hi = boot(v[m], pcell[m])
        print(f"   {BIN_NAMES[b]:12s} {int(m.sum()):5d}   {np.median(padd[m]):8.1f}   "
              f"{np.median(pfrac[m])*100:8.2f}%    {v[m].mean():.3f} [{lo:.3f}, {hi:.3f}]")
        rep[tag]["by_bin"][BIN_NAMES[b]] = dict(
            n=int(m.sum()), median_added_um=float(np.median(padd[m])),
            median_frac=float(np.median(pfrac[m])), paired=float(v[m].mean()),
            ci=[lo, hi])
    # by fraction of arbor
    print(f"\n  {head} by wrong cable as a share of the arbor")
    rep[tag]["by_frac"] = {}
    fe = [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 1.0]
    for b in range(len(fe) - 1):
        m = (pfrac >= fe[b]) & (pfrac < fe[b + 1])
        if m.sum() < 15:
            continue
        lo, hi = boot(v[m], pcell[m])
        lab = f"{fe[b]*100:g}-{fe[b+1]*100:g}%"
        print(f"   {lab:12s} {int(m.sum()):5d}   {np.median(padd[m]):8.1f} um"
              f"            {v[m].mean():.3f} [{lo:.3f}, {hi:.3f}]")
        rep[tag]["by_frac"][lab] = dict(n=int(m.sum()), paired=float(v[m].mean()),
                                        ci=[lo, hi])
    # all sets on the biggest grafts
    m = pfrac >= 0.05
    if m.sum() >= 15:
        print(f"\n  wrong cable >= 5% of the arbor ({int(m.sum())} pairs)")
        rep[tag]["ge5pct"] = {}
        for sname, (ss, vv) in scores.items():
            lo, hi = boot(vv[m], pcell[m])
            print(f"   {sname:15s} paired {vv[m].mean():.3f} [{lo:.3f}, {hi:.3f}]"
                  f"   absolute AUC {roc_auc(y[np.isin(pair, np.flatnonzero(m))] == 1, ss[np.isin(pair, np.flatnonzero(m))]):.3f}")
            rep[tag]["ge5pct"][sname] = dict(paired=float(vv[m].mean()), ci=[lo, hi])

    # single descriptors, within-site, cell-bootstrapped
    print("\n  single descriptors, within-site (all pairs)")
    uni = []
    for c in cols:
        d = X[i1, ci[c]] - X[i0, ci[c]]
        acc = float((d > 0).mean() + 0.5 * (d == 0).mean())
        uni.append((abs(acc - 0.5), acc, c))
    uni.sort(reverse=True)
    rep[tag]["top_descriptors"] = []
    for a, acc, c in uni[:10]:
        d = X[i1, ci[c]] - X[i0, ci[c]]
        vv = (d > 0).astype(float) + 0.5 * (d == 0)
        lo, hi = boot(vv, pcell)
        print(f"   {c:22s} {acc:.3f} [{lo:.3f}, {hi:.3f}]")
        rep[tag]["top_descriptors"].append(dict(col=c, paired=acc, ci=[lo, hi]))
    return rep


if __name__ == "__main__":
    rep = {}
    run("features.npz", "arm1_random_cut", rep)
    if (OUT / "features_arm2.npz").exists():
        run("features_arm2.npz", "arm2_real_breaks", rep)
    json.dump(rep, open(OUT / "result.json", "w"), indent=1)
    print("\nwrote", OUT / "result.json")
