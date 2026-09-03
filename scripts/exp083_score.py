"""EXP-083 scoring: held-out separation of correct from corrupted assemblies."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from neuronauts.harness.baselines import GradientBoostedStumps, LogisticRegression
from neuronauts.metrics.ranking import roc_auc
import exp083_shape_lib as L

OUT = R / "results/EXP-083"
BIN_NAMES = ["1-3", "3-10", "10-30", "30-100", "100-300", "300-1000", ">1000"]
N_FOLD = 5
SEED = 0

SETS = {
    "size_only":   L.SIZE_COLS,
    "topology":    L.TOPO_COLS,
    "extent":      L.EXTENT_COLS,
    "shape_geom":  L.TOPO_COLS + L.EXTENT_COLS,
    "shape+size":  L.SIZE_COLS + L.TOPO_COLS + L.EXTENT_COLS,
    "shape+radius": L.TOPO_COLS + L.EXTENT_COLS + L.RADIUS_COLS,
    "all":         L.ALL_COLS,
}


def folds(cells, n_fold, seed=SEED):
    uc = np.unique(cells)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uc))
    assign = {int(uc[perm[i]]): i % n_fold for i in range(len(uc))}
    return np.array([assign[int(c)] for c in cells])


def oof_scores(X, y, fold, model="gbs"):
    s = np.zeros(len(y))
    for k in range(N_FOLD):
        tr, te = fold != k, fold == k
        if not te.any() or y[tr].sum() in (0, tr.sum()):
            continue
        if model == "gbs":
            m = GradientBoostedStumps.fit(X[tr], y[tr], n_rounds=120, lr=0.15,
                                          min_leaf=20, seed=SEED)
        else:
            m = LogisticRegression.fit(X[tr], y[tr])
        s[te] = m.decision(X[te])
    return s


def oneclass(X, y, fold):
    """Normality score: fit on CORRECT training assemblies only, no corrupted
    example ever seen.  Mahalanobis distance under a shrunk Gaussian."""
    s = np.zeros(len(y))
    for k in range(N_FOLD):
        tr, te = (fold != k) & (y == 0), fold == k
        if tr.sum() < 20 or not te.any():
            continue
        Z = X[tr]
        mu, sd = Z.mean(0), Z.std(0)
        sd[sd < 1e-9] = 1.0
        Zs = (Z - mu) / sd
        C = np.cov(Zs.T) + 0.10 * np.eye(Zs.shape[1])
        P = np.linalg.pinv(C)
        D = (X[te] - mu) / sd
        s[te] = np.einsum("ij,jk,ik->i", D, P, D)
    return s


def paired(score, meta_pair, y):
    """Fraction of matched pairs where the corrupted twin scores higher."""
    d = {}
    for i, p in enumerate(meta_pair):
        d.setdefault(int(p), {})[int(y[i])] = score[i]
    v = [(x[1] - x[0]) for x in d.values() if 0 in x and 1 in x]
    v = np.asarray(v)
    return float((v > 0).mean() + 0.5 * (v == 0).mean()), len(v)


def boot_ci(vals, cells, n=2000, seed=SEED):
    """Bootstrap over cells on a per-pair statistic."""
    rng = np.random.default_rng(seed)
    uc = np.unique(cells)
    by = {int(c): vals[cells == c] for c in uc}
    out = []
    for _ in range(n):
        pick = rng.choice(uc, len(uc), replace=True)
        v = np.concatenate([by[int(c)] for c in pick])
        out.append(v.mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    d = np.load(OUT / "features.npz", allow_pickle=True)
    X, cols = d["X"], [str(c) for c in d["cols"]]
    meta = json.loads(str(d["meta"]))
    y = np.array([m["y"] for m in meta])
    cell = np.array([m["cell"] for m in meta])
    pair = np.array([m["pair"] for m in meta])
    bins = np.array([m["bin"] for m in meta])
    added = np.array([m["added_um"] for m in meta])
    same_type = np.array([m["same_type"] for m in meta])
    comp_match = np.array([int(m["comp_u"] == m["comp_w"]) for m in meta])
    total = np.array([m["total_um"] for m in meta])
    fold = folds(cell, N_FOLD)
    ci = {c: i for i, c in enumerate(cols)}
    rep = {}

    print(f"pairs {len(y)//2}  cells {len(np.unique(cell))}  "
          f"assemblies {len(y)}")
    print("per-bin pair counts:",
          {BIN_NAMES[b]: int((bins[y == 1] == b).sum()) for b in range(7)})
    dc = np.abs(np.array([m["donor_um"] - m["added_um"] for m in meta])[y == 1])
    ac = added[y == 1]
    print(f"donor cable mismatch: median {np.median(dc/ac)*100:.2f}% of the "
          f"added cable; added cable is median "
          f"{np.median(ac/total[y==1])*100:.2f}% of the arbor")
    rep["n_pairs"] = int(len(y) // 2)
    rep["n_cells"] = int(len(np.unique(cell)))
    rep["bin_counts"] = {BIN_NAMES[b]: int((bins[y == 1] == b).sum())
                         for b in range(7)}
    rep["donor_mismatch_pct"] = float(np.median(dc / ac) * 100)

    # ---- feature sets, held out by cell -------------------------------
    print("\nheld-out (5-fold, cell-disjoint)      AUC    paired-acc")
    rep["sets"] = {}
    for name, sc in SETS.items():
        Xs = X[:, [ci[c] for c in sc]]
        for mdl in ("gbs", "logit"):
            s = oof_scores(Xs, y, fold, mdl)
            auc = roc_auc(y == 1, s)
            pa, npair = paired(s, pair, y)
            print(f"  {name:14s} {mdl:6s}  {auc:.3f}   {pa:.3f}  ({npair})")
            rep["sets"][f"{name}|{mdl}"] = dict(auc=float(auc), paired=pa,
                                                n_pairs=npair)
    s1 = oneclass(X[:, [ci[c] for c in SETS["shape_geom"]]], y, fold)
    auc1 = roc_auc(y == 1, s1)
    pa1, _ = paired(s1, pair, y)
    print(f"  {'shape_geom':14s} {'1class':6s}  {auc1:.3f}   {pa1:.3f}   "
          f"(fit on correct assemblies only)")
    rep["one_class"] = dict(auc=float(auc1), paired=pa1)

    # ---- the headline model, by added-cable bin -----------------------
    best = "shape_geom"
    Xs = X[:, [ci[c] for c in SETS[best]]]
    s = oof_scores(Xs, y, fold, "gbs")
    np.save(OUT / "oof_score.npy", s)
    print(f"\n{best} (gradient-boosted stumps), by wrong cable added")
    print("  bin (um)     pairs   AUC    paired-acc [95% CI over cells]")
    rep["by_bin"] = {}
    for b in range(7):
        m = bins == b
        if m.sum() < 8:
            continue
        auc = roc_auc(y[m] == 1, s[m])
        # per-pair paired outcome, for a cell-level bootstrap
        dd = {}
        for i in np.flatnonzero(m):
            dd.setdefault(int(pair[i]), {})[int(y[i])] = (s[i], int(cell[i]))
        vals, cs = [], []
        for p, x in dd.items():
            if 0 in x and 1 in x:
                vals.append(1.0 * (x[1][0] > x[0][0]) + 0.5 * (x[1][0] == x[0][0]))
                cs.append(x[0][1])
        vals, cs = np.asarray(vals), np.asarray(cs)
        lo, hi = boot_ci(vals, cs)
        med = float(np.median(added[m & (y == 1)]))
        print(f"  {BIN_NAMES[b]:11s} {len(vals):5d}   {auc:.3f}  {vals.mean():.3f} "
              f"[{lo:.3f}, {hi:.3f}]  median added {med:.1f} um")
        rep["by_bin"][BIN_NAMES[b]] = dict(n_pairs=int(len(vals)), auc=float(auc),
                                           paired=float(vals.mean()),
                                           ci=[lo, hi], median_added_um=med)

    # ---- subgroups -----------------------------------------------------
    print("\nsubgroups (shape_geom, gbs)")
    rep["subgroups"] = {}
    for lab, m in (("donor same cell type", same_type == 1),
                   ("donor different type", same_type == 0),
                   ("same compartment (axon->axon / dend->dend)", comp_match == 1),
                   ("different compartment", comp_match == 0)):
        if m.sum() < 8:
            continue
        auc = roc_auc(y[m] == 1, s[m])
        pa, n = paired(s[m], pair[m], y[m])
        print(f"  {lab:44s} {n:5d} pairs  AUC {auc:.3f}  paired {pa:.3f}")
        rep["subgroups"][lab] = dict(n_pairs=n, auc=float(auc), paired=pa)

    # ---- which descriptors carry it (paired, no model) ------------------
    print("\nsingle descriptors, paired direction (|0.5 - acc| ranked)")
    uni = []
    for c in cols:
        v = X[:, ci[c]]
        pa, n = paired(v, pair, y)
        uni.append((abs(pa - 0.5), pa, c))
    uni.sort(reverse=True)
    for a, pa, c in uni[:12]:
        print(f"  {c:22s} paired {pa:.3f}")
    rep["top_descriptors"] = [dict(col=c, paired=pa) for _, pa, c in uni[:12]]

    # large-graft slice, for the operating question
    m = added >= 100
    if m.sum() > 8:
        pa, n = paired(s[m], pair[m], y[m])
        rep["ge_100um"] = dict(n_pairs=n, auc=float(roc_auc(y[m] == 1, s[m])),
                               paired=pa)
    json.dump(rep, open(OUT / "result.json", "w"), indent=1)
    print("\nwrote", OUT / "result.json")


if __name__ == "__main__":
    main()
