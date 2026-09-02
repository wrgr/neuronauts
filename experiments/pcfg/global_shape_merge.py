#!/usr/bin/env python3
"""Global whole-object shape test for false merges -- on cached synapse clouds (no fetch).

The local methods plateaued (gap=0.81 on splits, merge recall 0.02, neural at chance on
gap-free merges) and the group-level guardrail voided them all. The reframe: a neuron is a
connected global object; a false merge is a globally implausible whole -- two somas / two
spatial lobes / two clusters. Test that at the OBJECT level, where local features can't look.

This is the skeleton-topology idea on a coarse substrate (synapse positions instead of the
cable tree), runnable while the MICrONS skeleton service is down. It will catch spatially
separated merges and MISS intertwined ones -- quantifying exactly where the skeleton is needed.

Object = a v117 root's synapse cloud.  Label = false merge (spans >=2 v1718 roots).
Features are GLOBAL shape only (no local junction/gap features). Because false merges are
bigger (two cells), we ablate size-only vs shape-only vs full, and report the do-nothing
guardrail: object base rate, precision/recall, vs do-nothing (flags nothing, recall 0).

Usage:
    python -m experiments.pcfg.global_shape_merge --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg.synapse_correction import (  # noqa: E402
    SideTable,
    cell_components,
)


def global_features(pts: np.ndarray):
    """Whole-object shape descriptors from a synapse cloud (no local junction info)."""
    n = len(pts)
    c = pts - pts.mean(0)
    cov = np.cov(c.T) if n > 1 else np.zeros((3, 3))
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1] if n > 1 else np.zeros(3)
    evals = np.clip(evals, 0, None)
    extent = float(np.linalg.norm(pts.max(0) - pts.min(0))) if n else 0.0
    # 2-means bimodality: how much a 2-cluster split reduces inertia vs 1 cluster
    bimod = center_gap = balance = 0.0
    ndb = 1
    largest_frac = 1.0
    if n >= 6:
        from sklearn.cluster import KMeans, DBSCAN
        km1 = KMeans(1, n_init=1, random_state=0).fit(pts)
        km2 = KMeans(2, n_init=2, random_state=0).fit(pts)
        i1, i2 = km1.inertia_, km2.inertia_
        bimod = float(1.0 - i2 / i1) if i1 > 0 else 0.0
        c0, c1 = km2.cluster_centers_
        center_gap = float(np.linalg.norm(c0 - c1) / (extent + 1.0))
        lab = km2.labels_
        n0, n1 = int((lab == 0).sum()), int((lab == 1).sum())
        balance = min(n0, n1) / max(n0, n1)
        # DBSCAN connected blobs at a fixed physical scale
        db = DBSCAN(eps=5000.0, min_samples=3).fit(pts)
        labs = db.labels_[db.labels_ >= 0]
        if len(labs):
            counts = np.bincount(labs)
            ndb = len(counts)
            largest_frac = float(counts.max() / counts.sum())
    # soma proxy: number of well-separated high-local-density peaks
    return np.array([
        np.log1p(n), np.log1p(extent),
        np.sqrt(evals[0]) / (extent + 1.0), np.sqrt(evals[1]) / (extent + 1.0),
        np.sqrt(evals[2]) / (extent + 1.0),
        bimod, center_gap, balance, np.log1p(ndb), largest_frac,
    ])


SIZE_COLS = [0]                 # log n synapses
SHAPE_COLS = list(range(2, 10))  # everything except size + extent
FULL = list(range(10))


def _grouped_auc(X, y, groups, cols, n_splits, n_perm, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xc = X[:, cols]
    ng = len(np.unique(groups))
    gkf = GroupKFold(n_splits=max(2, min(n_splits, ng)))
    out = {}
    for name, mk in {
        "logreg": lambda: make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=2000, class_weight="balanced")),
        "rf": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                             class_weight="balanced", random_state=seed, n_jobs=-1),
    }.items():
        oof = np.full(len(y), np.nan)
        for tr, te in gkf.split(Xc, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            m = mk(); m.fit(Xc[tr], y[tr]); oof[te] = m.predict_proba(Xc[te])[:, 1]
        ok = ~np.isnan(oof)
        auc = roc_auc_score(y[ok], oof[ok])
        rng = np.random.default_rng(seed)
        null = [roc_auc_score(rng.permutation(y[ok]), oof[ok]) for _ in range(n_perm)]
        # precision/recall at a high-precision operating point (top 2% flagged)
        thr = np.quantile(oof[ok], 0.98)
        flagged = oof[ok] >= thr
        prec = (y[ok][flagged] == 1).mean() if flagged.sum() else float("nan")
        rec = (flagged & (y[ok] == 1)).sum() / max(1, (y[ok] == 1).sum())
        out[name] = (auc, float(np.mean(null)), float(np.std(null)), prec, rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--min-syn", type=int, default=8)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    comp = cell_components(tab)
    valid = tab.root_later > 0

    by_v117_pts: dict[int, list] = defaultdict(list)
    by_v117_later: dict[int, set] = defaultdict(set)
    for i in np.nonzero(valid)[0]:
        rv = int(tab.root_v117[i])
        by_v117_pts[rv].append(tab.pt[i])
        by_v117_later[rv].add(int(tab.root_later[i]))

    X, y, groups, sizes = [], [], [], []
    n_fm = n_clean = 0
    for rv, pl in by_v117_pts.items():
        if len(pl) < args.min_syn:
            continue
        pts = np.array(pl)
        is_fm = int(len(by_v117_later[rv]) >= 2)
        X.append(global_features(pts)); y.append(is_fm)
        groups.append(comp.get(rv, -1)); sizes.append(len(pl))
        n_fm += is_fm; n_clean += (1 - is_fm)
    X = np.array(X); y = np.array(y); groups = np.array(groups)
    base = y.mean()
    print(f"objects (v117 roots, >= {args.min_syn} syn) = {len(y):,}")
    print(f"  false merges = {n_fm}  clean = {n_clean}  base rate = {base:.2%}")
    print(f"  do-nothing: flags 0 -> recall 0.0, misses all {n_fm} merges by construction")
    print(f"  median size: merge={np.median(np.array(sizes)[y==1]):.0f}  "
          f"clean={np.median(np.array(sizes)[y==0]):.0f}  (size is a confound -> ablate)")

    print(f"\n  {'features':10s}{'model':8s}{'AUC':>7s}{'null':>13s}{'prec@2%':>9s}{'rec@2%':>8s}")
    for cols, name in ((SIZE_COLS, "size-only"), (SHAPE_COLS, "shape"), (FULL, "full")):
        res = _grouped_auc(X, y, groups, cols, args.cv_folds, args.n_perm, args.seed)
        for mdl, (auc, nm, ns, prec, rec) in res.items():
            print(f"  {name:10s}{mdl:8s}{auc:>7.3f}{nm:>8.2f}±{ns:.2f}{prec:>9.2f}{rec:>8.2f}")
    print("\n  shape>>size-only would mean global structure (not just 'it's big') flags merges.")
    print("  prec@2% is the do-nothing-relevant number: can we flag merges at usable precision?")
    print("  NOTE: synapse-cloud shape catches SEPARATED merges, misses INTERTWINED ones ->")
    print("        that residual is the case for skeleton topology (2 somas/2 trunks).")


if __name__ == "__main__":
    main()
