#!/usr/bin/env python3
"""(b) Merge/JOIN direction: correct false-SPLITS by joining adjacent fragments.

A false-split = a v1718 cell broken into >=2 v117 roots. Correction = JOIN the fragments. This
is the intertwined second direction (the per-object grammar provably can't do it -- it's
relational). Candidate joins = spatially-adjacent cross-v117-root synapse pairs; a pair of
roots is joined (union-find) if the decision says same cell. Decisions:
  * oracle   -- join iff the two roots share a v1718 cell (ceiling).
  * learned  -- logistic regression on raw pair geometry (min gap, centroid gap, size ratio),
                grouped-by-cell CV; the de-split discriminator, gated to avoid over-merging.

Evaluated via conn_metric (its union-find scope captures cross-object joins): net pair-error
reduction = fixing "same-cell synapses split apart". #merges (joins) reported.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg_synapse_partitions.synapse_correction import SideTable, cell_components  # noqa: E402
from experiments.pcfg_synapse_partitions import conn_metric  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidetable", default="data/sidetable_7box.npz")
    ap.add_argument("--min-syn", type=int, default=8)
    ap.add_argument("--radius-nm", type=float, default=4000.0)
    ap.add_argument("--decision", choices=["oracle", "learned"], default="oracle")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from scipy.spatial import cKDTree

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    comp = cell_components(tab)
    valid = tab.root_later > 0
    rows_by = defaultdict(list)
    for i in np.nonzero(valid)[0]:
        rows_by[int(tab.root_v117[i])].append(int(i))
    roots = {rv: r for rv, r in rows_by.items() if len(r) >= args.min_syn}
    maj = {rv: Counter(tab.root_later[r].tolist()).most_common(1)[0][0] for rv, r in roots.items()}
    cent = {rv: tab.pt[r].mean(0) for rv, r in roots.items()}

    # candidate join pairs: spatially-adjacent cross-v117-root synapses
    all_rows = [i for rv in roots for i in roots[rv]]
    pos = tab.pt[all_rows]; owner = np.array([tab.root_v117[i] for i in all_rows])
    tree = cKDTree(pos)
    pairs = {}
    kq = min(8, len(all_rows))
    dnn, inn = tree.query(pos, k=kq, workers=-1)
    for a in range(len(all_rows)):
        ra = int(owner[a])
        for s in range(1, kq):
            if dnn[a, s] > args.radius_nm:
                break
            rb = int(owner[int(inn[a, s])])
            if rb == ra:
                continue
            key = (min(ra, rb), max(ra, rb))
            if key not in pairs or dnn[a, s] < pairs[key]:
                pairs[key] = float(dnn[a, s])
    cand = list(pairs.keys())
    y = np.array([int(maj[a] == maj[b]) for a, b in cand])
    print(f"join candidates = {len(cand)} root-pairs  ({y.mean():.1%} truly same cell)")

    # decide which pairs to join
    if args.decision == "oracle":
        join = y.astype(bool)
    else:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X = np.array([[np.log1p(pairs[(a, b)]),
                       np.log1p(np.linalg.norm(cent[a] - cent[b])),
                       np.log1p(min(len(roots[a]), len(roots[b]))),
                       abs(len(roots[a]) - len(roots[b])) / (len(roots[a]) + len(roots[b]))]
                      for a, b in cand])
        grp = np.array([comp.get(a, -1) for a, b in cand])
        oof = np.full(len(cand), np.nan)
        gkf = GroupKFold(n_splits=max(2, min(args.folds, len(np.unique(grp)))))
        for tr, te in gkf.split(X, y, grp):
            if len(np.unique(y[tr])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
            m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
        from sklearn.metrics import roc_auc_score
        ok = ~np.isnan(oof)
        print(f"  learned join AUC = {roc_auc_score(y[ok], oof[ok]):.3f}")
        join = np.zeros(len(cand), bool)
        join[ok] = oof[ok] >= 0.5

    # union-find the joined roots -> corrected partition
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for (a, b), j in zip(cand, join):
        if j:
            parent[find(a)] = find(b)
    corrected = tab.root_v117.copy()
    cellid = {}
    n_joins = sum(int(j) for j in join)
    for rv, r in roots.items():
        c = find(rv)
        cid = cellid.setdefault(c, int(tab.root_v117.max()) + 1 + len(cellid))
        for i in r:
            corrected[i] = cid
    used = [i for rv in roots for i in roots[rv]]
    print(f"\njoin corrector (decision={args.decision}): {n_joins} joins applied")
    conn_metric.evaluate(tab, corrected, used, n_splits=0, n_merges=n_joins)


if __name__ == "__main__":
    main()
