#!/usr/bin/env python3
"""Close the loop: global whole-object detector -> CUT -> net partition improvement vs do-nothing.

The do-nothing guardrail voided the pairwise corrector (net-negative). The strongest detector
is the global whole-object one (flag an over-merged v117 root from its shape, ~0.88 AUC /
41% precision). This turns that flag into a real correction and asks the only question that
matters: does flagging + cutting beat doing nothing, at the synapse-partition level?

Loop:
  1. score each v117 object (>=min-syn synapses) with the global detector (grouped-CV OOF).
  2. at a flag threshold, CUT each flagged object into 2 groups (2-means on its synapses).
  3. corrected partition = v117, except flagged objects replaced by their 2 cut subgroups.
  4. score vs v1718 truth with Rand-disagreement pair counting (fixing AND breaking both count),
     summed over within-object synapse pairs. do-nothing = leave v117 (net 0 by definition).

net_fixed = do_nothing_pair_errors - corrected_pair_errors  (>0 == actually better than nothing).
Merge direction only; the split/join half needs the relational model.

    python -m attic.pcfg_one_offs.close_loop_merge --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg.synapse_correction import SideTable, cell_components  # noqa: E402
from experiments.pcfg.global_shape_merge import global_features  # noqa: E402


def _pairs(n):
    return n * (n - 1) // 2


def disagreement(truth, pred):
    """# within-object synapse pairs where (same pred) != (same truth) -- Rand disagreement."""
    from collections import Counter
    n = len(truth)
    same_t = sum(_pairs(c) for c in Counter(truth).values())
    same_p = sum(_pairs(c) for c in Counter(pred).values())
    cont = Counter(zip(truth, pred))
    same_b = sum(_pairs(c) for c in cont.values())
    return same_t + same_p - 2 * same_b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--min-syn", type=int, default=8)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    comp = cell_components(tab)
    valid = tab.root_later > 0

    pts_by, later_by = defaultdict(list), defaultdict(list)
    for i in np.nonzero(valid)[0]:
        rv = int(tab.root_v117[i])
        pts_by[rv].append(tab.pt[i]); later_by[rv].append(int(tab.root_later[i]))

    rids, X, y, groups, objpts, objlater = [], [], [], [], [], []
    for rv, pl in pts_by.items():
        if len(pl) < args.min_syn:
            continue
        P = np.asarray(pl); lat = np.asarray(later_by[rv])
        rids.append(rv); X.append(global_features(P)); objpts.append(P); objlater.append(lat)
        y.append(int(len(set(lat.tolist())) >= 2)); groups.append(comp.get(rv, -1))
    X = np.array(X); y = np.array(y); groups = np.array(groups)
    print(f"objects={len(y)}  real merges={int(y.sum())} ({y.mean():.1%})  cells={len(np.unique(groups))}")

    # grouped-CV out-of-fold merge probability (RF -- best precision in the detector study)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=max(2, min(args.cv_folds, len(np.unique(groups)))))
    prob = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced",
                                   random_state=args.seed, n_jobs=-1).fit(X[tr], y[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(prob)

    # precompute do-nothing error (all-same) and the best-possible 2-cut error per object
    from sklearn.cluster import KMeans
    do_nothing_err = np.array([disagreement(objlater[i], np.zeros(len(objlater[i]), int))
                               for i in range(len(y))])
    cut_lab = [None] * len(y)
    cut_err = do_nothing_err.copy()
    for i in range(len(y)):
        P = objpts[i]
        if len(P) >= 2:
            lab = KMeans(2, n_init=2, random_state=args.seed).fit_predict(P)
            cut_lab[i] = lab
            cut_err[i] = disagreement(objlater[i], lab)

    base = int(do_nothing_err[ok].sum())
    print(f"\ndo-nothing within-object pair errors (merges to cut) = {base:,}")
    print(f"  {'flag thr':>9s}{'flagged':>9s}{'true+':>7s}{'false+':>8s}"
          f"{'net_fixed':>11s}{'%ofbase':>9s}")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        flagged = ok & (prob >= thr)
        # corrected error: flagged -> cut error; else do-nothing error
        corr = np.where(flagged, cut_err, do_nothing_err)
        net = int(do_nothing_err[ok].sum() - corr[ok].sum())
        nflag = int(flagged.sum())
        tp = int((flagged & (y == 1)).sum()); fp = int((flagged & (y == 0)).sum())
        print(f"  {thr:>9.2f}{nflag:>9d}{tp:>7d}{fp:>8d}{net:>+11d}{100*net/max(1,base):>8.1f}%")
    # oracle: cut every real merge with the best 2-means cut (ceiling of this cut operator)
    orc = np.where(ok & (y == 1), cut_err, do_nothing_err)
    onet = int(do_nothing_err[ok].sum() - orc[ok].sum())
    print(f"\n  oracle (cut every TRUE merge, 2-means): net_fixed={onet:+d} "
          f"({100*onet/max(1,base):.1f}% of base) -- ceiling of the 2-means cut operator")
    print("  net_fixed>0 == the global detector + cut beats do-nothing on the partition.")


if __name__ == "__main__":
    main()
