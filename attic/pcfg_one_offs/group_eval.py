#!/usr/bin/env python3
"""Group-level, do-nothing-protected evaluation of the correction model.

Pairwise AUC is misleading here: corrections are ~1% of decisions, so a model that
proposes ZERO edits ("do nothing") scores ~99% accuracy and a deceptively fine AUC. This
script evaluates at the synapse-group level and scores every model against the do-nothing
baseline, which by construction has edit-recall 0.

For the decision-relevant pairs (within-root = split candidates, cross-root = merge
candidates), built and grouped-by-cell exactly as the corrector sees them:

  * counts: how many real splits and merges are needed.
  * do-nothing baseline: leave v117 unchanged -> fixes 0 edits, makes N errors.
  * model: at a threshold, how many edits it proposes, edit PRECISION and RECALL, and the
    NET change in partition errors vs do-nothing (positive = actually better than nothing).

Usage:
    python -m attic.pcfg_one_offs.group_eval --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg.synapse_correction import (  # noqa: E402
    SideTable,
    build_correction_pairs,
    summarize_edits,
)
def _oof_single(X, y, groups, model, n_splits, seed):
    """Out-of-fold P(same) for ONE model (avoids training the unused one)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ng = len(np.unique(groups))
    gkf = GroupKFold(n_splits=max(2, min(n_splits, ng)))
    mk = (lambda: make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced"))) \
        if model == "logreg" else \
        (lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                        class_weight="balanced", random_state=seed, n_jobs=-1))
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = mk(); m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    print("edit events:", summarize_edits(tab))

    X, y, groups, strata = build_correction_pairs(tab, rng=np.random.default_rng(args.seed))
    # do-nothing prediction per pair: same-cell iff same v117 root (within-root stratum)
    do_same = (strata == 0).astype(int)
    is_edit = (y != do_same)                     # the pairs do-nothing gets wrong
    n_split = int(((strata == 0) & (y == 0)).sum())   # within-root, truly different -> CUT
    n_merge = int(((strata == 1) & (y == 1)).sum())   # cross-root, truly same      -> JOIN
    print(f"\ndecision-relevant pairs={len(y):,}  grouped cells={len(np.unique(groups))}")
    print(f"  splits needed (cut)  = {n_split:,}")
    print(f"  merges needed (join) = {n_merge:,}")
    print(f"  do-nothing baseline: fixes 0/{int(is_edit.sum()):,} edits, "
          f"makes {int(is_edit.sum()):,} partition errors (recall 0.0 by construction)")

    oof = _oof_single(X, y, groups, args.model, args.cv_folds, args.seed)
    ok = ~np.isnan(oof)

    print(f"\nmodel={args.model}, grouped-CV out-of-fold, vs do-nothing:")
    print(f"  {'thr':>5s}{'edits':>9s}{'eprec':>8s}{'erecall':>9s}"
          f"{'split_rec':>11s}{'merge_rec':>11s}{'net_fixed':>11s}")
    base_err = int(is_edit[ok].sum())
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        pred_same = (oof >= t).astype(int)
        proposed = ok & (pred_same != do_same)             # disagrees with do-nothing
        correct = proposed & (pred_same == y)
        n_prop = int(proposed.sum())
        eprec = correct.sum() / n_prop if n_prop else float("nan")
        erecall = correct.sum() / max(1, base_err)
        sm = ok & (strata == 0)
        mm = ok & (strata == 1)
        srec = (proposed & (pred_same == y) & (y == 0))[sm].sum() / max(1, (is_edit & sm).sum())
        mrec = (proposed & (pred_same == y) & (y == 1))[mm].sum() / max(1, (is_edit & mm).sum())
        model_err = int((pred_same[ok] != y[ok]).sum())
        net = base_err - model_err                         # >0 => better than do-nothing
        print(f"  {t:>5.2f}{n_prop:>9d}{eprec:>8.2f}{erecall:>9.2f}"
              f"{srec:>11.2f}{mrec:>11.2f}{net:>+11d}")
    print("\n  net_fixed > 0 means the corrected partition has fewer errors than doing nothing.")
    print("  A model that only chases AUC can still sit at net<=0 here -- that's the guardrail.")


if __name__ == "__main__":
    main()
