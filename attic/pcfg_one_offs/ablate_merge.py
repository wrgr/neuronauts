#!/usr/bin/env python3
"""Robustness battery for the synapse-level correction model -- is the merge-stratum
signal real morphology, or a mechanical/ordering artifact?

The RF-vs-logreg split on the merge stratum (RF 0.98, logreg 0.34) smells like a leaky
cue rather than learned grammar (cf. the berlin cell-identity-leakage retraction). This
script loads a cached SideTable, rebuilds the pairs, and re-scores each stratum under:

  * full           -- all features
  * no-grammar     -- zero out the 2x17 grammar dims (keep geometry + sizes + dist)
  * geom-only      -- distance + axial/lateral/gap/density only
  * dist-only      -- single feature: log1p(distance)
  * size-only      -- log root sizes only
  * order-random   -- randomly swap A/B in every pair (kills any A/B asymmetry artifact)

A signal that survives *no-grammar* and *order-random* but collapses to ~chance under
dist-only/size-only is mechanical, not morphological.  Report per-stratum AUC vs the
permutation null for both LogReg and RandomForest.

Usage:
    python -m experiments.pcfg.ablate_merge --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg.synapse_correction import (  # noqa: E402
    FEAT_DIM,
    PAIR_DIM,
    SideTable,
    build_correction_pairs,
)
from experiments.pcfg.run_synapse_correction import evaluate  # noqa: E402

# Feature index map (see synapse_correction._pair_features / PAIR_DIM layout)
I_DIST = 0
I_SAME = 1
I_NA, I_NB = 2, 3
GRAMMAR_A = slice(4, 4 + FEAT_DIM)
GRAMMAR_B = slice(4 + FEAT_DIM, 4 + 2 * FEAT_DIM)
GEOM = slice(4 + 2 * FEAT_DIM, PAIR_DIM)  # axial, lateral, gap, densA, densB


def _swap_AB(X: np.ndarray) -> np.ndarray:
    """Swap the A/B halves of every pair feature vector (order-symmetry probe).

    Distance/same/gap are symmetric; na<->nb, grammarA<->grammarB swap; the geometry
    block is computed in A's frame so it is only approximately symmetric -- swapping it
    is a conservative perturbation that still removes any learned A/B ordering cue.
    """
    Xs = X.copy()
    Xs[:, I_NA], Xs[:, I_NB] = X[:, I_NB], X[:, I_NA]
    Xs[:, GRAMMAR_A], Xs[:, GRAMMAR_B] = X[:, GRAMMAR_B], X[:, GRAMMAR_A]
    return Xs


def _mask(X: np.ndarray, keep: list) -> np.ndarray:
    m = np.zeros(X.shape[1], bool)
    for k in keep:
        m[k] = True
    return X[:, m]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True, help="cached SideTable .npz")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    X, y, groups, strata = build_correction_pairs(tab, rng=np.random.default_rng(args.seed))
    print(f"pairs={len(y)}  merge={int((strata==1).sum())}  split={int((strata==0).sum())}")

    all_cols = list(range(X.shape[1]))
    geom_cols = list(range(GEOM.start, GEOM.stop))
    variants = {
        "full":        X,
        "no-grammar":  _mask(X, [I_DIST, I_SAME, I_NA, I_NB] + geom_cols),
        "geom-only":   _mask(X, [I_DIST] + geom_cols),
        "dist-only":   _mask(X, [I_DIST]),
        "size-only":   _mask(X, [I_NA, I_NB]),
        "order-random": None,  # special: swap A/B on a random half
    }

    rng = np.random.default_rng(args.seed)
    for name, Xv in variants.items():
        if name == "order-random":
            Xv = X.copy()
            flip = rng.random(len(Xv)) < 0.5
            Xv[flip] = _swap_AB(X[flip])
        print(f"\n===== variant: {name}  ({Xv.shape[1]} feats) =====")
        evaluate(Xv, y, groups, strata, n_splits=args.cv_folds,
                 n_perm=args.n_perm, seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()
