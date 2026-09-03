#!/usr/bin/env python3
"""Richer self-supervised geometric grammar -- climb the label-free anchor gate.

grammar_regime.py showed: contamination is free (KL~0) but the F/B/L/R token grammar is
blind to seams (AUC 0.54); a trivial 2-D Gaussian on (log-len, turn-angle) gets 0.66. This
pushes the SAME idea -- learn "what a coherent neurite junction looks like" from raw arbors,
score errors as surprise -- with a richer local feature set and a nonparametric (kNN)
density. Still label-free in training (the v1718 partition only marks seams for scoring).

Trained on clean (v1718) arbor junctions, scored on v117 arbor junctions; report
AUC(seam | surprise) vs the supervised ceiling (0.85) and the earlier baselines.

Usage:
    python -m attic.pcfg_one_offs.selfsup_grammar --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg.synapse_correction import SideTable  # noqa: E402


def junction_features(pts: np.ndarray, later: np.ndarray):
    """Per-junction geometric features (PCA-ordered arbor), aligned with seams.

    At interior synapse k (between step k-1 and step k):
      [ log1p(len_before), log1p(len_after), turn_angle,
        log len ratio, log1p(dist to nearest non-adjacent synapse) ]
    seam[k] = step k crosses a later-root boundary.  All features are raw geometry
    (no labels); seam is only returned for scoring.
    """
    n = len(pts)
    if n < 4:
        return np.zeros((0, 5)), np.zeros(0, bool)
    pts = pts.astype(np.float64)
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    order = np.argsort(centered @ Vt[0])
    ps, lab = pts[order], later[order]
    steps = np.diff(ps, axis=0)
    L = np.linalg.norm(steps, axis=1)
    u = steps / np.where(L[:, None] < 1e-9, 1.0, L[:, None])
    # nearest non-adjacent synapse distance (local crowding), per ordered synapse
    dall = np.linalg.norm(ps[:, None, :] - ps[None, :, :], axis=-1)
    for k in range(n):
        for j in (k - 1, k, k + 1):
            if 0 <= j < n:
                dall[k, j] = np.inf
    nnd = dall.min(axis=1)
    feats, seams = [], []
    for k in range(1, n - 1):                      # interior synapses only
        ang = float(np.arccos(np.clip(u[k - 1] @ u[k], -1.0, 1.0)))
        feats.append([
            np.log1p(L[k - 1]), np.log1p(L[k]), ang,
            np.log((L[k] + 1.0) / (L[k - 1] + 1.0)),
            np.log1p(nnd[k]),
        ])
        seams.append(bool(lab[k] != lab[k + 1]))   # step k after synapse k
    return np.array(feats), np.array(seams, bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--side", choices=["pre", "post", "both"], default="both")
    ap.add_argument("--min-syn", type=int, default=4)
    ap.add_argument("--k", type=int, default=16, help="kNN for density surprise")
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    rng = np.random.default_rng(args.seed)
    side_codes = {"pre": [0], "post": [1], "both": [0, 1]}[args.side]
    sel = np.isin(tab.side, side_codes) & (tab.root_later > 0)
    pt, rv, rl = tab.pt[sel], tab.root_v117[sel], tab.root_later[sel]

    by_v117: dict[int, list[int]] = defaultdict(list)
    by_v1718: dict[int, list[int]] = defaultdict(list)
    for i in range(len(pt)):
        by_v117[int(rv[i])].append(i)
        by_v1718[int(rl[i])].append(i)

    # clean (v1718) junctions -> training set for "normal" geometry (label-free)
    clean = [junction_features(pt[idxs], rl[idxs])[0]
             for idxs in by_v1718.values() if len(idxs) >= args.min_syn]
    clean = np.vstack([c for c in clean if len(c)])
    # v117 junctions -> scored; seams from v1718 only mark which to evaluate
    qf, qs = [], []
    for idxs in by_v117.values():
        if len(idxs) >= args.min_syn:
            f, s = junction_features(pt[idxs], rl[idxs])
            if len(f):
                qf.append(f); qs.append(s)
    qf = np.vstack(qf); qs = np.concatenate(qs)
    print(f"clean junctions={len(clean):,}  scored junctions={len(qf):,}  "
          f"seams={int(qs.sum()):,} ({qs.mean():.2%})")

    # standardize on clean stats, then kNN-density surprise (mean dist to k clean neighbours)
    mu, sd = clean.mean(0), clean.std(0) + 1e-9
    cz, qz = (clean - mu) / sd, (qf - mu) / sd
    from scipy.spatial import cKDTree
    from sklearn.metrics import roc_auc_score
    tree = cKDTree(cz)
    dk, _ = tree.query(qz, k=args.k, workers=-1)
    surprise = dk.mean(axis=1)

    auc = roc_auc_score(qs, surprise)
    null = np.array([roc_auc_score(rng.permutation(qs), surprise) for _ in range(args.n_perm)])
    # single-feature reference: gap-after alone
    auc_gap = roc_auc_score(qs, qf[:, 1])
    print("\nself-supervised geometric grammar (kNN density, 5 features)")
    print(f"    surprise(seam)={surprise[qs].mean():.3f}  surprise(non-seam)={surprise[~qs].mean():.3f}")
    print(f"    AUC(seam | kNN-surprise) = {auc:.3f}   null={null.mean():.3f}±{null.std():.3f}  "
          f"p={(null>=auc).mean():.3f}")
    print(f"    reference: gap-after alone={auc_gap:.3f} | bigram-token=0.539 | "
          f"2-D Gaussian=0.658 | supervised ceiling=0.85")


if __name__ == "__main__":
    main()
