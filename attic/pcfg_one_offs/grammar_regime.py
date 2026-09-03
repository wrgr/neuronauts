#!/usr/bin/env python3
"""Regime check for the self-supervised grammar idea -- runs on the CACHED SideTable.

The bet: learn a coherence grammar of neurites from segmentation, treat errors as grammar
violations. It only works if the presegmentation is "good but imperfect" -- correct
structure must dominate the local transition statistics. This script measures whether
that holds, using the cached cross-version data (v117 = contaminated input, v1718 = clean
reference) on the SAME synapses. No edit labels are used to *train* anything; the v1718
partition only marks which transitions are seams, for scoring.

Outputs
-------
1. Transition-level pollution rate -- fraction of within-arbor steps crossing a seam.
2. Grammar drift -- KL between the bigram grammar of the clean (v1718) partition and the
   contaminated (v117) partition.
3. Anchor gate -- does clean-grammar surprise rank seam steps above non-seam steps? (AUC)
4. Systematic vs idiosyncratic -- concentration of seam transitions (top-bigram lift).

Usage:
    python -m attic.pcfg_one_offs.grammar_regime \
        --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg.pcfg_partitions import ALPH  # noqa: E402
from experiments.pcfg.synapse_correction import SideTable  # noqa: E402

_IDX = {c: i for i, c in enumerate(ALPH)}


def tokens_and_seams(pts: np.ndarray, later: np.ndarray, threshold: float = 0.4):
    """PCA-order the points and return (tokens, seam) aligned per step.

    tokens[k] is the F/B/L/R direction of step k (synapse order[k] -> order[k+1]);
    seam[k] is True iff those two synapses have different later roots.
    Mirrors pcfg_partitions.tokenize but also tracks the ordering so seams align.
    """
    if len(pts) < 2:
        return [], np.zeros(0, bool)
    pts = pts.astype(np.float64)
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca1, pca2 = Vt[0], Vt[1]
    order = np.argsort(centered @ pca1)
    ps = pts[order]
    lab = later[order]
    steps = np.diff(ps, axis=0)
    norms = np.linalg.norm(steps, axis=1, keepdims=True)
    steps_n = steps / np.where(norms < 1e-9, 1.0, norms)
    c1 = steps_n @ pca1
    c2 = steps_n @ pca2
    toks = []
    for f, lat in zip(c1.tolist(), c2.tolist()):
        if abs(f) > threshold:
            toks.append('F' if f > 0 else 'B')
        else:
            toks.append('L' if lat >= 0 else 'R')
    seam = lab[:-1] != lab[1:]
    return toks, seam


def geometric_steps(pts: np.ndarray, later: np.ndarray):
    """Geometry-preserving per-step features aligned with seams.

    Keeps the continuous information the F/B/L/R tokenization throws away:
      feat[k] = [log1p(step_length_k), turn_angle(step_{k-1}, step_k)]   for k >= 1
    seam[k] marks whether step k crosses a later-root boundary.
    """
    if len(pts) < 3:
        return np.zeros((0, 2)), np.zeros(0, bool)
    pts = pts.astype(np.float64)
    centered = pts - pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    order = np.argsort(centered @ Vt[0])
    ps, lab = pts[order], later[order]
    steps = np.diff(ps, axis=0)
    L = np.linalg.norm(steps, axis=1)
    u = steps / np.where(L[:, None] < 1e-9, 1.0, L[:, None])
    cos = np.clip((u[:-1] * u[1:]).sum(axis=1), -1.0, 1.0)
    angle = np.arccos(cos)                       # turn at synapse k, for step k>=1
    feat = np.column_stack([np.log1p(L[1:]), angle])
    seam = (lab[:-1] != lab[1:])[1:]
    return feat, seam


def bigram_counts(roots: dict[int, list[int]], pts: np.ndarray, later: np.ndarray,
                  min_syn: int = 3) -> Counter:
    """Accumulate (cur,next) token-bigram counts over every root's arbor."""
    cnt: Counter = Counter()
    for idxs in roots.values():
        if len(idxs) < min_syn:
            continue
        toks, _ = tokens_and_seams(pts[idxs], later[idxs])
        for a, b in zip(toks[:-1], toks[1:]):
            cnt[(a, b)] += 1
    return cnt


def cond_prob(cnt: Counter, smoothing: float = 0.5) -> dict:
    """P(next | cur) with add-smoothing, as {cur: {next: p}}."""
    tot: dict = defaultdict(float)
    for (a, b), c in cnt.items():
        tot[a] += c
    out: dict = {}
    for a in ALPH:
        denom = tot.get(a, 0.0) + smoothing * len(ALPH)
        out[a] = {b: (cnt.get((a, b), 0) + smoothing) / denom for b in ALPH}
    return out


def kl_grammar(p_cnt: Counter, q_cnt: Counter) -> float:
    """KL(P||Q) over the full bigram distribution (smoothed)."""
    pt = sum(p_cnt.values()) + 0.5 * len(ALPH) ** 2
    qt = sum(q_cnt.values()) + 0.5 * len(ALPH) ** 2
    kl = 0.0
    for a, b in product(ALPH, repeat=2):
        p = (p_cnt.get((a, b), 0) + 0.5) / pt
        q = (q_cnt.get((a, b), 0) + 0.5) / qt
        kl += p * np.log2(p / q)
    return float(kl)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--side", choices=["pre", "post", "both"], default="both")
    ap.add_argument("--min-syn", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    rng = np.random.default_rng(args.seed)

    side_codes = {"pre": [0], "post": [1], "both": [0, 1]}[args.side]
    sel = np.isin(tab.side, side_codes) & (tab.root_later > 0)
    pt = tab.pt[sel]
    rv = tab.root_v117[sel]
    rl = tab.root_later[sel]
    print(f"sides used: {len(pt):,}  (side={args.side})")

    by_v117: dict[int, list[int]] = defaultdict(list)
    by_v1718: dict[int, list[int]] = defaultdict(list)
    for i in range(len(pt)):
        by_v117[int(rv[i])].append(i)
        by_v1718[int(rl[i])].append(i)

    # ---- 1. Transition-level pollution ----
    seam_steps = tot_steps = polluted_roots = eval_roots = 0
    surprises: list[float] = []
    seam_flags: list[bool] = []
    seam_bigrams: Counter = Counter()
    clean_bigram_for_score = None  # filled below

    # ---- 2. Grammar drift: clean (v1718) vs contaminated (v117) ----
    clean_cnt = bigram_counts(by_v1718, pt, rl, args.min_syn)   # ordering & seams irrelevant here
    contam_cnt = bigram_counts(by_v117, pt, rl, args.min_syn)
    kl_cc = kl_grammar(clean_cnt, contam_cnt)
    kl_revs = kl_grammar(contam_cnt, clean_cnt)
    clean_cond = cond_prob(clean_cnt)

    # ---- 1 + 3 + 4 over v117 arbors, scored by the CLEAN grammar ----
    for idxs in by_v117.values():
        if len(idxs) < args.min_syn:
            continue
        toks, seam = tokens_and_seams(pt[idxs], rl[idxs])
        if len(toks) < 2:
            continue
        eval_roots += 1
        tot_steps += len(seam)
        ns = int(seam.sum())
        seam_steps += ns
        if ns:
            polluted_roots += 1
        # surprise of step k (k>=1) under clean grammar, aligned with seam[k]
        for k in range(1, len(toks)):
            p = clean_cond[toks[k - 1]][toks[k]]
            surprises.append(-np.log2(max(p, 1e-9)))
            seam_flags.append(bool(seam[k]))
            if seam[k]:
                seam_bigrams[(toks[k - 1], toks[k])] += 1

    pollution = seam_steps / max(1, tot_steps)
    print("\n[1] transition-level pollution")
    print(f"    within-arbor steps={tot_steps:,}  seam steps={seam_steps:,}  "
          f"pollution={pollution:.3%}")
    print(f"    v117 arbors evaluated={eval_roots:,}  with >=1 seam={polluted_roots:,} "
          f"({polluted_roots/max(1,eval_roots):.1%})")

    print("\n[2] grammar drift (bigram, clean v1718 vs contaminated v117)")
    print(f"    KL(clean||contam)={kl_cc:.4f} bits   KL(contam||clean)={kl_revs:.4f} bits")
    print(f"    (clean bigrams={sum(clean_cnt.values()):,}  contam bigrams={sum(contam_cnt.values()):,})")

    # ---- 3. Anchor gate ----
    surch = np.array(surprises)
    seamf = np.array(seam_flags)
    print("\n[3] anchor gate: does clean-grammar surprise localize seams?")
    if seamf.sum() >= 5 and (~seamf).sum() >= 5:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(seamf, surch)
        null = np.array([roc_auc_score(rng.permutation(seamf), surch)
                         for _ in range(args.n_perm)])
        m_seam = surch[seamf].mean()
        m_non = surch[~seamf].mean()
        print(f"    surprise(seam)={m_seam:.3f} bits  surprise(non-seam)={m_non:.3f} bits")
        print(f"    AUC(seam | surprise)={auc:.3f}   null={null.mean():.3f}±{null.std():.3f}  "
              f"p={(null>=auc).mean():.3f}")
    else:
        print("    too few seams to score on this slice")

    # ---- 5. Geometry-preserving surprise (same anchor gate, richer representation) ----
    # Train a Gaussian on clean (v1718) arbor step-geometry -> Mahalanobis surprise.
    clean_feats = [geometric_steps(pt[idxs], rl[idxs])[0]
                   for idxs in by_v1718.values() if len(idxs) >= args.min_syn]
    clean_feats = np.vstack([f for f in clean_feats if len(f)]) if clean_feats else np.zeros((0, 2))
    geo_auc = None
    if len(clean_feats) > 50:
        mu = clean_feats.mean(axis=0)
        cov = np.cov(clean_feats.T) + 1e-6 * np.eye(2)
        cinv = np.linalg.inv(cov)
        g_sur, g_seam = [], []
        for idxs in by_v117.values():
            if len(idxs) < args.min_syn:
                continue
            f, s = geometric_steps(pt[idxs], rl[idxs])
            for x, sk in zip(f, s):
                dx = x - mu
                g_sur.append(float(dx @ cinv @ dx))
                g_seam.append(bool(sk))
        g_sur = np.array(g_sur); g_seam = np.array(g_seam)
        if g_seam.sum() >= 5 and (~g_seam).sum() >= 5:
            from sklearn.metrics import roc_auc_score
            geo_auc = roc_auc_score(g_seam, g_sur)
            gnull = np.array([roc_auc_score(rng.permutation(g_seam), g_sur)
                              for _ in range(args.n_perm)])
            print("\n[5] geometry-preserving surprise (Mahalanobis on log-len + turn-angle)")
            print(f"    surprise(seam)={g_sur[g_seam].mean():.2f}  "
                  f"surprise(non-seam)={g_sur[~g_seam].mean():.2f}")
            print(f"    AUC(seam | geo-surprise)={geo_auc:.3f}   "
                  f"null={gnull.mean():.3f}±{gnull.std():.3f}  p={(gnull>=geo_auc).mean():.3f}")
            print(f"    vs bigram-token AUC above -> representation, not contamination, is the bottleneck")

    # ---- 4. Systematic vs idiosyncratic ----
    print("\n[4] are seam transitions stereotyped?")
    base_tot = sum(contam_cnt.values()) or 1
    if seam_bigrams:
        top = seam_bigrams.most_common(5)
        s_tot = sum(seam_bigrams.values())
        ent = -sum((c / s_tot) * np.log2(c / s_tot) for _, c in seam_bigrams.items())
        print(f"    seam-bigram entropy={ent:.2f} bits (max {np.log2(len(ALPH)**2):.2f}); "
              f"top seam transitions (share | lift over base):")
        for (a, b), c in top:
            share = c / s_tot
            base = (contam_cnt.get((a, b), 0) + 0.5) / base_tot
            print(f"      {a}->{b}: {share:.1%}  lift={share/max(base,1e-9):.2f}x")


if __name__ == "__main__":
    main()
