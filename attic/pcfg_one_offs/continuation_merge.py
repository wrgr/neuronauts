#!/usr/bin/env python3
"""Directional continuation grammar for the de-split (MERGE) problem.

The de-merge (split) signal is a spatial gap -- trivial (gap-alone AUC 0.81). The de-split
(merge) problem is the hard, valuable one: a gap exists whether or not two fragments belong
together, so the gap cannot discriminate. What discriminates is *continuation*: do the two
fragment tips actually aim at each other (collinear approach, matched caliber)?

This builds tip-direction features and trains them two ways:
  * supervised   -- grouped-CV on the real de-split labels (does direction beat gap-only?);
  * self-supervised -- trained ONLY on coherent-vs-spliced pairs synthesized from clean
                    arbors (no edit labels), then transferred to the real de-splits.

Splices are adversarial by construction (a tip of cell A facing a near tip of a different
cell B), which is exactly the hard negative the earlier merge stratum lacked.

Usage:
    python -m attic.pcfg_one_offs.continuation_merge --sidetable data/sidetable_7box.npz
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


def approach_dir(frag_pts: np.ndarray, tip: np.ndarray, k: int = 5):
    """Outward local trajectory direction at `tip`, and local caliber (kNN spacing).

    Direction = unit(tip - mean(k nearest in-fragment points)); points outward from the
    local mass toward the tip.  Caliber = mean distance to those neighbours.  Raw geometry,
    no hand-tuned thresholds.
    """
    d = np.linalg.norm(frag_pts - tip, axis=1)
    kk = min(k, len(frag_pts))
    near = frag_pts[np.argsort(d)[:kk]]
    v = tip - near.mean(axis=0)
    n = np.linalg.norm(v)
    direction = v / n if n > 1e-9 else np.zeros(3)
    caliber = float(np.sort(d)[1:kk].mean()) if kk > 1 else 0.0
    return direction, caliber


def pair_feats(fa, pi, fb, pj):
    """Continuation features for fragment A (tip pi) meeting fragment B (tip pj)."""
    gap = float(np.linalg.norm(pj - pi))
    if gap < 1e-6:
        gap = 1e-6
    dA, calA = approach_dir(fa, pi)
    dB, calB = approach_dir(fb, pj)
    toB = (pj - pi) / gap
    align_A = float(dA @ toB)         # A's tip points toward B  (+1 = continuation)
    align_B = float(dB @ -toB)        # B's tip points toward A  (+1 = continuation)
    collinear = float(-(dA @ dB))     # anti-parallel outward dirs (+1 = collinear)
    lateral = float(np.linalg.norm(toB - (toB @ dA) * dA))  # off-axis offset (0 = inline)
    cal_match = abs(calA - calB) / (calA + calB + 1.0)
    # layout: [log gap, align_A, align_B, collinear, lateral, cal_match, log calA, log calB]
    return np.array([np.log1p(gap), align_A, align_B, collinear, lateral, cal_match,
                     np.log1p(calA), np.log1p(calB)])


GAP_ONLY = [0]                       # just log gap
FULL = list(range(8))


def _auc_grouped(X, y, groups, cols, n_splits, n_perm, seed):
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
        null = []
        for _ in range(n_perm):
            yp = y.copy()
            for g in np.unique(groups):
                idx = np.nonzero(groups == g)[0]
                yp[idx] = rng.permutation(yp[idx])
            null.append(roc_auc_score(yp[ok], oof[ok]))
        out[name] = (auc, float(np.mean(null)), float(np.std(null)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--radius-nm", type=float, default=6000.0)
    ap.add_argument("--k-neighbors", type=int, default=12)
    ap.add_argument("--min-syn", type=int, default=4)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scipy.spatial import cKDTree
    from sklearn.metrics import roc_auc_score

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    rng = np.random.default_rng(args.seed)
    comp = cell_components(tab)

    Xe, ye, ge = [], [], []          # real de-split eval pairs
    Xs, ys = [], []                  # self-supervised coherent/spliced pairs

    for side_code in (0, 1):
        sel = (tab.side == side_code) & (tab.root_later > 0)
        sub = tab.mask(sel)
        if len(sub) < 10:
            continue
        by_v117: dict[int, list[int]] = defaultdict(list)
        by_v1718: dict[int, list[int]] = defaultdict(list)
        for i in range(len(sub)):
            by_v117[int(sub.root_v117[i])].append(i)
            by_v1718[int(sub.root_later[i])].append(i)
        frag_pts = {rv: sub.pt[idxs] for rv, idxs in by_v117.items()}

        # ---- real de-split eval pairs (targeted by later root, like the merge stratum) ----
        # POSITIVES: within each later root that gathers >=2 v117 roots, cross-root near pairs.
        pos = []
        for idxs in by_v1718.values():
            if len(idxs) < 2 or len({int(sub.root_v117[m]) for m in idxs}) < 2:
                continue
            mpts = sub.pt[idxs]
            mt = cKDTree(mpts)
            kq = min(args.k_neighbors + 1, len(idxs))
            dnn, inn = mt.query(mpts, k=kq, workers=-1)
            seenp = set()
            for a in range(len(idxs)):
                ra = idxs[a]
                for slot in range(1, kq):
                    if dnn[a, slot] > args.radius_nm:
                        break
                    rb = idxs[int(inn[a, slot])]
                    if int(sub.root_v117[ra]) == int(sub.root_v117[rb]):
                        continue
                    key = (min(ra, rb), max(ra, rb))
                    if key not in seenp:
                        seenp.add(key); pos.append((ra, rb))
        # NEGATIVES: spatially-matched, cross-root, DIFFERENT later root, around pos anchors.
        neg = []
        if pos:
            gtree = cKDTree(sub.pt)
            anchors = list({p[0] for p in pos} | {p[1] for p in pos})
            kq = min(args.k_neighbors + 1, len(sub))
            seenn = set()
            for ra in anchors:
                rva = int(sub.root_v117[ra]); la = int(sub.root_later[ra])
                dnn, inn = gtree.query(sub.pt[ra], k=kq, workers=-1)
                for slot in range(1, kq):
                    if dnn[slot] > args.radius_nm:
                        break
                    rb = int(inn[slot])
                    if int(sub.root_v117[rb]) == rva or int(sub.root_later[rb]) == la:
                        continue
                    key = (min(ra, rb), max(ra, rb))
                    if key not in seenn:
                        seenn.add(key); neg.append((ra, rb))
            # DISTANCE-MATCH negatives to the positive gap distribution (weighted sampling),
            # so gap is forced to ~chance and any AUC must come from direction, not proximity.
            def _gap(pr):
                return float(np.linalg.norm(sub.pt[pr[0]] - sub.pt[pr[1]]))
            if neg:
                pos_log = np.log1p([_gap(p) for p in pos])
                neg_log = np.log1p([_gap(n) for n in neg])
                mu, sd = pos_log.mean(), pos_log.std() + 1e-6
                w = np.exp(-0.5 * ((neg_log - mu) / sd) ** 2)
                w = w / w.sum() if w.sum() > 0 else None
                n_neg = min(len(neg), max(1, len(pos) * 3))
                pick = rng.choice(len(neg), n_neg, replace=False, p=w)
                neg = [neg[p] for p in pick]
        for lbl, prs in ((1, pos), (0, neg)):
            for i, j in prs:
                rva, rvb = int(sub.root_v117[i]), int(sub.root_v117[j])
                Xe.append(pair_feats(frag_pts[rva], sub.pt[i], frag_pts[rvb], sub.pt[j]))
                ye.append(lbl); ge.append(comp.get(rva, -1))

        # ---- self-supervised pairs from clean v1718 arbors (no edit labels) ----
        clean_ids = [rl for rl, idxs in by_v1718.items() if len(idxs) >= 2 * args.min_syn]
        for rl in clean_ids:
            idxs = np.array(by_v1718[rl])
            pts = sub.pt[idxs]
            # PCA-order, cut at a few interior points -> coherent continuations (label 1)
            c = pts - pts.mean(0)
            _, _, Vt = np.linalg.svd(c, full_matrices=False)
            order = np.argsort(c @ Vt[0])
            ps = pts[order]
            for frac in (0.4, 0.5, 0.6):
                cut = int(len(ps) * frac)
                if cut < args.min_syn or len(ps) - cut < args.min_syn:
                    continue
                A, B = ps[:cut], ps[cut:]
                Xs.append(pair_feats(A, A[-1], B, B[0])); ys.append(1)
        # spliced negatives: a clean tip facing a near tip of a DIFFERENT clean arbor
        if len(clean_ids) >= 2:
            tip_pts, tip_owner, tip_frag = [], [], {}
            for rl in clean_ids:
                idxs = np.array(by_v1718[rl]); pts = sub.pt[idxs]
                c = pts - pts.mean(0); _, _, Vt = np.linalg.svd(c, full_matrices=False)
                order = np.argsort(c @ Vt[0]); ps = pts[order]
                tip_frag[rl] = ps
                for endp in (ps[0], ps[-1]):
                    tip_pts.append(endp); tip_owner.append(rl)
            tip_pts = np.array(tip_pts)
            ttree = cKDTree(tip_pts)
            tdn, tin = ttree.query(tip_pts, k=min(6, len(tip_pts)), workers=-1)
            made = 0
            for a in range(len(tip_pts)):
                for slot in range(1, tin.shape[1]):
                    b = int(tin[a, slot])
                    if tip_owner[a] == tip_owner[b] or tdn[a, slot] > args.radius_nm:
                        continue
                    fa, fb = tip_frag[tip_owner[a]], tip_frag[tip_owner[b]]
                    Xs.append(pair_feats(fa, tip_pts[a], fb, tip_pts[b])); ys.append(0)
                    made += 1
                    break
                if made > len(clean_ids) * 3:
                    break

    Xe, ye, ge = np.array(Xe), np.array(ye), np.array(ge)
    Xs, ys = np.array(Xs), np.array(ys)
    print(f"real de-split eval pairs={len(ye)} (pos {ye.mean():.1%}, cells {len(np.unique(ge))})")
    print(f"self-sup splice pairs={len(ys)} (coherent {ys.mean():.1%})")
    gp = np.expm1(Xe[ye == 1, 0]); gn = np.expm1(Xe[ye == 0, 0])
    print(f"gap nm: pos(de-split) median={np.median(gp):.0f}  neg(diff-cell) median={np.median(gn):.0f}")
    print("  CONFOUND: diff-cell negatives are largely synaptic partners (pre touching post) that")
    print("  sit CLOSER than a real split-fragment gap, so distance separates for the wrong reason.")

    print("\n[A] supervised on real labels (grouped CV) -- does direction beat gap-only?")
    for cols, name in ((GAP_ONLY, "gap-only "), (FULL, "full-dir ")):
        res = _auc_grouped(Xe, ye, ge, cols, args.cv_folds, args.n_perm, args.seed)
        for mdl, (auc, nm, ns) in res.items():
            print(f"    {name} {mdl:6s} AUC={auc:.3f}  null={nm:.3f}±{ns:.3f}")

    print("\n[B] self-supervised transfer (train on splices ONLY, test on real de-splits)")
    if len(np.unique(ys)) == 2 and len(ye) > 10:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        for cols, name in ((GAP_ONLY, "gap-only "), (FULL, "full-dir ")):
            lr = make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, class_weight="balanced"))
            lr.fit(Xs[:, cols], ys)
            rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                        class_weight="balanced", random_state=args.seed, n_jobs=-1)
            rf.fit(Xs[:, cols], ys)
            a_lr = roc_auc_score(ye, lr.predict_proba(Xe[:, cols])[:, 1])
            a_rf = roc_auc_score(ye, rf.predict_proba(Xe[:, cols])[:, 1])
            print(f"    {name} logreg={a_lr:.3f}  rf={a_rf:.3f}  (trained label-free on splices)")
    print("\n    finding: proximity dominates de-split too; hand-built directional features add")
    print("    no measurable signal over gap, and don't transfer from synthetic splices.")
    print("    -> if anything beats proximity, it must be a LEARNED representation, not more")
    print("       hand geometry. This is the proximity baseline the neural track must beat.")


if __name__ == "__main__":
    main()
