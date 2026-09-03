#!/usr/bin/env python3
"""Error-site hashing: is the local context around a merge SEAM a hashable cluster?

Sketch/prototype of a content-addressable error prior. For each v117 over-merged object we take
its oracle best-cut tree edge as the SEAM and label the seam vertex; every vertex gets a
pose-invariant, LABEL-FREE descriptor from v117 skeleton geometry + synapse sides (no v1718 --
that only defines the seam target for scoring). We binarize descriptors with SimHash (cosine LSH)
and ITQ, then ask, in grouped-by-cell CV: does a held-out seam context retrieve OTHER seam
contexts above base rate? precision@k / base_rate (lift) is the answer -- lift >> 1 means error
sites are hashable and the index is a usable prior.

    python -m experiments.pcfg.seam_hash --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg.synapse_correction import SideTable, cell_components  # noqa: E402
from experiments.pcfg.close_loop_cut import (  # noqa: E402
    load_skels, disagreement_from_counts, do_nothing_err, root_and_subtrees,
)


def _adj(n, E):
    a = defaultdict(list)
    for u, v in E:
        a[int(u)].append(int(v)); a[int(v)].append(int(u))
    return a


def _ball(adj, src, hops):
    """vertices within `hops` graph-steps of src (small geodesic ball)."""
    seen = {src}; frontier = [src]
    for _ in range(hops):
        nxt = []
        for u in frontier:
            for w in adj[u]:
                if w not in seen:
                    seen.add(w); nxt.append(w)
        frontier = nxt
    return seen


def descriptor(V, R, adj, v, pre_at, post_at, hops=4):
    """9-dim pose-invariant, label-free local context at vertex v."""
    nb = adj[v]
    deg = len(nb)
    logR = np.log1p(R[v])
    if deg:
        jumps = [abs(np.log1p(R[v]) - np.log1p(R[u])) for u in nb]
        caliber_jump = max(jumps)
        rr = [R[u] for u in nb]
        cross_ratio = (min(rr) + 1) / (max(rr) + 1)
    else:
        caliber_jump = 0.0; cross_ratio = 1.0
    # turn angle for degree-2 (collinearity); else neutral
    if deg == 2:
        d1 = V[nb[0]] - V[v]; d2 = V[nb[1]] - V[v]
        n1 = np.linalg.norm(d1); n2 = np.linalg.norm(d2)
        cos = float(np.dot(d1, d2) / (n1 * n2 + 1e-9)) if n1 > 0 and n2 > 0 else -1.0
        kink = 1.0 + cos            # 0 straight (cos=-1) -> 2 hairpin (cos=+1)
    else:
        kink = 0.5
    ball = _ball(adj, v, hops)
    pre = sum(pre_at.get(w, 0) for w in ball); post = sum(post_at.get(w, 0) for w in ball)
    dens = np.log1p(pre + post)
    prefrac = (pre + 0.5) / (pre + post + 1.0)
    # polarity contrast across v: split ball by nearest arm, compare pre-fraction of the two sides
    if deg >= 2:
        sideA = _ball(adj, nb[0], hops); sideB = _ball(adj, nb[1], hops)
        pa = sum(pre_at.get(w, 0) for w in sideA); qa = sum(post_at.get(w, 0) for w in sideA)
        pb = sum(pre_at.get(w, 0) for w in sideB); qb = sum(post_at.get(w, 0) for w in sideB)
        fa = (pa + 0.5) / (pa + qa + 1.0); fb = (pb + 0.5) / (pb + qb + 1.0)
        polarity_contrast = abs(fa - fb)
    else:
        polarity_contrast = 0.0
    # tortuosity over the ball: geodesic proxy (n vertices) vs euclidean spread
    pts = V[list(ball)]
    eucl = np.linalg.norm(pts.max(0) - pts.min(0)) + 1.0
    tort = np.log1p(len(ball)) / np.log1p(eucl / 1000.0 + 1.0)
    return np.array([logR, caliber_jump, cross_ratio, kink, min(deg, 5),
                     dens, prefrac, polarity_contrast, tort], np.float64)


def build_sites(sidetable, skel_dir, min_syn, hops, seam_tol):
    from scipy.spatial import cKDTree
    d = np.load(sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    comp = cell_components(tab)
    valid = tab.root_later > 0
    pts_by, lat_by, side_by = defaultdict(list), defaultdict(list), defaultdict(list)
    for i in np.nonzero(valid)[0]:
        rv = int(tab.root_v117[i])
        pts_by[rv].append(tab.pt[i]); lat_by[rv].append(int(tab.root_later[i]))
        side_by[rv].append(int(tab.side[i]))
    skels = load_skels(skel_dir, 117)
    X, y, grp = [], [], []
    n_obj = 0
    for rv, (V, E, R) in skels.items():
        if rv not in pts_by:
            continue
        P = np.asarray(pts_by[rv]); lat = np.asarray(lat_by[rv]); sd = np.asarray(side_by[rv])
        if len(P) < min_syn or len(set(lat.tolist())) < 2:
            continue
        labs, lab_index = np.unique(lat, return_inverse=True)
        nlab = len(labs); tot = np.bincount(lab_index, minlength=nlab).astype(np.int64)
        if do_nothing_err(tot) == 0:
            continue
        syn_vert = cKDTree(V).query(P)[1]
        rs = root_and_subtrees(V, E, list(zip(syn_vert.tolist(), lab_index.tolist())), nlab)
        if rs is None:
            continue
        parent, order, sub = rs
        # oracle seam = tree edge (parent[v],v) minimizing disagreement
        best_v, best_dis = None, None
        for v in order:
            p = parent[v]
            if p < 0:
                continue
            s = sub[v]; A = int(s.sum()); B = int(tot.sum()) - A
            if min(A, B) < 2:
                continue
            dis = disagreement_from_counts(s, tot)
            if best_dis is None or dis < best_dis:
                best_dis = dis; best_v = v
        if best_v is None:
            continue
        adj = _adj(len(V), E)
        pre_at, post_at = defaultdict(int), defaultdict(int)
        for vv, s in zip(syn_vert.tolist(), sd.tolist()):
            (pre_at if s == 0 else post_at)[vv] += 1
        # seam vertices: best_v and any vertex within seam_tol hops of it
        seam_set = _ball(adj, int(best_v), seam_tol)
        # sample vertices: all synapse-bearing + the seam neighborhood (keeps it tractable, on-cable)
        cand = set(syn_vert.tolist()) | _ball(adj, int(best_v), hops + seam_tol)
        for v in cand:
            X.append(descriptor(V, R, adj, int(v), pre_at, post_at, hops))
            y.append(1 if v in seam_set else 0)
            grp.append(comp.get(rv, -1))
        n_obj += 1
    return np.array(X), np.array(y), np.array(grp), n_obj


def simhash(Xs, bits, seed):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((Xs.shape[1], bits))
    return (Xs @ H > 0)


def itq(Xs, bits, seed, iters=50):
    rng = np.random.default_rng(seed)
    bits = min(bits, Xs.shape[1])      # PCA can't exceed feature dim (descriptor is 9-d)
    # PCA to `bits` dims then learn a rotation minimizing quantization error (Gong & Lazebnik)
    Xc = Xs - Xs.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:bits].T
    Z = Xc @ W
    Rm = rng.standard_normal((bits, bits)); Rm, _ = np.linalg.qr(Rm)
    for _ in range(iters):
        Bc = np.sign(Z @ Rm); Bc[Bc == 0] = 1
        Up, _, Vp = np.linalg.svd(Bc.T @ Z)
        Rm = (Vp.T @ Up.T)
    return (Z @ Rm > 0)


def eval_retrieval(codes, y, grp, ks):
    """grouped CV: query held-out seam codes vs TRAIN codes; precision@k & lift over base rate."""
    from sklearn.model_selection import GroupKFold
    ug = np.unique(grp)
    gkf = GroupKFold(n_splits=max(2, min(5, len(ug))))
    out = {k: [] for k in ks}; base_rates = []
    seam_q, nonseam_q = {k: [] for k in ks}, {k: [] for k in ks}
    for tr, te in gkf.split(codes, y, grp):
        ctr, ytr = codes[tr], y[tr]
        base = ytr.mean(); base_rates.append(base)
        if ytr.sum() == 0:
            continue
        for qi in te:
            ham = (codes[qi][None, :] != ctr).sum(1)   # Hamming to all train
            order = np.argsort(ham)
            for k in ks:
                topk = ytr[order[:k]]
                prec = topk.mean()
                out[k].append(prec / (base + 1e-9))
                (seam_q if y[qi] == 1 else nonseam_q)[k].append(prec)
    return out, np.mean(base_rates), seam_q, nonseam_q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", default="data/sidetable_7box.npz")
    ap.add_argument("--skel-cache", default="data/skel_v117")
    ap.add_argument("--min-syn", type=int, default=8)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--seam-tol", type=int, default=2)
    ap.add_argument("--bits", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y, grp, n_obj = build_sites(args.sidetable, args.skel_cache, args.min_syn, args.hops, args.seam_tol)
    print(f"objects={n_obj}  vertices={len(y)}  seam vertices={int(y.sum())} ({y.mean():.1%})", flush=True)
    if len(y) < 50 or y.sum() < 10:
        print("too few sites; let the v117 fetch grow."); return
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ks = [1, 3, 5, 10]
    for name, codes in (("SimHash", simhash(Xs, args.bits, args.seed)),
                        ("ITQ    ", itq(Xs, args.bits, args.seed))):
        _, base, sq, nq = eval_retrieval(codes, y, grp, ks)
        print(f"\n  {name} ({args.bits} bits)   base rate seam={base:.1%}")
        print(f"  {'k':>4s}{'prec@k(seam q)':>16s}{'prec@k(non q)':>15s}{'lift(seam/base)':>17s}{'discrim(s/n)':>14s}")
        for k in ks:
            s, nn = np.mean(sq[k]), np.mean(nq[k])
            print(f"  {k:>4d}{s:>15.1%}{nn:>15.1%}{s/(base+1e-9):>16.2f}x{s/(nn+1e-9):>13.2f}x")
    print("\n  lift>1 == seam contexts cluster in Hamming space (error sites are hashable).")


if __name__ == "__main__":
    main()
