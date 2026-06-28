#!/usr/bin/env python3
"""Connectivity-aware CUT operator on the real v117 over-merged skeletons -- re-close the loop.

The naive 2-means cut was net-negative even with a perfect detector, because merges are
imbalanced (small fragment fused on a big cell) and the seam is a CONNECTIVITY feature, not a
spatial gap (the two cells touch). This cuts along the actual v117 skeleton.

For each real false-merge object (v117 skeleton + its synapses + v1718 truth):
  * ORACLE single-edge cut -- the best skeleton edge to remove (ceiling: is the seam ONE edge?)
  * unsupervised cuts (deploy-fair, no labels): min-radius edge (thin neck) and max-radius-jump
    edge (caliber discontinuity), each restricted to non-trivial splits.
  * 2-means (reference).
Scored by Rand-disagreement within-object pair errors vs v1718; net vs do-nothing.

    python -m experiments.pcfg_synapse_partitions.close_loop_cut --sidetable data/sidetable_7box.npz
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg_synapse_partitions.synapse_correction import SideTable  # noqa: E402


def load_skels(skel_dir, version):
    out = {}
    pat = re.compile(rf"v{version}_rid(\d+)_skv")
    for f in glob.glob(str(Path(skel_dir) / f"v{version}_rid*.npz")):
        m = pat.search(Path(f).name)
        if not m:
            continue
        dd = np.load(f)
        V = dd["vertices"].astype(np.float64)
        E = dd["edges"].astype(np.int64) if "edges" in dd else np.zeros((0, 2), np.int64)
        if len(V) < 8 or len(E) < 4:
            continue
        R = dd["radius"].astype(np.float64) if "radius" in dd else np.full(len(V), 300.0)
        out[int(m.group(1))] = (V, E, R)
    return out


def _C2(n):
    return n * (n - 1) // 2


def disagreement_from_counts(sub, tot):
    """Rand disagreement for a 2-way cut, given per-label subtree counts `sub` and totals `tot`."""
    A = sub.sum(); B = tot.sum() - A
    same_t = _C2(int(tot.sum())) - sum(_C2(int(c)) for c in tot)  # cross-label pairs (= do-nothing err); not used here
    # err = same_t_full + same_p - 2*same_b ; compute directly:
    same_t_full = sum(_C2(int(c)) for c in tot)
    same_p = _C2(int(A)) + _C2(int(B))
    same_b = sum(_C2(int(a)) + _C2(int(t - a)) for a, t in zip(sub, tot))
    return same_t_full + same_p - 2 * same_b


def do_nothing_err(tot):
    return _C2(int(tot.sum())) - sum(_C2(int(c)) for c in tot)


def root_and_subtrees(V, E, syn_lab, nlab):
    """Spanning tree from vertex 0; return per-vertex subtree label-count vectors + tree edges."""
    g = defaultdict(list)
    for a, b in E:
        a, b = int(a), int(b)
        if a != b:
            g[a].append(b); g[b].append(a)
    if not g:
        return None
    root = next(iter(g))
    parent = {root: -1}; order = []
    dq = deque([root])
    while dq:
        u = dq.popleft(); order.append(u)
        for w in g[u]:
            if w not in parent:
                parent[w] = u; dq.append(w)
    own = defaultdict(lambda: np.zeros(nlab, np.int64))
    for vtx, lab in syn_lab:
        own[vtx][lab] += 1
    sub = {v: own[v].copy() for v in order}
    for v in reversed(order):                       # post-order accumulate
        p = parent[v]
        if p >= 0:
            sub[p] += sub[v]
    return parent, order, sub


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--skel-cache", default="data/skel_v117")
    ap.add_argument("--min-syn", type=int, default=8)
    ap.add_argument("--min-side", type=int, default=2)
    args = ap.parse_args()

    from scipy.spatial import cKDTree

    skels = load_skels(args.skel_cache, 117)
    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    valid = tab.root_later > 0
    pts_by, lat_by = defaultdict(list), defaultdict(list)
    for i in np.nonzero(valid)[0]:
        rv = int(tab.root_v117[i])
        pts_by[rv].append(tab.pt[i]); lat_by[rv].append(int(tab.root_later[i]))

    tot_dn = 0
    sums = {k: 0 for k in ("oracle", "min_radius", "radius_jump", "kmeans")}
    n_obj = 0
    from sklearn.cluster import KMeans
    for rv, V_E_R in skels.items():
        if rv not in pts_by:
            continue
        P = np.asarray(pts_by[rv]); lat = np.asarray(lat_by[rv])
        if len(P) < args.min_syn or len(set(lat.tolist())) < 2:
            continue
        V, E, R = V_E_R
        labs, lab_index = np.unique(lat, return_inverse=True)
        nlab = len(labs)
        tot = np.bincount(lab_index, minlength=nlab).astype(np.int64)
        dn = do_nothing_err(tot)
        if dn == 0:
            continue
        n_obj += 1; tot_dn += dn

        syn_vert = cKDTree(V).query(P)[1]
        syn_lab = list(zip(syn_vert.tolist(), lab_index.tolist()))
        rs = root_and_subtrees(V, E, syn_lab, nlab)
        if rs is None:
            sums["oracle"] += dn; sums["min_radius"] += dn; sums["radius_jump"] += dn
        else:
            parent, order, sub = rs
            best = dn; cand = []                     # (err, radius_at_child, radius_jump)
            for v in order:
                p = parent[v]
                if p < 0:
                    continue
                s = sub[v]; A = int(s.sum()); B = int(tot.sum()) - A
                if min(A, B) < args.min_side:
                    continue
                err = disagreement_from_counts(s, tot)
                best = min(best, err)
                cand.append((err, min(R[v], R[p]), abs(R[v] - R[p])))
            sums["oracle"] += best
            if cand:
                sums["min_radius"] += min(cand, key=lambda c: c[1])[0]      # thin-neck cut
                sums["radius_jump"] += max(cand, key=lambda c: c[2])[0]     # caliber-jump cut
            else:
                sums["min_radius"] += dn; sums["radius_jump"] += dn
        # 2-means reference
        km = KMeans(2, n_init=2, random_state=0).fit_predict(P)
        ksub = np.zeros(nlab, np.int64)
        for li, kk in zip(lab_index.tolist(), km.tolist()):
            if kk == 0:
                ksub[li] += 1
        sums["kmeans"] += disagreement_from_counts(ksub, tot)

    print(f"merge objects with v117 skeleton + >=2 cells = {n_obj}")
    print(f"do-nothing within-object pair errors = {tot_dn:,}\n")
    print(f"  {'cut operator':16s}{'pair errors':>13s}{'net_fixed':>12s}{'% of base':>11s}")
    for k in ("kmeans", "min_radius", "radius_jump", "oracle"):
        net = tot_dn - sums[k]
        print(f"  {k:16s}{sums[k]:>13,d}{net:>+12,d}{100*net/max(1,tot_dn):>10.1f}%")
    print("\n  oracle = best single skeleton-edge cut (ceiling: is the seam ONE edge?)")
    print("  net>0 == a connectivity cut beats do-nothing where 2-means could not.")


if __name__ == "__main__":
    main()
