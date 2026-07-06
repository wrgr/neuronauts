"""The goal metric: synapse-pair F1 before -> after JOINING L2 fragments with the
follower (global cut-face matching), at held precision.

Substrate (the correct one): L2 chunks are the fragmented baseline (F1~0.234 vs the
proofread v1822 grouping); a neuron is shattered across many L2 fragments, so recall is
low but precision is ~1 (L2 fragments never wrongly merge).  The follower JOINS adjacent
L2 fragments by cut-face continuity with **global one-to-one matching** (the
precision-preserving mechanism); each committed join raises synapse-pair recall.  We
sweep the accept threshold and report F1 before -> after and the **precision of the joins
themselves** (fraction of committed joins whose two L2s really share a v1822 root) — the
number the prior AUC work deferred.

All truth from CAVE v1822 (same machinery as the rest of the repo); no appearance.
"""
from __future__ import annotations

import numpy as np


class _DSU:
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r: r = self.p[r]
        while self.p[x] != r: self.p[x], x = r, self.p[x]
        return r
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def _pair_f1(labels, truth):
    """Synapse-pair line-graph F1 of a partition vs truth (pair-counting)."""
    from collections import Counter
    C2 = lambda n: n * (n - 1) // 2
    k = (labels > 0) & (truth > 0)
    a, b = labels[k], truth[k]
    est = sum(C2(c) for c in Counter(a.tolist()).values())
    tru = sum(C2(c) for c in Counter(b.tolist()).values())
    tp = sum(C2(c) for c in Counter(zip(a.tolist(), b.tolist())).values())
    P = tp / est if est else 0.0
    R = tp / tru if tru else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def build_join_edges(l2_vol, vox, root_of, *, gaps=(1, 2), traj_k=3, min_area=8,
                     search_nm=1500.0, verbose=True):
    """Global cut-face matching across z on the L2 volume -> candidate join edges.

    Returns edges [(weight, l2_a, l2_b, correct)] where correct = (root_of[a]==root_of[b]).
    l2_a != l2_b (a real fragment-join, not self-continuation).
    """
    from experiments.proofread.cutface_slices import (
        _footprints, _iou, _shift_mask, _centroid_on)
    d = l2_vol; vox = np.asarray(vox, float); nz = d.shape[2]; shape2d = d.shape[:2]
    fp = {}
    def foot(z):
        if z not in fp: fp[z] = _footprints(d[:, :, z], min_area)
        return fp[z]
    edges = []
    for gap in gaps:
        for z0 in range(traj_k + 1, nz - gap - 1, 1):
            A = foot(z0); B = foot(z0 + gap)
            if not A or not B: continue
            b_ids = list(B); b_cent = np.array([B[j][2] for j in b_ids]) * vox[:2]
            block = []   # (w, a, b) for this slice-pair, for global matching
            for oid, (ays, axs, acent, aarea) in A.items():
                c_prev = _centroid_on(d, oid, z0 - traj_k)
                vel = (acent - c_prev) / traj_k if c_prev is not None else np.zeros(2)
                sy, sx = _shift_mask(ays, axs, vel * gap, shape2d)
                near = np.where(np.linalg.norm(b_cent - acent * vox[:2], axis=1) <= search_nm)[0]
                for bi in near:
                    j = b_ids[bi]
                    if j == oid: continue                 # self-continuation, not a join
                    w = _iou(sy, sx, B[j][0], B[j][1], shape2d)
                    if w > 0: block.append((w, oid, j))
            block.sort(reverse=True)
            ua, ub = set(), set()
            for w, oid, j in block:                        # global one-to-one per slice-pair
                if oid in ua or j in ub: continue
                ua.add(oid); ub.add(j)
                edges.append((w, int(oid), int(j),
                              int(root_of.get(int(oid), -1) == root_of.get(int(j), -2))))
    if verbose:
        print(f"[join] {len(edges)} candidate L2-join edges")
    return edges


def synapse_f1_curve(l2_of_syn, root_of_syn, edges, *, thresholds=None):
    """Union-find L2 joins above a threshold; synapse-pair F1 before->after + join precision."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.8, 30)
    l2 = np.asarray(l2_of_syn); root = np.asarray(root_of_syn)
    Pb, Rb, Fb = _pair_f1(l2, root)
    rows = [{"thr": None, "stage": "before", "P": Pb, "R": Rb, "F1": Fb,
             "join_precision": float("nan"), "n_joins": 0}]
    for t in thresholds:
        dsu = _DSU()
        nj = ntp = 0
        for w, a, b, correct in edges:
            if w >= t:
                dsu.union(a, b); nj += 1; ntp += correct
        merged = np.array([dsu.find(int(x)) if x > 0 else 0 for x in l2])
        P, R, F = _pair_f1(merged, root)
        rows.append({"thr": float(t), "stage": "after", "P": P, "R": R, "F1": F,
                     "join_precision": (ntp / nj if nj else float("nan")), "n_joins": nj})
    return rows
