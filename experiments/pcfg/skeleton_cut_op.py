#!/usr/bin/env python3
"""Phase 2: connectivity skeleton CUT operator (replaces the spectral/Fiedler cut).

The validated +79% operator: cut an over-merged object along ONE skeleton edge. This wraps it
as a recursion-ready operator -- prepare an object once, then `best_cut(obj, subtree_root)`
bisects any connected subtree, so Phase 4 can split, then re-cut the still-merged child.

Edge scorers are pluggable:
  * oracle  -- uses root_later truth to pick the disagreement-minimizing edge (ceiling/eval).
  * learned -- per-edge benefit from the seam GNN (seam_detector), with abstention (returns
               None when the best edge's predicted benefit < tau).

Object = a set of synapse-SIDE rows snapped to its v117 skeleton. Subtrees are addressed by
their root vertex via Euler (tin/tout) intervals, so "rows of subtree(v)" is an O(1)-per-row test.
"""
from __future__ import annotations

import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg.close_loop_cut import (  # noqa: E402
    disagreement_from_counts, do_nothing_err,
)


def _adj(E, n):
    g = defaultdict(list)
    for a, b in E:
        a, b = int(a), int(b)
        if a != b:
            g[a].append(b); g[b].append(a)
    return g


def _far(g, V, src):
    """Farthest vertex from src by cable distance (tree DFS)."""
    seen = {src: 0.0}; stack = [src]; best, bd = src, 0.0
    while stack:
        u = stack.pop()
        for w in g[u]:
            if w not in seen:
                seen[w] = seen[u] + float(np.linalg.norm(V[u] - V[w]))
                if seen[w] > bd:
                    bd, best = seen[w], w
                stack.append(w)
    return best


@dataclass
class SkelObject:
    rows: np.ndarray          # global SideTable row indices in this object
    row_vert: np.ndarray      # nearest skeleton vertex per row (local to this object's V)
    parent: dict
    children: dict
    order: list
    root: int
    tin: dict
    tout: dict
    sub: dict | None          # per-vertex subtree label-count vector (oracle only)
    tot: np.ndarray | None    # total label counts (oracle only)
    benefit: dict | None = None   # per-child-vertex learned cut benefit (learned scorer)


def prepare_object(V, E, R, rows, pts, truth=None):
    """Build the spanning tree + Euler intervals + row->vertex snap for one object."""
    from scipy.spatial import cKDTree
    g = _adj(E, len(V))
    if not g:
        return None
    root = _far(g, V, _far(g, V, next(iter(g))))
    parent = {root: -1}; order = []; children = defaultdict(list); dq = deque([root])
    while dq:
        u = dq.popleft(); order.append(u)
        for w in g[u]:
            if w not in parent:
                parent[w] = u; children[u].append(w); dq.append(w)
    tin, tout, t = {}, {}, 0
    st = [(root, False)]
    while st:
        u, closing = st.pop()
        if closing:
            tout[u] = t; t += 1; continue
        tin[u] = t; t += 1; st.append((u, True))
        for w in children[u]:
            st.append((w, False))
    row_vert = cKDTree(V).query(np.asarray(pts))[1]
    sub = tot = None
    if truth is not None:
        labs, lab_index = np.unique(truth, return_inverse=True)
        nlab = len(labs)
        own = defaultdict(lambda: np.zeros(nlab, np.int64))
        for ri, vv in enumerate(row_vert.tolist()):
            if vv in tin:
                own[vv][lab_index[ri]] += 1
        sub = {v: own[v].copy() for v in order}
        for v in reversed(order):
            if parent[v] >= 0:
                sub[parent[v]] += sub[v]
        tot = sub[root].copy()
    return SkelObject(np.asarray(rows), row_vert, parent, children, order, root,
                      tin, tout, sub, tot)


def _subtree_rows(obj, vr):
    """Row indices (into obj.rows) whose vertex is in subtree(vr), via Euler interval."""
    lo, hi = obj.tin[vr], obj.tout[vr]
    return np.array([i for i, vv in enumerate(obj.row_vert.tolist())
                     if vv in obj.tin and lo <= obj.tin[vv] < hi], dtype=int)


def _candidates(obj, vr, min_side, sub_size_rows):
    """Child vertices w strictly inside subtree(vr) whose cut leaves both sides >= min_side rows."""
    lo, hi = obj.tin[vr], obj.tout[vr]
    nvr = sub_size_rows[vr]
    out = []
    for w in obj.order:
        if w == vr:
            continue
        if lo <= obj.tin[w] < hi:                    # w inside subtree(vr)
            a = sub_size_rows[w]
            if min(a, nvr - a) >= min_side:
                out.append(w)
    return out


def _row_subtree_sizes(obj):
    """#rows in subtree(v) per vertex (label-free), via post-order accumulation."""
    size = {v: 0 for v in obj.order}
    for vv in obj.row_vert.tolist():
        if vv in size:
            size[vv] += 1
    for v in reversed(obj.order):
        p = obj.parent[v]
        if p >= 0:
            size[p] += size[v]
    return size


def best_cut(obj, vr=None, *, scorer="oracle", min_side=2, tau=0.0):
    """Bisect subtree(vr). Returns (rowsA_global, rowsB_global, child_vertex) or None (abstain)."""
    if vr is None:
        vr = obj.root
    sizes = _row_subtree_sizes(obj)
    cands = _candidates(obj, vr, min_side, sizes)
    if not cands:
        return None
    if scorer == "oracle":
        assert obj.sub is not None, "oracle scorer needs truth-prepared object"
        svr = obj.sub[vr]
        best_w = min(cands, key=lambda w: disagreement_from_counts(obj.sub[w], svr))
    elif scorer == "learned":
        assert obj.benefit is not None, "learned scorer needs obj.benefit populated"
        best_w = max(cands, key=lambda w: obj.benefit.get(w, -1e9))
        if obj.benefit.get(best_w, -1e9) <= tau:     # abstain
            return None
    else:
        raise ValueError(scorer)
    a_rows = _subtree_rows(obj, best_w)
    all_rows = _subtree_rows(obj, vr)
    a_set = set(a_rows.tolist())
    b_rows = np.array([i for i in all_rows.tolist() if i not in a_set], dtype=int)
    return obj.rows[a_rows], obj.rows[b_rows], best_w


# ---------------------------------------------------------------------------
# Eval: oracle single top-level cut -> conn_metric (must reproduce Phase 1 +78%)
# ---------------------------------------------------------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidetable", default="data/sidetable_7box.npz")
    ap.add_argument("--skel-cache", default="data/skel_v117")
    args = ap.parse_args()
    from experiments.pcfg.close_loop_cut import load_skels
    from experiments.pcfg.synapse_correction import SideTable
    from experiments.pcfg import conn_metric

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    skels = load_skels(args.skel_cache, 117)
    valid = tab.root_later > 0
    rows_by = defaultdict(list)
    for i in np.nonzero(valid)[0]:
        rows_by[int(tab.root_v117[i])].append(int(i))

    corrected = tab.root_v117.copy()
    next_id = int(tab.root_v117.max()) + 1
    used, n_splits = [], 0
    for rv, (V, E, R) in skels.items():
        idxs = rows_by.get(rv, [])
        if len(idxs) < 8:
            continue
        lat = tab.root_later[idxs]
        if len(set(lat.tolist())) < 2:
            continue
        labs, li = np.unique(lat, return_inverse=True)
        tot = np.bincount(li, minlength=len(labs)).astype(np.int64)
        if do_nothing_err(tot) == 0:
            continue
        used.extend(idxs)
        obj = prepare_object(V, E, R, idxs, tab.pt[idxs], truth=lat)
        if obj is None:
            continue
        cut = best_cut(obj, scorer="oracle", min_side=2)
        if cut is None:
            continue
        rowsA, _rowsB, _w = cut
        n_splits += 1
        for r in rowsA.tolist():
            corrected[r] = next_id
        next_id += 1
    print(f"skeleton_cut_op oracle single-cut: {n_splits} splits applied")
    conn_metric.evaluate(tab, corrected, used, n_splits=n_splits, n_merges=0)


if __name__ == "__main__":
    main()
