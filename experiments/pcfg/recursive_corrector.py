#!/usr/bin/env python3
"""Phase 4: recursive split(-merge) corrector.

Reuses the assembly.py recursion SHAPE (split a still-merged object, re-test, recurse on
children) but with the validated connectivity skeleton cut as the operator and a supervised
atomicity head as the STOP test. Each cut peels one fragment along one skeleton edge; the
recursion handles >2-cell objects (single cut couldn't). Evaluated end-to-end with the
pre/post connectivity metric vs do-nothing, reporting #splits applied.

Stop test options: `pure` (recurse until each cell is one v1718 label -- the oracle ceiling),
or `detector` (the Phase-3 RF p_merge; stop when a child looks like one clean neuron).
Cut scorer: `oracle` (disagreement-minimizing edge) -- the realistic learned/abstaining cut
plugs in via skeleton_cut_op once a trained seam model is supplied.

    python -m experiments.pcfg_synapse_partitions.recursive_corrector --stop pure
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg_synapse_partitions.synapse_correction import SideTable  # noqa: E402
from experiments.pcfg_synapse_partitions.close_loop_cut import (  # noqa: E402
    load_skels, disagreement_from_counts, do_nothing_err,
)
from experiments.pcfg_synapse_partitions.skeleton_cut_op import prepare_object  # noqa: E402
from experiments.pcfg_synapse_partitions import conn_metric  # noqa: E402


def _best_edge(obj, rows_local, lab_index, nlab, min_side):
    """Best single tree-edge cut over the CURRENT rows: post-order count then argmin disagreement."""
    own = defaultdict(lambda: np.zeros(nlab, np.int64))
    for i in rows_local:
        own[int(obj.row_vert[i])][lab_index[i]] += 1
    sub = {v: own[v].copy() if v in own else np.zeros(nlab, np.int64) for v in obj.order}
    for v in reversed(obj.order):
        p = obj.parent[v]
        if p >= 0:
            sub[p] += sub[v]
    tot = sub[obj.root]
    ntot = int(tot.sum())
    best_v, best_e = None, do_nothing_err(tot)
    for v in obj.order:
        if v == obj.root:
            continue
        a = int(sub[v].sum())
        if min(a, ntot - a) < min_side:
            continue
        e = disagreement_from_counts(sub[v], tot)
        if e < best_e:
            best_e, best_v = e, v
    if best_v is None:
        return None
    lo, hi = obj.tin[best_v], obj.tout[best_v]
    A = [i for i in rows_local if lo <= obj.tin[int(obj.row_vert[i])] < hi]
    Aset = set(A)
    B = [i for i in rows_local if i not in Aset]
    return A, B, best_v


def correct_object(obj, lab_index, nlab, *, stop, detector_pmerge, pts_local,
                   max_rounds=8, min_side=2, stop_thresh=0.5):
    """Recursively split one object's rows into corrected cells. Returns list[list[row_local]]."""
    out = []
    stack = [(list(range(len(lab_index))), 0)]
    while stack:
        rows_local, depth = stack.pop()
        cur = lab_index[rows_local]
        if len(rows_local) < 2 * min_side or depth >= max_rounds:
            out.append(rows_local); continue
        # STOP test
        if stop == "pure":
            if len(set(cur.tolist())) <= 1:
                out.append(rows_local); continue
        elif stop == "detector":
            if detector_pmerge(pts_local[rows_local]) < stop_thresh:
                out.append(rows_local); continue
        res = _best_edge(obj, rows_local, lab_index, nlab, min_side)
        if res is None:
            out.append(rows_local); continue
        A, B, _v = res
        if not A or not B:
            out.append(rows_local); continue
        stack.append((A, depth + 1)); stack.append((B, depth + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidetable", default="data/sidetable_7box.npz")
    ap.add_argument("--skel-cache", default="data/skel_v117")
    ap.add_argument("--stop", choices=["pure", "detector"], default="pure")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--stop-thresh", type=float, default=0.5)
    args = ap.parse_args()

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    skels = load_skels(args.skel_cache, 117)
    valid = tab.root_later > 0
    rows_by = defaultdict(list)
    for i in np.nonzero(valid)[0]:
        rows_by[int(tab.root_v117[i])].append(int(i))

    detector = None
    if args.stop == "detector":
        from experiments.pcfg_synapse_partitions.atomicity_detector import AtomicityDetector, build_objects
        detector = AtomicityDetector().fit(build_objects(args.sidetable))

    corrected = tab.root_v117.copy()
    next_id = int(tab.root_v117.max()) + 1
    used, n_splits, n_objs = [], 0, 0
    for rv, (V, E, R) in skels.items():
        idxs = rows_by.get(rv, [])
        if len(idxs) < 8:
            continue
        lat = tab.root_later[idxs]
        if len(set(lat.tolist())) < 2:
            continue
        labs, lab_index = np.unique(lat, return_inverse=True)
        if do_nothing_err(np.bincount(lab_index, minlength=len(labs))) == 0:
            continue
        obj = prepare_object(V, E, R, idxs, tab.pt[idxs], truth=None)
        if obj is None:
            continue
        used.extend(idxs); n_objs += 1
        pts_local = tab.pt[idxs]
        cells = correct_object(obj, lab_index, len(labs), stop=args.stop,
                               detector_pmerge=(detector.p_merge if detector else None),
                               pts_local=pts_local, max_rounds=args.max_rounds,
                               stop_thresh=args.stop_thresh)
        n_splits += max(0, len(cells) - 1)            # cuts applied = #cells - 1
        for cell in cells:
            cid = next_id; next_id += 1
            for li in cell:
                corrected[idxs[li]] = cid

    print(f"recursive corrector (stop={args.stop}): {n_objs} merge objects, "
          f"{n_splits} splits applied")
    conn_metric.evaluate(tab, corrected, used, n_splits=n_splits, n_merges=0)


if __name__ == "__main__":
    main()
