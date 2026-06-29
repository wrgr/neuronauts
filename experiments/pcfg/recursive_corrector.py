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


# ---------------------------------------------------------------------------
# (a) LEARNED cut: seam GNN per-edge benefit, autonomous (abstain) + human-assist (top-k)
# ---------------------------------------------------------------------------
def _row_counts(obj, rows_local):
    """#current rows in subtree(v) per vertex (label-free), via post-order."""
    own = defaultdict(int)
    for i in rows_local:
        own[int(obj.row_vert[i])] += 1
    cnt = {v: own.get(v, 0) for v in obj.order}
    for v in reversed(obj.order):
        p = obj.parent[v]
        if p >= 0:
            cnt[p] += cnt[v]
    return cnt


def _split_rows(obj, rows_local, v):
    lo, hi = obj.tin[v], obj.tout[v]
    A = [i for i in rows_local if lo <= obj.tin[int(obj.row_vert[i])] < hi]
    As = set(A)
    return A, [i for i in rows_local if i not in As]


def correct_object_cb(obj, lab_index, choose_fn, *, max_rounds, min_side):
    """Recurse using a pluggable choose_fn(obj, rows_local, cnt, tot)->vertex|None."""
    out = []
    stack = [(list(range(len(lab_index))), 0)]
    while stack:
        rows_local, depth = stack.pop()
        if len(rows_local) < 2 * min_side or depth >= max_rounds:
            out.append(rows_local); continue
        cnt = _row_counts(obj, rows_local)
        tot = len(rows_local)
        v = choose_fn(obj, rows_local, cnt, tot)
        if v is None:
            out.append(rows_local); continue
        A, B = _split_rows(obj, rows_local, v)
        if not A or not B:
            out.append(rows_local); continue
        stack.append((A, depth + 1)); stack.append((B, depth + 1))
    return out


def _candidates(obj, rows_local, cnt, tot, min_side):
    return [v for v in obj.order if v != obj.root and 0 < cnt[v] < tot
            and min(cnt[v], tot - cnt[v]) >= min_side]


def _train_seam(train_objs, epochs, seed):
    import torch
    import torch.nn.functional as F
    from experiments.pcfg_synapse_partitions.seam_detector import build_model
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = build_model(); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-5)
    for ep in range(epochs):
        net.train(); order = rng.permutation(len(train_objs)); opt.zero_grad(); acc = 0
        for k, i in enumerate(order.tolist()):
            o = train_objs[i]
            pred = net(torch.tensor(o["feat"]), torch.tensor(o["edges"], dtype=torch.long))
            loss = F.smooth_l1_loss(pred, torch.tensor(o["benefit"]))
            loss.backward(); acc += 1
            if acc == 8 or k == len(order) - 1:
                opt.step(); opt.zero_grad(); acc = 0
    return net


def _edge_benefit(net, o):
    """{frozenset{global_u,global_v}: predicted benefit} from a seam-detector object dict."""
    import torch
    with torch.no_grad():
        pred = net(torch.tensor(o["feat"]), torch.tensor(o["edges"], dtype=torch.long)).numpy()
    order = o["order"]
    out = {}
    for k, (lp, lv) in enumerate(o["edges"].tolist()):
        out[frozenset((int(order[lp]), int(order[lv])))] = float(pred[k])
    return out


def run_learned_cv(sidetable, skel_cache, folds, epochs, seed, max_rounds, min_side, tau, topk):
    from sklearn.model_selection import GroupKFold
    from experiments.pcfg_synapse_partitions.seam_detector import build_objects
    d = np.load(sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    skels = load_skels(skel_cache, 117)
    valid = tab.root_later > 0
    rows_by = defaultdict(list)
    for i in np.nonzero(valid)[0]:
        rows_by[int(tab.root_v117[i])].append(int(i))
    objs = build_objects(sidetable, skel_cache, min_syn=8, min_side=min_side)
    groups = np.array([o["group"] for o in objs])
    print(f"learned-cut CV: {len(objs)} merge objects, {len(np.unique(groups))} cells")

    corr_auto = tab.root_v117.copy()
    corr_assist = tab.root_v117.copy()
    nid = [int(tab.root_v117.max()) + 1]
    used, ns_auto, ns_assist = [], 0, 0
    gkf = GroupKFold(n_splits=max(2, min(folds, len(np.unique(groups)))))
    for fold, (tr, te) in enumerate(gkf.split(objs, np.zeros(len(objs)), groups)):
        net = _train_seam([objs[i] for i in tr], epochs, seed)
        print(f"  fold {fold+1}: trained on {len(tr)} objs, applying to {len(te)}", flush=True)
        for ti in te:
            o = objs[ti]; rv = o["rv"]; idxs = rows_by[rv]
            V, E, R = skels[rv]
            lat = tab.root_later[idxs]
            labs, lab_index = np.unique(lat, return_inverse=True)
            obj = prepare_object(V, E, R, idxs, tab.pt[idxs], truth=None)
            if obj is None:
                continue
            ben = _edge_benefit(net, o)
            used.extend(idxs)

            def edge_ben(obj, v):
                return ben.get(frozenset((int(obj.parent[v]), int(v))), -1e9)

            # autonomous: max benefit, abstain if <= tau
            def choose_auto(obj, rows_local, cnt, tot):
                cands = _candidates(obj, rows_local, cnt, tot, min_side)
                if not cands:
                    return None
                best = max(cands, key=lambda v: edge_ben(obj, v))
                return best if edge_ben(obj, best) > tau else None

            # human-assist: top-k by benefit, human (oracle) verifies -> pick min disagreement, only if it helps
            def choose_assist(obj, rows_local, cnt, tot):
                cands = _candidates(obj, rows_local, cnt, tot, min_side)
                if not cands:
                    return None
                cands.sort(key=lambda v: edge_ben(obj, v), reverse=True)
                top = cands[:topk]
                own = defaultdict(lambda: np.zeros(len(labs), np.int64))
                for i in rows_local:
                    own[int(obj.row_vert[i])][lab_index[i]] += 1
                sub = {v: own[v].copy() if v in own else np.zeros(len(labs), np.int64) for v in obj.order}
                for v in reversed(obj.order):
                    p = obj.parent[v]
                    if p >= 0:
                        sub[p] += sub[v]
                tt = sub[obj.root]; base = do_nothing_err(tt)
                best_v, best_e = None, base
                for v in top:
                    e = disagreement_from_counts(sub[v], tt)
                    if e < best_e:
                        best_e, best_v = e, v
                return best_v

            for choose, corr, kind in ((choose_auto, corr_auto, "auto"),
                                       (choose_assist, corr_assist, "assist")):
                cells = correct_object_cb(obj, lab_index, choose, max_rounds=max_rounds, min_side=min_side)
                nsplits = max(0, len(cells) - 1)
                for cell in cells:
                    cid = nid[0]; nid[0] += 1
                    for li in cell:
                        corr[idxs[li]] = cid
                if kind == "auto":
                    ns_auto += nsplits
                else:
                    ns_assist += nsplits

    print("\n=== LEARNED cut, AUTONOMOUS (abstaining) ===")
    conn_metric.evaluate(tab, corr_auto, used, n_splits=ns_auto, n_merges=0)
    print(f"\n=== LEARNED cut, HUMAN-ASSIST (top-{topk}, verified) ===")
    conn_metric.evaluate(tab, corr_assist, used, n_splits=ns_assist, n_merges=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidetable", default="data/sidetable_7box.npz")
    ap.add_argument("--skel-cache", default="data/skel_v117")
    ap.add_argument("--stop", choices=["pure", "detector"], default="pure")
    ap.add_argument("--cut", choices=["oracle", "learned"], default="oracle")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--stop-thresh", type=float, default=0.5)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.cut == "learned":
        run_learned_cv(args.sidetable, args.skel_cache, args.folds, args.epochs, args.seed,
                       args.max_rounds, 2, args.tau, args.topk)
        return

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
