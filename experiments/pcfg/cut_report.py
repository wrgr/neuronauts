#!/usr/bin/env python3
"""Re-report the connectivity cut with intuitive metrics + explicit operation counts.

The +79% headline is a PAIR metric (Rand disagreement over within-object synapse pairs). This
adds the plainer numbers: how many SPLITS were applied, how many synapses end up in the right
cell, and what fraction of merges are cleanly separated by one cut -- alongside the pair metric
so the +79% is interpretable. Oracle single-edge cut on real v117 over-merged skeletons.
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
    load_skels, disagreement_from_counts, do_nothing_err, root_and_subtrees,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
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

    n_obj = n_syn = splits_applied = 0
    dn_pair = cut_pair = 0
    syn_correct = 0; obj_acc = []; clean = 0; ncells = []
    for rv, (V, E, R) in skels.items():
        if rv not in pts_by:
            continue
        P = np.asarray(pts_by[rv]); lat = np.asarray(lat_by[rv])
        if len(P) < args.min_syn or len(set(lat.tolist())) < 2:
            continue
        labs, lab_index = np.unique(lat, return_inverse=True)
        nlab = len(labs); tot = np.bincount(lab_index, minlength=nlab).astype(np.int64)
        dn = do_nothing_err(tot)
        if dn == 0:
            continue
        syn_vert = cKDTree(V).query(P)[1]
        rs = root_and_subtrees(V, E, list(zip(syn_vert.tolist(), lab_index.tolist())), nlab)
        if rs is None:
            continue
        parent, order, sub = rs
        children = defaultdict(list)
        for v in order:
            if parent[v] >= 0:
                children[parent[v]].append(v)
        # oracle best single edge
        best_v, best_err = None, dn
        for v in order:
            if parent[v] < 0:
                continue
            s = sub[v]; A = int(s.sum()); B = int(tot.sum()) - A
            if min(A, B) < args.min_side:
                continue
            e = disagreement_from_counts(s, tot)
            if e < best_err:
                best_err, best_v = e, v
        n_obj += 1; n_syn += len(P); ncells.append(nlab)
        dn_pair += dn
        if best_v is None:
            cut_pair += dn; obj_acc.append(0.0); continue
        splits_applied += 1
        # synapse partition for the chosen edge: side A = subtree(best_v) vertices
        stack = [best_v]; subset = set()
        while stack:
            u = stack.pop(); subset.add(u); stack.extend(children.get(u, []))
        side = np.array([1 if v in subset else 0 for v in syn_vert.tolist()])
        cut_pair += best_err
        # object accuracy: assign each side to its plurality true cell, count matches
        acc = 0
        for grp in (0, 1):
            mask = side == grp
            if mask.sum() == 0:
                continue
            maj = np.bincount(lab_index[mask], minlength=nlab).argmax()
            acc += int((lab_index[mask] == maj).sum())
        syn_correct += acc
        a = acc / len(P); obj_acc.append(a); clean += int(a >= 0.90)

    print(f"\nmerge objects (real false-merges, with v117 skeleton) = {n_obj}")
    print(f"synapses in them = {n_syn:,}   (median cells per object = {int(np.median(ncells))})")
    print(f"\nOPERATIONS APPLIED (oracle single-edge cut):")
    print(f"  splits applied (cuts) = {splits_applied}   merges applied (joins) = 0")
    print(f"  -> one cut per object; objects with >2 true cells need >1 cut (not done here)")
    print(f"\nINTUITIVE OUTCOME:")
    print(f"  synapses placed in the correct cell after the cut = {syn_correct:,}/{n_syn:,} "
          f"= {100*syn_correct/max(1,n_syn):.1f}%")
    print(f"  per-object separation accuracy: median={np.median(obj_acc):.2f}  mean={np.mean(obj_acc):.2f}")
    print(f"  merges CLEANLY separated by one cut (>=90% correct) = {clean}/{n_obj} "
          f"= {100*clean/max(1,n_obj):.0f}%")
    print(f"\nPAIR METRIC (the +79% headline, for reference):")
    print(f"  within-object synapse-pair errors: do-nothing={dn_pair:,}  after-cut={cut_pair:,}")
    print(f"  reduction = {dn_pair-cut_pair:,} = {100*(dn_pair-cut_pair)/max(1,dn_pair):.1f}% of do-nothing")
    print("  (a 'pair error' = two synapses grouped together that belong to different cells,")
    print("   or split apart that belong together; quadratic, so big cells dominate.)")


if __name__ == "__main__":
    main()
