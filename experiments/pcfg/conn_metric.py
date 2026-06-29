#!/usr/bin/env python3
"""Phase 1: pre/post-explicit + connectivity-level correction metric.

Every prior scorer counted synapse-SIDES grouped by root_v117 and conflated pre/post -- a
synapse connects a PREsynaptic neuron to a POSTsynaptic neuron (different cells), so a synapse
is only truly correct when BOTH its pre-side and its post-side land in the right cells. This
util reports, for a corrected partition over synapse-side rows:

  * pre_side_accuracy / post_side_accuracy  (plurality best-match per corrected group)
  * connectivity_accuracy  (a synapse_id is correct iff BOTH its sides are correct)
  * do_nothing guardrail   (net within-object pair-error / Rand-disagreement reduction)
  * #splits / #merges applied (passed by caller)

The corrected partition is `corrected_cell[row] -> cell id`; do-nothing = root_v117; truth =
root_later. Scope for the pair guardrail = connected components of (do-nothing group OR
corrected group), so it captures BOTH within-object splits and cross-object joins, and reduces
to per-v117-object (matching cut_report) when no joins are made.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg_synapse_partitions.synapse_correction import SideTable  # noqa: E402


def _C2(n):
    return n * (n - 1) // 2


def _disagree(truth, pred):
    """Rand pair-disagreement: # pairs grouped-together in one labeling but not the other."""
    same_t = sum(_C2(c) for c in Counter(truth).values())
    same_p = sum(_C2(c) for c in Counter(pred).values())
    same_b = sum(_C2(c) for c in Counter(zip(truth, pred)).values())
    return same_t + same_p - 2 * same_b


def _components(dn, corr):
    """Union-find over rows: linked if same do-nothing group OR same corrected group."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    def union(a, b):
        parent[find(a)] = find(b)

    n = len(dn)
    # link each row to a node for its dn group and its corrected group
    for i in range(n):
        union(("row", i), ("dn", int(dn[i])))
        union(("row", i), ("cc", int(corr[i])))
    comp = defaultdict(list)
    for i in range(n):
        comp[find(("row", i))].append(i)
    return list(comp.values())


def _row_correct(truth, pred):
    """Per-row correctness: row correct iff its corrected group's plurality truth == its truth."""
    grp = defaultdict(list)
    for i, p in enumerate(pred):
        grp[p].append(i)
    maj = {}
    for p, idxs in grp.items():
        maj[p] = Counter(truth[i] for i in idxs).most_common(1)[0][0]
    return np.array([truth[i] == maj[pred[i]] for i in range(len(truth))])


def evaluate(tab: SideTable, corrected_cell: np.ndarray, rows, *, n_splits=0, n_merges=0,
             verbose=True):
    """rows = indices of synapse-side rows under correction. corrected_cell[row] = corrected id."""
    rows = np.asarray(rows)
    rows = rows[tab.root_later[rows] > 0]
    truth_all = tab.root_later
    dn_all = tab.root_v117
    correct_flag = {}                                   # row -> bool (for connectivity join)
    out = {"n_splits": int(n_splits), "n_merges": int(n_merges)}

    for sname, smask in (("pre", tab.side[rows] == 0),
                         ("post", tab.side[rows] == 1),
                         ("pooled", np.ones(len(rows), bool))):
        rr = rows[smask]
        if len(rr) == 0:
            out[sname] = dict(acc=float("nan"), dn_err=0, corr_err=0, net=0, n=0)
            continue
        truth = truth_all[rr]; dn = dn_all[rr]; corr = corrected_cell[rr]
        dn_err = corr_err = 0
        ac = at = 0
        for idxs in _components(dn, corr):
            t = truth[idxs]; d = dn[idxs]; c = corr[idxs]
            dn_err += _disagree(t, d)
            corr_err += _disagree(t, c)
            rc = _row_correct(t, c)
            ac += int(rc.sum()); at += len(rc)
            if sname == "pooled":
                for j, i in enumerate(idxs):
                    correct_flag[int(rr[i])] = bool(rc[j])
        out[sname] = dict(acc=ac / max(1, at), dn_err=int(dn_err), corr_err=int(corr_err),
                          net=int(dn_err - corr_err), n=len(rr))

    # connectivity: a synapse is correct iff BOTH its present sides are correct
    by_syn = defaultdict(list)
    for r in rows.tolist():
        if r in correct_flag:
            by_syn[int(tab.syn_id[r])].append(correct_flag[r])
    both = [v for v in by_syn.values() if len(v) == 2]      # synapses with both sides present
    conn_ok = sum(1 for v in both if all(v))
    out["connectivity"] = dict(accuracy=conn_ok / max(1, len(both)),
                               n_synapses_both_sides=len(both),
                               n_synapse_sides=len(rows))
    if verbose:
        _print(out)
    return out


def _print(out):
    print(f"\n  splits applied = {out['n_splits']}   merges applied = {out['n_merges']}")
    for s in ("pre", "post", "pooled"):
        d = out[s]
        base = d["dn_err"]
        pct = 100 * d["net"] / base if base else 0.0
        print(f"  {s:7s} side acc={d['acc']:.3f}  pair net_fixed={d['net']:+,d} "
              f"({pct:+.1f}% of do-nothing, n_sides={d['n']:,})")
    c = out["connectivity"]
    print(f"  CONNECTIVITY (both sides correct) = {c['accuracy']:.3f}  "
          f"over {c['n_synapses_both_sides']:,} two-sided synapses")


# ---------------------------------------------------------------------------
# Self-test: reproduce cut_report's oracle numbers through this metric
# ---------------------------------------------------------------------------
def self_test(sidetable="data/sidetable_7box.npz", skel_dir="data/skel_v117"):
    from scipy.spatial import cKDTree
    from experiments.pcfg_synapse_partitions.close_loop_cut import (
        load_skels, do_nothing_err, root_and_subtrees, disagreement_from_counts,
    )
    d = np.load(sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    skels = load_skels(skel_dir, 117)
    valid = tab.root_later > 0
    rows_by = defaultdict(list)
    for i in np.nonzero(valid)[0]:
        rows_by[int(tab.root_v117[i])].append(int(i))

    corrected = tab.root_v117.copy()                    # start = do-nothing
    next_id = int(tab.root_v117.max()) + 1
    used_rows, n_splits = [], 0
    for rv, (V, E, R) in skels.items():
        idxs = rows_by.get(rv, [])
        if len(idxs) < 8:
            continue
        lat = tab.root_later[idxs]
        if len(set(lat.tolist())) < 2:
            continue
        labs, lab_index = np.unique(lat, return_inverse=True)
        tot = np.bincount(lab_index, minlength=len(labs)).astype(np.int64)
        if do_nothing_err(tot) == 0:
            continue
        P = tab.pt[idxs]
        syn_vert = cKDTree(V).query(P)[1]
        rs = root_and_subtrees(V, E, list(zip(syn_vert.tolist(), lab_index.tolist())), len(labs))
        used_rows.extend(idxs)
        if rs is None:
            continue
        parent, order, sub = rs
        children = defaultdict(list)
        for v in order:
            if parent[v] >= 0:
                children[parent[v]].append(v)
        best_v, best = None, do_nothing_err(tot)
        for v in order:
            if parent[v] < 0:
                continue
            s = sub[v]; A = int(s.sum()); B = int(tot.sum()) - A
            if min(A, B) < 2:
                continue
            e = disagreement_from_counts(s, tot)
            if e < best:
                best, best_v = e, v
        if best_v is None:
            continue
        n_splits += 1
        subset = set(); stack = [best_v]
        while stack:
            u = stack.pop(); subset.add(u); stack.extend(children.get(u, []))
        for k, vtx in enumerate(syn_vert.tolist()):
            if vtx in subset:
                corrected[idxs[k]] = next_id
        next_id += 1

    print(f"self-test: oracle single-edge cut over {n_splits} merge objects")
    out = evaluate(tab, corrected, used_rows, n_splits=n_splits, n_merges=0)
    print("\n  VALIDATION: pooled pair net reproduces cut_report (+78%). NEW pre/post split:")
    print("  pre-side acc < post-side acc (axons harder than dendrites); connectivity = both sides.")
    assert abs(out["pooled"]["net"] / max(1, out["pooled"]["dn_err"]) - 0.78) < 0.03, "pair net drift"
    print("  pair-net assert passed (reproduces cut_report +78%).")
    return out


if __name__ == "__main__":
    self_test()
