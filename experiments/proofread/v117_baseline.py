"""Honest raw-v117 baseline: matched synapse-half confusion vs v1507 truth.

The question this answers precisely: **how well does the pre-proofreading v117
agglomeration place each neuron's synapses onto a single object, measured against
the proofread v1507 truth**, evaluated only on rooted-soma neurons (the cells that
*have* ground truth) but with the full volume present as realistic context.

Metric (matched, per-neuron — the agreed definition)
----------------------------------------------------
A synapse *half* (the pre side = an axonal output; the post side = a dendritic input)
belongs at v1507 to exactly one true neuron.  Under v117 it lands in some object.
For each true neuron we **match it to the single v117 object holding the most of its
halves** (largest-overlap correspondence) and count:

* **TP** — its halves in that matched object (correctly grouped),
* **FN** — its halves that ended up in *other* v117 objects (split away),
* **FP** — halves in the matched object belonging to a *different* true neuron
  (merge contamination).

``P = TP/(TP+FP)``, ``R = TP/(TP+FN)``.  We report micro (pooled) and macro
(per-neuron mean), plus recall broken out by **side** (axon-output vs
dendrite-input), because the two compartments fail completely differently.

Only truthed↔truthed merges count as FP (we cannot verify a merge with an untruthed
fragment); those untruthed fragments still exist in the volume and are a real risk
for any downstream joiner, just not scorable here.

Substrate: proofread column ``allen_v1_column_types_slanted_ref`` (~1355 rooted
cells), split into spatially-disjoint train/eval/gap slabs by ``column_split`` so the
numbers can be read leakage-safe.  Truth = materialization v1507 (a supported
release > 1400); prediction = v117 via ``chunkedgraph.get_roots`` at the v117
timestamp.  Root ids are version-specific, so everything is anchored on supervoxel
ids mapped through ``get_roots`` at each timestamp.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import time
from collections import Counter, defaultdict

import numpy as np


# --------------------------------------------------------------------------- #
# pure metric (offline-testable)                                              #
# --------------------------------------------------------------------------- #
def matched_confusion(pred, truth, side=None):
    """Matched per-neuron confusion of a v117 partition vs truth.

    ``pred``  : int array [N] — v117 object id of each synapse half (0 = unmapped).
    ``truth`` : int array [N] — true (v1507) neuron of each half (0 = none).
    ``side``  : optional int8 [N] — 0 = pre/axon-output, 1 = post/dendrite-input;
                enables the per-side recall breakdown.

    Returns a dict with micro/macro P,R,F1, TP/FP/FN, #catastrophic merges (a v117
    object that is the best match of >=2 true neurons), objects-per-neuron stats and
    (if ``side`` given) per-side recall (micro, per-neuron median, frac<0.5).
    """
    pred = np.asarray(pred); truth = np.asarray(truth)
    ok = (pred > 0) & (truth > 0)
    p, t = pred[ok], truth[ok]
    sd = np.asarray(side)[ok] if side is not None else None

    per_neuron = defaultdict(Counter)   # truth -> Counter(obj -> halves)
    obj_total = Counter()               # obj -> total truthed halves
    for a, b in zip(p.tolist(), t.tolist()):
        per_neuron[b][a] += 1
        obj_total[a] += 1

    TP = FP = FN = 0
    matched = {}                        # neuron -> matched obj
    matched_objs = Counter()
    macroP, macroR = [], []
    for neuron, cnts in per_neuron.items():
        M, tp = cnts.most_common(1)[0]
        tot = sum(cnts.values())
        fn = tot - tp
        fp = obj_total[M] - tp
        TP += tp; FP += fp; FN += fn
        matched[neuron] = M
        matched_objs[M] += 1
        macroP.append(tp / (tp + fp) if tp + fp else 1.0)
        macroR.append(tp / (tp + fn) if tp + fn else 1.0)

    catastrophic = sum(1 for o, c in matched_objs.items() if c > 1)
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    mP, mR = float(np.mean(macroP)), float(np.mean(macroR))
    mF = 2 * mP * mR / (mP + mR) if mP + mR else 0.0
    nobj_per = [len(c) for c in per_neuron.values()]

    out = dict(neurons=len(per_neuron), halves=int(len(p)),
               v117_objects=len(obj_total), P=P, R=R, F1=F, TP=int(TP),
               FP=int(FP), FN=int(FN), macroP=mP, macroR=mR, macroF1=mF,
               catastrophic=catastrophic,
               objs_per_neuron_median=float(np.median(nobj_per)),
               objs_per_neuron_max=int(np.max(nobj_per)))
    if sd is not None:
        inM = np.array([a == matched[b] for a, b in zip(p.tolist(), t.tolist())])
        perside = {}
        for lbl, code in (("pre_axon_out", 0), ("post_dend_in", 1)):
            sel = sd == code
            micro = float(inM[sel].mean()) if sel.sum() else float("nan")
            macro = [float(inM[sel & (t == b)].mean())
                     for b in per_neuron if (sel & (t == b)).sum() >= 5]
            perside[lbl] = dict(
                halves=int(sel.sum()), micro_recall=micro,
                median_per_neuron_recall=float(np.median(macro)) if macro else float("nan"),
                frac_neurons_below_half=float(np.mean(np.array(macro) < 0.5)) if macro else float("nan"))
        out["perside"] = perside
    return out


# --------------------------------------------------------------------------- #
# data collection (network)                                                   #
# --------------------------------------------------------------------------- #
def _tz(ts):
    return ts.replace(tzinfo=dt.timezone.utc) if ts.tzinfo is None else ts


def rooted_neurons(client):
    """Return ``[(root_id, region)]`` for every proofread column neuron (v1507 roots)."""
    from experiments.pcfg.column_split import load_column_neurons, make_split
    neurons = load_column_neurons(client)
    sp = make_split(neurons)
    region = {}
    for n in sp.train: region[n.root_id] = "train"
    for n in sp.eval:  region[n.root_id] = "eval"
    for n in sp.gap:   region[n.root_id] = "gap"
    return [(n.root_id, region[n.root_id])
            for grp in (sp.eval, sp.gap, sp.train) for n in grp if n.proofread]


def fetch_halves(neurons, client, cache_dir, *, verbose=True):
    """Resumable fetch of each neuron's synapse halves (supervoxel + side).

    One ``<root>.npz`` per neuron in ``cache_dir`` (skipped if present).  Query at the
    client's current version (must be the truth version, e.g. 1507) so pre/post
    supervoxels are the neuron's own halves.
    """
    os.makedirs(cache_dir, exist_ok=True)

    def q(kw, r):
        for i in range(5):
            try:
                return client.materialize.synapse_query(**{kw: [r]})
            except Exception:
                if i == 4:
                    raise
                time.sleep(2 * 2 ** i)

    t0 = time.time()
    for k, (root, reg) in enumerate(neurons):
        fp = os.path.join(cache_dir, f"{root}.npz")
        if os.path.exists(fp):
            continue
        sv, side = [], []
        for kw, col, sc in (("pre_ids", "pre_pt_supervoxel_id", 0),
                            ("post_ids", "post_pt_supervoxel_id", 1)):
            d = q(kw, root)
            if d is not None and len(d):
                s = d[col].values.astype(np.int64)
                sv.append(s); side.append(np.full(len(s), sc, np.int8))
        sv = np.concatenate(sv) if sv else np.array([], np.int64)
        side = np.concatenate(side) if side else np.array([], np.int8)
        np.savez(fp, sv=sv, side=side, root=root, region=reg)
        if verbose and k % 50 == 0:
            print(f"  fetched {k + 1}/{len(neurons)} ({time.time() - t0:.0f}s)", flush=True)


def map_supervoxels_to_v117(supervoxels, client, *, cache_path=None, chunk=50000,
                            verbose=True):
    """Map supervoxel ids -> v117 object id via ``get_roots`` at the v117 timestamp.

    Cached to ``cache_path`` (npz of k/v) so repeat scoring is cheap.
    """
    ts117 = _tz(client.materialize.get_timestamp(117))
    cache = {}
    if cache_path and os.path.exists(cache_path):
        z = np.load(cache_path)
        cache = dict(zip(z["k"].tolist(), z["v"].tolist()))
    need = np.array(sorted(set(int(x) for x in np.unique(supervoxels)) - set(cache)))
    if len(need):
        t0 = time.time()
        for s in range(0, len(need), chunk):
            bb = need[s:s + chunk].tolist()
            r = client.chunkedgraph.get_roots(bb, timestamp=ts117)
            cache.update({int(k): int(v) for k, v in zip(bb, r.tolist())})
            if verbose and (s // chunk) % 5 == 0:
                print(f"  mapped {s + len(bb)}/{len(need)} sv ({time.time() - t0:.0f}s)",
                      flush=True)
        if cache_path:
            np.savez(cache_path, k=np.array(list(cache), np.int64),
                     v=np.array(list(cache.values()), np.int64))
    return cache


def load_cached_halves(cache_dir):
    """Load all per-neuron half caches -> (sv, truth, side, region) arrays."""
    sv, truth, side, region = [], [], [], []
    for f in glob.glob(os.path.join(cache_dir, "*.npz")):
        z = np.load(f, allow_pickle=True)
        s = z["sv"]
        if len(s) == 0:
            continue
        sv.append(s); truth.append(np.full(len(s), int(z["root"]), np.int64))
        side.append(z["side"]); region.append(np.array([str(z["region"])] * len(s)))
    sv = np.concatenate(sv); truth = np.concatenate(truth)
    side = np.concatenate(side); region = np.concatenate(region)
    keep = sv > 0
    return sv[keep], truth[keep], side[keep], region[keep]


def main():
    import json
    from caveclient import CAVEclient
    tok = os.environ["token"]
    cl = CAVEclient("minnie65_public", auth_token=tok)
    cl.version = 1507
    here = os.path.dirname(__file__)
    cache_dir = os.path.join(here, "..", "..", "cache", "v117_halves")
    map_path = os.path.join(here, "..", "..", "cache", "v117_sv2obj.npz")

    neurons = rooted_neurons(cl)
    print(f"{len(neurons)} rooted proofread column neurons", flush=True)
    fetch_halves(neurons, cl, cache_dir)
    sv, truth, side, region = load_cached_halves(cache_dir)
    print(f"{len(sv)} truthed halves", flush=True)
    m = map_supervoxels_to_v117(sv, cl, cache_path=map_path)
    pred = np.array([m[int(x)] for x in sv], np.int64)

    results = {}
    for name, mask in (("overall", np.ones(len(sv), bool)),
                       ("train", region == "train"),
                       ("eval", region == "eval"),
                       ("gap", region == "gap")):
        if not mask.any():
            continue
        r = matched_confusion(pred[mask], truth[mask], side[mask])
        results[name] = r
        ps = r["perside"]
        print(f"\n[{name}] neurons={r['neurons']} halves={r['halves']} "
              f"v117_objects={r['v117_objects']}")
        print(f"  micro P={r['P']:.3f} R={r['R']:.3f} F1={r['F1']:.3f} "
              f"(TP={r['TP']} FP={r['FP']} FN={r['FN']})  catastrophic={r['catastrophic']}")
        print(f"  macro P={r['macroP']:.3f} R={r['macroR']:.3f} F1={r['macroF1']:.3f}  "
              f"objs/neuron median={r['objs_per_neuron_median']:.0f}")
        print(f"  axon-out  recall micro={ps['pre_axon_out']['micro_recall']:.3f} "
              f"median/neuron={ps['pre_axon_out']['median_per_neuron_recall']:.3f}")
        print(f"  dend-in   recall micro={ps['post_dend_in']['micro_recall']:.3f} "
              f"median/neuron={ps['post_dend_in']['median_per_neuron_recall']:.3f}")
    json.dump(results, open(os.path.join(here, "v117_baseline.json"), "w"),
              indent=2, default=float)
    print("\nsaved v117_baseline.json")


if __name__ == "__main__":
    main()
