"""v117-object joiner: recover shattered axonal OUTPUT synapses by joining a neuron's
v117 pieces, without merging different cells (hold precision ~1.0).

Framed by the honest baseline (`v117_baseline.py`): v117 precision is already ~1.0 and
dendritic-input recall ~0.95, but axon-OUTPUT recall is ~0.09/neuron — the axon is
shattered across dozens of objects.  So the task is axon reconstruction: join a
neuron's output-bearing v117 fragments to its arbor.  Axon and dendrite are scored
separately.

Self-contained (fetches + caches per neuron under repo ``cache/v117_joiner``):

* synapse halves (supervoxel + side) and their v117 object (``get_roots`` @ ts117);
* per-v117-object L2 point cloud (``get_leaves`` -> ``l2cache`` rep_coord_nm) and
  caliber (``max_dt_nm``).

Candidate joins are built **globally** across all sampled neurons (KDTree over object
point clouds) so wrong-neuron distractors are real; each is scored by proximity +
trajectory colinearity + caliber match and labelled correct iff the two objects share
a true (v1507) neuron.  We commit joins with greedy vs merge-aware constrained
union-find (`merge_aware_join`), sweep the accept threshold, and report axon-output
recall before->after **at held precision**, against the oracle (correct-joins-only)
ceiling.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
import time
from collections import Counter, defaultdict

import numpy as np

from experiments.proofread.merge_aware_join import (
    _CDSU, apply_partition, constrained_union_find)
from experiments.proofread.v117_baseline import matched_confusion

CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "cache", "v117_joiner")


def _tz(t):
    return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t


def _tangent(pts, tip, k=8):
    if len(pts) < 3:
        return np.zeros(3)
    d = np.linalg.norm(pts - tip, axis=1)
    knn = pts[np.argsort(d)[:k]]
    if len(knn) < 3:
        return np.zeros(3)
    cc = knn - knn.mean(0)
    _, _, vt = np.linalg.svd(cc, full_matrices=False)
    return vt[0]


def neuron_objects(client, root, region, ts117, *, max_pts=300):
    """Per-v117-object geometry + synapse halves for one proofread neuron (cached)."""
    os.makedirs(CACHE, exist_ok=True)
    fp = os.path.join(CACHE, f"{root}.npz")
    if os.path.exists(fp):
        z = np.load(fp, allow_pickle=True)
        return z["objs"].item(), z["halves"].item()
    cg = client.chunkedgraph

    # --- synapse halves on this neuron, mapped to v117 object ---
    def q(kw):
        for i in range(5):
            try:
                return client.materialize.synapse_query(**{kw: [root]})
            except Exception:
                if i == 4:
                    raise
                time.sleep(2 * 2 ** i)
    sv, side = [], []
    for kw, col, sc in (("pre_ids", "pre_pt_supervoxel_id", 0),
                        ("post_ids", "post_pt_supervoxel_id", 1)):
        d = q(kw)
        if d is not None and len(d):
            s = d[col].values.astype(np.int64)
            sv.append(s); side.append(np.full(len(s), sc, np.int8))
    sv = np.concatenate(sv) if sv else np.array([], np.int64)
    side = np.concatenate(side) if side else np.array([], np.int8)
    keep = sv > 0; sv, side = sv[keep], side[keep]
    sv_obj = {}
    usv = np.unique(sv)
    for b0 in range(0, len(usv), 50000):
        bb = usv[b0:b0 + 50000].tolist()
        r = cg.get_roots(bb, timestamp=ts117)
        sv_obj.update({int(k): int(v) for k, v in zip(bb, r.tolist())})
    half_obj = np.array([sv_obj.get(int(x), 0) for x in sv], np.int64)
    n_out, n_in = Counter(), Counter()
    for o, sd in zip(half_obj.tolist(), side.tolist()):
        if o > 0:
            (n_out if sd == 0 else n_in)[o] += 1
    halves = dict(obj=half_obj, side=side, root=int(root))

    # --- per-object L2 point cloud + caliber ---
    lvs = np.asarray(cg.get_leaves(int(root), stop_layer=2))
    objs = {}
    if len(lvs):
        hist = np.asarray(cg.get_roots(lvs, timestamp=ts117))
        pos = np.full((len(lvs), 3), np.nan); cal = np.full(len(lvs), np.nan)
        ids = [int(x) for x in lvs]
        for b0 in range(0, len(ids), 1000):
            chunk = ids[b0:b0 + 1000]
            d = client.l2cache.get_l2data(chunk, attributes=["rep_coord_nm", "max_dt_nm"])
            for k, lid in enumerate(chunk):
                e = d.get(str(lid), {})
                if e.get("rep_coord_nm"): pos[b0 + k] = e["rep_coord_nm"]
                if e.get("max_dt_nm"): cal[b0 + k] = e["max_dt_nm"]
        rng = np.random.default_rng(int(root) % 2 ** 31)
        for o in np.unique(hist):
            m = (hist == o) & ~np.isnan(pos[:, 0])
            p = pos[m]
            if len(p) == 0:
                continue
            if len(p) > max_pts:
                p = p[rng.choice(len(p), max_pts, replace=False)]
            c = float(np.nanmedian(cal[m])) if np.any(~np.isnan(cal[m])) else 0.0
            objs[int(o)] = dict(truth=int(root), region=region, pts=p.astype(np.float32),
                                caliber=c, n_out=int(n_out.get(int(o), 0)),
                                n_in=int(n_in.get(int(o), 0)))
    np.savez(fp, objs=objs, halves=halves)
    return objs, halves


def build_candidates(objs, *, max_gap=3000.0):
    """Global cross-object candidate joins: proximity + trajectory + caliber."""
    from scipy.spatial import cKDTree
    ids = list(objs)
    allp, owner = [], []
    for o in ids:
        p = objs[o]["pts"]; allp.append(p); owner.append(np.full(len(p), o))
    allp = np.vstack(allp); owner = np.concatenate(owner)
    tree = cKDTree(allp)
    best = {}
    for ia, ib in tree.query_pairs(max_gap, output_type="ndarray"):
        a, b = int(owner[ia]), int(owner[ib])
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        gap = float(np.linalg.norm(allp[ia] - allp[ib]))
        g = (allp[ib] - allp[ia]) / (gap + 1e-9)
        ta = _tangent(objs[a]["pts"], allp[ia]); tb = _tangent(objs[b]["pts"], allp[ib])
        colin = 0.5 * (abs(float(ta @ g)) + abs(float(tb @ g)))
        ca, cb = objs[a]["caliber"], objs[b]["caliber"]
        cmatch = min(ca, cb) / (max(ca, cb) + 1e-9) if max(ca, cb) > 0 else 1.0
        score = 0.5 * colin + 0.3 * (1.0 - gap / max_gap) + 0.2 * cmatch
        correct = int(objs[a]["truth"] == objs[b]["truth"])
        if key not in best or score > best[key][0]:
            best[key] = (score, correct)
    return [(s, a, b, c) for (a, b), (s, c) in best.items()]


def _half_labels(dsu, half_obj):
    return np.array([dsu.find(int(x)) if x > 0 else 0 for x in half_obj])


def evaluate(objs, halves_list, edges, *, tag="", thresholds=None):
    """Sweep greedy vs merge-aware; report axon-output recall at held precision."""
    if thresholds is None:
        thresholds = np.linspace(0.2, 0.9, 25)
    obj_h = np.concatenate([h["obj"] for h in halves_list])
    side = np.concatenate([h["side"] for h in halves_list])
    truth = np.concatenate([np.full(len(h["obj"]), h["root"], np.int64) for h in halves_list])
    keep = obj_h > 0
    obj_h, side, truth = obj_h[keep], side[keep], truth[keep]

    pre_count = {o: objs[o]["n_out"] for o in objs}
    post_count = {o: objs[o]["n_in"] for o in objs}
    area_of = {o: objs[o]["caliber"] for o in objs}
    contaminated = {o for o in objs if objs[o]["n_out"] >= 3 and objs[o]["n_in"] >= 3}
    soma_frags = set()
    byneu = defaultdict(list)
    for o in objs:
        byneu[objs[o]["truth"]].append(o)
    for _, os_ in byneu.items():
        soma_frags.add(max(os_, key=lambda o: objs[o]["n_in"]))

    base = matched_confusion(obj_h, truth, side)

    def sweep(**flags):
        rows = []
        for thr in thresholds:
            dsu, rej = constrained_union_find(
                edges, pre_count=pre_count, post_count=post_count,
                soma_frags=soma_frags, contaminated=contaminated, area_of=area_of,
                threshold=float(thr), caliber_ratio=3.0, **flags)
            r = matched_confusion(_half_labels(dsu, obj_h), truth, side)
            jp = rej["committed_correct"] / rej["committed"] if rej["committed"] else float("nan")
            rows.append(dict(thr=float(thr), P=r["P"], FP=r["FP"], catastrophic=r["catastrophic"],
                             axon_micro=r["perside"]["pre_axon_out"]["micro_recall"],
                             axon_median=r["perside"]["pre_axon_out"]["median_per_neuron_recall"],
                             dend_micro=r["perside"]["post_dend_in"]["micro_recall"],
                             joins=rej["committed"], join_P=jp))
        return rows

    greedy = sweep(use_ad=False, use_soma=False, use_caliber=False, use_quarantine=False)
    aware = sweep()
    dsu = _CDSU(pre_count, post_count, soma_frags)
    for s, a, b, c in edges:
        if c:
            dsu.union(a, b)
    oracle = matched_confusion(_half_labels(dsu, obj_h), truth, side)
    return dict(tag=tag, base=base, oracle=oracle, greedy=greedy, aware=aware,
                n_objects=len(objs), n_edges=len(edges),
                n_edges_correct=sum(e[3] for e in edges))


def _collect(client, neurons, ts117, *, verbose=True):
    objs = {}
    halves_list = []
    t0 = time.time()
    for i, (root, region) in enumerate(neurons):
        o, h = neuron_objects(client, root, region, ts117)
        objs.update(o); halves_list.append(h)
        if verbose and i % 10 == 0:
            print(f"  geom {i + 1}/{len(neurons)} objs={len(objs)} ({time.time() - t0:.0f}s)", flush=True)
    return objs, halves_list


def run(client, neurons, ts117, *, tag="", max_gap=3000.0, verbose=True):
    objs, halves_list = _collect(client, neurons, ts117, verbose=verbose)
    edges = build_candidates(objs, max_gap=max_gap)
    res = evaluate(objs, halves_list, edges, tag=tag)
    _report(res)
    return res


def sweep_gaps(client, neurons, ts117, *, tag="", gaps=(2000, 3000, 5000, 8000),
               verbose=True):
    """Map the oracle ceiling and the best-at-held-precision axon-output recall vs the
    candidate max_gap.  The key diagnostic: is the limit candidate generation (oracle
    rises with gap) or the discriminator (held-P lift stays flat)?"""
    objs, halves_list = _collect(client, neurons, ts117, verbose=verbose)
    rows = []
    print(f"[{tag}] {len(objs)} objects", flush=True)
    print(f"{'gap_um':>6} {'edges':>6} {'corr':>6} {'base_med':>9} {'oracle_med':>10} "
          f"{'held_P_med':>10} {'greedy_max(P)':>16}", flush=True)
    for gap in gaps:
        edges = build_candidates(objs, max_gap=float(gap))
        res = evaluate(objs, halves_list, edges, tag=f"g{gap}")
        b = res["base"]["perside"]["pre_axon_out"]["median_per_neuron_recall"]
        omed = res["oracle"]["perside"]["pre_axon_out"]["median_per_neuron_recall"]
        hp = [r for r in res["aware"] if r["P"] >= 0.999]
        held = max((r["axon_median"] for r in hp), default=float("nan"))
        gm = max(res["greedy"], key=lambda r: r["axon_micro"])
        rows.append(dict(gap=gap, base_median=b, oracle_median=omed, held_P_median=held,
                         greedy_max_micro=gm["axon_micro"], greedy_max_P=gm["P"],
                         n_edges=len(edges), n_correct=res["n_edges_correct"]))
        print(f"{gap / 1000:6.0f} {len(edges):6d} {res['n_edges_correct']:6d} "
              f"{b:9.3f} {omed:10.3f} {held:10.3f} {gm['axon_micro']:.3f}(P{gm['P']:.2f})", flush=True)
    return rows


def _report(res):
    b = res["base"]["perside"]["pre_axon_out"]
    o = res["oracle"]["perside"]["pre_axon_out"]
    print(f"\n[{res['tag']}] {res['n_objects']} objects, {res['n_edges']} candidate joins "
          f"({res['n_edges_correct']} correct)")
    print(f"  BASE   axon-out recall micro={b['micro_recall']:.3f} median/neuron={b['median_per_neuron_recall']:.3f}  P={res['base']['P']:.4f}")
    print(f"  ORACLE axon-out recall micro={o['micro_recall']:.3f} median/neuron={o['median_per_neuron_recall']:.3f}  P={res['oracle']['P']:.4f}")
    for name in ("greedy", "aware"):
        rows = res[name]
        hp = [r for r in rows if r["P"] >= 0.999]
        best = max(hp, key=lambda r: r["axon_micro"]) if hp else None
        mx = max(rows, key=lambda r: r["axon_micro"])
        s = (f"axon_micro={best['axon_micro']:.3f} joins={best['joins']} joinP={best['join_P']:.3f} thr={best['thr']:.2f}"
             if best else "none reached")
        print(f"  {name:11s} best@P>=.999: {s}")
        print(f"  {'':11s} max-recall : axon_micro={mx['axon_micro']:.3f} P={mx['P']:.4f} joinP={mx['join_P']:.3f} FP={mx['FP']} cat={mx['catastrophic']}")


def main():
    import json
    import sys
    from caveclient import CAVEclient
    from experiments.pcfg.column_split import load_column_neurons, make_split
    tok = os.environ["token"]
    cl = CAVEclient("minnie65_public", auth_token=tok)
    cl.version = 1507
    ts117 = _tz(cl.materialize.get_timestamp(117))
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    tag = sys.argv[2] if len(sys.argv) > 2 else "train"
    mode = sys.argv[3] if len(sys.argv) > 3 else "run"      # "run" | "sweep"
    neurons = load_column_neurons(cl)
    sp = make_split(neurons)
    grp = {"train": sp.train, "eval": sp.eval, "gap": sp.gap}[tag]
    pf = [n for n in grp if n.proofread]
    somas = np.array([n.soma_nm for n in pf])
    order = np.argsort(np.linalg.norm(somas - somas[0], axis=1))[:N]   # spatial cluster
    sample = [(pf[i].root_id, tag) for i in order]
    here = os.path.dirname(__file__)
    if mode == "sweep":
        rows = sweep_gaps(cl, sample, ts117, tag=f"{tag}[{N}]")
        json.dump(rows, open(os.path.join(here, f"v117_joiner_sweep_{tag}.json"), "w"),
                  indent=2, default=float)
    else:
        res = run(cl, sample, ts117, tag=f"{tag}[{N}]")
        json.dump(res, open(os.path.join(here, f"v117_joiner_{tag}.json"), "w"),
                  indent=2, default=float)


if __name__ == "__main__":
    main()
