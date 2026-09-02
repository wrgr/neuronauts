"""EXP-072 — can proposal work at the object level once the substrate is complete?

EXP-060B is the number to beat: reducing by atom instead of endpoint, a panel of
20 candidate objects reaches **12%** of the spanning links assembly needs, and
65% needs a median panel of 3,870. EXP-070 showed the distance metric was wrong
(skeleton tips, not objects) and fixing it moved the ceiling without moving the
verdict. EXP-071 then found the actual cause: the population admits a v117 object
only if it owns a synapse in the cube, so the connective cable — a passing
neurite with no synapse of its own — was never enumerated, and every one of the
2,147 objects holding it was missing.

This re-runs the proposal measurement with both corrections and nothing else new:

  * **substrate** — every v117 object with a voxel in the region
    (``enumerate_region_objects.py``), synapse-free ones included;
  * **geometry** — object point clouds straight from the segmentation volume
    (``build_object_clouds.py``), no skeleton, no endpoints, no per-object fetch.

The proposer stays deliberately stupid: objects within a radius, ranked by
closest approach, capped at a panel size. No tangent, no caliber, no grammar,
nothing learned. If a substrate fix alone moves recall, that is the finding; if
it does not, no scorer downstream was ever going to rescue it.

**Direct and chained recall are both reported, and the distinction is the whole
point.** A spanning link joins two *labelled* atoms, but the object between them
is usually neither — it is connective cable with no synapse and no label. So a
proposer that only ever links labelled atoms to each other has to jump the gap,
while one that proposes over all objects can reach the partner in two steps
through the piece in between. Chained recall at ``h`` hops asks whether the
partner is reachable in the panel graph, which is what an assembler with
transitive closure actually consumes.

**The control is the same code on the old object set.** Restricting the panel to
population objects isolates the substrate change from the metric change, so the
comparison cannot be flattered by the new geometry.

Declared before the run: on the widened substrate, chained recall of MST spanning
links at a panel cap of 20 and a 2 um radius, within 3 hops, must exceed 50% AND
beat the population-only control by at least 20 points.

    python -m neuronauts.experiments.exp072_object_proposal
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.labels import TIER_NONE, load_labels

TOPOLOGY = "data/substrate/topology/kall.npz"
LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"
CLOUDS = "data/substrate/c100um/object_clouds_mip5.npz"
OBJECTS = "data/substrate/c100um/objects_v117_mip5.npz"

#: 5 um was in this sweep and is not any more. At mip-5 cloud density (~73
#: points/um^3) a 5 um ball returns ~2,400 points per query point, and a 40 um
#: probe died at 10 GB before finishing that arm. 1 and 2 um bracket the
#: measured nearest-sibling gap (median ~1.6 um, EXP-071), which is the range
#: the question lives in; a radius that has to reach 5 um is already conceding
#: the panel-size argument.
RADII_NM = (1000.0, 2000.0)
CAPS = (5, 10, 20, 50, 100)
HOPS = (1, 2, 3)
#: Query points per labelled atom. A large atom carries thousands of supervoxel
#: centroids and querying them all is quadratic in dense tissue for no gain --
#: the closest approach to a neighbour is found from a spread sample. Sampled on
#: a fixed stride, not randomly, so the run is reproducible.
MAX_SEED_POINTS = 64
#: Objects below this many voxels are dropped from the panel. EXP-071 measured
#: the connective cable at a median of 5 L2 nodes with 98.6% of its mass in
#: objects of 2+ nodes, so single-voxel dust can go without losing the material
#: that matters -- and it is most of the point count.
MIN_VOXELS = 2

BAR_CHAINED_RECALL = 0.50
BAR_MARGIN_OVER_CONTROL = 0.20
BAR_RADIUS_NM = 2000.0
BAR_CAP = 20
BAR_HOPS = 3
#: Recall through a chained graph is only meaningful next to the size of the
#: set it reaches. A 40 um probe of this same code hit 99.5% chained recall at a
#: panel cap of 50 -- which is the EXP-058 union-find result wearing a new hat
#: unless the reachable set stayed small. Added to the bar BEFORE the registered
#: run, for that reason, and recorded rather than quietly applied.
BAR_MAX_REACHABLE_LABELLED = 50.0

SPEC = Spec(
    id="EXP-072",
    title="Object-level proposal on the widened substrate",
    question="Does proposing at the object level, over every v117 object rather "
             "than only synapse-carrying ones, reach the spanning links at a "
             "usable panel size?",
    criterion=f"on the widened substrate, chained recall of MST spanning links "
              f"at radius {BAR_RADIUS_NM/1000:.0f} um, panel cap {BAR_CAP} and "
              f"at most {BAR_HOPS} hops must exceed "
              f"{BAR_CHAINED_RECALL:.0%}, AND beat the population-only control "
              f"-- the same code on the old object set -- by at least "
              f"{BAR_MARGIN_OVER_CONTROL:.0%}, AND keep the median number of "
              f"reachable LABELLED atoms at or under "
              f"{BAR_MAX_REACHABLE_LABELLED:.0f}. The third clause was added "
              f"before the run: chained recall without a bound on what it "
              f"reaches is EXP-058's union-find result, which scored recall 1.0 "
              f"at pair precision 0.0006 by collapsing the population into one "
              f"cluster. The proposer uses distance alone -- no tangent, "
              f"caliber, grammar or learned score",
    requires=["EXP-071"], requires_ran=["EXP-060B", "EXP-070"],
    inputs=[TOPOLOGY, LABELS_NPZ, CLOUDS, OBJECTS],
    params={"radii_nm": list(RADII_NM), "caps": list(CAPS), "hops": list(HOPS),
            "min_voxels": MIN_VOXELS, "bar_chained_recall": BAR_CHAINED_RECALL,
            "bar_margin": BAR_MARGIN_OVER_CONTROL,
            "bar_radius_nm": BAR_RADIUS_NM, "bar_cap": BAR_CAP,
            "bar_hops": BAR_HOPS},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


def load_clouds(path):
    with np.load(Path(path), allow_pickle=False) as z:
        return (z["object_id"], z["node_ptr"], z["pos_nm"], z["n_voxels"],
                json.loads(bytes(z["meta"]).decode()) if "meta" in z else {})


def _points_of(obj_id, ptr, pos, want):
    """dict: object id -> its points, for a chosen subset."""
    row = {int(a): k for k, a in enumerate(obj_id.tolist())}
    out = {}
    for a in want:
        k = row.get(int(a))
        out[int(a)] = (pos[ptr[k]:ptr[k + 1]] if k is not None
                       else np.empty((0, 3), np.float32))
    return out


def _mst(groups, pts):
    """Spanning links per owner under object distance (EXP-070's ground truth)."""
    links = set()
    for grp in groups:
        grp = [g for g in grp if len(pts.get(g, ()))]
        n = len(grp)
        if n < 2:
            continue
        trees = {g: cKDTree(pts[g]) for g in grp}
        D = np.full((n, n), np.inf)
        for i in range(n):
            Pi = pts[grp[i]]
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = float(trees[grp[j]].query(Pi, k=1)[0].min())
        intree, rest = [0], list(range(1, n))
        while rest:
            _, x, y = min((D[p, q], p, q) for p in intree for q in rest)
            links.add(tuple(sorted((grp[x], grp[y]))))
            intree.append(y); rest.remove(y)
    return links


def _neighbour_map(seed_ids, seed_pts, tree, owner_of_point, radius):
    """``{seed: {other object: closest approach}}`` within ``radius``, uncapped.

    Computed once per radius; the panel cap is a truncation of this, so a cap
    sweep costs one range query rather than one per cap. Vectorised per seed --
    a first version looped over every returned neighbour in Python and did not
    finish a 40 um cube inside ten minutes.
    """
    out: dict[int, dict[int, float]] = {}
    for a in seed_ids:
        a = int(a)
        P = seed_pts.get(a)
        if P is None or not len(P):
            out[a] = {}
            continue
        if len(P) > MAX_SEED_POINTS:
            P = P[::max(1, len(P) // MAX_SEED_POINTS)][:MAX_SEED_POINTS]
        res = tree.query_ball_point(P, r=radius)
        lens = np.fromiter((len(v) for v in res), np.int64, len(res))
        if not lens.sum():
            out[a] = {}
            continue
        idx = np.concatenate([np.asarray(v, np.int64) for v in res if len(v)])
        src = np.repeat(np.arange(len(P)), lens)
        own = owner_of_point[idx]
        keep = own != np.uint64(a)
        if not keep.any():
            out[a] = {}
            continue
        idx, src, own = idx[keep], src[keep], own[keep]
        d = np.linalg.norm(np.asarray(tree.data)[idx] - P[src], axis=1)
        order = np.argsort(own, kind="stable")
        own_s, d_s = own[order], d[order]
        uo, st = np.unique(own_s, return_index=True)
        out[a] = dict(zip(uo.tolist(), np.minimum.reduceat(d_s, st).tolist()))
    return out


def _cap_panel(nmap, cap):
    return {a: sorted(v.items(), key=lambda kv: kv[1])[:cap]
            for a, v in nmap.items()}


def _chained(panel, truth, labelled, owner_of, max_hops):
    """Reachability through the panel graph within ``h`` hops -- with its cost.

    Edges run between any two objects the panel links, labelled or not, so a
    path may pass through connective cable carrying no label of its own, which
    EXP-071 says is the typical case.

    **Recall alone is not interpretable here and is never reported alone.**
    Chaining grows the reachable set fast, and a graph dense enough to reach
    every true partner also reaches everything else -- which is precisely how
    EXP-058's proximity union-find scored recall 1.0 at a pair precision of
    0.0006, having collapsed the population into a single cluster. So each hop
    count also reports how many *labelled* atoms become reachable (the panel an
    assembler would actually have to score) and what share of those are true
    partners. A recall that rises while precision collapses is the union-find
    result again, not progress.
    """
    nodes = sorted({a for a in panel} | {b for v in panel.values() for b, _ in v})
    pos = {a: i for i, a in enumerate(nodes)}
    n = len(nodes)
    rows, cols = [], []
    for a, v in panel.items():
        for b, _ in v:
            rows.append(pos[a]); cols.append(pos[b])
    empty = {h: {"recall": 0.0, "precision": float("nan"),
                 "median_reachable_labelled": 0.0, "n_pairs": 0}
             for h in range(1, max_hops + 1)}
    if not rows or n == 0:
        return empty
    A = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)),
                   shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.int8)

    lab_idx = np.array([pos[a] for a in sorted(labelled) if a in pos], np.int64)
    lab_ids = np.array([a for a in sorted(labelled) if a in pos], np.uint64)
    truth_set = set(truth)

    out = {}
    cum = A.copy()
    frontier = A.copy()
    for h in range(1, max_hops + 1):
        if h > 1:
            frontier = ((frontier @ A) > 0).astype(np.int8)
            cum = ((cum + frontier) > 0).astype(np.int8)
        # restrict to labelled-atom rows/cols: the pairs an assembler scores
        sub = cum[lab_idx][:, lab_idx].tocoo()
        m = sub.row < sub.col                       # upper triangle, no self
        pa, pb = sub.row[m], sub.col[m]
        n_pairs = int(len(pa))
        tp = 0
        for i, j in zip(pa.tolist(), pb.tolist()):
            key = tuple(sorted((int(lab_ids[i]), int(lab_ids[j]))))
            if key in truth_set:
                tp += 1
        deg = np.bincount(np.concatenate([pa, pb]), minlength=len(lab_idx))
        out[h] = {
            "recall": round(tp / max(len(truth_set), 1), 6),
            "precision": round(tp / n_pairs, 6) if n_pairs else float("nan"),
            "median_reachable_labelled": float(np.median(deg)),
            "p90_reachable_labelled": float(np.percentile(deg, 90)),
            "n_pairs": n_pairs,
        }
    return out


def measure(clouds_path, objects_path, topo_path, labels_path, *, verbose=True):
    """The whole measurement, factored out so a sub-volume can be probed with it."""
    obj_id, ptr, pos, nvox, meta = load_clouds(clouds_path)
    with np.load(Path(objects_path), allow_pickle=False) as z:
        in_pop_id, in_pop = z["object_id"], z["in_population"]
    pop_set = set(in_pop_id[in_pop].tolist())

    with np.load(Path(topo_path), allow_pickle=False) as z:
        atoms = z["atom_id"]
    labels = load_labels(labels_path)
    idx = labels.index_of(atoms); has = idx >= 0
    owner = np.zeros(len(atoms), np.int64); pure = np.zeros(len(atoms), bool)
    tier = np.full(len(atoms), TIER_NONE, np.int8)
    owner[has] = labels.owner[idx[has]].astype(np.int64)
    pure[has] = labels.pure[idx[has]]; tier[has] = labels.owner_tier[idx[has]]
    keep = pure & (tier > TIER_NONE) & (owner > 0)
    ids, own = atoms[keep], owner[keep]

    # Keep only labelled atoms the clouds actually cover, carrying their owner
    # across by id rather than by position -- the filter reorders nothing, but
    # pairing two arrays by index after a filter is how alignment bugs start.
    o_map = dict(zip(ids.tolist(), own.tolist()))
    have = set(obj_id.tolist())
    ids = np.array([i for i in ids.tolist() if int(i) in have], np.uint64)
    own = np.array([o_map[int(i)] for i in ids.tolist()], np.int64)

    seed_pts = _points_of(obj_id, ptr, pos, ids.tolist())
    groups = [[int(x) for x in np.sort(ids[own == o]).tolist()]
              for o in np.unique(own)]
    truth = sorted(_mst(groups, seed_pts))
    if verbose:
        print(f"  labelled atoms with geometry {len(ids):,}; "
              f"MST spanning links {len(truth):,}", flush=True)

    # ``n_voxels`` in the clouds file is per POINT (one point per supervoxel),
    # so an object's size is the sum over its points -- not the array itself.
    # Getting that wrong broadcasts a per-point mask against a per-object one.
    per = np.diff(ptr)
    obj_voxels = np.add.reduceat(nvox, ptr[:-1]) if len(ptr) > 1 else nvox
    big = obj_voxels >= MIN_VOXELS
    point_obj = np.repeat(obj_id, per)
    point_big = np.repeat(big, per)
    labelled_set = set(ids.tolist())
    if verbose:
        print(f"  objects {len(obj_id):,}; >= {MIN_VOXELS} voxels: "
              f"{int(big.sum()):,} ({big.mean():.1%})", flush=True)

    results = {}
    for label, restrict in (("widened", None), ("population_only", pop_set)):
        sel = point_big.copy()
        if restrict is not None:
            sel &= np.isin(point_obj, np.fromiter(restrict, np.uint64,
                                                  len(restrict)))
        P = pos[sel]
        owner_of_point = point_obj[sel]
        if verbose:
            print(f"  [{label}] {len(np.unique(owner_of_point)):,} objects, "
                  f"{len(P):,} points", flush=True)
        tree = cKDTree(P)
        rows = {}
        for r in RADII_NM:
            nmap = _neighbour_map(ids.tolist(), seed_pts, tree, owner_of_point, r)
            for cap in CAPS:
                panel = _cap_panel(nmap, cap)
                pairs = {tuple(sorted((a, b))) for a, v in panel.items()
                         for b, _ in v}
                direct_tp = len(pairs & set(truth))
                direct = direct_tp / max(len(truth), 1)
                # direct precision over labelled-atom pairs only, so it is
                # comparable with the chained figure
                lab_pairs = {p for p in pairs
                             if p[0] in labelled_set and p[1] in labelled_set}
                ch = _chained(panel, truth, labelled_set, None, max(HOPS))
                sizes = np.array([len(v) for v in panel.values()], float)
                rows[f"r{int(r)}_cap{cap}"] = {
                    "radius_nm": r, "cap": cap,
                    "direct_recall": round(direct, 6),
                    "direct_precision_labelled_pairs":
                        round(len(lab_pairs & set(truth)) / len(lab_pairs), 6)
                        if lab_pairs else float("nan"),
                    "n_direct_labelled_pairs": len(lab_pairs),
                    **{f"chained_h{h}": ch[h] for h in HOPS},
                    "median_panel": float(np.median(sizes)) if len(sizes) else 0.0,
                    "p90_panel": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
                }
                if verbose:
                    q = rows[f"r{int(r)}_cap{cap}"]
                    c3 = q[f"chained_h{max(HOPS)}"]
                    print(f"    {label:<16} r={int(r/1000)}um cap={cap:<4} "
                          f"direct {direct:6.1%}  chained h3 R {c3['recall']:6.1%} "
                          f"P {c3['precision']:6.2%}  reachable-labelled "
                          f"{c3['median_reachable_labelled']:6.0f}", flush=True)
        results[label] = rows

    return {"results": results, "n_truth": len(truth),
            "n_labelled": int(len(ids)), "clouds_meta": meta,
            "n_objects_total": int(len(obj_id)),
            "n_objects_in_population": int(len(pop_set))}


def run(ctx: Context) -> Outcome:
    root = ctx.root
    m = measure(root / CLOUDS, root / OBJECTS, root / TOPOLOGY,
                root / LABELS_NPZ)
    key = f"r{int(BAR_RADIUS_NM)}_cap{BAR_CAP}"
    field = f"chained_h{BAR_HOPS}"
    wide = m["results"]["widened"][key][field]["recall"]
    ctrl = m["results"]["population_only"][key][field]["recall"]
    wide_p = m["results"]["widened"][key][field]["precision"]
    ctrl_p = m["results"]["population_only"][key][field]["precision"]
    reach = m["results"]["widened"][key][field]["median_reachable_labelled"]
    gain = wide - ctrl
    passed = bool(wide >= BAR_CHAINED_RECALL
                  and gain >= BAR_MARGIN_OVER_CONTROL
                  and reach <= BAR_MAX_REACHABLE_LABELLED)

    best_direct = max(v["direct_recall"]
                      for v in m["results"]["widened"].values())
    return Outcome(
        passed=passed,
        observed={
            "widened_chained_recall_at_bar": wide,
            "population_only_chained_recall_at_bar": ctrl,
            "gain_from_widening": round(gain, 6),
            "widened_chained_precision_at_bar": wide_p,
            "population_only_chained_precision_at_bar": ctrl_p,
            "median_reachable_labelled_at_bar": reach,
            "widened_direct_recall_at_bar":
                m["results"]["widened"][key]["direct_recall"],
            "best_direct_recall_any_setting": best_direct,
            "median_panel_at_bar":
                m["results"]["widened"][key]["median_panel"],
            "n_spanning_links": m["n_truth"],
        },
        population={"n_labelled_atoms": m["n_labelled"],
                    "n_spanning_links": m["n_truth"],
                    "n_objects_total": m["n_objects_total"],
                    "n_objects_in_population": m["n_objects_in_population"]},
        tables={"by_substrate": m["results"], "clouds_meta": m["clouds_meta"]},
        note=(f"widening the substrate takes chained recall at radius "
              f"{BAR_RADIUS_NM/1000:.0f} um, panel {BAR_CAP}, {BAR_HOPS} hops "
              f"from {ctrl:.1%} to {wide:.1%} (+{gain:.1%}), on distance alone "
              f"with no scorer. EXP-060B's comparable figure on the old "
              f"substrate was 12% at a panel of 20."
              if passed else
              f"bar not met: widened {wide:.1%} (bar "
              f"{BAR_CHAINED_RECALL:.0%}), control {ctrl:.1%}, gain "
              f"{gain:+.1%} (bar +{BAR_MARGIN_OVER_CONTROL:.0%}). Widening the "
              f"substrate is not sufficient for proposal by distance alone"),
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
