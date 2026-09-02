"""EXP-073 — can structural constraints make the chained panel sparse enough to use?

EXP-071 fixed the substrate: the connective cable a cell's fragments are joined
through was never in the synapse-anchored population, and once you walk the real
graph the nearest sibling is ~1.6 um and 2-3 hops away. EXP-072 then showed that
fixing the substrate is not sufficient. Chaining through it by distance alone
collapses exactly as EXP-058's union-find did: at radius 1 um and a panel cap of
20, chained recall reaches 63.6% while pair precision sits at 0.06% and 1,586 of
4,801 labelled atoms become mutually reachable. Dense neuropil connects
everything to everything.

What distance cannot express is *structure*, and structure is what the skeleton
layer was always for. This experiment asks whether the cheapest structural
constraints -- computable from the object clouds already on disk, with no
skeleton, no fetch and nothing learned -- buy back the precision.

Three constraints on a two-hop path ``A -> X -> B``, in increasing strength:

``cable``      X must look like a piece of neurite, not a blob: its point cloud
               is elongated (first principal axis dominant) and its extent is
               bounded. A soma or a large merged object is not a bridge.
``through``    A and B must attach at *opposite ends* of X, measured as the
               angle at X between the directions to each one's closest point.
               Two fragments hanging off the same spot on X are not a
               continuation through it -- this is the grammar rule, in one
               number, and it is the one that should do the work.
``collinear``  the whole path A -> X -> B must be roughly straight, comparing
               the A->X and X->B directions.

Each is a *hard* constraint: it prunes, it does not score. That is deliberate.
EXP-060B's problem was never that the right pair was absent -- it was that the
panel needed to be 3,870 objects for the right pair to be in it. A filter that
removes candidates at near-zero recall cost is worth more here than a scorer
that ranks them, and it is testable without training anything.

**The bar is EXP-072's, unchanged, so the two are a direct A/B.** Chained recall
at radius 2 um, cap 20, at most 3 hops must exceed 50% while the median number
of reachable labelled atoms stays at or under 50. EXP-072 met the recall clause
and failed the reachability clause by a factor of 30. If constraints cannot fix
that, geometry is finished as a proposer and the next lever is an embedding
(EXP-057C) or a learned scorer over a much smaller panel.

    python -m neuronauts.experiments.exp073_constrained_chain
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.experiments.exp072_object_proposal import (
    CLOUDS, LABELS_NPZ, MAX_SEED_POINTS, MIN_VOLUME_UM3, OBJECTS, TOPOLOGY,
    _chained, _mst, _neighbour_map, _points_of, load_clouds,
)
from neuronauts.harness.labels import TIER_NONE, load_labels

RADIUS_NM = 2000.0
CAP = 20
HOPS = (1, 2, 3)

#: Constraint settings, swept weakest to strongest. ``min_elong`` is the ratio
#: of the first to the second principal standard deviation of the bridge's
#: cloud; ``max_extent_nm`` bounds its size; ``min_through_deg`` is the angle at
#: the bridge between the two attachment directions, where 180 deg means the
#: two fragments sit at opposite ends.
SETTINGS = [
    ("none", {}),
    ("cable", {"min_elong": 2.0, "max_extent_nm": 30_000.0}),
    ("through_90", {"min_elong": 2.0, "max_extent_nm": 30_000.0,
                    "min_through_deg": 90.0}),
    ("through_120", {"min_elong": 2.0, "max_extent_nm": 30_000.0,
                     "min_through_deg": 120.0}),
    ("through_150", {"min_elong": 2.0, "max_extent_nm": 30_000.0,
                     "min_through_deg": 150.0}),
    ("through_120_tight", {"min_elong": 3.0, "max_extent_nm": 15_000.0,
                           "min_through_deg": 120.0}),
]

BAR_RECALL = 0.50
BAR_MAX_REACHABLE = 50.0

SPEC = Spec(
    id="EXP-073",
    title="Constrained chaining: does structure prune the panel?",
    question="Do cheap structural constraints on the bridging object make the "
             "chained panel sparse enough to use, where distance alone could "
             "not?",
    criterion=f"EXP-072's bar, unchanged, so the two are a direct A/B: chained "
              f"recall of MST spanning links at radius {RADIUS_NM/1000:.0f} um, "
              f"cap {CAP}, at most 3 hops must exceed {BAR_RECALL:.0%} while "
              f"the median number of reachable LABELLED atoms stays at or under "
              f"{BAR_MAX_REACHABLE:.0f}. EXP-072 met the first clause and "
              f"missed the second by ~30x. Constraints are hard filters on the "
              f"bridging object, computed from the object clouds -- no "
              f"skeleton, no fetch, nothing learned",
    requires_ran=["EXP-072"], requires=["EXP-071"],
    inputs=[TOPOLOGY, LABELS_NPZ, CLOUDS, OBJECTS],
    params={"radius_nm": RADIUS_NM, "cap": CAP,
            "settings": {k: v for k, v in SETTINGS},
            "bar_recall": BAR_RECALL, "bar_max_reachable": BAR_MAX_REACHABLE},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


def object_shape(obj_id, ptr, pos, rows):
    """Centroid, principal axis, elongation and extent for the objects in ``rows``.

    Only the objects that actually appear in some panel need a shape -- a
    handful of thousand, not every root id the enumeration touched (909,888 at
    mip 5, most of them sub-5-voxel debris; see results/EXP-071/CORRECTION.md).
    A first version ran the SVD over every object in a Python loop; on the full
    cube that is the whole budget spent before the first constraint is tested. Objects with fewer than
    three points get an undefined axis and an elongation of 1, which every
    ``cable`` setting rejects -- a two-point cloud has no direction worth trusting.
    """
    out = {}
    for k in rows:
        P = pos[ptr[k]:ptr[k + 1]]
        if not len(P):
            out[k] = (np.zeros(3, np.float32), np.zeros(3, np.float32), 1.0, 0.0)
            continue
        c = P.mean(0)
        ext = float(np.linalg.norm(P.max(0) - P.min(0)))
        if len(P) < 3:
            out[k] = (c, np.zeros(3, np.float32), 1.0, ext)
            continue
        _, s, vt = np.linalg.svd(P - c, full_matrices=False)
        sd = s / np.sqrt(max(len(P) - 1, 1))
        el = float(sd[0] / (sd[1] + 1e-6)) if len(sd) > 1 else 1.0
        out[k] = (c, vt[0], el, ext)
    return out


def _through_angle(pa, px, pb):
    """Angle at the bridge between the directions to each fragment, degrees.

    180 means the two fragments sit at opposite ends of the bridge; 0 means
    they hang off the same spot.
    """
    v1, v2 = pa - px, pb - px
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 <= 0 or n2 <= 0:
        return 180.0
    c = float(np.dot(v1 / n1, v2 / n2))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def measure(clouds_path, objects_path, topo_path, labels_path, *, verbose=True):
    """The whole measurement, factored out so a sub-volume can be probed with it."""
    obj_id, ptr, pos, nvox, meta = load_clouds(clouds_path)

    with np.load(topo_path, allow_pickle=False) as z:
        atoms = z["atom_id"]
    labels = load_labels(labels_path)
    idx = labels.index_of(atoms); has = idx >= 0
    owner = np.zeros(len(atoms), np.int64); pure = np.zeros(len(atoms), bool)
    tier = np.full(len(atoms), TIER_NONE, np.int8)
    owner[has] = labels.owner[idx[has]].astype(np.int64)
    pure[has] = labels.pure[idx[has]]; tier[has] = labels.owner_tier[idx[has]]
    keep = pure & (tier > TIER_NONE) & (owner > 0)
    ids, own = atoms[keep], owner[keep]
    o_map = dict(zip(ids.tolist(), own.tolist()))
    have = set(obj_id.tolist())
    ids = np.array([i for i in ids.tolist() if int(i) in have], np.uint64)
    own = np.array([o_map[int(i)] for i in ids.tolist()], np.int64)

    seed_pts = _points_of(obj_id, ptr, pos, ids.tolist())
    groups = [[int(x) for x in np.sort(ids[own == o]).tolist()]
              for o in np.unique(own)]
    truth = sorted(_mst(groups, seed_pts))
    print(f"  labelled atoms {len(ids):,}; MST spanning links {len(truth):,}",
          flush=True)

    per = np.diff(ptr)
    obj_voxels = np.add.reduceat(nvox, ptr[:-1]) if len(ptr) > 1 else nvox
    # Same dust floor as EXP-072, in physical units, synapse-carriers exempt.
    with np.load(objects_path, allow_pickle=False) as z:
        _pid, _pin = z["object_id"], z["in_population"]
    voxel_um3 = float(np.prod(meta["resolution_nm"])) / 1e9
    has_synapse = np.isin(obj_id, _pid[_pin])
    big = has_synapse | (obj_voxels * voxel_um3 >= MIN_VOLUME_UM3)
    point_obj = np.repeat(obj_id, per)
    sel = np.repeat(big, per)
    P_all, owner_of_point = pos[sel], point_obj[sel]
    print(f"  panel substrate: {int(big.sum()):,} objects, {len(P_all):,} points",
          flush=True)
    tree = cKDTree(P_all)
    row_of = {int(a): k for k, a in enumerate(obj_id.tolist())}

    nmap = _neighbour_map(ids.tolist(), seed_pts, tree, owner_of_point, RADIUS_NM)
    base = {a: sorted(v.items(), key=lambda kv: kv[1])[:CAP]
            for a, v in nmap.items()}
    labelled_set = set(ids.tolist())

    # shape only for objects that can act as a bridge: those in some panel
    member_rows = sorted({row_of[int(b)] for v in base.values() for b, _ in v
                          if int(b) in row_of})
    if verbose:
        print(f"  computing shape for {len(member_rows):,} panel members "
              f"(of {len(obj_id):,} objects)...", flush=True)
    shape = object_shape(obj_id, ptr, pos, member_rows)
    def centroid_of(k): return shape[k][0]
    def elong_of(k): return shape[k][2]
    def extent_of(k): return shape[k][3]

    # closest point of each labelled atom to each of its panel members, reused
    # by every setting: the attachment geometry does not depend on the filter
    attach: dict[tuple[int, int], np.ndarray] = {}
    for a, v in base.items():
        Pa = seed_pts[a]
        if len(Pa) > MAX_SEED_POINTS:
            Pa = Pa[::max(1, len(Pa) // MAX_SEED_POINTS)][:MAX_SEED_POINTS]
        for b, _ in v:
            k = row_of.get(int(b))
            if k is None:
                continue
            Pb = pos[ptr[k]:ptr[k + 1]]
            if not len(Pb) or not len(Pa):
                continue
            d, j = cKDTree(Pb).query(Pa, k=1)
            s = int(np.argmin(d))
            attach[(a, int(b))] = Pb[j[s]]

    rows = {}
    for name, cfg in SETTINGS:
        min_el = cfg.get("min_elong", 0.0)
        max_ex = cfg.get("max_extent_nm", np.inf)
        min_th = cfg.get("min_through_deg", 0.0)

        # 1. filter which objects may act as a bridge at all
        ok_bridge = set()
        for b in {int(x) for v in base.values() for x, _ in v}:
            k = row_of.get(b)
            if k is None or k not in shape:
                continue
            if b in labelled_set or (elong_of(k) >= min_el
                                     and extent_of(k) <= max_ex):
                ok_bridge.add(b)

        panel = {a: [(b, d) for b, d in v if int(b) in ok_bridge]
                 for a, v in base.items()}

        # 2. the through-angle rule, applied per bridging object
        if min_th > 0:
            by_bridge: dict[int, list[int]] = {}
            for a, v in panel.items():
                for b, _ in v:
                    if int(b) not in labelled_set:
                        by_bridge.setdefault(int(b), []).append(a)
            drop: set[tuple[int, int]] = set()
            for b, seeds in by_bridge.items():
                if len(seeds) < 2:
                    continue
                k = row_of[b]
                px = centroid_of(k)
                keep_pairs = set()
                for i in range(len(seeds)):
                    for j in range(i + 1, len(seeds)):
                        a1, a2 = seeds[i], seeds[j]
                        p1, p2 = attach.get((a1, b)), attach.get((a2, b))
                        if p1 is None or p2 is None:
                            continue
                        if _through_angle(p1, px, p2) >= min_th:
                            keep_pairs.add(a1); keep_pairs.add(a2)
                for a in seeds:
                    if a not in keep_pairs:
                        drop.add((a, b))
            panel = {a: [(b, d) for b, d in v if (a, int(b)) not in drop]
                     for a, v in panel.items()}

        ch = _chained(panel, truth, labelled_set, max(HOPS))
        sizes = np.array([len(v) for v in panel.values()], float)
        c3 = ch[max(HOPS)]
        rows[name] = {
            "config": cfg,
            "n_bridge_objects_allowed": len(ok_bridge),
            "median_panel": float(np.median(sizes)) if len(sizes) else 0.0,
            **{f"chained_h{h}": ch[h] for h in HOPS},
        }
        print(f"  {name:<20} recall {c3['recall']:6.1%}  "
              f"precision {c3['precision']:7.3%}  "
              f"reachable-labelled {c3['median_reachable_labelled']:7.0f}  "
              f"bridges {len(ok_bridge):>7,}", flush=True)

    return {"rows": rows, "n_truth": len(truth), "n_labelled": int(len(ids)),
            "n_objects": int(len(obj_id)), "clouds_meta": meta}


def run(ctx: Context) -> Outcome:
    root = ctx.root
    m = measure(root / CLOUDS, root / OBJECTS, root / TOPOLOGY, root / LABELS_NPZ)
    rows, truth_n, ids_n = m["rows"], m["n_truth"], m["n_labelled"]

    best_name, best = None, None
    for name, r in rows.items():
        if name == "none":
            continue
        c = r[f"chained_h{max(HOPS)}"]
        if (c["recall"] >= BAR_RECALL
                and c["median_reachable_labelled"] <= BAR_MAX_REACHABLE):
            if best is None or c["recall"] > best["recall"]:
                best_name, best = name, c
    passed = best is not None

    base3 = rows["none"][f"chained_h{max(HOPS)}"]
    tightest = rows[SETTINGS[-1][0]][f"chained_h{max(HOPS)}"]
    return Outcome(
        passed=passed,
        observed={
            "best_setting": best_name or "none met the bar",
            "best_recall": best["recall"] if best else None,
            "best_precision": best["precision"] if best else None,
            "best_reachable_labelled":
                best["median_reachable_labelled"] if best else None,
            "unconstrained_recall": base3["recall"],
            "unconstrained_precision": base3["precision"],
            "unconstrained_reachable_labelled":
                base3["median_reachable_labelled"],
            "tightest_recall": tightest["recall"],
            "tightest_precision": tightest["precision"],
            "tightest_reachable_labelled": tightest["median_reachable_labelled"],
            "n_spanning_links": truth_n,
        },
        population={"n_labelled_atoms": ids_n,
                    "n_spanning_links": truth_n,
                    "n_objects": m["n_objects"]},
        tables={"by_setting": rows, "clouds_meta": m["clouds_meta"]},
        note=(f"{best_name} reaches {best['recall']:.1%} of spanning links at a "
              f"median of {best['median_reachable_labelled']:.0f} reachable "
              f"labelled atoms (precision {best['precision']:.2%}), where "
              f"unconstrained chaining needed "
              f"{base3['median_reachable_labelled']:.0f}. Structure prunes what "
              f"distance could not."
              if passed else
              f"no constraint setting met the bar. Unconstrained: recall "
              f"{base3['recall']:.1%} at {base3['median_reachable_labelled']:.0f} "
              f"reachable labelled atoms. Tightest "
              f"({SETTINGS[-1][0]}): recall {tightest['recall']:.1%} at "
              f"{tightest['median_reachable_labelled']:.0f}, precision "
              f"{tightest['precision']:.3%}. Cheap object-level structure does "
              f"not separate a real continuation from an incidental "
              f"neighbour; the constraints that would have to work are the ones "
              f"needing a skeleton (tangent at the cut face, caliber "
              f"continuity) or an identity signal (EXP-057C)"),
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
