"""EXP-074 -- soma-seeded growth: what does distance alone achieve on the real task?

Every proposer from EXP-060 through EXP-073 scored pairwise join-finding, and
collapsed at about 0.09 percent precision on every substrate, radius, dust floor
and read resolution tried. The task a grammar performs is different: start at a
cell body and grow a tree outward, judging each join by whether the growing tree
stays well formed. This is the first experiment scored on that task.

The method here is deliberately NOT a grammar. It takes the nearest unclaimed
object within a radius of the tree it has built so far, adds it, and repeats
until nothing qualifies. No compartment, no caliber, no direction, nothing
learned. Its job is to establish what distance alone achieves so that a
grammar's contribution is measurable against it rather than asserted -- and if
distance alone clears the bars, to say so plainly.

Two populations, scored separately and never pooled, because they ask opposite
things of a grower (see docs/threads/soma_seeded_targets.md):

  67 cells that need joining   299 target fragments, 200 links, gap median
                               1,202 nm, and 80 percent of links within 2 um
  36 cells already whole       the soma fragment already holds the entire
                               in-box arbor; the correct output is nothing

Scoring separates three outcomes rather than folding them into one number. An
object the grower adds is *recovered* when it is in the target, *contamination*
when our own overlay says it belongs to a different proofread cell, and
*unknown* when it carries no label -- which is the usual case for connective
cable and is neither right nor wrong. Counting unknowns as errors would punish
the bridging the task requires; counting them as successes would hide
everything. They are reported on their own.

Bars, declared in docs/threads/exp074_spec.md before this was written:
recovery at least 60 percent at 2 um, purity at least 80 percent, abstention on
at least 70 percent of the already-whole cells.

    python -m neuronauts.experiments.exp074_seeded_growth
"""

from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.labels import TIER_NONE, load_labels

CLOUDS = "data/substrate/c100um/object_clouds_mip5.npz"
OBJECTS = "data/substrate/c100um/objects_v117_mip5.npz"
TOPOLOGY = "data/substrate/topology/kall.npz"
LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"
CARDS = "data/external/cell_cards"
SEEDS = "data/external/soma_viz/seed_census.json"

RADII_NM = (500.0, 1000.0, 2000.0, 3000.0)
BAR_RADIUS_NM = 2000.0
#: Same physical dust floor EXP-072 settled on: synapse-free objects below this
#: are debris, synapse-carrying ones stay whatever their size.
MIN_VOLUME_UM3 = 0.041
#: A grower with no stopping rule would eventually absorb the cube. Capped at a
#: multiple of the target size so a runaway is visible as a purity failure
#: rather than as an hour of compute.
MAX_ADDS_PER_CELL = 200

BAR_RECOVERY = 0.60
BAR_PURITY = 0.80
BAR_ABSTENTION = 0.70

SPEC = Spec(
    id="EXP-074",
    title="Soma-seeded growth, distance only",
    question="Can a grower seeded at a cell body recover that cell's in-box root "
             "process, and does it know when to stop?",
    criterion=f"scored on two populations separately, never pooled. On the 67 "
              f"cells that need joining, at radius {BAR_RADIUS_NM/1000:.0f} um: "
              f"recovery of target fragments at least {BAR_RECOVERY:.0%} "
              f"(micro-averaged) AND purity at least {BAR_PURITY:.0%}, where "
              f"purity counts only LABELLED objects added, since unlabelled "
              f"connective cable is neither right nor wrong and is reported "
              f"separately. On the 36 already-whole cells, add nothing in at "
              f"least {BAR_ABSTENTION:.0%} of them. The method uses distance "
              f"alone -- no compartment, caliber, direction or learned score",
    requires_ran=["EXP-071", "EXP-072"],
    inputs=[CLOUDS, OBJECTS, TOPOLOGY, LABELS_NPZ],
    params={"radii_nm": list(RADII_NM), "bar_radius_nm": BAR_RADIUS_NM,
            "min_volume_um3": MIN_VOLUME_UM3, "max_adds_per_cell": MAX_ADDS_PER_CELL,
            "bar_recovery": BAR_RECOVERY, "bar_purity": BAR_PURITY,
            "bar_abstention": BAR_ABSTENTION},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


#: Query points per object per growth step. A soma fragment carries a median
#: 72,210 cloud points, so querying all of them is the difference between a run
#: and a hang. A spread sample on a fixed stride finds the same neighbouring
#: objects: what matters is whether ANY point of the object comes within reach,
#: and the sample covers its extent.
MAX_QUERY_POINTS = 1500
#: Neighbours per query point. Growth wants the NEAREST unclaimed object, which
#: is a nearest-neighbour question, not a ball query -- a first version asked for
#: every point within the radius and got about 2,400 per query point, 175 million
#: for one step of a soma fragment. Asking for the nearest 32 within the radius
#: gives the same answer for anything the frontier would actually pop next. The
#: approximation only bites if an object's nearest point to the tree is beyond
#: the 32nd nearest overall, which cannot matter for a nearest-first heap.
NEIGHBOURS_PER_POINT = 32


def _sample(P, cap=MAX_QUERY_POINTS):
    return P if len(P) <= cap else P[::max(1, len(P) // cap)][:cap]


def grow(seed, tree, point_owner, obj_points, radius, max_adds):
    """Nearest-first growth from ``seed``. Returns the objects added, in order.

    Prim-shaped: when an object joins, only ITS points are queried and what they
    find goes on a heap keyed by distance, so the frontier is never rebuilt.

    A per-cell "local substrate" was tried first and abandoned: a soma
    fragment's bounding box plus a 20 um margin covers 100% of this cube
    (measured on 40 seeds), so masking to it kept 72.5M of 72.5M points and
    bought nothing.
    """
    claimed = {int(seed)}
    order, heap = [], []
    n_pts = len(point_owner)
    k = min(NEIGHBOURS_PER_POINT, n_pts)

    def push(obj):
        P = _sample(obj_points(obj))
        if not len(P):
            return
        d, j = tree.query(P, k=k, distance_upper_bound=radius)
        d = np.atleast_2d(d); j = np.atleast_2d(j)
        ok = np.isfinite(d) & (j < n_pts)      # scipy marks misses with inf / n
        if not ok.any():
            return
        own = point_owner[j[ok]]
        dd = d[ok]
        keep = ~np.isin(own, np.fromiter(claimed, np.uint64, len(claimed)))
        if not keep.any():
            return
        own, dd = own[keep], dd[keep]
        o = np.argsort(own, kind="stable")
        own_s, d_s = own[o], dd[o]
        uo, st = np.unique(own_s, return_index=True)
        for obj_id_, best in zip(uo.tolist(), np.minimum.reduceat(d_s, st).tolist()):
            heapq.heappush(heap, (float(best), int(obj_id_)))

    push(int(seed))
    while heap and len(order) < max_adds:
        dist, o = heapq.heappop(heap)
        if o in claimed:
            continue
        claimed.add(o)
        order.append({"object": o, "gap_nm": round(float(dist), 1)})
        push(o)
    return order


def run(ctx: Context) -> Outcome:
    root = ctx.root
    with np.load(root / CLOUDS, allow_pickle=False) as z:
        obj_id, ptr, pos = z["object_id"], z["node_ptr"], z["pos_nm"]
        nvox = (z["n_voxels_per_point"] if "n_voxels_per_point" in z.files
                else z["n_voxels"])
        meta = json.loads(bytes(z["meta"]).decode()) if "meta" in z.files else {}
    with np.load(root / OBJECTS, allow_pickle=False) as z:
        pop_ids, in_pop = z["object_id"], z["in_population"]
    vox_um3 = float(meta.get("resolution_nm", [256, 256, 160])[0]) * \
        float(meta.get("resolution_nm", [256, 256, 160])[1]) * \
        float(meta.get("resolution_nm", [256, 256, 160])[2]) / 1e9

    row_of = {int(a): k for k, a in enumerate(obj_id.tolist())}

    def obj_points(o):
        k = row_of.get(int(o))
        if k is None:
            return np.empty((0, 3), np.float32)
        return pos[int(ptr[k]):int(ptr[k + 1])]

    # dust floor, in physical units, synapse-carriers exempt
    per_obj_vox = np.add.reduceat(nvox, ptr[:-1]) if len(ptr) > 1 else nvox
    vol = per_obj_vox * vox_um3
    carries_syn = np.isin(obj_id, pop_ids[in_pop])
    keep_obj = carries_syn | (vol >= MIN_VOLUME_UM3)
    per = np.diff(ptr)
    point_owner_all = np.repeat(obj_id, per)
    sel = np.repeat(keep_obj, per)
    P_all, point_owner = pos[sel], point_owner_all[sel]
    print(f"  substrate: {int(keep_obj.sum()):,} of {len(obj_id):,} objects kept "
          f"({int(carries_syn.sum()):,} synapse-carrying), {len(P_all):,} points",
          flush=True)
    tree = cKDTree(P_all)
    print("  tree built", flush=True)

    # --- truth: the seeded targets and the cells' own labels ------------------
    labels = load_labels(root / LABELS_NPZ)
    with np.load(root / TOPOLOGY, allow_pickle=False) as z:
        atoms = z["atom_id"]
    idx = labels.index_of(atoms); has = idx >= 0
    owner = np.zeros(len(atoms), np.int64); pure = np.zeros(len(atoms), bool)
    tier = np.full(len(atoms), TIER_NONE, np.int8)
    owner[has] = labels.owner[idx[has]].astype(np.int64)
    pure[has] = labels.pure[idx[has]]; tier[has] = labels.owner_tier[idx[has]]
    trusted = pure & (tier > TIER_NONE) & (owner > 0)
    owner_of = {int(a): int(o) for a, o, t in zip(atoms, owner, trusted) if t}

    cards = {}
    for f in sorted((root / CARDS).glob("*.json")):
        if f.name.startswith("_"):
            continue
        c = json.load(open(f))
        if c.get("coverage", {}).get("graph"):
            cards[int(c["cell"])] = c

    rows, by_radius = [], {}
    for r in RADII_NM:
        need_rec_num = need_rec_den = pure_num = pure_den = 0
        unknown = 0
        abst_ok = abst_n = 0
        per_cell = []
        for cell, c in sorted(cards.items()):
            seed = int(c["seed"]["v117_fragment"])
            tgt = set(int(x) for x in c["structure"]["seeded_target"]) - {seed}
            whole = bool(c["structure"]["already_whole"])
            if seed not in row_of:
                continue
            added = grow(seed, tree, point_owner, obj_points, r, MAX_ADDS_PER_CELL)
            got = [a["object"] for a in added]
            rec = len(set(got) & tgt)
            lab = [o for o in got if o in owner_of]
            right = [o for o in lab if owner_of[o] == cell]
            unk = len(got) - len(lab)
            if whole:
                abst_n += 1; abst_ok += int(len(got) == 0)
            else:
                need_rec_num += rec; need_rec_den += len(tgt)
                pure_num += len(right); pure_den += len(lab)
                unknown += unk
            per_cell.append({"cell": str(cell), "already_whole": whole,
                             "target": len(tgt), "added": len(got),
                             "recovered": rec, "labelled_added": len(lab),
                             "correct_labelled": len(right), "unknown_added": unk})
        recovery = need_rec_num / max(need_rec_den, 1)
        purity = pure_num / max(pure_den, 1)
        abst = abst_ok / max(abst_n, 1)
        by_radius[f"r{int(r)}"] = {
            "radius_nm": r, "recovery": round(recovery, 6), "purity": round(purity, 6),
            "abstention": round(abst, 6), "target_fragments": need_rec_den,
            "recovered": need_rec_num, "labelled_added": pure_den,
            "correct_labelled": pure_num, "unknown_added": unknown,
            "abstained": abst_ok, "already_whole_cells": abst_n,
            "median_added_per_needing_cell": float(np.median(
                [p["added"] for p in per_cell if not p["already_whole"]] or [0])),
        }
        if r == BAR_RADIUS_NM:
            rows = per_cell
        q = by_radius[f"r{int(r)}"]
        print(f"  r={int(r/1000) if r>=1000 else r/1000:>4}um  recovery {recovery:6.1%}  "
              f"purity {purity:6.1%}  abstention {abst:6.1%}  "
              f"unknown added {unknown:>6,}  median added/cell "
              f"{q['median_added_per_needing_cell']:.0f}", flush=True)

    b = by_radius[f"r{int(BAR_RADIUS_NM)}"]
    passed = bool(b["recovery"] >= BAR_RECOVERY and b["purity"] >= BAR_PURITY
                  and b["abstention"] >= BAR_ABSTENTION)
    fails = [n for n, v, bar in (("recovery", b["recovery"], BAR_RECOVERY),
                                 ("purity", b["purity"], BAR_PURITY),
                                 ("abstention", b["abstention"], BAR_ABSTENTION))
             if v < bar]
    return Outcome(
        passed=passed,
        observed={"recovery_at_bar": b["recovery"], "purity_at_bar": b["purity"],
                  "abstention_at_bar": b["abstention"],
                  "unknown_objects_absorbed": b["unknown_added"],
                  "median_added_per_needing_cell": b["median_added_per_needing_cell"],
                  "failed_clauses": fails},
        population={"cells": len(cards),
                    "needing_joins": sum(1 for c in cards.values()
                                         if not c["structure"]["already_whole"]),
                    "already_whole": sum(1 for c in cards.values()
                                         if c["structure"]["already_whole"]),
                    "target_fragments": b["target_fragments"],
                    "objects_in_substrate": int(keep_obj.sum())},
        tables={"by_radius": by_radius, "per_cell_at_bar": rows},
        note=(f"distance alone at {BAR_RADIUS_NM/1000:.0f} um: recovery "
              f"{b['recovery']:.1%}, purity {b['purity']:.1%}, abstention "
              f"{b['abstention']:.1%}" +
              ("; clears every clause, so the grammar must be justified against "
               "this rather than against the pairwise line"
               if passed else
               f"; fails on {', '.join(fails)}. The grammar has to supply what "
               f"distance cannot, and this says which clause it must fix")),
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
