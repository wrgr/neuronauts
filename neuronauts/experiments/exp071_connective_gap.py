"""EXP-071 — are a cell's fragments separated by distance, or by objects we omitted?

Three experiments have now failed to generate candidates by geometry. EXP-060's
ceiling: only 47.4% of true pairs have an endpoint inside 5 um. EXP-060B: 12%
recall at a usable panel, 65% only at a median panel of 3,870 objects. EXP-061:
the cone reaches 40% at best, at 42,000 distractors. EXP-070 then showed the
metric itself was wrong — endpoints, not objects — and fixing it moved the
ceiling to 75.7% without changing the verdict.

All four measured the same thing: the distance between two *synapse-anchored*
atoms of one cell. This asks whether that is the right measurement at all.

The population is built synapse-first — a v117 object enters it by owning a
synapse in the cube. A passing stretch of neurite with no synapse of its own
never enters, and that is exactly the material that joins two fragments which
do. If so, the geometry experiments were measuring the width of a hole the
substrate made, and the fix is upstream of every scorer and solver.

The test walks the **real level-2 graph of the proofread cell** — ground truth
a proposer does not have, used here only to measure, never to propose — and asks
three things of it:

1. **Direct contacts.** Do two v117 atoms of one cell ever share a level-2 edge?
   The expected answer is no, structurally: had the chunkedgraph joined them
   they would be one atom. A nonzero count would mean the atom definition or the
   join is wrong, so this is a correctness check as much as a measurement.
2. **Nearest-sibling hops.** For each labelled fragment, the graph distance to
   the nearest *other* labelled fragment of the same cell, counted through
   everything. Nearest sibling, not the clique: EXP-060 measured all same-owner
   pairs and got a median of 6.5 um from a distribution dominated by distal
   pairs no proposer should be asked to find, and CORRECTION.md had to withdraw
   the conclusion. The same error on this graph gives a median of 102 hops
   against a nearest-sibling median of 2.
3. **What the connective material is.** Every level-2 node on those paths that
   belongs to no population atom, resolved back to the v117 object that owned
   it, and sized. Dust and cable have different implications: cable means the
   population is missing ordinary objects and should be widened; dust means the
   fragments are joined through debris, which is a worse problem.

**The bar comes from a probe on disjoint cells.** This was first run by hand on
the twelve cells with the most labelled fragments, which found a nearest-sibling
median of 2 hops, 74.7% within 3 (exploratory probe; held-out result: median 3,
60.1%), and zero direct contacts. Declaring a bar
after seeing that would be worthless, so those twelve are **excluded here** and
the bar below is the probe's finding, tested on cells it never saw.

    python -m neuronauts.experiments.exp071_connective_gap
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import requests
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from neuronauts.data import lineage as L
from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.labels import TIER_NONE, load_labels
from neuronauts.harness.substrate import region_bounds

TOPOLOGY = "data/substrate/topology/kall.npz"
OBJGEOM = "data/substrate/geom/objgeom_kall.npz"
LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"
CACHE = "data/external/cell_l2_graphs"

CENTRE_UM = [663.0, 591.0, 860.0]
OUTER_SIDE_UM = 200.0          # geometry bounds; matches the atom fetch
V117_TS = 1623399000

#: The twelve cells the exploratory probe used. Excluded so the bar is tested
#: on cells that did not set it.
PROBE_CELLS = {
    864691135566902039, 864691135942123174, 864691135499624723,
    864691136312057178, 864691135928477966, 864691135493686111,
    864691136392806271, 864691136312445786, 864691135867057372,
    864691135375888840, 864691136145574196, 864691135124690471,
}

MIN_FRAGMENTS = 4              # a cell needs several fragments to have a "nearest"
N_CELLS = 40

#: Declared from the probe, before this run, on cells the probe never touched.
BAR_MEDIAN_HOPS = 5.0
BAR_FRAC_WITHIN_3 = 0.50
BAR_CONNECTIVE_MISSING = 0.80  # share of connective objects absent from the population

SPEC = Spec(
    id="EXP-071",
    title="Contact adjacency and the connective gap",
    question="Are the fragments of one cell separated by distance, or by "
             "objects the synapse-anchored population omits?",
    criterion=f"on at least 20 proofread cells DISJOINT from the twelve the "
              f"exploratory probe used: median nearest-sibling hop distance at "
              f"most {BAR_MEDIAN_HOPS:.0f} AND at least "
              f"{BAR_FRAC_WITHIN_3:.0%} of fragments within 3 hops AND at least "
              f"{BAR_CONNECTIVE_MISSING:.0%} of the objects holding the "
              f"connective material absent from the population. Direct "
              f"atom-to-atom level-2 contacts are reported as a correctness "
              f"check and are expected to be zero. Hops are to the NEAREST "
              f"sibling, never the clique",
    requires_ran=["EXP-057", "EXP-070"],
    inputs=[TOPOLOGY, OBJGEOM, LABELS_NPZ],
    params={"centre_um": CENTRE_UM, "outer_side_um": OUTER_SIDE_UM,
            "v117_timestamp": V117_TS, "n_cells": N_CELLS,
            "min_fragments": MIN_FRAGMENTS,
            "excluded_probe_cells": sorted(PROBE_CELLS),
            "bar_median_hops": BAR_MEDIAN_HOPS,
            "bar_frac_within_3": BAR_FRAC_WITHIN_3,
            "bar_connective_missing": BAR_CONNECTIVE_MISSING},
    flags={"network": True, "synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


def _cell_graph(cell: int, bstr: str, cache: Path) -> np.ndarray | None:
    """Level-2 edge list of one proofread cell, cached so a re-run is offline."""
    f = cache / f"{cell}.npz"
    if f.exists():
        with np.load(f, allow_pickle=False) as z:
            return z["edges"]
    url = (f"{L.CG_SERVER}/segmentation/api/v1/table/{L.SEG_TABLE}"
           f"/node/{cell}/lvl2_graph")
    for attempt in range(3):
        try:
            r = requests.get(url, headers=L._headers(L.DEFAULT_TOKEN),
                             params={"bounds": bstr}, timeout=180)
        except Exception:                                     # noqa: BLE001
            time.sleep(2.0 * (attempt + 1)); continue
        if r.status_code == 200:
            e = r.json().get("edge_graph", [])
            E = (np.asarray(e, np.uint64).reshape(-1, 2) if len(e)
                 else np.zeros((0, 2), np.uint64))
            cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, edges=E)
            return E
        if r.status_code in (400, 404):
            return None
        time.sleep(2.0 * (attempt + 1))
    return None


def _labelled(atoms, labels):
    idx = labels.index_of(atoms)
    has = idx >= 0
    owner = np.zeros(len(atoms), np.int64)
    pure = np.zeros(len(atoms), bool)
    tier = np.full(len(atoms), TIER_NONE, np.int8)
    owner[has] = labels.owner[idx[has]].astype(np.int64)
    pure[has] = labels.pure[idx[has]]
    tier[has] = labels.owner_tier[idx[has]]
    keep = pure & (tier > TIER_NONE) & (owner > 0)
    return atoms[keep], owner[keep]


def run(ctx: Context) -> Outcome:
    root = ctx.root
    with np.load(root / TOPOLOGY, allow_pickle=False) as z:
        atoms = z["atom_id"]
    labels = load_labels(root / LABELS_NPZ)
    ids, own = _labelled(atoms, labels)

    with np.load(root / OBJGEOM, allow_pickle=False) as z:
        oa, optr, ol2 = z["atom_id"], z["node_ptr"], z["l2_id"]
    l2_atom = np.repeat(oa, np.diff(optr))
    o = np.argsort(ol2)
    ol2_s, l2_atom_s = ol2[o], l2_atom[o]
    pop_atoms = np.unique(atoms)

    _lo, _hi, seg = region_bounds(CENTRE_UM, OUTER_SIDE_UM)
    bstr = "_".join(f"{int(seg[i][0])}-{int(seg[i][1])}" for i in range(3))
    cache = root / CACHE

    uo, cnt = np.unique(own, return_counts=True)
    eligible = [(int(uo[i]), int(cnt[i])) for i in np.argsort(-cnt)
                if cnt[i] >= MIN_FRAGMENTS and int(uo[i]) not in PROBE_CELLS]
    targets = eligible[:N_CELLS]
    print(f"  {len(eligible):,} eligible cells (>= {MIN_FRAGMENTS} fragments, "
          f"probe's 12 excluded); using {len(targets)}", flush=True)

    rows, all_hops, unknown_l2 = [], [], []
    total_cross = n_ok = 0
    for cell, n_frag in targets:
        E = _cell_graph(cell, bstr, cache)
        if E is None or not len(E):
            continue
        members = set(ids[own == cell].tolist())
        nodes = np.unique(E)
        n = len(nodes)
        pos_of = {int(v): k for k, v in enumerate(nodes.tolist())}

        j = np.clip(np.searchsorted(ol2_s, nodes), 0, len(ol2_s) - 1)
        known = ol2_s[j] == nodes
        node_atom = np.where(known, l2_atom_s[j], np.uint64(0))
        is_mem = np.array([int(a) in members for a in node_atom.tolist()])
        unknown_l2.append(nodes[~known])

        ei = np.array([[pos_of[int(a)], pos_of[int(b)]] for a, b in E.tolist()])
        ka, kb = known[ei[:, 0]], known[ei[:, 1]]
        cross = int((ka & kb & (node_atom[ei[:, 0]] != node_atom[ei[:, 1]])).sum())
        total_cross += cross

        g = coo_matrix((np.ones(len(ei), np.int8), (ei[:, 0], ei[:, 1])),
                       shape=(n, n))
        ncomp, _ = connected_components(g, directed=False)

        adj: list[list[int]] = [[] for _ in range(n)]
        for a, b in ei.tolist():
            adj[a].append(b); adj[b].append(a)
        mem_nodes: dict[int, list[int]] = {}
        for k in np.flatnonzero(is_mem).tolist():
            mem_nodes.setdefault(int(node_atom[k]), []).append(k)
        mem_ids = sorted(mem_nodes)
        if len(mem_ids) < 2:
            continue

        nearest, clique = [], []
        for src in mem_ids:
            dist = np.full(n, -1, np.int32)
            q = deque()
            for s in mem_nodes[src]:
                dist[s] = 0; q.append(s)
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1; q.append(v)
            best = None
            for dst in mem_ids:
                if dst == src:
                    continue
                c = [int(dist[k]) for k in mem_nodes[dst] if dist[k] >= 0]
                if not c:
                    continue
                d = min(c)
                clique.append(d)
                if best is None or d < best:
                    best = d
            if best is not None:
                nearest.append(best)
        if not nearest:
            continue

        nn = np.asarray(nearest)
        all_hops.extend(nearest)
        n_ok += 1
        rows.append({
            "cell": str(cell), "n_labelled_fragments": n_frag,
            "fragments_present": len(mem_ids), "n_nodes": n,
            "n_edges": int(len(E)), "n_components": int(ncomp),
            "frac_l2_known_to_population": round(float(known.mean()), 4),
            "direct_atom_contacts": cross,
            "nearest_sibling_median": float(np.median(nn)),
            "nearest_sibling_frac_le_3": float((nn <= 3).mean()),
            "clique_median": float(np.median(clique)) if clique else None,
        })
        print(f"  {cell}  frags {len(mem_ids):>3}  nodes {n:>6,}  "
              f"known {known.mean():5.1%}  contacts {cross}  "
              f"nearest med {np.median(nn):.0f}  (clique med "
              f"{np.median(clique) if clique else float('nan'):.0f})", flush=True)

    hops = np.asarray(all_hops)
    med = float(np.median(hops)) if len(hops) else float("nan")
    f3 = float((hops <= 3).mean()) if len(hops) else float("nan")
    clique_med = float(np.median([r["clique_median"] for r in rows
                                  if r["clique_median"] is not None])) if rows else None

    # --- what holds the connective material -------------------------------
    unk = np.unique(np.concatenate(unknown_l2)) if unknown_l2 else np.zeros(0, np.uint64)
    conn = {"n_unknown_l2": int(len(unk))}
    if len(unk):
        roots = np.asarray(L.roots_at(unk.tolist(), V117_TS), np.uint64)
        ok = roots > 0
        ur, uc = np.unique(roots[ok], return_counts=True)
        in_pop = np.isin(ur, pop_atoms)
        conn.update({
            "n_resolved_to_v117": int(ok.sum()),
            "frac_resolved": round(float(ok.mean()), 4),
            "n_objects": int(len(ur)),
            "n_objects_in_population": int(in_pop.sum()),
            "n_objects_missing": int((~in_pop).sum()),
            "frac_objects_missing": round(float((~in_pop).mean()), 4),
            "missing_l2_per_object": {
                "median": float(np.median(uc[~in_pop])) if (~in_pop).any() else None,
                "p90": float(np.percentile(uc[~in_pop], 90)) if (~in_pop).any() else None,
                "max": int(uc[~in_pop].max()) if (~in_pop).any() else None,
            },
            "missing_objects_with_ge_2_l2": int((uc[~in_pop] >= 2).sum()),
            "share_of_connective_nodes_in_ge2_objects": round(float(
                uc[~in_pop][uc[~in_pop] >= 2].sum() / max(uc[~in_pop].sum(), 1)), 4),
        })
        print(f"\n  connective material: {len(unk):,} L2 nodes unknown to the "
              f"population, {conn['frac_resolved']:.1%} resolve to a v117 object",
              flush=True)
        print(f"  -> {conn['n_objects']:,} objects, "
              f"{conn['frac_objects_missing']:.1%} absent from the population; "
              f"median {conn['missing_l2_per_object']['median']:.0f} L2 nodes each",
              flush=True)

    passed = bool(
        n_ok >= 20 and np.isfinite(med) and med <= BAR_MEDIAN_HOPS
        and f3 >= BAR_FRAC_WITHIN_3
        and conn.get("frac_objects_missing", 0.0) >= BAR_CONNECTIVE_MISSING)

    note = (
        f"on {n_ok} held-out cells the nearest labelled sibling is a median "
        f"{med:.0f} level-2 hops away ({f3:.0%} within 3), through material "
        f"{conn.get('frac_objects_missing', float('nan')):.0%} of which belongs "
        f"to v117 objects the synapse-anchored population never enumerated "
        f"(median {conn.get('missing_l2_per_object', {}).get('median', float('nan')):.0f} "
        f"L2 nodes each -- ordinary cable, not dust). Direct atom-to-atom "
        f"contacts: {total_cross}, as expected -- two v117 atoms cannot be "
        f"level-2 adjacent or they would be one atom, so 'is there a contact' "
        f"is the wrong query. The clique median is {clique_med:.0f} hops against "
        f"a nearest-sibling median of {med:.0f}: the same denominator error "
        f"CORRECTION.md caught in EXP-060, on a different substrate. The "
        f"proximity experiments were measuring the width of a hole the "
        f"substrate made, not the distance to the neighbour."
        if passed else
        f"bar not met: {n_ok} cells, median {med:.1f} hops (bar "
        f"{BAR_MEDIAN_HOPS:.0f}), {f3:.0%} within 3 (bar "
        f"{BAR_FRAC_WITHIN_3:.0%}), connective objects missing "
        f"{conn.get('frac_objects_missing', float('nan')):.0%} (bar "
        f"{BAR_CONNECTIVE_MISSING:.0%})")

    return Outcome(
        passed=passed,
        observed={
            "n_cells": n_ok,
            "nearest_sibling_median_hops": med,
            "nearest_sibling_frac_within_3": round(f3, 4) if np.isfinite(f3) else None,
            "clique_median_hops": clique_med,
            "direct_atom_contacts": total_cross,
            "frac_connective_objects_missing":
                conn.get("frac_objects_missing"),
            "connective_objects_missing": conn.get("n_objects_missing"),
            "median_l2_nodes_per_missing_object":
                conn.get("missing_l2_per_object", {}).get("median"),
        },
        population={
            "n_labelled_atoms": int(len(ids)),
            "n_eligible_cells": len(eligible),
            "n_cells_measured": n_ok,
            "n_fragments_measured": int(len(hops)),
            "probe_cells_excluded": len(PROBE_CELLS),
        },
        tables={"per_cell": rows, "connective_material": conn,
                "hop_histogram": {
                    str(k): int((hops == k).sum()) for k in range(2, 13)
                } if len(hops) else {}},
        note=note,
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
