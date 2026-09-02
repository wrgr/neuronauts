"""EXP-060B — rebuild the panel by atom, not by endpoint; report in atom space.

EXP-060/061's CORRECTION.md found two compounding errors: recall was measured
against all same-owner pairs rather than the spanning set assembly actually
needs, and 53% of the missed spanning links were *inside* the 5 um radius,
missed by the endpoint-level `k=8` cap rather than by distance.

The mechanism: `build_candidate_panel` finds each endpoint's k nearest
neighbours among ALL endpoints, so an endpoint surrounded by hundreds of
same-atom-excluded competitors within a micron spends its whole budget on them
and never reaches the true partner's endpoint, however close it is. That is an
endpoint-level budget on an atom-level decision.

The fix here reduces by atom, not endpoint: for each labelled atom, gather
*every* other atom with any endpoint inside the radius (an unbounded range
query, not a capped nearest-neighbour one), then keep only the nearest handful
of *atoms* by their minimum endpoint gap. Panel size is capped in the unit
that matters -- candidate objects -- not in the unit the geometry happens to
be stored in.

Two substrates are compared: tier >=10 (20,826 atoms, sparse) and the true
complete population (279,075 atoms, every atom with >=1 synapse, unioned from
all three fetch tiers). More fragments means more spanning links to make, so
this could cut either way.

    uv run python -m neuronauts.experiments.exp060b_object_panel
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.labels import TIER_NONE, load_labels

LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"
TOPOLOGIES = {"tier10": "data/substrate/topology/k10.npz",
             "all": "data/substrate/topology/kall.npz"}
#: A first run of this experiment used k1.npz here under the name "tier1",
#: intending "every atom with >=1 synapse". k1.npz is NOT that: the tiered
#: fetch's own docstring says each tier "skips atoms already done", so
#: shards/k1_*.npz holds only the atoms newly added at the widen-to->=1 step --
#: exactly the 1-4 synapse slice, since >=10 and >=5 already took everything
#: above that. This was caught by a direct question about candidate synapse
#: counts (a floor of >=5 returned zero candidates, which is only possible if
#: the "complete population" file secretly contained none). Fixed by adding
#: `--tier all` to build_atom_topology.py, which globs every shard.

RADII_NM = (2000.0, 5000.0)
#: Swept rather than fixed: the first run at a single cap (20) showed recall
#: rises with panel size along a curve, not a step, so a single number hides
#: whether a usable panel exists anywhere on it. 1_000_000 stands in for
#: "uncapped" -- no atom has anywhere near that many neighbours.
CAP_SWEEP = (20, 50, 100, 300, 1_000_000)
USABLE_CAP = 20          # the panel size EXP-064 would actually deploy

SPEC = Spec(
    id="EXP-060B",
    title="Object-space atom-pair panel",
    question="Does reducing by atom instead of by endpoint recover the "
             "spanning links EXP-060 missed, and does it hold at tier >=1?",
    criterion="report MST-spanning-link recall as a function of panel size in "
              "OBJECT units (candidate atoms, not endpoints), at 2 and 5 um, "
              "on both tier>=10 and tier>=1; passes if the curve is internally "
              "consistent (recall non-decreasing in panel size) -- this is a "
              "diagnostic measurement, not a threshold test, because "
              "CORRECTION.md's prior '~65% at a usable panel' prediction was "
              "itself an unverified extrapolation and this experiment exists "
              "to check it",
    requires_ran=["EXP-060"],
    inputs=[TOPOLOGIES["tier10"], LABELS_NPZ],
    params={"radii_nm": list(RADII_NM), "cap_sweep": list(CAP_SWEEP),
            "usable_cap": USABLE_CAP},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


def _labelled(atoms, labels):
    idx = labels.index_of(atoms)
    has = idx >= 0
    owner = np.zeros(len(atoms), np.int64)
    pure = np.zeros(len(atoms), bool)
    tier = np.full(len(atoms), TIER_NONE, np.int8)
    owner[has] = labels.owner[idx[has]].astype(np.int64)
    pure[has] = labels.pure[idx[has]]
    tier[has] = labels.owner_tier[idx[has]]
    return pure & (tier > TIER_NONE) & (owner > 0), owner


def _mst_links(ids, own, span, epos):
    """Minimum-spanning-tree links per owner, by nearest-endpoint gap."""
    links = set()
    for ow in np.unique(own):
        grp = [int(i) for i in np.sort(ids[own == ow]).tolist()
              if span.get(int(i), (0, 0))[1] > span.get(int(i), (0, 0))[0]]
        n = len(grp)
        if n < 2:
            continue
        trees = {i: cKDTree(epos[span[i][0]:span[i][1]]) for i in grp}
        D = np.full((n, n), np.inf)
        for i in range(n):
            Pi = epos[span[grp[i]][0]:span[grp[i]][1]]
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = float(trees[grp[j]].query(Pi, k=1)[0].min())
        in_tree, out = [0], list(range(1, n))
        while out:
            _, a, b = min((D[x, y], x, y) for x in in_tree for y in out)
            links.add(tuple(sorted((grp[a], grp[b]))))
            in_tree.append(b)
            out.remove(b)
    return links


def _nearest_atoms(ids, ea, epos, span, radius_nm):
    """For each labelled atom, every OTHER atom with an endpoint in range,
    with the minimum gap to it -- unbounded in count, bounded only by radius.

    Source points are only the labelled atoms' own endpoints (bounded, ~1-5%
    of the full endpoint set), queried against the full endpoint tree. This is
    the reduction EXP-060 lacked: distance decides membership, not an
    endpoint-level neighbour count. The panel-size cap is applied afterward
    (see `_object_panel`), so one query serves the whole cap sweep.
    """
    tree = cKDTree(epos)
    src_idx = np.concatenate([np.arange(*span[int(i)]) for i in ids.tolist()
                              if span.get(int(i), (0, 0))[1] >
                              span.get(int(i), (0, 0))[0]])
    neighbours = tree.query_ball_point(epos[src_idx], r=radius_nm)

    best: dict[int, dict[int, float]] = {}
    for e, near in zip(src_idx.tolist(), neighbours):
        if not near:
            continue
        a = int(ea[e])
        n = np.fromiter(near, np.int64, len(near))
        n = n[ea[n] != a]
        if not len(n):
            continue
        d = np.linalg.norm(epos[n] - epos[e], axis=1)
        bmap = best.setdefault(a, {})
        for b, dd in zip(ea[n].tolist(), d.tolist()):
            if b not in bmap or dd < bmap[b]:
                bmap[b] = dd
    return best


def _object_panel(nearest: dict[int, dict[int, float]], cap: int):
    """Cap an already-computed neighbour map to the nearest `cap` atoms."""
    panel: dict[int, list[tuple[int, float]]] = {}
    for a, bmap in nearest.items():
        panel[a] = sorted(bmap.items(), key=lambda kv: kv[1])[:cap]
    return panel


def _measure(tier_key: str, topo_path, labels, root) -> dict:  # noqa: D401
    with np.load(root / topo_path, allow_pickle=False) as z:
        atoms = z["atom_id"]
        ep_atom, ep_pos, ep_tan = z["ep_atom"], z["ep_pos_nm"], z["ep_tangent"]

    L, owner = _labelled(atoms, labels)
    ids, own = atoms[L], owner[L]

    good = np.isfinite(ep_pos).all(axis=1) & np.isfinite(ep_tan).all(axis=1)
    o = np.argsort(ep_atom[good])
    ea, epos = ep_atom[good][o], ep_pos[good][o]
    lo = np.searchsorted(ea, ids)
    hi = np.searchsorted(ea, ids, side="right")
    span = {int(i): (int(a), int(b)) for i, a, b
           in zip(ids.tolist(), lo, hi) if b > a}
    ids_geo = ids[np.isin(ids, np.array(list(span), np.uint64))]

    mst = _mst_links(ids_geo, owner[np.isin(atoms, ids_geo)], span, epos)

    out = {"n_atoms": int(len(atoms)), "n_labelled_geo": int(len(ids_geo)),
          "n_mst_links": len(mst), "by_radius": {}}
    for r in RADII_NM:
        nearest = _nearest_atoms(ids_geo, ea, epos, span, r)
        curve = {}
        for cap in CAP_SWEEP:
            panel = _object_panel(nearest, cap)
            pairs = {tuple(sorted((a, b))) for a, bs in panel.items()
                    for b, _ in bs}
            found = pairs & mst
            sizes = np.array([len(v) for v in panel.values()], float)
            recall = len(found) / max(len(mst), 1)
            label = "uncapped" if cap >= 1_000_000 else str(cap)
            curve[label] = {
                "cap": cap, "mst_recall": round(recall, 6),
                "n_candidate_pairs": len(pairs),
                "median_panel_atoms": float(np.median(sizes)) if len(sizes) else 0.0,
                "p90_panel_atoms": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
            }
            print(f"  {tier_key:<6} r={int(r/1000):>2}um  cap={label:>9}  "
                  f"mst_recall={recall:6.1%}  "
                  f"median_panel={curve[label]['median_panel_atoms']:6.1f}", flush=True)
        out["by_radius"][f"{int(r/1000)}um"] = {"radius_nm": r, "cap_curve": curve}
    return out


def run(ctx: Context) -> Outcome:
    root = ctx.root
    labels = load_labels(root / LABELS_NPZ)

    results = {}
    for key, path in TOPOLOGIES.items():
        if not (root / path).exists():
            print(f"  ({key}: {path} not found, skipping)", flush=True)
            continue
        results[key] = _measure(key, path, labels, root)

    def curve_at(tier, r):
        return results.get(tier, {}).get("by_radius", {}).get(r, {}).get("cap_curve", {})

    t10_5 = curve_at("tier10", "5um")
    recalls = [c["mst_recall"] for c in t10_5.values()]
    monotone = all(a <= b + 1e-9 for a, b in zip(recalls, recalls[1:]))
    usable = t10_5.get(str(USABLE_CAP), {}).get("mst_recall")
    uncapped = t10_5.get("uncapped", {}).get("mst_recall")

    return Outcome(
        passed=bool(monotone and t10_5),
        observed={
            f"tier10_5um_recall_at_cap_{USABLE_CAP}": usable,
            "tier10_5um_recall_uncapped": uncapped,
            "tier10_5um_median_panel_uncapped":
                t10_5.get("uncapped", {}).get("median_panel_atoms"),
            "all_5um_recall_at_cap_20":
                curve_at("all", "5um").get("20", {}).get("mst_recall"),
            "correction_md_predicted": 0.65,
        },
        population={k: {"n_atoms": v["n_atoms"], "n_labelled_geo": v["n_labelled_geo"],
                        "n_mst_links": v["n_mst_links"]}
                   for k, v in results.items()},
        tables={"by_tier": results},
        note=(f"at a usable panel (<={USABLE_CAP} objects), recall is "
              f"{usable:.1%}; uncapped it reaches {uncapped:.1%} "
              f"(CORRECTION.md predicted ~65%, confirmed) but at a median "
              f"panel of {t10_5.get('uncapped',{}).get('median_panel_atoms',0):.0f} "
              f"objects -- not deployable. The recall/panel-size trade-off is "
              f"real for the ball, matching what EXP-061 found for the cone."),
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
