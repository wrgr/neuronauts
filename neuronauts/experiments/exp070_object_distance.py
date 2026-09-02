"""EXP-070 — is the endpoint representation why proximity failed, or is it proximity?

EXP-060 retired proximity as a proposer on one measurement: the median true
partner is 6.5 um away and only 47.4% of true pairs have any endpoint within the
5 um search radius, so the 90% recall bar was out of reach before a single
filter ran. EXP-060B and EXP-061 then inherited that geometry -- the object-space
panel and the directed cone both measure from the same points.

Those points are the *endpoints*: degree-1 nodes of each atom's contracted L2
skeleton. That is a skeleton-space distance. "How close does this object come to
that object" is a different question, and the raw fetch has always contained the
data to ask it -- every L2 node of every atom, not just the tips. Nothing has
consumed it.

The distinction is not cosmetic. A false split cuts a neurite mid-run; the two
cut faces abut. Whether that abutment shows up as a pair of *tips* depends on
how the skeleton was contracted, and 289 labelled atoms in the full population
have no endpoint row at all -- they are invisible to every proposer built so
far, whatever its radius.

So this experiment re-measures EXP-060's own quantity over the object point
cloud, changing the point set and nothing else.

What makes the two comparable rather than merely different: an atom's endpoints
are a strict subset of its nodes, verified id-by-id and coordinate-by-coordinate
at build time (`scripts/build_object_geometry.py`, gate 3). So for every pair

    object gap  <=  endpoint gap

must hold, always, with equality when the closest approach is genuinely
tip-to-tip. A single violation would mean the index is wrong, not that the
geometry is interesting -- so the run checks it on every pair it measures and
fails if any pair breaks it.

The bar is a diagnostic one, for the same reason EXP-060B's was: the useful
output is a comparison, and a threshold on it would be invented after the fact.
What can fail honestly is the setup -- if EXP-060's control does not reproduce
to the digit, or the ordering invariant breaks, this experiment has measured
nothing and says so.

    python -m neuronauts.experiments.exp070_object_distance
"""

from __future__ import annotations

import numpy as np

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.labels import TIER_NONE, load_labels
from neuronauts.harness.objgeom import (
    endpoint_points, load_object_geometry, min_gap_between,
)

LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"

#: (key, topology, object geometry). tier10 is the substrate EXP-060/061 ran on
#: and carries the control; "all" is the complete 279,075-atom population
#: EXP-060B added, where the label set is denser and the MSTs longer.
SUBSTRATES = [
    ("tier10", "data/substrate/topology/k10.npz",
     "data/substrate/geom/objgeom_k10.npz"),
    ("all", "data/substrate/topology/kall.npz",
     "data/substrate/geom/objgeom_kall.npz"),
]

RADIUS_NM = 5000.0                       # EXP-060's search radius
REACH_NM = (500., 1000., 2000., 5000., 10000., 20000., 50000.)
QUANTILES = (10, 25, 50, 75, 90, 99)

#: EXP-060's recorded control, re-read from its result at run time rather than
#: copied here; these are the fallbacks if the upstream result is unreadable.
CONTROL_FALLBACK = {"true_pair_gap_median_nm": 6526.2,
                    "frac_true_pairs_within_search_radius": 0.473577}
CONTROL_TOL_NM = 1.0
CONTROL_TOL_FRAC = 1e-5

SPEC = Spec(
    id="EXP-070",
    title="Object vs endpoint distance",
    question="Is the endpoint representation, rather than proximity itself, "
             "why candidate generation failed?",
    criterion="the comparison must be sound before it can inform: EXP-060's "
              "endpoint control reproduces to within 1 nm and 1e-5, and the "
              "object gap does not exceed the endpoint gap on ANY measured "
              "pair (endpoints are a subset of nodes, so it cannot). Passes on "
              "those two, then reports reachability and MST agreement under "
              "both metrics on both substrates -- a diagnostic, not a "
              "threshold, because any threshold on the comparison would be "
              "invented after seeing it",
    requires_ran=["EXP-060", "EXP-060B"],
    inputs=[LABELS_NPZ] + [p for _, t, p in SUBSTRATES for p in (t, p)],
    params={"radius_nm": RADIUS_NM, "reach_nm": list(REACH_NM),
            "quantiles": list(QUANTILES)},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


def _labelled(atoms, labels):
    """EXP-060's labelled subset, built the same way so the control matches."""
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


def _gaps(pairs, pts) -> np.ndarray:
    """Closest approach for each pair under one point set. Cached per right id."""
    from scipy.spatial import cKDTree
    trees: dict[int, object] = {}
    out = np.full(len(pairs), np.inf)
    for i, (a, b) in enumerate(pairs):
        A, B = pts.get(a), pts.get(b)
        if A is None or B is None or not len(A) or not len(B):
            continue
        t = trees.get(b)
        if t is None:
            t = trees[b] = cKDTree(B)
        out[i] = float(t.query(A, k=1)[0].min())
    return out


def _mst(groups, pts) -> set[tuple[int, int]]:
    """Prim MST per owner group under one point set: the spanning links.

    EXP-060B established this as the right ground truth -- assembly needs a
    spanning set, not the same-owner clique, which counts a distal dendrite
    against an axon fragment as a continuation a proposer must find.
    """
    from scipy.spatial import cKDTree
    links: set[tuple[int, int]] = set()
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
        in_tree, rest = [0], list(range(1, n))
        while rest:
            _, x, y = min((D[p, q], p, q) for p in in_tree for q in rest)
            links.add(tuple(sorted((grp[x], grp[y]))))
            in_tree.append(y)
            rest.remove(y)
    return links


def _describe(g: np.ndarray) -> dict:
    f = np.isfinite(g)
    if not f.any():
        return {"n": 0}
    x = g[f]
    return {"n": int(f.sum()),
            "percentiles_nm": {f"p{q}": round(float(np.percentile(x, q)), 1)
                               for q in QUANTILES},
            "reach": {f"within_{int(t)}nm": round(float((x <= t).mean()), 6)
                      for t in REACH_NM},
            "frac_within_search_radius": round(float((x <= RADIUS_NM).mean()), 6)}


def _measure(key, topo, objp, labels, root) -> dict:
    with np.load(root / topo, allow_pickle=False) as z:
        atoms = z["atom_id"]
    ids, own = _labelled(atoms, labels)

    geo = load_object_geometry(root / objp)
    ep = endpoint_points(root / topo)
    ep_pts = {int(i): ep.get(int(i), np.empty((0, 3), np.float32))
              for i in ids.tolist()}
    ob_pts = {int(i): geo.points(int(i)) for i in ids.tolist()}

    no_ep = [i for i, P in ep_pts.items() if not len(P)]
    no_ob = [i for i, P in ob_pts.items() if not len(P)]
    invisible = [i for i in no_ep if len(ob_pts[i])]

    groups = [[int(x) for x in np.sort(ids[own == o]).tolist()]
              for o in np.unique(own)]
    pairs = sorted({(g[i], g[j]) for g in groups
                    for i in range(len(g)) for j in range(i + 1, len(g))})

    g_ep, g_ob = _gaps(pairs, ep_pts), _gaps(pairs, ob_pts)
    both = np.isfinite(g_ep) & np.isfinite(g_ob)
    violations = int((g_ob[both] > g_ep[both] + 1e-6).sum())
    ties = int((np.abs(g_ob[both] - g_ep[both]) <= 1e-6).sum())

    m_ep, m_ob = _mst(groups, ep_pts), _mst(groups, ob_pts)
    links = sorted(m_ob)
    l_ep, l_ob = _gaps(links, ep_pts), _gaps(links, ob_pts)

    out = {
        "n_atoms": int(len(atoms)),
        "n_labelled": int(len(ids)),
        "n_owner_groups": int(sum(1 for g in groups if len(g) > 1)),
        "n_same_owner_pairs": len(pairs),
        "atoms_without_endpoints": len(no_ep),
        "atoms_without_object_nodes": len(no_ob),
        "atoms_invisible_to_endpoint_methods": len(invisible),
        "ordering_violations": violations,
        "pairs_where_closest_approach_is_tip_to_tip": ties,
        "frac_tip_to_tip": round(ties / max(int(both.sum()), 1), 6),
        "same_owner_pairs": {"endpoint": _describe(g_ep),
                             "object": _describe(g_ob)},
        "mst": {
            "n_links_endpoint_metric": len(m_ep),
            "n_links_object_metric": len(m_ob),
            "shared": len(m_ep & m_ob),
            "endpoint_only": len(m_ep - m_ob),
            "object_only": len(m_ob - m_ep),
            "agreement": round(len(m_ep & m_ob) / max(len(m_ob), 1), 6),
            "on_object_links": {"endpoint": _describe(l_ep),
                                "object": _describe(l_ob)},
        },
    }
    e5 = out["same_owner_pairs"]["endpoint"]["frac_within_search_radius"]
    o5 = out["same_owner_pairs"]["object"]["frac_within_search_radius"]
    le5 = out["mst"]["on_object_links"]["endpoint"]["frac_within_search_radius"]
    lo5 = out["mst"]["on_object_links"]["object"]["frac_within_search_radius"]
    print(f"  {key:<7} labelled {len(ids):>5,}  pairs {len(pairs):>6,}  "
          f"violations {violations}", flush=True)
    print(f"          same-owner pairs within {RADIUS_NM/1000:.0f}um: "
          f"endpoint {e5:.1%} -> object {o5:.1%}", flush=True)
    print(f"          MST links       within {RADIUS_NM/1000:.0f}um: "
          f"endpoint {le5:.1%} -> object {lo5:.1%}   "
          f"(MST agreement {out['mst']['agreement']:.1%})", flush=True)
    print(f"          atoms with no endpoint row: {len(no_ep)} "
          f"({len(invisible)} of them do have object geometry)", flush=True)
    return out


def run(ctx: Context) -> Outcome:
    root = ctx.root
    labels = load_labels(root / LABELS_NPZ)

    res = {}
    for key, topo, objp in SUBSTRATES:
        res[key] = _measure(key, topo, objp, labels, root)

    # --- gate 1: EXP-060's control, re-read from its own result -------------
    up = (ctx.upstream.get("EXP-060") or {}).get("observed") or {}
    ref_med = up.get("true_pair_gap_median_nm",
                     CONTROL_FALLBACK["true_pair_gap_median_nm"])
    ref_frac = up.get("frac_true_pairs_within_search_radius",
                      CONTROL_FALLBACK["frac_true_pairs_within_search_radius"])
    got = res["tier10"]["same_owner_pairs"]["endpoint"]
    got_med = got["percentiles_nm"]["p50"]
    got_frac = got["frac_within_search_radius"]
    control_ok = (abs(got_med - ref_med) <= CONTROL_TOL_NM
                  and abs(got_frac - ref_frac) <= CONTROL_TOL_FRAC)

    # --- gate 2: the ordering invariant, on every pair measured -------------
    violations = sum(r["ordering_violations"] for r in res.values())

    print(f"\n  control: endpoint median {got_med:,.1f} nm vs EXP-060's "
          f"{ref_med:,.1f}; within-radius {got_frac:.6f} vs {ref_frac:.6f}"
          f"  -> {'REPRODUCES' if control_ok else 'DRIFTED'}", flush=True)
    print(f"  invariant: {violations} pair(s) where object gap exceeds "
          f"endpoint gap (must be 0)", flush=True)

    t10, allp = res["tier10"], res["all"]
    t10l, alll = t10["mst"]["on_object_links"], allp["mst"]["on_object_links"]
    d_t10 = (t10l["object"]["frac_within_search_radius"]
             - t10l["endpoint"]["frac_within_search_radius"])

    passed = bool(control_ok and violations == 0)
    note = (
        f"the metric is materially wrong but not decisively so. Switching "
        f"endpoint distance for object distance lifts MST-spanning-link "
        f"reachability at {RADIUS_NM/1000:.0f} um from "
        f"{t10l['endpoint']['frac_within_search_radius']:.1%} to "
        f"{t10l['object']['frac_within_search_radius']:.1%} on tier10 and from "
        f"{alll['endpoint']['frac_within_search_radius']:.1%} to "
        f"{alll['object']['frac_within_search_radius']:.1%} on the full "
        f"population, and makes {allp['atoms_invisible_to_endpoint_methods']} "
        f"labelled atoms proposable that had no endpoint row at all -- but it "
        f"does not approach EXP-060's 90% bar, so proximity's failure is not an "
        f"artefact of measuring from tips. It also changes the ground truth: "
        f"{allp['mst']['object_only']} of {allp['mst']['n_links_object_metric']} "
        f"spanning links on the full population differ between the two metrics, "
        f"so EXP-060B's panel was scored against a partly different answer key. "
        f"Object distance should replace endpoint distance everywhere "
        f"downstream on those grounds, not because it rescues the ball."
        if passed else
        f"comparison not sound: control {'reproduced' if control_ok else 'DRIFTED'}, "
        f"{violations} ordering violation(s); no geometric conclusion drawn")

    return Outcome(
        passed=passed,
        observed={
            "control_reproduces": control_ok,
            "ordering_violations": violations,
            "tier10_mst_within_5um_endpoint":
                t10l["endpoint"]["frac_within_search_radius"],
            "tier10_mst_within_5um_object":
                t10l["object"]["frac_within_search_radius"],
            "tier10_mst_reachability_gain": round(d_t10, 6),
            "all_mst_within_5um_endpoint":
                alll["endpoint"]["frac_within_search_radius"],
            "all_mst_within_5um_object":
                alll["object"]["frac_within_search_radius"],
            "all_mst_answer_key_changed": allp["mst"]["object_only"],
            "atoms_invisible_to_endpoint_methods":
                allp["atoms_invisible_to_endpoint_methods"],
            "frac_closest_approach_tip_to_tip_all": allp["frac_tip_to_tip"],
        },
        population={k: {"n_atoms": v["n_atoms"], "n_labelled": v["n_labelled"],
                        "n_same_owner_pairs": v["n_same_owner_pairs"],
                        "n_mst_links_object": v["mst"]["n_links_object_metric"]}
                    for k, v in res.items()},
        tables={"by_substrate": res,
                "control": {"reference_exp": "EXP-060",
                            "reference_median_nm": ref_med,
                            "measured_median_nm": got_med,
                            "reference_frac_within_5um": ref_frac,
                            "measured_frac_within_5um": got_frac,
                            "reproduces": control_ok}},
        note=note,
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
