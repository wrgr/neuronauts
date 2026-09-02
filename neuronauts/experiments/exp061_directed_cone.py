"""EXP-061 — can a directed cone reach what a proximity ball cannot?

EXP-060 retired proximity as a proposer: the median true partner is 6.5 um
away and p90 is 56 um, so a 5 um ball reaches 47.4% of true pairs at best, and
a ball wide enough to reach 90% would hold ~2.7 million endpoints.

A cone is the obvious replacement because it spends its budget along the
neurite instead of around it. A 50 um cone at 15 degrees has under 2% of the
volume of a 50 um ball, so it can be long without being large. The question is
whether true partners actually lie along the outward tangent — if a fragment's
continuation is not where it was pointing, the cone buys nothing.

Two measurements, because the bar needs both:

**Reachability.** For each true pair, the angle between the outward tangent at
one atom's closest endpoint and the direction to the partner. A cone of
half-angle theta reaches the pair when that angle is under theta. This is an
upper bound on what any cone-based proposer can propose at that angle.

**Panel size.** For a sample of labelled endpoints, how many endpoints of
*other* atoms fall inside the same cone. A cone that reaches everything by
being wide is just a ball.

Both are measured against the real endpoint cloud, with all 20,826 tier-10
atoms present as distractors. The comparison is EXP-060's measured proximity
numbers: 47.4% reachable within 5 um, 17.5% actually proposed at k=8, median
panel 819.

**The chance baseline is measured, not assumed.** The reachability statistic
is a *minimum* over two directions (A's tangent toward B, and B's toward A), so
the chance that a pair is "reached" with random tangents is not the
single-direction (1 - cos theta)/2 -- it is the chance that the better of two
draws lands inside the cone, roughly twice that at small angles. The first run
used the single-direction number and reported 3-6x enrichment over chance; the
QA repro with random unit tangents through this same loop measured the real
null at about twice the stated one, so the enrichment was closer to 2-3x. The
null is now computed empirically: the same loop, run ``N_NULL`` times with
random unit tangents in place of the real ones, on the same pairs. The
closed-form single-direction number is kept in the tables, labelled as what it
is, so the correction is visible. The pass/fail bar does not involve the null
and is unchanged.

    python -m neuronauts.experiments.exp061_directed_cone
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.labels import TIER_NONE, load_labels

TOPOLOGY = "data/substrate/topology/k10.npz"
LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"

#: (max_distance_nm, half_angle_degrees)
CONES = [
    (10_000.0, 15.0), (10_000.0, 30.0), (10_000.0, 45.0),
    (25_000.0, 15.0), (25_000.0, 30.0), (25_000.0, 45.0),
    (50_000.0, 15.0), (50_000.0, 30.0), (50_000.0, 45.0),
]

#: Endpoints sampled for the panel-size measurement. The reachability figure
#: uses every true pair; only the distractor count is sampled, because it needs
#: a query against all 5.1M endpoints per endpoint.
PANEL_SAMPLE = 4000
SEED = 0
#: Random-tangent draws for the empirical chance baseline. The loop is cheap
#: (a few hundred pairs), so this is not a budget decision.
N_NULL = 20

REACH_BAR = 0.70
PANEL_BAR = 20.0

#: EXP-060's measured proximity numbers, for the comparison the bar is stated
#: against rather than an abstract target.
PROXIMITY_REACHABLE = 0.474
PROXIMITY_PROPOSED = 0.175
PROXIMITY_MEDIAN_PANEL = 819.0

SPEC = Spec(
    id="EXP-061",
    title="Directed cone vs proximity ball",
    question="Do true partners lie along the outward tangent, and is the cone "
             "that reaches them small?",
    criterion=f"a cone reaches at least {REACH_BAR:.0%} of true pairs at a "
              f"median panel of at most {PANEL_BAR:.0f} distractor endpoints -- "
              f"beating EXP-060's measured proximity ceiling of "
              f"{PROXIMITY_REACHABLE:.1%} reachable / {PROXIMITY_PROPOSED:.1%} "
              f"proposed at median panel {PROXIMITY_MEDIAN_PANEL:.0f}",
    requires_ran=["EXP-060"],
    inputs=[TOPOLOGY, LABELS_NPZ],
    params={"cones": CONES, "panel_sample": PANEL_SAMPLE, "seed": SEED},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


def _reach(pairs, span, epos, etan):
    """Best-of-two angle to the true partner, and the closest endpoint gap.

    Parameterised on the tangent field so the same code measures the real
    tangents and the random-direction null; anything the loop does to the
    statistic (the argmin over endpoints, the min over directions) is then
    done to the null too.
    """
    best_angle = np.full(len(pairs), np.nan)
    gap = np.full(len(pairs), np.inf)
    for i, (a, b) in enumerate(pairs):
        (a0, a1), (b0, b1) = span.get(a, (0, 0)), span.get(b, (0, 0))
        A, B = epos[a0:a1], epos[b0:b1]
        if not len(A) or not len(B):
            continue
        angs = []
        for P, T, Q in ((A, etan[a0:a1], B), (B, etan[b0:b1], A)):
            d, j = cKDTree(Q).query(P, k=1)
            s = int(np.argmin(d))
            v = Q[j[s]] - P[s]
            nv = float(np.linalg.norm(v))
            if nv <= 0:
                continue
            c = float(np.dot(T[s], v / nv))
            angs.append(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
            gap[i] = min(gap[i], float(d[s]))
        if angs:
            best_angle[i] = min(angs)
    return best_angle, gap


def run(ctx: Context) -> Outcome:
    root = ctx.root
    with np.load(root / TOPOLOGY, allow_pickle=False) as z:
        atoms = z["atom_id"]
        ep_atom, ep_pos, ep_tan = z["ep_atom"], z["ep_pos_nm"], z["ep_tangent"]

    labels = load_labels(root / LABELS_NPZ)
    idx = labels.index_of(atoms)
    has = idx >= 0
    owner = np.zeros(len(atoms), np.int64)
    pure = np.zeros(len(atoms), bool)
    tier = np.full(len(atoms), TIER_NONE, np.int8)
    owner[has] = labels.owner[idx[has]].astype(np.int64)
    pure[has] = labels.pure[idx[has]]
    tier[has] = labels.owner_tier[idx[has]]
    L = pure & (tier > TIER_NONE) & (owner > 0)
    ids, own = atoms[L], owner[L]

    good = np.isfinite(ep_pos).all(axis=1) & np.isfinite(ep_tan).all(axis=1)
    o = np.argsort(ep_atom[good])
    ea = ep_atom[good][o]
    epos = ep_pos[good][o]
    etan = ep_tan[good][o]
    lo = np.searchsorted(ea, ids)
    hi = np.searchsorted(ea, ids, side="right")
    span = {int(i): (int(a), int(b)) for i, a, b in zip(ids.tolist(), lo, hi)}

    pairs = []
    for ow in np.unique(own):
        g = np.sort(ids[own == ow])
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                pairs.append((int(g[i]), int(g[j])))

    # --- reachability: angle from the tangent to the partner ----------------
    # For each ordered direction of a true pair, take the endpoint of A closest
    # to B and ask how far off its outward tangent B lies. The pair is reached
    # at half-angle theta if either direction is within theta.
    best_angle, gap = _reach(pairs, span, epos, etan)
    ok = np.isfinite(best_angle) & np.isfinite(gap)

    # The chance baseline, through the same code path: random unit tangents
    # in place of the real ones, same pairs, same closest-endpoint choice, so
    # the best-of-two-directions structure of the statistic is in the null.
    rng_null = np.random.default_rng(SEED + 1)
    null_angles = []
    for _ in range(N_NULL):
        rt = rng_null.normal(size=etan.shape)
        rt /= np.maximum(np.linalg.norm(rt, axis=1, keepdims=True), 1e-12)
        na, _ = _reach(pairs, span, epos, rt.astype(etan.dtype))
        null_angles.append(na[ok])
    null_angles = np.stack(null_angles) if null_angles else np.empty((0, 0))

    # --- panel size: distractor endpoints inside the same cone --------------
    rng = np.random.default_rng(SEED)
    lab_ep = np.concatenate([np.arange(*span[int(i)]) for i in ids.tolist()
                             if span.get(int(i), (0, 0))[1] >
                             span.get(int(i), (0, 0))[0]])
    sample = rng.choice(lab_ep, size=min(PANEL_SAMPLE, len(lab_ep)),
                        replace=False)
    tree = cKDTree(epos)

    rows: dict[str, dict] = {}
    best = None
    for max_d, half_deg in CONES:
        reach = float((best_angle[ok] <= half_deg).mean()) if ok.any() else 0.0
        reach_d = float(((best_angle[ok] <= half_deg)
                         & (gap[ok] <= max_d)).mean()) if ok.any() else 0.0

        counts = []
        cos_t = np.cos(np.radians(half_deg))
        for e in sample.tolist():
            near = tree.query_ball_point(epos[e], r=max_d)
            if not near:
                counts.append(0)
                continue
            n = np.fromiter(near, np.int64, len(near))
            n = n[ea[n] != ea[e]]                       # other atoms only
            if not len(n):
                counts.append(0)
                continue
            v = epos[n] - epos[e]
            nv = np.linalg.norm(v, axis=1)
            keep = nv > 0
            c = (v[keep] @ etan[e]) / nv[keep]
            counts.append(int((c >= cos_t).sum()))
        cts = np.asarray(counts, float)

        # Chance is measured (random tangents, same loop). The single-direction
        # closed form (1 - cos theta)/2 is what the first run used; it is the
        # wrong null for a best-of-two statistic and is kept only as a record.
        single = (1.0 - np.cos(np.radians(half_deg))) / 2.0
        per_seed = ((null_angles <= half_deg).mean(axis=1)
                    if null_angles.size else np.array([np.nan]))
        chance = float(np.nanmean(per_seed))
        chance_sd = float(np.nanstd(per_seed))

        key = f"{int(max_d/1000)}um_{int(half_deg)}deg"
        row = {"max_distance_nm": max_d, "half_angle_deg": half_deg,
               "reach_angle_only": round(reach, 6),
               "reach_if_direction_were_random": round(chance, 6),
               "reach_if_random_sd_over_seeds": round(chance_sd, 6),
               "naive_single_direction_chance": round(float(single), 6),
               "closed_form_best_of_two_chance":
                   round(float(1.0 - (1.0 - single) ** 2), 6),
               "enrichment_over_chance": round(float(reach / chance), 2)
               if chance > 0 else None,
               "enrichment_as_first_reported": round(float(reach / single), 2)
               if single > 0 else None,
               "reach_within_distance": round(reach_d, 6),
               "median_panel": float(np.median(cts)),
               "p90_panel": float(np.percentile(cts, 90)),
               "mean_panel": round(float(cts.mean()), 2)}
        meets = (row["reach_within_distance"] >= REACH_BAR
                 and row["median_panel"] <= PANEL_BAR)
        row["meets_bar"] = bool(meets)
        rows[key] = row
        if meets and (best is None
                      or row["reach_within_distance"]
                      > rows[best]["reach_within_distance"]):
            best = key
        print(f"  cone {key:<14} reach {row['reach_within_distance']:6.1%}  "
              f"(angle-only {reach:5.1%} vs {chance:5.1%} chance, "
              f"{row['enrichment_over_chance']:4.1f}x)  "
              f"median panel {row['median_panel']:7.1f}  "
              f"{'MEETS' if meets else ''}", flush=True)

    ang = {f"p{q}": round(float(np.percentile(best_angle[ok], q)), 1)
           for q in (10, 25, 50, 75, 90)}
    # Chance percentiles from the empirical null (random tangents, same loop).
    # The first run used the single-direction closed form arccos(1 - 2q/100),
    # whose median is 90 degrees; the best-of-two median is ~65.5 degrees.
    ang_chance = {f"p{q}": round(float(np.percentile(null_angles.ravel(), q)), 1)
                  for q in (10, 25, 50, 75, 90)} if null_angles.size else {}
    ang_single = {f"p{q}": round(float(np.degrees(np.arccos(1 - 2 * q / 100))), 1)
                  for q in (10, 25, 50, 75, 90)}
    print(f"\n  angle to true partner : "
          + ", ".join(f"{k}={v}deg" for k, v in ang.items()), flush=True)
    print(f"  same, if random       : "
          + ", ".join(f"{k}={v}deg" for k, v in ang_chance.items()), flush=True)

    return Outcome(
        passed=best is not None,
        observed={
            "best_cone": best or "none",
            "best_reach": rows[best]["reach_within_distance"] if best else 0.0,
            "best_median_panel": rows[best]["median_panel"] if best else None,
            "median_angle_to_partner_deg": ang["p50"],
            "median_angle_if_random_deg": ang_chance["p50"],
            "proximity_reference_reach": PROXIMITY_REACHABLE,
        },
        population={
            "n_true_pairs": len(pairs),
            "n_pairs_measurable": int(ok.sum()),
            "n_labelled_atoms": int(L.sum()),
            "n_endpoints": int(good.sum()),
            "panel_sample": int(len(sample)),
        },
        tables={"cone_sweep": rows,
                "angle_to_true_partner_deg": {
                    k: {"observed_deg": ang[k],
                        "if_random_deg": ang_chance.get(k),
                        "if_random_single_direction_deg_as_first_reported":
                            ang_single[k]}
                    for k in ang},
                "null": {"method": "empirical: random unit tangents through "
                                   "the same reachability loop, same pairs",
                         "n_seeds": N_NULL, "n_pairs": int(ok.sum())}},
        note=(f"{best} reaches {rows[best]['reach_within_distance']:.1%} at "
              f"median panel {rows[best]['median_panel']:.0f}" if best else
              f"no cone reached {REACH_BAR:.0%} at a median panel of "
              f"{PANEL_BAR:.0f}; median angle to the true partner is "
              f"{ang['p50']:.0f} degrees"),
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
