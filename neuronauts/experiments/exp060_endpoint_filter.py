"""EXP-060 — which of the 5.1M endpoints are real split sites, not spines?

EXP-058 established that recall is free and precision is everything: proximity
union-find recovers all 492 true pairs at a pair precision of 0.0006, because
61% of candidate pairs sit within a micron of each other. The first place to
buy precision back is before scoring, by not proposing most of the panel at
all.

At L2 resolution a degree-1 node is usually a spine or a small protrusion, not
a cut. The contracted topology carries two label-free quantities that should
separate them — the length of the leaf segment the tip terminates, and the
caliber at the tip — and this experiment measures what they buy.

The bar, declared before the sweep: **at least 90% recall of true continuation
pairs, at a median panel of at most 20 partners per labelled atom, while
keeping at most 1% of endpoints.** All three at once. Recall alone is easy
(keep everything), and a small panel alone is easy (keep nothing).

Each setting rebuilds the panel rather than filtering the finished one, because
a proposer filters *then* searches: dropping a spine changes which endpoints
are in its neighbours' k-nearest sets. Post-filtering the unfiltered panel
would flatter every row.

    uv run python -m neuronauts.experiments.exp060_endpoint_filter
"""

from __future__ import annotations

import numpy as np

from scipy.spatial import cKDTree

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.candidates import build_candidate_panel
from neuronauts.harness.labels import TIER_NONE, load_labels

TOPOLOGY = "data/substrate/topology/k10.npz"
LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"

RADIUS_NM = 5000.0
K = 8

#: (min_leaf_nm, min_caliber_nm). The first row is the unfiltered ceiling.
SWEEP = [
    (0.0, 0.0),
    (1000.0, 0.0), (1000.0, 30.0), (1000.0, 50.0),
    (2000.0, 30.0), (2000.0, 50.0), (2000.0, 80.0),
    (5000.0, 30.0), (5000.0, 50.0), (5000.0, 80.0),
]

RECALL_BAR = 0.90
PANEL_BAR = 20.0
KEEP_BAR = 0.01

SPEC = Spec(
    id="EXP-060",
    title="Endpoint filter",
    question="Which of the 5.1M endpoints are real split sites rather than "
             "spines?",
    criterion=f"at least {RECALL_BAR:.0%} recall of true continuation pairs, "
              f"at a median panel of at most {PANEL_BAR:.0f} partners per "
              f"labelled atom, while keeping at most {KEEP_BAR:.0%} of "
              f"endpoints -- all three at once",
    requires_ran=["EXP-057"],
    inputs=[TOPOLOGY, LABELS_NPZ],
    params={"radius_nm": RADIUS_NM, "k": K, "sweep": SWEEP},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


def _true_pairs(atoms, owner, labelled) -> set[tuple[int, int]]:
    """Every same-owner pair of labelled atoms, as sorted atom-id tuples."""
    ids, own = atoms[labelled], owner[labelled]
    out: set[tuple[int, int]] = set()
    for o in np.unique(own):
        grp = np.sort(ids[own == o])
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                out.add((int(grp[i]), int(grp[j])))
    return out


def _true_pair_gaps(atoms, ep_atom, ep_pos, pairs) -> np.ndarray:
    """Minimum endpoint-to-endpoint distance between each pair of true partners.

    This is what decides whether a proximity search can reach a pair at all,
    independently of any filter, so it is measured here rather than inferred
    from a panel that already applied one.
    """
    o = np.argsort(ep_atom)
    ea, ep = ep_atom[o], ep_pos[o]
    need = np.unique(np.asarray([a for a, _ in pairs] + [b for _, b in pairs],
                                np.uint64))
    lo = np.searchsorted(ea, need)
    hi = np.searchsorted(ea, need, side="right")
    pts = {}
    for i, a, b in zip(need.tolist(), lo, hi):
        P = ep[a:b]
        pts[int(i)] = P[np.isfinite(P).all(axis=1)]

    out = np.full(len(pairs), np.inf)
    for i, (a, b) in enumerate(pairs):
        A, B = pts.get(int(a)), pts.get(int(b))
        if A is None or B is None or not len(A) or not len(B):
            continue
        out[i] = float(cKDTree(B).query(A, k=1)[0].min())
    return out


def run(ctx: Context) -> Outcome:
    root = ctx.root
    with np.load(root / TOPOLOGY, allow_pickle=False) as z:
        atoms = z["atom_id"]
        ep_atom, ep_pos = z["ep_atom"], z["ep_pos_nm"]
        ep_tan, ep_len, ep_cal = (z["ep_tangent"], z["ep_seg_len_nm"],
                                  z["ep_caliber_nm"])

    labels = load_labels(root / LABELS_NPZ)
    idx = labels.index_of(atoms)
    has = idx >= 0
    owner = np.zeros(len(atoms), np.int64)
    pure = np.zeros(len(atoms), bool)
    tier = np.full(len(atoms), TIER_NONE, np.int8)
    owner[has] = labels.owner[idx[has]].astype(np.int64)
    pure[has] = labels.pure[idx[has]]
    tier[has] = labels.owner_tier[idx[has]]
    labelled = pure & (tier > TIER_NONE) & (owner > 0)

    truth = _true_pairs(atoms, owner, labelled)
    labelled_ids = set(atoms[labelled].tolist())
    n_ep = len(ep_atom)

    # How far apart are true partners? This bounds what any proximity search
    # can propose, before a single filter is applied.
    gaps = _true_pair_gaps(atoms, ep_atom, ep_pos, sorted(truth))
    finite = np.isfinite(gaps)
    reach = {f"within_{int(t/1000)}um": {
                "n": int((gaps[finite] <= t).sum()),
                "frac": round(float((gaps[finite] <= t).mean()), 6)}
             for t in (1000., 2000., 5000., 10000., 20000., 50000., 100000.)}
    gap_pct = {f"p{q}": round(float(np.percentile(gaps[finite], q)), 1)
               for q in (10, 25, 50, 75, 90, 99)}
    reachable_at_radius = float((gaps[finite] <= RADIUS_NM).mean())
    print(f"  true-partner gap: median {gap_pct['p50']:,.0f} nm, "
          f"p90 {gap_pct['p90']:,.0f} nm; "
          f"{reachable_at_radius:.1%} within the {RADIUS_NM/1000:.0f} um "
          f"search radius\n", flush=True)

    rows: dict[str, dict] = {}
    best = None
    for min_leaf, min_cal in SWEEP:
        panel = build_candidate_panel(
            ep_atom, ep_pos, ep_tan, ep_len, ep_cal,
            min_leaf_nm=min_leaf, min_caliber_nm=min_cal,
            radius_nm=RADIUS_NM, k=K)

        kept = int(((ep_len >= min_leaf) & (ep_cal >= min_cal)
                    & np.isfinite(ep_pos).all(axis=1)
                    & np.isfinite(ep_tan).all(axis=1)).sum())

        a, b = panel.atom_a, panel.atom_b
        found = {(int(x), int(y)) for x, y in zip(a.tolist(), b.tolist())
                 if (int(x), int(y)) in truth}
        recall = len(found) / max(len(truth), 1)

        # Panel size counted per *labelled* atom: the decision unit downstream.
        deg: dict[int, int] = {}
        for x, y in zip(a.tolist(), b.tolist()):
            if int(x) in labelled_ids:
                deg[int(x)] = deg.get(int(x), 0) + 1
            if int(y) in labelled_ids:
                deg[int(y)] = deg.get(int(y), 0) + 1
        sizes = np.array([deg.get(i, 0) for i in labelled_ids], float)

        key = f"leaf>={int(min_leaf)}nm_cal>={int(min_cal)}nm"
        row = {
            "min_leaf_nm": min_leaf, "min_caliber_nm": min_cal,
            "endpoints_kept": kept,
            "frac_endpoints_kept": round(kept / max(n_ep, 1), 6),
            "n_pairs": int(len(panel)),
            "recall": round(recall, 6),
            "true_pairs_found": len(found),
            "median_panel": float(np.median(sizes)),
            "p90_panel": float(np.percentile(sizes, 90)),
        }
        meets = (recall >= RECALL_BAR
                 and row["median_panel"] <= PANEL_BAR
                 and row["frac_endpoints_kept"] <= KEEP_BAR)
        row["meets_bar"] = bool(meets)
        rows[key] = row
        if meets and (best is None or recall > rows[best]["recall"]):
            best = key
        print(f"  {key:<34} keep {row['frac_endpoints_kept']:7.3%}  "
              f"recall {recall:6.1%}  median panel {row['median_panel']:6.1f}  "
              f"{'MEETS' if meets else ''}", flush=True)

    ceiling = rows["leaf>=0nm_cal>=0nm"]
    return Outcome(
        passed=best is not None,
        observed={
            "best_setting": best or "none",
            "best_recall": rows[best]["recall"] if best else 0.0,
            "unfiltered_recall_ceiling": ceiling["recall"],
            "n_settings_meeting_bar": sum(1 for r in rows.values()
                                          if r["meets_bar"]),
            "true_pair_gap_median_nm": gap_pct["p50"],
            "frac_true_pairs_within_search_radius": round(reachable_at_radius, 6),
        },
        population={
            "n_endpoints": n_ep,
            "n_labelled_atoms": int(labelled.sum()),
            "n_true_pairs": len(truth),
        },
        tables={"filter_sweep": rows,
                "true_pair_gap_percentiles_nm": {k: {"nm": v}
                                                 for k, v in gap_pct.items()},
                "true_pairs_reachable_by_radius": reach},
        note=(f"{best} meets all three bars" if best else
              (f"no setting met all three bars, and none could: only "
               f"{reachable_at_radius:.1%} of true pairs have any endpoint "
               f"within the {RADIUS_NM/1000:.0f} um search radius, so the "
               f"{RECALL_BAR:.0%} recall bar was unreachable before any filter "
               f"was applied")),
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
