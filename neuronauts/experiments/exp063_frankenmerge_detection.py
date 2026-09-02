"""EXP-063 — does anything label-blind flag a frankenmerge, and are CB2's labels real?

A frankenmerge is one v117 object that spans two cells. The PCFG report found
that *detecting* one is comparatively easy -- a random forest on ten whole-object
shape descriptors of the synapse cloud reached AUC 0.875 and precision 0.41 at
the top 2%, on v117 roots at a 3.78% base rate -- and that *cutting* one is the
hard part. This experiment re-asks the detection question on the harness
substrate, with three things the report could not do:

1. **Polarity as a competitor (H8).** An atom whose synapses are half pre and
   half post is an axon merged to a dendrite. That signal is free (H1) and was
   never run against the shape detector.
2. **Object geometry as a competitor.** EXP-070 showed the L2 node cloud is a
   strictly tighter description of the object than its skeleton tips; extent,
   anisotropy and bimodality of that cloud are new here.
3. **CB2's tiers as held-out positives.** EXP-057B's expanded seam set carries a
   corroboration tier: tiers 3 and 2 are atoms our own v1822 tally already
   calls mixed; tiers 1 and 0 are atoms *only* ConnectomeBench2 says a human
   operated on. Tiers 1 and 0 are outside the positive definition used to
   train, so scoring them with the trained detector asks, without circularity:
   do they look like frankenmerges? That is the label-validity check EXP-057B's
   own caveat asked for.

**Size is the confound and is controlled by substrate.** On the full population
a mixed atom has a median 818 L2 nodes and a trustworthy negative 28, so "is it
big" is nearly a perfect classifier there and would flatter every feature set.
Tier >=10 puts a floor under both classes (945 vs 809) and is the only substrate
this experiment runs on; a size-only rung is reported anyway, and the bar
requires beating it.

**Two things this experiment deliberately does not do.**

*Bar 3 is not measured here.* The registry's first draft of this criterion
carried "Bar 3 above 0.5". Bar 3 is frankenmerge *split* recall -- the fraction
of true frankenmerges a method actually cuts -- and it needs a cut operator,
which this experiment does not have and EXP-062 owns. Reporting 0.000 for it
here, as every prior run did, would say nothing about detection. It moves to
EXP-062, and that move is recorded before this run, not after.

*Precision at the top 2% is reported, not gated.* With trustworthy negatives
(pure atoms with a proofread owner) the strict base rate is ~60% positive, which
makes precision@2% trivially high and uninformative. It is reported against the
lenient negative set (every pure atom) where the base rate is realistic, and the
gate is AUC, which is base-rate invariant.

CB2-touched atoms are excluded from both negative sets: if a tier-0 atom is a
real frankenmerge our tally cannot see, leaving it among the negatives would be
label noise of exactly the kind this run is trying to measure.

    python -m neuronauts.experiments.exp063_frankenmerge_detection
"""

from __future__ import annotations

import time

import numpy as np

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.atom_features import (
    GLOBAL_SHAPE_COLS, GLOBAL_SHAPE_SHAPE_COLS, OBJECT_GEOMETRY_COLS,
    POLARITY_COLS, TOPOLOGY_COLS, global_shape_features,
    object_geometry_features, polarity_features, precision_at_top,
    topology_features,
)
from neuronauts.harness.baselines import GradientBoostedStumps, LogisticRegression
from neuronauts.harness.cb2_positives import TIER_NAMES, load_cb2_positives
from neuronauts.harness.labels import TIER_NONE, load_labels
from neuronauts.harness.objgeom import load_object_geometry
from neuronauts.harness.population import load_population
from neuronauts.harness.spatial_split import (
    SPLIT_TRAIN, SPLIT_VAL, assign_split, describe,
)
from neuronauts.metrics.ranking import average_precision, roc_auc

TOPOLOGY = "data/substrate/topology/k10.npz"
OBJGEOM = "data/substrate/geom/objgeom_k10.npz"
POPULATION = "data/substrate/c100um/population.npz"
LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"
CB2_NPZ = "data/substrate/c100um/cb2_seam_positives.npz"

SPLIT_AXIS = 0
SPLIT_BUFFER_NM = 10_000.0     # 2x the 5 um candidate radius; see
                               # docs/threads/seam_positive_sample_size.md
TOP_FRAC = 0.02
SEED = 0

AUC_BAR = 0.875                # the PCFG report's shape detector, on its substrate
SIZE_MARGIN = 0.02             # must beat log(n) alone by this much

FEATURE_SETS = {
    "size_only":          ["log_n_syn"],
    "polarity":           POLARITY_COLS,
    "global_shape":       GLOBAL_SHAPE_COLS,
    "global_shape_noSize": [GLOBAL_SHAPE_COLS[i] for i in GLOBAL_SHAPE_SHAPE_COLS],
    "topology":           TOPOLOGY_COLS,
    "object_geometry":    OBJECT_GEOMETRY_COLS,
    "polarity+shape":     POLARITY_COLS + GLOBAL_SHAPE_COLS,
    "all":                (POLARITY_COLS + GLOBAL_SHAPE_COLS + TOPOLOGY_COLS
                           + OBJECT_GEOMETRY_COLS),
}

SPEC = Spec(
    id="EXP-063",
    title="Frankenmerge detection",
    question="Does mixed polarity, object shape, or their combination flag a "
             "false merge on the harness substrate -- and do CB2's "
             "uncorroborated tiers look like frankenmerges to a detector "
             "trained without them?",
    criterion=f"tier>=10 only (size-controlled); positives = atoms our own "
              f"v1822 tally calls mixed-lineage; strict negatives = pure atoms "
              f"with a proofread owner, CB2-touched atoms excluded from every "
              f"negative set; held out by a positives-centred spatial split "
              f"with a {SPLIT_BUFFER_NM/1000:.0f} um buffer. PASS when the best "
              f"non-size feature set reaches held-out AUC >= {AUC_BAR} AND "
              f"exceeds the size-only rung by >= {SIZE_MARGIN}. Precision at "
              f"the top {TOP_FRAC:.0%} is reported against lenient (all-pure) "
              f"negatives, not gated. Bar 3 (split recall) is a cut metric and "
              f"moves to EXP-062. CB2 tiers 1 and 0 are scored by the trained "
              f"detector as a label-validity diagnostic, no bar.",
    requires=["EXP-057B"], requires_ran=["EXP-057", "EXP-070"],
    inputs=[TOPOLOGY, OBJGEOM, POPULATION, LABELS_NPZ, CB2_NPZ],
    params={"split_axis": SPLIT_AXIS, "split_buffer_nm": SPLIT_BUFFER_NM,
            "top_frac": TOP_FRAC, "seed": SEED, "auc_bar": AUC_BAR,
            "size_margin": SIZE_MARGIN,
            "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()}},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": False,
           "labels_used_for_training": "own v1822 mixed-lineage tally, "
                                       "train side of the spatial split only"},
)


# ---------------------------------------------------------------------------
# substrate assembly
# ---------------------------------------------------------------------------

def _synapse_clouds(pop, atoms: np.ndarray):
    """CSR of synapse centres per atom: a synapse is in an atom's cloud when
    the atom is on either side of it, once."""
    order = np.argsort(atoms, kind="stable")
    srt = atoms[order]

    def hit(side):
        j = np.clip(np.searchsorted(srt, side), 0, len(srt) - 1)
        ok = srt[j] == side
        return order[j[ok]], np.flatnonzero(ok)

    ra, sa = hit(pop.syn_atom_pre)
    rb, sb = hit(pop.syn_atom_post)
    atom_row = np.concatenate([ra, rb])
    syn_row = np.concatenate([sa, sb])
    key = atom_row.astype(np.int64) * len(pop.syn_id) + syn_row
    _, u = np.unique(key, return_index=True)
    atom_row, syn_row = atom_row[u], syn_row[u]
    o = np.argsort(atom_row, kind="stable")
    atom_row, syn_row = atom_row[o], syn_row[o]
    ptr = np.searchsorted(atom_row, np.arange(len(atoms) + 1))
    return ptr, pop.syn_ctr_nm[syn_row].astype(np.float64)


def _features(atoms, topo, geo, pop) -> tuple[np.ndarray, list[str]]:
    """``[A, F]`` and the column names, families concatenated in a fixed order."""
    t0 = time.time()
    pol = polarity_features(topo["n_pre"], topo["n_post"])
    top = topology_features(topo)

    ptr, pts = _synapse_clouds(pop, atoms)
    shape = np.stack([global_shape_features(pts[ptr[i]:ptr[i + 1]], seed=SEED)
                      for i in range(len(atoms))])
    print(f"  synapse-cloud shape features: {len(atoms):,} atoms "
          f"({time.time()-t0:.0f}s)", flush=True)

    obj = np.stack([object_geometry_features(geo.points(int(a)), geo.radii(int(a)),
                                             seed=SEED)
                    for a in atoms.tolist()])
    print(f"  object-geometry features: ({time.time()-t0:.0f}s)", flush=True)

    X = np.concatenate([pol, shape, top, obj], axis=1)
    cols = POLARITY_COLS + GLOBAL_SHAPE_COLS + TOPOLOGY_COLS + OBJECT_GEOMETRY_COLS
    assert X.shape[1] == len(cols), (X.shape, len(cols))
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), cols


def _eval(y, s) -> dict:
    y = np.asarray(y, bool)
    out = {"n": int(len(y)), "n_pos": int(y.sum()),
           "base_rate": round(float(y.mean()), 6) if len(y) else float("nan")}
    if y.any() and (~y).any():
        out["auc"] = round(float(roc_auc(y, s)), 6)
        out["ap"] = round(float(average_precision(y, s)), 6)
        p = precision_at_top(y, s, TOP_FRAC)
        out[f"precision_at_top{int(TOP_FRAC*100)}pct"] = round(p["precision"], 6)
        out[f"recall_at_top{int(TOP_FRAC*100)}pct"] = round(p["recall"], 6)
        out["n_flagged"] = p["n_flagged"]
    else:
        out["auc"] = out["ap"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def run(ctx: Context) -> Outcome:
    root = ctx.root
    with np.load(root / TOPOLOGY, allow_pickle=False) as z:
        atoms = z["atom_id"]
        topo = {k: z[k] for k in z.files if k != "meta"}
    pop = load_population(root / POPULATION)
    labels = load_labels(root / LABELS_NPZ)
    cb2 = load_cb2_positives(root / CB2_NPZ)
    geo = load_object_geometry(root / OBJGEOM)

    # --- per-atom label masks ------------------------------------------------
    li = labels.index_of(atoms)
    has = li >= 0
    own_mixed = np.zeros(len(atoms), bool)
    mixed_pr = np.zeros(len(atoms), bool)
    pure = np.zeros(len(atoms), bool)
    tier_of_owner = np.full(len(atoms), TIER_NONE, np.int8)
    own_mixed[has] = labels.mixed[li[has]]
    mixed_pr[has] = labels.mixed_proofread[li[has]]
    pure[has] = labels.pure[li[has]]
    tier_of_owner[has] = labels.owner_tier[li[has]]

    ci = cb2.index_of(atoms)
    cb2_touched = ci >= 0
    cb2_tier = np.full(len(atoms), -1, np.int8)
    cb2_tier[cb2_touched] = cb2.tier[ci[cb2_touched]]
    cb2_sb = np.zeros(len(atoms), bool)
    cb2_sb[cb2_touched] = cb2.split_before[ci[cb2_touched]]

    neg_strict = pure & (tier_of_owner > TIER_NONE) & ~cb2_touched
    neg_lenient = pure & ~cb2_touched
    n_neg_strict_excluded = int((pure & (tier_of_owner > TIER_NONE) & cb2_touched).sum())

    # --- centroids and the split --------------------------------------------
    pi = np.clip(np.searchsorted(np.sort(pop.atom_id), atoms), 0, len(pop.atom_id) - 1)
    porder = np.argsort(pop.atom_id)
    prow = porder[pi]
    assert (pop.atom_id[prow] == atoms).all(), "k10 atom missing from population"
    centroid = pop.centroid_nm[prow]
    centre = float(np.median(centroid[own_mixed, SPLIT_AXIS]))
    split = assign_split(centroid, axis=SPLIT_AXIS, centre_nm=centre,
                         buffer_nm=SPLIT_BUFFER_NM)
    tr, va = split == SPLIT_TRAIN, split == SPLIT_VAL

    counts = {
        "n_atoms": int(len(atoms)),
        "own_mixed": int(own_mixed.sum()),
        "mixed_proofread": int(mixed_pr.sum()),
        "neg_strict": int(neg_strict.sum()),
        "neg_lenient": int(neg_lenient.sum()),
        "neg_strict_excluded_as_cb2_touched": n_neg_strict_excluded,
        "cb2_touched": int(cb2_touched.sum()),
        "cb2_by_tier": {TIER_NAMES[t]: int((cb2_tier == t).sum())
                        for t in sorted(TIER_NAMES)},
        "cb2_tier_in_own_mixed": {TIER_NAMES[t]: int(((cb2_tier == t) & own_mixed).sum())
                                  for t in sorted(TIER_NAMES)},
        "split": {"centre_nm": centre, **describe(split)},
        "train": {"pos": int((own_mixed & tr).sum()),
                  "neg_strict": int((neg_strict & tr).sum())},
        "val": {"pos": int((own_mixed & va).sum()),
                "neg_strict": int((neg_strict & va).sum()),
                "neg_lenient": int((neg_lenient & va).sum())},
    }
    print(f"  positives (own mixed) {counts['own_mixed']:,}: train "
          f"{counts['train']['pos']:,} / val {counts['val']['pos']:,}", flush=True)
    print(f"  strict negatives {counts['neg_strict']:,} (excluded "
          f"{n_neg_strict_excluded} CB2-touched): train "
          f"{counts['train']['neg_strict']:,} / val {counts['val']['neg_strict']:,}",
          flush=True)

    # --- features --------------------------------------------------------------
    X, cols = _features(atoms, topo, geo, pop)
    col_idx = {c: i for i, c in enumerate(cols)}

    train_mask = (own_mixed | neg_strict) & tr
    y_train = own_mixed[train_mask]

    # --- one model per feature set -------------------------------------------
    rows: dict[str, dict] = {}
    scores: dict[str, np.ndarray] = {}
    for name, fcols in FEATURE_SETS.items():
        j = [col_idx[c] for c in fcols]
        Xtr = X[train_mask][:, j]
        models = {
            "gbdt": GradientBoostedStumps.fit(Xtr, y_train, seed=SEED),
            "logistic": LogisticRegression.fit(Xtr, y_train),
        }
        rows[name] = {}
        for mname, m in models.items():
            s = m.decision(X[:, j])
            scores[f"{name}/{mname}"] = s
            vs = (own_mixed | neg_strict) & va
            vl = (own_mixed | neg_lenient) & va
            rows[name][mname] = {
                "strict": _eval(own_mixed[vs], s[vs]),
                "lenient": _eval(own_mixed[vl], s[vl]),
                "train_strict": _eval(own_mixed[train_mask], s[train_mask]),
            }
            r = rows[name][mname]
            print(f"  {name:<20} {mname:<9} val AUC strict {r['strict']['auc']:.3f}  "
                  f"lenient {r['lenient']['auc']:.3f}  "
                  f"P@{int(TOP_FRAC*100)}% lenient "
                  f"{r['lenient'].get('precision_at_top2pct', float('nan')):.3f}  "
                  f"(train AUC {r['train_strict']['auc']:.3f})", flush=True)

    # --- the bar ---------------------------------------------------------------
    def auc_of(name, mname="gbdt"):
        return rows[name][mname]["strict"]["auc"]

    size_auc = max(auc_of("size_only", m) for m in ("gbdt", "logistic"))
    best_name, best_model, best_auc = None, None, -1.0
    for name in FEATURE_SETS:
        if name == "size_only":
            continue
        for m in ("gbdt", "logistic"):
            a = auc_of(name, m)
            if np.isfinite(a) and a > best_auc:
                best_name, best_model, best_auc = name, m, a
    shape_auc = max(auc_of("global_shape", m) for m in ("gbdt", "logistic"))
    passed = bool(best_auc >= AUC_BAR and best_auc - size_auc >= SIZE_MARGIN)

    # --- CB2 tiers, scored by the best detector ------------------------------
    s_best = scores[f"{best_name}/{best_model}"]
    tiers: dict[str, dict] = {}
    neg_va = neg_strict & va
    for t in sorted(TIER_NAMES):
        m = (cb2_tier == t) & va
        # tiers 3/2 overlap own_mixed and may have trained; say so per row
        tiers[TIER_NAMES[t]] = {
            "n_val": int(m.sum()),
            "n_val_in_own_mixed": int((m & own_mixed).sum()),
            "n_val_split_before": int((m & cb2_sb).sum()),
            "mean_score": round(float(s_best[m].mean()), 4) if m.any() else float("nan"),
            "auc_vs_strict_neg": (round(float(roc_auc(
                np.r_[np.ones(int(m.sum()), bool), np.zeros(int(neg_va.sum()), bool)],
                np.r_[s_best[m], s_best[neg_va]])), 6)
                if m.any() and neg_va.any() else float("nan")),
            "frac_above_median_positive_score": (round(float(
                (s_best[m] > np.median(s_best[own_mixed & va])).mean()), 4)
                if m.any() and (own_mixed & va).any() else float("nan")),
        }
    tiers["_reference"] = {
        "own_mixed_val_mean_score": round(float(s_best[own_mixed & va].mean()), 4),
        "strict_neg_val_mean_score": round(float(s_best[neg_va].mean()), 4),
        "mixed_proofread_val": {
            "n": int((mixed_pr & va).sum()),
            "auc_vs_strict_neg": (round(float(roc_auc(
                np.r_[np.ones(int((mixed_pr & va).sum()), bool),
                      np.zeros(int(neg_va.sum()), bool)],
                np.r_[s_best[mixed_pr & va], s_best[neg_va]])), 6)
                if (mixed_pr & va).any() else float("nan"))},
    }
    for t in sorted(TIER_NAMES, reverse=True):
        r = tiers[TIER_NAMES[t]]
        print(f"  CB2 {TIER_NAMES[t]:<22} n_val {r['n_val']:>4}  "
              f"AUC vs strict-neg {r['auc_vs_strict_neg']:.3f}  "
              f"mean score {r['mean_score']:+.3f}", flush=True)

    t1, t0_ = tiers["new_mixed_raw_only"], tiers["new_no_v1822_signal"]
    note = (
        f"best detector {best_name}/{best_model}: held-out AUC {best_auc:.3f} "
        f"(bar {AUC_BAR}), size-only {size_auc:.3f}, global-shape rung "
        f"{shape_auc:.3f}. CB2 tiers held out of training: tier 1 "
        f"(raw-mixed only, n={t1['n_val']}) scores AUC {t1['auc_vs_strict_neg']:.3f} "
        f"against strict negatives, tier 0 (no v1822 signal, n={t0_['n_val']}) "
        f"scores {t0_['auc_vs_strict_neg']:.3f} -- "
        + ("tier 0 looks like the positives, so CB2's uncorroborated labels "
           "carry signal our tally cannot see"
           if np.isfinite(t0_["auc_vs_strict_neg"]) and t0_["auc_vs_strict_neg"] >= 0.75
           else "tier 0 does not look like the positives to this detector; "
                "treat it as unverified, not as noise")
        + ". Bar 3 not measured here; it is a cut metric and moves to EXP-062."
    )

    return Outcome(
        passed=passed,
        observed={
            "best_feature_set": f"{best_name}/{best_model}",
            "best_val_auc_strict": round(best_auc, 6),
            "size_only_val_auc_strict": round(size_auc, 6),
            "global_shape_val_auc_strict": round(shape_auc, 6),
            "polarity_val_auc_strict": max(auc_of("polarity", m)
                                           for m in ("gbdt", "logistic")),
            "best_lenient_precision_at_top2pct":
                rows[best_name][best_model]["lenient"].get(
                    "precision_at_top2pct", float("nan")),
            "cb2_tier1_auc_vs_strict_neg": t1["auc_vs_strict_neg"],
            "cb2_tier0_auc_vs_strict_neg": t0_["auc_vs_strict_neg"],
            "n_train_positives": counts["train"]["pos"],
        },
        population=counts,
        tables={"by_feature_set": rows, "cb2_tiers": tiers,
                "feature_columns": cols},
        note=note,
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
