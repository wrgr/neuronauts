"""EXP-058 — the floor and the ceiling, so later numbers mean something.

Every result in this repo's history was reported without the two numbers that
make it interpretable: what you get for doing nothing, and what you would get
if you were right. Without those, "ARI 0.75" and "merge precision 0.95" cannot
be placed. This experiment publishes the whole ladder on one substrate, through
one metric package, so every later experiment has rows to sit between.

The rungs, weakest to strongest:

``do_nothing``      every atom alone. The v117 segmentation untouched. Perfect
                    precision by construction and zero recall; it is the
                    honest floor and a surprising amount of the job is already
                    done here.
``random_matched``  random atom pairs from the same candidate surface, at the
                    same count as the proximity rung. Controls for "any
                    merging moves ARI".
``proximity_*``     union-find over candidate pairs whose endpoint gap is under
                    1, 2 or 5 um. The naive method every reader will ask about.
``oracle``          cluster the labelled atoms by their proofread owner. The
                    ceiling *given this candidate surface*, which is lower than
                    a true ceiling wherever the panel missed a pair.

The candidate surface is shared: one bounded k-nearest-neighbour panel over
every tier-10 endpoint, built once and cached, so the rungs differ only in
which pairs they accept. It is bounded (k=8 within 5 um), not all-pairs, and
that is a real limitation of the ceiling rather than a detail.

Evaluation is restricted to atoms with a proofread owner, because those are the
only ones whose merges can be judged; the unlabelled atoms remain in the
clustering as distractors, which is what makes the task honest.

    uv run python -m neuronauts.experiments.exp058_baseline_ladder
"""

from __future__ import annotations

import numpy as np

from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.harness.candidates import build_candidate_panel, load_panel
from neuronauts.harness.evaluation import union_find_components
from neuronauts.harness.labels import TIER_NONE, load_labels
from neuronauts.metrics import evaluate_partition_suite

TOPOLOGY = "data/substrate/topology/k10.npz"
LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"
PANEL_CACHE = "data/substrate/panels/k10_proximity.npz"

PANEL_RADIUS_NM = 5000.0
PANEL_K = 8
PROXIMITY_THRESHOLDS_NM = (1000.0, 2000.0, 5000.0)
SEED = 0

SPEC = Spec(
    id="EXP-058",
    title="Baseline ladder",
    question="What are the floor and the ceiling on this substrate?",
    criterion="every rung reports through neuronauts.metrics AND the ladder is "
              "correctly ordered on finite numbers: oracle ARI above the best "
              "proximity rung, proximity at or above random, do-nothing at "
              "exactly zero pair recall and oracle at exactly one. A ladder "
              "out of order means the evaluation is broken, not that a method "
              "is good; a non-finite rung is not a pass.",
    requires_ran=["EXP-057"],
    inputs=[TOPOLOGY, LABELS_NPZ],
    params={"panel_radius_nm": PANEL_RADIUS_NM, "panel_k": PANEL_K,
            "proximity_thresholds_nm": list(PROXIMITY_THRESHOLDS_NM),
            "seed": SEED},
    flags={"synthetic_fallback": False,
           "labels_used_only_for_evaluation": True},
)


def _panel(root, ep):
    """Build the shared candidate surface once, then reuse it."""
    cache = root / PANEL_CACHE
    if cache.exists():
        return load_panel(cache), True
    panel = build_candidate_panel(
        ep["atom"], ep["pos"], ep["tan"], ep["leaf"], ep["cal"],
        min_leaf_nm=0.0, min_caliber_nm=0.0,
        radius_nm=PANEL_RADIUS_NM, k=PANEL_K,
        meta={"purpose": "EXP-058 proximity ladder; deliberately unfiltered"})
    panel.save(cache)
    return panel, False


def _score(pred, true_owner, labelled) -> dict:
    """Metrics over the atoms whose merges can be judged."""
    m = evaluate_partition_suite(pred[labelled], true_owner[labelled],
                                 ignore=0, naive_baseline=False)
    # The suite names these pair_*, not merge_*. Asking for the wrong key
    # returned NaN for every rung on the first run, and the ordering check
    # accepted it -- hence the explicit key list below and the assertion.
    want = ("ari", "pair_precision", "pair_recall", "pair_f1", "pair_tp",
            "pair_fp", "pair_fn", "n_clusters_pred")
    missing = [k for k in want if k not in m]
    if missing:
        raise KeyError(f"metric suite did not return {missing}; "
                       f"it returned {sorted(m)}")
    _, inv = np.unique(pred[labelled], return_inverse=True)
    sizes = np.bincount(inv)
    return {
        "ari": float(m["ari"]),
        "pair_precision": float(m["pair_precision"]),
        "pair_recall": float(m["pair_recall"]),
        "pair_f1": float(m["pair_f1"]),
        "pair_tp": int(m["pair_tp"]), "pair_fp": int(m["pair_fp"]),
        "pair_fn": int(m["pair_fn"]),
        "n_clusters_pred": int(m["n_clusters_pred"]),
        "largest_cluster": int(sizes.max()) if len(sizes) else 0,
    }


def run(ctx: Context) -> Outcome:
    root = ctx.root
    with np.load(root / TOPOLOGY, allow_pickle=False) as z:
        atoms = z["atom_id"]
        ep = {"atom": z["ep_atom"], "pos": z["ep_pos_nm"],
              "tan": z["ep_tangent"], "leaf": z["ep_seg_len_nm"],
              "cal": z["ep_caliber_nm"]}

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

    panel, cached = _panel(root, ep)
    order = np.argsort(atoms)
    srt = atoms[order]
    ia = order[np.searchsorted(srt, panel.atom_a)]
    ib = order[np.searchsorted(srt, panel.atom_b)]
    gap = panel.col("gap_nm")
    n = len(atoms)

    rungs: dict[str, dict] = {}

    # 1. the floor
    rungs["do_nothing"] = _score(np.arange(n), owner, labelled)

    # 3. proximity, first, because the random rung matches its edge count
    n_prox_1um = int((gap <= PROXIMITY_THRESHOLDS_NM[0]).sum())
    for thr in PROXIMITY_THRESHOLDS_NM:
        m = gap <= thr
        pred = union_find_components(n, ia[m], ib[m])
        rungs[f"proximity_{int(thr)}nm"] = {
            **_score(pred, owner, labelled), "n_pairs_accepted": int(m.sum())}

    # 2. random at a matched count
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(gap), size=min(n_prox_1um, len(gap)), replace=False)
    pred = union_find_components(n, ia[pick], ib[pick])
    rungs["random_matched"] = {**_score(pred, owner, labelled),
                               "n_pairs_accepted": int(len(pick))}

    # 4. the ceiling, given this candidate surface
    pred = np.arange(n)
    pred[labelled] = n + owner[labelled]          # keep distractors singleton
    _, pred = np.unique(pred, return_inverse=True)
    rungs["oracle"] = _score(pred, owner, labelled)

    best_prox = max(rungs[f"proximity_{int(t)}nm"]["ari"]
                    for t in PROXIMITY_THRESHOLDS_NM)
    # Every term must be a real number: a NaN slipping through here is how the
    # first run "passed" with no merge metrics at all.
    terms = [rungs["oracle"]["ari"], best_prox,
             rungs["random_matched"]["ari"], rungs["do_nothing"]["pair_recall"]]
    all_real = all(np.isfinite(t) for t in terms)
    ordered = bool(
        all_real
        and rungs["oracle"]["ari"] > best_prox
        and best_prox >= rungs["random_matched"]["ari"]
        and rungs["do_nothing"]["pair_recall"] == 0.0
        and rungs["oracle"]["pair_recall"] == 1.0)

    n_true_pairs = int(sum(c * (c - 1) // 2 for c in
                           np.bincount(np.unique(owner[labelled],
                                                 return_inverse=True)[1])))

    return Outcome(
        passed=bool(ordered),
        observed={
            "oracle_ari": round(rungs["oracle"]["ari"], 6),
            "best_proximity_ari": round(best_prox, 6),
            "random_ari": round(rungs["random_matched"]["ari"], 6),
            "do_nothing_pair_recall": rungs["do_nothing"]["pair_recall"],
            "best_proximity_pair_precision":
                max(rungs[f"proximity_{int(t)}nm"]["pair_precision"]
                    for t in PROXIMITY_THRESHOLDS_NM),
        },
        population={
            "n_atoms": int(n),
            "n_labelled_atoms": int(labelled.sum()),
            "n_distinct_owners": int(len(np.unique(owner[labelled]))),
            "n_true_same_owner_pairs": n_true_pairs,
            "n_endpoints": int(len(ep["atom"])),
            "n_candidate_pairs": int(len(panel)),
            "panel_from_cache": bool(cached),
        },
        tables={"ladder": rungs},
        note=("ladder correctly ordered" if ordered else
              ("LADDER OUT OF ORDER -- the evaluation, not a method, is "
               "suspect" if all_real else
               "a rung reported a non-finite metric; the ladder cannot be "
               "ordered and must not be called passing")),
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
