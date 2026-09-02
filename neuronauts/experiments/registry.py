"""The experiment program as data: what to run, in what order, against what bar.

`docs/consolidation_plan.md` §6 sets out a program in strict dependency order —
substrate → baselines → propose → cut → score → assemble → re-derive. Written
as prose it needs a human to decide what may run next. Written here it does
not: every experiment declares its prerequisites, its inputs, and the bar it
must clear, so :func:`next_runnable` can answer "what is ready right now?"
without judgement, and a loop over it explores the program autonomously.

Three states, and the distinction matters:

``ready``            prerequisites pass, inputs exist, entry point exists.
``blocked``          something upstream has no passing result, or a declared
                     input is not on disk. The reason is reported, never
                     guessed at.
``not_implemented``  the plan declares it but no module runs it yet. This is
                     deliberately visible: a program you cannot run is a wish
                     list, and counting the gap is how it stops being one.

Nothing here judges a result. The bar in each ``criterion`` was written before
the data existed; the runner records the author's verdict against it.

    from neuronauts.experiments.registry import next_runnable, status_table
    print(status_table())
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from neuronauts.experiments._runner import (
    STATUS_PASSED, Spec, check_inputs, check_prerequisites, load_result,
)
from neuronauts.report.provenance import repo_root

# Paths the substrate experiments depend on. Declared once so a rename shows up
# as "blocked: missing input" rather than as a crash three hours in.
TOPOLOGY_K10 = "data/substrate/topology/k10.npz"
POPULATION = "data/substrate/c100um/population.npz"
LABELS = "results/atom_labels_v1822.json"
LABELS_NPZ = "data/substrate/c100um/labels_v1822.npz"
TOPOLOGY_KALL = "data/substrate/topology/kall.npz"
#: The object point cloud -- every L2 node of every atom, not just the skeleton
#: tips the endpoint table holds. Built by scripts/build_object_geometry.py.
OBJGEOM_K10 = "data/substrate/geom/objgeom_k10.npz"
OBJGEOM_KALL = "data/substrate/geom/objgeom_kall.npz"
CB2_POSITIVES = "data/substrate/c100um/cb2_seam_positives.npz"
# ConnectomeBench2 intake (EXP-057B). The CAVE lineage resolution is a recorded
# input, not a step: it was run once and its output is hashed like any file.
CB2_RAW_ROWS = "data/external/cb2/full_mouse_rows_raw.parquet"
CB2_INCUBE_EDITS = "data/external/cb2/incube_edits.json"
CB2_RESOLUTION = "data/external/cb2/final_resolution.json"


@dataclass
class Entry:
    """One registered experiment: its spec, where it lives, and its series."""

    spec: Spec
    series: str
    module: Optional[str] = None      # dotted path implementing run()
    est_minutes: Optional[int] = None
    note: str = ""

    @property
    def id(self) -> str:
        return self.spec.id

    def implemented(self) -> bool:
        if not self.module:
            return False
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            return False


def _s(**kw) -> Spec:
    return Spec(**kw)


# ---------------------------------------------------------------------------
# The program. Series letters follow docs/consolidation_plan.md §6.3.
# ---------------------------------------------------------------------------

REGISTRY: list[Entry] = [
    # --- A. substrate and baselines -----------------------------------------
    Entry(series="A", est_minutes=60, spec=_s(
        id="EXP-057", title="GT overlay and spatial split",
        question="What fraction of atoms and synapse mass carry unambiguous "
                 "ground truth, and where?",
        criterion="at least 30% of synapse mass on single-lineage atoms with a "
                  "proofread owner; else widen the tier before proceeding",
        requires=[], inputs=[POPULATION, LABELS_NPZ],
        flags={"synthetic_fallback": False,
               "labels_used_only_for_evaluation": True}),
        module="neuronauts.experiments.exp057_gt_overlay",
        note="Substrate for everything downstream. Labels exist "
             "(results/atom_labels_v1822.json): 2,357 pure-gold atoms, 2,444 "
             "mixed-lineage, 56 mixed AND proofread-owned."),

    Entry(series="A", est_minutes=2, spec=_s(
        id="EXP-057B", title="ConnectomeBench2 intake",
        question="Can an external corpus lift us past 56 trustworthy "
                 "seam positives?",
        criterion="at least 1,000 merge or split decisions map onto v117 roots "
                  "in or near the harness cube",
        requires=[],
        inputs=[CB2_RAW_ROWS, CB2_INCUBE_EDITS, CB2_RESOLUTION, POPULATION,
               LABELS_NPZ],
        flags={"network": False}),
        module="neuronauts.experiments.exp057b_cb2_intake",
        note="716,485 expert proofreading decisions (FlyWire/MICrONS/Fish1/H01). "
             "Its third task, mask segmentation for merge-error correction, is "
             "our seam-location problem renamed. MEASURED, not projected: of "
             "2,514 in-cube merge/split decisions, all 7,220 referenced root "
             "ids resolved to nonzero v117 roots and 2,392 decisions (95.1%) "
             "put at least one of them inside the 279,075-atom population. The "
             "strictest cut -- split_edit before-roots, the closest analogue of "
             "the 56 -- is 1,508 decisions over 1,116 distinct new atoms, 574 "
             "of which our own v1822 crosswalk independently calls mixed "
             "lineage; 37 of the existing 56 are re-found by this corpus. "
             "est_minutes dropped 30 -> 2 because the CAVE lineage resolution "
             "(31.6 min, 7,220/7,220) was run once and is now a recorded, "
             "hashed input, so flags={'network': False} is true of the run. "
             "Emits data/substrate/c100um/cb2_seam_positives.npz (2,067 atoms, "
             "two strictness axes) for EXP-062/063. Open caveat carried in the "
             "artifact meta: resolution went through one arbitrary supervoxel "
             "per root, not the decision's edit_point_nm, so these are not yet "
             "LOCATED seam positives."),

    Entry(series="A", est_minutes=30, spec=_s(
        id="EXP-057C", title="SegCLR embedding intake",
        question="Do the published SegCLR per-segment embeddings separate "
                 "same-cell from different-cell fragments on our substrate?",
        criterion="same-cell vs different-cell fragment cosine separation at or "
                  "above our own tree-DNA (within-type AUC 0.829), on atoms "
                  "the v117->v343 crosswalk resolves",
        requires_ran=["EXP-057"], inputs=[LABELS_NPZ],
        flags={"network": True}),
        note="Reclassified to requires_ran under the same rule as EXP-058/060: "
             "it needs EXP-057's overlay to pair same-cell against "
             "different-cell fragments, not the synapse-mass density EXP-057 "
             "failed on. Promoted to the critical path by EXP-061 -- retrieval "
             "over an embedding has neither radius nor cone, which is the one "
             "thing geometry cannot do. "
             "RETARGETED from GraphDINO to SegCLR after checking the papers. "
             "GraphDINO (Weis et al. 2025) is disqualified, verified verbatim "
             "in its Methods: it removed 6,304 cells predicted to be "
             "fragmented, and 'we focused on the dendritic skeleton only and "
             "removed segments labeled as axon'. Its coordinates are "
             "soma-relative and skeletons are subsampled to 200 nodes assuming "
             "a whole tree. Our atoms are fragments, often axonal, with no "
             "soma -- the representation cannot apply. SegCLR is the opposite "
             "profile: per-segment and fragment-native by construction, axon "
             "and dendrite equally, and originally trained on segmentation "
             "v117, our exact version. Public embeddings verified to exist at "
             "gs://iarpa_microns/minnie/minnie65/embeddings_m343/"
             "segclr_nm_coord_public_offset_csvzips/ (~220MB shards) with a "
             "135MB TF2 encoder checkpoint at .../embeddings/models/"
             "segclr-216000/. Catch: the public release is keyed to v343/v943, "
             "so it needs a supervoxel crosswalk -- the same mechanism the "
             "v1822 GT overlay already uses. See "
             "docs/threads/embedding_availability.md."),

    Entry(series="A", est_minutes=90, spec=_s(
        id="EXP-058", title="Baseline ladder",
        question="What are the floor and the ceiling on this substrate?",
        criterion="every rung reports through neuronauts.metrics AND the "
                  "ladder is correctly ordered: oracle ARI above the best "
                  "proximity rung, proximity above random, do-nothing at zero "
                  "merge recall",
        requires_ran=["EXP-057"], inputs=[TOPOLOGY_K10, LABELS_NPZ],
        flags={"synthetic_fallback": False}),
        module="neuronauts.experiments.exp058_baseline_ladder",
        note="Rungs: untouched v117, random at matched count, proximity "
             "union-find at 1/2/5 um, GT-lineage oracle. Trained checkpoints "
             "are deliberately omitted -- the treestitch models expect a "
             "different input contract, and wiring them badly would be worse "
             "than saying they are absent. Comparison methods (published "
             "approaches) are a separate axis and are paused; see plan "
             "section 7.2. Scoped to EXP-057's labelled subset (16.2% of "
             "synapse mass), which every rung shares, so the comparison "
             "between rungs is fair."),

    Entry(series="A", est_minutes=15, spec=_s(
        id="EXP-059", title="Metric agreement",
        question="Do the legacy metric implementations agree with metrics/?",
        criterion="agreement to 1e-6, or every difference documented as a "
                  "deliberate convention change",
        requires_ran=["EXP-058"], inputs=[]),
        module="neuronauts.experiments.exp059_metric_agreement",
        note="Five of six callers already delegate; experiments/pcfg/"
             "conn_metric.py is the outstanding one."),

    # --- B. candidate generation --------------------------------------------
    Entry(series="B", est_minutes=45, spec=_s(
        id="EXP-060", title="Endpoint filter",
        question="Which of the 5.1M endpoints are real split sites rather "
                 "than spines?",
        criterion="at least 90% recall of true continuation pairs at a median "
                  "panel size of at most 20; and at most 1% of endpoints kept",
        requires_ran=["EXP-057"], inputs=[TOPOLOGY_K10, LABELS_NPZ],
        flags={"synthetic_fallback": False}),
        module="neuronauts.experiments.exp060_endpoint_filter",
        note="Redo of EXP-053B on a substrate that has coverage. Bar adopted "
             "verbatim from the PCFG report's E4. Needs EXP-057's overlay, not "
             "its density bar: continuation pairs come from the 4,802 "
             "proofread-owned single-lineage atoms, which EXP-057 confirmed "
             "exist even as it failed."),

    Entry(series="B", est_minutes=45, spec=_s(
        id="EXP-060B", title="Object-space atom-pair panel",
        question="Does atom-pair reduction recover the spanning links "
                 "endpoint k-NN missed, and does it hold at tier >=1?",
        criterion="report recall-vs-panel-size as a curve (object units), not "
                  "a single number; passes on internal consistency, since the "
                  "prior single-cap bar was itself an unverified extrapolation",
        requires_ran=["EXP-060"],
        inputs=["data/substrate/topology/k10.npz",
               "data/substrate/c100um/labels_v1822.npz"],
        flags={"synthetic_fallback": False,
              "labels_used_only_for_evaluation": True}),
        module="neuronauts.experiments.exp060b_object_panel",
        note="Direct test of CORRECTION.md's two fixes: reduce by atom not "
             "endpoint, and report/cap panel size in atom units. Compares "
             "tier>=10 (sparse, 20,826 atoms) against the true complete "
             "population (279,075 atoms, every atom, unioned from all fetch "
             "tiers). A first run mislabeled k1.npz -- the incremental "
             "1-4-synapse-only shard, not the union -- as this; fixed. "
             "Moved after EXP-060 in program order: it was listed before its "
             "own prerequisite, which "
             "test_prerequisites_come_earlier_in_program_order caught."),

    Entry(series="B", est_minutes=45, spec=_s(
        id="EXP-061", title="Directed cone vs proximity ball",
        question="Does the axon proximity failure hold for dendrites too?",
        criterion="a directed cone reaches at least 70% of true pairs at a "
                  "median panel of at most 20 -- i.e. beats EXP-060's measured "
                  "proximity ceiling of 47.4% reachable / 17.5% proposed, at "
                  "comparable panel size",
        requires_ran=["EXP-060"], inputs=[TOPOLOGY_K10, LABELS_NPZ]),
        module="neuronauts.experiments.exp061_directed_cone",
        note="Rewritten after EXP-060: the cone is not an improvement on the "
             "proximity ball, it is the replacement. EXP-060 measured the "
             "median true partner at 6.5 um and p90 at 56 um, so a 5 um ball "
             "reaches 47.4% at best while a 50 um ball would hold ~2.7M "
             "endpoints. A cone projects along the neurite instead, reaching "
             "far partners without the volume. Stratify by the polarity "
             "compartment signal (H1), which is free."),

    Entry(series="B", est_minutes=5, spec=_s(
        id="EXP-070", title="Object vs endpoint distance",
        question="Is the endpoint representation, rather than proximity "
                 "itself, why candidate generation failed?",
        criterion="the comparison must be sound before it can inform: "
                  "EXP-060's endpoint control reproduces to within 1 nm and "
                  "1e-5, and the object gap does not exceed the endpoint gap "
                  "on ANY measured pair. Passes on those two, then reports "
                  "reachability and MST agreement under both metrics on both "
                  "substrates -- a diagnostic, not a threshold",
        requires_ran=["EXP-060", "EXP-060B"],
        inputs=[LABELS_NPZ, TOPOLOGY_K10, OBJGEOM_K10,
               TOPOLOGY_KALL, OBJGEOM_KALL],
        flags={"synthetic_fallback": False,
              "labels_used_only_for_evaluation": True}),
        module="neuronauts.experiments.exp070_object_distance",
        note="EXP-060/060B/061 all measure distance between ENDPOINTS -- the "
             "degree-1 nodes of the contracted L2 skeleton -- which is a "
             "skeleton-space distance, not an object one. The raw fetch always "
             "held every L2 node of every atom; nothing consumed it until "
             "scripts/build_object_geometry.py. Answer: the metric was wrong "
             "but is not the reason proximity failed. Object distance lifts "
             "MST-spanning-link reachability at 5 um from 64.9% to 75.7% "
             "(tier10) and 47.7% to 56.8% (all), and recovers 280 labelled "
             "atoms that have NO endpoint row at all, but does not approach "
             "the 90% bar. It also moves the answer key: 463 of 3,538 spanning "
             "links on the full population differ between the two metrics, so "
             "EXP-060B's recall carries an error bar nobody drew. Adopt object "
             "distance downstream because it is the correct and strictly "
             "tighter quantity, not because it rescues the ball."),

    # --- C. cuts and frankenmerge detection ---------------------------------
    Entry(series="C", est_minutes=90, spec=_s(
        id="EXP-062", title="Real-L2 cuts and seam location",
        question="Do cuts on real L2 adjacency beat MST-geometry cuts, and can "
                 "we pick the right edge?",
        criterion="at least 90% same-lineage pair recall AND at least 50% "
                  "cross-lineage split recall; seam top-1 above 25%",
        requires=["EXP-057B"], requires_ran=["EXP-057"],
        inputs=[TOPOLOGY_K10, LABELS_NPZ, CB2_POSITIVES, OBJGEOM_K10],
        flags={"synthetic_fallback": False}),
        note="Merges the plan's 062 with the PCFG report's E2/E3. EXP-056 "
             "falsified every single global edge-length threshold. "
             "DEPENDENCY SWAPPED, deliberately: was requires=[EXP-057] + "
             "requires_ran=[EXP-057B], now the reverse. EXP-057's failing bar "
             "was about the DENSITY of proofread synapse mass (16.2% vs 30%); "
             "what this experiment actually needs is a COUNT of trustworthy "
             "seam positives, which EXP-057B now supplies from an external "
             "corpus. So EXP-057B is the one that must have passed, and "
             "EXP-057 is the one whose overlay must merely exist -- the same "
             "rule already applied to EXP-058/060/057C. "
             "SAMPLE SIZE IS STILL THE OPEN RISK, measured not assumed: under "
             "EXP-057's own split (axis 0, population median, 20 um buffer) "
             "the strictest cut gives 143 train positives, below the 513 at "
             "which this repo's seam GNN first cleared zero. Re-centring the "
             "split on the positives and narrowing the buffer to 10 um takes "
             "that to 264, and a tier>=1 cut to 508. Choose the split before "
             "the model -- see docs/threads/seam_positive_sample_size.md."),

    Entry(series="C", est_minutes=45, spec=_s(
        id="EXP-063", title="Frankenmerge detection",
        question="Does mixed polarity, or a grammar, flag a false merge?",
        criterion="AUC at least 0.875 and precision at top 2% at least 0.41, "
                  "beating the global-shape baseline; Bar 3 above 0.5",
        requires=["EXP-057B"], requires_ran=["EXP-057"],
        inputs=[TOPOLOGY_K10, LABELS_NPZ, CB2_POSITIVES]),
        note="Bar adopted from the PCFG report's E1, which is stronger than the "
             "polarity-only bar first drafted. Polarity is the cheap baseline "
             "in the same run. Bar 3 has been 0.000 in every real run to date. "
             "Dependency swapped for the same reason as EXP-062. Of the two "
             "seam experiments this is the more runnable: its bar is an "
             "evaluation metric (AUC, precision at top 2%), which spends "
             "positives on the TEST side, and the unbalanced split leaves "
             "305-1,045 positives in val depending on the cut -- so it is less "
             "exposed to the train-side shortfall that gates EXP-062."),

    # --- D. scoring ----------------------------------------------------------
    Entry(series="D", est_minutes=120, spec=_s(
        id="EXP-064", title="Fixed-panel scorer bake-off",
        question="Which signal actually separates true continuations?",
        criterion="some single scorer reaches held-out AUROC 0.80; the stacked "
                  "model beats the best single by at least 0.03",
        requires=["EXP-060"], inputs=[TOPOLOGY_K10, LABELS],
        flags={"synthetic_fallback": False}),
        note="One fixed panel, many scorers: distance, tangent alignment, "
             "caliber continuity, tree-DNA cosine, PCFG grammar, neural-emission "
             "grammar, cut-face texture combiner, VLM verifier, stacked. "
             "Report calibration (ECE), not only ranking."),

    Entry(series="D", est_minutes=60, spec=_s(
        id="EXP-065", title="Scorer ablation",
        question="What does each feature actually contribute?",
        criterion="reported with 3 seeds and confidence intervals; no pass/fail",
        requires=["EXP-064"], inputs=[]),
        note="Leave-one-out from the stacked model."),

    # --- E. assembly ---------------------------------------------------------
    Entry(series="E", est_minutes=120, spec=_s(
        id="EXP-066", title="Solver bake-off",
        question="At fixed scores, which solver wins?",
        criterion="beats proximity union-find on ARI while holding merge "
                  "precision at or above 0.95",
        requires=["EXP-058", "EXP-064"], inputs=[TOPOLOGY_K10, LABELS]),
        note="Union-find, GAEC correlation clustering, constrained multicut, "
             "soma-seeded forest with the one-soma rule and the certified "
             "dendritic scaffold."),

    Entry(series="E", est_minutes=45, spec=_s(
        id="EXP-067", title="Abstention curve",
        question="Is there an operating point a proofreader would use?",
        criterion="at least 20% coverage at merge precision 0.95",
        requires=["EXP-066"], inputs=[]),
        note="The cut-face combiner reached precision 1.0 at 11% coverage on a "
             "different task; this asks whether assembly can too."),

    Entry(series="E", est_minutes=180, spec=_s(
        id="EXP-068", title="Scale and tiling",
        question="Does the result hold at 200 um and under tiling?",
        criterion="delta ARI within the confidence interval of EXP-066",
        requires=["EXP-066"], inputs=[]),
        note="100 -> 200 um; 2x2 vs 6x2 tiles of 100 um. Prior work: right-"
             "sized tiles gained ARI, enlarged ones did not."),

    # --- F. re-derivation ----------------------------------------------------
    Entry(series="F", est_minutes=120, spec=_s(
        id="EXP-069", title="Attic re-derivation",
        question="Does any retired morpho_grammar engine earn its numbers back?",
        criterion="beats the stacked EXP-064 scorer on the same panel with a "
                  "trained grammar; else EXPERIMENT_LOG.md stays superseded",
        requires=["EXP-064"], inputs=[TOPOLOGY_K10, LABELS],
        flags={"synthetic_fallback": False}),
        note="The only route out of attic/morpho_grammar/. Real substrate, "
             "EXP-064 protocol, trained checkpoint."),
]


def by_id(exp_id: str) -> Optional[Entry]:
    return next((e for e in REGISTRY if e.id == exp_id), None)


def state(entry: Entry, root: Optional[Path] = None) -> tuple[str, list[str]]:
    """``(state, reasons)`` for one entry. Never guesses; reads disk."""
    root = root or repo_root()
    res = load_result(entry.id, root)
    if res is not None and res.get("status"):
        return res["status"], [res.get("note", "")] if res.get("note") else []

    reasons = check_prerequisites(entry.spec, root)
    missing = check_inputs(entry.spec, root)
    reasons += [f"missing input: {m}" for m in missing]
    if reasons:
        return "blocked", reasons
    if not entry.implemented():
        return "not_implemented", [
            f"no module at {entry.module}" if entry.module
            else "no entry point declared"]
    return "ready", []


def next_runnable(root: Optional[Path] = None) -> list[Entry]:
    """Every experiment that could start right now, in program order.

    This is the autonomy primitive: run one, then ask again.
    """
    root = root or repo_root()
    return [e for e in REGISTRY if state(e, root)[0] == "ready"]


def unblocked_but_unwritten(root: Optional[Path] = None) -> list[Entry]:
    """Prerequisites and inputs satisfied; only the entry point is missing.

    Distinct from :func:`next_runnable` because the action differs: these are
    ready to *write*, not ready to *run*. Collapsing the two hides the fact
    that the program is gated on unwritten code rather than on evidence.
    """
    root = root or repo_root()
    out = []
    for e in REGISTRY:
        st, _ = state(e, root)
        if st == "not_implemented":
            out.append(e)
    return out


def blocked(root: Optional[Path] = None) -> list[tuple[Entry, list[str]]]:
    root = root or repo_root()
    out = []
    for e in REGISTRY:
        st, why = state(e, root)
        if st == "blocked":
            out.append((e, why))
    return out


def summary(root: Optional[Path] = None) -> dict:
    root = root or repo_root()
    counts: dict[str, int] = {}
    for e in REGISTRY:
        st = state(e, root)[0]
        counts[st] = counts.get(st, 0) + 1
    done = counts.get(STATUS_PASSED, 0)
    return {"total": len(REGISTRY), "by_state": counts,
            "passed": done, "fraction_passed": done / max(len(REGISTRY), 1)}


_MARK = {STATUS_PASSED: "PASS", "failed": "FAIL", "prerequisite_failed": "BLOCKED",
         "error": "ERROR", "ready": "ready", "blocked": "blocked",
         "not_implemented": "todo"}


def status_table(root: Optional[Path] = None) -> str:
    """Plain-text program status, in dependency order."""
    root = root or repo_root()
    rows = []
    width = max(len(e.id) for e in REGISTRY)
    for e in REGISTRY:
        st, why = state(e, root)
        mark = _MARK.get(st, st)
        line = f"  [{mark:>7}] {e.id:<{width}}  {e.spec.title}"
        if why and st in ("blocked", "not_implemented"):
            line += f"\n{'':>13}      -- {why[0]}"
        rows.append(line)
    s = summary(root)
    head = (f"Experiment program: {s['passed']}/{s['total']} passed  "
            + "  ".join(f"{k}={v}" for k, v in sorted(s["by_state"].items())))
    return head + "\n" + "\n".join(rows)
