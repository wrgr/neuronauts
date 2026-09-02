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

    Entry(series="A", est_minutes=30, spec=_s(
        id="EXP-057B", title="ConnectomeBench2 intake",
        question="Can an external corpus lift us past 56 trustworthy "
                 "seam positives?",
        criterion="at least 1,000 merge or split decisions map onto v117 roots "
                  "in or near the harness cube",
        requires=[], inputs=[],
        flags={"network": True}),
        note="716,485 expert proofreading decisions (FlyWire/MICrONS/Fish1/H01). "
             "Its third task, mask segmentation for merge-error correction, is "
             "our seam-location problem renamed. Gates the sample size of "
             "EXP-062/063, so run it early."),

    Entry(series="A", est_minutes=30, spec=_s(
        id="EXP-057C", title="Published embedding intake",
        question="Is a usable tree-DNA already published for this volume?",
        criterion="same-cell vs different-cell fragment cosine separation at or "
                  "above our own tree-DNA on the harness",
        requires=["EXP-057"], inputs=[LABELS],
        flags={"network": True}),
        note="Weis et al. 2025 GraphDINO over >30,000 MICrONS excitatory "
             "neurons. Free class-conditioning for the grammar mixture if the "
             "per-root embeddings are released."),

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
        requires=["EXP-058"], inputs=[]),
        note="Five of six callers already delegate; experiments/pcfg/"
             "conn_metric.py is the outstanding one."),

    # --- B. candidate generation --------------------------------------------
    Entry(series="B", est_minutes=60, spec=_s(
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

    # --- C. cuts and frankenmerge detection ---------------------------------
    Entry(series="C", est_minutes=90, spec=_s(
        id="EXP-062", title="Real-L2 cuts and seam location",
        question="Do cuts on real L2 adjacency beat MST-geometry cuts, and can "
                 "we pick the right edge?",
        criterion="at least 90% same-lineage pair recall AND at least 50% "
                  "cross-lineage split recall; seam top-1 above 25%",
        requires=["EXP-057"], requires_ran=["EXP-057B"],
        inputs=[TOPOLOGY_K10, LABELS_NPZ],
        flags={"synthetic_fallback": False}),
        note="Merges the plan's 062 with the PCFG report's E2/E3. EXP-056 "
             "falsified every single global edge-length threshold. Correctly "
             "blocked: EXP-057 measured 56 trustworthy seam positives, 15 of "
             "them in train, against a seam GNN that was net-negative at 150. "
             "EXP-057B is the unblock."),

    Entry(series="C", est_minutes=45, spec=_s(
        id="EXP-063", title="Frankenmerge detection",
        question="Does mixed polarity, or a grammar, flag a false merge?",
        criterion="AUC at least 0.875 and precision at top 2% at least 0.41, "
                  "beating the global-shape baseline; Bar 3 above 0.5",
        requires=["EXP-057"], requires_ran=["EXP-057B"],
        inputs=[TOPOLOGY_K10, LABELS_NPZ]),
        note="Bar adopted from the PCFG report's E1, which is stronger than the "
             "polarity-only bar first drafted. Polarity is the cheap baseline "
             "in the same run. Bar 3 has been 0.000 in every real run to date."),

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
