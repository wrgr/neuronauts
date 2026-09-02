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
#: The widened, label-blind object set: every v117 object with a voxel in the
#: region, synapse-free ones included, plus object clouds read straight from the
#: segmentation volume. Built by scripts/enumerate_region_objects.py and
#: scripts/build_object_clouds.py.
OBJECTS_V117 = "data/substrate/c100um/objects_v117_mip5.npz"
OBJECT_CLOUDS = "data/substrate/c100um/object_clouds_mip5.npz"
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
             "(tier10) and 43.3% to 56.8% (all, same pair universe for both "
             "columns -- a first run took the endpoint column over finite "
             "gaps only and read 47.7%; the QA pass caught it), and recovers "
             "280 labelled atoms that have NO endpoint row at all, but does "
             "not approach the 90% bar. It also moves the answer key: on the "
             "full population, 325 links touch an atom with no endpoint row "
             "(this equals the count of endpoint-unreachable links by "
             "construction), and separately 138 object-only + 187 "
             "endpoint-only = 325 links are re-routed between atoms both "
             "metrics see -- the two 325s are a coincidence, not one number "
             "-- so EXP-060B's recall carries an error bar nobody drew. Adopt "
             "object distance downstream because it is the correct and "
             "strictly tighter quantity, not because it rescues the ball."),

    Entry(series="B", est_minutes=20, spec=_s(
        id="EXP-071", title="Contact adjacency and the connective gap",
        question="Are the fragments of one cell separated by distance, or by "
                 "objects the synapse-anchored population omits?",
        criterion="on at least 20 proofread cells DISJOINT from the twelve the "
                  "exploratory probe used: median nearest-sibling hop distance "
                  "at most 5 AND at least 50% of fragments within 3 hops AND at "
                  "least 80% of the objects holding the connective material "
                  "absent from the population. Direct atom-to-atom level-2 "
                  "contacts reported as a correctness check, expected zero. "
                  "Hops to the NEAREST sibling, never the clique",
        requires_ran=["EXP-057", "EXP-070"],
        inputs=[TOPOLOGY_KALL, OBJGEOM_KALL, LABELS_NPZ],
        flags={"network": True, "synthetic_fallback": False,
               "labels_used_only_for_evaluation": True}),
        module="neuronauts.experiments.exp071_connective_gap",
        note="The one that reframes series B. EXP-060/060B/061/070 all measured "
             "the distance between two SYNAPSE-ANCHORED atoms of one cell; the "
             "population admits a v117 object only if it owns a synapse in the "
             "cube, so the connective cable -- a passing neurite with no "
             "synapse of its own -- is never enumerated. Walking the proofread "
             "cell's own L2 graph instead, the nearest labelled sibling is a "
             "median 3 hops away (probe: 2 and ~102) and every object holding "
             "the material in between is missing from the population. Direct "
             "atom-to-atom L2 contacts are ZERO by construction and always "
             "will be: had the chunkedgraph joined two atoms they would be "
             "one atom, so "
             "'is there a contact' is the wrong query -- which is why this "
             "reports hop distance instead. Note the denominator trap repeats "
             "here: the clique median is 54.75 hops against a nearest-sibling "
             "median of 3, the same error CORRECTION.md caught in EXP-060. "
             "The bar was set by a hand probe on the twelve cells with the most "
             "fragments and is tested here on cells that probe never saw. "
             "Consequence: candidate generation was being asked to bridge a gap "
             "the SUBSTRATE created, so the fix is upstream of every scorer and "
             "solver -- see scripts/enumerate_region_objects.py."),

    Entry(series="B", est_minutes=45, spec=_s(
        id="EXP-072", title="Object-level proposal on the widened substrate",
        question="Does proposing at the object level, over every v117 object "
                 "rather than only synapse-carrying ones, reach the spanning "
                 "links at a usable panel size?",
        criterion="on the widened substrate, chained recall of MST spanning "
                  "links at radius 2 um, panel cap 20 and at most 3 hops must "
                  "exceed 50%, AND beat the population-only control by at "
                  "least 20 points, AND keep the median number of reachable "
                  "LABELLED atoms at or under 50. The third clause was added "
                  "before the run: chained recall without a bound on what it "
                  "reaches is EXP-058's union-find result, which scored recall "
                  "1.0 at pair precision 0.0006 by collapsing the population "
                  "into one cluster",
        requires=["EXP-071"], requires_ran=["EXP-060B", "EXP-070"],
        inputs=[TOPOLOGY_KALL, LABELS_NPZ, OBJECT_CLOUDS, OBJECTS_V117],
        flags={"synthetic_fallback": False,
               "labels_used_only_for_evaluation": True}),
        module="neuronauts.experiments.exp072_object_proposal",
        note="The test of whether EXP-071's substrate fix is SUFFICIENT, not "
             "just necessary. Proposer is deliberately stupid -- objects within "
             "a radius, ranked by closest approach, capped. No tangent, "
             "caliber, grammar or learned score, so a gain is attributable to "
             "the substrate alone. A 40 um probe already says it is not "
             "sufficient: at r=2 um chained recall rises 10.9% -> 100% as the "
             "cap goes 5 -> 100, while precision sits at 0.08-0.13% throughout "
             "and the reachable-labelled set goes 21 -> 703 of 704. Dense "
             "neuropil means chaining reaches everything. Direct recall is also "
             "WORSE than EXP-060B's 12% at the same cap, because the panel "
             "budget now gets spent on connective and dust objects physically "
             "closer than the true partner -- more objects ranked by distance "
             "alone is actively harmful. Read together with EXP-071: the "
             "substrate omission was real and fixing it is necessary, but "
             "distance over a complete object set still cannot propose. What "
             "is missing is structural constraint, which is what the skeleton "
             "layer was always for. A physical dust floor (synapse-free "
             "objects under 0.041 um^3 dropped, synapse-carriers exempt) is "
             "canonical from the second run on; a sweep on the 40 um mip-2 "
             "substrate (results/EXP-072/probe_40um_mip2_dust_floor.md) "
             "showed it removes 87% of objects and moves precision by nothing "
             "-- the collapse is the synapse-carrying population connecting "
             "itself at 2 um, not debris."),

    Entry(series="B", est_minutes=60, spec=_s(
        id="EXP-073", title="Constrained chaining: does structure prune the panel?",
        question="Do cheap structural constraints on the bridging object make "
                 "the chained panel sparse enough to use, where distance alone "
                 "could not?",
        criterion="EXP-072's bar, unchanged, so the two are a direct A/B: "
                  "chained recall of MST spanning links at radius 2 um, cap 20, "
                  "at most 3 hops must exceed 50% while the median number of "
                  "reachable LABELLED atoms stays at or under 50. Constraints "
                  "are hard filters on the bridging object computed from the "
                  "object clouds -- cable-like (elongated, bounded extent), "
                  "through (the two fragments attach at opposite ends of the "
                  "bridge), collinear. No skeleton, no fetch, nothing learned",
        requires=["EXP-071"], requires_ran=["EXP-072"],
        inputs=[TOPOLOGY_KALL, LABELS_NPZ, OBJECT_CLOUDS, OBJECTS_V117],
        flags={"synthetic_fallback": False,
               "labels_used_only_for_evaluation": True}),
        module="neuronauts.experiments.exp073_constrained_chain",
        note="The user's hypothesis, made falsifiable: skeletons were never "
             "mainly a distance substrate, they carry structure -- tree-ness, "
             "degree, the one-soma rule, caliber continuity -- and structure is "
             "exactly what EXP-072's collapse lacks. This tests the cheapest "
             "form of it, at the OBJECT level (no L2 graph, no skeleton fetch): "
             "is the bridge a piece of cable, and do the two fragments hang off "
             "opposite ends of it? Hard filters, because EXP-060B's problem was "
             "panel SIZE, and a filter that prunes at near-zero recall cost is "
             "worth more there than a scorer that ranks. Both this and EXP-072 "
             "now apply a physical dust floor -- synapse-free objects under "
             "0.041 um^3 (1,000 voxels at 32x32x40 nm) are dropped, "
             "synapse-carriers exempt -- after the first runs thresholded at 2 "
             "read voxels, which excluded nothing. Canonical substrate is the "
             "near-isotropic mip-2 (32x32x40 nm) read once its full-cube "
             "enumeration lands; mip-5 results before that are superseded."),

    # --- C. cuts and frankenmerge detection ---------------------------------
    Entry(series="C", est_minutes=90, spec=_s(
        id="EXP-062", title="Real-L2 cuts and seam location",
        question="Do cuts on real L2 adjacency beat MST-geometry cuts, and can "
                 "we pick the right edge?",
        criterion="at least 90% same-lineage pair recall AND at least 50% "
                  "cross-lineage split recall; seam top-1 above 25%; and Bar 3 "
                  "(frankenmerge split recall) above 0.5 -- moved here from "
                  "EXP-063 because it is a property of the cut, not of "
                  "detection",
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

    Entry(series="C", est_minutes=15, spec=_s(
        id="EXP-063", title="Frankenmerge detection",
        question="Does mixed polarity, object shape, or their combination flag "
                 "a false merge on the harness substrate -- and do CB2's "
                 "uncorroborated tiers look like frankenmerges to a detector "
                 "trained without them?",
        criterion="tier>=10 only (size-controlled); positives = atoms our own "
                  "v1822 tally calls mixed-lineage; strict negatives = pure "
                  "atoms with a proofread owner, CB2-touched atoms excluded "
                  "from every negative set; held out by a positives-centred "
                  "spatial split with a 10 um buffer. PASS when the best "
                  "non-size feature set reaches held-out AUC >= 0.875 AND "
                  "exceeds the size-only rung by >= 0.02. Precision at the top "
                  "2% reported against lenient negatives, not gated. Bar 3 is "
                  "a cut metric and moves to EXP-062. CB2 tiers 1 and 0 scored "
                  "as a label-validity diagnostic, no bar",
        requires=["EXP-057B"], requires_ran=["EXP-057", "EXP-070"],
        inputs=[TOPOLOGY_K10, OBJGEOM_K10, POPULATION, LABELS_NPZ,
               CB2_POSITIVES],
        flags={"synthetic_fallback": False,
               "labels_used_only_for_evaluation": False}),
        module="neuronauts.experiments.exp063_frankenmerge_detection",
        note="CRITERION AMENDED BEFORE THE FIRST RUN, and why: the first draft "
             "carried 'precision@2% >= 0.41' and 'Bar 3 above 0.5'. Bar 3 is "
             "frankenmerge SPLIT recall -- it needs a cut operator this "
             "experiment does not have and EXP-062 owns; reporting 0.000 for "
             "it here, as every prior run did, says nothing about detection. "
             "Precision@2% at the ~60% strict base rate is uninformative, so "
             "it is reported under lenient negatives and the gate is AUC, "
             "which is base-rate invariant. The 0.875 came from the PCFG "
             "report's shape detector on v117 roots at a 3.78% base rate; it "
             "is kept as the bar and the same ten shape features are run as a "
             "rung on this substrate (numpy port of global_shape_merge, "
             "validated against scikit-learn in tests/test_atom_features.py) "
             "so 'beating the global-shape baseline' is measured here, not "
             "compared across substrates. Size is the confound: on the full "
             "population a mixed atom has a median 818 L2 nodes vs 28 for a "
             "trustworthy negative, so this runs on tier>=10 only (945 vs 809) "
             "and must beat a size-only rung. The CB2 tier-1/tier-0 scoring is "
             "the label-validity check EXP-057B's own caveat asked for: those "
             "tiers are outside the training positive definition, so a "
             "detector that fires on them is evidence they are real."),

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
