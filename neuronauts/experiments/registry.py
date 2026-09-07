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
# The human proofreading corpus (EXP-082): every logged operation endpoint
# joined to the final skeleton of the cell it built, and the skeletons
# themselves. Written by results/EXP-082/build_join.py and
# scripts/fetch_seed_skeletons.py.
EDIT_JOIN_V082 = "data/external/edit_join_v082.npz"
CELL_SKELETONS = "data/external/cell_skeletons"
EDIT_HISTORY = "data/external/edit_history"
EXP083_SHAPE_LIB = "scripts/exp083_shape_lib.py"


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

    Entry(series="B", est_minutes=30, spec=_s(
        id="EXP-074", title="Soma-seeded growth, distance only",
        question="Can a grower seeded at a cell body recover that cell's in-box "
                 "root process, and does it know when to stop?",
        criterion="two populations scored separately, never pooled. On the 67 "
                  "cells that need joining, at radius 2 um: recovery of target "
                  "fragments at least 60% AND purity at least 80%, counting "
                  "only labelled objects added. On the 36 already-whole cells, "
                  "add nothing in at least 70%. Distance alone -- no "
                  "compartment, caliber, direction or learned score",
        requires_ran=["EXP-071", "EXP-072"],
        inputs=[OBJECT_CLOUDS, OBJECTS_V117, TOPOLOGY_KALL, LABELS_NPZ],
        flags={"synthetic_fallback": False,
               "labels_used_only_for_evaluation": True}),
        module="neuronauts.experiments.exp074_seeded_growth",
        note="The first experiment scored on the task a grammar performs, after "
             "EXP-060 through EXP-073 all scored pairwise join-finding and "
             "collapsed at ~0.09% precision. Target is box_truth.seeded_target: "
             "the seed's own in-box component, with compartment-crossing links "
             "dropped. Bars derive from the 103-cell census in "
             "docs/threads/exp074_spec.md -- 60% recovery because 80% of scored "
             "links are within 2 um and that is the distance-only ceiling; 80% "
             "purity is a judgement, stated as one; 70% abstention is set low "
             "because nothing has ever measured it. Scoring separates recovered "
             "/ contamination / unknown, because unlabelled connective cable is "
             "neither right nor wrong and folding it either way would hide the "
             "result. Deliberately not a grammar: it exists so EXP-075's "
             "grammar has something to be measured against."),

    Entry(series="B", est_minutes=40, spec=_s(
        id="EXP-086", title="Is an unexplained cut surface a real split?",
        question="Of the cut surfaces EXP-085's terminal grammar calls "
                 "unexplained, what fraction are genuine segmentation splits "
                 "-- cable continuing in another object owned by the same "
                 "proofread cell -- rather than artifacts of our own pipeline?",
        criterion="on tips of objects carrying a trustworthy proofread owner, "
                  "with tips and their grammar class computed label-blind "
                  "exactly as EXP-085 computed them, at continuation radius "
                  "5 um inside a 60 degree outward cone, and under the fixed "
                  "precedence tip_detection > true_split > search_exits_box > "
                  "dust_floor > unresolved: (1) at least 60% of UNEXPLAINED "
                  "tips are true splits, denominator every unexplained tip on "
                  "a labelled object including those the test cannot "
                  "adjudicate; AND (2) the synaptic-terminal class's "
                  "true-split rate is at least 3x lower than the unexplained "
                  "class's. BOTH must hold. If either fails, EXP-085's "
                  "unexplained population is contaminated and must not be used "
                  "as a training negative class as it stands. The "
                  "field-boundary class is counted as a third, non-binding "
                  "control for false boundaries. Labels are evaluation-only",
        requires_ran=["EXP-071", "EXP-072"],
        inputs=[OBJECT_CLOUDS, OBJECTS_V117, LABELS_NPZ, POPULATION],
        flags={"synthetic_fallback": False,
               "labels_used_only_for_evaluation": True}),
        module="neuronauts.experiments.exp086_unexplained_split",
        note="EXP-085 offered ~281,790 extrapolated 'unexplained' cut surfaces "
             "as the negative class a stop-versus-extend decision would train "
             "on, and said in its own Limits that unexplained has never been "
             "verified to mean true split. This measures that before anything "
             "trains on it. The bar is where it is for two different reasons, "
             "and they are not equally strong. Clause 2 -- the "
             "synaptic-terminal control at 3x -- is the clause that actually "
             "tests the grammar, and it is the robust one: whatever suppresses "
             "the measured true-split rate suppresses it in both classes and "
             "largely cancels in a ratio, so a grammar whose STOP label means "
             "something must separate the two by a wide factor. 3x is a "
             "judgement, stated as one, chosen so a modest separation cannot "
             "be read as a result. Clause 1 -- 60% absolute -- is a floor "
             "rather than an estimate, because a continuation into a fragment "
             "of the same cell that carries no trustworthy label cannot be "
             "counted and lands in 'unresolved' instead. That is the known "
             "risk to this bar: it can fail because labels are sparse rather "
             "than because the tips are artifacts. The design makes those two "
             "failures distinguishable rather than pretending they cannot "
             "happen -- the artifact-cause breakdown (tip_detection, "
             "search_exits_box, dust_floor) says how much is our pipeline, and "
             "the size of the unresolved bucket says how much is label reach. "
             "60% is kept unchanged from the brief because nothing in "
             "EXP-081/085 measures label reach at a tip, so lowering it here "
             "would be a guess dressed as a correction. Three outcomes are "
             "reported and never pooled, on the EXP-074 principle that folding "
             "an unknown into either side hides the result: true split, "
             "artifact with its named cause, unresolved. Resolution limit "
             "carried in the result: tips and the continuation test both run "
             "on mip-5 centroid clouds as EXP-085's did, so 'does it continue "
             "when examined more finely' is answered only as full point "
             "density against the 1,500-point tip-finding subsample, not at "
             "mip 2 -- no cube-wide mip-2 cloud exists yet."),

    Entry(series="B", est_minutes=120, spec=_s(
        id="EXP-089", title="Where-to-edit prior on v117-only features",
        question="Does EXP-082's where-to-edit prior survive when every feature "
                 "is computed from what a grower actually has -- v117 fragments "
                 "-- instead of from the final proofread reconstruction?",
        criterion="Same cells, same lattice (one row per final-skeleton "
                  "vertex), same label (a merge endpoint within 2 um) and the "
                  "same held-out-by-cell protocol (5-fold GroupKFold on cell "
                  "root id) as EXP-082, so the two numbers are comparable; only "
                  "the features change. PASS when (a) the v117-only arm -- "
                  "caliber from EXP-088's v117 measurement, path distance "
                  "inside the vertex's own v117 fragment, degree inside that "
                  "fragment, position, and compartment DROPPED rather than "
                  "substituted with its true label -- reaches held-out area "
                  "under the curve >= 0.70 against the 0.779 proofread-feature "
                  "ceiling, AND (b) the proofread-feature control refitted "
                  "under this harness reproduces 0.779 within +/- 0.03. If (b) "
                  "fails, the reimplementation is the difference and no claim "
                  "about the substitution is made. The feature-by-feature drop, "
                  "proofread versus v117, is reported for every column and is "
                  "the result that matters more than the headline; the "
                  "v117_plus and mean-distance-transform arms are reported, "
                  "not gated",
        requires=[], requires_ran=[],
        inputs=["data/external/cell_skeletons", "data/external/edit_history"],
        params={"merge_match_nm": 2000.0, "n_folds": 5, "v117_auc_bar": 0.70,
                "exp082_auc": 0.779, "control_tol": 0.03},
        flags={"synthetic_fallback": False, "network": True,
               "labels_used_only_for_evaluation": False,
               "labels_used_for_training": "human proofreading merge "
                                           "endpoints, train folds only, "
                                           "grouped by cell"},
        budget_minutes=150),
        module="neuronauts.experiments.exp089_edit_prior_v117",
        note="EXP-082's own limits section says the transfer this experiment "
             "measures is 'assumed, not shown': skeleton radius, compartment "
             "and path distance all come from the final reconstruction, and its "
             "usable-signal table asks for exactly one thing before the prior "
             "is spent -- 'recompute radius from v117 fragments instead of the "
             "proofread skeleton'. Since the signal is dominated by caliber "
             "(radius alone 0.750 of the 0.779; dropping radius costs 0.057 "
             "while dropping anything else costs 0.000-0.004), the whole "
             "usability of the prior rests on that one substitution. "
             "THE BAR, AND WHY THESE NUMBERS. 0.70 is not a hedge: EXP-081 "
             "measured local pairwise geometry at the grower's frontier at "
             "0.63, so a v117-only prior below ~0.65 buys nothing over what we "
             "already have and 0.70 is the first value that is clearly a "
             "different regime while allowing real loss from the substitution. "
             "The +/- 0.03 control is the tighter of the two and is the one "
             "that can invalidate the run: EXP-082 was a one-off script, not a "
             "registered experiment, so its 0.779 has never been reproduced "
             "under this harness. If the control misses, the reimplementation "
             "is the difference and the v117 number says nothing about v117 -- "
             "the module is written so that outcome is reported rather than "
             "explained away. "
             "COMPARTMENT IS DROPPED, NOT SUBSTITUTED. No compartment "
             "predictor in this repository runs on a v117 fragment, and "
             "feeding in the true pcg_skel label would smuggle the proofread "
             "reconstruction back into a v117-only arm. So it is dropped, and "
             "a seven-column proofread arm is fitted alongside so the cost of "
             "losing compartment is separated from the cost of substituting "
             "caliber instead of pooled with it. EXP-082's own ablation says "
             "this should be cheap (0.775-0.779 without it); that prediction is "
             "checked here rather than assumed. "
             "CALIBER IS EXP-088'S, IMPORTED, NEVER REIMPLEMENTED. This uses "
             "neuronauts.harness.v117_caliber.load_l2_caliber plus "
             "vertex_radii_from_l2 -- the level-2 route that module's own "
             "docstring names as 'the route EXP-082 needs, because its unit is "
             "a skeleton vertex and it has 650,200 of them'. Two independent "
             "caliber definitions would make the comparison against 0.779 "
             "uninterpretable, so a missing or renamed module stops the run "
             "with a named error rather than falling back to a local estimator. "
             "That module also says neither of its estimators has been "
             "validated against the proofread radius it replaces and that the "
             "caller must measure the agreement; this run reports Spearman, "
             "Pearson and the median ratio against the skeleton radius at the "
             "same vertices, reported and not gated -- a weak correlation with "
             "a surviving area under the curve would be a finding, not a "
             "failure. max_dt_nm is the gated statistic and mean_dt_nm is run "
             "alongside, because a level-2 chunk maximum over-reports a shaft "
             "sharing a chunk with a bouton and the mean under-reports; which "
             "was used is recorded rather than left implicit. "
             "NO requires/requires_ran ON PURPOSE. EXP-082 is a set of "
             "scripts under results/ with no result.json, so declaring it "
             "would block this permanently on a prerequisite that will never "
             "be written; EXP-088 is gated by the import instead, which fails "
             "loudly and says why. Add requires_ran=['EXP-088'] once that "
             "experiment writes a result. "
             "WHAT THIS DOES NOT MEASURE, stated before the run: the "
             "evaluation lattice stays EXP-082's proofread skeleton vertices, "
             "because changing lattice and features at once would make the two "
             "numbers incomparable -- a v117-native lattice is a separate "
             "experiment. A v117 fragment is measured only where it overlaps "
             "this cell's cable, so fragment size and reach are underestimates "
             "and sit in the ungated v117_plus arm. Caliber is evaluated at "
             "positions the proofread skeletonization chose. Needs the "
             "chunkedgraph once (level-2 ids -> roots at the v117 timestamp, "
             "the same call scripts/probe_v117_geometry_route.py already gets "
             "usable answers from) and caches the answer per cell under "
             "data/external/v117_fragment_map/, so only the first run is "
             "online. It also needs a level-2 attribute cache carrying "
             "max_dt_nm -- data/substrate/geom/l2_attributes.npz and/or "
             "data/external/soma_viz/connective_l2_attrs.npz, both tried and "
             "merged for coverage, per-source hit counts reported -- and stops "
             "rather than proceeding if neither exists, since without one "
             "there is no v117 caliber to substitute. Vertices whose level-2 "
             "node is in neither cache are excluded from BOTH arms so the "
             "comparison stays like-for-like."),

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

    Entry(series="C", est_minutes=45, spec=_s(
        id="EXP-090", title="Does tree-level evidence compound?",
        question="Does an aggregate of per-bifurcation conservation residuals "
                 "separate a correctly assembled arbor from one carrying k "
                 "wrong joins, and does the separation grow with k?",
        criterion="held out by cell, on the aggregate declared before the run "
                  "(fraction of bifurcations with |p-3| above 2.0, unrooted "
                  "role assignment), all three of: pooled AUC at least 0.80 at "
                  "k=3 wrong joins against 0.675 for EXP-084's single branch "
                  "point; the AUC monotone in k, k=1 < k=3 < k=5; and the "
                  "same-cell displacement control no higher than the "
                  "foreign-cable arm at k=3, within 0.02. Every aggregate, both "
                  "role assignments, both residual families, both controls and "
                  "a label-shuffle null are reported, so the primary was not "
                  "chosen after the fact",
        inputs=[CELL_SKELETONS, EXP083_SHAPE_LIB],
        flags={"synthetic_fallback": False,
               "labels_used_only_for_evaluation": True}),
        module="neuronauts.experiments.exp090_compounding_evidence",
        note="EXP-083 supplies the graft construction and is the counterweight this experiment is read against, but it is NOT declared as a prerequisite: it has a result.json and no registry entry, so declaring it fails the registry's own consistency tests. Registering it now would mean writing its bar after its data, which is the one thing this program does not do. Named here, gated nowhere -- and the same is true of EXP-084, whose 0.675 is the number this bar is set against. EXP-084 found Cajal's material conservation holds in this tissue "
             "-- Murray exponent median 3.18 over 3,781 real bifurcations -- "
             "and that a caliber-mismatched branch point separates at 0.675 "
             "with no parameters. Its honest reading then claimed the value of "
             "that weak signal is that it COMPOUNDS over a tree. Nothing tested "
             "it. This does, and it is the property the program needs and does "
             "not have, so it is worth a bar rather than a paragraph. "
             "THE BAR. 0.675 is the thing to beat, not to match: an aggregate "
             "over a few hundred branch points that lands where one branch "
             "point already lands has not compounded, it has averaged. 0.80 at "
             "k=3 is a judgement and is stated as one -- it sits above "
             "everything EXP-083's whole-cell shape score reached in the same "
             "absolute setting (0.505 at random-cut scale, 0.562 with a third "
             "of the neuron foreign) and well below EXP-063's 0.958, which was "
             "a different substrate without size matching inside the pair. "
             "Monotonicity in k is a CLAUSE and not a remark, because a single "
             "high number at k=5 is equally consistent with a size or "
             "bifurcation-count artifact; the k=1 sites are a prefix of the k=3 "
             "sites which are a prefix of the k=5 sites, and only cells that "
             "supply all five are used, so the curve is not a changing cell "
             "set. The same-cell displacement control is the clause that can "
             "sink the experiment regardless of the headline: in EXP-083 "
             "moving a cell's OWN branch to the wrong site was detected BETTER "
             "(0.710) than importing another cell's (0.642), which showed the "
             "score was reading local placement wearing a global costume. If "
             "the conservation aggregate reproduces that ordering it is reading "
             "the same thing and the compounding story is wrong. "
             "CONSTRUCTION IS EXP-083'S, imported from "
             "scripts/exp083_shape_lib.py rather than rebuilt -- same "
             "breadth-first rooted tree, same size-matched foreign subtree "
             "grafted at the same stem edge, same second removal so the correct "
             "assembly is partial too. Rebuilding it differently would make the "
             "result incomparable with EXP-083's, which is most of its value. "
             "ONE DEVIATION, deliberate and stated: sites are restricted to "
             "vertices whose parent is a two-child bifurcation whose sibling "
             "survives, because a graft replacing the only child of a path "
             "vertex creates no branch point and a conservation law has nothing "
             "to read there. That is a strict subset of EXP-083's sites. "
             "It also means the corruption here is WEAKER than EXP-084's, which "
             "swapped both daughters; here one daughter is foreign, so the "
             "run also measures the single-join-site AUC inside this very "
             "construction as the like-for-like reference, alongside the "
             "declared 0.675. A blind whole-tree aggregate and an aggregate "
             "over the k KNOWN join sites are reported separately, because an "
             "assembler knows where it joined and an auditor of someone else's "
             "assembly does not, and conflating them would answer an easier "
             "question than the one asked. "
             "EXP-084 IS NOT LISTED AS A PREREQUISITE even though this tests "
             "its claim: it has results/EXP-084/evaluation.md and no "
             "result.json, so requires_ran would block this on bookkeeping "
             "rather than on science. EXP-083 is listed because its artifact "
             "exists. INPUTS are the same 103 proofread arbors EXP-083 used, "
             "built by scripts/fetch_seed_skeletons.py, which needs a CAVE "
             "token; no cell cards, no labels, no training data. "
             "OPEN RISK, measured not assumed: a cell that cannot supply five "
             "disjoint sites in the 30-300 um band with a size-matched donor is "
             "dropped from every k. The run reports cells_contributing and a "
             "per-reason rejection tally, so if the site filter rather than the "
             "biology is the binding constraint that is visible in the result "
             "instead of inferred from a low pair count."),

    Entry(series="D", est_minutes=90, spec=_s(
        id="EXP-087", title="Terminal grammar as a stop-versus-extend classifier",
        question="Trained on the grammar's label-free terminal classes, does a "
                 "stop-versus-extend classifier reach the precision the real "
                 "frontier demands at the real base rate?",
        criterion="Trained ONLY on EXP-085's label-free terminal classes (unexplained = "
                  "extend, synaptic terminal = stop, field boundary dropped); scored on "
                  "EXP-081's frontier -- tips on the soma fragment of the first 40 "
                  "soma-seeded cells, live when a fragment of box_truth.seeded_target lies "
                  "within 5 um. Held out BY CELL: every fragment named by any cell card is "
                  "removed from the training object pool, so no cell trains the model that "
                  "scores it. PASS when the better of the two declared models (the repo's "
                  "GradientBoostedStumps, and a scikit-learn RandomForest when that package "
                  "is installed) clears ALL THREE clauses: (1) precision at the top-k "
                  "operating point >= 0.30, k = the number of live sites on the held-out "
                  "frontier, i.e. one extension per cell -- EXP-081's top-34 row, where "
                  "local geometry scored 0.0% at a 1.6% base rate; AND (2) false-extend "
                  "rate on true dead ends <= 2% at a threshold calibrated on the TRAINING "
                  "population alone (the score at which the classifier extends at 2% of the "
                  "grammar's own legitimate stops); no evaluation label sets that "
                  "threshold; AND (3) at that same threshold the grower actually makes the "
                  "join -- at least one true extension, and at least as many true "
                  "extensions as false ones, which is EXP-081's own inequality ('a "
                  "false-extend rate of 5% per tip yields about 2.3 false joins for every 1 "
                  "correct one'). Clause 3 is added because clause 2 alone is cleared by a "
                  "model that never extends, which would let abstention pass as transfer; "
                  "it makes the bar stricter, not looser. Two models on one split is "
                  "selection optimism and a small margin between them is noise. Reported, "
                  "not gated: area under the curve, the same pipeline with training labels "
                  "shuffled (5 seeds), the hypergeometric tail and the Wilson interval on "
                  "precision at top-k, the unlearned EXP-081 rung (alignment x proximity) "
                  "on this exact frontier, and a stricter region-disjoint split (axis 0, 20 "
                  "um buffer). Features are mip-5 v117 fragment geometry only -- no synapse "
                  "feature (it defines the training label), no box-face distance, no "
                  "proofread skeleton",
        requires=["EXP-086"],
        inputs=[OBJECT_CLOUDS, POPULATION],
        params={"n_eval_cells": 40, "n_train_objects": 3000,
                "max_train_tips": 8000, "live_radius_nm": 5000.0,
                "synapse_radius_nm": 1500.0, "edge_margin_nm": 3000.0,
                "split_axis": 0, "split_buffer_nm": 20000.0, "seed": 0,
                "n_null_repeats": 5, "bar_precision_at_k": 0.30,
                "bar_false_extend": 0.02, "calibration_false_extend": 0.02},
        flags={"synthetic_fallback": False,
               "labels_used_only_for_evaluation": True,
               "training_labels": "EXP-085 terminal grammar, derived from the "
                                  "segmentation and its own synapses; no "
                                  "proofreading, no cell identity",
               "labels_used_for_leakage_exclusion": "cell-card fragment ids are "
                                                    "removed from the training "
                                                    "pool, never used to fit"},
        budget_minutes=120),
        module="neuronauts.experiments.exp087_terminal_classifier",
        note="The experiment EXP-081 asked for and EXP-085 made possible: score "
             "every tip of the real frontier at its real 1.6% base rate, with a "
             "stop rule trained on labels no human produced. EXP-081 measured "
             "2,137 tips over 40 cells, 34 of them live, and local geometry "
             "found 0 of the top 34 -- 0.0% precision, best single feature AUC "
             "0.630. EXP-085 then classified 8,183 cut surfaces label-free into "
             "synaptic terminal (25.8%), field boundary (30.6%) and unexplained "
             "(43.6%), which is a training population three orders of magnitude "
             "larger than the 56 proofread seam positives everything else in "
             "series C runs on. This asks whether that population transfers. "
             "WHY THE BAR IS 0.30 AND 2%: the 2% is not a judgment, it is "
             "EXP-081's derived arithmetic -- 46 tips per cell and one true "
             "join means a 5% per-tip false-extend rate buys 2.3 false joins "
             "per correct one, and the rate has to sit below roughly 2% for the "
             "true join to win. The 0.30 is a judgment, stated as one: twenty "
             "times the 1.6% base rate, against a measured 0.0%, chosen high "
             "enough that clearing it would mean the grammar had done something "
             "local geometry could not rather than something a wide confidence "
             "interval could explain. With k about 34 the estimate is coarse, "
             "so precision at top-k carries a Wilson interval and a "
             "hypergeometric tail and neither is gated. A THIRD CLAUSE WAS "
             "ADDED BEFORE THE FIRST REAL RUN, and it makes the bar stricter, "
             "not looser: a synthetic dry run of the module PASSED clause 2 at "
             "0% recall, because a model that never extends has a 0% "
             "false-extend rate by construction. Clause 3 requires the grower "
             "to actually make the join -- at least one true extension and at "
             "least as many true as false -- which is EXP-081's own inequality "
             "rather than a new number. THE GATE IS A CELL-DISJOINT SPLIT, not "
             "the axis-0 spatial seam: the predeclared operating point is "
             "EXP-081's top-34, which needs all 40 cells, and the seam roughly "
             "halves them. Every fragment any cell card names is barred from "
             "the training pool, so no object a cell is made of trains the "
             "model that scores it; the seam split is run and reported anyway "
             "as the stricter control, partly to bound the one leak "
             "cell-disjointness cannot close -- unlabeled connective cable of "
             "an evaluation cell, which nothing names. THE CEILING IS "
             "STRUCTURAL AND IS NOT HIDDEN: the training positive is "
             "'unexplained cut surface', a probable split of SOME object, while "
             "the evaluation positive is 'a fragment of THIS seed's target "
             "within 5 um'. A tip can be honestly unexplained and still be a "
             "dead end for this grower, so a failure is ambiguous between the "
             "grammar not transferring and the target being narrower than the "
             "class, and the run says so rather than reading a failure as a "
             "verdict on the grammar. Features are mip-5 v117 fragment geometry "
             "only -- fourteen columns, self geometry of the ending plus "
             "EXP-081's own neighborhood family reproduced exactly so its rung "
             "is measured on this frontier instead of quoted across runs. NO "
             "SYNAPSE FEATURE OF ANY KIND: 'the object's own synapse is within "
             "1.5 um' IS the training negative, and EXP-085 already caught the "
             "ownership-free version returning 99.96% explained by chance at "
             "1.04 um mean synapse spacing. No box-face distance either, "
             "because it defines the dropped class. What mip-5 cannot supply is "
             "stated instead of faked: a 256x256x160 nm voxel cannot resolve "
             "the flare of a bouton against a cut face, so a negative on the "
             "shape columns is a statement about mip-5 and a mip-2 pass is the "
             "next step, exactly as EXP-081 said of its own numbers. "
             "requires=['EXP-086'] and nothing else: the unexplained class is "
             "only a negative population if it is really splits, which is what "
             "EXP-086 measures. EXP-081 and EXP-085 are deliberately NOT in "
             "requires_ran even though this is built on them -- they were run "
             "as ad-hoc scripts and wrote no results/<id>/result.json, so "
             "gating on them would block this experiment forever on an artifact "
             "that does not exist. Instead the run re-derives EXP-081's "
             "frontier from its own constants and prints the reproduced tip / "
             "live / base-rate counts against the published 2,137 / 34 / 1.6%, "
             "which is a check rather than a citation. est_minutes 90 is an "
             "estimate and not a measurement: the run builds one cKDTree over "
             "every mip-5 cloud point in the cube (about 72.5M) and issues one "
             "6 um ball query per tip, for roughly 10,000 tips; EXP-074 built "
             "the same tree and finished in 15 minutes with a heavier per-cell "
             "workload."),

    Entry(series="D", est_minutes=240, spec=_s(
        id="EXP-088", title="Conservation priors on real joins, v117 radii",
        question="Does the Murray/Cajal conservation prior separate a real "
                 "human join from a plausible wrong one at the same site, and "
                 "does it survive caliber measured on v117 fragments instead "
                 "of a proofread skeleton?",
        criterion="one site set, three scorings, held out by cell in 5 folds. "
                  "Sites are post-v117 human merges from the EXP-082 corpus "
                  "whose snapped skeleton vertex has degree 3 and whose three "
                  "arms carry the v117 pattern host/host/added. PASS requires "
                  "all of: (1) at least 300 such sites survive the funnel -- "
                  "below that the run reports the funnel and no area under the "
                  "curve; (2) arm C, v117-measured radii against real nearby "
                  "wrong objects offered at the same cut end, reaches held-out "
                  "area under the curve at least 0.65 by the parameter-free "
                  "Murray score |p-3|, pooled over up to 3 distractors per "
                  "site; (3) arm A, the same construction scored on PROOFREAD "
                  "radii with EXP-084's permuted distractor, reproduces "
                  "EXP-084's 0.675 within +-0.05. Clause 3 is not decoration: "
                  "if it fails, the construction moved the number and no drop "
                  "may be attributed to the v117 radius. Also gated: at most "
                  "20% of otherwise-valid sites lost to the read-box size "
                  "guard. Proofread identity is used only to build the site "
                  "set, to say which offered object is correct, and to keep "
                  "the cell's own tissue out of the distractor pool; no score "
                  "reads a label",
        requires_ran=[],
        inputs=[EDIT_JOIN_V082, CELL_SKELETONS],
        params={"v117_timestamp": 1623399000, "mip": 2, "arm_nm": 1500.0,
                "distractor_max_gap_nm": 2000.0, "n_distractors": 3,
                "n_folds": 5, "seed": 0, "min_sites": 300,
                "bar_auc_v117": 0.65, "exp084_auc": 0.675,
                "control_tolerance": 0.05},
        flags={"network": True, "synthetic_fallback": False,
               "labels_used_only_for_evaluation": True}),
        module="neuronauts.experiments.exp088_conservation_joins",
        note="The follow-up EXP-084 asked for in its own closing line: score "
             "REAL proposed joins, not the mismatch proxy. Two things were "
             "unestablished there and both must hold before the prior is "
             "usable, so they are asked on ONE site set: the joins are the "
             "28,012 located post-v117 human merges of EXP-082, and every "
             "radius is remeasured on v117 fragment geometry, which is all a "
             "grower has. That remeasurement is EXP-082's own stated "
             "verification for its 0.779 where-to-edit prior (radius alone "
             "0.750), so it lives in neuronauts/harness/v117_caliber.py and "
             "both experiments call it rather than each writing its own -- "
             "harness, not metrics, because harness already owns substrate "
             "geometry (geometry.py fetches the level-2 max_dt_nm; "
             "objgeom.radii exposes it as 'local radius') while metrics is "
             "declared as the one home for EVALUATION metrics, which a "
             "caliber is not. "
             "BARS TAKEN FROM THE PRIOR RESULTS, NOT INVENTED. 0.65 is "
             "EXP-084's 0.675 minus the width of one honest step down: the "
             "prior has to survive real decoys and a coarser ruler, and a "
             "number below that is not worth compounding over a tree. The "
             "+-0.05 control is the load-bearing clause. EXP-084 measured "
             "0.675 on ARBITRARY bifurcations of proofread skeletons; this "
             "measures join sites, which are a different and thinner "
             "population (EXP-082: 95% of merges are on axon, 26x enriched in "
             "sub-110-nm cable). If the proofread-radius arm does not land "
             "back on 0.675, the construction is what moved the number and no "
             "drop may be blamed on resolution -- so the run reports that "
             "instead of an attribution it has not earned. A third arm, v117 "
             "radii with the SAME permuted distractor, is reported to split "
             "the difference between the two causes; it adds a decomposition, "
             "not a bar. "
             "The distractor is a real nearby v117 object at the same cut "
             "end, sampled at the same stand-off distance as the true arm it "
             "replaces, with the cell's own tissue excluded -- EXP-083's "
             "lesson that gap, direction and parent must be equalised inside "
             "the pair before a shape score means anything. EXP-082 measured "
             "23+ adjacent decoys per micron at a seam, so 'nearest first' is "
             "the hard version of this test, not the easy one. "
             "REFUSAL IS PREDECLARED: below 300 sites, or above a 20% "
             "read-box skip rate, the run reports the funnel and no area "
             "under the curve. The funnel is not free -- degree 3 at the "
             "snapped vertex plus a verified host/host/added arm pattern will "
             "reject most merges, because an end-to-end continuation join "
             "makes no bifurcation for Murray's law to apply to -- and the "
             "attrition is the thing worth knowing if the bar cannot be "
             "measured. What this does NOT test is the compounding EXP-084 "
             "says is the prior's real value: one branch point stays weak "
             "evidence, and a whole-tree accumulation is a separate "
             "experiment."),

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
