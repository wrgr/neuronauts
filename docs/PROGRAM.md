# The program: staged experiments, ready to iterate

One page. What we believe, what is dead, what runs next, and where the code is.
Every stage has a concept, a falsifiable claim, an entry point and a bar.

## The task, stated once

Start at a cell body — real, from `nucleus_detection_v0`, exhaustive, and
independent of the segmentation, so segmentation error cannot corrupt the seed
set. Grow outward and recover that cell's cable. Stop when it ends.

The shape of the problem, measured (`results/EXP-081/`):

| | |
|---|---|
| cut ends a grower must judge, per cell | median **46** |
| of which are real extension sites | median **1** |
| base rate across 40 cells | **1.6%** (34 of 2,137) |

Precision at that base rate is the binding constraint. A 5% per-tip false-extend
rate produces about 2.3 wrong joins for every right one.

## What is dead, with evidence

Do not restart these. Each was measured on correct data.

| | result |
|---|---|
| **pairwise proximity at the frontier** | best local feature area under the curve 0.630; **0% precision in the top 34** (EXP-081) |
| **SegCLR embeddings as a selector** | 0 of 44 top-1 against geometry's 22, below chance, 94% coverage (EXP-080) |
| **synapse pattern** (density, spacing, polarity) | chance; degraded the combination from 22 to 16 (EXP-080 follow-up) |
| **polarity agreement as continuation evidence** | costs 0.058 area under the curve — the soma-to-axon transition is a *required* polarity flip |
| **abstention by any distance threshold** | **provably impossible**: all 33 already-whole terminal panels have a candidate touching at exactly 32 nm (median = min = max), and so do the true partners. Everything touches (EXP-074 re-run) |

The lesson under all five: **local pairwise evidence reproduces a segmentation
effort that already exists, at worse quality, on one cube.** The contribution has
to be at a level the local view cannot see.

### Two corrections to the record

**Correcting the substrate fix.** Through this session I attributed a large
ranking improvement to fixing voxel identity. A matched control — same panels,
same candidates, same box, only the geometry changing — says otherwise: the
partner's mid-rank goes from 38.0 on centroids to 47.5 corrected, marginally
*worse*, because correct geometry collapses four times as many distractors into
an exact tie. The move from rank 2,265 to about 38 was **scope**, a box rather
than the cube, not identity. The identity defect is real and worth fixing, and
it was not what made the numbers move. EXP-070 never read those centroid clouds
at all; it used level-2 representative coordinates. And the mip-5 cloud, which I
called broken, measures **tightest of the three coarse metrics** — median 224 nm
above true contact against 645 nm for level-2 and 1,247 nm for endpoints.

**Correcting the failure story.** EXP-074's re-run reaches **95.1% recovery** at
cap 200 (bar 0.60, passed) with purity 5.8% and abstention 0.0% (both failed).
Its radius sweep never bound; the cap did. But recovery here is an upper bound —
the panel is a single hop inside a box centred on the answer, ~450 candidates
rather than ~300,000.

## What works, and why it is the template

**Cajal/Murray caliber conservation** (`results/EXP-084/`,
`scripts/test_cajal_conservation.py`). Real branch points obey `r0^3 = r1^3 +
r2^3`: median exponent **3.18** against an ideal 3.0 across **3,781** real
bifurcations. Mismatched branches separate at area under the curve **0.675**,
with zero parameters and no training.

0.675 from one branch point is weak. It is the template anyway, because it has
the three properties the program needs:

1. **It scores a structure.** You cannot evaluate it without a tree.
2. **It compounds.** A cell has many branch points; a wrong join spoils one.
3. **It needs no labels.** It is biophysics, so it cannot overfit our splits.

## The stages

Each stage is a concept with a falsifiable claim. Order is by evidence, not
ambition. `neuronauts/program/` holds the entry points.

### Stage 1 — Conservation over a whole tree  · READY
**Concept.** A wrong join violates conservation laws at the branch it creates,
and a whole assembled cell offers many independent chances to notice.
**Claim.** Summing Cajal evidence over an assembly separates correct from
corrupted assemblies better than any single branch (0.675) does.
**Needs.** Cached skeletons; no training. **Entry:** `stage1_conservation.py`.
**Bar.** Area under the curve above 0.85 over whole assemblies, and a stated
minimum detectable wrong-join size.

### Stage 2 — Joint assignment instead of greedy choice  · READY
**Concept.** An object can continue at most one cut end, and a cut end takes at
most one object. Greedy scoring ignores that; Hungarian assignment enforces it,
with a per-tip dummy column so declining is part of the optimisation rather than
a threshold bolted on.
**Claim.** The same local scores under a global constraint beat greedy precision
at the 1.6% base rate.
**Needs.** `scipy.optimize.linear_sum_assignment`; no training.
**Entry:** `stage2_assignment.py`. **Bar.** Precision above 20% at recall 0.3,
against greedy's 0%.

### Stage 3 — Human proofreading history as supervision  · IN FLIGHT
**Concept.** Humans already solved thousands of these decisions. Where they
edited, and what they joined that geometry ranks poorly, is a label set nobody
here has exploited.
**Claim.** Edit locations are predictable from structure, and human joins
include cases local adjacency cannot propose (skip connections).
**Needs.** `data/external/edit_history/`, ConnectomeBench2, v1822 versus v117.
**Entry:** `stage3_edit_history.py`. **Bar.** Report the fraction of human joins
that bridge non-adjacent fragments — if it is material, adjacency-based
candidate generation is refuted outright.

### Stage 4 — Whole-cell shape as a global check  · IN FLIGHT
**Concept.** A wrong join is detectable from the shape it produces, not from
evidence at the join. EXP-063 detects a frankenmerge in an existing object at
held-out area under the curve **0.958**; it has never been applied to a
*proposed* join.
**Claim.** A whole-cell shape score separates correct assemblies from ones
corrupted by a real foreign fragment.
**Needs.** Cached skeletons, `box_truth`. **Entry:** `stage4_shape.py`.
**Bar.** Area under the curve above 0.9, plus the smallest detectable wrong join.

### Stage 5 — Grammar infilling  · SPECIFIED
**Concept.** The grammar predicts what *should* be at a frontier — a dendrite of
this caliber at this centrifugal order continues, branches, or terminates with
known probabilities. Infilling turns growth from search into prediction, and a
beam search over depth avoids syntactic dead ends.
**Claim.** Expected-continuation from the grammar beats geometric alignment at
finding live sites.
**Needs.** Production rules (`docs/grammar/santiago_grammar_extracted.md`);
`attic/morpho_grammar/tree_grammar_infiller.py` for reference.
**Entry:** `stage5_infill.py`. **Bar.** Beat area under the curve 0.630 at the
1.6% base rate.

### Stage 6 — A learned tree scorer  · SPECIFIED
**Concept.** Stages 1–5 are hand-specified. Once one of them separates
assemblies, train a model on that representation. Training is in scope.
**Claim.** A learned scorer over tree features beats the hand-built combination.
**Needs.** Whatever representation wins above; a root-disjoint split — the
current top-1 of 22 of 66 was selected on the same 66 panels it is reported on,
which is selection bias and must not be repeated.
**Entry:** `stage6_learned.py`. **Bar.** Beat the hand-built score on a held-out
split, reported with the effective sample size.

## Rules for every stage

Five separate confident results dissolved on inspection in one session, every
one a substrate error rather than a wrong hypothesis. So:

1. **Print the audit before the result** — substrate, identity resolution,
   sample size, and the confound this comparison is vulnerable to.
2. **Read objects, never supervoxels** — CloudVolume `agglomerate=True,
   timestamp=V117_TS`, mip 2. `object_clouds_mip5.npz` is supervoxel centroids
   at ~20% coverage: fine at micron scale, wrong for contact.
3. **State the effective sample size.** The abstention estimate swung 0.44 to
   0.64 on one feature with 21 terminals. That is noise, not a result.
4. **Evaluate at the real base rate.** Accuracy on a balanced set is meaningless
   when the operating point is 1.6%.
