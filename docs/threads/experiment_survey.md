# Experiment survey — what has been tried, what is still believed, what to read first

*Written 2026-09-02 from a direct read of every source listed at the bottom of
this document — the registry, every `results/EXP-*/` file, `RESULTS.md`,
`EXPERIMENT_LOG.md`, the 34 scripts in `attic/benchmarks_semi_synthetic/`, the
26 engines in `attic/morpho_grammar/`, every `experiments/*/README.md` and
`docs/threads/*.md`, `docs/consolidation_plan.md`, `docs/pcfg_global_assembly_report.md`,
`docs/tree_assembly_handoff.md`, and `docs/archive/2026-09/STATUS.md`. Every
number below is quoted from one of those files; where I sampled rather than
read exhaustively, that is stated. Nothing here was run — this document only
reads what is already on disk.*

## 0. The one-minute answer

**~80 experiment-shaped things exist in this repo, spanning three eras.** Most
of what makes the tree feel large is one era that is not evidence at all:

| Era | Count | Grade | What it is |
|---|---:|---|---|
| **The registered program** (`neuronauts/experiments/registry.py`, EXP-057–074) | 21 declared | 8 REAL · 3 REAL BUT SUPERSEDED · 10 UNRUN | Real MICrONS data, fail-closed, one substrate, running this week |
| **The pre-registry real-data series** (EXP-051–056, `scripts/benchmark_exp05*.py`) | 6 | 6 REAL (mostly honest negatives) | Real data, no registry, predates the runner |
| **The semi-synthetic generation** (EXP-020–050, `attic/benchmarks_semi_synthetic/`) | 34 scripts | 34 SEMI-SYNTHETIC | Synthetic cuts/frankenmerges on real skeletons — **`EXPERIMENT_LOG.md`'s whole SOTA table is built on this era and carries a superseded banner** |
| **`attic/morpho_grammar/`** engines scored by the above | 26 | 25 NOT EVIDENCE (no checkpoint) | Untrained models scored on synthetic damage |
| **Other threads that are experiments in all but name** (fingerprints, pcfg, tree-DNA, treestitch, dendritic scaffold, tile stitching, CellGNN, grammar/GAT, topology validator, and four incubating/archived threads) | ~14 | 5 REAL · 2 SEMI-SYNTHETIC · 1 misleadingly-quoted · rest incubating/no result | Mixed; see Part 1D |

**The load-bearing bottom line, in one sentence:** on real data, the harness
substrate is real and well-verified, frankenmerge *detection* works
(held-out AUC 0.958), but candidate *proposal* — finding the right partner
fragment to score in the first place — has failed every method tried
(distance, direction, widened object set, cheap structural filters), and nine
of the ten experiments that would test scoring and assembly have never run
because they sit behind that unmet bar.

If you read nothing else, read **Part 4** for the five documents that carry
the current state of belief, in order.

---

## Part 1 — the inventory

### 1A. The registered program (`neuronauts/experiments/registry.py`, series A–F)

This is the live spine: every entry below is declared in the registry with a
predeclared criterion. "Grade" is mine, applying the rubric in the task; the
registry's own `status` (pass/fail/blocked) is a different, narrower judgment
against the criterion as written — I report both because they disagree in
instructive ways (e.g. EXP-072 *fails* its bar but is still real, honest
evidence).

| ID | Title | Question | Registry status | Grade | Date · commit |
|---|---|---|---|---|---|
| EXP-057 | GT overlay and spatial split | What fraction of atoms/synapse mass has unambiguous ground truth? | fail (16.2% vs 30% bar) | **REAL** | 2026-09-01 · `06e8b44a6` |
| EXP-057B | ConnectomeBench2 intake | Can an external corpus lift us past 56 seam positives? | pass | **REAL** | 2026-09-02 · `fa5db41f8` |
| EXP-057C | SegCLR embedding intake | Do public embeddings separate same-cell from different-cell fragments? | not_implemented | **UNRUN** | never run |
| EXP-058 | Baseline ladder | What are the floor and ceiling on this substrate? | pass | **REAL** | 2026-09-02 · `8a148bdba`, corrected `3f8c6bb98` |
| EXP-059 | Metric agreement | Do the metric implementations agree? | pass | **REAL** | 2026-09-02 · `06e069e0b` |
| EXP-060 | Endpoint filter | Which endpoints are real split sites? | fail | **REAL BUT SUPERSEDED** (headline numbers) | 2026-09-02 · `677d3d9bf`, corrected `2cbcdb465` |
| EXP-060B | Object-space atom-pair panel | Does atom-pair reduction recover spanning links? | pass (internal consistency) | **REAL** | 2026-09-02 · `28039e9f3` |
| EXP-061 | Directed cone vs proximity ball | Does a tangent cone beat the proximity ball? | fail | **REAL BUT SUPERSEDED** (enrichment stat) | 2026-09-02 · `fb662e13d`, corrected 2026-09-02 |
| EXP-062 | Real-L2 cuts and seam location | Do real-adjacency cuts beat MST-geometry cuts? | not_implemented | **UNRUN** | never run |
| EXP-063 | Frankenmerge detection | Does polarity/shape flag a false merge? | pass (AUC 0.958) | **REAL** | 2026-09-02 · `81aec2b81` |
| EXP-064 | Fixed-panel scorer bake-off | Which signal separates true continuations? | not_implemented | **UNRUN** | never run |
| EXP-065 | Scorer ablation | What does each feature contribute? | not_implemented | **UNRUN** | never run |
| EXP-066 | Solver bake-off | At fixed scores, which solver wins? | not_implemented | **UNRUN** | never run |
| EXP-067 | Abstention curve | Is there a usable proofreading operating point? | not_implemented | **UNRUN** | never run |
| EXP-068 | Scale and tiling | Does the result hold at 200 µm and under tiling? | not_implemented | **UNRUN** | never run |
| EXP-069 | Attic re-derivation | Does any retired morpho_grammar engine earn its numbers back? | not_implemented | **UNRUN** | never run |
| EXP-070 | Object vs endpoint distance | Was the endpoint representation, not proximity itself, why proposal failed? | pass | **REAL** | 2026-09-02 · `8b509ef12` |
| EXP-071 | Contact adjacency and the connective gap | Are fragments separated by distance, or by objects the population omits? | pass | **REAL BUT SUPERSEDED** (object-count headline) | 2026-09-02 · `867cba04b`, corrected `81aec2b81`, `4a5e4d9b5` |
| EXP-072 | Object-level proposal on the widened substrate | Does proposing over every v117 object reach spanning links at a usable panel? | fail | **REAL** | 2026-09-02 · `81aec2b81` |
| EXP-073 | Constrained chaining | Do cheap structural constraints prune the panel? | **blocked** (prerequisite EXP-072 failed) | **UNRUN** (registered run) / a probe exists, graded separately below | 2026-09-02 · `81aec2b81` |
| EXP-074 | Soma-seeded growth, distance only | Can a soma-seeded grower recover a cell's in-box process? | no result on disk | **UNRUN** | module + spec written 2026-09-02; `results/EXP-074/` is empty |

**Counts: 8 REAL, 3 REAL BUT SUPERSEDED, 10 UNRUN, 21 total.**

#### Notes on the ones that need more than one line

**EXP-057 — failed, and the failure is the finding.** 291,931 of 1,802,996
synapse sides (16.2%) sit on single-lineage atoms with a proofread owner,
against a 30% bar; every atom got a label row, so this is a real ground-truth
shortage, not a coverage bug. Only 56 atoms in the whole 279,075-atom
population are both mixed-lineage and proofread-owned — the "seam positive"
count that gates cut/detection work. Source: `results/EXP-057/evaluation.md`.

**EXP-058 — proximity is not merely weak, it is indistinguishable from
random.** Over 1,297 tier-≥10 proofread-owned atoms (947 owners, 492 true
same-owner pairs), every proximity threshold from 1–5 µm collapses the
population into one giant cluster: pair precision 0.0006, matching a random
baseline at the same edge count to four decimal places. The oracle rung hits
ARI 1.0. An earlier claim in this file — "the panel happens to contain all 492
true pairs" — was **wrong** and is corrected in the same evaluation.md: the
oracle's `pair_tp=492` came from transitive closure over a collapsed cluster,
not from panel coverage; EXP-060 later measured the panel directly and found
17.5%. Source: `results/EXP-058/evaluation.md`.

**EXP-060 → CORRECTION.md → EXP-060B: three drafts of the same finding.**
EXP-060 first reported 17.5% panel recall against a median true-partner gap of
6.5 µm and concluded "geometry cannot propose candidates." `CORRECTION.md`
(2026-09-02) found two real errors — the denominator should be minimum-spanning-tree
links (350), not all same-owner pairs (492), which raises recall to 24.6%;
and the *nearest* partner's median gap is 1.3 µm, not 6.5 µm, because the
6.5 µm figure was dominated by distal same-cell pairs no proposer should be
asked to find. It provisionally **withdrew** the "cannot propose" conclusion.
EXP-060B then measured the full recall-vs-panel-size curve rather than a
single point and found the correction's own ~65% prediction was right **only
at an unusable panel size** (median 3,870 candidates for 64.6% recall at
tier ≥10); at a panel size a scorer could use (≤20–100), recall is 12–23%,
close to the original, differently-computed number. **So the corrected
conclusion reinstates the original one**, on the right numbers this time. A
second EXP-060B addendum caught a mislabeled substrate file (`k1.npz` was the
1–4-synapse shard, not the full-population union) and re-ran the full-tier
comparison. Sources: `results/EXP-060/CORRECTION.md`, `results/EXP-060/evaluation.md`,
`results/EXP-060B/evaluation.md`.

**EXP-061 — the tangent is real signal, just not sharp enough, and the
enrichment number was overstated 2×.** No cone configuration reached 70%
recall at panel ≤20; the best (50 µm, 45°) reaches 40.2% at a panel of ~2,174
objects (corrected from an object/endpoint unit-conflation that first read
42,160). The angular enrichment over chance was first reported as 3–6×; the
2026-09-02 QA pass found the "chance" comparison used a single-direction null
against a best-of-two-directions statistic and recomputed the true null with
20 random-tangent seeds — the real enrichment is **2–3×**. The pass/fail
verdict is unchanged either way. Sources: `results/EXP-061/evaluation.md`,
`docs/threads/qa_pass_2026-09-02.md`.

**EXP-063 — held-out AUC 0.958, and I found a discrepancy I could not
resolve.** All-feature GBDT reaches held-out AUC 0.958 against a 0.875 bar;
polarity alone (5 free columns of synapse counts) reaches 0.914, beating the
published shape detector's 0.875; size alone is at or near chance. **The
"size only" number disagrees between the two files on disk**: the current
`results/EXP-063/result.json` (the later of two logged runs, timestamped
2026-09-02T17:08:53 in `RESULTS.md`) records `size_only_val_auc_strict =
0.653845`, but `results/EXP-063/evaluation.md` (written after the *first*
logged run) states "size only (log n synapses) | 0.483 | 0.427" and reads
this as "size carries nothing... chance." **I believe the `result.json` value
(0.654) is the more current one** because `RESULTS.md` records it as the
later run against the identical criterion, but `evaluation.md` was not
regenerated after that run, so the file a reader opens first states a number
one full run out of date. This does not change the pass/fail verdict (0.654
is still far below the 0.958 stack and below the 0.875 bar's competitor), but
it does mean "size carries nothing" is not quite what the current numbers
show — 0.65 is above chance. Neither file explains the discrepancy; I have
not resolved it and am reporting it rather than guessing at a cause.

**EXP-070 — the metric was wrong, but that is not why proximity failed.**
Verified id-by-id that endpoints are a strict subset of L2 nodes (0 violations
across 15,235 + 492 pairs where object gap exceeds endpoint gap — a single
violation would have meant the index was wrong). Reachability at 5 µm moves
from 47.4% → 55.5% (all pairs, tier ≥10) and 64.9% → 75.7% (MST links, tier
≥10) under the tighter object metric — real, but still short of the 90% bar,
and the panel-size problem is untouched. The QA pass caught a denominator bug
in the same experiment (a first pass excluded pairs with no endpoint row from
the endpoint denominator, flattering the endpoint column by ~4 points on the
full population) and it is fixed in the reported numbers above. Source:
`results/EXP-070/evaluation.md`.

**EXP-071 → CORRECTION.md, twice: this is the sharpest reversal in the
record.** The original run reported "2,147 objects, 100% absent from the
population" as the connective material joining a cell's fragments. Hours
later, the full-cube enumeration recovered only 8.6% of those 2,147 objects —
looked like an enumeration bug, was not: EXP-071 fetched cells' level-2
graphs at the **200 µm** box the geometry fetch used, but the enumeration only
covers the **100 µm** harness cube, so 94% of the "missing" material was
never in scope to begin with. Rescoped to the 146 objects actually inside the
100 µm cube, the enumeration recovers 141 (96.6%) — the enumeration was fine.
**Separately, a peer review found the "100% absent" clause could not fail**:
`objgeom_kall` by construction only knows population-atom nodes, so a node it
does not recognize cannot resolve to a population atom — an identity, not a
measurement. It was withdrawn and replaced with a real, falsifiable test: of
230 distinct bridging nodes on ≤3-hop nearest-sibling paths, 230 (100%)
resolve to a real v117 object that existed at v117 (not a later-edit
artifact) and none is in the population — a result that could have come out
differently and did not. **What survives unchanged**: nearest-sibling
distance median 3 hops (~1.6 µm / 1,604 nm), zero direct atom-to-atom L2
contacts (structural, not a measurement — two v117 atoms sharing a level-2
edge would be one atom), and the qualitative conclusion that the
synapse-anchored population omits ordinary connective cable. Sources:
`results/EXP-071/CORRECTION.md`, `results/EXP-071/evaluation.md`.

**EXP-072 — fails its bar, and the failure mode is itself informative.**
Widened chained recall clears the 50% floor (63.6%), but the *control*
(population-only, no widening) scores **higher** (71.1%), so the required
"beat control by 20 points" clause fails by 27 points in the wrong direction,
and the median reachable-labelled-atoms clause (bar ≤50) misses by 30×
(1,586). Widening the object set to include synapse-free material made
chained recall *worse* because dense neuropil connects nearly everything to
everything within a few hops — the same collapse pattern as EXP-058, now at
the object level. A same-day probe (`probe_40um_mip2_dust_floor.md`) isolated
that this is not a debris artifact: removing 87% of objects by a physical
dust floor moves precision by nothing (0.09% at every floor tested). Source:
`results/EXP-072/result.json`, `results/EXP-072/probe_40um_mip2_dust_floor.md`.

**EXP-073 — blocked as registered; its probe is a real negative result.**
The registered run is blocked because its prerequisite (EXP-072) did not
pass. A same-day, non-canonical 40 µm probe (`results/EXP-073/probe_40um_mip2.md`)
tested whether cheap object-level shape filters (elongation, attachment
angle) prune the chained panel enough to use: no setting clears EXP-072's
bar, and the filters prune true links about three times as fast as they
improve precision (recall 89.8% → 30.2% while precision only doubles,
0.09% → 0.18%). The "through-angle" clause — the actual structural-grammar
rule — adds nothing measurable on top of the elongation filter alone. This
falsifies the *cheap, object-level* form of the structural hypothesis; it
does not test skeleton-level constraints (tangent/caliber continuity), which
remain untested.

**EXP-074 — declared and specified, not yet run.** `docs/threads/exp074_spec.md`
derives every bar from a real 103-cell census (67 cells that need joining, 36
already whole) and `neuronauts/experiments/exp074_seeded_growth.py` exists,
but `results/EXP-074/` holds no files and `RESULTS.md` has no row for it — I
am grading this UNRUN on the direct evidence of an empty results directory,
not inferring anything about why.

---

### 1B. The pre-registry real-data series (EXP-051–056)

These predate `neuronauts/experiments/registry.py` and the runner; they are
plain scripts (`scripts/benchmark_exp05{1..6}*.py`) with hand-written
evaluation files in `results/`. `docs/consolidation_plan.md` §0.2 calls this
"the real, defensible" replacement for the semi-synthetic series below, and
`EXPERIMENT_LOG.md`'s own superseded banner names it as what replaces §1–3.
**All six are graded REAL** — every one ran on real MICrONS synapses/skeletons
with target lineage withheld from inference, fail-closed on missing
prerequisites, and reported honest (mostly negative) numbers rather than
flattering ones.

| ID | Title | What it found | Grade |
|---|---|---|---|
| EXP-051 | Real dense soma-seeded grammar, 30 µm box | Box held only 1 true merge pair among 9,333 roots; ARI ≈ 0, merge P/R 0/0. A valid negative, not a fair positive test — underpowered box, not a method failure. | REAL |
| EXP-052 | Proofread-soma-anchored 30 µm run | 14 true fragment-merge pairs, 116 mixed-lineage roots. No global score threshold separates true continuations from confusers: at recall 0.929, precision 0.000026 (one cluster absorbs 997/1,023 roots). | REAL |
| EXP-053A | Checkpoint bake-off | No existing checkpoint (raw skeleton, +GAT, legacy, root-neighborhood) separates real pairs from confusers; every non-collapsing threshold recovers zero real merges. | REAL |
| EXP-053B | Real-L2 candidate-panel recall | Prerequisite coverage gate failed: only 284/1,023 roots (27.8%) had ≥2 L2 coordinates; only 1/14 true pairs had geometry on both sides, and it was not proposed at any radius/cone. Explicitly **not** evidence that L2 geometry cannot work — evidence the retrieval path was inadequate. | REAL |
| EXP-054 | Fixed-panel scorer bake-off | Prerequisite failed (0 recall from EXP-053B's panel); no scorer metrics computed, by design. | REAL (fail-closed) |
| EXP-055 | Conservative soma-seeded forest | Prerequisite failed (cascaded from EXP-054); no assembly run, by design. | REAL (fail-closed) |
| EXP-056 | Real-root atomization | On 116 real mixed-lineage roots, no single global edge-length threshold rule meets both bars (90% same-lineage recall AND 50% cross-lineage split recall). Falsifies the single-threshold atomizer. | REAL |

---

### 1C. The semi-synthetic generation (EXP-020–050) and the engines it scored

This is most of the "80% of the files are old" the user is reacting to, and
it is the single most important thing to label correctly.

**What it is.** `attic/benchmarks_semi_synthetic/` holds 34 scripts (moved
there 2026-09-01, `docs/consolidation_plan.md` §8b execution log, commit
row "2 (partial)"). I did not take the consolidation plan's claim on faith —
I sample-checked five directly and grepped all 34:

```
grep -l "treestitch.worldbuild" attic/benchmarks_semi_synthetic/*.py | wc -l
→ 32
```

32 of the 34 import `treestitch.worldbuild.frankenmerge_adjacent` directly
and apply it to real proofread skeletons at 45% (confirmed by reading
`benchmark_exp021_3d.py`, `benchmark_exp035_restored_dual_engine.py` — the
"restored SOTA" `EXPERIMENT_LOG.md` quotes — and `benchmark_exp048_grand_unified_engine.py`
line by line: each calls `frankenmerge_adjacent(pieces_rec, 0.45, rng,
radius_nm=9000–9500)`). The other two generate synthetic data by different
means, also confirmed by reading them: `benchmark_exp050_interneuron_stratified.py`
builds entire neurons from scratch (`generate_pyramidal_skeleton`,
`generate_basket_skeleton`, `generate_martinotti_skeleton`,
`generate_vip_bipolar_skeleton` — random walks, not fetched data), and
`benchmark_exp049_dense_subvolume.py` calls `generate_dense_subvolume_fallback`
unconditionally. **Verdict: SEMI-SYNTHETIC, confirmed on a sample of 5 read in
full plus a grep across all 34 — not merely repeated from the plan.**

The models these scripts score are the 26 engines in `attic/morpho_grammar/`.
The claim that they load no trained checkpoint is likewise independently
checked, not repeated: `grep -L "torch.load\|\.pt'\|\.pt\"" attic/morpho_grammar/*.py`
(excluding `__init__.py`) returns **25 of 26** files with zero checkpoint
reference at all — stricter than the earlier claim of "15 of 26 draw random
numbers at runtime," and consistent with it. **Verdict: NOT EVIDENCE — no
trained checkpoint loaded, scored against synthetically damaged data.**

**What this means for the numbers people might still cite.** `EXPERIMENT_LOG.md`
(97 lines, superseded banner added 2026-09-01, still on disk) is the home of
the "Merge Precision 0.70 / path_P 0.84 / 99.1% synapse precision" table and
the head-to-head comparison against FFN, Janelia Multicut, DeepMulticut and
FlyWire. Every one of those rows is this era: untrained engines scored on
45%-synthetic damage. The banner at the top of the file says so in the file's
own words and names the specific scripts. **I did not find any daylight
between what the banner claims and what the code shows** — this is the one
case in the whole survey where the self-correction is fully corroborated by
independent inspection.

---

### 1D. Other threads that are experiments in all but name

These live in `experiments/*/README.md` or `docs/threads/*.md` rather than
the registry, but each makes a testable claim on real or synthetic data.

| Thread | Claim | Grade | Basis |
|---|---|---|---|
| **Cut-face fingerprints** (`experiments/fingerprints/`) | A cross-section image hash re-identifies the true partner of a severed neurite; combiner + abstention reaches precision 1.0 at 11% coverage (8/73 held-out sites per `docs/tree_assembly_handoff.md` row 5) | **REAL**, with a caveat: the thread's own README reports a different set of numbers — "smoke runs, single ~4–8 µm box," patch top-1 0.716→0.101 as the gap widens 40→640 nm, ~117–121 candidates per row. Ground truth is free (proofread segmentation gives both cut faces the same id) and evaluated on real v117 split sites, not synthetic ones. I read the thread's own README in full but did not open the specific evaluation output backing the "8/73 sites" figure cited secondhand in the handoff doc, so that exact count is one hop removed from a file I opened myself. |
| **PCFG cross-region holdout** (`experiments/pcfg/HOLDOUT_RESULTS.md`) | A learned grammar reassembles synthetically fragmented real neurons across a spatially disjoint region: cross-region AUC 0.816 [0.754, 0.874]; recovers 60% of true cross-region merges vs 0% doing nothing | **SEMI-SYNTHETIC** — real skeletons and synapses (three disjoint real 60 µm regions), but the "true" break points are synthetically introduced at chosen weak points, not adjudicated proofreading edits — the same pattern flagged for EXP-020–050. It is methodologically more careful than that era (proper region-disjoint split, an explicitly identified and removed distance confound), which is worth crediting, but the ground truth is still constructed, not observed. |
| **PCFG half-partition bigram grammar** (`experiments/pcfg/README.md`) | A cheap bigram grammar over synapse half-partitions predicts merges | **UNRUN / UNCLEAR** — the README's own "expected output" table shows placeholder values (`CV AUC = 0.XX`); I found no completed result file for this specific sub-experiment (distinct from `HOLDOUT_RESULTS.md`, which is a different, completed run). |
| **Tree-DNA half-skeleton identity** (`scripts/half_split_ablation.py`, numbers in `docs/archive/2026-09/STATUS.md`) | A trained SkeletonGNN separates same-cell from different-cell half-skeletons within a cell type: trained AUC 0.829 vs a 0.75 bar | **REAL**, with a caveat worth keeping attached to the number: random-init AUC is already 0.768, so the *trained* lift is +0.061, not the full 0.829 — both clear a 0.475–0.488 spatial-baseline/chance control, and no collapse was observed. **The same method fails at quarter-skeleton granularity** (0.599→0.687 or 0.740→0.725, sometimes training *hurts*) — the result is scale-specific and STATUS.md says so explicitly. |
| **treestitch global partition** | Out-of-sample ARI 0.752, merge precision 0.951, merge recall 0.865, "leak-fixed" | **REAL**, sourced from the top-level `README.md`'s own results table (verified directly — `grep -n "0.752" README.md`); I did not additionally open the underlying treestitch run artifacts, so this is one level of indirection from the rawest evidence. |
| **Dendritic scaffold** (`docs/tree_assembly_handoff.md`) | The dendrite/soma side of the graph is certifiable from nucleus lineage alone: 59.6% of synapses at 99.8% mass purity (unbiased dual-side census, 200 µm) | **REAL** — verified directly in the handoff doc, which itself warns not to conflate this number with a *different*, higher (79–99.8%) figure from a biased soma-seeded substrate quoted elsewhere; I kept the two separate as the source insists. |
| **Tile stitching** (`docs/tree_assembly_handoff.md`) | Two-level tiling improves cross-tile assembly at near-zero precision cost: ΔARI +0.10 to +0.11 over 3 independent 200 µm runs; right-sized (100 µm) tiles help, larger (300 µm) tiles do not | **REAL** — verified directly in the handoff doc. |
| **Proximity attribution for axon fragments** | Falsified: nearest-fragment proximity carries no identity signal for axons | **REAL, negative** — 0/32 doubly-adjudicated links at every threshold swept; 9/1,063 axon fragments have their true neuron among in-box soma anchors at all. Verified directly in `docs/tree_assembly_handoff.md`. |
| **CellGNN** (`neuronauts/cell_graph.py`, box-local pipeline) | Held-out test F1 ≈ 0.272, "the pipeline that runs by default" | **REAL**, but the thread page itself calls the ceiling structural (a 30 µm box cannot hold a neuron larger than the box) — a real number describing a known architectural limit, not a candidate for improvement in place. |
| **Grammar / SharedGrammarModel merge scoring** (`docs/threads/grammar.md`) | "Pairwise merge accuracy is strong (~85–87%)" | **Misleadingly quoted, not evidence as stated** — this figure traces to `experiments/pcfg/README.md`'s reference line ("Neuronauts PathEncoder merge acc = 0.856, 85 boxes"), an earlier in-sample/CV number on a curated candidate panel. It is **directly contradicted in spirit** by the harder, later, real test of essentially the same task: EXP-053A found *no checkpoint* separates real continuation pairs from dense confusers on real data, and EXP-058 found proximity-based candidate scoring collapses to pair precision 0.0006. The thread page carries no caveat connecting the two. See Part 4's "most likely to mislead" note. |
| **Topology / atomicity validator** (`neuronauts/topology_model.py`) | Flags clusters formed by merging two distinct roots | **No real-data result recorded** — thread page status is "optional... smoke only," no checkpoint tracked. |
| **root_neighborhood, soma_graph, low_res_segmentation** (experiment threads) | Various (better training-cache seeding; global soma-graph GAT; coarse-resolution segmentation) | **NOT EVIDENCE / no result yet** — root_neighborhood is "incubating," a cache-building strategy with no comparison result; soma_graph's own README describes only a synthetic smoke test (`smoke_test.py`, synthetic soma graph); low_res_segmentation's README states "Runs on synthetic connectomes," with a real-CAVE scaffold only "tested," not scored, against a stated graduation bar (match CellGNN F1 0.272) it has not yet cleared. |
| **Berlin proofreading-language grammar** (external prior work, cited in `docs/consolidation_plan.md` §6.3a) | A first-order Markov model over UI action tokens separates expert from proto-expert proofreaders, LOO AUC 0.95 | **REAL, but not this project's data** — explicitly flagged in the plan as "prior work, verified; not ours to extend without data access," and a different alphabet (UI actions) from anything scored above. Included for completeness since the consolidation plan places it alongside this project's grammars and a reader could conflate the three. |

---

## Part 3 — what is still load-bearing

Cross-checked against `docs/consolidation_plan.md` §1.5 ("What has actually
held up on real data"), which is from an earlier session (2026-09-01) and is
now stale on at least two rows — flagged below where the plan and a later
correction disagree, with the correction winning.

1. **Real L2 adjacency, 100% id resolution, endpoints a verified subset of
   nodes.** *Established by:* `scripts/build_object_geometry.py`'s build-time
   gates, re-verified independently by EXP-070 (0 violations across 15,727
   pairs). *Licenses:* any downstream code may treat the object/endpoint
   geometry as internally consistent; a violation there would mean a real
   bug, not noise.
2. **Frankenmerge detection is solved on this substrate; cutting is not.**
   *Established by:* EXP-063, held-out AUC 0.958, polarity alone 0.914 beats
   the published shape baseline (0.875). *Licenses:* treating "is this atom a
   false merge" as answered for the purpose of downstream design; does
   **not** license any claim about *where* to cut it (Bar 3 / EXP-062 is
   unrun).
3. **Proximity (distance or direction) cannot generate a usable candidate
   panel on this substrate, at any tier, at any resolution.** *Established
   by:* EXP-058 (rank), EXP-060/060B (propose by ball), EXP-061 (propose by
   cone), EXP-070 (propose by object metric instead of endpoint metric — same
   verdict), EXP-072 (propose over the widened object set — same verdict, and
   *worse* by the chained metric), EXP-073's probe (cheap structural filters
   don't fix it either). *Licenses:* not spending further effort on
   distance-only or direction-only candidate generation without a new
   ingredient (identity/embedding retrieval, or skeleton-level structural
   constraints, both still untested).
4. **The candidate-generation ceiling is real, not a bug in one filter.**
   *Established by:* EXP-070's object metric puts the uncapped MST-recall
   ceiling at 75.7% (tier ≥10) / 56.8% (full population) at 5 µm — below any
   90% bar, however the panel is built. *Licenses:* treating a 90%-recall
   proposal bar on this substrate as unmeetable without a non-geometric
   channel, per the consolidation plan's own protocol rule (§6.2).
5. **The population omits ordinary connective cable, not debris.**
   *Established by:* EXP-071 (nearest-sibling median 3 hops / ~1.6 µm on 40
   held-out cells; 230/230 sampled bridging nodes resolve to real,
   pre-existing v117 objects, none in the population), corroborated by
   EXP-072's dust-floor probe (removing 87% of objects by size moves
   precision by 0.00 percentage points). *Licenses:* treating "widen the
   object population" as necessary; **does not** license "widening is
   sufficient" — EXP-072 shows it is not, and makes chained recall worse.
6. **The dendritic scaffold is certifiable from nucleus lineage alone.**
   *Established by:* `docs/tree_assembly_handoff.md` row 3, unbiased 200 µm
   dual-side census: 59.6% of synapses at 99.8% mass purity. *Licenses:*
   treating the dendrite/soma side of assembly as solved by one exact lookup,
   with no learning required; the 79–99.8% purity figure quoted elsewhere is
   from a *different*, biased substrate and should not be substituted for
   this one.
7. **Right-sized tiling helps; enlarging tiles does not.** *Established by:*
   `docs/tree_assembly_handoff.md` row 1–2, 3 independent 200 µm runs, ΔARI
   +0.10 to +0.11 at 100 µm tiles, ΔARI +0.003 (dead) at 300 µm tiles.
   *Licenses:* choosing ~100 µm as the tile size for any future tiled
   assembly, not choosing "bigger."
8. **The EXP-061 angular enrichment is real but was overstated 2×; use the
   corrected 2–3×, not the original 3–6×.** *Where the plan and a correction
   disagree, the correction wins.* `docs/consolidation_plan.md` does not
   carry this table so there is no direct conflict there, but any other doc
   (or the reader's memory) quoting "3–6×" is quoting a retracted number —
   see `docs/threads/qa_pass_2026-09-02.md`.
9. **The EXP-071 connective-object count is 146 in-cube, not 2,147.** *The
   plan predates this correction entirely (written 2026-09-01, before
   EXP-071 ran on 2026-09-02); `results/EXP-071/CORRECTION.md` is the only
   authority and it wins by default.* The 2,147 figure describes a 200 µm box
   scope EXP-071's own bar did not intend to measure.
10. **Do not cite "grammar merge accuracy ~85–87%" as a real-data
    candidate-scoring result.** No correction file says so explicitly — this
    is my own cross-check, not a recorded reversal — but the number's origin
    (an in-sample/CV figure from `experiments/pcfg/README.md`) and the
    harder, later, real tests that address essentially the same question
    (EXP-053A: no checkpoint separates real pairs from confusers; EXP-058:
    proximity-based scoring at pair precision 0.0006) point the same
    direction hard enough that treating the 85–87% figure as current
    evidence would mislead. See Part 4.

---

## Part 4 — a reading order

**If you are new, read these five, in this order:**

1. **`docs/consolidation_plan.md`** — the map. §0–§1 explain why the repo
   looks the way it does (three architectural regimes, four metric
   implementations, the EXP-020–050 provenance problem); §6 is the program
   this survey's Part 1A tracks. Fifteen minutes buys you the whole repo's
   shape.
2. **`docs/threads/feasibility_2026-09-02.md`** — the honest current verdict:
   assisted proofreading is feasible today (detection 0.96, a proposer that
   hands a human 50–100 candidates), autonomous global assembly is not, and
   the blocker is candidate generation, not scoring or assembly. Read the
   "Superseded in part, same day, by EXP-071" box at the top first — it is
   the document correcting itself in real time, which is instructive on its
   own.
3. **`results/EXP-060/CORRECTION.md`** — the clearest single lesson in the
   repo: two real errors (wrong denominator, wrong units) that *looked* like
   a finding because the flawed measurement told a clean story. Read it
   alongside `results/EXP-060B/evaluation.md`, which shows the corrected
   conclusion arrived back near the original one, on the right numbers.
4. **`results/EXP-063/evaluation.md`** — the one clearly positive real-data
   result on the current substrate (held-out AUC 0.958), and a template for
   what a well-documented experiment file looks like here (explicit criterion
   amendment before the run, explicit statement of what is deliberately not
   measured).
5. **`docs/tree_assembly_handoff.md`** — the load-bearing findings that
   predate and sit alongside the registry program: the dendritic scaffold,
   tile stitching, and the falsified proximity-attribution result, all with
   real numbers and explicit caveats about which of two similarly-named
   figures to trust.

**Safe to ignore unless you are re-deriving a specific claim:**

- `EXPERIMENT_LOG.md` and everything in `attic/` — read only if you need to
  know exactly what a specific EXP-020–050 number claimed, never to learn
  what is currently true.
- `docs/paper/`, `docs/latex/`, `docs/*_slides.*` — inherit the same
  superseded numbers per `docs/consolidation_plan.md` §4.4.
- `docs/pcfg_global_assembly_report.md` and `docs/grammar_literature_directions.md`
  — real, careful design documents, but explicitly "no code changes... nothing
  here has been run." Useful for *why* a grammar is proposed, not for *what
  has been shown*.
- Any individual `results/EXP-072/probe_*.md` or `results/EXP-073/probe_*.md`
  file on its own — each is a same-day pilot at a different resolution/box
  size than the canonical run; read the parent experiment's `result.json`
  first and the probes only if you need the resolution-sensitivity detail.

---

## Part 5 — corrections and reversals, as a timeline

| Date | What was claimed | What was wrong | How it was caught |
|---|---|---|---|
| (era) EXP-020–050, `EXPERIMENT_LOG.md` §1–3 | Merge precision 0.70, path_P 0.84, 99.1% synapse precision, compared head-to-head against published FFN/Multicut/DeepMulticut/FlyWire numbers | Built on real skeletons that were **synthetically cut** and scored by **untrained** morpho_grammar engines (no checkpoint) | Direct inspection of the scripts (2026-09-01), independently re-confirmed here on a sample of 5 scripts read in full + a grep across all 34 |
| 2026-09-02 | EXP-058: "the panel happens to contain all 492 true pairs" | The oracle's `pair_tp=492` came from transitive closure over a collapsed single cluster, not from panel membership | EXP-060 measured panel membership directly, one experiment later, and found 17.5% |
| 2026-09-02 | EXP-060: 17.5% panel recall, median true-partner gap 6.5 µm, "geometry cannot propose candidates" | Wrong denominator (all-pairs, not MST spanning links) and a gap distribution dominated by distal pairs no proposer needs to find; nearest-partner median is actually 1.3 µm | A direct question ("shouldn't proximity depend on which atoms are filtered out?") prompted a recheck; `results/EXP-060/CORRECTION.md` |
| 2026-09-02 | EXP-061: cone panel sizes reported in endpoint units (up to 42,640) | Panel size should be counted in **object** units — one partner atom contributing 40 endpoints is one candidate, not 40; inflation up to 19.6× | Same correction pass, "Error 3" in `CORRECTION.md` |
| 2026-09-02 | CORRECTION.md's own "revised bottom line": geometry *can* propose, ~65% recall achievable | True only at an unusable panel size (median 3,870 objects); at ≤20–100 objects, recall is 12–23% | EXP-060B measured the full recall-vs-panel-size curve instead of a single point; its own addendum calls this "premature" |
| 2026-09-02 | EXP-060B's "tier ≥1 (complete substrate)" comparison | `k1.npz` was the 1–4-synapse incremental shard, not the full-population union; every "tier ≥1" number in the file was wrong | A direct question about candidate synapse counts (filtering to `n_synapses >= 5` returned zero, only possible if every atom had 1–4) |
| 2026-09-02 | EXP-061: "3–6× enrichment over chance" | Chance was computed for a single random direction against a best-of-two-directions statistic; the true empirical null (20 random-tangent seeds) is about half the claimed enrichment | `docs/threads/qa_pass_2026-09-02.md`, a requested proximity-code audit |
| 2026-09-02 | EXP-070 (first pass): reach fractions over pairs with a finite gap | Pairs where an atom has no endpoint row dropped from the endpoint denominator while staying in the object one, flattering the endpoint column by ~4 points | Same QA pass |
| 2026-09-02 | EXP-071: "2,147 objects, 100% absent from the population" | The enumeration covers a 100 µm cube; EXP-071 fetched a 200 µm box, so 94% of the counted material was out of scope by construction | The full-cube enumeration recovering only 8.6% of the 2,147 looked like an enumeration failure; checking the box sizes found the real cause |
| 2026-09-02 | EXP-071's bar clause 3, "≥80% of connective objects absent from the population" | Could not fail: `objgeom_kall` only knows population-atom nodes by construction, so any unknown node is trivially "absent" — an identity, not a measurement | A peer review of the criterion itself |
| 2026-09-02 | (replacing the withdrawn clause) | — | Measured instead: 230/230 sampled bridging nodes resolve to real, pre-existing v117 objects, none in the population — a test that could have failed and did not |
| 2026-09-02 | EXP-063: `evaluation.md` states size-only AUC 0.483 ("chance") | The current `result.json` (a later logged run) records 0.654 for the same field; `evaluation.md` was not regenerated after the later run | Found in this survey by comparing the two files directly; **not resolved** — flagged, not explained |

**The recurring pattern, stated once:** almost every correction in this
project follows the same shape — a real computation on a subtly wrong
quantity (wrong denominator, wrong units, a clique instead of a nearest
neighbor, a scope mismatch between two box sizes, a bar clause that could not
fail) produced a number that *looked* like a clean finding, and the error was
caught only when someone asked a pointed follow-up question rather than
accepting that the number told a coherent story. `results/EXP-060/CORRECTION.md`
names this directly: "Both were correct computations of the wrong quantity...
[a] check... costs one line and was not run because the all-pairs number
already told a clean story." This matches `CLAUDE.md`'s own operating rule
("assume the bug is yours") applied to the project's own numbers, not just to
external systems.

---

## Appendix — method and open items

**What I read in full:** `neuronauts/experiments/registry.py` (664 lines),
`docs/consolidation_plan.md` (663 lines), every file under `results/EXP-*/`
(all `result.json`, `evaluation.md`, `CORRECTION.md`, and probe files),
`results/RESULTS.md`, `EXPERIMENT_LOG.md`, `results/exp05{1..6}_evaluation.md`,
`experiments/README.md`, `experiments/fingerprints/README.md`,
`experiments/pcfg/README.md`, `experiments/pcfg/HOLDOUT_RESULTS.md`,
`experiments/{root_neighborhood,soma_graph,low_res_segmentation,minnie_column}/README.md`,
`docs/threads/{error_correction,cell_assignment,tree_dna,topology,grammar,
qa_pass_2026-09-02,feasibility_2026-09-02,exp074_spec}.md`,
`docs/tree_assembly_handoff.md`, `docs/archive/2026-09/STATUS.md` (excerpts),
`docs/pcfg_global_assembly_report.md` (excerpts), and the top-level `README.md`.

**What I sampled rather than read exhaustively, and how many:** the 34
`attic/benchmarks_semi_synthetic/*.py` scripts — read 5 in full
(`benchmark_exp021_3d.py`, `benchmark_exp035_restored_dual_engine.py`,
`benchmark_exp048_grand_unified_engine.py`, `benchmark_exp049_dense_subvolume.py`,
`benchmark_exp050_interneuron_stratified.py`), grepped all 34 for the
`treestitch.worldbuild` import (32/34 hit); the 26 `attic/morpho_grammar/*.py`
engines — grepped all 26 for checkpoint-loading code (25/26 have none), did
not read each file's logic in full.

**What I could not verify and did not guess at:**
- The exact "8/73 held-out sites, precision 1.0 at 11% coverage" figure for
  the cut-face combiner — cited in `docs/tree_assembly_handoff.md` but I did
  not locate and open the specific evaluation output that produced it inside
  `experiments/fingerprints/`.
- Why `results/EXP-063/evaluation.md` and `results/EXP-063/result.json`
  disagree on the size-only AUC (0.483 vs 0.654) — reported as a discrepancy,
  not resolved.
- Whether EXP-074, if run today, would find EXP-072 (its `requires_ran`
  prerequisite) sufficiently "run" despite EXP-072 having *failed* its bar —
  I did not read `neuronauts/experiments/_runner.py`'s prerequisite-checking
  logic closely enough to state this with confidence, so I graded EXP-074
  UNRUN on the direct evidence (empty results directory) rather than
  predicting what a future run would do.

**Total experiment-shaped artifacts surveyed:** 21 (registry) + 6 (EXP-051–056)
+ 34 (semi-synthetic scripts, graded as one bloc) + ~14 (other threads) ≈ 75,
consistent with the user's sense that the tree is large — the registry's 21
plus the six pre-registry real runs (27 total) are the part worth tracking
closely; the rest is provenance, not a live decision surface.
