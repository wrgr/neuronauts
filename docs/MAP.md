# MAP — what is alive in this repo, and what is only history

*Written 2026-09-02. This is the orientation page: one line per thing, with its
honest status. It does not re-argue any result — for that, read
[`docs/threads/experiment_survey.md`](threads/experiment_survey.md), which
grades every experiment by evidence quality.*

505 tracked Python files sit in `neuronauts/` (128), `attic/` (130),
`tests/` (110), `scripts/` (60), `experiments/` (57) and `treestitch/` (20)
— counted 2026-09-02, and the tree is under active development, so treat these
as an order of magnitude, not a checksum.
Most of that is provenance rather than working code: real work a later,
better-controlled run replaced, plus one large generation of benchmarks whose
test worlds were synthetic. Nothing has been deleted — everything below says
where a thing is and why it is there.

## The tree, in ten seconds

| Directory | Rule | Count |
|---|---|---:|
| `neuronauts/` | The package. `harness/`, `metrics/`, `report/`, `data/`, `meshing/`, `experiments/` are the live spine. | 128 |
| `scripts/` | **Only** what builds or refreshes current data, the probes, `train.py`, and the seven `benchmark_exp05*` scripts a resolver in `neuronauts/report/` pins here. Has its own [`README.md`](../scripts/README.md). | 56 |
| `experiments/` | **Only** live research threads: `pcfg/`, `fingerprints/`, `minnie_column/`, `soma_graph/`, `root_neighborhood/`. | 57 |
| `treestitch/` | The global-partition package. Real results; nothing depends on it. | 20 |
| `tests/` | The default suite. | 110 |
| `attic/` | **The archive.** Everything superseded, in ten subdirectories, each with a README saying what it was and what replaced it. Excluded from `pytest`. | 130 |

**On 2026-09-02, 54 Python files moved out of `scripts/` and `experiments/` into
`attic/`** — 31 scripts, 15 probabilistic-context-free-grammar modules, and one
whole thread of 8. Everything moved by `git mv`; every live reference outside
`results/` and `docs/archive/` was rewritten and re-checked with `git grep`, and
the suite was re-run. §3b has the table, including what was deliberately *not*
moved.

## Read these three first

1. **[`docs/threads/experiment_survey.md`](threads/experiment_survey.md)** — every
   experiment graded (REAL / REAL-BUT-SUPERSEDED / SEMI-SYNTHETIC / UNRUN), with
   a corrections timeline. The single most useful file in the repo.
2. **[`docs/threads/feasibility_2026-09-02.md`](threads/feasibility_2026-09-02.md)**
   — the current verdict: assisted proofreading looks feasible, autonomous
   global assembly does not, and the blocker is candidate *generation*.
3. **[`docs/consolidation_plan.md`](consolidation_plan.md)** — why the tree looks
   the way it does, and the disposition (KEEP / FOLD / ATTIC) behind most of the
   moves recorded on this page.

## Check this page against disk instead of trusting it

```bash
.venv/bin/python -c "from neuronauts.experiments.registry import status_table; print(status_table())"
.venv/bin/python scripts/status.py          # consolidation + program state, derived from disk
```

Prose goes stale; those two do not. Where this page and that output disagree,
the output wins.

---

## 1. The live program

### 1.1 Registered experiments

The spine. Each is declared in
[`neuronauts/experiments/registry.py`](../neuronauts/experiments/registry.py)
with a bar written *before* the data existed, run by
[`_runner.py`](../neuronauts/experiments/_runner.py), and recorded as one row in
[`results/RESULTS.md`](../results/RESULTS.md). A run without a row does not
exist.

Status column is the registry's own verdict as read from disk on 2026-09-02
(**7 passed · 5 failed · 6 blocked · 2 not implemented · 1 ready**). "Failed"
means it missed its predeclared bar — most of these are real, useful negative
results, not broken code.

| ID | Status | Module | What it settled |
|---|---|---|---|
| EXP-057 | fail | `exp057_gt_overlay.py` | Only 16.2% of synapse mass has unambiguous ground truth (bar was 30%); 56 seam positives exist in the whole population. |
| EXP-057B | pass | `exp057b_cb2_intake.py` | ConnectomeBench2 intake maps 2,392 external decisions onto population atoms. |
| EXP-057C | not implemented | — | SegCLR embedding intake: declared, no entry point. |
| EXP-058 | pass | `exp058_baseline_ladder.py` | Floor and ceiling on this substrate; proximity clustering is indistinguishable from random (pair precision 0.0006). |
| EXP-059 | pass | `exp059_metric_agreement.py` | The metric implementations agree — 0 disagreements over 200 cases, 11 quantities. |
| EXP-060 | fail | `exp060_endpoint_filter.py` | No endpoint filter proposes candidates; see its `CORRECTION.md` before quoting any number from it. |
| EXP-060B | pass | `exp060b_object_panel.py` | The full recall-vs-panel-size curve: 64.6% recall only at a median panel of 3,870; 12–23% at a usable panel. |
| EXP-061 | fail | `exp061_directed_cone.py` | A tangent cone beats chance by 2–3× (not the retracted 3–6×) but never reaches a usable panel. |
| EXP-062 | not implemented | — | Cuts on real level-2 (L2) adjacency and seam location: declared, no entry point. Cutting is still unmeasured. |
| EXP-063 | pass | `exp063_frankenmerge_detection.py` | **The one clean positive.** Held-out area under the curve (AUC) 0.958 for false-merge detection; polarity alone 0.914. |
| EXP-064–069 | blocked | — | Scorer bake-off, ablation, solver, abstention, scale, attic re-derivation. All sit behind EXP-060/064, which have no passing result. |
| EXP-070 | pass | `exp070_object_distance.py` | The endpoint metric was wrong, but fixing it does not rescue proximity (75.7% ceiling, bar was 90%). |
| EXP-071 | pass | `exp071_connective_gap.py` | The population omits ordinary connective cable: 230/230 bridging nodes are real v117 objects absent from it. Read its `CORRECTION.md`. |
| EXP-072 | fail | `exp072_object_proposal.py` | Widening the object set makes chained recall *worse* (63.6% vs 71.1% control) — dense neuropil connects everything to everything. |
| EXP-073 | ready | `exp073_constrained_chain.py` | Registered run not yet done; its 40 µm probe already falsified the cheap object-level structural filter. |
| EXP-074 | fail | `exp074_seeded_growth.py` | Ran 2026-09-02 20:35 UTC: recovery 0.43%, purity 0.06%, abstention 0.0%. **The survey lists this UNRUN; the survey is one run stale here.** |

Two things on disk that this table cannot cover honestly:

- **EXP-075, EXP-076 and EXP-077** have directories under `results/` but **no
  registry entry and no `RESULTS.md` row**. By the runner's own rule they do not
  exist yet — the registry table above ends at EXP-074.
  - **EXP-075** asked whether local geometry supplies a grower's stop rule. Its
    headline (max-score AUC 0.304) is **withdrawn as unverified** by EXP-076: the
    "already whole, nothing continues here" panels were centered on a point where
    the interior mask clipped the arbor, not where the arbor ends, so in 28 of 35
    whole cells the box sat mid-cable. Read the correction box at the top of
    [`results/EXP-075/evaluation.md`](../results/EXP-075/evaluation.md) before
    quoting anything from that file. Its join-side numbers are unaffected.
  - **EXP-076** ([`evaluation.md`](../results/EXP-076/evaluation.md)) tested the
    seed's own end shape and polarity agreement. Both fail: end shape separates a
    cut from a genuine terminal at AUC 0.476 once matched for distance from the
    soma, and polarity agreement inverts on axons.
  - **EXP-077** is an empty directory with probes in flight
    (`scripts/probe_exp077_*.py`). Nothing to read yet.
  - The pattern EXP-076 names about itself is worth carrying forward: **three
    box-placement errors in one experiment.** On this substrate, where the box
    sits does as much work as what is computed inside it.
- **Probe files** (`results/EXP-07*/probe_*.md`) are same-day pilots at a
  different box size or resolution from the canonical run. Read the parent
  `result.json` first.

### 1.2 The harness — the substrate everything above runs on

[`neuronauts/harness/`](../neuronauts/harness/) is one substrate, built once,
shared by every registered experiment. This is the part of the repo with the
most verification behind it.

| Module | What it holds |
|---|---|
| `substrate.py` | Region bounds and the box/cube conventions every experiment shares. |
| `population.py` | The label-blind v117 atom population (279,075 atoms in the 100 µm cube). |
| `labels.py` | Proofread ground truth attached to atoms, with tiers; label use is evaluation-only. |
| `geometry.py`, `objgeom.py` | Endpoint tables and full object point clouds; endpoints verified a strict subset of L2 nodes. |
| `topology.py` | Contracted L2 adjacency (`k1` / `k10` / `kall` shards). |
| `candidates.py`, `baselines.py` | Candidate-panel construction and the floor/ceiling rungs. |
| `evaluation.py`, `box_truth.py`, `spatial_split.py` | Scoring, in-box truth, and the region-disjoint split. |
| `atom_features.py`, `cb2_positives.py` | Per-atom features; the ConnectomeBench2 positive set. |

[`neuronauts/metrics/`](../neuronauts/metrics/) is the single metric home —
EXP-059 is the test that the implementations agree.
[`neuronauts/report/`](../neuronauts/report/) renders `results/reports/`.
[`neuronauts/data/`](../neuronauts/data/) is the CAVE access layer;
`neuronauts/fetch.py` is the entry point named in `CLAUDE.md`.

### 1.3 Scripts that build the current data

These write `data/substrate/` and `data/external/`. If a registered experiment
is "blocked: missing input", one of these is what produces it.

| Script | Produces |
|---|---|
| `scripts/extract_region_synapses.py` | Region synapses from the local static 337M-row table, one streaming pass. |
| `scripts/build_population.py` | The label-blind atom population for a region. |
| `scripts/build_atom_labels.py` | Proofread ground truth joined onto the population. |
| `scripts/fetch_atom_geometry.py` | Per-atom L2 geometry, fetched in widening synapse tiers. |
| `scripts/build_atom_topology.py` | The contracted adjacency shards (`k1`/`k10`/`kall`). |
| `scripts/build_object_geometry.py` | The object point cloud — every L2 node, not just skeleton tips. |
| `scripts/enumerate_region_objects.py` | Every v117 object in the region, synapse-free ones included (EXP-072's widened set). |
| `scripts/build_object_clouds.py` | Object clouds read straight from the segmentation volume. |
| `scripts/build_object_polarity.py` | Per-object synapse polarity — the grammar's one hard constraint. |
| `scripts/build_contact_panels.py` | The contact panels behind EXP-075/076 (corrected terminal panels land in `data/external/panels_tip/`). |
| `scripts/build_reports.py` | Markdown reports, figures and Neuroglancer views for every result. |
| `scripts/status.py` | Consolidation and program state, derived from disk. |
| `scripts/seed_census.py`, `scripts/tier_census.py`, `scripts/scaffold_census.py` | The censuses the EXP-074 bars were derived from. |
| `scripts/probe_*.py` (~10) | Minimal-repro probes. Kept deliberately: they are the repo's "smallest experiment" habit. |
| `scripts/fetch_*.py` (~10) | CAVE fetches into `data/external/` (skeletons, seed graphs, edit history, proofread manifest). |
| `scripts/warm_cache.py`, `scripts/warm_synapses_1M.py`, `scripts/wait_for_cave.py` | Cache warming and a CAVE-outage watcher. |

---

## 2. Superseded but real

Genuinely measured on real data, then replaced by something better controlled.
Kept in place — these are not archive candidates, and two of them are the
current front-page numbers.

| Thing | Where | What replaced it |
|---|---|---|
| **Pre-registry series EXP-051–056** | `scripts/benchmark_exp051…056*.py`, `results/exp05*_evaluation.md` | The registered program (§1.1). All six are graded REAL — fail-closed, honest negatives. **Attempted and reverted 2026-09-02** — they cannot leave `scripts/`: `neuronauts/report/registry.py:331` resolves a result's script by globbing `scripts/benchmark_exp<id>*.py`, and `tests/test_report.py::test_discover_real_results_parse` asserts it resolves. See §3b. |
| **treestitch global partition** (adjusted Rand index, ARI, 0.752; merge precision 0.951) | `treestitch/` (in place) + `attic/prior_results/{train_l2_partition,multi_region_train,two_level_stitch,spatial_variance,real_lineage_partition,real_region_partition}.py` | Nothing yet. Still the top-line result in `README.md`. The scripts moved to `attic/prior_results/` on 2026-09-02 — archived as *not the current program*, not as wrong; that directory's README says so first. |
| **Tile stitching** (ΔARI +0.10 at 100 µm tiles; 300 µm tiles dead) | `docs/tree_assembly_handoff.md`, `attic/prior_results/two_level_stitch.py` | Nothing — it is why ~100 µm is the harness cube size. |
| **Dendritic scaffold** (59.6% of synapses at 99.8% mass purity) | `docs/tree_assembly_handoff.md` | Nothing. Do not substitute the higher 79–99.8% figure — that is a different, biased substrate. |
| **Tree-DNA half-skeleton identity** (AUC 0.829, random-init 0.768) | `attic/prior_results/{half_split_ablation,ablate_dna,multi_fragment_ablation,half_synapse_ablation,within_type_ablation,global_gnn_ablation,fetch_real_skeletons}.py` | Nothing at half granularity; the same method *fails* at quarter granularity. The seven files are a sibling group — each does a bare `from ablate_dna import …`, so they only run while they sit in one directory. |
| **Cut-face fingerprints** (precision 1.0 at 11% coverage) | `experiments/fingerprints/` (+ its own `archive/`) | Nothing — the plan marks it PROMOTE. Its README's own smoke numbers differ from the handoff's; read both. |
| **Probabilistic context-free grammar (PCFG) cross-region holdout** (AUC 0.816) | `experiments/pcfg/HOLDOUT_RESULTS.md` | Not replaced, but graded SEMI-SYNTHETIC: real skeletons, synthetically introduced break points. |
| **CellGNN box-local baseline** (held-out test F1 score 0.272) | `neuronauts/cell_graph.py`, `scripts/train.py` (kept in `scripts/` — five test modules drive it) | Nothing — the ceiling is structural (a 30 µm box cannot hold a bigger neuron), not a tuning gap. |
| **EXP-060/061/070/071 headline numbers** | `results/EXP-06*/CORRECTION.md`, `docs/threads/qa_pass_2026-09-02.md` | Corrected in place, same day. Always open the `CORRECTION.md` next to a `result.json` before quoting it. |

---

## 3. The archive — `attic/`

Retired pathways. Kept for history, excluded from `pytest` (`testpaths =
["tests"]`), not on any active path. Everything arrived by `git mv`, so
`git log --follow` still works. **[`attic/README.md`](../attic/README.md)** is
the index — three eras, then a per-directory table of what was checked, by what
re-runnable criterion, and the route back for each. Every subdirectory has its
own README.

### 3a. What was here before 2026-09-02

| Path | What | Why it is here |
|---|---|---|
| [`attic/benchmarks_semi_synthetic/`](../attic/benchmarks_semi_synthetic/) ([README](../attic/benchmarks_semi_synthetic/README.md)) | 40 scripts: the original 34 (`benchmark_exp021`–`exp050` + siblings) **plus the 6 added 2026-09-02, below** | 32 of 34 inject `treestitch.worldbuild.frankenmerge_adjacent` at 45% into real skeletons they first cut synthetically; the other two generate neurons or subvolumes outright. |
| [`attic/morpho_grammar/`](../attic/morpho_grammar/) ([README](../attic/morpho_grammar/README.md)) | 26 "engines" | 25 of 26 contain no checkpoint reference at all — untrained models scored on fabricated damage. Route back is EXP-069. |
| [`attic/outer_loop_and_viz/`](../attic/outer_loop_and_viz/) | 9 outer-loop shell scripts, the Flask/Streamlit dashboard, `viz_pipeline.py`, two synthetic viz artifacts | Training loops around the box-local CellGNN track, plus viz built on the synthetic frankenmerge world. No registered route back. |
| [`attic/superseded_modules/`](../attic/superseded_modules/) | `cave_synapse_{counts,degrees}_v1412.py` | Predecessors of `neuronauts/cave_synapse.py`; equivalence checked numerically before the move. |
| [`attic/tests/`](../attic/tests/) | `test_morpho_grammar.py` | Tests attic code; moved with its subject. |

### 3b. Moved 2026-09-02 — the reorganization

54 Python files, all by `git mv`, all with their live references rewritten and
proved. An eighth batch — the seven `benchmark_exp051`–`056` scripts — was moved
and then **reverted**; see the refusals table.

**Verified after the moves**, on the tree as it then stood: `import neuronauts`
and `status_table()` both work; `pytest tests/ -q` gives **1 failed, 1,580
passed, 2 skipped** — the one failure is the pre-existing
`test_multitask_convergence.py::test_loss_decreases_over_steps`; `scripts/status.py`
reports 16/24 consolidation checks with **0 broken markdown links** across the
repository; and `report.registry.discover()` resolves a script for every result
record except `atom_labels_v1822`, which never had one.

| Path | Count | What | Why it is here |
|---|---:|---|---|
| [`attic/prior_results/`](../attic/prior_results/README.md) | 20 | The treestitch partition trainers and the tree-DNA identity ablations | **Real results nothing has replaced** — two of them are the front-page numbers. Not part of the registered program. |
| [`attic/superseded_training/`](../attic/superseded_training/README.md) | 4 | `train_shared_grammar.py`, `train_topology_model.py`, `export_topology_dataset.py`, `inspect_topology_metric.py` | Training entry points with no real-data result. The model code itself stays in `neuronauts/`. |
| [`attic/pcfg_one_offs/`](../attic/pcfg_one_offs/README.md) | 15 | The one-off `*_merge` / `seam_*` / `*_ssl_grammar` scripts from `experiments/pcfg/` | Nothing in the remaining package imports any of them; each was imported from the new path to prove it still resolves. |
| [`attic/one_off_analyses/`](../attic/one_off_analyses/README.md) | 7 | Single-use analyses, exports, characterizations | Never wired into a pipeline; five of the seven are referenced by nothing at all. |
| [`attic/incubating_threads/`](../attic/incubating_threads/README.md) | 8 | `low_res_segmentation/` | Incubating since April with no number against its own graduation bar. Its two tests stayed in `tests/` and still pass. |

**What was deliberately refused, and why.** Each of these met some archive
criterion and was kept anyway:

| Kept | Reason |
|---|---|
| `scripts/benchmark_exp051`–`056` (7 files) | **Moved, then reverted.** `neuronauts/report/registry.py:331` resolves a result record's script by globbing `scripts/benchmark_exp<id>*.py` — a *glob*, so a `git grep` for the basenames does not find it. Moving them silently set `script=None` on five records and broke `tests/test_report.py::test_discover_real_results_parse`, which asserts EXP-056 resolves. The seven were restored byte-identically from `dd49b5cbb^` and the suite re-run. They can only leave `scripts/` together with a change to that resolver, which is package internals. |
| `experiments/soma_graph/` | Phase 3 of `docs/roadmap_global_assembly.md`, which names `build_graph.py:97` as the next line to change. A thread the canonical roadmap tells you to build is not an archive candidate. |
| `experiments/root_neighborhood/` | Reachable from a live, tested command line — `scripts/train.py build-dataset --strategy proofread-core` imports from it. |
| `scripts/train.py` | `README.md`, `INTRO.md` and `CONTRIBUTING.md` document it as the pipeline entry point and five test modules drive it. `docs/consolidation_plan.md` §4.3 marks it SPLIT; that is package surgery, not a move. |
| `scripts/verify_attribution.py` | `docs/tree_assembly_handoff.md` names it as the file the next experiment extends. |
| `scripts/coassign_demo.py`, `scripts/v117_coassign.py` | `INTRO.md` and `neuronauts/coassign/README.md` cite them as the runnable demo of a KEEP package; one is covered by `tests/test_cave_coassign.py`. |
| `scripts/inspect_pipeline.py`, `scripts/fetch_cave_boxes.py` | Covered by `tests/test_inspect_pipeline.py` and `tests/test_pipeline_commands.py`, and both are live paths. |
| `experiments/pcfg/conn_metric.py`, `global_shape_merge.py` | Imported by `neuronauts/report/tracker.py` and `neuronauts/harness/atom_features.py`. Live package dependencies. |
| The name `attic/` itself | Renaming it to `archive/` would mean editing `neuronauts/report/tracker.py` (six path literals) and `neuronauts/morpho_grammar/__init__.py` (the `__path__` shim). Both are package internals this pass was told not to touch, and the gain is cosmetic. `attic/README.md` now opens by saying it *is* the archive. |

**The number nobody should quote.** `EXPERIMENT_LOG.md` §1–3 (merge precision
0.70, path_P 0.84, 99.1% synapse precision, head-to-head against flood-filling
networks, Janelia multicut, DeepMulticut and FlyWire) is entirely this era: untrained engines on
45%-synthetic damage. The file carries a superseded banner. Read it only to
learn what a specific old number claimed, never to learn what is true.

### Added to `attic/benchmarks_semi_synthetic/` on 2026-09-02

Same criterion as the original pass — the script builds its own test world by
synthetically cutting or fusing skeletons, or generates fragments outright —
plus a check that nothing in the repo imports it.

| File (was `scripts/`) | Why |
|---|---|
| `real_franken_partition.py` | Splits real skeletons into pieces, then calls `frankenmerge_adjacent` + `build_world_from_pieces`. The exact attic criterion. |
| `real_skeleton_partition.py` | Calls `split_skeleton_n_pieces` unconditionally — the "20 neurons × 3 pieces" ARI number in the archived `STATUS.md`. |
| `test_global_merge_franken.py` | Splits skeletons into pieces and injects `frankenmerge_adjacent`. Also misleadingly named: it is not a test. |
| `optimize_tree_stitch.py` | Random hyperparameter search over a world made by splitting real skeletons into N pieces. |
| `run_global_merge.py` | `run_synthetic_demo` fabricates two-vertex fragments from scratch; no real data path. |
| `sota_benchmark.py` | "All data is generated synthetically" (its own docstring); produced the superseded state-of-the-art (SOTA) comparison table. |

---

## 4. Support code

| Area | Files | Status |
|---|---|---|
| [`neuronauts/`](../neuronauts/) | 126 `.py` | The package. `harness/`, `metrics/`, `report/`, `data/`, `meshing/`, `experiments/` are live (§1). `cell_graph.py`, `grammar.py`, `path_dataset.py`, `topology_model.py`, `em_corridor.py` are older tracks the plan marks SPLIT or ATTIC — still imported, not yet moved. `legacy/` is quarantined behind `pytest -m 'not legacy'`. |
| [`treestitch/`](../treestitch/) | 20 modules | Real results (§2). `worldbuild.py` and `synthetic.py` are the synthetic-world builders the attic criterion keys on — live code, but a script that calls them is building a synthetic test world. |
| [`tests/`](../tests/) | 110 `.py` | The default suite. Measured immediately before the 2026-09-02 moves and again after them, identical both times: **1 failed, 1,579 passed, 2 skipped** (878s / 869s) — the one failure is `test_multitask_convergence.py::test_loss_decreases_over_steps`, a pre-existing legacy-model regression. |
| [`experiments/`](../experiments/README.md) | 57 `.py` | Thread index with a status legend. `pcfg/` (16 files after the split) and `fingerprints/` (23) are active; `minnie_column/` is a live data pipeline; `root_neighborhood/` (live command line) and `soma_graph/` (roadmap Phase 3) stay despite having no result on the board. `low_res_segmentation/` moved to the archive. |
| [`docs/threads/`](threads/) | 16 pages | Per-thread state, the quality-assurance pass, the feasibility verdict, the survey. The most current prose in the repo. |
| [`docs/archive/2026-09/`](archive/2026-09/) | 9 docs | `STATUS.md`, `program.md`, `NEXT_STEPS.md` and friends. Superseded by `docs/roadmap_global_assembly.md`; several of the moved scripts above are cited only from here. |
| `docs/paper/`, `docs/latex/`, `docs/*_slides.*` | — | Inherit the superseded EXP-020–050 numbers. Not a source of truth. |
| `scripts/` | 60 `.py` tracked (31 moved out 2026-09-02; new EXP-079/081/083 probes moved in) | §1.3 covers the live builders, and [`scripts/README.md`](../scripts/README.md) is now the directory's own index. What is left writes `data/`, renders a report, probes one question, or is `train.py`. |

---

## 5. Known loose ends this page does not fix

- **`experiments/root_neighborhood/` and `soma_graph/`** are still marked ATTIC
  by the consolidation plan and graded "no result yet" by the survey, and were
  **kept anyway** — one is reachable from a live tested command line, the other
  is Phase 3 of the canonical roadmap (§3b). Only the thread owner can turn
  "incubating" into "dead"; `low_res_segmentation/`, which had neither a live
  caller nor a roadmap entry, moved.
- **`experiments/pcfg/` is split (2026-09-02): 16 modules stay, 15 moved** to
  [`attic/pcfg_one_offs/`](../attic/pcfg_one_offs/README.md). The rule applied,
  and it is the conservative half of the consolidation plan's "~20 one-off
  scripts to the attic": a module moved only if *all four* held — nothing left in
  `experiments/pcfg/` imports it; it is not in that README's `Files` table or any
  documented `python -m` line; no test imports it; and nothing in `neuronauts/`,
  `scripts/`, `tests/` or `run_bigdata.sh` references it. `run_experiment.py` and
  `v117_pcfg.py` stayed on the second clause even though nothing imports them,
  which is exactly the trap this page warned about. All 15 were imported from
  their new path afterwards to prove they still resolve — their
  `from experiments.pcfg.<core> import …` lines are unchanged, and
  `sys.path.insert(…, parents[2])` still points at the repository root because
  the nesting depth is the same. What did **not** move: the core
  (`pcfg_partitions`, `synapse_correction`, `skeleton_tokens`, `learned_grammar`),
  the documented entry points, and `conn_metric` / `global_shape_merge`, which
  `neuronauts/report/tracker.py` and `neuronauts/harness/atom_features.py` import.
  (Names appearing in `results/EXP-0*/result.json` are file-manifest provenance
  hashes, not usages — do not mistake them for live references.)
- **`scripts/coassign_demo.py`** is a synthetic-split demo (it says so), but
  `INTRO.md` and `neuronauts/coassign/README.md` cite it as the runnable demo of
  a KEEP package, so it stays. It also carries a **hardcoded CAVE token on line
  37**, against `CLAUDE.md`'s "keep secrets out of files"; that is a separate
  fix, and rotating the token matters more than editing the line.
- **`EXP-063`'s two files disagree** on size-only AUC (0.483 in `evaluation.md`,
  0.654 in the later `result.json`). Unresolved; flagged in the survey.
- **`results/EXP-075/`** has a finding but no registry entry (§1.1).
- **`experiments/pcfg/README.md` line 189** documents `python scripts/v117_pcfg.py …`,
  but that file lives at `experiments/pcfg/v117_pcfg.py`; there is no
  `scripts/v117_pcfg.py`. Pre-existing, not caused by any move on this page.
- **Stale script references in `docs/archive/2026-09/` and in `results/`** point
  at the old `scripts/` paths for every file moved above. Left alone deliberately,
  and for two different reasons: `docs/archive/` holds dated snapshots that would
  be misrepresented by rewriting, and `results/` is a record of what produced a
  number and is not edited at all. `git log --follow` resolves any of them. Every
  reference *outside* those two trees was rewritten and re-checked with
  `git grep`.
- **The 2026-09-02 moves landed inside someone else's commit.** They were staged,
  not committed, but a concurrent session on this branch committed the index as
  part of `dd49b5cbb` ("EXP-080: SegCLR does not select the true continuation").
  `git log --follow` on any moved file still works; the rename is intact. Noting
  it because the commit message does not mention the reorganization at all.
