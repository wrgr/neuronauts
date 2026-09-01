# Neuronauts consolidation plan

> Written 2026-09-01 from a full survey of the working tree at
> `feat/global-merge-assembly`. Two deliverables in one document: (A) how to
> reorganize the repo so it is easier to develop, understand and maintain —
> including what to deprecate — and (B) an experiment program that tests every
> live hypothesis against one substrate, one metric package, and one baseline
> ladder. Companion HTML version published as an artifact.

## 0a. The name: ConnectomeForge

The thing being built here — the label-blind v117 atom substrate, its ground-
truth overlay, the fixed candidate panels, the one metric package, and the
registered experiment program that runs against them — is **ConnectomeForge**.

Naming it matters because it is a benchmark suite and the field's obvious names
are taken: *ConnectomeBench* and *ConnectomeBench2* are existing published
benchmarks (the latter is the 716,485-decision corpus this plan proposes to
import in EXP-057B), and "bench" is crowded generally. Citing someone else's
benchmark while our own suite has no name of its own is how the two get
conflated in a paper.

*Forge* is the right register for what this is: not a leaderboard, but the
place the substrate is made and tested. Where this document says "the harness",
read ConnectomeForge.

Scope of the name, so it does not sprawl:

| Is ConnectomeForge | Is not |
|---|---|
| `neuronauts/harness/` — population, geometry, topology, labels, split, panels | `neuronauts/metrics/` — a general metric package, usable outside it |
| The EXP-057…069 program and its registry | `neuronauts/report/`, `neuronauts/meshing/` — general tooling |
| `results/RESULTS.md` and the per-experiment records | The retired `attic/` line |

The Python package stays `neuronauts.harness` for now; renaming it is a
separate, mechanical change and should not ride along with research work.

## 0. What this plan asserts

1. The repo is ~116k lines of Python carrying **three architectural regimes**
   and **four independent metric implementations**. Nobody can tell which
   number is the current baseline. Fixing that is worth more than any model.
2. **The EXP-020–050 result series is not evidence.** Every benchmark script in
   that range builds its test world with `treestitch.worldbuild`
   (synthetic cuts, synthetic frankenmerges at 45%), and the 26
   `morpho_grammar` engines load no trained checkpoint. The SOTA comparison
   table in `EXPERIMENT_LOG.md` and both paper drafts inherit this. EXP-051's
   own audit reached the same conclusion for EXP-049.
3. The real, defensible results are few and should be the spine of the
   codebase: label-blind atoms + real L2 topology (harness), tree-DNA identity
   at half-skeleton scale, the certifiable dendritic scaffold, tile stitching,
   and the cut-face texture combiner. Everything else is a candidate.
4. Reorganize by **pipeline stage with typed artifacts**, put the metrics in
   **one package**, move dead pathways to an **attic** with import shims, and
   make every future number come from a registered, fail-closed experiment.
5. Run experiments in the order **substrate → baselines → propose → cut →
   score → assemble → re-derive prior claims**. Each stage's result is an
   input the next stage needs; running them out of order is how EXP-053B/054/
   055 all failed on prerequisites.

## 1. Diagnosis — what the survey found

### 1.1 Size and shape

| Area | Files | Lines | Notes |
|---|---:|---:|---|
| `neuronauts/` | 101 | 33,604 | 9 subpackages; `cell_graph.py` 3,958 lines |
| `scripts/` | 106 | 33,970 | `train.py` 3,391 lines / 17 subcommands; 36 `benchmark_exp*` |
| `experiments/` | 80 | 18,957 | 10 threads; `pcfg/` alone is 34 files |
| `tests/` | 87 | 20,251 | 960 pass, 7 fail, 1 collection error, 192 quarantined; 8m53s |
| `treestitch/` | 20 | 7,066 | Depends on `neuronauts` (6 files); nothing depends on it |
| `docs/` | 40+ | — | Five "direction" docs, three of which disagree on the canonical pipeline |

Layering today is actually clean in one direction: `neuronauts` imports nothing
from `treestitch` or `experiments`, and `scripts` imports nothing from
`experiments`. That makes the fold-in below mechanical rather than surgical.

### 1.2 Three regimes, still coexisting

| Regime | Where it lives | Real-data status |
|---|---|---|
| v1 agent / membrane simulation | `neuronauts/legacy/` (192 tests) | Quarantined; dead |
| v2 shared grammar + GAT | `grammar.py`, `shared_grammar_model.py`, `scripts/train.py` | EXP-053A: no checkpoint separates real continuation pairs from dense confusers |
| Box-local CellGNN | `cell_graph.py`, `scripts/train.py` | F1 0.27 ceiling, architectural (`docs/architecture.md`); its 7 CLI tests are the ones failing |
| treestitch global partition | `treestitch/` | Real: out-of-sample ARI 0.752, merge_P 0.951; Bar 3 = 0 |
| morpho_grammar "engines" | `neuronauts/morpho_grammar/` (26 files) | Semi-synthetic only (see §1.4) |
| harness (label-blind atoms, real L2) | `neuronauts/harness/` (uncommitted) | Real; substrate built and verified this week |

### 1.3 Four metric implementations

| Module | Computes | Used by |
|---|---|---|
| `neuronauts/line_graph.py` | line-graph F1 (terminal metric), sampled-pair F1 | CellGNN, 25 EXP scripts |
| `treestitch/metrics.py` + `partition.py` | ARI, merge P/R, completeness, connectome edge F1, cable | treestitch |
| `neuronauts/global_merge/eval/benchmark.py` | ARI, pairwise merge P/R, frankenmerge split rate, path-length/ERL | 25 EXP scripts |
| `experiments/pcfg/conn_metric.py` | component-level connectome accuracy | pcfg |

They overlap on ARI and pairwise P/R but with different conventions (e.g.
`benchmark.py` returns precision 1.0 when no merges are predicted, which is
why "untouched v117" scores perfectly on precision in EXP-052). No test
asserts they agree.

**Since this survey began, `neuronauts/metrics/` has landed** (uncommitted): one
sparse-contingency core, NaN for undefined ratios, partition / edge /
connectome / frankenmerge / calibration / ranking modules, `evaluate_partition_suite`,
and 206 passing tests together with `report/` and `harness/`. Only
`global_merge/eval` delegates to it so far; `line_graph.py`, `treestitch/partition.py`,
`treestitch/metrics.py` and `pcfg/conn_metric.py` still carry their own maths.
Phase 1 below is therefore "finish the migration", not "create the package".

### 1.4 The EXP-020–050 provenance problem

Checked directly, not inferred:

- `scripts/benchmark_exp0{21,26,30,35,40,44,45,48,49}*.py` each import
  `treestitch.worldbuild.frankenmerge_adjacent` (or a dense "fallback") and
  apply it at 45% to real proofread skeletons that were themselves
  synthetically cut. EXP-035 — the "restored SOTA" the paper walkthrough cites —
  is `frankenmerge_adjacent(pieces_rec, 0.45, rng, radius_nm=9500)` on line 158.
- 15 of the 26 `neuronauts/morpho_grammar/*.py` engines draw random numbers at
  runtime; none contains `torch.load` or a `.pt` path. EXP-051 recorded the
  same for the SANTIAGO infiller.
- `results/exp051_evaluation.md` found EXP-049's fallback unconditional and
  the visualizer rendering synthetic IDs.

Consequence: the Merge Precision 0.70 / path_P 0.84 / "99.1% synapse
precision" rows in `EXPERIMENT_LOG.md` §3, and the numbers in
`docs/paper/*.tex`, describe performance on synthetically damaged data scored
by untrained engines. They must be labelled **superseded** until re-derived
under §6. This is the single most important cleanup item.

### 1.5 What has actually held up on real data

| Result | Where | Evidence |
|---|---|---|
| Polarity → compartment, label-blind (~95% pure vs binomial null) | harness | `results/figures/06_polarity_compartments.png` |
| Real L2 adjacency for 40,109 atoms, 100% attribute coverage | harness | `results/atom_geometry_tiers.json`, verified independently |
| Contracted topology: 5.1M endpoints, 245/atom, 4.3 components/atom | harness | `results/atom_topology_k10.json` |
| Tree-DNA individual identity at half-skeleton scale (within-type AUC 0.829) | STATUS.md | `scripts/half_split_ablation.py --encoder gnn` |
| Dendritic scaffold certifiable by nucleus lineage (59.6% of synapses at 99.8% purity) | tree_assembly handoff | 200 µm dual-side census |
| Tile stitching +0.10 ARI at ~0 precision cost; 100 µm tiles, not 300 | tree_assembly handoff | 3 runs, 200 µm box |
| Cut-face texture combiner: precision 1.0 at 11% coverage on real split sites | `experiments/fingerprints/` | 73 held-out sites |
| treestitch out-of-sample ARI 0.752, merge_P 0.951 | README | Phase 2.11, seam-buffered |
| **Falsified:** proximity attribution of axon fragments (0/32; 9/1,063) | tree_assembly handoff | real, adjudicated |
| **Falsified:** single global edge-length threshold as atomizer | EXP-056 | 116 real mixed-lineage roots |

## 2. Principles for the reorganization

1. **One pipeline, by stage, with typed artifacts.** `schemas.py` already
   defines `Region → Fragment → NeuronHypothesis → ConnectomeGraph`; the
   directory tree should mirror that and nothing else.
2. **One metric package.** Every number in a results file comes from
   `neuronauts.eval`. Legacy implementations become adapters with agreement
   tests, then go away.
3. **Deprecate by moving, not deleting.** Dead pathways go to `attic/` with a
   one-line epitaph and an import shim that raises `DeprecationWarning`. Git
   history is preserved with `git mv`. A pathway comes back only by passing a
   registered experiment.
4. **Every claim is a command.** A result file names the script, config, git
   SHA and data manifest that produced it. If it cannot be regenerated it is
   not a result.
5. **Fail closed, predeclare, report distributions.** EXP-051–056 already do
   this; make it the runner's job so no one has to remember.
6. **Synthetic data says so in its name.** Any world built by
   `synthetic.py`/`worldbuild.py` writes to `results/synthetic/`, never to
   `results/EXP-*`.
7. **Small experiments first.** ≤2 h per EXP on the cached substrate; network
   only through `neuronauts.data` with rate limiting and failure accounting
   (the lesson of this week's silent 14% loss).

## 3. Target layout

Packages marked ● already exist in the working tree (uncommitted); the layout
keeps their names rather than churning them.

```
neuronauts/
  data/         CAVE access + caches. fetch, lineage, synapse tables,
                supervoxel→root mapping, L2 geometry (rate-limited, fail-loud),
                proofread manifests, minnie_column constants.
● harness/      The substrate. Label-blind atom population, contracted L2
                topology, GT overlay (labels), spatial split, endpoint
                candidate panel, baseline ladder, rank + assembly evaluation.
                Gains: kimimaro skeletons (from cell_graph.py).
  represent/    Encoders only. dna (TreeDNA), skeleton_gnn, path_edge_encoder,
                grammar encoders, texture (cut-face) encoder.
  score/        Scorers over a fixed candidate panel. pcfg, learned pairwise,
                texture combiner, stacked. Each returns calibrated P(same).
  assemble/     Solvers over scored candidates. union-find, GAEC correlation
                clustering, constrained multicut, soma-seeded forest.
● metrics/      THE metric package (§7). Sparse contingency core; partition,
                edges, connectome, frankenmerge, calibration, ranking, suite.
● report/       Provenance capture + audit, result registry, Markdown render,
                figures, Neuroglancer state builder. Gains: the runner (§5).
● meshing/      Precomputed mesh / skeleton encoding + local server for
                Neuroglancer views. Replaces dashboard/ and viz/.
  experiments/  One module per EXP: config, predeclared criterion, entry
                point; run through report's runner. Writes results/EXP-xxx/.
  schemas.py    Typed artifacts (unchanged).
attic/          Quarantined code, importable with a warning, excluded from CI.
scripts/        Thin CLIs only (`neuronauts <stage> <verb>`); no logic.
tests/          unit/ (<2 min, no network)  ·  slow/  ·  network/ (opt-in)
docs/           README · ARCHITECTURE · EXPERIMENTS · RESULTS · CONTRIBUTING
                + threads/ (research-thread pages) + archive/ (dated)
results/        EXP-xxx/{report.md, metrics.json, config.json, provenance.json}
                + reports/ (rendered by scripts/build_reports.py)
```

`propose/` from the first draft is absorbed: `harness/candidates.py` already is
the endpoint-panel generator, and cut proposals belong beside the topology
they cut. `treestitch/` folds in: `data.py`/`realworld.py` → `data/` and
`harness/`; `partition.py`/`stitch.py`/`assemble.py` → `assemble/`;
`metrics.py` → a delegating wrapper over `metrics/`, then gone; `embed.py` →
`represent/`; `risk.py`/`calibration.py` → `metrics/calibration` (exists);
`synthetic.py`/`worldbuild.py` → `experiments/synthetic/`. Keep the name
"treestitch" as the research-thread page, not as a package.

## 4. Deprecation triage

Verdicts: **KEEP** (stays, possibly moved) · **FOLD** (merge into a stage
package) · **ATTIC** (quarantine with shim; may return via experiment) ·
**DELETE** (pure duplicate or scratch).

### 4.1 Packages

| Path | Lines | Verdict | Why |
|---|---:|---|---|
| `neuronauts/legacy/` | ~3,500 | ATTIC | Already quarantined; 192 tests still run on `pytest` without `-m`. Move the whole thing so the default suite is clean. |
| `neuronauts/morpho_grammar/` (26 engines) | ~9,000 | ATTIC | No engine loads a checkpoint; every consuming benchmark is semi-synthetic (§1.4). Re-admit any single engine only by passing EXP-064/069. Exception: `synapse_segment_typer.py` if the harness needs it — evaluate in EXP-063. |
| `neuronauts/global_merge/solver/constrained_multicut.py` | 734 | FOLD → `assemble/` | Real solver, needed for EXP-066. |
| `neuronauts/global_merge/eval/benchmark.py` | ~200 | FOLD → `metrics/` | Already a delegating wrapper (`eval/__init__.py`); retire after EXP-059 confirms agreement. |
| `neuronauts/global_merge/represent/*` (vicreg, tangent_flow, EM sampler, local_em_verifier) | ~1,500 | ATTIC | Built for the synthetic series; no real-data evidence. `tangent_flow` may return for EXP-064 tangent features — as a feature, not an engine. |
| `neuronauts/coassign/` | ~500 | FOLD → `assemble/` + `represent/` | The "correct core idea" per `NEXT_STEPS.md` (synapses as invariant nodes, calibrated P(same), K materializations). Small and clean. |
| `neuronauts/cell_graph.py` | 3,958 | SPLIT | `precompute_self_skeletons_for_cache` (kimimaro) → `substrate/skeleton.py` (**keep, chosen path**). `partition_from_embeddings` → `assemble/`. `CellGNN`, `build_synapse_graph`, beam search, tangledness → ATTIC as the box-local baseline reference. |
| `neuronauts/grammar.py`, `shared_grammar_model.py` | 1,424 | SPLIT | `TorchPathEncoder`/`MergeScorer` → `represent/` as scorer candidates for EXP-064. `GlobalAssemblyGAT`, `gat_train_step`, `BridgeHead` → ATTIC (EXP-053A: GAT variant no better; idle since June). |
| `neuronauts/path_dataset.py` | 1,776 | SPLIT | Edit-pair mining (v117→v1412 false-merge/split labels) → `data/edits.py` — this is the error_correction thread's real training signal. Path-sequence builders for CellGNN → ATTIC. |
| `neuronauts/em_corridor.py`, `topology_model.py`, `topology_dataset.py` | ~1,600 | ATTIC | Marked "optional, not wired in" since April; no real-data result. |
| `neuronauts/harness/` (10 modules) | ~3,000 | KEEP | The substrate package: population, geometry, topology, labels, spatial_split, candidates, baselines, evaluation. **Uncommitted — commit first.** Only `geometry.py`'s CAVE calls move to `data/`. |
| `neuronauts/metrics/` (12 modules) | ~2,000 | KEEP | The one metric home (§1.3). Uncommitted. Missing from `pyproject` packages. |
| `neuronauts/report/` (5 modules) | ~2,500 | KEEP | provenance / registry / render / figures / ngl; already renders `results/reports/`. Uncommitted. Missing from `pyproject`. |
| `neuronauts/meshing/` (6 modules) | ~1,500 | KEEP | Neuroglancer precomputed mesh + skeleton serving. Uncommitted; in `pyproject`. Supersedes `dashboard/`, `viz/`. |
| `neuronauts/{assembly,merge,merge_dataset,dijkstra,helpers,viz}.py` | ~2,000 | AUDIT each | Keep only what `assemble/` and `eval/` import after the fold; the rest → ATTIC. |
| `neuronauts/{cave_synapse*,bulk_synapses,synapse_root_counts_static,cave_root_mapping}.py` | ~2,500 | FOLD → `data/` | Five overlapping CAVE synapse fetchers. Consolidate behind one `data.synapses` API with the count-validation rule from `CLAUDE.md` as a test. |
| `treestitch/` | 7,066 | FOLD (see §3) | Real results; keep the code, retire the top-level package. |

### 4.2 `experiments/` threads

| Thread | Verdict | Why |
|---|---|---|
| `fingerprints/` (cut-face combiner) | **PROMOTE** → `represent/texture.py`, `score/texture.py`, `propose/cone_panel.py` | The one channel the tree-assembly handoff says works (precision 1.0 @ 11% coverage). Its `evaluate.py` (panel recall / top-1 / abstention) → `eval/`. Drop the checked-in `.pt` files from git; they are regenerable. |
| `pcfg/` (34 files) | SPLIT | `pcfg_partitions.py`, `skeleton_tokens.py`, `learned_grammar.py` core → `score/pcfg/`. The ~20 one-off `*_merge.py`/`*_cut.py`/`seam_*` scripts → ATTIC. Keep `HOLDOUT_RESULTS.md` and `FINDINGS_*.md` in `docs/threads/pcfg.md`. |
| `minnie_column/` | FOLD → `data/minnie_column.py` | Live data pipeline (column bbox, nucleus manifest, tubes). |
| `error_correction/`, `grammar/`, `cell_assignment/`, `tree_dna/`, `topology/` | DOCS ONLY | README-only thread pages → `docs/threads/`. |
| `root_neighborhood/`, `soma_graph/`, `low_res_segmentation/` | ATTIC | Incubating since June with no result on the board; two open stale PRs (#9). |

### 4.3 `scripts/` (106 files)

| Group | Count | Verdict |
|---|---:|---|
| Harness: `fetch_*`, `build_*`, `extract_region_synapses`, `probe_*`, `viz_*` | ~17 | KEEP → become `neuronauts data|substrate` CLI verbs; probes stay as `scripts/probes/` (they are the repo's minimal-repro habit). |
| `benchmark_exp051`–`056` | 7 | KEEP → `experiments/exp051…056.py` (they are the template for the runner). |
| `benchmark_exp021`–`050` | 28 | ATTIC → `attic/benchmarks_semi_synthetic/` with a README stating exactly what §1.4 says. |
| `train.py` (17 subcommands) | 1 | SPLIT: `build-dataset`/`remap-roots`/`fetch-cave-edits*` → `data` CLI; `train-*` for CellGNN/GAT → ATTIC with their code; `evaluate` → `eval` CLI. |
| treestitch trainers (`train_l2_partition`, `multi_region_train`, `real_*_partition`, `two_level_stitch`, `spatial_variance`) | ~9 | KEEP → `experiments/` as the treestitch baseline reproduction (EXP-058 needs them). |
| Ablations (`*_ablation*`, `ablate_dna`) | ~8 | KEEP → `experiments/` (they back the tree-DNA results in §1.5). |
| `codex_optimize*`, `watch_and_eval.sh`, `eval_*.sh`, `run_*.sh`, `generate_dashboard`, `export_viz_data`, `render_whitepaper_pdf.sh` | ~12 | ATTIC (outer-loop optimizer and viz that rendered synthetic data). |
| Remaining one-offs | ~24 | AUDIT: DELETE if not referenced by any doc or test after the moves. |

### 4.4 Docs, results, models

| Item | Verdict |
|---|---|
| `program.md`, `pipeline_state.md`, `model.md`, `global_inference_roadmap.md`, `global_topological_merge_plan.md`, `docs/TODO.md`, `NEXT_STEPS.md`, `STATUS.md` | ARCHIVE → `docs/archive/2026-09/` with a date stamp; the surviving `ARCHITECTURE.md` cites them. |
| `EXPERIMENT_LOG.md` §1–3 | Mark **superseded** in place (§1.4); becomes the first entry in `docs/archive/`. |
| `docs/paper/`, `docs/latex/`, `docs/*_slides.*`, `dashboard/`, `viz/` | ARCHIVE → `docs/papers/` with a header: "numbers pending re-derivation under EXP-064/069". |
| `docs/roadmap_global_assembly.md`, `architecture.md`, `tree_assembly_handoff.md`, `grammar_harness_handoff.md`, `synapse_pair_architecture.md` | KEEP → source material for `ARCHITECTURE.md`; handoffs → `docs/threads/`. |
| `models/*.pt` (12 tracked) | KEEP `neuronauts_l2_partition*.pt`, `cell_gnn_seg.pt` as baseline references for EXP-058. Others → `models/attic/` (still tracked, not on the default path). |
| `run_logs/` (44 dirs) | Untracked already; delete locally once the ledger is regenerated by the runner. |
| Remote branches (`claude/*`, `codex/*`, 5 of them) | Decide per `experiments/README.md` §"branches": merge or delete within Phase 0. |

### 4.5 Tests

- The 7 failures (`test_pipeline_commands` ×6, `test_multitask_convergence`) and
  the collection error (`test_synapse_table_filter`) are all on ATTIC code.
  They move with it.
- Split into `tests/unit` (no network, <2 min — the current suite is 9 min),
  `tests/slow`, `tests/network` (opt-in marker; `CLAUDE.md`'s "validate counts
  against a trusted query" becomes an actual test here).
- Add `tests/eval/test_metric_agreement.py` before retiring any legacy metric
  (§5, Phase 1).

## 5. Migration sequence

Each phase is one PR, leaves the default test suite green, and uses `git mv`.

| Phase | Scope | Effort | Exit criterion |
|---|---|---:|---|
| **0 · Freeze** | Tag `pre-consolidation`. Commit `harness/` + this week's fixes. Add the superseded header to `EXPERIMENT_LOG.md` and the paper dirs. Skip the 7 failing tests with a reason string. Decide the 5 stale branches. | 1 day | `pytest -m 'not legacy'` green; provenance note merged. |
| **1 · `metrics/`** | Finish what exists: make `line_graph.py`, `treestitch/partition.py`, `treestitch/metrics.py`, `pcfg/conn_metric.py` delegating wrappers (only `global_merge/eval` is today); agreement tests on shared fixtures (a known partition with known ARI/P/R); write `docs/metrics.md` (the `__init__` docstring already promises it). Retire nothing yet. | 1–2 days | Agreement tests pass or every disagreement is documented as a convention choice. |
| **2 · Attic** | `git mv` everything marked ATTIC; add `attic/__init__.py` shims that re-export with `DeprecationWarning`; drop attic from `testpaths`. Add `attic/README.md` with the §4 table. | 2 days | Default suite < 3 min; `import neuronauts.morpho_grammar` warns but works. |
| **3 · Fold** | treestitch + harness + coassign + fingerprints into the stage packages; split `cell_graph.py` and `train.py`; `scripts/` → thin CLIs. Update `pyproject` packages list, drop the `neuronauts = legacy.run:main` console script. | 3–5 days | Every KEEP script runs via the new CLI; `docs/threads/*` link to the new paths. |
| **4 · Docs** | Write `ARCHITECTURE.md` (from roadmap + architecture + handoffs), `EXPERIMENTS.md` (registry), `RESULTS.md` (ledger), refresh `README.md`, `CONTRIBUTING.md`. Archive the rest. | 1–2 days | A new contributor can find the current baseline in one click. |
| **5 · Runner** | On top of `report.provenance` (`write_result`, already used by the harness scripts) and `report.registry` (already discovers and grades results): a `run()` that reads the predeclared criterion and prerequisite EXP ids from the module header, fails closed, and writes `results/EXP-xxx/`; `build_reports.py` then renders it. Port EXP-051–056 onto it. | 1–2 days | EXP-056 reproduces byte-for-byte through the runner and appears in `results/reports/README.md` with grade A provenance. |

Total ≈ 2.5 working weeks for one person, parallelizable across Phases 1/2/4.
Nothing in §6 needs to wait for Phase 3; EXP-057/058 can start after Phase 1.

## 6. Experiment program

### 6.1 Hypothesis inventory

Drawn from every direction doc, handoff, and results file. Status uses the
repo's own evidence standard: **supported** = real data, held out, reported
with counts; **falsified** = same standard, negative; **unverified** =
only semi-synthetic or in-sample evidence; **untested** = no run.

| # | Hypothesis | Status | Evidence / gap |
|---|---|---|---|
| H1 | Synapse polarity gives axon/dendrite compartments without GT | supported | 95% pure vs binomial null, 20,826 atoms |
| H2 | Skeleton morphology (tree-DNA) identifies individual neurons, even within cell type | supported at half-skeleton; fails at quarter | AUC 0.829 vs 0.60 collapse; data limit, not loss |
| H3 | Endpoint proximity alone identifies continuations | **falsified for axons**; works for dendrites via tiling | 0/32 adjudicated links; +0.10 ARI tiling |
| H4 | The dendritic scaffold is certifiable from nucleus lineage at ~100% precision | supported | 59.6% of synapses, 99.8% purity |
| H5 | Directed continuation (tangent cone) + EM texture beats proximity | supported on split sites; untested on link certification | precision 1.0 @ 11% coverage |
| H6 | Grammar (PCFG) priors improve merge scoring over learned-only | **unverified** | EXP-02x–050 semi-synthetic; EXP-053A real: nothing separates |
| H7 | Real L2 adjacency gives better cut surfaces than synapse-MST geometry | untested | EXP-056 falsified MST thresholds; real L2 now available |
| H8 | A mixed-polarity atom is a frankenmerge flag (Bar 3 signal) | untested | Bar 3 = 0.000 in every real run to date |
| H9 | Correlation clustering (GAEC / multicut) beats threshold union-find on frankenmerges | unverified (synthetic only) | `NEXT_STEPS.md` |
| H10 | Cross-box (global) assembly breaks the 0.27 line-graph F1 ceiling | partially supported | tiling result; no line-graph F1 reported at scale |
| H11 | Caliber continuity is a useful merge feature | untested | `mean_dt_nm` now on every L2 node for free |
| H12 | Calibrated abstention yields a proofreading-usable precision/coverage curve | supported on cut-face; untested on assembly | 8/73 sites at precision 1.0 |
| H13 | Endpoint filtering by leaf length × caliber isolates real split sites from spines | untested | 5.1M endpoints; 6.9% survive a 2 µm/50 nm filter |

### 6.2 Common protocol

- **Substrate:** the 100 µm harness cube (centre 663/591/860 µm), tier ≥10
  atoms first, ≥5 and ≥1 as coverage allows; GT = proofread status v1822 gold,
  attached for evaluation only; spatial train/val/test split with a 50 µm seam
  buffer (Phase 2.11's leak fix) — **EXP-057 builds this once**.
- **Candidates fixed before scoring.** Scorers are compared on identical
  candidate sets (EXP-054's design), solvers on identical scores.
- **Predeclared criterion** in the module header; the runner refuses to
  report metrics when a prerequisite fails (EXP-053B–055 behaviour, automated).
- **Report:** the §7 table, per-atom distributions, 3 seeds with 95% CI where
  anything is learned, and population counts (atoms, positives, candidates).
- **Budget:** ≤2 h wall-clock per EXP on the cached substrate.

### 6.3 Series

**A — Substrate and baselines (prerequisite for everything)**

| EXP | Question | Design | Criterion |
|---|---|---|---|
| **057** GT overlay + split | What fraction of atoms/synapses have unambiguous GT, and where? | Attach v1822 gold ids to tier ≥10 atoms via supervoxel majority; classify atoms {single-lineage, mixed-lineage, no-GT}; spatial 60/20/20 split with seam buffer; report coverage per compartment (H1). | ≥30% of synapse mass with single-lineage GT, else widen tier. |
| **058** Baseline ladder | What are the floor and ceiling on this substrate? | Run: (i) untouched v117; (ii) random merges at matched count; (iii) proximity union-find at 1/2/5 µm; (iv) `neuronauts_l2_partition.pt`; (v) `cell_gnn_seg.pt` within tiles; (vi) GT-lineage oracle. All through `eval/`. | Publishes `RESULTS.md` row 1; every later EXP must beat (iii). |
| **057B** ConnectomeBench2 intake | Can we buy our way past the 56-positive wall? | Download the MICrONS split of ConnectomeBench2 (716,485 expert proofreading decisions across FlyWire / MICrONS / Fish1 / H01). Record segmentation version, id space, coordinate frame, task format; count decisions landing in or near the harness cube and how many map to v117 roots via lineage. Its third task, "mask segmentation for merge error correction", is our seam-location problem renamed. | ≥1,000 mapped merge-or-split decisions in or near the cube; else record the version gap and move on. **Half a day, and it gates 062/063's sample size** — run it first in series A. |
| **057C** Embedding intake | Is tree-DNA already published for this volume? | Check whether Weis et al. 2025 release per-root GraphDINO embeddings (>30,000 MICrONS excitatory neurons); if so join to the gold manifest and test cosine separation of same-cell vs different-cell fragment pairs — the same test our own tree-DNA must pass. | Separation at or above tree-DNA's on the harness; free class-conditioning for the grammar mixture if it holds. Half a day. |
| **059** Metric agreement | Do the four legacy metrics agree? | Run all four on the 058 outputs; diff. | Agree to 1e-6 or the difference is documented; three retire. |

**B — Candidate generation (H3, H5, H13)**

| EXP | Question | Design | Criterion |
|---|---|---|---|
| **060** Endpoint filter | Which endpoints are split sites? | Recall of GT continuation pairs vs leaf-length × caliber × tangent-cone thresholds; panel size distribution. Redo of 053B on a substrate with coverage. | ≥90% recall at median panel ≤20. |
| **061** Proximity vs cone, by compartment | Does H3's failure hold for dendrites too? | Same panels stratified axon/dendrite via H1; compare radius-only vs radius+cone. | Cone improves precision at equal recall on axons. |

**C — Cuts and frankenmerge detection (H7, H8)**

| EXP | Question | Design | Criterion |
|---|---|---|---|
| **062** Real-L2 cuts | Do real adjacency cuts beat MST-geometry cuts? | On the 116 mixed-lineage roots (056) plus 057's; cut candidates = L2 bridge edges ranked by caliber drop / branch context; EXP-056's metrics. | ≥90% same-lineage pair recall AND ≥50% cross-lineage split recall. |
| **063** Polarity flag | Is mixed polarity a frankenmerge detector? | Precision/recall of "pre-fraction in (0.3,0.7)" vs GT mixed-lineage; compare `synapse_segment_typer`. | Bar 3 (fk_split) > 0.5 at precision ≥0.8. |

**D — Scoring (H2, H5, H6, H11)**

| EXP | Question | Design | Criterion |
|---|---|---|---|
| **064** Fixed-panel scorer bake-off | Which signal separates true continuations? | On the 060 panel: distance; tangent alignment; caliber continuity; tree-DNA cosine; PCFG grammar score; cut-face texture combiner; stacked logistic. AUROC, precision@k, calibration (ECE). | Any single scorer AUROC ≥0.80 on held-out split; stacked ≥ best single + 0.03. |
| **065** Scorer ablation | What does each feature add? | Leave-one-out from the stacked model, 3 seeds. | Reported; no criterion. |

**E — Assembly (H4, H9, H10, H12)**

| EXP | Question | Design | Criterion |
|---|---|---|---|
| **066** Solver bake-off | At fixed scores, which solver wins? | Union-find threshold; GAEC; constrained multicut; soma-seeded forest with one-soma rule and H4 scaffold. ARI, merge P/R, ERL, is_tree, Bar 3, line-graph F1. | Beats 058(iii) on ARI with merge_P ≥0.95. |
| **067** Abstention curve | Is there a proofreading operating point? | Precision vs coverage sweep; report the point at merge_P ≥0.95 and ≥0.99. | Coverage ≥20% at merge_P ≥0.95. |
| **068** Scale | Does it hold at 200 µm and under tiling? | 100 → 200 µm cube; 2×2 vs 6×2 tiles of 100 µm. | ΔARI within CI of 066. |

**F — Re-derivation of prior claims (H6)**

| EXP | Question | Design | Criterion |
|---|---|---|---|
| **069** Morpho-grammar on real data | Does any attic engine earn its way back? | Best-performing engine from the EXP-035/048 line, run under the 064 protocol with a *trained* grammar; same panel, same metrics. | Beats stacked 064 scorer; else `EXPERIMENT_LOG.md` §3 stays superseded. |

Order is strict A → B → C → D → E; F after D. Each EXP is one module in
`neuronauts/experiments/` and one PR.

### 6.4 Relation to the grammar track

Two companion documents landed the same day and are folded in here rather than
run as a parallel track:

- `docs/pcfg_global_assembly_report.md` — the case for a parsed, typed tree
  grammar as **verifier and seam locator** on the harness substrate, with
  experiments E0–E5.
- `docs/grammar_literature_directions.md` — what the wider literature says
  about that plan, and ten ranked directions. Its §2 independently corroborates
  four of our own falsifications (synthetic splices transfer at chance;
  linearised sequence grammars score geometry not topology; hand oddness rules
  over-flag; detection is easy and the cut operator is hard).

**The blocking constraint both documents converge on, now measurable.** The
v1822 overlay exists (`results/atom_labels_v1822.json`), and it is thin where
it matters:

| Quantity | Count | Share of 279,075 atoms |
|---|---:|---:|
| Pure, gold-owned atoms | 2,357 | 0.84% |
| Pure, silver-owned | 2,445 | 0.88% |
| **Mixed-lineage (false merges)** | **2,444** | 0.88% |
| **Mixed-lineage *and* proofread-owned** | **56** | 0.02% |
| Distinct proofread owner roots | 1,255 (474 gold) | — |

Those 56 atoms are the trustworthy positives for seam location. The repo's own
seam GNN was **net-negative at 150 objects and only cleared zero at 513**. So
EXP-062/063 as drafted are below the data-starvation wall before they start —
which is precisely why the grammar (a few hundred parameters, learned from
*correct* gold cells rather than from rare error objects) is the right first
model, and why the ConnectomeBench2 intake probe below is urgent rather than
optional. The 2,444 unrestricted mixed atoms are a larger but weaker pool: they
rest on non-proofread lineage, so they belong in a clearly separated stratum,
never pooled with the 56.

The E0–E5 map:

| PCFG | Series slot | Note |
|---|---|---|
| E0 fit + parse sanity | prerequisite of D | Needs 057's gold overlay; no equivalent in A–F because it validates a model, not a hypothesis. |
| E1 false-merge detection | **063** | E1's bar (AUC ≥0.875, beat `global_shape_merge`) is stronger than 063's polarity bar; adopt it and treat polarity as the cheap baseline in the same run. |
| E2 seam location | **062** | The sharper framing: 062 asks "do real-L2 cuts beat MST cuts", E2 asks "which edge" with a top-1 bar. Merge them; run E2's metric. |
| E3 grammar atomiser | **062** | Same protocol as EXP-056, same bar. One run, two scorers. |
| E4 `CUT`-typed tips | **060** | E4 states the bar 060 needs: ≥0.9 recall of true split sites at ≤1% of endpoints. Adopt verbatim. |
| E5 `Δ_attach` in stitching | **066** | A solver term, so it belongs in the solver bake-off. |

Practical consequence: the grammar is one scorer in the 064 bake-off and one
term in the 066 solver, not a parallel track. Its detection and seam bars
replace the weaker ones I drafted for 062/063. E0 runs as part of Phase 1's
substrate work.

From the literature survey, three further additions earn slots — the rest of
its ranked list is explicitly *not scheduled* (generative tree transformers for
completion, graph-grammar form discovery, point-affinity clustering unless
retrained on real errors):

| Addition | Slot | Bar |
|---|---|---|
| **Neural emissions in the grammar** (Torch-Struct-style differentiable inside pass; compound-PCFG per-object latent for cell class / layer) | second grammar scorer in **064** | Seam top-1 and net pair-error above the fitted grammar on the same folds, hard zeros unchanged. |
| **VLM verifier channel** (ConnectomeBench protocol: three orthographic mesh renders + our parse overlay, multiple-choice with decoys) | channel in **064**, battery member in **066/067** | ≥70% multiple-choice *and* errors uncorrelated with the geometry channel — independence is the point, not raw accuracy. `neuronauts/meshing/` already renders what this needs. |
| **Tree-structured transformer** (Transformer-Grammars masks / tree positional encodings over contracted segments) | conditional on 057B | Only if 057B delivers labels at scale; otherwise it re-hits the wall. Beats the neural-emission grammar on held-out seams, axon side reported separately. |

LLM rule induction (`program.md`'s outer loop over an interpretable search
space of productions and emission features) stays gated on the grammar existing
as an executor — i.e. after E0–E2.

## 7. Baselines and the metric package

### 7.1 `neuronauts.eval` contract

```python
report = evaluate(pred, gt, geometry=None, *, unit="synapse"|"atom")
```

One call, one flat dict, grouped by prefix, printed as one table:

| Group | Metrics | Source implementation |
|---|---|---|
| `partition_*` | ARI, homogeneity, V-measure, n_clusters pred/true | treestitch |
| `pair_*` | same-neuron pair precision / recall / F1 **and cross-lineage split recall** (EXP-056's lesson: pair-F1 is class-imbalance driven) | benchmark.py + treestitch |
| `cable_*` | ERL, path-weighted precision/recall, total cable | benchmark.py |
| `connectome_*` | synapse **line-graph F1** (terminal), sampled-pair F1 | line_graph.py |
| `franken_*` | frankenmerge split rate (Bar 3), mixed-lineage roots resolved | benchmark.py |
| `abstain_*` | precision @ coverage curve, operating points at 0.95/0.99 | fingerprints evaluate + treestitch risk |
| `pop_*` | atoms, synapses, positives, candidates, GT coverage | new |

Conventions to fix on day one (each currently differs between implementations):
undefined precision when nothing is predicted is **NaN, not 1.0**; pairs are
counted over the evaluated unit only; ERL uses skeleton cable, not synapse
chords.

### 7.2 Two different things called "baselines"

Keep these separate; conflating them is how a weak result looks strong.

- **The ladder** (below) — floor, controls, and ceiling *on our substrate*. It
  answers "did this method do anything?" Every row is cheap and must appear in
  every results table.
- **Comparison methods** — published approaches (NEURD, RoboEM-style
  continuation, point-affinity clustering, multicut variants) reimplemented or
  run on *our* substrate under *our* protocol. This answers "is this
  competitive?" and it is the harder, more valuable axis.

`neuronauts/harness/baselines.py` currently implements the ladder plus two
learned scorers (logistic, gradient-boosted stumps); it was intended as the
comparison-methods surface. **Paused as of 2026-09-01** — resume by deciding
which published methods are in scope and what a faithful reimplementation on
the harness substrate requires. Until then, EXP-058 runs the ladder only, and
no result claims competitiveness with published work.

Literature numbers as *published* stay in their own table labelled "different
data, different protocol" and never share a row with ours; a comparison method
only earns a shared row once it has been run on our substrate.

### 7.2.1 The ladder

Every results table has these rows, in this order, on the same substrate:

1. **Untouched v117** — the do-nothing floor. Perfect precision by
   construction, zero recall; shows how much the segmentation already gets.
2. **Random merges at matched count** — controls for "any merging helps ARI".
3. **Proximity union-find** at 1/2/5 µm — the naive method everyone will ask about.
4. **Current best checkpoints** — `neuronauts_l2_partition.pt` and
   `cell_gnn_seg.pt`, run through the harness, so old numbers and new numbers
   finally sit in the same table.
5. **Oracle** — GT lineage as the solver; the ceiling given the candidate set.

Literature numbers (FFN, multicut, FlyWire) live in a separate table labelled
"different data, different protocol" and never share a row with ours.

### 7.3 Rules the runner enforces

- Criterion string and prerequisite EXP ids in the module header; fail closed.
- `results/EXP-xxx/provenance.json`: git SHA, dirty flag, config, substrate
  manifest hash, data versions (v117 / v1822), wall-clock, seeds.
- One ledger row appended to `RESULTS.md` per run; a run without a row does
  not exist.
- Means **and** per-atom quantiles; CIs where anything is trained.

## 8. Decisions needed from you

0. **Commit the untracked tree today.** Four whole packages (`harness/`,
   `metrics/`, `report/`, `meshing/`), 20+ scripts, 10 test files and this
   plan are untracked, and more than one session is editing them. Add
   `neuronauts.harness`, `neuronauts.metrics`, `neuronauts.report` to
   `[tool.setuptools] packages` (only `meshing` was added). Phase 0's tag means
   nothing until this lands.
1. **Attic vs delete for `morpho_grammar`.** I recommend attic with the §1.4
   note; the code is large but the ideas (tangent flow, caliber continuity)
   return as *features* in EXP-064 regardless.
2. **Paper drafts.** Mark superseded in place (recommended) or withdraw.
3. **Kimimaro scope** (open since the harness handoff): per-atom synapse-bbox
   skeletonization vs tile the 100 µm region once and slice by seg id. Affects
   `substrate/skeleton.py` in Phase 3 and EXP-064's tree-DNA feature.
4. **Python version.** `pyproject` says ≥3.10, `.venv` is 3.14. Pin one.
5. **Who owns what.** `docs/stage_ownership.md` has an empty Owner column; the
   stage layout in §3 is designed so it can be filled.

## 8b. Execution log

| Date | Phase | What landed |
|---|---|---|
| 2026-09-01 | 2 (partial) | **Attic created.** `neuronauts/morpho_grammar/` → `attic/morpho_grammar/` (26 engines) with a `__path__`-redirecting shim that keeps `neuronauts.morpho_grammar.*` importable under a `DeprecationWarning`. 34 benchmark scripts → `attic/benchmarks_semi_synthetic/`; `tests/test_morpho_grammar.py` → `attic/tests/`. Selection criterion was objective (`grep -l "treestitch.worldbuild"`), plus three scripts verified by hand: `benchmark_exp049` (unconditional fallback), `benchmark_exp050` (neurons generated from random walks), `benchmark_pcfg_infiller` (imports a class that does not exist). EXP-051–056 stayed in `scripts/`. All 34 moved scripts parse and every import still resolves. |
| 2026-09-01 | 0 | **Provenance headers.** Superseded banner on `EXPERIMENT_LOG.md`; new `docs/paper/README.md`; `% SUPERSEDED` stamps in all three LaTeX sources. `attic/README.md` carries the full audit. |
| 2026-09-01 | 0 | **Collection error fixed.** `tests/test_synapse_table_filter.py` tested `SynapseTable.filter_clutter` / `line_graph._clutter_keep_indices`, which live only on the unmerged branch `claude/remove-connectome-clutter-CkKag`; the test was merged without the implementation, so a bare `pytest` aborted with `ImportError`. Now skipped with that reason. **`pytest` collects cleanly for the first time: 1,494 tests, 0 errors.** |
| 2026-09-01 | 0 | **Decision 0 done — the untracked tree is committed** in five reviewed commits: `harness/` (substrate), `report/` (provenance + rendering), `meshing/` (Neuroglancer export), `metrics/` (the consolidation), and the two grammar docs. `neuronauts.{harness,report,meshing}` registered in `pyproject`. Suite after: **7 failed, 1,289 passed, 9 skipped** — the same 7 CellGNN failures as before the work started, so nothing regressed. |
| 2026-09-01 | 1 | **Metrics consolidated** (parallel session). `line_graph.py`, `treestitch/{partition,connectivity,calibration}.py` and `global_merge/eval/benchmark.py` are now delegating shims over `neuronauts/metrics/`, cross-checked numerically against the pre-consolidation implementations rather than assumed equivalent. 149 tests. `experiments/pcfg/conn_metric.py` still to migrate. |
| 2026-09-01 | 5 | **Runner landed** (`neuronauts/experiments/_runner.py`). Declares the bar before the run, blocks on unmet prerequisites without executing the body, checks inputs up front, records exceptions instead of swallowing them, stamps provenance, appends exactly one ledger row. A `Spec` with a blank criterion is rejected at construction. 15 tests, one per refusal. EXP-051–056 still to port onto it. |

Still open from Phases 0–2: committing the other three untracked packages (§8
decision 0), `neuronauts/legacy/` → attic (drags `topology_dataset.py`,
`shared_grammar_model.py`, `scripts/train.py` and 13 test files, so it belongs
with Phase 3's splits), and the ~12 outer-loop / viz scripts.

## 9. This week

Phase 0 and Phase 1 in parallel with EXP-057/058. The code for both
experiments already exists as modules — `harness.labels`, `spatial_split`,
`candidates`, `baselines`, `evaluation` — so the experiments are the *runs*:
commit the tree; add the superseded headers; wire the four legacy metric
callers to `metrics/`; run the GT overlay and baseline ladder on tier ≥10
while the tier ≥1 fetch finishes. From then on every number in the repo has a
floor and a ceiling next to it.
