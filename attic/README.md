# attic — retired pathways

Code here is **kept for history, excluded from the default test run, and not
part of the active pipeline.** Nothing is deleted: git history is preserved
(everything arrived via `git mv`), and import shims keep old call sites
working with a `DeprecationWarning`.

> **The rule for coming back.** A pathway leaves the attic by *passing a
> registered experiment*, not by being imported again. See
> [`docs/consolidation_plan.md`](../docs/consolidation_plan.md) §6 for the
> experiment that governs each one.

## Contents

| Path | What it is | Why it is here | Route back |
|---|---|---|---|
| [`morpho_grammar/`](morpho_grammar/) | 26 "engines": dual-engine infillers, SANTIAGO grammars, MCTS handshake, Cajal geodesic tracers, global hypothesis search, Hungarian assembler, NEURD/autoproof baselines | **No engine loads a trained checkpoint.** 15 of 26 draw random numbers at runtime; none contains a `torch.load` or a `.pt` path. Every benchmark that consumed them ran on synthetic damage (below). So the reported accuracies are those of randomly initialised models on fabricated errors. | **EXP-069** — same engine, real harness substrate, EXP-064 fixed-panel protocol, trained grammar. Individual ideas (tangent flow, caliber continuity, conservation priors) are expected back sooner as *features* in the EXP-064 scorer bake-off. |
| [`benchmarks_semi_synthetic/`](benchmarks_semi_synthetic/) | 34 benchmark scripts: `benchmark_exp021`–`exp050` plus 6 unnumbered siblings | Their test worlds are not real. See the audit below. | **EXP-058** replaces them with a real baseline ladder on one substrate. |
| [`tests/`](tests/) | `test_morpho_grammar.py` | Tests attic code; moved with it so the default suite stays clean. | Moves back with its subject. |
| [`outer_loop_and_viz/`](outer_loop_and_viz/) | 9 outer-loop scripts (`watch_and_eval.sh`, `eval_at_t099.sh`, `eval_path_models.sh`, `run_feature_ablation.sh`, `run_k_ablation.sh`, `run_timing_pipeline.sh`, `render_whitepaper_pdf.sh`, `generate_dashboard.py`, `export_viz_data.py`); `viz_pipeline.py` (was `tools/`); the `dashboard/` Flask + Streamlit app (5 files); two stale synthetic visualizer artifacts in `viz_synthetic_artifacts/` | Shell loops that train/evaluate the box-local CellGNN track outside any tracked pipeline, plus viz tooling built on top of the synthetic-frankenmerge world (§1.4). See the audits below. | No registered route back. `neuronauts/meshing/` (§3) is the live replacement for `dashboard/`'s Neuroglancer views; a dashboard/viz feature returns only as a `report/` or `meshing/` module built against real data. |
| [`superseded_modules/`](superseded_modules/) | `cave_synapse_counts_v1412.py`, `cave_synapse_degrees_v1412.py` | Predecessors that a later consolidation replaced and then left in the tree. `neuronauts/cave_synapse.py`'s own docstring names them as the two of three single-purpose scripts it consolidated; nothing imports them and no test imports them. See the audit below. | Not a research pathway, so no experiment gates them. They come back only if `neuronauts/cave_synapse.py` turns out to have lost a behaviour they had — and the equivalence check below is the thing to re-run first. |

## The audit that put the benchmarks here

Verified by direct inspection of the scripts on 2026-09-01, not inferred. The
selection criterion was objective — `grep -l "treestitch.worldbuild"` — plus
three scripts checked individually:

| Finding | Scripts |
|---|---|
| Import `treestitch.worldbuild.frankenmerge_adjacent` and apply it at **45–50%** to real proofread skeletons that were themselves **synthetically cut** | 32 |
| `benchmark_exp049_dense_subvolume.py` calls `generate_dense_subvolume_fallback(...)` **unconditionally** at line 279 — the comment says "Attempt CAVE fetch or dense fallback" but there is no fetch attempt | 1 |
| `benchmark_exp050_interneuron_stratified.py` **generates the neurons themselves** from random walks (`rng.normal` steps building dendrites and axons) — not real skeletons at all | 1 |
| `benchmark_pcfg_infiller.py` imports `TreeGrammarInfiller`, which **does not exist** (the module defines `EnhancedTreeGrammarInfiller`); it also fabricates synapses and partner ids | 1 |

`benchmark_exp035_restored_dual_engine.py` — the "restored SOTA" quoted in
`docs/paper/walkthrough.md` — is
`frankenmerge_adjacent(pieces_rec, 0.45, rng, radius_nm=9500.0)` on line 158.

This was found independently once before: `results/exp051_evaluation.md`
audited EXP-049 and reached the same conclusion, including that the SANTIAGO
infiller "initializes random matrices at runtime rather than loading a trained
real-data grammar checkpoint."

**Consequence for published claims.** The SOTA comparison table in
`EXPERIMENT_LOG.md` §3 places these numbers beside FFN (Januszewski 2018),
Janelia multicut (Beier 2017), DeepMulticut (Li 2024) and FlyWire (Dorkenwald
2024) — all measured on real data. That comparison is not sound. Both paper
drafts inherit it; see [`docs/paper/README.md`](../docs/paper/README.md).

## The outer-loop / dashboard / viz move (2026-09-01)

Verified individually, per `docs/consolidation_plan.md` §4.3's "~12 outer-loop
/ viz scripts" line. `scripts/codex_optimize.py` — named in that line and in
`program.md`/`docs/model.md`/`docs/roadmap_global_assembly.md`/
`docs/whitepaper.md` as the outer optimizer loop — was already removed from
the tree in commit `012bb50e6` ("Remove dead modules..."); there was nothing
left to move, and those four docs' references to it were already dangling
before this change and are unrelated to it.

| File | What it does | Verified before moving |
|---|---|---|
| `watch_and_eval.sh` | Polls `models/` for new `cell_gnn_path16*.pt` checkpoints and sweeps eval thresholds as they appear | No importer/caller anywhere (`grep -rn watch_and_eval`); only self-references in its own usage comment |
| `eval_at_t099.sh`, `run_k_ablation.sh`, `run_feature_ablation.sh` | Outer `for`-loops that call `scripts/train.py train-cell-gnn`/`evaluate` to produce the K-hop and per-feature ablation checkpoints for the box-local CellGNN (F1 0.27 ceiling, `docs/architecture.md`) | Referenced only as historical "Reproduce" commands in `docs/ablation_results.md` and a checked-off TODO in `docs/TODO.md`; no `.py`, `Makefile`, or `pyproject.toml` reference. `docs/ablation_results.md`'s three commands were updated in place to the new `attic/outer_loop_and_viz/` path so they still run; `docs/TODO.md` is separately mid-move to `docs/archive/2026-09/` by another in-flight change and was left untouched |
| `eval_path_models.sh` | Same outer-loop pattern for the Option-2 path-embedding checkpoints | No references anywhere outside itself |
| `run_timing_pipeline.sh` | End-to-end timing harness that shells out through all of `scripts/train.py`'s subcommands (build-dataset → train-cell-gnn → evaluate) | Standalone; hardcodes `PROJECT_ROOT`; nothing sources or calls it |
| `render_whitepaper_pdf.sh` | `pandoc`/`xelatex` wrapper that renders `docs/whitepaper.md` to PDF | No reference anywhere in `*.py`/`*.md`/`*.sh`/`Makefile`/`pyproject.toml` besides the plan line that named it |
| `generate_dashboard.py` | Builds a standalone Three.js HTML dashboard from `neuronauts.global_merge.represent.{vicreg_gnn,asymmetric_relational_gnn,local_em_verifier}` — the ATTIC-verdict "synthetic series" `represent/` modules (§4.1) — with hardcoded KPI numbers (e.g. `kpi-erl` = "3,595.4 μm") baked into the HTML string, not computed | Not imported anywhere; run only via `if __name__ == "__main__"` |
| `export_viz_data.py` | Confirmed by reading the file: `_split_skeleton_n_pieces` cuts real proofread skeletons into synthetic pieces (line 33), `rng.choice`/`rng.integers` fabricate `syn_coords`/`syn_types`/`syn_partners` with no real synapse fetch (lines 38–44), and `treestitch.worldbuild.frankenmerge_adjacent` injects frankenmerges at 35% (line 58) before writing `viz/sample_connectome_viz.json`. Matches `results/exp051_evaluation.md`'s independent finding verbatim | No importer; the two files it produces are moved with it (below) |
| `dashboard/` (`app.py`, `neuroglancer_export.py`, `results_explorer.py`, `streamlit_app.py`, `templates/index.html`) | Flask "v2 performance dashboard" (`app.py`, keyed to the v1 `run_research_cycle` pipeline) plus two Streamlit result-bundle viewers | `grep -rn "import dashboard\|from dashboard"` outside the directory found only its own internal `results_explorer.py → dashboard.neuroglancer_export` import. `neuronauts/report/ngl.py`'s docstring names `dashboard/neuroglancer_export.py` only as a comparison of voxel-grid conventions — not an import — and `neuronauts/report/` is explicitly out of scope for this move, so that stale path in the comment was left as-is. `scripts/spatial_variance.py`'s `--save-bundle` help string pointed at the old `dashboard/` path; updated to the new `attic/outer_loop_and_viz/dashboard/` path |
| `viz_synthetic_artifacts/connectome_visualizer.html`, `viz_synthetic_artifacts/sample_connectome_viz.json` | The Three.js viewer and its companion data file that `export_viz_data.py` produces | Not `viz/` itself (below) — just these two generated files, which are the synthetic output `results/exp051_evaluation.md` already flagged ("contains synthetic IDs and zero links") |

**`viz/` the directory was not moved.** `neuronauts/meshing/__init__.py`'s
own doctring example writes to `viz/mesh/demo`, `scripts/mesh_results.py`
defaults `--out` to `viz/mesh/demo` and documents `viz/mesh/` throughout, and
`docs/meshing.md` gives the same paths — `neuronauts/meshing/` (kept; §3) is
the live, current writer of that directory. Only the two stale files sitting
in it were moved; the empty directory stays as `meshing`'s default output
root.

**`treestitch/ngl_export.py` was not moved** despite being the second half of
`neuronauts/report/ngl.py`'s "two state builders disagree" comment: it is
imported by `treestitch/stitch_viz.py`, `scripts/out_of_column_eval.py`, and
`neuronauts/meshing/bundle.py` — the last one a KEEP package — so it is a live
dependency, not a retired one.

**`neuronauts/viz.py`** (a single module, unrelated to the top-level `viz/`
directory despite the name collision) was not touched: it is imported by
`tests/test_viz_smoke.py`, `tests/test_scaffold.py`, and
`attic/outer_loop_and_viz/dashboard/app.py` (now attic code depending on live
code, which is fine — the dependency does not run the other way).

## The 2026-09-02 pass — three files nothing referenced

Two moves, both chosen by a criterion that can be re-run rather than by
judgement, and both verified against the same test baseline
(**1 failed, 1,569 passed, 2 skipped** — the one failure is
`tests/test_multitask_convergence.py::test_loss_decreases_over_steps`, a
pre-existing legacy-model regression). The suite reported exactly those counts
before the pass, after the first move, and after the second.

### `tools/viz_pipeline.py` → `outer_loop_and_viz/viz_pipeline.py`

A 588-line Plotly HTML viewer with three subcommands — `path-encoder`,
`grammar`, `cell-gnn` — over `neuronauts.dataset_builder.BoxCache`,
`neuronauts.shared_grammar_model`, and `neuronauts.cell_graph`. It is the same
box-local CellGNN / path-encoder track, and the same kind of artifact, as the
`generate_dashboard.py` and `export_viz_data.py` already here; `tools/` does
not appear anywhere in `docs/consolidation_plan.md` §3's target layout.

Criterion, re-runnable: `git grep -n viz_pipeline` returns **nothing** outside
the file itself — no importer, no test, no doc, no `Makefile`, no
`pyproject.toml` entry. `tools/` held this one file and is now empty. Two of
the three checkpoints its usage block names (`models/path_encoder_v3_ep10.pt`,
`models/cell_gnn_path16_ep50.pt`) are not in `models/`.

### `neuronauts/cave_synapse_{counts,degrees}_v1412.py` → `superseded_modules/`

These are not a research pathway; they are leftovers of a completed
consolidation. `neuronauts/cave_synapse.py` opens with:

> Consolidates three former single-purpose scripts
> (`cave_synapse_degrees_v1412`, `cave_synapse_counts_v1412`,
> `cave_synapse_degrees_1078_to_1412`) into one module with three flows

The third was already gone; these two were still sitting in the package.

Criterion, and it was checked rather than taken from the docstring:

| Check | Result |
|---|---|
| Anything imports them? | No. `git grep` finds only two *docstring* mentions: `neuronauts/cave_synapse.py`'s header (above) and two test docstrings in `tests/test_cave_synapse.py` that name the old modules while importing the new one. |
| Any test imports them? | No. `tests/test_cave_synapse.py` imports `neuronauts.cave_synapse` only. |
| Named in any doc as current? | No. |
| Does the successor actually reproduce them? | **Yes, checked numerically.** Running `tests/test_cave_synapse.py`'s own fixtures through both sides, `cave_synapse.build_degree_root_table` and `cave_synapse_degrees_v1412.build_root_table` returned identical DataFrames, as did `build_counts_root_table` vs `cave_synapse_counts_v1412.build_root_table` (3 fixtures, `DataFrame.equals` → True on all). The successor's function surface is the union of both predecessors', with each `build_root_table` renamed to `build_degree_root_table` / `build_counts_root_table`. |

**No import shim was added, for either move**, because there is nothing to
shim: no call site exists to keep working. That is a deliberate difference
from `morpho_grammar/`, whose shim exists because 26 benchmark scripts still
import it.

**Known stale reference, left alone.** `tests/test_cave_synapse.py`'s two class
docstrings still say "Test `build_root_table` in `cave_synapse_counts_v1412`"
/ "`…_degrees_v1412`". They were already stale before this move — the test has
imported the consolidated module all along — so they are recorded here rather
than edited, to keep this pass to moves and documentation.

## What stayed out of the attic

`scripts/benchmark_exp051`–`exp056` are the **fail-closed real-data series**
and remain in `scripts/`. They fetch real synapses, resolve endpoint
supervoxels at exact v117/v1412 timestamps, refuse to run without a real
checkpoint and a real merge-pair signal, and report prerequisite failures
instead of numbers (EXP-053B, 054, 055 all correctly refused to report). They
are the template for the experiment runner.

`benchmark_boundary_search.py` and `benchmark_synapse_membership_box.py` use no
synthetic world and stayed put.

## Running something from the attic

```bash
# scripts are self-contained; run them from the repo root
uv run python attic/benchmarks_semi_synthetic/benchmark_exp035_restored_dual_engine.py

# imports still work, with a warning
python -c "from neuronauts.morpho_grammar.santiago_v2_grammar import *"

# outer-loop scripts run unchanged from their new path
bash attic/outer_loop_and_viz/run_k_ablation.sh
streamlit run attic/outer_loop_and_viz/dashboard/results_explorer.py

# the viz_pipeline viewer still runs from here (it imports live code, not the
# other way round), though two of the three checkpoints it wants are gone
python attic/outer_loop_and_viz/viz_pipeline.py grammar \
    --checkpoint models/grammar_cave_real_50.pt --cache-dir data/boxes_30um

# superseded_modules/ have no importers, so run them as scripts if ever needed
python attic/superseded_modules/cave_synapse_degrees_v1412.py --help
```

The attic is excluded from `pytest` (`testpaths = ["tests"]`). Any result
produced from here belongs in `results/synthetic/`, never in `results/EXP-*`.

## Bringing something back

`git mv` it to the path it came from — the moves above were all `git mv`, so
`git log --follow <attic path>` shows the original location and the full
history is intact:

```bash
git log --follow --oneline attic/superseded_modules/cave_synapse_counts_v1412.py
git mv attic/superseded_modules/cave_synapse_counts_v1412.py neuronauts/
```

Then run `.venv/bin/python -m pytest tests/ -q` and check the counts against
the baseline recorded above. For a *research* pathway (`morpho_grammar/`,
`benchmarks_semi_synthetic/`) the mechanical move is not the bar — the bar is
the registered experiment named in the Contents table.
