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
```

The attic is excluded from `pytest` (`testpaths = ["tests"]`). Any result
produced from here belongs in `results/synthetic/`, never in `results/EXP-*`.
