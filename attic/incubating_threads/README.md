# incubating_threads — threads that never put a result on the board

**Era.** March–June 2026, alongside the box-local CellGNN pipeline.

## `low_res_segmentation/`

Reconstructing neurons at coarse resolution (128×128×120 nm) from EM volumes,
growing per-cell segments from membrane guidance and synapse positions instead of
full-resolution tracing.

**Why it is here.** Its own README says "Runs on synthetic connectomes; real-CAVE
scaffold **tested** against Minnie65 soma boxes" — tested, never *scored*. The
thread set itself a graduation bar (match the CellGNN held-out F1 of 0.272) and
never reported a number against it. The branch carrying it,
`claude/low-res-segmentation-pipeline-fwIHN`, has been stale since 2026-04-07 with
pull request #9 still open. `docs/consolidation_plan.md` §4.2 marks it ATTIC and
`docs/threads/experiment_survey.md` grades it "not evidence / no result yet."

**Its tests came too.** `tests/test_low_res_segmentation.py` and
`tests/test_high_precision_segmentation.py` stayed in `tests/` and now import from
`attic.incubating_threads.low_res_segmentation`; they still run in the default
suite. The two `test_*.py` files *inside* this directory are the thread's own
scaffolding checks, never collected (`testpaths = ["tests"]`).

**What replaced it.** Nothing directly. The question it was asking — can coarse
geometry alone group cable into cells — was answered negatively at a different
scale by EXP-058 (proximity clustering is indistinguishable from random) and
EXP-072 (widening the object set makes chained recall worse).

**Route back.** Score it against its own stated bar on the harness substrate. Any
number it produces needs a registry entry to count.

## What deliberately did *not* move here

- **`experiments/soma_graph/`** stayed in `experiments/`. It is Phase 3 of
  `docs/roadmap_global_assembly.md`, which names `build_graph.py:97` as the exact
  line to change next. A thread the canonical roadmap tells you to build is not
  an archive candidate, however thin its current evidence.
- **`experiments/root_neighborhood/`** stayed in `experiments/`. It is reachable
  from a live, tested command line —
  `scripts/train.py build-dataset --strategy proofread-core` imports
  `build_root_neighborhood_cache` from it — and it is the cache strategy that
  surfaced real edit signal where random boxes showed `0 merge pairs, 0 split
  pairs`.
