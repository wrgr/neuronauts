# Semi-synthetic benchmarks — not evidence about real segmentation

**Every script in this directory scores a method against artificially damaged
skeletons, not against v117 segmentation.** 32 of the 34 import the synthetic
world builder directly. A "split" here is an intact skeleton cut in software, so
the two halves still carry matching geometry, matching caliber and a matching
tangent at the cut. Real v117 fragments do not.

That difference is the whole difficulty of the program. Numbers from this
directory are therefore **not comparable** to anything in `results/`, and the
headline figures they produced — including the "~85–87% pairwise merge accuracy"
quoted for a long time in `docs/threads/grammar.md` — did not survive contact
with real data:

| question | on synthetic damage | on real v117 |
|---|---|---|
| pairwise merge proposal | ~85–87% | ~0.09% precision (EXP-060, 060B, 061, 070) |
| candidate generation after widening | — | collapses (EXP-072) |

What *has* held up on real data lives in `results/`: EXP-063 detects a
frankenmerge at held-out AUC 0.958, EXP-071 explains why the synapse-anchored
population omits connective cable, and EXP-075/076 argue that local geometry
cannot supply a grower's stop rule.

> Update, later on 2026-09-02: EXP-076 **withdrew EXP-075's 0.304 headline as
> unverified** (a box-placement error on the "already whole" panels) while
> reaching the same negative on the seed's own end shape (AUC 0.476, matched for
> distality). The direction above stands; the specific number 0.304 should not be
> quoted. See `results/EXP-075/evaluation.md`'s correction box.

Keep these scripts for their mechanics — the damage model, the metric
implementations and the harness plumbing are all reusable. Do not quote their
scores. See `docs/threads/experiment_survey.md` for the evidence grade of every
experiment in the repo.

---

## The 2026-09-02 addition — six more scripts that build their own world

The original 34 arrived by one re-runnable criterion (`grep -l
"treestitch.worldbuild"`). These six arrived by the same criterion widened to
catch the scripts that cut skeletons without going through `worldbuild`:

> **the script constructs its own test world** — by splitting real skeletons into
> pieces, by fusing pieces of different neurons, or by fabricating fragments
> outright — **and nothing in the repo imports it.**

Both halves were checked before each move, not inferred:

| File (was `scripts/`) | How it builds its world | Who referenced it |
|---|---|---|
| `real_franken_partition.py` | `_split_skeleton_n_pieces` → `frankenmerge_adjacent(…, 0.25)` → `build_world_from_pieces` (lines 64, 126, 130) | One line in `docs/archive/2026-09/STATUS.md`. No importer. |
| `real_skeleton_partition.py` | `split_skeleton_n_pieces` unconditionally (lines 164–165, 223); produced the "20 real neurons × 3 pieces" ARI number in the archived `STATUS.md` | Two archived docs, plus a *comment* in `scripts/coassign_demo.py`. No importer. |
| `test_global_merge_franken.py` | `_split_skeleton_n_pieces` then `frankenmerge_adjacent` (lines 51, 72). Not a test despite the name — `pytest` never collected it (`testpaths = ["tests"]`) | Nothing, anywhere. |
| `optimize_tree_stitch.py` | Random hyperparameter search over `load_minnie65_world(..., n_pieces=3)` — real skeletons, software cuts | Nothing, anywhere. |
| `run_global_merge.py` | `run_synthetic_demo` fabricates two-vertex fragments in a loop (105 lines, no real-data path; `--demo` defaults to `True` and is the only mode). It also prints `Multi-Soma Violations: 0` as a string literal, not a measurement | Nothing, anywhere. |
| `sota_benchmark.py` | "all data is generated synthetically" (its own docstring, line 5); `_make_benchmark_graph` builds separable graphs with distinct DNA per object | Nothing, anywhere. |

Re-run the reference half of the criterion yourself:

```bash
git grep -w sota_benchmark            # → only its own file
git grep -w real_skeleton_partition   # → two archived docs and one comment
```

**What deliberately did not move, and why**, so the line is reproducible rather
than a judgement call:

- `attic/prior_results/real_lineage_partition.py`, `real_region_partition.py` — same family
  of names, but they run on **real v117 roots with no synthetic cut**; they call
  no splitter. They back the treestitch results and the plan marks them KEEP.
- `attic/prior_results/two_level_stitch.py`, `compare_partition_methods.py` — reach
  `treestitch.synthetic` only behind an opt-in flag; the default path is real,
  and `two_level_stitch.py` backs the tile-stitching result (ΔARI +0.10).
- `scripts/coassign_demo.py` — is a synthetic-split demo, but `INTRO.md` and
  `neuronauts/coassign/README.md` cite it as the runnable demo of a KEEP package,
  so moving it would break live documentation.
- `scripts/benchmark_exp051`–`056` — the fail-closed **real**-data series. See
  "What stayed out of the attic" in [`../README.md`](../README.md).
- The tree-DNA ablations (`ablate_dna.py`, `half_split_ablation.py`,
  `multi_fragment_ablation.py`, `half_synapse_ablation.py`,
  `global_gnn_ablation.py`) — they split skeletons, but that split *is* the
  experiment (half- versus quarter-skeleton identity), it is measured on real
  skeletons, and the survey grades the result REAL.

Test baseline, measured immediately before the move and required to be unchanged
after it: **1 failed, 1,579 passed, 2 skipped** (`.venv/bin/python -m pytest
tests/ -q --continue-on-collection-errors`, 878s). The one failure is
`tests/test_multitask_convergence.py::test_loss_decreases_over_steps`, the same
pre-existing legacy-model regression recorded for the earlier passes.
