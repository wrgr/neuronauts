# Evaluation metrics: one home, `neuronauts/metrics/`

Before this consolidation, the project had four independent "metrics homes"
that didn't import each other: `neuronauts/line_graph.py`,
`neuronauts/global_merge/eval/benchmark.py`, `treestitch/metrics.py` /
`treestitch/partition.py` / `treestitch/connectivity.py` /
`treestitch/calibration.py`, plus a long tail of one-off pairwise-P/R, ARI,
AUC and completeness implementations across `scripts/`, `experiments/pcfg/`,
and `neuronauts/*`. Several disagreed on edge cases (empty-set convention,
ignore-label convention, int64 overflow on real ~1e18 root ids) without
anyone noticing, because nothing cross-checked them against each other.

`neuronauts/metrics/` is now the one place the math lives. Everything else
that used to define a metric either imports from there or is a thin
backward-compatible shim that delegates to it — see "What moved" below.

## Package layout

| Module | What it computes |
|---|---|
| `neuronauts/metrics/_core.py` | Shared primitives: `safe_div`/`prf1` (NaN-by-default ratios), `align_labels` (the `ignore`/`pred_ignore` conventions), `contingency`/`pair_confusion`/`weighted_pair_confusion` (one sparse contingency table, O(N log N)) |
| `neuronauts/metrics/partition.py` | ARI, pairwise merge P/R/F1, homogeneity/completeness/V-measure, variation of information (`vi_split`/`vi_merge`), cluster purity, expected run length (ERL) |
| `neuronauts/metrics/edges.py` | Candidate-edge merge/split confusion (`edge_merge_metrics`) — the "Bar 1/2" metric, plus edge-level frankenmerge split recall ("Bar 3") |
| `neuronauts/metrics/frankenmerge.py` | Item/parent-level frankenmerge separation (`frankenmerge_metrics`) — global-merge's `fk_split` and treestitch's `fk_separation` computed together so they can't drift apart |
| `neuronauts/metrics/completeness.py` | Fragment completeness (needs-no-edit) P/R/F1 |
| `neuronauts/metrics/connectome.py` | Neuron-to-neuron connectome edge F1, single- and dual-side protocols |
| `neuronauts/metrics/line_graph.py` | Synapse line-graph F1, all four variants (`pre_only`/`or_metric`/`post_only`/`and_metric`) |
| `neuronauts/metrics/ranking.py` | ROC-AUC, average precision (both sklearn-free, rank-based), threshold sweeps, `edit_metrics_vs_baseline` (the do-nothing-relative guardrail) |
| `neuronauts/metrics/calibration.py` | Reliability bins, ECE, Brier score |
| `neuronauts/metrics/suite.py` | `evaluate_partition_suite` — one call that runs every applicable block based on what inputs you pass |
| `neuronauts/metrics/report.py` | `KEY_DOCS`/`describe_key` (the key registry), `format_metrics`, `metrics_to_json` |

Import everything from the top level:

```python
from neuronauts.metrics import evaluate_partition_suite, format_metrics

m = evaluate_partition_suite(
    pred, true,                       # per-item predicted cluster / ground truth
    weights=cable_um,                 # optional: cable-weighted pairs + ERL
    src=edge_src, dst=edge_dst,       # optional: candidate-edge merge/split block
    fragment_id=atom_id,              # optional: frankenmerge + naive baseline
    root_label_map=root_label_map,    # optional: completeness (needs fragment_id)
    true_post=post_root_id,           # optional: connectome + line-graph blocks
)
print(format_metrics(m, title="PCFG vs learned, val region"))
```

## Conventions (read this before adding a new metric)

- **Inputs are aligned per-item arrays.** `pred`, `true`, optional `weights`,
  any dtype `np.unique` accepts (including real ~8.6e17 CAVE root ids — every
  contingency table remaps to compact indices first, so nothing overflows
  int64). Dict-shaped `{item: cluster}` inputs go through
  `neuronauts.metrics.labels_from_maps`.
- **`ignore` marks unknown ground truth** (default `0`, the repo-wide
  convention); those items are dropped everywhere. **`pred_ignore` marks an
  abstained prediction**; that item is kept but relabelled as its own
  singleton, so two abstentions never count as "merged" with each other.
- **Undefined ratios are NaN**, not 0.0 or 1.0. `0/0` genuinely is undefined —
  see `CLAUDE.md` §"assume the bug is yours" for why silently picking a
  convention here has bitten this project before. Pass `undefined=` to
  override when a specific historical convention must be preserved (the
  backward-compat shims do this explicitly — see below).
- **Everything is O(N log N)** via one sparse contingency table
  (`neuronauts.metrics.contingency`); nothing materializes an N×N matrix or an
  explicit pair list. `neuronauts/coassign/cluster.py`'s dense boolean-matrix
  pairwise P/R and `global_merge/eval/benchmark.py`'s old O(N²) path-length
  loop are gone — the new `wpair_*`/`erl` fields in `partition_metrics`
  replace both and are verified numerically identical (see the shim's
  docstring and `tests/test_metrics_partition.py`).
- **Every key `evaluate_partition_suite` can emit is documented** in
  `neuronauts.metrics.report.KEY_DOCS` (exact keys) or `PATTERN_DOCS` (the
  line-graph variant families). `tests/test_metrics_report.py::
  test_full_suite_output_has_no_undocumented_keys` fails the build if a
  metric starts returning an undocumented key — keep the registry in sync
  when you add one.

## What moved (old location → new implementation)

| Old location | Status | New home |
|---|---|---|
| `neuronauts/line_graph.py` | Rewritten as a shim (re-exports everything; keeps the 3 `ConnectivityGraph`-based legacy entry points: `evaluate`, `evaluate_sampled`, `build_estimated_line_graph`) | `neuronauts.metrics.line_graph` |
| `treestitch/connectivity.py` | Rewritten as a shim (`Region`-object calls adapted to the array API); `_prf1`, `_match_clusters_to_neurons` kept as aliases | `neuronauts.metrics.connectome` |
| `treestitch/partition.py` (`fragment_completeness`, `pred_fragment_completeness`, `completeness_metrics`) | Bodies replaced with one-line delegating calls | `neuronauts.metrics.completeness` |
| `treestitch/partition.py` (`evaluate_partition`, `merge_metrics`) | Unchanged — already thin re-export shims into `neuronauts.assemble.partition_gnn` / `edge_partition`, which independently agree with the new package (see cross-check tests) and were left alone rather than risking their existing test coverage for no functional gain | `neuronauts.assemble.partition_gnn.evaluate_partition_ari`, `neuronauts.assemble.edge_partition.edge_merge_metrics` |
| `treestitch/calibration.py` (`expected_calibration_error`) | Body replaced with a delegating call; `fit_temperature`/`calibrated_obs_confidence`/`reliability_diagram` (torch-model-driving code) stay here since they don't belong in a model-free metrics package | `neuronauts.metrics.calibration` |
| `neuronauts/global_merge/eval/benchmark.py` | Rewritten as a shim: `adjusted_rand_index`, `compute_pairwise_partition_metrics`, `evaluate_frankenmerge_split_rate`, `compute_path_length_metrics` all preserved bit-for-bit (numerically verified against the originals with a 500-trial randomized fuzz test — see PR notes). One quirk preserved deliberately: `evaluate_frankenmerge_split_rate` gives every fragment missing from `pred_map` a single shared placeholder cluster (matching the old `dict.get()` semantics), not a singleton — a genuinely different, intentional convention from `pred_ignore` | `neuronauts.metrics.partition`, `neuronauts.metrics.frankenmerge` |
| `neuronauts/global_merge/eval/` had no `__init__.py` | Fixed (was a namespace package by accident) | — |
| `neuronauts/coassign/cluster.py` (`pairwise_precision_recall`) | Body replaced; the dense N×N boolean pair matrix this built is gone. Verified numerically identical (300-trial fuzz test) | `neuronauts.metrics.partition_metrics` |
| `neuronauts/atomization.py` (`pair_counts`, `metrics_from_counts`) | Bodies replaced with delegating calls. Verified numerically identical (300-trial fuzz test) | `neuronauts.metrics._core` |
| `treestitch/stitch.py` (`pairwise_merge_metrics`) | Body replaced. One **deliberate behaviour change**: when precision and recall are both legitimately 0.0 (not undefined — there were predictions/truths, just no correct ones), `merge_f1` is now 0.0, matching every other F1 in the codebase, instead of the old code's NaN (an `(prec + rec) > 0` guard that conflated "genuinely zero" with "undefined"). No test relied on the old value | `neuronauts.metrics.partition_metrics` |
| `neuronauts/assemble/partition_gnn.py` (`_adjusted_rand_score_np`, `_homogeneity_completeness_np`), `neuronauts/assemble/edge_partition.py` (`edge_merge_metrics`) | **Not touched.** Already the de-facto standard, heavily tested (Bar 1/2/3 viability gates), and independently verified to agree with `neuronauts.metrics` (`tests/test_metrics_partition.py`, `tests/test_metrics_edges.py`) | left in place |
| `treestitch/atomize.py` (`frankenmerge_separation`) | **Not touched** (still the production entry point for treestitch); cross-checked against `neuronauts.metrics.frankenmerge_metrics` for numerical agreement | left in place |

### Not consolidated (out of scope)

- **`experiments/pcfg/conn_metric.py`** (`SideTable`-based pre/post/pooled
  connectivity accuracy) is a genuinely different protocol — it operates on
  synapse *sides* joined across two segmentation versions via a `SideTable`,
  not a generic partition — so it wasn't folded into
  `neuronauts.metrics.connectome`. (A pre-existing, unrelated break -- this
  file, `attic/pcfg_one_offs/group_eval.py`, and 21 other files under
  `experiments/pcfg/` still imported from the pre-rename package name
  `experiments.pcfg_synapse_partitions` -- was fixed in the 2026-09 code-health
  pass by rewriting those imports to `experiments.pcfg`.)
- **Purity plotting** (`neuronauts/viz.py::plot_scaffold_purity`,
  `scripts/scaffold_census.py`, `scripts/tier_census.py`) — these compute
  majority-label purity inline for a plot, same math as the new
  `neuronauts.metrics.partition.cluster_purity`, but weren't rewired since
  they're presentation code, not a metrics call site with its own tests.
- **Soft-F1 training surrogates** (`neuronauts/shared_grammar_model.py`) —
  intentionally differentiable approximations used inside a training loop,
  not evaluation metrics; out of scope by design.
- The many `scripts/benchmark_exp0*.py` one-off `eval_lg`/
  `evaluate_model_pipeline` closures that duplicate the line-graph adapter or
  a union-find + full-metric battery — these are semi-synthetic benchmark
  scripts, not part of the metrics package's own call surface. They still
  work unchanged (they call into `neuronauts.line_graph` /
  `neuronauts.global_merge.eval.benchmark`, both now shims), but weren't
  individually rewritten to call `neuronauts.metrics` directly.

## Testing

`tests/test_metrics_*.py` (11 files, ~150 tests) cover every module:
edge cases (empty input, `n<2`, all-one-cluster, all-singletons, real root-id
magnitudes), agreement with scikit-learn where one exists (`ari`,
`homogeneity_completeness_v_measure`, `roc_auc_score`,
`average_precision_score` — skipped if `sklearn` isn't installed), and a
direct numerical cross-check against **every** pre-consolidation
implementation it replaces (`neuronauts.assemble.partition_gnn`,
`neuronauts.assemble.edge_partition`, `treestitch.connectivity`,
`treestitch.atomize`, `neuronauts.line_graph`, `treestitch.calibration`).

The full non-legacy suite (`pytest -q -m 'not legacy'`, ~1300 tests) was run
before and after this consolidation; every failure traced back to causes
unrelated to this change (a pre-existing `scripts/train.py` bug that calls
`evaluate()` with a keyword argument it never accepted, one flaky
`torch`-training-convergence test, and one transient collection error from
concurrent file operations in another session working in the same tree —
none reproduced in isolation).
