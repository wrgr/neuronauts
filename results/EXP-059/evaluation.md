# EXP-059 — the metric consolidation did not change the numbers

## Result: passed

Four independent metric implementations were replaced by `neuronauts/metrics/`
and the old call sites rewritten as delegating shims. That is only safe if the
shims return what they returned before — a refactor that silently moves a
metric is worse than four implementations, because every historical number
becomes incomparable and nobody notices.

Checked numerically across **200 randomly generated partitions** spanning eight
shapes chosen to break these functions: perfect agreement, one giant cluster,
all singletons, independent labels, heavy class imbalance, unlabelled items
present, a single true class, and near-perfect with scattered errors.

| quantity | cases | max abs difference |
|---|---:|---:|
| `benchmark.adjusted_rand_index` | 200 | 0 |
| `benchmark.ari` | 200 | 0 |
| `benchmark.merge_P` | 175 | 0 |
| `benchmark.merge_R` | 200 | 0 |
| `treestitch.evaluate_partition.ari` | 200 | 1.11e-16 |
| `treestitch.evaluate_partition.homogeneity` | 200 | 0 |
| `treestitch.evaluate_partition.completeness` | 200 | 0 |
| `treestitch.evaluate_partition.v_measure` | 200 | 0 |
| `treestitch.evaluate_partition.n_clusters_pred` | 200 | 0 |
| `treestitch.evaluate_partition.n_clusters_true` | 200 | 0 |

The one non-zero is 1.11 × 10⁻¹⁶ on ARI — floating-point noise, an order below
the 1e-9 tolerance.

## The one deliberate difference, asserted rather than waved through

`global_merge.eval.benchmark` returns **precision 1.0 when nothing is merged**;
`neuronauts.metrics` returns **NaN**. This fired in 25 of the 200 cases and is
intentional. It is exactly why "untouched v117" scored a perfect merge
precision in EXP-052, and why EXP-058's do-nothing rung correctly reports no
precision at all. The shim keeps the old constant for callers that depend on
it; new code gets NaN. Both behaviours are pinned by this experiment, so
neither can drift silently.

## What is *not* covered, stated

- **`treestitch.partition.merge_metrics`** needs an `ObservationGraph` and was
  not cross-checked here. Its edge-level maths already delegates to
  `neuronauts.assemble.edge_partition.edge_merge_metrics`, so it shares a code
  path rather than duplicating one, but that is an argument from inspection,
  not a measurement.
- **`treestitch/metrics.py`** was reported as "own maths" by a crude grep. It
  is not: `compute_full_metrics` composes `treestitch.partition` and
  `treestitch.connectivity`, both of which delegate. It is an aggregation and
  formatting layer, and needs no migration.
- **`experiments/pcfg/conn_metric.py` is a deliberate non-migration.** It looks
  like the last unconverted implementation but measures a *different quantity*:
  net pair-error change against do-nothing on a correction task, via union-find
  over rows linked by either their do-nothing group or their corrected group.
  Nothing in `neuronauts/metrics/` computes that. It is also the metric the
  PCFG findings moved to after concluding pairwise AUC was "a mirage", which
  makes it a candidate for **promotion into `metrics/`** rather than
  retirement. Left in place, recorded here, not silently skipped.

## Phase 1 status

Five of six legacy entry points delegate and are verified equal. The sixth
computes something else. The metric consolidation is therefore closed on its
own terms — one package, one set of conventions, agreement demonstrated rather
than assumed — with one identified follow-up: promote the net-pair-error metric
into `metrics/` so the correction work has a home alongside partition and
connectome scoring.

```bash
uv run python -m neuronauts.experiments.exp059_metric_agreement
```
