# EXP-053B — EXP-053B real-L2 candidate-panel recall

**Status: FAILED (gate not met)** — `success_criterion.passed` = False

- **Result file** [`results/exp053b_l2_candidate_panel.json`](../exp053b_l2_candidate_panel.json)
- **Script** [`scripts/benchmark_exp053b_l2_candidate_panel.py`](../../scripts/benchmark_exp053b_l2_candidate_panel.py)
- **Evaluation note** [`results/exp053b_evaluation.md`](../exp053b_evaluation.md)
- **Elapsed** 9.9 min

## What this experiment does

EXP-053B: candidate-panel recall from bounded real v117 L2 geometry.

### From the evaluation note

The experiment failed its prerequisite coverage gate. Of 1,023 eligible v117
roots selected from the real synapse table, only 284 (27.8%) had at least two
bounded L2 representative coordinates in the 30 um box plus a 10 um halo. Only
one of the 14 true two-root lineages had L2 geometry on both sides. That one
pair was not proposed at any tested radius/cone setting, including the fully
open 10 um / 180 degree panel.

Therefore this is **not evidence that L2 endpoint geometry cannot recover real
continuations**. It is evidence that the current bounded v117 L2 retrieval path
does not provide a valid positive panel for this experiment.

*Full note: [`exp053b_evaluation.md`](../exp053b_evaluation.md)*

## Provenance

- **Commit** [`13a7e2864228`](https://github.com/wrgr/neuronauts/commit/13a7e28642281fa8b636e11339fdd766e30a4c6e) — *not present in the local repository*
- **Result file written** 2026-09-01T18:44:32+00:00 (file mtime)
- **Provenance completeness** 35% — missing `git_dirty`, `timestamp_utc`, `inputs`, `argv`, `packages`

**Honesty flags** (recorded by the script itself)

| flag | value |
|---|---|
| `candidate_generation_used_target_lineage` | no |
| `labels_used_only_for_evaluation` | yes |
| `synthetic_fallback` | no |

**Recorded run parameters**

| field | value |
|---|---|
| `bbox_nm` | [[844776.0, 700136.0, 875040.0], [874776.0, 730136.0, 905040.0]] |
| `geometry_halo_nm` | 10,000.0 |
| `l2_topology` | MST over real bounded L2 rep_coord_nm points |
| `root_selection` | synapse-table v117 roots with >=10 observations, plus soma roots |
| `target_timestamp` | 1,745,921,401 |
| `target_version` | 1,412 |

## Gate

| gate | requirement | required | observed |
|---|---|---:|---:|
| success_criterion | max_median_panel_size | 20 | — |
| success_criterion | recall | 0.9000 | — |

## Population

| field | value |
|---|---|
| `eligible_roots` | 1,023 |
| `endpoint_paths` | 2,178 |
| `l2_covered_roots` | 284 |
| `l2_covered_true_pairs` | 1 |
| `raw_pairs_within_max_radius` | 15,292 |
| `synapses` | 24,573 |
| `true_merge_pairs` | 14 |

## Headline values

| field | value |
|---|---|
| `best_recall_configuration.candidate_pairs` | 0 |
| `best_recall_configuration.cone_degrees` | 30 |
| `best_recall_configuration.panel_size.max` | 0 |
| `best_recall_configuration.panel_size.median` | 0 |
| `best_recall_configuration.panel_size.p90` | 0 |
| `best_recall_configuration.radius_um` | 0.5000 |
| `best_recall_configuration.recall_all_true_pairs` | 0 |
| `best_recall_configuration.recall_l2_covered_true_pairs` | 0 |
| `best_recall_configuration.true_pairs_recovered` | 0 |

## Tables

### `grid`

30 rows × 7 columns.

| grid | candidate_pairs | panel_size.max | panel_size.median | panel_size.p90 | recall_all_true_pairs | recall_l2_covered_true_pairs | true_pairs_recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| `r0.5_cone30` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `r0.5_cone45` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `r0.5_cone60` | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| `r0.5_cone90` | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| `r0.5_cone180` | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| `r1_cone30` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `r1_cone45` | 3 | 1 | 0 | 0 | 0 | 0 | 0 |
| `r1_cone60` | 5 | 2 | 0 | 0 | 0 | 0 | 0 |
| `r1_cone90` | 15 | 2 | 0 | 0.7000 | 0 | 0 | 0 |
| `r1_cone180` | 38 | 3 | 0 | 1 | 0 | 0 | 0 |
| `r2_cone30` | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| `r2_cone45` | 10 | 1 | 0 | 0 | 0 | 0 | 0 |
| `r2_cone60` | 25 | 2 | 0 | 1 | 0 | 0 | 0 |
| `r2_cone90` | 113 | 4 | 1 | 2 | 0 | 0 | 0 |
| `r2_cone180` | 230 | 6 | 1 | 3 | 0 | 0 | 0 |
| `r2.5_cone30` | 4 | 1 | 0 | 0 | 0 | 0 | 0 |
| `r2.5_cone45` | 16 | 2 | 0 | 1 | 0 | 0 | 0 |
| `r2.5_cone60` | 50 | 2 | 0 | 1 | 0 | 0 | 0 |
| `r2.5_cone90` | 202 | 5 | 1 | 3 | 0 | 0 | 0 |
| `r2.5_cone180` | 412 | 11 | 3 | 5 | 0 | 0 | 0 |
| `r5_cone30` | 34 | 3 | 0 | 1 | 0 | 0 | 0 |
| `r5_cone45` | 128 | 5 | 1 | 2 | 0 | 0 | 0 |
| `r5_cone60` | 358 | 9 | 2 | 5 | 0 | 0 | 0 |
| `r5_cone90` | 1,207 | 18 | 8 | 13 | 0 | 0 | 0 |
| `r5_cone180` | 2,373 | 35 | 16 | 26 | 0 | 0 | 0 |
| `r10_cone30` | 199 | 8 | 1 | 3 | 0 | 0 | 0 |
| `r10_cone45` | 859 | 19 | 6 | 11 | 0 | 0 | 0 |
| `r10_cone60` | 2,309 | 37 | 16 | 24.7 | 0 | 0 | 0 |
| `r10_cone90` | 7,405 | 89 | 53 | 69 | 0 | 0 | 0 |
| `r10_cone180` | 15,292 | 186 | 106 | 146.7 | 0 | 0 | 0 |

## Figures

![`grid`: recall_all_true_pairs, recall_l2_covered_true_pairs, candidate_pairs, panel_size.median, panel_size.max, panel_size.p90 per row.](figures/EXP-053B_grid_panels.png)

`grid`: recall_all_true_pairs, recall_l2_covered_true_pairs, candidate_pairs, panel_size.median, panel_size.max, panel_size.p90 per row. **Check:** Does the best row on each metric agree with the evaluation note?

![`grid` as a grid: recall_all_true_pairs.](figures/EXP-053B_grid_recall_all_true_pairs_grid.png)

`grid` as a grid: recall_all_true_pairs. **Check:** Does the metric vary smoothly with both parameters?

![`grid` as a grid: recall_l2_covered_true_pairs.](figures/EXP-053B_grid_recall_l2_covered_true_pairs_grid.png)

`grid` as a grid: recall_l2_covered_true_pairs. **Check:** Does the metric vary smoothly with both parameters?

![`grid` as a grid: candidate_pairs.](figures/EXP-053B_grid_candidate_pairs_grid.png)

`grid` as a grid: candidate_pairs. **Check:** Does the metric vary smoothly with both parameters?

## Neuroglancer

- **experiment box** — [state JSON](ngl/EXP-053B_bbox.json) · [open in Neuroglancer](https://spelunker.cave-explorer.org/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B1e-09%2C%22m%22%5D%2C%22y%22%3A%5B1e-09%2C%22m%22%5D%2C%22z%22%3A%5B1e-09%2C%22m%22%5D%7D%2C%22position%22%3A%5B859776.0%2C715136.0%2C890040.0%5D%2C%22crossSectionScale%22%3A33.33%2C%22projectionScale%22%3A48000.0%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22precomputed%3A%2F%2Fhttps%3A%2F%2Fbossdb-open-data.s3.amazonaws.com%2Fiarpa_microns%2Fminnie%2Fminnie65%2Fem%22%2C%22name%22%3A%22em%22%2C%22shader%22%3A%22void%20main%28%29%20%7B%20emitGrayscale%28toNormalized%28getDataValue%28%29%29%29%3B%20%7D%22%7D%2C%7B%22type%22%3A%22segmentation%22%2C%22source%22%3A%22graphene%3A%2F%2Fhttps%3A%2F%2Fminnie.microns-daf.com%2Fsegmentation%2Ftable%2Fminnie65_public%22%2C%22name%22%3A%22segmentation%22%2C%22segments%22%3A%5B%5D%2C%22objectAlpha%22%3A0.6%2C%22hideSegmentZero%22%3Atrue%7D%2C%7B%22type%22%3A%22annotation%22%2C%22name%22%3A%22experiment%20bbox%22%2C%22source%22%3A%7B%22url%22%3A%22local%3A%2F%2Fannotations%22%2C%22transform%22%3A%7B%22outputDimensions%22%3A%7B%22x%22%3A%5B1e-09%2C%22m%22%5D%2C%22y%22%3A%5B1e-09%2C%22m%22%5D%2C%22z%22%3A%5B1e-09%2C%22m%22%5D%7D%7D%7D%2C%22annotationColor%22%3A%22%232a78d6%22%2C%22annotations%22%3A%5B%7B%22type%22%3A%22axis_aligned_bounding_box%22%2C%22id%22%3A%22experiment%20bbox-box%22%2C%22pointA%22%3A%5B844776.0%2C700136.0%2C875040.0%5D%2C%22pointB%22%3A%5B874776.0%2C730136.0%2C905040.0%5D%2C%22description%22%3A%22EXP-053B%20box%22%7D%5D%7D%5D%2C%22showSlices%22%3Afalse%2C%22layout%22%3A%224panel%22%7D) — bounding box, anchor soma and anchor root from `provenance`

Load a state JSON by pasting it into the viewer's `{}` (edit JSON state) panel, or serve it locally with `scripts/ngl_view.py state <file> --serve`.

## Reproduce

```bash
uv run --extra cave python scripts/benchmark_exp053b_l2_candidate_panel.py
```

Regenerate this report: `uv run python scripts/build_reports.py --only EXP-053B`
