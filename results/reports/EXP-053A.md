# EXP-053A — EXP-053A checkpoint bake-off

**Status: COMPLETED (no gate)** — no gate recorded; the run finished and wrote its result

- **Result file** [`results/exp053a_checkpoint_bakeoff.json`](../exp053a_checkpoint_bakeoff.json)
- **Script** [`scripts/benchmark_exp053a_checkpoint_bakeoff.py`](../../scripts/benchmark_exp053a_checkpoint_bakeoff.py)
- **Evaluation note** [`results/exp053a_evaluation.md`](../exp053a_evaluation.md)
- **Elapsed** 6.2 min

## What this experiment does

EXP-053A: checkpoint bake-off on one fixed edit-bearing v117 population.

### From the evaluation note

No existing checkpoint separates real continuation pairs from dense spatial
confusers on the EXP-052 population. Permissive thresholds recover most of the
14 real merge pairs while collapsing hundreds of unrelated roots; the first
threshold that avoids collapse recovers no real merge pair.

| Checkpoint | Last high-recall threshold | Recall | Precision | Largest cluster | First non-collapse threshold | Recall there |
|---|---:|---:|---:|---:|---:|---:|
| raw skeleton | 0 | 0.929 | 0.000026 | 997 | 3 | 0 |
| raw skeleton + GAT | 0 | 0.929 | 0.000026 | 999 | 4 | 0 |
| legacy real | 0 | 0.786 | 0.000023 | 979 | 3 | 0 |
| root neighborhood | 0 | 0.929 | 0.000027 | 981 | 1 | 0 |

The raw-skeleton checkpoint at threshold 3 makes only two joins, both false.
The root-neighborhood checkpoint at threshold 1 makes 15 pairwise joins, all
false. At abstention, the untouched-v117 baseline is restored: expected run
length (ERL) 81.34 um and circuit F1 0.9868, with zero merge recall.

*Full note: [`exp053a_evaluation.md`](../exp053a_evaluation.md)*

## Provenance

- **Commit** [`87a594dcc7a0`](https://github.com/wrgr/neuronauts/commit/87a594dcc7a08c06069a40ff8c7a65da55540974) — *not present in the local repository*
- **Result file written** 2026-09-01T18:44:32+00:00 (file mtime)
- **Provenance completeness** 35% — missing `git_dirty`, `timestamp_utc`, `inputs`, `argv`, `packages`

**Honesty flags** (recorded by the script itself)

| flag | value |
|---|---|
| `benchmark_selection_used_target_lineage` | yes |
| `ground_truth_used_during_inference` | no |
| `synthetic_fallback` | no |
| `thresholds_are_post_hoc` | yes |

**Recorded run parameters**

| field | value |
|---|---|
| `anchor_soma_nm` | [859776.0, 715136.0, 890040.0] |
| `anchor_target_root` | 864,691,135,106,016,333 |
| `bbox_nm` | [[844776.0, 700136.0, 875040.0], [874776.0, 730136.0, 905040.0]] |
| `target_timestamp` | 1,745,921,401 |
| `target_version` | 1,412 |

## Gate

*No gate recorded.*

## Population

| field | value |
|---|---|
| `active_path_roots` | 1,023 |
| `mixed_lineage_roots` | 116 |
| `singleton_confusers` | 10,218 |
| `soma_seeds` | 6 |
| `synapse_bearing_v117_roots` | 11,241 |
| `synapses` | 24,573 |
| `true_merge_pairs` | 14 |

## Headline values

| field | value |
|---|---|
| `untouched_v117_baseline.ari` | 0 |
| `untouched_v117_baseline.erl_um` | 81.34 |
| `untouched_v117_baseline.merge_precision` | 1 |
| `untouched_v117_baseline.merge_recall` | 0 |
| `untouched_v117_baseline.n_labeled_fragments` | 1,023 |

## Tables

### `checkpoints`

4 rows × 10 columns.

| checkpoints | candidate_edges | checkpoint_sha256 | path_feature_mode | score_quantiles.0.0 | score_quantiles.0.25 | score_quantiles.0.5 | score_quantiles.0.75 | score_quantiles.0.9 | score_quantiles.0.99 | score_quantiles.1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `shared_grammar_raw_skel_50e.pt` | 29,985 | 1e704ff943647965b5d1771a89d8143efa6ea3aeef5909f7190965cd8... | raw_delta3+skeleton | -30.2 | -7.7 | -3.514 | -0.5566 | 1.344 | 3.544 | 5.504 |
| `shared_grammar_raw_skel_gat50e.pt` | 29,985 | 9cada042dd7cb356579eab42fb2a54ee22e7ebdec9deb29af21fe92c3... | raw_delta3+skeleton | -18.96 | -4.164 | -1.561 | 0.4941 | 1.944 | 3.747 | 5.42 |
| `shared_grammar_real.pt` | 29,985 | cf2e3118d627ef493a0d11a1576c7d0d33aa32f89facc551a804b1086... | legacy_geom3 | -66.24 | -16.45 | -7.764 | -1.509 | 1.023 | 3.191 | 4.929 |
| `shared_grammar_root_neighborhood_run001.pt` | 29,985 | 706f3b537a99f3318a36df59783b622f22d6ceefd6b456533b8dc787f... | legacy_geom3 | -2.441 | -1.472 | -0.7391 | 0.0288 | 0.5131 | 0.9913 | 1.469 |

### `checkpoints.shared_grammar_raw_skel_50e.pt.threshold_sweep`

7 rows × 16 columns.

| threshold_sweep | ari | circuit_f1 | circuit_precision | circuit_recall | erl_um | largest_cluster_roots | merge_precision | merge_recall | multi_soma_clusters | n_circuit_synapses | n_clusters_all_roots | n_labeled_fragments | n_pred_clusters | n_soma_clusters | predicted_join_pairs | single_soma_compliance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.0` | -1.20e-06 | 0.0002 | 9.50e-05 | 0.9929 | 5.487 | 997 | 2.62e-05 | 0.9286 | 0 | 24,573 | 10,242 | 1,023 | 24 | 6 | 496,510 | 1 |
| `1.0` | -2.91e-06 | 0.0002 | 0.0001 | 0.9912 | 7.942 | 974 | 2.53e-05 | 0.8571 | 0 | 24,573 | 10,265 | 1,023 | 47 | 6 | 473,855 | 1 |
| `2.0` | -5.62e-06 | 0.0004 | 0.0002 | 0.9901 | 18.39 | 867 | 2.40e-05 | 0.6429 | 0 | 24,573 | 10,373 | 1,023 | 155 | 6 | 375,413 | 1 |
| `3.0` | -6.70e-06 | 0.9841 | 0.9830 | 0.9852 | 79.71 | 2 | 0 | 0 | 0 | 24,573 | 11,239 | 1,023 | 1,021 | 6 | 2 | 1 |
| `4.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |
| `5.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |
| `6.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |

### `checkpoints.shared_grammar_raw_skel_gat50e.pt.threshold_sweep`

7 rows × 16 columns.

| threshold_sweep | ari | circuit_f1 | circuit_precision | circuit_recall | erl_um | largest_cluster_roots | merge_precision | merge_recall | multi_soma_clusters | n_circuit_synapses | n_clusters_all_roots | n_labeled_fragments | n_pred_clusters | n_soma_clusters | predicted_join_pairs | single_soma_compliance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.0` | -1.42e-06 | 0.0002 | 9.71e-05 | 0.9929 | 1.117 | 999 | 2.61e-05 | 0.9286 | 0 | 24,573 | 10,227 | 1,023 | 9 | 6 | 498,596 | 1 |
| `1.0` | -4.63e-07 | 0.0002 | 9.97e-05 | 0.9929 | 1.302 | 990 | 2.65e-05 | 0.9286 | 0 | 24,573 | 10,236 | 1,023 | 18 | 6 | 489,650 | 1 |
| `2.0` | -2.15e-07 | 0.0002 | 0.0001 | 0.9912 | 8.464 | 949 | 2.67e-05 | 0.8571 | 0 | 24,573 | 10,282 | 1,023 | 64 | 6 | 449,882 | 1 |
| `3.0` | -2.79e-06 | 0.0009 | 0.0004 | 0.9901 | 26.91 | 688 | 2.54e-05 | 0.4286 | 0 | 24,573 | 10,548 | 1,023 | 330 | 6 | 236,344 | 1 |
| `4.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |
| `5.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |
| `6.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |

### `checkpoints.shared_grammar_real.pt.threshold_sweep`

7 rows × 16 columns.

| threshold_sweep | ari | circuit_f1 | circuit_precision | circuit_recall | erl_um | largest_cluster_roots | merge_precision | merge_recall | multi_soma_clusters | n_circuit_synapses | n_clusters_all_roots | n_labeled_fragments | n_pred_clusters | n_soma_clusters | predicted_join_pairs | single_soma_compliance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.0` | -7.61e-06 | 0.0002 | 0.0001 | 0.9918 | 12.21 | 979 | 2.30e-05 | 0.7857 | 0 | 24,573 | 10,256 | 1,023 | 38 | 6 | 478,749 | 1 |
| `1.0` | -3.29e-06 | 0.0003 | 0.0002 | 0.9918 | 20.61 | 936 | 2.51e-05 | 0.7857 | 0 | 24,573 | 10,305 | 1,023 | 87 | 6 | 437,581 | 1 |
| `2.0` | 6.30e-06 | 0.0009 | 0.0004 | 0.9901 | 37.67 | 776 | 2.99e-05 | 0.6429 | 0 | 24,573 | 10,466 | 1,023 | 248 | 6 | 300,700 | 1 |
| `3.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |
| `4.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |
| `5.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |
| `6.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |

### `checkpoints.shared_grammar_root_neighborhood_run001.pt.threshold_sweep`

7 rows × 16 columns.

| threshold_sweep | ari | circuit_f1 | circuit_precision | circuit_recall | erl_um | largest_cluster_roots | merge_precision | merge_recall | multi_soma_clusters | n_circuit_synapses | n_clusters_all_roots | n_labeled_fragments | n_pred_clusters | n_soma_clusters | predicted_join_pairs | single_soma_compliance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.0` | 5.01e-07 | 0.0002 | 9.71e-05 | 0.9934 | 2.325 | 981 | 2.70e-05 | 0.9286 | 0 | 24,573 | 10,225 | 1,023 | 7 | 6 | 480,917 | 1 |
| `1.0` | -2.77e-05 | 0.9828 | 0.9803 | 0.9852 | 79.14 | 4 | 0 | 0 | 0 | 24,573 | 11,233 | 1,023 | 1,015 | 6 | 15 | 1 |
| `2.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |
| `3.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |
| `4.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |
| `5.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |
| `6.0` | 0 | 0.9868 | 0.9884 | 0.9852 | 81.34 | 1 | 1 | 0 | 0 | 24,573 | 11,241 | 1,023 | 1,023 | 6 | 0 | 1 |

## Figures

![`checkpoints`: candidate_edges per row.](figures/EXP-053A_checkpoints_panels.png)

`checkpoints`: candidate_edges per row. **Check:** Does the best row on each metric agree with the evaluation note?

![`checkpoints.*.threshold_sweep`: one line per checkpoints entry.](figures/EXP-053A_checkpoints_threshold_sweep_compare.png)

`checkpoints.*.threshold_sweep`: one line per checkpoints entry. **Check:** Does any line separate from the others, or do they all fail together?

## Neuroglancer

- **experiment box** — [state JSON](ngl/EXP-053A_bbox.json) · [open in Neuroglancer](https://spelunker.cave-explorer.org/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B1e-09%2C%22m%22%5D%2C%22y%22%3A%5B1e-09%2C%22m%22%5D%2C%22z%22%3A%5B1e-09%2C%22m%22%5D%7D%2C%22position%22%3A%5B859776.0%2C715136.0%2C890040.0%5D%2C%22crossSectionScale%22%3A33.33%2C%22projectionScale%22%3A48000.0%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22precomputed%3A%2F%2Fhttps%3A%2F%2Fbossdb-open-data.s3.amazonaws.com%2Fiarpa_microns%2Fminnie%2Fminnie65%2Fem%22%2C%22name%22%3A%22em%22%2C%22shader%22%3A%22void%20main%28%29%20%7B%20emitGrayscale%28toNormalized%28getDataValue%28%29%29%29%3B%20%7D%22%7D%2C%7B%22type%22%3A%22segmentation%22%2C%22source%22%3A%22graphene%3A%2F%2Fhttps%3A%2F%2Fminnie.microns-daf.com%2Fsegmentation%2Ftable%2Fminnie65_public%22%2C%22name%22%3A%22segmentation%22%2C%22segments%22%3A%5B%22864691135106016333%22%5D%2C%22objectAlpha%22%3A0.6%2C%22hideSegmentZero%22%3Atrue%7D%2C%7B%22type%22%3A%22annotation%22%2C%22name%22%3A%22experiment%20bbox%22%2C%22source%22%3A%7B%22url%22%3A%22local%3A%2F%2Fannotations%22%2C%22transform%22%3A%7B%22outputDimensions%22%3A%7B%22x%22%3A%5B1e-09%2C%22m%22%5D%2C%22y%22%3A%5B1e-09%2C%22m%22%5D%2C%22z%22%3A%5B1e-09%2C%22m%22%5D%7D%7D%7D%2C%22annotationColor%22%3A%22%232a78d6%22%2C%22annotations%22%3A%5B%7B%22type%22%3A%22axis_aligned_bounding_box%22%2C%22id%22%3A%22experiment%20bbox-box%22%2C%22pointA%22%3A%5B844776.0%2C700136.0%2C875040.0%5D%2C%22pointB%22%3A%5B874776.0%2C730136.0%2C905040.0%5D%2C%22description%22%3A%22EXP-053A%20box%22%7D%5D%7D%2C%7B%22type%22%3A%22annotation%22%2C%22name%22%3A%22anchor%20soma%22%2C%22source%22%3A%7B%22url%22%3A%22local%3A%2F%2Fannotations%22%2C%22transform%22%3A%7B%22outputDimensions%22%3A%7B%22x%22%3A%5B1e-09%2C%22m%22%5D%2C%22y%22%3A%5B1e-09%2C%22m%22%5D%2C%22z%22%3A%5B1e-09%2C%22m%22%5D%7D%7D%7D%2C%22annotationColor%22%3A%22%23eb6834%22%2C%22annotations%22%3A%5B%7B%22type%22%3A%22point%22%2C%22id%22%3A%22anchor%20soma-0%22%2C%22point%22%3A%5B859776.0%2C715136.0%2C890040.0%5D%2C%22description%22%3A%22anchor%20target%20root%20864691135106016333%22%7D%5D%7D%5D%2C%22showSlices%22%3Afalse%2C%22layout%22%3A%224panel%22%7D) — bounding box, anchor soma and anchor root from `provenance`

Load a state JSON by pasting it into the viewer's `{}` (edit JSON state) panel, or serve it locally with `scripts/ngl_view.py state <file> --serve`.

## Reproduce

```bash
uv run --extra cave python scripts/benchmark_exp053a_checkpoint_bakeoff.py
```

Regenerate this report: `uv run python scripts/build_reports.py --only EXP-053A`
