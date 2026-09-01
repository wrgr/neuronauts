# probe_population_scale — probe population scale

**Status: COMPLETED (no gate)** — no gate recorded; the run finished and wrote its result

- **Result file** [`results/probe_population_scale.json`](../probe_population_scale.json)
- **Script** [`scripts/probe_population_scale.py`](../../scripts/probe_population_scale.py)

## What this experiment does

Size the label-blind atom population for candidate region sizes.

The atom population must be enumerated without using ground truth, or the task
is rigged. The GT-free filter is the one described for this work: every v117
fragment carrying at least k synapses in the region. Ground truth is attached
afterward, only where it happens to exist.

The cost driver is mapping synapse supervoxels to v117 roots, so this reports,
per region size: synapse count, unique supervoxel count, and the projected
mapping time at the measured batched ``roots_at`` rate. It also runs a real
timed sample so the projection is anchored to a measurement rather than a guess.

Reads the already-extracted region NPZ; no full table rescan.

## Provenance

- **Commit** *not recorded*
- **Result file written** 2026-09-01T19:52:02+00:00 (file mtime)
- **Provenance completeness** 0% — missing `git_commit`, `git_dirty`, `timestamp_utc`, `inputs`, `argv`, `packages`, `synthetic_fallback`

## Gate

*No gate recorded.*

## Population

*none recorded*

## Headline values

| field | value |
|---|---|
| `centre_um` | [663, 591, 860] |
| `rate_sv_per_s` | 5,831.6 |
| `resolve_fraction` | 1 |

## Tables

### `rows`

4 rows × 4 columns.

| rows | side_um | n_synapses | n_unique_sv | est_v117_map_min |
|---|---:|---:|---:|---:|
| `0` | 60 | 198,892 | 395,478 | 1.13 |
| `1` | 100 | 901,498 | 1,792,309 | 5.122 |
| `2` | 150 | 3,084,646 | 6,131,190 | 17.52 |
| `3` | 200 | 7,514,814 | 14,933,676 | 42.68 |

## Figures

![`rows`: side_um, n_synapses, n_unique_sv, est_v117_map_min per row.](figures/probe_population_scale_rows_panels.png)

`rows`: side_um, n_synapses, n_unique_sv, est_v117_map_min per row. **Check:** Does the best row on each metric agree with the evaluation note?

## Neuroglancer

*No spatial provenance (bounding box, anchor, or atom id) recorded, so no view was built.*

## Reproduce

```bash
uv run --extra cave python scripts/probe_population_scale.py
```

Regenerate this report: `uv run python scripts/build_reports.py --only probe_population_scale`
