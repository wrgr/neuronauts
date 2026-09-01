# probe_l2_throughput — probe l2 throughput

**Status: COMPLETED (no gate)** — no gate recorded; the run finished and wrote its result

- **Result file** [`results/probe_l2_throughput.json`](../probe_l2_throughput.json)
- **Script** [`scripts/probe_l2_throughput.py`](../../scripts/probe_l2_throughput.py)

## What this experiment does

Timing probe: how fast can we retrieve L2 geometry for many v117 roots?

EXP-053B concluded the complete-root L2 route "is not a practical 1,023-root
dense substrate" after a 10-root probe ran >14 min. That probe used the
per-root path in ``neuronauts.data.lineage.l2_skeleton``, which does
1 root_leaves call + ceil(n_l2/500) attribute calls per root, each preceded by
a 0.25 s sleep, strictly serially.

This script measures three routes on the SAME roots so the comparison is fair:

  A. current  -- lineage.l2_skeleton(), serial, as used by EXP-053B
  B. caveclient -- l2cache.get_l2data(root_ids=[...]) per root
  C. pooled   -- threaded root_leaves, then pool every L2 id across all roots
                 into large batched attribute POSTs with no sleep

Route C is the hypothesis: the attributes endpoint is keyed on l2_ids, not on
root, so per-root batching wastes almost all of the request budget.

Roots come from the offline v117 box cache (no network needed to pick them).
Nothing is written except the shared L2 cache.

## Provenance

- **Commit** *not recorded*
- **Result file written** 2026-09-01T19:17:56+00:00 (file mtime)
- **Provenance completeness** 0% — missing `git_commit`, `git_dirty`, `timestamp_utc`, `inputs`, `argv`, `packages`, `synthetic_fallback`

## Gate

*No gate recorded.*

## Population

*none recorded*

## Headline values

| field | value |
|---|---|
| `config.attr_batch` | 2,000 |
| `config.attr_workers` | 8 |
| `config.box_dir` | data/boxes_v117 |
| `config.leaf_workers` | 16 |
| `config.min_syn` | 10 |
| `config.n_roots` | 12 |
| `config.out` | results/probe_l2_throughput.json |
| `config.routes` | A,B,C |
| `meta.box` | 002b1cf087ab2d16.npz |
| `meta.n_roots_eligible` | 2,158 |
| `meta.n_roots_total` | 23,329 |
| `meta.n_synapses` | 58,483 |

## Tables

### `results`

3 rows × 11 columns.

| results | route | seconds | n_roots | ok | vertices | sec_per_root | pooled_l2_ids | n_attr_requests | t_root_leaves | t_attributes | errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | A_current_serial | 122.3 | 12 | 12 | 23,992 | 10.19 | — | — | — | — | — |
| `1` | B_caveclient_per_root | 4.815 | 12 | 0 | 0 | 0.4012 | — | — | — | — | — |
| `2` | C_pooled_threaded | 30.77 | 12 | 12 | 161,682 | 2.564 | 161,682 | 81 | 5.373 | 25.37 | [] |

## Figures

![`results`: seconds, n_roots, ok, vertices, sec_per_root, pooled_l2_ids per row.](figures/probe_l2_throughput_results_panels.png)

`results`: seconds, n_roots, ok, vertices, sec_per_root, pooled_l2_ids per row. **Check:** Does the best row on each metric agree with the evaluation note?

## Neuroglancer

*No spatial provenance (bounding box, anchor, or atom id) recorded, so no view was built.*

## Reproduce

```bash
uv run --extra cave python scripts/probe_l2_throughput.py
```

Regenerate this report: `uv run python scripts/build_reports.py --only probe_l2_throughput`
