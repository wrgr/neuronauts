# probe_v117_geometry_route — probe v117 geometry route

**Status: COMPLETED (no gate)** — no gate recorded; the run finished and wrote its result

- **Result file** [`results/probe_v117_geometry_route.json`](../probe_v117_geometry_route.json)
- **Script** [`scripts/probe_v117_geometry_route.py`](../../scripts/probe_v117_geometry_route.py)

## What this experiment does

Can we get real L2 geometry for a *stale* v117 root?

With a label-blind population the atoms are v117 roots enumerated from synapses,
not from proofread cells, so we must fetch geometry keyed on the v117 root
itself. The catch: a v117 root that was later edited is no longer a live root in
the current chunkedgraph. Endpoints may or may not serve historical ids.

This distinguishes two classes explicitly:
  current v117 roots -- never edited, so no merge signal
  stale   v117 roots -- merged/split since, i.e. exactly the interesting atoms

and tries three routes on each: /root, /leaves?stop_layer=2, and lvl2_graph
(true adjacency). Errors are reported, never swallowed.

Stale roots are sourced by walking a proofread cell's L2 nodes back to v117,
which guarantees they participated in a real edit.

## Provenance

- **Commit** *not recorded*
- **Result file written** 2026-09-01T19:53:28+00:00 (file mtime)
- **Provenance completeness** 0% — missing `git_commit`, `git_dirty`, `timestamp_utc`, `inputs`, `argv`, `packages`, `synthetic_fallback`

## Gate

*No gate recorded.*

## Population

*none recorded*

## Tables

### `rows`

24 rows × 9 columns.

| rows | root | is_current | root_status | leaves_status | leaves_s | n_l2_leaves | lvl2graph_status | lvl2graph_s | n_adjacency_edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `864691135615779689` | 864,691,135,615,779,689 | yes | 200 | 200 | 0.7586 | 3,053 | 200 | 1.365 | 3,468 |
| `864691135994546090` | 864,691,135,994,546,090 | yes | 200 | 200 | 0.6871 | 2,598 | 200 | 1.563 | 2,941 |
| `864691136098990973` | 864,691,136,098,990,973 | yes | 200 | 200 | 0.4600 | 578 | 200 | 0.7722 | 637 |
| `864691136485518308` | 864,691,136,485,518,308 | yes | 200 | 200 | 0.2956 | 133 | 200 | 0.3427 | 147 |
| `864691135605162095` | 864,691,135,605,162,095 | yes | 200 | 200 | 0.2465 | 58 | 200 | 5.327 | 66 |
| `864691135790419878` | 864,691,135,790,419,878 | yes | 200 | 200 | 0.2729 | 55 | 200 | 0.2903 | 57 |
| `864691135634990444` | 864,691,135,634,990,444 | yes | 200 | 200 | 0.3227 | 205 | 200 | 0.4189 | 223 |
| `864691136162616981` | 864,691,136,162,616,981 | yes | 200 | 200 | 0.2157 | 45 | 200 | 0.3014 | 47 |
| `864691136087812221` | 864,691,136,087,812,221 | yes | 200 | 200 | 0.3414 | 144 | 200 | 0.5416 | 155 |
| `864691135894113941` | 864,691,135,894,113,941 | yes | 200 | 200 | 4.914 | 60 | 200 | 0.2726 | 64 |
| `864691136716521444` | 864,691,136,716,521,444 | yes | 200 | 200 | 0.1818 | 31 | 200 | 4.641 | 33 |
| `864691135360160583` | 864,691,135,360,160,583 | yes | 200 | 200 | 1.669 | 4,707 | 200 | 2.325 | 5,501 |
| `864691136585469924` | 864,691,136,585,469,924 | yes | 200 | 200 | 0.2424 | 27 | 200 | 0.2155 | 28 |
| `864691135633536364` | 864,691,135,633,536,364 | yes | 200 | 200 | 0.2245 | 39 | 200 | 0.2427 | 40 |
| `864691136124084817` | 864,691,136,124,084,817 | yes | 200 | 200 | 0.5463 | 705 | 200 | 0.8801 | 766 |
| `864691135603299436` | 864,691,135,603,299,436 | yes | 200 | 200 | 0.2303 | 29 | 200 | 0.2215 | 31 |
| `864691136637123905` | 864,691,136,637,123,905 | yes | 200 | 200 | 0.2323 | 25 | 200 | 0.2176 | 26 |
| `864691136364046364` | 864,691,136,364,046,364 | yes | 200 | 200 | 0.2048 | 20 | 200 | 0.2523 | 20 |
| `864691136099948881` | 864,691,136,099,948,881 | yes | 200 | 200 | 0.2327 | 21 | 200 | 0.2453 | 22 |
| `864691135101514280` | 864,691,135,101,514,280 | yes | 200 | 200 | 0.2566 | 24 | 200 | 0.2876 | 26 |
| `864691136139881623` | 864,691,136,139,881,623 | yes | 200 | 200 | 0.2334 | 18 | 200 | 0.2700 | 17 |
| `864691134865713002` | 864,691,134,865,713,002 | yes | 200 | 200 | 0.2031 | 16 | 200 | 0.1980 | 17 |
| `864691135631918956` | 864,691,135,631,918,956 | yes | 200 | 200 | 0.1907 | 15 | 200 | 0.2072 | 15 |
| `864691135382617050` | 864,691,135,382,617,050 | yes | 200 | 200 | 0.3184 | 289 | 200 | 0.3902 | 325 |

## Figures

![`rows`: root, root_status, leaves_status, leaves_s, n_l2_leaves, lvl2graph_status per row.](figures/probe_v117_geometry_route_rows_panels.png)

`rows`: root, root_status, leaves_status, leaves_s, n_l2_leaves, lvl2graph_status per row. **Check:** Does the best row on each metric agree with the evaluation note?

## Neuroglancer

*No spatial provenance (bounding box, anchor, or atom id) recorded, so no view was built.*

## Reproduce

```bash
uv run --extra cave python scripts/probe_v117_geometry_route.py
```

Regenerate this report: `uv run python scripts/build_reports.py --only probe_v117_geometry_route`
