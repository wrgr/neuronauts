# probe_substrate_pilot — probe substrate pilot

**Status: COMPLETED (no gate)** — no gate recorded; the run finished and wrote its result

- **Result file** [`results/probe_substrate_pilot.json`](../probe_substrate_pilot.json)
- **Script** [`scripts/probe_substrate_pilot.py`](../../scripts/probe_substrate_pilot.py)

## What this experiment does

Pilot the proofread-cell-first substrate on a handful of cells.

The planned substrate is: proofread cell at v1822 -> real L2 adjacency graph
(``level2_chunk_graph``) -> each L2 node labelled with the v117 root it belonged
to (``roots_at`` at the v117 timestamp) -> v117 fragments become the atoms the
grammar assembles, with the proofread cell id as ground truth.

That whole design rests on one unverified assumption: that ``roots_at`` returns
a sensible v117 root for a *current* L2 node id. L2 nodes are chunk-level
objects that edits can create or destroy, so current L2 ids need not have
existed at v117. ``build_region_world_l2`` assumes this works, but it swallows
failures, so a silent zero rate would look like sparse data.

This measures, on a few cells: fetch cost, node counts, v117 resolution rate,
and how many v117 fragments each proofread cell breaks into (the merge signal
we need to exist for the task to be learnable at all).

## Provenance

- **Commit** *not recorded*
- **Result file written** 2026-09-01T19:48:42+00:00 (file mtime)
- **Provenance completeness** 0% — missing `git_commit`, `git_dirty`, `timestamp_utc`, `inputs`, `argv`, `packages`, `synthetic_fallback`

## Gate

*No gate recorded.*

## Population

*none recorded*

## Headline values

| field | value |
|---|---|
| `config.centre_um` | [663, 591, 860] |
| `config.n_cells` | 5 |
| `config.out` | results/probe_substrate_pilot.json |
| `config.side_um` | 200 |
| `config.tier` | gold |
| `config.version` | 1,822 |
| `config.workers` | 6 |
| `coord_rate` | 1 |
| `frag_counts` | [99, 54, 70, 52, 45] |
| `n_cells_in_region` | 247 |
| `n_shared_fragments` | 0 |
| `pooled_l2` | 20,493 |
| `timing.coords_s` | 5.13 |
| `timing.graphs_s` | 5.202 |
| `timing.v117_s` | 2.093 |
| `v117_resolve_rate` | 0.9607 |

## Tables

### `cells`

5 rows × 4 columns.

| cells | root | t_graph_s | n_edges | n_l2 |
|---|---:|---:|---:|---:|
| `864691135361314119` | 864,691,135,361,314,119 | 3.638 | 4,395 | 3,913 |
| `864691136043283030` | 864,691,136,043,283,030 | 3.642 | 4,335 | 3,827 |
| `864691135294441654` | 864,691,135,294,441,654 | 3.599 | 3,672 | 3,158 |
| `864691136968429774` | 864,691,136,968,429,774 | 3.622 | 4,935 | 4,373 |
| `864691135562842337` | 864,691,135,562,842,337 | 4.665 | 6,036 | 5,222 |

## Figures

![`cells`: n_l2, n_edges, root, t_graph_s per row.](figures/probe_substrate_pilot_cells_panels.png)

`cells`: n_l2, n_edges, root, t_graph_s per row. **Check:** Does the best row on each metric agree with the evaluation note?

## Neuroglancer

- **region** — [state JSON](ngl/probe_substrate_pilot_region.json) · [open in Neuroglancer](https://spelunker.cave-explorer.org/#!%7B%22dimensions%22%3A%7B%22x%22%3A%5B1e-09%2C%22m%22%5D%2C%22y%22%3A%5B1e-09%2C%22m%22%5D%2C%22z%22%3A%5B1e-09%2C%22m%22%5D%7D%2C%22position%22%3A%5B663000.0%2C591000.0%2C860000.0%5D%2C%22crossSectionScale%22%3A222.22%2C%22projectionScale%22%3A320000.0%2C%22layers%22%3A%5B%7B%22type%22%3A%22image%22%2C%22source%22%3A%22precomputed%3A%2F%2Fhttps%3A%2F%2Fbossdb-open-data.s3.amazonaws.com%2Fiarpa_microns%2Fminnie%2Fminnie65%2Fem%22%2C%22name%22%3A%22em%22%2C%22shader%22%3A%22void%20main%28%29%20%7B%20emitGrayscale%28toNormalized%28getDataValue%28%29%29%29%3B%20%7D%22%7D%2C%7B%22type%22%3A%22segmentation%22%2C%22source%22%3A%22graphene%3A%2F%2Fhttps%3A%2F%2Fminnie.microns-daf.com%2Fsegmentation%2Ftable%2Fminnie65_public%22%2C%22name%22%3A%22segmentation%22%2C%22segments%22%3A%5B%5D%2C%22objectAlpha%22%3A0.6%2C%22hideSegmentZero%22%3Atrue%7D%2C%7B%22type%22%3A%22annotation%22%2C%22name%22%3A%22region%22%2C%22source%22%3A%7B%22url%22%3A%22local%3A%2F%2Fannotations%22%2C%22transform%22%3A%7B%22outputDimensions%22%3A%7B%22x%22%3A%5B1e-09%2C%22m%22%5D%2C%22y%22%3A%5B1e-09%2C%22m%22%5D%2C%22z%22%3A%5B1e-09%2C%22m%22%5D%7D%7D%7D%2C%22annotationColor%22%3A%22%232a78d6%22%2C%22annotations%22%3A%5B%7B%22type%22%3A%22axis_aligned_bounding_box%22%2C%22id%22%3A%22region-box%22%2C%22pointA%22%3A%5B563000.0%2C491000.0%2C760000.0%5D%2C%22pointB%22%3A%5B763000.0%2C691000.0%2C960000.0%5D%2C%22description%22%3A%22200.0%20um%20cube%20at%20%5B663%2C%20591%2C%20860%5D%20um%22%7D%5D%7D%5D%2C%22showSlices%22%3Afalse%2C%22layout%22%3A%224panel%22%7D) — 200.0 um cube at [663, 591, 860] um from `config`

Load a state JSON by pasting it into the viewer's `{}` (edit JSON state) panel, or serve it locally with `scripts/ngl_view.py state <file> --serve`.

## Reproduce

```bash
uv run --extra cave python scripts/probe_substrate_pilot.py
```

Regenerate this report: `uv run python scripts/build_reports.py --only probe_substrate_pilot`
