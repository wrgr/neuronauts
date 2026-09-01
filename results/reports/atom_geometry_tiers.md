# atom_geometry_tiers — atom geometry tiers

**Status: COMPLETED (no gate)** — no gate recorded; the run finished and wrote its result

- **Result file** [`results/atom_geometry_tiers.json`](../atom_geometry_tiers.json)
- **Script** [`scripts/fetch_atom_geometry.py`](../../scripts/fetch_atom_geometry.py)

## What this experiment does

Fetch L2 geometry for the label-blind atom population, in widening tiers.

Runs ``>=10``, then ``>=5``, then ``>=1`` synapses. Each tier only fetches
atoms the previous tiers did not cover, so the sequence costs the same as going
straight to ``>=1`` while producing a usable substrate after the first tier and
letting us look before committing to the next.

Geometry is always fetched against the *outer* region bounds so one cache
serves every nested region and, later, a scale-up to all somata.

Safe to interrupt and rerun; progress is on disk.

## Provenance

- **Commit** *not recorded*
- **Result file written** 2026-09-01T22:08:23+00:00 (file mtime)
- **Provenance completeness** 0% — missing `git_commit`, `git_dirty`, `timestamp_utc`, `inputs`, `argv`, `packages`, `synthetic_fallback`

## Gate

*No gate recorded.*

## Population

*none recorded*

## Tables

### `rows`

2 rows × 7 columns.

| rows | k | n_atoms | n_with_geom | total_l2_nodes | coord_coverage | caliber_coverage | elapsed_min |
|---|---:|---:|---:|---:|---:|---:|---:|
| `10` | 10 | 20,826 | 20,826 | 18,519,922 | 1.0000 | 1.0000 | 2.532 |
| `5` | 5 | 40,109 | 40,109 | 20,956,855 | 1.0000 | 1.0000 | 2.506 |

## Figures

*No sweep-like table or percentile series to plot.*

## Neuroglancer

*No spatial provenance (bounding box, anchor, or atom id) recorded, so no view was built.*

## Reproduce

```bash
uv run --extra cave python scripts/fetch_atom_geometry.py
```

Regenerate this report: `uv run python scripts/build_reports.py --only atom_geometry_tiers`
