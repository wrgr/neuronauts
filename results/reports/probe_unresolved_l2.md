# probe_unresolved_l2 — probe unresolved l2

**Status: COMPLETED (no gate)** — no gate recorded; the run finished and wrote its result

- **Result file** [`results/probe_unresolved_l2.json`](../probe_unresolved_l2.json)
- **Script** [`scripts/probe_unresolved_l2.py`](../../scripts/probe_unresolved_l2.py)

## What this experiment does

Why do 3.6% of L2 nodes have no v117 root?

Three candidate explanations, with different consequences:

  A. the L2 node was created by a post-v117 edit (a split makes new within-chunk
     components), but the supervoxels underneath did exist at v117
     -> the cable is attributable; assign via supervoxel majority
  B. the underlying supervoxels were unsegmented at v117
     -> genuinely unattributable; must be excluded and declared in the ceiling
  C. transient API failure
     -> a bug on our side; retry

The test drops one level: for L2 nodes that return no v117 root, fetch their
supervoxels and ask *those* for a v117 root. Supervoxel ids are immutable, so
they discriminate A from B directly.

A labelled control group (L2 nodes that DID resolve) is run through the same
path, so we can tell a real effect from a broken query.

## Provenance

- **Commit** *not recorded*
- **Result file written** 2026-09-01T20:09:06+00:00 (file mtime)
- **Provenance completeness** 0% — missing `git_commit`, `git_dirty`, `timestamp_utc`, `inputs`, `argv`, `packages`, `synthetic_fallback`

## Gate

*No gate recorded.*

## Population

*none recorded*

## Headline values

| field | value |
|---|---|
| `n_l2` | 38,431 |
| `n_unresolved` | 1,368 |

## Tables

### `summary`

2 rows × 4 columns.

| summary | n | median_frac | mean_frac | frac_nodes_majority_resolved |
|---|---:|---:|---:|---:|
| `unresolved` | 29 | 1 | 1 | 1 |
| `control` | 45 | 1 | 1 | 1 |

## Figures

*No sweep-like table or percentile series to plot.*

## Neuroglancer

*No spatial provenance (bounding box, anchor, or atom id) recorded, so no view was built.*

## Reproduce

```bash
uv run --extra cave python scripts/probe_unresolved_l2.py
```

Regenerate this report: `uv run python scripts/build_reports.py --only probe_unresolved_l2`
