# atom_topology_k10 — atom topology k10

**Status: COMPLETED (no gate)** — no gate recorded; the run finished and wrote its result

- **Result file** [`results/atom_topology_k10.json`](../atom_topology_k10.json)
- **Script** [`scripts/build_atom_topology.py`](../../scripts/build_atom_topology.py)

## What this experiment does

Contract every atom's L2 adjacency into the harness topology table.

Input is the raw fetch (``data/substrate/geom``): per-atom L2 node sets, real
``lvl2_graph`` adjacency, and pooled L2 attributes. Output is one NPZ holding

  * a per-atom row -- node/edge/component counts, endpoint and branch counts,
    cable length, caliber, and the atom's presynaptic/postsynaptic tallies;
  * a per-endpoint row -- position, outward tangent, the length and caliber of
    the leaf segment it terminates.

The endpoint table is the surface candidate generation runs on: a false split
appears as two endpoints facing each other. Endpoints are *not* scarce at L2
resolution, so the caliber and leaf-length columns are what a proposer filters
on; this script reports their distribution so that filter can be chosen from
data rather than guessed.

Cable length is NaN for any segment with a coordinate-less node, and the NaN
share is reported rather than silently dropped.

    uv run python scripts/build_atom_topology.py --tier 10

## Provenance

- **Commit** *not recorded*
- **Result file written** 2026-09-01T22:12:52+00:00 (file mtime)
- **Provenance completeness** 0% — missing `git_commit`, `git_dirty`, `timestamp_utc`, `inputs`, `argv`, `packages`, `synthetic_fallback`

## Gate

*No gate recorded.*

## Population

*none recorded*

## Headline values

| field | value |
|---|---|
| `cable_m` | 32.87 |
| `endpoint_caliber_nm_pct.10` | 8 |
| `endpoint_caliber_nm_pct.25` | 12.03 |
| `endpoint_caliber_nm_pct.50` | 26.34 |
| `endpoint_caliber_nm_pct.75` | 41.41 |
| `endpoint_caliber_nm_pct.90` | 56.12 |
| `endpoint_caliber_nm_pct.99` | 78.56 |
| `leaf_len_nm_pct.10` | 383.5 |
| `leaf_len_nm_pct.25` | 805.7 |
| `leaf_len_nm_pct.50` | 1,477.8 |
| `leaf_len_nm_pct.75` | 2,180.4 |
| `leaf_len_nm_pct.90` | 3,153.3 |
| `leaf_len_nm_pct.99` | 9,691.2 |
| `n_atoms` | 20,826 |
| `n_branch` | 5,037,329 |
| `n_components` | 89,751 |
| `n_edges` | 21,030,275 |
| `n_endpoints` | 5,103,160 |
| `n_l2` | 18,519,922 |
| `n_leaf_segments` | 5,076,213 |
| `n_segments` | 12,650,842 |
| `n_segments_without_length` | 734 |
| `out` | data/substrate/topology/k10.npz |
| `tier` | 10 |

## Figures

![Percentile curve of `leaf_len_nm_pct` (log axis when the range spans more than 50×).](figures/atom_topology_k10_leaf_len_nm_pct_pct.png)

Percentile curve of `leaf_len_nm_pct` (log axis when the range spans more than 50×). **Check:** Where does the tail start, and is a threshold there defensible?

![Percentile curve of `endpoint_caliber_nm_pct` (log axis when the range spans more than 50×).](figures/atom_topology_k10_endpoint_caliber_nm_pct_pct.png)

Percentile curve of `endpoint_caliber_nm_pct` (log axis when the range spans more than 50×). **Check:** Where does the tail start, and is a threshold there defensible?

## Neuroglancer

*No spatial provenance (bounding box, anchor, or atom id) recorded, so no view was built.*

## Reproduce

```bash
uv run --extra cave python scripts/build_atom_topology.py
```

Regenerate this report: `uv run python scripts/build_reports.py --only atom_topology_k10`
