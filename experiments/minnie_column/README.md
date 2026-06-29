# Minnie Column experiments

> **Status: active (data)** — experiment thread; see the index in
> [`../README.md`](../README.md). The bounded, full-depth ROI testbed for global
> assembly.

Implements the workflow in [`docs/minnie_column_paradigm.md`](../../docs/minnie_column_paradigm.md) and **[`docs/minnie_column_downloads.md`](../../docs/minnie_column_downloads.md)** (EM, seg, meshes, skeletons, synapse tables).

## Prerequisites

- `pip install caveclient pandas` (CAVE is not in core `neuronauts` deps).
- Network access for CAVE.
- Materialization **1718** (or set `--version`).

## 1. Bbox for the Minnie Column (recommended path)

**Discover bounds from CAVE** using the column typing table (nucleus `target_id` list → soma positions → padded bbox):

```bash
python -m experiments.minnie_column.discover_column_bbox \
  --version 1718 \
  --margin-um 2 \
  --out-json data/minnie_column_bbox.json \
  --out-nuclei-tsv run_logs/minnie_column_nuclei_for_bbox.tsv
```

Writes `data/minnie_column_bbox.json` with key **`bbox_nm`**.  
If the default `--column-table allen_v1_column_types_slanted_ref` is missing at your release, pass `--column-table` per [release manifests](https://tutorial.microns-explorer.org/).

**Manual path:** copy coordinates from Neuroglancer or edit [`data/minnie_column_bbox_example.json`](../../data/minnie_column_bbox_example.json).

JSON format:

```json
{
  "bbox_nm": [[x0, y0, z0], [x1, y1, z1]]
}
```

Or pass `--bbox-nm x0,y0,z0,x1,y1,z1` to `build_manifest`.

## 2. Build nucleus manifest

```bash
# From repo root (use data/minnie_column_bbox.json after discover_column_bbox)
python -m experiments.minnie_column.build_manifest \
  --bbox-json data/minnie_column_bbox.json \
  --version 1718 \
  --bin-width-um 50 \
  --bin-height-um 100 \
  --auto-median-test \
  --out-tsv run_logs/minnie_column_manifest.tsv \
  --out-meta-json run_logs/minnie_column_manifest_meta.json
```

Explicit train/test bins (integers matching `bin_id` in the TSV):

```bash
python -m experiments.minnie_column.build_manifest \
  --bbox-json data/minnie_column_bbox_example.json \
  --version 1718 \
  --train-bins 0 \
  --test-bins 1 \
  --out-tsv run_logs/minnie_column_manifest.tsv
```

Columns include: `id` (nucleus id), `pt_root_id`, `center_*_nm`, `bin_id`, `split`, and optional proofreading fields + `difficulty_heuristic`.

Use `--no-proofread` to skip the proofreading table join (faster).

## 3. Tube synapses (optional)

After you have a manifest, gather synapses per root with an axis-aligned tube bbox:

```bash
python -m experiments.minnie_column.fetch_tube_synapses \
  --manifest-tsv run_logs/minnie_column_manifest.tsv \
  --split train \
  --radius-xy-um 15 \
  --z-half-extent-um 40 \
  --version 1718 \
  --out-dir run_logs/minnie_column_synapses \
  --max-rows 20
```

Outputs one `.npz` / `.json` per root (same pattern as `BoxCache` synapse-only), plus a dedup index. See `fetch_tube_synapses.py --help`.

## 4. Attach download URLs (no large downloads)

Enriches the manifest with EM/seg/synapse/skeleton **endpoints** and a JSON sidecar:

```bash
python -m experiments.minnie_column.attach_assets \
  --manifest-tsv run_logs/minnie_column_manifest.tsv \
  --version 1718 \
  --out-tsv run_logs/minnie_column_manifest_assets.tsv \
  --out-sidecar-json run_logs/minnie_column_asset_urls.json
```

See **`docs/minnie_column_downloads.md`** for meshes (CAVE), static SWC naming, and v117 vs 1718 synapse products.

## Module layout

| Module | Role |
|--------|------|
| `spatial.py` | Bin assignment, bbox parsing |
| `paradigm.py` | Easy/medium/hard heuristics, default tube radii |
| `dedup.py` | Stable keys + overlap weights |
| `tubes.py` | Non-cube bbox for CAVE synapse queries |
| `cave_queries.py` | Nucleus + proofread batch queries |
| `build_manifest.py` | CLI for manifest TSV |
| `discover_column_bbox.py` | CLI: bbox from column table + nuclei |
| `attach_assets.py` | CLI: add URL columns + sidecar JSON |
| `asset_urls.py` | Canonical cloud URLs |
