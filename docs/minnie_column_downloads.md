# Attaching MICrONS downloads (Minnie Column experiments)

This note lists **what to download or stream** to pair **geometry** (EM, meshes, skeletons) with **connectivity** (synapses, roots) for `experiments/minnie_column/`.  
Materialization **1718** is the default target; align **segmentation / root IDs** to the same release you use in CAVE.

Official overview: [Static Repositories – MICrONS Tutorial](https://tutorial.microns-explorer.org/static-repositories.html), [Proofreading and Data Quality](https://tutorial.microns-explorer.org/proofreading.html).

---

## 1. Column bbox (required first)

**Preferred:** derive bounds from CAVE so the ROI matches the **Minnie Column** census.

```bash
python -m experiments.minnie_column.discover_column_bbox \
  --version 1718 \
  --margin-um 2 \
  --out-json data/minnie_column_bbox.json \
  --out-nuclei-tsv run_logs/minnie_column_nuclei_for_bbox.tsv
```

This reads **`allen_v1_column_types_slanted_ref`** (nucleus `target_id`s), loads **`nucleus_detection_v0`** for those ids, and writes **`bbox_nm`**. If your release renames the table, pass `--column-table` (e.g. alternatives listed in release manifests).

Then build the nucleus manifest:

```bash
python -m experiments.minnie_column.build_manifest \
  --bbox-json data/minnie_column_bbox.json \
  --version 1718 \
  --bin-width-um 50 --bin-height-um 100 \
  --auto-median-test \
  --out-tsv run_logs/minnie_column_manifest.tsv
```

---

## 2. Synapses

| Product | Use | URL / access |
|--------|-----|----------------|
| **Live CAVE** @1718 | Training labels, `fetch_synapses` in code | `minnie65_public` + `client.materialize.synapse_query` |
| **mat_dbs archived CSV** | Offline full table, bbox filter | `gs://mat_dbs/public/minnie65_phase3_v1/{v}/synapses_pni_2_v1_filtered_view.csv.gz` + header (HTTPS in `asset_urls.py`) |
| **BossDB v117 CSV** | Historical edges (roots pinned to v117 in docs) | `synapse_graph/synapses_pni_2.csv` (~47.5 GB) — [static repos](https://tutorial.microns-explorer.org/static-repositories.html) |

**v117 vs 1718:** use **`scripts/train.py remap-roots`** (or your own `chunkedgraph` map) so the **same physical synapses** get two root label columns.

---

## 3. Electron microscopy (EM)

| Product | CloudVolume URI |
|--------|------------------|
| Minnie65 EM (BossDB) | `precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em` |
| Mirror (GCS) | `precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/em` |

Crop with **bounding boxes in voxel space** at chosen MIP (often **MIP 2**: 32×32×40 nm). Convert from nm using dataset voxel sizes.

---

## 4. Segmentation

| Product | Use |
|--------|-----|
| **Dynamic (graphene)** `minnie65_public` | Meshes, latest roots, Neuronauts-style fetches |
| **Flat static `seg_m1300`** (GCS) | Fixed voxel seg for a **specific** materialization — align version to analysis |

Graphene path (see `asset_urls.GRAPHENE_MINNIE65_PUBLIC`):

`graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public`

---

## 5. Meshes

Meshes are **not** a single URL per neuron in the static bucket; use **CAVE / CloudVolume** meshing:

- Tutorial: [Download Meshes](https://tutorial.microns-explorer.org/quickstart_notebooks/06-cloudvolume-download-mesh.html)
- Typical pattern: mesh client associated with the **segmentation layer** for `minnie65_public` at your materialization **1718**, keyed by **`root_id`** (or supervoxel id depending on API).

**Attach:** store **per-root** mesh path or cache directory after download; the manifest enricher adds a human-readable `asset_mesh_note` column pointing here.

---

## 6. Skeletons (SWC)

Static public trees (proofread vs dendrite-only):

| Set | Prefix |
|-----|--------|
| Proofread (axon included) | `https://storage.googleapis.com/microns-static-links/skel/swc/proofread/` |
| Dendrite (axon stripped) | `https://storage.googleapis.com/microns-static-links/skel/swc/dendrite/` |

Filenames may be **`{root_id}.swc`** or **`{root_id}_{nucleus_id}.swc`** depending on collection. **Listing** the prefix or using **CAVE “download skeletons”** is more reliable than guessing URLs.

**Attach:** `experiments/minnie_column/attach_assets.py` adds **candidate** `asset_skeleton_swc_*_url` columns; verify with `curl -I` or use CAVE.

---

## 7. Enrich manifest with URLs (no downloads)

```bash
python -m experiments.minnie_column.attach_assets \
  --manifest-tsv run_logs/minnie_column_manifest.tsv \
  --version 1718 \
  --out-tsv run_logs/minnie_column_manifest_assets.tsv \
  --out-sidecar-json run_logs/minnie_column_asset_urls.json
```

Optional: `--static-synapse-version 1078` if you want a specific archived gzip while **labels** stay at 1718 via remap.

---

## 8. What your training code should read

1. **`minnie_column_manifest_assets.tsv`** — one row per nucleus + **global URL columns** + per-root skeleton guesses.
2. **`minnie_column_asset_urls.json`** — single copy of global endpoints for pipelines.
3. **Tube synapses** (optional): `fetch_tube_synapses.py` output dirs.
4. **Meshes** — local paths after you run a mesh download script (not committed here).

---

## References

- [Annotation Tables – MICrONS Tutorial](https://tutorial.microns-explorer.org/annotation-tables.html)
- [Static Repositories](https://tutorial.microns-explorer.org/static-repositories.html) (EM, seg, synapse CSV, nucleus CSV)
- [Release manifests](https://tutorial.microns-explorer.org/release_manifests/version-1718.html) (version 1718)
