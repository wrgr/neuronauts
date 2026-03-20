# Minnie Column experiment paradigm

> Target materialization **1718** (or newer public release). Spatial CV on the **Minnie Column** (~100 × 100 µm footprint in V1, all layers in *z*), with **per-root tube** synapse collection and **easy / medium / hard** training tiers.

See [Proofreading and Data Quality](https://tutorial.microns-explorer.org/proofreading.html) for background on the column and `proofreading_status_and_strategy`.

## Goals

1. **High-quality supervised learning** on proofread-rich tissue before scaling to the full volume.
2. **Learn from segmentation improvement**: compare **historical** (e.g. v117) vs **latest** (1718) root IDs on the same synapses where possible (separate tooling: root remap, static tables).
3. **Honest generalization**: train / val / test splits by **spatial bins** so nuclei (and their arbors) do not leak across splits.

## Column footprint and nucleus query

1. Define **3D bounds** for the Minnie Column in **nanometers** (from release notes, Neuroglancer, or a one-off query).
2. At materialization **1718**, query **`nucleus_detection_v0`** with a **bounding box** on `pt_position` (CAVE `query_table` + `bounding_box`).
3. Each nucleus row yields **`id`** (soma id), **`pt_root_id`**, and **`pt_position`** (convert to nm with `desired_resolution=[1,1,1]` and `split_positions=True`).

## Spatial bins (50 × 100 µm, distinct nuclei)

- Split the **xy** footprint into non-overlapping bins (e.g. **two** bins of **50 × 100 µm** along the short axis of the 100 × 100 µm column, or a **2×2** grid of 50 × 50 µm).
- Assign each nucleus to exactly one bin using **soma (x, y)** only; **z** is unrestricted so full depth stays in one bin.
- **Train / val / test** = **disjoint bin sets** (e.g. bin 0 → train, bin 1 → test). Do not split a single nucleus across bins.

Implementation: `experiments/minnie_column/spatial.py` (`assign_bins_xy`, `train_test_split_by_bin`).

## Per-root synapse gathering and “tubes”

For each **anchor root** (or proofread component):

1. Optionally use **mesh/skeleton** at 1718 to get spatial support per slab; for a first version, use **axis-aligned boxes** that approximate a **tube**: expand a soma-centered or arbor bbox by **lateral radius** \(R\) in xy and extent in z.
2. **Gather synapses** whose **`ctr_pt_position`** (or pre/post) lies inside the tube volume. This pulls candidate synapses near the neuron, including some not yet assigned to that root (useful for merge learning).
3. Record **`tube_radius_xy_um`** per difficulty tier (easy < medium < hard).

Synapse fetch uses existing **`neuronauts.fetch.fetch_synapses`** (CAVE bbox query).

## Easy / medium / hard

| Tier | Roots | Tube | Supervision |
|------|--------|------|-------------|
| **Easy** | Strict `proofreading_status_and_strategy` filters (e.g. dendrite clean + strategy not `none`) | Small lateral radius | 1718 roots only |
| **Medium** | Broader proofread mix | Medium radius | Same + optional auxiliary losses |
| **Hard** | Include weaker proofread / larger tubes | Large radius | Same labels; harder input distribution |

Implementation: `experiments/minnie_column/paradigm.py` (`difficulty_from_row`, `tube_radius_um`).

## Avoiding synapse count and exact-position memorization

When tubes overlap, the **same synapse** can appear under multiple roots or multiple passes.

1. **Dedup key**: `synapse_id` at 1718, or a stable hash of rounded `ctr_pt` in nm (`dedup.synapse_stable_key`).
2. **Do not supervise on** total synapse count per tube.
3. **Weighting**: optionally `1 / tube_degree(synapse)` when a synapse appears in \(k\) tubes (`dedup.tube_overlap_weights`).
4. **Features**: prefer **box-relative** or **anchor-relative** coordinates (aligned with `SynapseTable` in `fetch.py`), not raw dataset-global xyz in the loss.
5. **Sampling**: cap duplicates per epoch or sample each `synapse_id` at most once per task per step.

## Data products

| Artifact | Description |
|----------|-------------|
| **Nucleus manifest** | TSV: `nucleus_id`, `root_id`, `center_nm`, `bin_id`, `split`, proofread columns (if joined) |
| **Synapse index** (future) | Deduped synapse rows + optional v117 roots + `tube_roots` list |
| **Meshes / skeletons** | Fetched separately (CAVE or static GCS); keyed by `root_id` @1718 |

## Code layout

```
experiments/minnie_column/
  README.md           # CLI and env
  spatial.py          # Bins and splits
  paradigm.py         # Easy/medium/hard
  dedup.py            # Stable keys and overlap weights
  tubes.py            # Axis-aligned tube bboxes (nm)
  cave_queries.py     # Nucleus bbox query + optional proofread join
  build_manifest.py   # CLI: build nucleus manifest
```

## References

- [Proofreading and Data Quality – MICrONS Tutorial](https://tutorial.microns-explorer.org/proofreading.html)
- Schneider-Mizell et al., 2025 — Minnie Column census (column definition).
- **Attaching EM, meshes, skeletons, synapse tables:** [`minnie_column_downloads.md`](minnie_column_downloads.md)
