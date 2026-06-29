# Thread: fingerprints (tree-DNA)

**Goal.** Learn a per-fragment **morphological fingerprint** — a coordinate-free
embedding of local arbor structure (step deltas, skeleton features, tangent
flow) that says whether two pieces of neurite belong to the same cell. This is
the "tree-DNA" of [`docs/roadmap_global_assembly.md`](../../docs/roadmap_global_assembly.md):
the representation meant to replace the collapsed scalar edge features and,
eventually, to stitch neurons across box seams.

**Status:** incubating (core thread). The encoder and its training signal exist
and are tested; promoting it to the primary node representation (and the
cross-region stitch) is the open roadmap work.

## Code (lives in core)

| Module | Role |
|--------|------|
| [`neuronauts/path_edge_encoder.py`](../../neuronauts/path_edge_encoder.py) | `PathEdgeEncoder` — Transformer over per-step path features → fixed-size embedding |
| [`neuronauts/path_dataset.py`](../../neuronauts/path_dataset.py) | path-discrimination dataset + `train_path_encoder` |
| skeleton featurization in [`neuronauts/cell_graph.py`](../../neuronauts/cell_graph.py) | `precompute_self_skeletons_for_cache`, `precompute_skeleton_paths_for_cache` |

## Run

```bash
python scripts/train.py train-path-encoder \
  --cache-dir data/boxes_30um \
  --edit-pairs-tsv data/cave_edit_pairs_v3.tsv \
  --edit-chains-npz data/cave_edit_chains_v3.npz \
  --output models/scratch/path_encoder.pt --epochs 10 --seed 42
```

The fingerprint feeds the [grammar](../grammar/README.md) and
[cell_assignment](../cell_assignment/README.md) threads via
`--path-encoder-checkpoint`.

## Checkpoints

`path_encoder_v3*.pt` (best ~0.899 path-discrimination acc) are produced locally
and not tracked — write new runs under `models/scratch/`. See
[`models/README.md`](../../models/README.md).

## Graduation

Promote when a fingerprint-only same-neuron predictor beats the 6-scalar-feature
CellGNN baseline on the spatial val/test split (the "leaving signal on the table"
gap in [`docs/TODO.md`](../../docs/TODO.md)). That makes tree-DNA the primary node
feature for [cell_assignment](../cell_assignment/README.md).
