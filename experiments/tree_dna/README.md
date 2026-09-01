# Thread: tree_dna

**Goal.** Learn a per-fragment **morphological embedding** ("tree-DNA") of local
arbor structure — caliber/radius profile, branching, tortuosity, tangent flow —
that is translation-invariant and poolable to a per-neuron signature. It's the
representation the [roadmap](../../docs/roadmap_global_assembly.md) builds global
assembly on: fragments carry global coordinates and a shape signature, so they
can be stitched across box seams.

> **tree-DNA ≠ fingerprints.** tree-DNA is *morphology* (the fragment's own
> shape); [fingerprints](../fingerprints/README.md) is *connectivity* (who it
> synapses with). Two independent cues; different branches.

**Status:** incubating — a core seed lives in this tree; the full encoder is on a
branch (PR #17).

## Where the code is

- **Core seed (merged):** [`neuronauts/path_edge_encoder.py`](../../neuronauts/path_edge_encoder.py)
  (`PathEdgeEncoder` — Transformer over per-step path features) and the skeleton
  featurization / precompute paths in [`neuronauts/cell_graph.py`](../../neuronauts/cell_graph.py).
  This is the morphological-encoder seed already in the pipeline.
- **`claude/tree-dna-phase-1-G1DNn` — PR #17 (open).** Phase 1+2.1: a data-driven
  `SkeletonGNN` DNA encoder (triplet contrastive) + a half-synapse graph
  partitioner under `neuronauts/represent/` and `neuronauts/assemble/`. 211
  commits ahead / 24 behind `main`.
- **Successor:** the PR's own notes say the box-local approach is superseded by
  **`claude/synapse-coassign`** (correlation-clustering co-assignment) — see
  [cell_assignment](../cell_assignment/README.md). Resolve PR #17 by extracting
  the encoder and closing in favor of the co-assignment frontier, or rebasing.

## Run (the merged seed)

```bash
python scripts/train.py train-path-encoder \
  --cache-dir data/boxes_30um \
  --edit-pairs-tsv data/cave_edit_pairs_v3.tsv \
  --edit-chains-npz data/cave_edit_chains_v3.npz \
  --output models/scratch/path_encoder.pt --epochs 10 --seed 42
```

The embedding feeds [cell_assignment](../cell_assignment/README.md) via
`--path-encoder-checkpoint`.

## Checkpoints

`path_encoder_v3*.pt` (best ~0.899 path-discrimination acc) are produced locally
and not tracked — write runs under `models/scratch/`.

## Graduation

Promote when a tree-DNA-only same-neuron predictor beats the 6-scalar-feature
CellGNN baseline on the spatial val/test split (the "leaving signal on the table"
gap in [`docs/TODO.md`](../../docs/archive/2026-09/TODO.md) (archived)), making tree-DNA the primary node
feature for global assembly.
