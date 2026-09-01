> **Archived 2026-09-01.** Status: superseded. One of five "direction" docs
> that disagreed about the canonical pipeline (`docs/consolidation_plan.md`
> §1.1, §4.4); moved here with `git mv` so history is preserved. Its own
> banner below already said it was superseded by
> [`roadmap_global_assembly.md`](../../roadmap_global_assembly.md); this
> header just records the move. Content below is unchanged from the original
> (its internal relative links still assume the old `docs/` location).

---

## Neuronauts: Global Inference Roadmap

> ⚠️ **Superseded by [`roadmap_global_assembly.md`](../../roadmap_global_assembly.md).**
> Kept for history; see [`../README.md`](../README.md) for the pipeline that runs
> today.

This document tracks the planned transition from the current box-scale grammar and heuristic assembly toward a scaffold-aware, globally optimized connectome inference system.

### PR 1: Transformer-Based Multi-Modal Path Encoding ✓ COMPLETE

- **Goal**: Replace the heuristic "split-into-thirds" pooling in `grammar.py` with a sequence model that fuses path, skeleton, and mesh descriptors.
- **Implemented**: `TorchPathEncoder` now uses `nn.TransformerEncoder` + learned `[CLS]` token. `PathBatch` extended for multi-modal inputs. `fetch.py` gained `skeleton_tortuosity`, `skeleton_stepwise_features`, `mesh_volume_surface_ratio`, `mesh_stepwise_features`.

### PR 2: Trajectory Bridge Head & Dijkstra Proposals ✓ COMPLETE

- **Goal**: Infer neurite continuations across corrupted or missing EM data.
- **Implemented**: `BridgeHead` 6D MLP on `SharedGrammarModel`. `BridgeGraph` + Dijkstra in `neuronauts/dijkstra.py`. `_build_bridge_graph` / `_propose_bridges` in `run.py`. Self-supervised bridge loss in `multitask_train_step`.

### PR 3: Scaffold-Aware Graph Initialization ✓ COMPLETE

- **Goal**: Collapse the search space using CAVE segmentation IDs as noisy scaffold nodes.
- **Implemented**: `SynapseTable` extended with `pre_seg_id` / `post_seg_id`. `fetch_synapses` pulls supervoxel columns. `_scaffold_union_from_seg_ids` pre-merges same-segment agents. 2.5D `MembraneUNet` with `InstanceNorm2d` and context slices. Visualization helpers in `neuronauts/viz.py`.

### PR 4: Global GNN Assembly ✓ COMPLETE

- **Goal**: Replace local beam-style merge search with an explicit Graph Attention Network that operates over scaffold nodes and bridge candidates.
- **Files**: `neuronauts/assembly.py`, `neuronauts/shared_grammar_model.py`, `tests/test_gat_assembly.py`, `tests/test_gat_training.py`.
- **Implemented**:
  - `_SparseGATLayer` + `GlobalAssemblyGAT` (forward pass, `score_edges`, checkpoint I/O) in `shared_grammar_model.py`.
  - `gat_refine_connectivity` in `assembly.py` — applies the trained GAT to filter connectivity graph edges.
  - **Training loop** (`label_graph_edges`, `gat_train_step`, `GATTrainingConfig`, `train_global_assembly_gat`):
    - `label_graph_edges`: per-edge ground-truth labels derived from majority-vote root_id matching.
    - `gat_train_step`: single-example gradient step with combined BCE + soft-F1 surrogate loss; path encoder frozen.
    - `train_global_assembly_gat`: full epoch loop over synthetic ConnectivityGraph examples with train/val split and best-checkpoint saving.
  - 20 new tests (6 `LabelGraphEdgesTest`, 7 `GATTrainStepTest`, 3 `GATTrainingConfigTest`, 4 `TrainGlobalAssemblyGATTest`).

### PR 5: Heuristic Decommissioning ✓ COMPLETE

- **Goal**: Remove deterministic spatial thresholds as decision rules, leaving the learned grammar as the primary arbiter.
- **Implemented**: `HeuristicConfig` dataclass centralises all spatial thresholds. `run()` auto-selects `learned()` mode when any checkpoint is present, promoting spatial filters to candidate-generators. Hardcoded radii and overlap thresholds are configurable defaults only.

---

## Post-PR Real-Data Training Infrastructure

### Real Data Subset Fetcher (`neuronauts/dataset_builder.py`)

- `BoxCache` — disk-backed cache (npz + json per box, index.json manifest).
- `select_random_boxes(n)` — uniform spatial sampling over Minnie65 interior.
- `select_boxes_from_nucleus_table(counts_tsv, nucleus_csv, n)` — soma-centred selection from static synapse count table (produced by `synapse_root_counts_static.py`).
- `build_dataset(specs, cache)` — fetch EM + synapses for each spec, filter by synapse count, skip already-cached.
- `load_dataset(cache_dir)` — load existing cache and return `(BoxCache, records)`.

### End-to-End Training CLI (`scripts/train.py`)

Three subcommands:
- `build-dataset` — fetch and cache real MICrONS boxes.
- `train` — grammar + optional GAT training loop on cached boxes with val/checkpoint.
- `run` — build-dataset then train in one shot.

Training strategy:
- **Grammar** — trained from synapse tables alone (no simulation, ~0.3 s/box).
- **GAT** — optional (`--train-gat`), runs every `--gat-every-n-epochs` epochs using full agent simulation → `_build_graph` → `gat_train_step`.

