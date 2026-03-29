# neuronauts

`neuronauts` is a Python package for end-to-end connectome inference from electron microscopy data. It implements **Neuronauts v2: Scaffolded Global Grammar** — a multi-modal Transformer-GNN architecture that treats reconstruction as a graph-refinement problem over existing CAVE segmentations, evaluated directly against synapse line-graph F1.

## Architecture overview

The system is organized into five learned layers plus a global topological merge:

```
EM volume + CAVE synapses
        │
        ▼
1. Perception          fetch.py · fields.py · vectorized.py · membrane_unet.py
   Agent traces, 2.5D membrane U-Net, synapse hits

        ▼
2. Scaffold init       run.py (_scaffold_union_from_seg_ids)
   CAVE seg-IDs pre-group agents → collapse search space 10×

        ▼
3. Shared Grammar      grammar.py · shared_grammar_model.py
   Transformer PathEncoder ([CLS] token, multi-modal fusion)
   MergeScorer · BridgeHead · multitask training (merge + atomicity + bridge)

        ▼
4. Global Assembly     assembly.py · shared_grammar_model.py
   SparseGATLayer · GlobalAssemblyGAT
   Score edges with soft-F1 surrogate → trained end-to-end on line-graph F1

        ▼
5. Topological Merge   cell_graph.py · edit_history.py
   CellGNN: reachability-based GNN embeds synapses via K message-passing rounds
   Cluster embeddings → full cell assignments
   Edit-history pairs from proofreader merges/splits for hard-example training

        ▼
6. Evaluation          line_graph.py
   Synapse line-graph F1  (primary scalar throughout)
```

## Current pipeline status

As of 2026-03-29, the v2 architecture is **fully implemented and tested**.

### What's working

| Component | Module(s) | Status |
|---|---|---|
| Transformer path encoding | `grammar.py` | Multi-modal fusion with [CLS] token |
| Merge scoring + bridge head | `shared_grammar_model.py` | Multitask training (merge + atomicity + bridge) |
| Scaffold-aware init | `run.py` | CAVE seg-ID grouping, ~10× search reduction |
| Global assembly GAT | `assembly.py`, `shared_grammar_model.py` | Sparse attention, soft-F1 surrogate loss |
| CellGNN topological merge | `cell_graph.py` | 40+ tests, tangledness-aware sampling, spatial splits |
| Edit-history supervision | `edit_history.py` | Contrastive pairs from proofreader merges/splits |
| 2.5D membrane U-Net | `membrane_unet.py` | InstanceNorm2d, context slices |
| CAVE data pipeline | `dataset_builder.py`, `fetch.py` | Synapse-seeded boxes, retry with backoff, no-EM mode |
| Training CLI | `scripts/train.py` | 8 subcommands: build-dataset, train, train-cell-gnn, evaluate, sweep, scale-test, remap-roots, run |
| Pipeline inspector | `scripts/inspect_pipeline.py` | 6-stage Neuroglancer visualization |
| Line-graph F1 evaluation | `line_graph.py` | Primary + sampled-pair variants |

### Test coverage

- **42 test files**, covering all major modules
- Critical integration tests for `cmd_train_cell_gnn`, `cmd_evaluate`, `cmd_sweep`, `cmd_scale_test`
- Edge case coverage: degenerate inputs, threshold boundaries, out-of-range indices
- Mocked CAVE integration tests (no network required)

### What's next

1. **Real data training** — Run `fetch_cave_boxes.py` on a machine with CAVE access, then train CellGNN on 50-80 real MICrONS boxes
2. **Head-to-head evaluation** — Compare CellGNN vs grammar baseline on test split
3. **Hyperparameter sweep** — Grid search over `d_model`, `n_layers`, `proximity_radius`, `partition_threshold`
4. **Cell-level plausibility** (Phase 1 of topological merge roadmap) — Re-partition low-atomicity cells

See `docs/TODO.md` for the full prioritized backlog and `docs/global_topological_merge_plan.md` for the 4-phase roadmap.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

Run one synthetic benchmark:

```bash
python -m neuronauts.run
```

Run with a trained grammar checkpoint:

```bash
python -m neuronauts.run \
  --shared-grammar-checkpoint models/shared_grammar_smoke.pt
```

## Training on real MICrONS data

The recommended training path uses `scripts/train.py`, which handles box caching, grammar training, CellGNN training, evaluation, and sweeps in one CLI.

### 1. Build a box cache (CAVE-only, no token required)

```bash
pip install caveclient

python scripts/fetch_cave_boxes.py \
  --cache-dir data/boxes_cave \
  --n-boxes 80 \
  --no-em \
  --min-positive-pairs 5
```

Or via the training CLI with a specific strategy:

```bash
# Synapse-seeded (default, recommended)
python scripts/train.py build-dataset \
  --cache-dir data/boxes \
  --n-boxes 80 \
  --strategy synapse-seeded \
  --no-em \
  --min-positive-pairs 5

# From static nucleus table (fully offline)
python scripts/train.py build-dataset \
  --cache-dir data/boxes \
  --n-boxes 50 \
  --counts-tsv run_logs/synapse_root_counts_static.tsv \
  --nucleus-csv data/microns_static/v1078/nucleus_detection_v0.csv

# Proofread-core (highest quality, requires CAVE token)
python scripts/train.py build-dataset \
  --cache-dir data/boxes_proofread \
  --strategy proofread-core \
  --proofread-n-roots 30 \
  --proofread-radius-um 40 \
  --no-em \
  --cave-version 1412 \
  --cave-token "$CAVE_TOKEN"
```

### 2. Train grammar (fast, no simulation required)

```bash
python scripts/train.py train \
  --cache-dir data/boxes \
  --grammar-output models/shared_grammar.pt \
  --epochs 30
```

### 3. Train grammar + GAT (adds agent simulation)

```bash
python scripts/train.py train \
  --cache-dir data/boxes \
  --grammar-output models/shared_grammar.pt \
  --gat-output models/gat.pt \
  --epochs 30 \
  --train-gat \
  --gat-every-n-epochs 5
```

### 4. Train CellGNN (topological merge)

```bash
python scripts/train.py train-cell-gnn \
  --cache-dir data/boxes \
  --epochs 50 \
  --d-model 64 \
  --n-layers 3 \
  --cell-gnn-output models/cell_gnn.pt \
  --log-dir run_logs/cell_gnn
```

With edit-history supervision:

```bash
python scripts/train.py train-cell-gnn \
  --cache-dir data/boxes \
  --epochs 50 \
  --cell-gnn-output models/cell_gnn.pt \
  --edit-pairs-tsv data/edit_pairs.tsv \
  --edit-weight 2.0
```

### 5. Evaluate CellGNN vs baseline

```bash
python scripts/train.py evaluate \
  --cache-dir data/boxes \
  --cell-gnn-checkpoint models/cell_gnn.pt \
  --split test \
  --log-dir run_logs/eval
```

### 6. Hyperparameter sweep

```bash
python scripts/train.py sweep \
  --cache-dir data/boxes \
  --d-models "32,64,128" \
  --n-layers-list "2,3,4" \
  --proximity-radii "2000,5000,10000" \
  --partition-thresholds "0.3,0.5,0.7" \
  --epochs 30 \
  --best-output models/cell_gnn_sweep_best.pt
```

### 7. Scale profiling

```bash
python scripts/train.py scale-test \
  --cache-dir data/boxes \
  --cell-gnn-checkpoint models/cell_gnn.pt \
  --min-synapses 50 \
  --n-boxes 10
```

### 8. One-shot: build + train

```bash
python scripts/train.py run \
  --cache-dir data/boxes \
  --n-boxes 50 \
  --grammar-output models/shared_grammar.pt \
  --epochs 30
```

### 9. Proofread-core with root remapping

```bash
python scripts/train.py build-dataset \
  --cache-dir data/proofread_core_v117 \
  --strategy proofread-core \
  --cave-version 117 \
  --proofread-n-roots 50 \
  --proofread-radius-um 40

python scripts/train.py remap-roots \
  --cache-dir data/proofread_core_v117 \
  --base-version 117 \
  --target-version 1412 \
  --output data/proofread_core_v117/root_remap.tsv

python scripts/train.py train \
  --cache-dir data/proofread_core_v117 \
  --base-version 117 \
  --target-version 1412 \
  --root-remap-tsv data/proofread_core_v117/root_remap.tsv \
  --grammar-output models/shared_grammar_real.pt \
  --epochs 30
```

### Static data (offline synapse counts)

```bash
python -m neuronauts.synapse_root_counts_static \
  --version 1078 \
  --static-dir data/microns_static \
  --output run_logs/synapse_root_counts_static.tsv
```

## Install extras

```bash
pip install -e ".[membrane]"   # 2.5D membrane U-Net (InstanceNorm2d)
pip install -e ".[topology]"   # torch grammar + GAT training
```

## Membrane U-Net (2.5D)

The current U-Net fuses a central Z-slice with +/-2 neighbouring slices as extra input channels, providing 3D context at 2D inference cost.

```bash
python scripts/train_membrane_unet.py \
  --dataset-dir /path/to/unet_data \
  --output models/membrane_unet.pt

python scripts/cache_membrane_volume.py \
  --checkpoint models/membrane_unet.pt \
  --center-nm 1153592,793592,655640 \
  --cache-dir cache/membranes
```

## Shared grammar training (offline datasets)

Export datasets from real boxes and train independently:

```bash
python scripts/export_merge_dataset.py \
  --output data/merge_dataset.npz --box-indices 0,1,2

python scripts/export_topology_dataset.py \
  --output data/topology_dataset.npz --box-indices 0,1,2

python scripts/train_shared_grammar.py \
  --merge-dataset data/merge_dataset.npz \
  --topology-dataset data/topology_dataset.npz \
  --output models/shared_grammar.pt
```

## Assembly hypothesis reranker

```bash
python scripts/export_assembly_ranking_dataset.py \
  --output data/assembly_ranking.npz \
  --cases 3 \
  --thresholds=-0.5,0.0,0.5 \
  --beam-widths=1,2,4

python scripts/train_assembly_ranker.py \
  --dataset data/assembly_ranking.npz \
  --output models/assembly_reranker.npz
```

## Visualization

```python
from neuronauts.viz import plot_scaffold_groups, plot_bridge_proposals, plot_f1_history

fig = plot_scaffold_groups(synapses, group_map, title="Scaffold init")
fig = plot_bridge_proposals(synapses, proposals)
fig = plot_f1_history(history["val_f1"])
```

Pipeline inspector with Neuroglancer:

```bash
python scripts/inspect_pipeline.py \
  --center-nm 1153592,793592,655640 \
  --side-um 6.0
```

## Project layout

```text
neuronauts/
  __init__.py               Public API: Agent, ConnectivityGraph, MergedNeuron,
                              BridgeGraph, LineGraphMetrics, evaluate, UnionFind
  agent.py                  Agent config and step logic
  assembly.py               GlobalAssemblyGAT, gat_refine_connectivity,
                              label_graph_edges
  assembly_dataset.py       Hypothesis feature extraction
  cell_graph.py             CellGNN: synapse-level GNN for topological merge,
                              build_synapse_graph, infer_cells,
                              partition_from_embeddings, train_cell_gnn
  cave_root_mapping.py      Root ID mapping across materialization versions
  dataset_builder.py        BoxCache, select_synapse_seeded_boxes,
                              select_boxes_from_nucleus_table, build_dataset
  dijkstra.py               BridgeGraph (Dijkstra bridge proposals)
  edit_history.py            Proofreader merge/split pairs for CellGNN training
  experiment_driver.py      Canonical experiment cycle driver
  fetch.py                  MICrONS fetch, SynapseTable, SyntheticBenchmark,
                              skeleton/mesh feature extractors, retry with backoff
  fields.py                 Membrane, exploration, synapse attraction fields
  grammar.py                TorchPathEncoder (Transformer+CLS), MergeScorer,
                              PathBatch, build_multimodal_path_sequence
  helpers.py                UnionFind, safe_normalize, pairwise_edges
  hypothesis_reranker.py    Assembly reranker
  line_graph.py             Synapse line-graph F1 (primary metric)
  membrane_unet.py          2.5D MembraneUNet (InstanceNorm2d, context slices)
  merge.py                  MergedNeuron, ConnectivityGraph, union-find merge
  merge_dataset.py          Local merge example construction
  run.py                    Main runner, HeuristicConfig, simulate_paths_and_hits
  shared_grammar_model.py   SharedGrammarModel, BridgeHead, GlobalAssemblyGAT,
                              GATTrainingConfig, gat_train_step,
                              train_global_assembly_gat, multitask_train_step
  skeleton_graph.py         Skeleton-backed graph with leakage guards
  synapse_root_counts_static.py  Offline static MICrONS synapse counts
  topology_dataset.py       Atomicity example construction
  topology_model.py         AttentionArborValidator
  training_batches.py       Batch padding utilities
  vectorized.py             Vectorized agent simulation
  viz.py                    Matplotlib visualization helpers
  _scipy_compat.py          Scipy fallbacks (cKDTree, cdist, gaussian_filter, sobel)

scripts/
  train.py                  End-to-end training CLI (8 subcommands)
  fetch_cave_boxes.py       Standalone CAVE box fetcher (no token required)
  inspect_pipeline.py       6-stage pipeline inspector with Neuroglancer
  cache_membrane_volume.py  Predict + cache membrane for a real box
  export_assembly_ranking_dataset.py
  export_merge_dataset.py
  export_topology_dataset.py
  train_assembly_ranker.py
  train_membrane_unet.py
  train_shared_grammar.py
  train_topology_model.py
  run_research_cycle.py     Canonical research cycle (export -> train -> validate)
  validate_viz.py           Grammar validation visualizer with Neuroglancer
  codex_optimize.py         Codex outer-loop optimizer (legacy)
  gemini_researcher.py      Gemini outer-loop optimizer (legacy)

tests/                      42 test files
  test_pipeline_commands.py  CellGNN subcommand integration (23 tests)
  test_cell_graph.py         CellGNN core (40+ tests)
  test_gat_training.py       GAT training (20+ tests)
  test_run.py                Integration / oracle regression
  ...                        (see tests/ directory for full listing)

docs/
  TODO.md                   Prioritized open items
  whitepaper.md             Nature Methods-style paper
  global_topological_merge_plan.md  4-phase CellGNN roadmap
  global_inference_roadmap.md       PR-by-PR implementation log
  ...

experiments/
  minnie_column/            Minnie Column ROI tools
  soma_graph/               Soma-based connectivity graph
  root_neighborhood_dataset.py  Proofread-core dataset builder
```

## Key design decisions

| Decision | Rationale |
|---|---|
| Line-graph F1 as primary scalar | Closest box-scale proxy for downstream connectome correctness |
| Coordinate-free path descriptors | Reusable across volumes without registration |
| CAVE scaffold init | Collapses search space ~10x before learned grammar decisions |
| Transformer + [CLS] token | Global fragment embedding; multi-modal fusion of path, skeleton, mesh |
| Soft-F1 surrogate loss for GAT | Differentiable approximation of the terminal metric |
| `HeuristicConfig.learned()` | Spatial thresholds become candidate generators, not hard decisions |
| 2.5D UNet with InstanceNorm2d | 3D context at 2D inference cost; stable for batch size 1 |
| CellGNN with tangledness sampling | Focuses training on hard examples where cells overlap spatially |
| Edit-history contrastive pairs | Proofreader merges/splits provide ground-truth hard negatives |
| Retry with exponential backoff | Network resilience for CAVE data fetching |

## References

- MICrONS Consortium et al. Functional connectomics spanning multiple areas of mouse visual cortex. *Nature* 2021.
- Li, P. H. et al. RoboEM: neurite reconstruction from 3D EM by AI-based direct image-to-trace translation. *Nature Methods* 2024.
- NEURD: Morphology-based proofreading. *Nature* 2025.
- Silversmith, W. `cloud-volume`. https://github.com/seung-lab/cloud-volume
- CAVEconnectome. `CAVEclient`. https://github.com/CAVEconnectome/CAVEclient
- Bae, J. A. et al. Digital museum of retinal ganglion cells. *Cell* 2024.
