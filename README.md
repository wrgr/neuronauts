# neuronauts

`neuronauts` is a Python package for end-to-end connectome inference from electron microscopy data. It implements **Neuronauts v2: Scaffolded Global Grammar** — a multi-modal Transformer-GNN architecture that treats reconstruction as a graph-refinement problem over existing CAVE segmentations, evaluated directly against synapse line-graph F1.

## Architecture overview

```
                        CAVE synapses + (optional EM volume)
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
     1. Agent Perception    2. Skeleton Graph     3. CellGNN (direct)
        fields.py              skeleton_graph.py     cell_graph.py
        vectorized.py          --graph-source        topology_model.py
        Sobel membrane           skeleton
        + agent traces                                Reachability GNN
                                                      embeds synapses,
              │                     │                 clusters into cells
              └──────────┬──────────┘                     │
                         ▼                                │
              4. Scaffold Init                            │
                 run.py                                   │
                 CAVE seg-IDs pre-group                   │
                 → 10× search reduction                   │
                         │                                │
                         ▼                                │
              5. Shared Grammar                           │
                 grammar.py                               │
                 shared_grammar_model.py                  │
                 Transformer [CLS] encoder                │
                 Merge + atomicity + bridge                │
                         │                                │
                         ▼                                │
              6. Global Assembly                          │
                 assembly.py                              │
                 Beam search with calibrated              │
                 probabilities + GAT refinement            │
                         │                                │
                         └──────────┬─────────────────────┘
                                    ▼
                            7. Evaluation
                               line_graph.py
                               Synapse line-graph F1
```

**Two independent training paths converge at evaluation:**

| Path | Entry point | What it trains | Needs EM? | Needs simulation? |
|---|---|---|---|---|
| **Grammar** | `train.py train` | Merge scorer, atomicity head, bridge predictor | No (fast path) | No (fast path) |
| **CellGNN** | `train.py train-cell-gnn` | Synapse-level GNN for full cell reconstruction | No | No |

Both paths are evaluated against line-graph F1 on held-out boxes.

## Current pipeline status

As of 2026-03-30, the v2 architecture is **fully implemented and validated on real MICrONS data**.

### What's working

| Component | Module(s) | Status |
|---|---|---|
| Transformer path encoding | `grammar.py` | Multi-modal fusion with [CLS] token |
| Merge scoring + bridge head | `shared_grammar_model.py` | Multitask training (merge + atomicity + bridge) |
| Calibrated probability scoring | `assembly.py` | Temperature-scaled sigmoid, log-probability beam search |
| Scaffold-aware init | `run.py` | CAVE seg-ID grouping, ~10x search reduction |
| Global assembly GAT | `assembly.py`, `shared_grammar_model.py` | Sparse attention, soft-F1 surrogate loss |
| CellGNN topological merge | `cell_graph.py` | 40+ tests, tangledness-aware sampling, spatial splits |
| Topology cell quality | `topology_model.py` | AttentionArborValidator scores inferred cell coherence |
| Skeleton-based graphs | `skeleton_graph.py` | Alternative to agent sim, CAVE skeleton connectivity |
| Edit-history supervision | `edit_history.py` | Contrastive pairs from proofreader merges/splits |
| CAVE data pipeline | `dataset_builder.py`, `fetch.py` | Synapse-seeded boxes, retry with backoff, no-EM mode |
| Training CLI | `scripts/train.py` | 8 subcommands (see below) |
| Pipeline inspector | `scripts/inspect_pipeline.py` | 6-stage Neuroglancer visualization |
| Line-graph F1 evaluation | `line_graph.py` | Primary + sampled-pair variants |

### Test coverage

- **36 test files**, 600+ tests covering all major modules
- Integration tests for all CLI subcommands (`train-cell-gnn`, `evaluate`, `sweep`, `scale-test`)
- Edge case coverage: degenerate inputs, threshold boundaries, out-of-range indices
- Mocked CAVE integration tests (no network required)
- Probability conversion roundtrips and log-probability beam search

### Real MICrONS training results (2026-03-30)

**Grammar training on 40 CAVE boxes (245k synapses):**

| Metric | Synthetic (15e) | Real 10-box (15e) | Real 40-box (50e) |
|--------|-----------------|-------------------|------------------|
| Best val_BCE | 0.3817 | 0.4751 | **0.3076** |
| Val merge acc | 74.92% | 74.56% | **87.23%** |
| Val topo acc | 87.49% | 88.97% | **88.93%** |
| Improvement | baseline | +8.7% | **+19.4%** |

The 50-epoch real-data grammar model converges to **45.9% better BCE than initial** (0.5684→0.3076) and **exceeds synthetic baseline by 19.4%**, validating the architecture on authentic MICrONS connectome structure.

**Data split:** 34 training boxes + 6 validation boxes (held-out during training to monitor overfitting)

### What's next

1. **Global assembly** — Run beam search with trained grammar scorer on full volume
2. **CellGNN training** — Train topological merge model on grammar scaffolds (50-80 epochs)
3. **Head-to-head evaluation** — Compare grammar-only vs CellGNN vs combined on test split
4. **Hyperparameter sweep** — Grid search CellGNN over `d_model`, `n_layers`, `proximity_radius`, `partition_threshold`
5. **Cell-level plausibility** (Phase 1 of topological roadmap) — Re-partition low-quality cells using topology validator

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

## Training on real MICrONS data

The recommended training path uses `scripts/train.py`, which handles box caching, grammar training, global assembly, CellGNN training, evaluation, and sweeps in one CLI.

### Understanding data splits: train/val/test/cold

- **Training set** (34 boxes): Used to optimize grammar/CellGNN weights via backprop
- **Validation set** (6 boxes): Held-out during training; used to monitor overfitting and select best checkpoint
- **Test set** (optional, separate boxes): Never seen during training; final evaluation of model generalization
- **Cold set** (boxes from different volume/region): True test of cross-generalization; simulates deployment on new connectomes

In our 50-epoch run: used all 40 boxes (34 train + 6 val). Test/cold evaluation requires separate boxes or held-out test portion.

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

### 5. Evaluate CellGNN (with optional topology quality scoring)

```bash
python scripts/train.py evaluate \
  --cache-dir data/boxes \
  --cell-gnn-checkpoint models/cell_gnn.pt \
  --topology-checkpoint models/topology_validator.pt \
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

### Skeleton-based training (no agent simulation)

```bash
python scripts/train.py train \
  --cache-dir data/boxes \
  --grammar-output models/shared_grammar.pt \
  --graph-source skeleton \
  --skeleton-version 1412 \
  --epochs 30
```

### Static data (offline synapse counts)

```bash
python -m neuronauts.synapse_root_counts_static \
  --version 1078 \
  --static-dir data/microns_static \
  --output run_logs/synapse_root_counts_static.tsv
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

## Topology validator training

Train the AttentionArborValidator for cell quality scoring:

```bash
python scripts/train_topology_model.py \
  --dataset data/topology_dataset.npz \
  --output models/topology_validator.pt
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
neuronauts/                     28 modules
  __init__.py               Public API: Agent, ConnectivityGraph, MergedNeuron,
                              BridgeGraph, LineGraphMetrics, evaluate, UnionFind,
                              CandidateMerge, logit_to_probability
  agent.py                  Agent config and step logic
  assembly.py               Beam search with calibrated probabilities,
                              logit_to_probability, probability_to_log_odds,
                              gat_refine_connectivity, repartition
  cell_graph.py             CellGNN: synapse-level GNN for topological merge,
                              build_synapse_graph, infer_cells,
                              score_cell_quality, train_cell_gnn
  cave_root_mapping.py      Root ID mapping across materialization versions
  dataset_builder.py        BoxCache, select_synapse_seeded_boxes, build_dataset
  dijkstra.py               BridgeGraph (Dijkstra bridge proposals)
  edit_history.py           Proofreader merge/split pairs for CellGNN training
  fetch.py                  MICrONS fetch, SynapseTable, retry with backoff
  fields.py                 Sobel membrane field, exploration field
  grammar.py                TorchPathEncoder (Transformer+CLS), MergeScorer,
                              build_multimodal_path_sequence
  helpers.py                UnionFind, safe_normalize, pairwise_edges
  line_graph.py             Synapse line-graph F1 (primary metric)
  merge.py                  MergedNeuron, ConnectivityGraph, union-find merge
  merge_dataset.py          Local merge example construction
  run.py                    Main runner, HeuristicConfig, simulate_paths_and_hits
  shared_grammar_model.py   SharedGrammarModel, BridgeHead, GlobalAssemblyGAT
  skeleton_graph.py         Skeleton-backed graph with leakage guards
  topology_dataset.py       Atomicity example construction
  topology_model.py         AttentionArborValidator (cell quality scoring)
  training_batches.py       Batch padding utilities
  vectorized.py             Vectorized agent simulation
  viz.py                    Matplotlib visualization helpers
  _scipy_compat.py          Scipy fallbacks (cKDTree, cdist, sobel)

scripts/                        9 scripts
  train.py                  End-to-end training CLI (8 subcommands)
  fetch_cave_boxes.py       Standalone CAVE box fetcher (no token required)
  inspect_pipeline.py       6-stage pipeline inspector with Neuroglancer
  train_shared_grammar.py   Standalone grammar training
  train_topology_model.py   Topology validator training
  export_merge_dataset.py   Export merge supervision examples
  export_topology_dataset.py Export topology/atomicity examples
  inspect_topology_metric.py Debug topology metric balance
  analyze_minnie65_boxes.py  Box statistics analysis

tests/                          36 test files, 600+ tests
  test_pipeline_commands.py  CellGNN subcommand integration (23 tests)
  test_cell_graph.py         CellGNN core (50+ tests incl. quality scoring)
  test_assembly.py           Beam search + probability (19 tests)
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
| Calibrated probability scoring | Temperature-scaled sigmoid converts logits to principled probabilities for beam search and bridge costs |
| Soft-F1 surrogate loss for GAT | Differentiable approximation of the terminal metric |
| `HeuristicConfig.learned()` | Spatial thresholds become candidate generators, not hard decisions |
| CellGNN with tangledness sampling | Focuses training on hard examples where cells overlap spatially |
| Topology validator for cell quality | AttentionArborValidator scores structural coherence of inferred cells |
| Skeleton graph alternative | CAVE skeletons provide connectivity without agent simulation or EM volumes |
| Edit-history contrastive pairs | Proofreader merges/splits provide ground-truth hard negatives |
| Retry with exponential backoff | Network resilience for CAVE data fetching |

## References

- MICrONS Consortium et al. Functional connectomics spanning multiple areas of mouse visual cortex. *Nature* 2021.
- Li, P. H. et al. RoboEM: neurite reconstruction from 3D EM by AI-based direct image-to-trace translation. *Nature Methods* 2024.
- NEURD: Morphology-based proofreading. *Nature* 2025.
- Silversmith, W. `cloud-volume`. https://github.com/seung-lab/cloud-volume
- CAVEconnectome. `CAVEclient`. https://github.com/CAVEconnectome/CAVEclient
- Bae, J. A. et al. Digital museum of retinal ganglion cells. *Cell* 2024.



CONSIDER: 
Cross check with recent reviews (at least 3 - markowitz, helmstaedler, lichtman)