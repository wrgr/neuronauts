# neuronauts

`neuronauts` is a Python package for end-to-end connectome inference from electron microscopy data. It implements **Neuronauts v2: Scaffolded Global Grammar** — a multi-modal Transformer-GNN architecture that treats reconstruction as a graph-refinement problem over existing CAVE segmentations, evaluated directly against synapse line-graph F1.

## Architecture overview

The system is organized into five learned layers:

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
5. Evaluation          line_graph.py
   Synapse line-graph F1  (primary scalar throughout)
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pytest                            # 284 tests
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

The recommended training path uses `scripts/train.py`, which handles box caching, grammar training, and optional GAT training in one CLI.

### 1. Build a box cache

Random spatial sampling (no static tables needed):

```bash
python scripts/train.py build-dataset \
  --cache-dir data/boxes \
  --n-boxes 50 \
  --min-synapses 15
```

Or from a pre-downloaded nucleus table (see *Static data* below):

```bash
python scripts/train.py build-dataset \
  --cache-dir data/boxes \
  --n-boxes 50 \
  --counts-tsv run_logs/synapse_root_counts_static.tsv \
  --nucleus-csv data/microns_static/v1078/nucleus_detection_v0.csv
```

### 2. Train grammar (fast — no simulation required)

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

### 4. One-shot: build + train

```bash
python scripts/train.py run \
  --cache-dir data/boxes \
  --n-boxes 50 \
  --grammar-output models/shared_grammar.pt \
  --epochs 30
```

### Static data (offline synapse counts)

Download static MICrONS synapse/nucleus tables without CAVE authentication:

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

The current U-Net fuses a central Z-slice with ±2 neighbouring slices as extra input channels, providing 3D context at 2D inference cost.

Train on an external tif dataset:

```bash
python scripts/train_membrane_unet.py \
  --dataset-dir /path/to/unet_data \
  --output models/membrane_unet.pt
```

Cache predictions for a real box:

```bash
python scripts/cache_membrane_volume.py \
  --checkpoint models/membrane_unet.pt \
  --center-nm 1153592,793592,655640 \
  --cache-dir cache/membranes
```

## Shared grammar training (offline datasets)

Export datasets from real boxes and train independently:

```bash
# Merge supervision (synapse-cluster pairs, no simulation)
python scripts/export_merge_dataset.py \
  --output data/merge_dataset.npz --box-indices 0,1,2

# Topology / atomicity supervision (no simulation)
python scripts/export_topology_dataset.py \
  --output data/topology_dataset.npz --box-indices 0,1,2

# Train shared grammar
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

## Project layout

```text
neuronauts/
  agent.py                  Agent config and step logic
  assembly.py               GlobalAssemblyGAT, gat_refine_connectivity,
                              label_graph_edges
  assembly_dataset.py       Hypothesis feature extraction
  dataset_builder.py        BoxCache, select_random_boxes,
                              select_boxes_from_nucleus_table, build_dataset
  dijkstra.py               BridgeGraph (Dijkstra bridge proposals)
  experiment_driver.py      Canonical experiment cycle driver
  fetch.py                  MICrONS fetch, SynapseTable, SyntheticBenchmark,
                              skeleton/mesh feature extractors
  fields.py                 Membrane, exploration, synapse attraction fields
  grammar.py                TorchPathEncoder (Transformer+CLS), MergeScorer,
                              PathBatch, build_multimodal_path_sequence
  hypothesis_reranker.py    Assembly reranker
  line_graph.py             Synapse line-graph F1 (primary metric)
  membrane_unet.py          2.5D MembraneUNet (InstanceNorm2d, context slices)
  merge.py                  MergedNeuron, ConnectivityGraph, union-find merge
  merge_dataset.py          Local merge example construction
  run.py                    Main runner, HeuristicConfig, simulate_paths_and_hits
  shared_grammar_model.py   SharedGrammarModel, BridgeHead, GlobalAssemblyGAT,
                              GATTrainingConfig, gat_train_step,
                              train_global_assembly_gat, multitask_train_step
  synapse_root_counts_static.py  Offline static MICrONS synapse counts
  topology_dataset.py       Atomicity example construction
  topology_model.py         AttentionArborValidator
  training_batches.py       Batch padding utilities
  vectorized.py             Vectorized agent simulation
  viz.py                    Matplotlib visualization helpers

scripts/
  train.py                  ★ End-to-end training CLI (build-dataset / train / run)
  cache_membrane_volume.py  Predict + cache membrane for a real box
  codex_optimize.py         Codex outer-loop optimizer (legacy)
  export_assembly_ranking_dataset.py
  export_merge_dataset.py
  export_topology_dataset.py
  gemini_researcher.py      Gemini outer-loop optimizer (legacy)
  iterative_loop.py         Repeated-evaluation monitor
  plot_iterations.py        Iteration metric plots
  run_research_cycle.py     Canonical research cycle (export→train→validate)
  train_assembly_ranker.py
  train_membrane_unet.py
  train_shared_grammar.py
  train_topology_model.py
  view_research_ledger.py

tests/
  test_assembly.py          Assembly beam search
  test_assembly_ranking.py  Reranker
  test_bridge.py            BridgeGraph, BridgeHead, _propose_bridges
  test_experiment_driver.py
  test_fetch_geometry.py    Skeleton/mesh feature extractors
  test_fields.py            Field computation
  test_gat_assembly.py      _SparseGATLayer, GlobalAssemblyGAT, gat_refine_connectivity
  test_gat_training.py      label_graph_edges, gat_train_step, train_global_assembly_gat
  test_grammar_gaps.py      MergeScorer, build_multimodal_path_sequence, etc.
  test_heuristic_config.py  HeuristicConfig, learned/legacy mode switching
  test_line_graph.py        Line-graph F1
  test_membrane_unet.py     2.5D UNet, InstanceNorm, context slices
  test_merge_learning.py    Merge dataset construction
  test_research_ledger_viewer.py
  test_run.py               Integration / oracle regression
  test_scaffold.py          Scaffold union, seg-ID grouping, viz helpers
  test_shared_grammar_training.py  multitask_train_step, bridge loss
  test_topology_learning.py AttentionArborValidator

docs/
  whitepaper.md             Nature Methods–style paper
  global_inference_roadmap.md  PR-by-PR implementation log
  global_validation_layer.md
  global_validation_dataset.md
  topology_learning_test_plan.md
```

## Key design decisions

| Decision | Rationale |
|---|---|
| Line-graph F1 as primary scalar | Closest box-scale proxy for downstream connectome correctness |
| Coordinate-free path descriptors | Reusable across volumes without registration |
| CAVE scaffold init | Collapses search space ~10× before learned grammar decisions |
| Transformer + [CLS] token | Global fragment embedding; multi-modal fusion of path, skeleton, mesh |
| Soft-F1 surrogate loss for GAT | Differentiable approximation of the terminal metric |
| `HeuristicConfig.learned()` | Spatial thresholds become candidate generators, not hard decisions |
| 2.5D UNet with InstanceNorm2d | 3D context at 2D inference cost; stable for batch size 1 |

## References

- MICrONS Consortium et al. Functional connectomics spanning multiple areas of mouse visual cortex. *Nature* 2021. <https://www.nature.com/articles/s41586-021-03778-x>
- Li, P. H. et al. RoboEM: neurite reconstruction from 3D EM by AI-based direct image-to-trace translation. *Nature Methods* 2024. <https://www.nature.com/articles/s41592-024-02226-5>
- NEURD: Morphology-based proofreading. *Nature* 2025. <https://www.nature.com/articles/s41586-025-08660-5>
- Silversmith, W. `cloud-volume`. <https://github.com/seung-lab/cloud-volume>
- CAVEconnectome. `CAVEclient`. <https://github.com/CAVEconnectome/CAVEclient>
- Bae, J. A. et al. Digital museum of retinal ganglion cells. *Cell* 2024. <https://www.cell.com/cell/fulltext/S0092-8674(24)00308-4>
