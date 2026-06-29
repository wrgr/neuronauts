# neuronauts

`neuronauts` reconstructs neurons from electron microscopy connectome data. Given a CAVE synapse table, it learns to assign synapses to cells and evaluates the result as synapse line-graph F1.

The active pipeline requires **no EM volume** and **no agent simulation** — it runs entirely on synapse positions and CAVE root IDs.

## Conceptual pipeline (single-line view)

**Data curation → supervision mining from proofreading lineage → path-level representation learning → cell-assignment learning → graph-level evaluation.**

## Pipeline stages: action, inputs, outputs

| Stage | Action | Primary input(s) | Primary output(s) | CLI |
|---|---|---|---|---|
| 0. Box cache | Build/use spatially chunked synapse cache from CAVE. | CAVE synapse table (`minnie65_public`) | `data/boxes_30um/*.json` + `*.npz` | (pre-existing cache in repo) |
| 1. Edit supervision mining | Derive false-merge / false-split pairs from proofreading lineage (`v117 -> v1412`). Preferred path is cache-based mining. | Box cache + CAVE lineage APIs + token | `data/cave_edit_pairs_*.tsv`, `data/cave_edit_chains_*.npz` | `fetch-cave-edits-from-cache` (preferred), `fetch-cave-edits` |
| 2. Path encoder pretraining | Train transformer to discriminate coherent synapse-path continuations from spliced negatives. | Box cache + edit TSV/NPZ | `models/path_encoder*.pt` (+ best ckpt) | `train-path-encoder` |
| 3A. Grammar merge scorer | Train pairwise merge classifier (`PathEdgeEncoder + MergeScorer`) using cached synapses/chains. | Box cache (optionally path features via shared components) | `models/grammar_*.pt` | `train` |
| 3B. CellGNN | Train K-hop synapse graph model for cell membership; optionally fuse frozen pretrained path embeddings. | Box cache (+ optional path encoder ckpt) | `models/cell_gnn*.pt` | `train-cell-gnn` |
| 4. Evaluation | Compute line-graph F1 (CellGNN route); report merge accuracy for grammar route. | Test split boxes + checkpoints | Metrics JSON/logs (`logs/*/evaluate_results.json`) | `evaluate` |

## Dataflow diagram

```
CAVE synapse table
  -> [Stage 0] Box cache (30 um windows)
  -> [Stage 1] Edit supervision mining (v117->v1412 lineage)
  -> [Stage 2] Path encoder pretraining (path discrimination)
  -> [Stage 3A] Grammar merge scorer      \
  -> [Stage 3B] CellGNN (optional path features)  ---> [Stage 4] evaluation
```

## Current results  *(updated 2026-05-01)*

| Model | Val merge acc | Test line-graph F1 |
|-------|--------------|-------------------|
| Grammar (ep10/10 ✓) | **85.6%** val merge acc | N/A — see note below |
| CellGNN baseline (ep2/10, training) | — | TBD |
| CellGNN v3 + path encoder (ep2/10, training) | — | TBD |

**Metric note:**
- `evaluate` currently reports **line-graph F1** for CellGNN checkpoints.
- Grammar checkpoints are currently tracked by **pairwise merge accuracy** during training.
- A grammar-only line-graph F1 in the historical beam-search sense would require the full agent/EM route, which is not part of the current no-EM pipeline.

## Reproducible runbook (stage-by-stage)

## Prerequisites

```bash
pip install -r requirements-dev.txt    # or: make setup  (dev + topology + cave extras)

# CAVE token (for fetching data; not needed to train on an existing cache)
mkdir -p ~/.cloudvolume/secrets
echo '{"token": "YOUR_TOKEN"}' > ~/.cloudvolume/secrets/cave-secret.json
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md#dev-setup) for smaller installs and the
caveclient note; [`docs/CAVE_AUTHENTICATION_SETUP.md`](docs/CAVE_AUTHENTICATION_SETUP.md)
for token setup.

### Stage 0 — Check the box cache

247 boxes are already cached at `data/boxes_30um/`.

```bash
ls data/boxes_30um/ | wc -l   # should be ~494 (json + npz per box)
```

### Stage 1 — Fetch CAVE edit pairs (preferred: from cache)

Preferred command (spatially stratified over local cache; avoids row-cap limitations):

```bash
python scripts/train.py fetch-cave-edits-from-cache \
  --cache-dir data/boxes_30um \
  --min-synapses-per-root 8 \
  --output-tsv data/cave_edit_pairs_v3.tsv \
  --output-chains data/cave_edit_chains_v3.npz
```

Legacy direct-sampling command:

```bash
python scripts/train.py fetch-cave-edits \
  --n-sample 50000 \
  --max-false-merges 99999 \
  --min-synapses-per-root 8 \
  --output-tsv data/cave_edit_pairs_v3.tsv \
  --output-chains data/cave_edit_chains_v3.npz
```

### Stage 2 — Train path encoder

```bash
python scripts/train.py train-path-encoder \
  --cache-dir data/boxes_30um \
  --epochs 10 \
  --edit-pairs-tsv data/cave_edit_pairs_v3.tsv \
  --edit-chains-npz data/cave_edit_chains_v3.npz \
  --max-examples-per-epoch 50000 \
  --output models/path_encoder_v3.pt \
  --checkpoint-every 2 \
  --seed 42
```

### Stage 3A — Train grammar

```bash
python scripts/train.py train \
  --cache-dir data/boxes_30um \
  --epochs 10 \
  --grammar-output models/grammar_30um_v1.pt
```

### Stage 3B — Train CellGNN

**Baseline** (no path encoder):

```bash
python scripts/train.py train-cell-gnn \
  --cache-dir data/boxes_30um \
  --epochs 10 \
  --cell-gnn-output models/cell_gnn_30um_v1.pt \
  --checkpoint-every 2 \
  --n-layers 2 \
  --seed 42
```

**With pretrained path encoder** (recommended):

```bash
python scripts/train.py train-cell-gnn \
  --cache-dir data/boxes_30um \
  --epochs 10 \
  --path-encoder-checkpoint models/path_encoder_v3_ep8.pt \
  --pretrained-path-emb-dim 16 \
  --cell-gnn-output models/cell_gnn_v3.pt \
  --checkpoint-every 2 \
  --n-layers 2 \
  --seed 42
```

### Stage 4 — Evaluate

```bash
# CellGNN (+ optional grammar baseline branch when provided)
python scripts/train.py evaluate \
  --cache-dir data/boxes_30um \
  --cell-gnn-checkpoint models/cell_gnn_v3.pt \
  --split test
```

## How this maps to literature (RoboEM / Neurd style framing)

This section is a **conceptual mapping**, not a claim of architectural equivalence.

- **RoboEM-style systems** generally emphasize EM-image-driven tracing / policy execution over voxel fields.
  - `neuronauts` currently **does not** use that online tracing loop in its active pipeline.
  - Instead, it shifts supervision to **proofreading lineage edits** plus synapse geometry, then learns merge/cell assignment in graph space.

- **Neurd-style workflows** (broadly, proofreading-oriented reconstruction stacks) emphasize error discovery/correction loops and morphology-aware constraints.
  - `neuronauts` similarly leans on **proofreading signal** (false merges/splits from version deltas).
  - The current active path is narrower: synapse-level graph assignment + line-graph metrics, with EM/skeleton/topology modules present but not in the default training loop.

### Practical takeaway for comparison studies

For clean apples-to-apples experiments, treat `neuronauts` as a **synapse-graph + lineage-supervision baseline** and compare against EM-policy systems along three axes:
1. **Input modality:** synapse table only vs dense EM volumes.
2. **Supervision source:** proofreading lineage events vs manual traces/labels.
3. **Output target:** synapse clustering / merge decisions vs full neurite trajectory reconstruction.

## Run tests

```bash
pytest
pytest tests/test_cell_graph.py
```

## Saved checkpoints

Curated, representative checkpoints are tracked in `models/` and catalogued, with
metrics and provenance, in [`models/README.md`](models/README.md). Highlights:

| File | Description |
|------|-------------|
| `models/grammar_cave_real_50.pt` | Grammar, real boxes, **87.2% val merge acc** |
| `models/shared_grammar_raw_skel_gat50e.pt` | Shared grammar + GAT, `raw_delta3+skeleton` |
| `models/cell_gnn_seg.pt` | CellGNN, best **test line-graph F1 0.272** @ t=0.99 |
| `models/cell_gnn_real.pt` | CellGNN, first real-CAVE no-EM baseline |

Path-encoder checkpoints (`path_encoder_v3*.pt`) are produced locally and not
tracked; write new runs under `models/scratch/` (git-ignored).

## Key files

| Module | Purpose |
|--------|---------|
| `neuronauts/path_dataset.py` | `fetch_cave_false_merge_chains`, `train_path_encoder` |
| `neuronauts/grammar.py` | `PathEdgeEncoder`, `MergeScorer`, `ArborEncoder` |
| `neuronauts/cell_graph.py` | `CellGNN`, `build_synapse_graph`, `train_cell_gnn` |
| `neuronauts/shared_grammar_model.py` | `SharedGrammarModel`, multitask training |
| `neuronauts/dataset_builder.py` | `BoxCache`, box fetching and caching |
| `neuronauts/line_graph.py` | Line-graph F1 evaluation metric |
| `scripts/train.py` | All training and evaluation CLI |
| `data/boxes_30um/` | 247 cached CAVE boxes |

## Package surface policy

To keep the library import surface manageable, `neuronauts.__init__` now re-exports
only the active no-EM training/evaluation pipeline APIs (CellGNN, path encoder,
datasets, and evaluation helpers).

Legacy experimental modules are still available via direct module imports
(e.g. `from neuronauts import vectorized` is **not** supported; use
`import neuronauts.legacy.vectorized` explicitly when needed).

## Research threads

The work is organized as a series of experiments (research threads) feeding this
core pipeline. See [`experiments/README.md`](experiments/README.md) for the index
— fingerprints (connectivity signatures), tree-DNA (morphology),
error-correction (proofreading supervision), PCFG,
grammar variants, cell-assignment, root-neighborhood, soma-graph, minnie-column,
and topology — each with its status, entry point, and checkpoints. The
longer-range direction is [`docs/roadmap_global_assembly.md`](docs/roadmap_global_assembly.md).

## Architecture notes

**What is not used in the current pipeline:**
- EM volume fetching / Sobel membrane fields
- Agent simulation (`vectorized.py`, `fields.py`)
- Skeleton graph source (`skeleton_graph.py`)
- Topology validator (`topology_model.py`)
- Neuroglancer inspector (`scripts/inspect_pipeline.py`)

These modules are present and tested but are not part of the active training workflow.

## Data

- **Box cache**: 247 × 30 µm CAVE boxes, v1412 (proofread) root IDs, ~1.68M synapses total
- **Train/val/test split**: 148 / 30 / 49 boxes (spatial split, reproducible with seed=42)
- **Edit pairs v3**: 25 860 pairs from 2002 v117 roots (25 444 false-merge, 416 false-split)
