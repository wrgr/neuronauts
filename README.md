# neuronauts

`neuronauts` reconstructs neurons from electron microscopy connectome data. Given a CAVE synapse table, it learns to assign synapses to cells and evaluates the result as synapse line-graph F1.

The active pipeline requires **no EM volume** and **no agent simulation** — it runs entirely on synapse positions and CAVE root IDs.

## Pipeline

```
CAVE synapse table (MICrONS minnie65_public)
        │
        ▼
[1] Box cache  ──── data/boxes_30um/
    30 µm windows, v1412 root IDs, ~6800 synapses/box
        │
        ▼
[2] CAVE edit pairs  ──── data/cave_edit_pairs_v3.tsv
    False merges + false splits from v117→v1412 transfer
    (25 444 merge pairs, 416 split pairs)
        │
        ▼
[3] Path encoder  ──── models/path_encoder_v3_ep8.pt
    Transformer over synapse-chain windows
    Discriminates valid continuations from spliced paths
    Best: acc=0.899 (ep8/10)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
[4a] Grammar  ──── models/grammar_30um_v1.pt      [4b] CellGNN  ──── models/cell_gnn_v3.pt
     PathEdgeEncoder + MergeScorer                     K-hop synapse GNN
     val merge acc: 84.6% (ep6/10, still training)     Contrastive cell membership loss
                                                        + pretrained path encoder (frozen)
        │                                              │
        └──────────────────┬───────────────────────────┘
                           ▼
                   [5] Evaluate
                   Line-graph F1 on held-out test boxes
```

## Current results  *(updated 2026-05-01)*

| Model | Val merge acc | Test line-graph F1 |
|-------|--------------|-------------------|
| Grammar (ep6/10) | 84.6% | TBD — training in progress |
| CellGNN baseline (ep2/10) | — | TBD |
| CellGNN v3 + path encoder | — | TBD — ep1 in progress |

Previous best (older 40-box run): grammar 87.23% val merge acc.

## Prerequisites

```bash
pip install -e ".[dev]"

# CAVE token (for fetching data; not needed to train on existing cache)
mkdir -p ~/.cloudvolume/secrets
echo '{"token": "YOUR_TOKEN"}' > ~/.cloudvolume/secrets/cave-secret.json
```

## Step 0 — Check the box cache

247 boxes are already cached at `data/boxes_30um/`.

```bash
ls data/boxes_30um/ | wc -l   # should be ~494 (json + npz per box)
```

## Step 1 — Fetch CAVE edit pairs

Samples delta roots (v117 → v1412), yields false-merge and false-split chains.

```bash
python scripts/train.py fetch-cave-edits \
  --n-sample 50000 \
  --max-false-merges 99999 \
  --min-synapses-per-root 8 \
  --output-tsv data/cave_edit_pairs_v3.tsv \
  --output-chains data/cave_edit_chains_v3.npz
```

Expected: ~25 000 merge pairs + ~400 split pairs, ~10 min.

## Step 2 — Train path encoder

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

Expected: best acc ~0.90 (around ep8), ~10 min/epoch.
Best checkpoint saved as `models/path_encoder_v3_best.pt` (future runs).

## Step 3 — Train grammar

```bash
python scripts/train.py train \
  --cache-dir data/boxes_30um \
  --epochs 10 \
  --grammar-output models/grammar_30um_v1.pt
```

Resumes automatically from existing checkpoint if present.
Expected: val merge acc ~87% by ep10, ~30–45 min/epoch.

## Step 4 — Train CellGNN

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

Expected: val_loss converging over 10 epochs, ~75–100 min/epoch on CPU.

## Step 5 — Evaluate

```bash
# Grammar only
python scripts/train.py evaluate \
  --cache-dir data/boxes_30um \
  --cell-gnn-checkpoint models/cell_gnn_30um_v1.pt \
  --grammar-checkpoint models/grammar_30um_v1.pt \
  --split test

# CellGNN + path encoder only
python scripts/train.py evaluate \
  --cache-dir data/boxes_30um \
  --cell-gnn-checkpoint models/cell_gnn_v3.pt \
  --split test
```

## Run tests

```bash
pytest                          # all tests (~10 min)
pytest tests/test_cell_graph.py # CellGNN unit tests only
```

## Saved checkpoints

| File | Description |
|------|-------------|
| `models/path_encoder_v3_ep8.pt` | Best path encoder (acc=0.899) |
| `models/path_encoder_v3.pt` | Final path encoder (acc=0.896) |
| `models/grammar_30um_v1.pt` | Grammar, best val (ep6+, 84.6% merge acc) |
| `models/cell_gnn_30um_v1.pt` | CellGNN baseline (ep2) |
| `models/cell_gnn_v3.pt` | CellGNN + path encoder (training) |

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

## Architecture notes

**What is not used in the current pipeline:**
- EM volume fetching / Sobel membrane fields
- Agent simulation (`vectorized.py`, `fields.py`)
- Skeleton graph source (`skeleton_graph.py`)
- Topology validator (`topology_model.py`)
- Neuroglancer inspector (`scripts/inspect_pipeline.py`)

These modules are present and tested but are not part of the active training workflow.

**Why no EM volumes?** The synapse positions + root IDs from CAVE are sufficient to train the grammar and CellGNN. Adding EM requires BossDB access and kimimaro skeletonization (~10s/box), which is reserved for future skeleton-path feature work.

## Data

- **Box cache**: 247 × 30 µm CAVE boxes, v1412 (proofread) root IDs, ~1.68M synapses total
- **Train/val/test split**: 148 / 30 / 49 boxes (spatial split, reproducible with seed=42)
- **Edit pairs v3**: 25 860 pairs from 2002 v117 roots (25 444 false-merge, 416 false-split)
