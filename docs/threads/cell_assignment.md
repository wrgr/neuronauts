# Thread: cell_assignment

**Goal.** Assign synapses to cells with a **K-hop synapse-graph GNN** (`CellGNN`)
trained contrastively against CAVE root IDs, then partition the learned
embeddings into cells and score the result as synapse **line-graph F1**. This is
the pipeline that runs **by default** today.

**Status:** active (default). Best held-out test F1 ≈ **0.272** (`cell_gnn_seg`,
6 edge features incl. real seg signal, t=0.99). The ceiling is structural: the
graph is built per 30 µm box, so neurons larger than a box can't be assembled —
the motivation for the global-assembly roadmap.

## Code (lives in core)

| Module | Role |
|--------|------|
| [`neuronauts/cell_graph.py`](../../neuronauts/cell_graph.py) | `CellGNN`, `build_synapse_graph`, `partition_from_embeddings`, `train_cell_gnn` |
| [`neuronauts/assembly.py`](../../neuronauts/assembly.py) | candidate merges, connectivity assembly |
| [`neuronauts/line_graph.py`](../../neuronauts/line_graph.py) | `evaluate` → line-graph F1 (terminal metric) |

## Run

```bash
# Baseline
python scripts/train.py train-cell-gnn \
  --cache-dir data/boxes_30um --epochs 10 --n-layers 2 \
  --cell-gnn-output models/scratch/cell_gnn.pt --seed 42

# With a frozen path-encoder fingerprint (recommended)
python scripts/train.py train-cell-gnn \
  --cache-dir data/boxes_30um --epochs 10 --n-layers 2 \
  --path-encoder-checkpoint models/scratch/path_encoder.pt \
  --pretrained-path-emb-dim 16 \
  --cell-gnn-output models/scratch/cell_gnn.pt --seed 42

# Evaluate
python scripts/train.py evaluate \
  --cache-dir data/boxes_30um --cell-gnn-checkpoint models/cell_gnn_seg.pt --split test
```

## Checkpoints

| File | Metric |
|------|--------|
| `cell_gnn_seg.pt` | test F1 **0.272** @ t=0.99 (6-feat + real seg) — best |
| `cell_gnn_5feat.pt` | test F1 0.269 @ t=0.99 (5 scalar features) — benchmark baseline |
| `cell_gnn_real.pt` | test F1 0.264 @ t=0.99 (first real-CAVE no-EM baseline) |

Ablation findings (K-hop sweep, per-feature drop) are summarized in
[`docs/TODO.md`](../../docs/archive/2026-09/TODO.md) (archived) and [`models/README.md`](../../models/README.md);
the sweep checkpoints themselves were curated out.

## Graduation

Already core and default. Its successor is **global** assembly: replace the
box-local synapse graph with a fragment graph carrying
[tree-DNA](tree_dna.md) that stitches across box seams (roadmap
Phase 2).
