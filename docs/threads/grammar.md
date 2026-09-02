# Thread: grammar

**Goal.** Score whether two neurite fragments should **merge**, using a learned
path "grammar" — a Transformer `PathEncoder` with a `[CLS]` token plus a
`MergeScorer`, trained on local merge supervision (same-root positives,
nearby-different-root negatives) and global atomicity. The shared-grammar
variant adds a `GlobalAssemblyGAT` head that refines connectivity edges against
a soft-F1 surrogate of the terminal line-graph metric.

**Status:** active (core thread). Pairwise merge accuracy is strong (~85–87%);
the open problem is translating that into global line-graph F1 (see
[cell_assignment](../cell_assignment/README.md) and the roadmap).

## Code (lives in core)

| Module | Role |
|--------|------|
| [`neuronauts/grammar.py`](../../neuronauts/grammar.py) | `PathEdgeEncoder`/`TorchPathEncoder`, `MergeScorer`, `ArborEncoder` |
| [`neuronauts/shared_grammar_model.py`](../../neuronauts/shared_grammar_model.py) | `SharedGrammarModel`, `BridgeHead`, `GlobalAssemblyGAT`, `multitask_train_step` |
| [`neuronauts/assembly.py`](../../neuronauts/assembly.py) | `gat_refine_connectivity`, `label_graph_edges` |
| [`scripts/train_shared_grammar.py`](../../scripts/train_shared_grammar.py) | standalone shared-grammar trainer (helper) |

## Run

```bash
# Grammar merge scorer
python scripts/train.py train \
  --cache-dir data/boxes_30um --epochs 10 \
  --grammar-output models/scratch/grammar.pt

# Shared grammar + GAT, preferred feature mode
python scripts/train.py train \
  --cache-dir data/boxes_v117 --base-version 117 --target-version 1412 \
  --path-feature-mode raw_delta3+skeleton \
  --grammar-output models/scratch/shared_grammar.pt --train-gat --epochs 50
```

## Checkpoints

| File | What |
|------|------|
| `grammar_cave_real_50.pt` | best real-data grammar, val merge acc 87.2% |
| `grammar_synthetic.pt` | synthetic smoke baseline (no CAVE token) |
| `shared_grammar_real.pt` | shared grammar + GAT, real boxes |
| `shared_grammar_raw_skel_50e.pt` / `_gat50e.pt` | `raw_delta3+skeleton` feature mode, without / with GAT |
| `gat_skeleton_50e.pt` | GlobalAssemblyGAT over skeleton features |

See [`models/README.md`](../../models/README.md).

## Graduation

Already core. The thread "graduates the metric" when shared-grammar+GAT improves
held-out **line-graph F1** (not just local merge acc) over the CellGNN baseline —
at which point it becomes the assembly head of record.
