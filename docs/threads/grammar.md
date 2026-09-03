# Thread: grammar

**Goal.** Score whether two neurite fragments should **merge**, using a learned
path "grammar" — a Transformer `PathEncoder` with a `[CLS]` token plus a
`MergeScorer`, trained on local merge supervision (same-root positives,
nearby-different-root negatives) and global atomicity. The shared-grammar
variant adds a `GlobalAssemblyGAT` head that refines connectivity edges against
a soft-F1 surrogate of the terminal line-graph metric.

**Status:** superseded in its headline claim; the code is still the reference
implementation.

> **Read this before quoting the accuracy below.** The "~85–87% pairwise merge
> accuracy" this thread was written around is an **in-sample number on a
> synthetic substrate** — fragments produced by artificially damaging intact
> skeletons, where the two halves of a split still carry matching geometry.
> Every later attempt to reproduce it against real v117 segmentation came back
> near zero: EXP-060, EXP-060B, EXP-061 and EXP-070 all measured candidate
> generation at roughly **0.09% precision**, and EXP-072 collapsed the same way
> once the substrate was widened. The number is not fraudulent, it answers a
> different and much easier question than the one the program needs.
>
> What survives, measured on real data: **EXP-063** detects a frankenmerge at
> held-out AUC **0.958**, and polarity alone reaches 0.914. What does not:
> pairwise merge proposal on real segmentation, which is the task this thread
> names. See `docs/threads/experiment_survey.md` for the evidence grade of every
> experiment in the repo, and `results/EXP-075/evaluation.md` for the current
> state of the join problem.

## Code (lives in core)

| Module | Role |
|--------|------|
| [`neuronauts/grammar.py`](../../neuronauts/grammar.py) | `PathEdgeEncoder`/`TorchPathEncoder`, `MergeScorer`, `ArborEncoder` |
| [`neuronauts/shared_grammar_model.py`](../../neuronauts/shared_grammar_model.py) | `SharedGrammarModel`, `BridgeHead`, `GlobalAssemblyGAT`, `multitask_train_step` |
| [`neuronauts/assembly.py`](../../neuronauts/assembly.py) | `gat_refine_connectivity`, `label_graph_edges` |
| [`attic/superseded_training/train_shared_grammar.py`](../../attic/superseded_training/train_shared_grammar.py) | standalone shared-grammar trainer (helper) |

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
| `grammar_cave_real_50.pt` | best real-data grammar, val merge acc 87.2% — **in-sample, synthetic damage; see the status note above** |
| `grammar_synthetic.pt` | synthetic smoke baseline (no CAVE token) |
| `shared_grammar_real.pt` | shared grammar + GAT, real boxes |
| `shared_grammar_raw_skel_50e.pt` / `_gat50e.pt` | `raw_delta3+skeleton` feature mode, without / with GAT |
| `gat_skeleton_50e.pt` | GlobalAssemblyGAT over skeleton features |

See [`models/README.md`](../../models/README.md).

## Graduation

Already core. The thread "graduates the metric" when shared-grammar+GAT improves
held-out **line-graph F1** (not just local merge acc) over the CellGNN baseline —
at which point it becomes the assembly head of record.
