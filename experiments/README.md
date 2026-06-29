# Research threads

`neuronauts` is organized as a **series of experiments** (research threads) that
feed a shared core pipeline. This page is the index: what each thread is, its
status, where its code lives, how to run it, and which checkpoints it owns.

Two kinds of threads:

- **Core threads** — the code is part of the active pipeline and lives in
  `neuronauts/`. The thread page links into those modules; nothing is moved.
- **Experiment threads** — exploratory code lives self-contained under
  `experiments/<thread>/`.

> The canonical pipeline narrative is in the top-level [`README.md`](../README.md);
> the longer-range direction is [`docs/roadmap_global_assembly.md`](../docs/roadmap_global_assembly.md).
> Checkpoints are catalogued in [`models/README.md`](../models/README.md).

| Thread | Status | Kind | Code | Entry point | Checkpoints |
|--------|--------|------|------|-------------|-------------|
| [fingerprints](fingerprints/README.md) (tree-DNA) | incubating | core | `neuronauts/path_edge_encoder.py`, `path_dataset.py` | `train.py train-path-encoder` | `path_encoder_v3*` (local) |
| [error_correction](error_correction/README.md) | active | core | `neuronauts/edit_history.py`, `cave_root_mapping.py`, `path_dataset.py` | `train.py fetch-cave-edits-from-cache` | — (training signal) |
| [pcfg](pcfg/README.md) | active | experiment | `experiments/pcfg/` | `run_experiment.py` | — (non-neural) |
| [grammar](grammar/README.md) | active | core | `neuronauts/grammar.py`, `shared_grammar_model.py` | `train.py train` | `grammar_cave_real_50`, `shared_grammar_*`, `gat_skeleton_50e` |
| [cell_assignment](cell_assignment/README.md) | active (default) | core | `neuronauts/cell_graph.py`, `assembly.py` | `train.py train-cell-gnn` | `cell_gnn_seg`, `cell_gnn_5feat`, `cell_gnn_real` |
| [root_neighborhood](root_neighborhood/README.md) | incubating | experiment | `experiments/root_neighborhood/` | `train.py build-dataset --strategy proofread-core` | `shared_grammar_root_neighborhood_run001` |
| [soma_graph](soma_graph/README.md) | incubating | experiment | `experiments/soma_graph/` | `smoke_test.py` | — |
| [minnie_column](minnie_column/README.md) | active (data) | experiment | `experiments/minnie_column/` | see its README | — |
| [topology](topology/README.md) | optional | core | `neuronauts/topology_model.py`, `topology_dataset.py` | `scripts/train_topology_model.py` | — (smoke only) |
| legacy (v1) | quarantined | — | `neuronauts/legacy/` | `neuronauts` console script | — |

**Status legend:** *active* = part of the current workflow · *active (default)*
= the pipeline that runs by default · *incubating* = promising, not yet a
baseline · *optional* = wired but off the default path · *quarantined* = kept for
history, excluded from the default import surface and CI (`pytest -m 'not legacy'`).

## How threads graduate

Each thread page ends with a **graduation note**. The lifecycle:

```
incubating ──(beats the current baseline on a held-out split)──▶ promote to core
     │
     └────────(superseded / dead end)──────────────────────────▶ archive
```

Promoting a thread means folding its code into `neuronauts/` (a core thread) and
updating this table; archiving means moving it under `experiments/archive/` with
a one-line epitaph. Keep this index and `models/README.md` in sync when either
happens.
