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

**Only five threads still have code under `experiments/`:** `pcfg/`,
`fingerprints/`, `minnie_column/`, `soma_graph/` and `root_neighborhood/`. The
rest of this page's rows are *core* threads whose code lives in `neuronauts/`, or
history. Retired threads and one-off scripts are in
[`attic/`](../attic/README.md), which is the archive; each subdirectory there has
a README saying what it was and what replaced it.

Two threads moved out on 2026-09-02:

| Was | Now | Why |
|---|---|---|
| `experiments/low_res_segmentation/` | [`attic/incubating_threads/low_res_segmentation/`](../attic/incubating_threads/README.md) | Incubating since April with no number against its own graduation bar; branch stale since 2026-04-07. Its two tests stayed in `tests/` and still run. |
| 15 one-off scripts in `experiments/pcfg/` | [`attic/pcfg_one_offs/`](../attic/pcfg_one_offs/README.md) | Each answered one question once; nothing in the remaining package imports any of them. They still run from the new path. |

| Thread | Status | Kind | Code | Entry point | Checkpoints |
|--------|--------|------|------|-------------|-------------|
| [fingerprints](fingerprints/README.md) | external | branch | `claude/neuron-fingerprints-connectivity-jg95xp` (unmerged) | see thread README | — |
| [tree_dna](../docs/threads/tree_dna.md) | incubating | core seed + branch | `neuronauts/path_edge_encoder.py` + `claude/tree-dna-phase-1-G1DNn` | `train.py train-path-encoder` | `path_encoder_v3*` (local) |
| [error_correction](../docs/threads/error_correction.md) | active | core | `neuronauts/edit_history.py`, `cave_root_mapping.py`, `path_dataset.py` | `train.py fetch-cave-edits-from-cache` | — (training signal) |
| [pcfg](pcfg/README.md) | active | experiment | `experiments/pcfg/` | `run_experiment.py` | — (non-neural) |
| [grammar](../docs/threads/grammar.md) | active | core | `neuronauts/grammar.py`, `shared_grammar_model.py` | `train.py train` | `grammar_cave_real_50`, `shared_grammar_*`, `gat_skeleton_50e` |
| [cell_assignment](../docs/threads/cell_assignment.md) | active (default) | core | `neuronauts/cell_graph.py`, `assembly.py` | `train.py train-cell-gnn` | `cell_gnn_seg`, `cell_gnn_5feat`, `cell_gnn_real` |
| [root_neighborhood](root_neighborhood/README.md) | incubating | experiment | `experiments/root_neighborhood/` | `train.py build-dataset --strategy proofread-core` | `shared_grammar_root_neighborhood_run001` |
| [soma_graph](soma_graph/README.md) | incubating | experiment | `experiments/soma_graph/` | `smoke_test.py` | — |
| [minnie_column](minnie_column/README.md) | active (data) | experiment | `experiments/minnie_column/` | see its README | — |
| [topology](../docs/threads/topology.md) | optional | core | `neuronauts/topology_model.py`, `topology_dataset.py` | `attic/superseded_training/train_topology_model.py` | — (smoke only) |
| legacy (v1) | quarantined | — | `neuronauts/legacy/` | `neuronauts` console script | — |

**Status legend:** *active* = part of the current workflow · *active (default)*
= the pipeline that runs by default · *incubating* = promising, not yet a
baseline · *optional* = wired but off the default path · *external* = code lives
on an unmerged feature branch, not in this tree · *quarantined* = kept for
history, excluded from the default import surface and CI (`pytest -m 'not legacy'`).

## Research threads ↔ branches

Much of the active work lives on feature branches that aren't merged here yet.
This is the map (status as of 2026-06-29, `main` @ `3784cff`). Ahead/behind are
relative to `main`.

### Threads with live work on a branch

| Thread | Branch | Ahead/behind | Last commit | State |
|--------|--------|--------------|-------------|-------|
| fingerprints | `claude/neuron-fingerprints-connectivity-jg95xp` | 44 / 0 | 2026-06-29 | **active**, strictly ahead — PR-ready |
| error_correction | `claude/error-correction-model-jb0x1i` | 46 / 0 | 2026-06-29 | **active**, strictly ahead — PR-ready (SSL splice pretrain) |
| grammar (cross-region eval) | `claude/vibrant-wozniak-we54pv` | 30 / 0 | 2026-06-28 | **active**, strictly ahead — PR-ready (`HOLDOUT_RESULTS.md`) |
| pcfg (fetch follow-on) | `claude/pcfg-synapse-partitions-lo5nlu` | 4 / 0 | 2026-06-27 | **active**, strictly ahead — lightweight fetch + retry |
| cell_assignment / co-assignment | `claude/synapse-coassign` | 16 / 24 | 2026-06-10 | **active**, diverged — needs rebase (correlation-clustering frontier) |
| global stitch / lineage | `claude/abstract-tree-stitch` | 31 / 24 | 2026-06-12 | **active**, diverged — needs rebase (`treestitch/`, real v117→v1718) |
| tree_dna | `claude/tree-dna-phase-1-G1DNn` | 211 / 24 | 2026-06-25 | **PR #17 open** — author notes it's superseded by `synapse-coassign` → likely close/extract |
| low_res_segmentation | `claude/low-res-segmentation-pipeline-fwIHN` | 6 / 95 | 2026-04-07 | **PR #9 open**, stale → review or close |

### Needs a decision (unmerged, no current PR)

| Branch | Ahead/behind | What it is | Suggested |
|--------|--------------|------------|-----------|
| `claude/remove-connectome-clutter-CkKag` | 1 / 31 | `SynapseTable.filter_clutter` + `--min-root-synapses` (real feature, **not** in main) | rebase + PR, or discard |
| `claude/small-e2e-test-B9k2g` | 11 / 31 | v117-atom sibling retrieval (fingerprint-adjacent) | fold insight into fingerprints, then close |

### Safe to delete — fully merged into `main` (0 commits ahead)

| Branch | PR |
|--------|----|
| `claude/pcfg-synapse-partitions-hcd4c8` | #18 (merged) |
| `claude/todo-cave-coverage-E89bU` | #5 (merged) |
| `codex/scrub-package-to-deprecate-and-organize` | #15 (merged) |
| `claude/intelligent-planck-oCPwt` | #16 (merged); only extra commit is a stale `STATUS.md` |
| `claude/resume-pipeline-6g6wm` | #11 (merged); only extra commit is a doc tweak superseded by the reconciled README |

> Deletion is a manual step (not done automatically). Once a branch's work is
> merged or abandoned, delete it with `git push origin --delete <branch>` to keep
> this list short.

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
