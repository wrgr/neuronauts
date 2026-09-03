# Thread: topology

**Goal.** Validate whether a candidate cell is **topologically atomic** — an
attention-based arbor validator that flags clusters formed by merging two
distinct roots. Used as an optional quality signal on top of
[cell_assignment](cell_assignment.md).

**Status:** optional (core thread). Wired but off the default path:
`neuronauts/cell_graph.py` imports `topology_model` only inside
`score_cell_quality`. Because an active module references it, it stays at the top
level rather than under `legacy/` (see
[`docs/stage_ownership.md`](../../docs/stage_ownership.md)).

## Code (lives in core)

| Module | Role |
|--------|------|
| [`neuronauts/topology_model.py`](../../neuronauts/topology_model.py) | `AttentionArborValidator`, `TrainingConfig`, `train_iteration` |
| [`neuronauts/topology_dataset.py`](../../neuronauts/topology_dataset.py) | atomicity dataset construction |
| [`attic/superseded_training/train_topology_model.py`](../../attic/superseded_training/train_topology_model.py) | standalone trainer (helper) |
| [`attic/superseded_training/export_topology_dataset.py`](../../attic/superseded_training/export_topology_dataset.py) | export an atomicity dataset |

## Run

```bash
python attic/superseded_training/export_topology_dataset.py --output data/topology_dataset.npz
python attic/superseded_training/train_topology_model.py \
  --dataset data/topology_dataset.npz --output models/scratch/topology.pt
```

## Checkpoints

None tracked — only smoke artifacts existed and were curated out. Write runs to
`models/scratch/`.

## Graduation

Decide its fate per the open item in
[`docs/stage_ownership.md`](../../docs/stage_ownership.md): either make the
validator fully optional (decouple `cell_graph` and quarantine it) or fold a
proven version into `cell_assignment`'s scoring.
