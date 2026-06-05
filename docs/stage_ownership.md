# Stage ownership & module map

Companion to [`roadmap_global_assembly.md`](roadmap_global_assembly.md) and
[`../CONTRIBUTING.md`](../CONTRIBUTING.md). This is the living map of which files
back which pipeline stage, plus the dependency facts that govern the legacy
quarantine.

## Stage → modules → owner

Fill the **Owner** column with real names as the team forms.

| Stage | Artifact (`schemas.py`) | Current modules | Owner |
|-------|-------------------------|-----------------|-------|
| `data/` | `Region` | `fetch.py`, `dataset_builder.py`, `fetch_skeletons.py`, `cave_root_mapping.py`, `cave_synapse_*` | _TBD_ |
| `represent/` | `Fragment` (tree-DNA) | `path_edge_encoder.py`, `grammar.py` (encoder), skeleton featurization in `cell_graph.py` | _TBD_ |
| `assemble/` | `NeuronHypothesis` | `cell_graph.py` (CellGNN + partition), `assembly.py`, `merge.py` (graph types), `em_corridor.py` | _TBD_ |
| `connectome/` | `ConnectomeGraph` | `experiments/soma_graph/`, `shared_grammar_model.py` (GAT) | _TBD_ |
| `evaluate/` | metrics | `line_graph.py` | _TBD_ |
| `legacy/` (v1) | — | `vectorized.py`, `fields.py`, `agent.py`, agent half of `run.py`, `topology_*` | _maintainer only_ |

## Contracts

The five artifacts are defined in [`../neuronauts/schemas.py`](../neuronauts/schemas.py)
with `validate()` + pickle-free `.npz` I/O. Treat any change to these as a
cross-team change (see `CONTRIBUTING.md`).

## Legacy quarantine plan

**Goal:** move the v1 agent/membrane simulation stack into `neuronauts/legacy/`,
out of the default import surface and CI.

**Why it has not happened yet (the honest state, verified 2026-06-05):** a naive
file move breaks imports because active modules still depend on v1 modules. The
dependency facts:

| v1 module | Imported by (active) | Blocks move? |
|-----------|----------------------|--------------|
| `membrane_unet.py` | — (already deleted; `run.py:1280` is a removal stub) | n/a — only stale doc references remain |
| `agent.py` (`Agent`, `AgentConfig`) | `merge.py:10` (active — `MergedNeuron`/`ConnectivityGraph` live here too) | **yes** |
| `fields.py` | `shared_grammar_model.py:688`, `topology_dataset.py:12`, `scripts/train.py` (GAT path) | **yes** |
| `vectorized.py` | `run.py:28` (console entry point) | **yes** (via `run.py`) |
| `run.py` | pyproject console script `neuronauts.run:main`; `scripts/train.py` (GAT path); ~9 test files; `scripts/export_*`, `inspect_pipeline.py` | **yes** |
| `topology_model.py` | `cell_graph.py:2475` (optional validator in `score_cell_quality`), `scripts/train.py:1520` | semi-active |

> **Do not** "fix" this by having active modules import from `legacy/`. The
> allowed direction is `legacy/ → active`, never `active → legacy/`.

**Sequence that must land before the move (each step is its own behavior-
preserving PR, verified with the full suite):**

1. **Split `merge.py`.** Keep `MergedNeuron` and `ConnectivityGraph` (used by
   `cell_graph.py`, `skeleton_graph.py`, `assembly.py`) in an active module.
   Move `merge_agents` + the `from .agent import Agent` dependency to the v1
   side. This frees `agent.py`.
2. **Extract the active helpers out of `run.py`.** The pieces that are genuinely
   active — `_scaffold_union_from_seg_ids`, `_build_graph`, `_merge_role_groups`,
   `HeuristicConfig` (imported by `train.py` and several tests) — move into
   `assemble/`. The agent loop, `REAL_BOXES`, synthetic-benchmark code, and the
   `run()` orchestrator stay behind as v1. Note the `--train-gat` path in
   `train.py` depends on `simulate_paths_and_hits` and is itself v1-coupled;
   decide whether GAT-from-simulation is retired or ported onto `Fragment`s.
3. **Decouple `fields.py` consumers.** `compute_membrane_field` is used by
   `topology_dataset.py` and (lazily) `shared_grammar_model.py`. Quarantine
   `fields.py` together with the topology/simulation cluster, or keep it if the
   topology validator is retained as active.
4. **Move + re-point.** With 1–3 done, move `agent.py`, `vectorized.py`,
   `fields.py`, the agent half of `run.py`, and the membrane/topology sim into
   `neuronauts/legacy/`. Update `pyproject.toml` `packages` to include
   `neuronauts.legacy`, repoint the console script (or retire it), and mark the
   v1 tests with `@pytest.mark.legacy` (register the marker) so they drop from
   default CI.

**Stale references to clean up opportunistically** (they describe the deleted
`membrane_unet.py` / dead v1 path): `program.md`, `docs/model.md`,
`docs/whitepaper.md`, `docs/global_inference_roadmap.md`,
`docs/pipeline_state.md`.
