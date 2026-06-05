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
| `legacy/` (v1) | — | `legacy/run.py` (moved; `neuronauts/run.py` is a shim), `vectorized.py`, `fields.py`, `agent.py`, `agent_merge.py`, `topology_*` | _maintainer only_ |

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
| `agent.py` (`Agent`, `AgentConfig`) | none (step 1 done) — now only `agent_merge.py`, `vectorized.py`, `run.py`, all v1 | **no longer** — freed from the active surface |
| `fields.py` | `shared_grammar_model.py:688`, `topology_dataset.py:12`, `scripts/train.py` (GAT path) | **yes** |
| `vectorized.py` | `legacy/run.py`, tests (`test_vectorized_extras`) — all v1 | not by the active surface |
| `run.py` | ✅ **relocated to `legacy/run.py`** (step 2 done). `neuronauts/run.py` is now a deprecated delegating shim; importers (console script, `train.py` GAT path, `shared_grammar_model`, ~9 v1 tests) resolve through it unchanged. | no — off the active surface |
| `topology_model.py` | `cell_graph.py:2475` (optional validator in `score_cell_quality`), `scripts/train.py:1520` | semi-active |

> **Do not** "fix" this by having active modules import from `legacy/`. The
> allowed direction is `legacy/ → active`, never `active → legacy/`.

**Sequence that must land before the move (each step is its own behavior-
preserving PR, verified with the full suite):**

1. **Split `merge.py`. ✅ DONE (2026-06-05).** `MergedNeuron` /
   `ConnectivityGraph` stay in `merge.py` (now agent-free); the v1 `merge_agents`
   + its `from .agent import Agent` dependency moved to `neuronauts/agent_merge.py`.
   Verified: `import neuronauts` no longer transitively imports `agent.py`, and
   242 passed / 1 pre-existing failure across the merge/agent/run + active-core
   tests (the failure is the stale `test_core_types_importable`, unrelated).
   `agent.py` is now imported only by v1 modules.
2. **Relocate `run.py` to `legacy/`. ✅ DONE (2026-06-05).** Investigation
   corrected the premise of this step: the active import surface is exactly 8
   modules — `assembly`, `cell_graph`, `grammar`, `helpers`, `line_graph`,
   `merge`, `path_dataset`, `path_edge_encoder` (what `import neuronauts` loads)
   — and **none of them import `run.py`**. The helpers this step once named
   "active" (`_scaffold_union_from_seg_ids`, `_build_graph`, `_merge_role_groups`,
   `HeuristicConfig`) are consumed only by the idle GAT/simulation path
   (`shared_grammar_model` + `train.py::_run_gat_training_step`, gated by
   `--train-gat`) and by v1 tests. So `run.py` is **wholly v1**; there is
   nothing to extract into `assemble/`. It was therefore *relocated*, not
   gutted: the 1,884-line module now lives at `neuronauts/legacy/run.py`
   (relative imports rewritten to the parent package), and `neuronauts/run.py`
   is a thin deprecated shim that delegates every attribute to
   `legacy.run` via module `__getattr__`. All existing importers
   (`from neuronauts.run import ...`, the console script, `shared_grammar_model`,
   ~9 v1 tests) keep working unchanged. Verified: active surface still 8 modules
   (neither `run` nor `legacy.run` eagerly loaded); 211 passed / 1 pre-existing
   failure across every `run.py`-dependent test + active core.
   *Remaining for a later PR:* migrate those importers off the shim to
   `neuronauts.legacy.run`, then delete the shim.
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
