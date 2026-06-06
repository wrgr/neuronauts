# Contributing to neuronauts

This project is moving to a **staged pipeline with typed artifacts between
stages** so that a team can own and develop slices in parallel. Read
[`docs/roadmap_global_assembly.md`](docs/roadmap_global_assembly.md) for the
why; this file is the how.

## The pipeline is five stages

Each stage reads and writes a **typed artifact on disk** (defined in
[`neuronauts/schemas.py`](neuronauts/schemas.py)). You depend on the *schema* of
the upstream artifact, never on the upstream stage's code — so you can build,
cache, and test your stage in isolation.

| Stage | Produces (schema) | Today lives in | Ownership map |
|-------|-------------------|----------------|---------------|
| `data/` | `Region` | `fetch.py`, `dataset_builder.py`, `fetch_skeletons.py` | see [`docs/stage_ownership.md`](docs/stage_ownership.md) |
| `represent/` | `Fragment` (tree-DNA) | `path_edge_encoder.py`, skeleton featurization in `cell_graph.py` | ″ |
| `assemble/` | `NeuronHypothesis` | `cell_graph.py`, `assembly.py`, `merge.py` | ″ |
| `connectome/` | `ConnectomeGraph` | `experiments/soma_graph/`, `shared_grammar_model.py` (GAT) | ″ |
| `evaluate/` | metrics | `line_graph.py` | ″ |

> The `data/`, `represent/`, … package directories are the **target** layout.
> The code still lives in the flat module list above; Phase 0 splits the two
> monoliths (`scripts/train.py`, `cell_graph.py`) into these packages. Until
> then, treat the stage as a logical boundary and keep new code on the correct
> side of it.

## Golden rule: respect the contracts

- A change that alters a `schemas.py` type is a **cross-team change** — flag it
  in the PR description and bump `SCHEMA_VERSION` if the on-disk format changes.
- Produce artifacts via the schema constructors (e.g.
  `Region.from_synapse_table`) and call `.validate()` before writing.
- Coordinates in artifacts are **global nanometers**, not box-relative voxels.
  Convert at the `data/` boundary; everything downstream assumes global nm.

## Dev setup

```bash
pip install -e ".[dev,topology]"   # numpy, scipy, pytest, pandas, matplotlib, torch
# Optional, only for live CAVE fetching (not needed to run tests on a cache):
pip install -e ".[cave]"           # caveclient  (may need network + a CAVE token)
```

## Testing

```bash
pytest -q                                   # full suite
pytest -q --continue-on-collection-errors   # if caveclient is not installed
pytest tests/test_schemas.py -q             # one stage's contract tests
```

Conventions:

- **Every stage ships at least one smoke test** that constructs its artifact
  from a tiny in-memory fixture and round-trips it through disk. See
  `tests/test_schemas.py` for the pattern (duck-typed fixtures, `tmp_path`).
- Tests must not require network or a CAVE token. Tests that hit CAVE import
  `caveclient` at module top and so only collect when the `cave` extra is
  installed; keep those isolated and clearly named (`test_cave_*`).
- Keep new tests dependency-honest: if a test needs torch, it's fine (torch is a
  dev dep); if it needs a GPU, skip when unavailable.

## Legacy (v1) code

The original agent/membrane simulation stack (700-walker tracing) is **not part
of the active pipeline** and now lives under `neuronauts/legacy/` (`run`,
`agent`, `agent_merge`, `vectorized`, `fields`). `import neuronauts` does not
load any of it. The quarantine history and the remaining items are in
[`docs/stage_ownership.md`](docs/stage_ownership.md#legacy-quarantine-plan).

Guidelines:

- Don't add dependencies from active code on `neuronauts.legacy.*`. The allowed
  direction is `legacy → active`, never `active → legacy`.
- Run `pytest -m 'not legacy'` to skip the v1 simulation tests.
- `em_corridor.py` and `topology_model.py` are **not** legacy — they're still
  referenced by the active `cell_graph.py`, so they stay at the top level.

## Pull requests

- Keep PRs scoped to one stage where possible.
- Refactors that move code must be **behavior-preserving**: run the suite before
  and after and note the pass count in the PR.
- Update the relevant doc (`docs/roadmap_global_assembly.md`,
  `docs/stage_ownership.md`) in the same PR when you change structure or
  ownership.
