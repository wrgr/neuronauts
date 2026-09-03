# Contributing to neuronauts

This project uses a **staged pipeline with typed artifacts between stages** so
that a team can own and develop slices in parallel. Read
[`docs/roadmap_global_assembly.md`](docs/roadmap_global_assembly.md) for the
canonical north-star roadmap and [`docs/architecture.md`](docs/architecture.md)
for a detailed explanation of the system design and the problem it solves.

## The pipeline is five stages

Each stage reads and writes a **typed artifact on disk** (defined in
[`neuronauts/schemas.py`](neuronauts/schemas.py)). You depend on the *schema* of
the upstream artifact, never on the upstream stage's code — so you can build,
cache, and test your stage in isolation.

| Stage | Produces (schema) | Current modules | Ownership map |
|-------|-------------------|-----------------|---------------|
| `data/` | `Region` | `fetch.py`, `dataset_builder.py`, `fetch_skeletons.py`, `data/fragments.py` | see [`docs/stage_ownership.md`](docs/stage_ownership.md) |
| `represent/` | `Fragment` (tree-DNA) | `represent/dna.py`, `represent/enrich.py`, `grammar.py` | ″ |
| `assemble/` | `NeuronHypothesis` | `assemble/global_synapse_graph.py`, `assemble/synapse_gnn.py`, `cell_graph.py` (CellGNN backbone + legacy box-local) | ″ |
| `connectome/` | `ConnectomeGraph` | `experiments/soma_graph/`, `shared_grammar_model.py` (GAT) | ″ |
| `evaluate/` | metrics | `line_graph.py` | ″ |

The `data/`, `represent/`, and `assemble/` package directories are all live.
`assemble/global_synapse_graph.py` and `assemble/synapse_gnn.py` implement the
Phase 2 global pipeline. `cell_graph.py` remains the source for `CellGNN` and
`partition_from_embeddings`, which `assemble/` imports directly. Keep new
assembly code in `assemble/` rather than adding to `cell_graph.py`.

## Architecture overview

See [`docs/architecture.md`](docs/architecture.md) for a detailed walkthrough of:
- Why the box-local CellGNN has a hard F1 ceiling (~0.27) and what causes it
- How tree-DNA (learned skeleton morphology embeddings) breaks that ceiling
- The three-stage pipeline: data → represent → assemble
- Each stage's typed artifact contract (input/output schema)
- The DNA encoder design (path sampling, Transformer, triplet loss)
- The global synapse graph design (k-NN, DNA node features, message passing)
- Ablation results validating the approach (Phase 1 and Phase 2)
- Phase 2 open questions and what comes next

## Golden rule: respect the contracts

- A change that alters a `schemas.py` type is a **cross-team change** — flag it
  in the PR description and bump `SCHEMA_VERSION` if the on-disk format changes.
- Produce artifacts via the schema constructors (e.g.
  `Region.from_synapse_table`) and call `.validate()` before writing.
- Coordinates in artifacts are **global nanometers**, not box-relative voxels.
  Convert at the `data/` boundary; everything downstream assumes global nm.

## Dev setup

One command installs the package (editable) with every extra the full test
suite needs — `dev` (pytest/pandas/matplotlib), `topology` (torch), and `cave`
(caveclient):

```bash
# Preferred: uv handles the virtual environment automatically
uv sync

# Alternative: plain pip editable install
pip install -e ".[dev,topology]"   # numpy, scipy, pytest, pandas, matplotlib, torch

# Optional, only for live CAVE fetching:
pip install -e ".[cave]"           # caveclient (needs network + a CAVE token)
```

After `uv sync`, activate the environment with `source .venv/bin/activate` or
prefix commands with `uv run`.

## Testing

```bash
# Recommended: skip the quarantined v1 simulation tests
pytest -m "not legacy" -q

# Full suite (includes legacy tests — slow and noisy)
pytest -q

# If caveclient is not installed
pytest -q --continue-on-collection-errors

# One stage at a time
pytest tests/test_schemas.py -q
pytest tests/test_data_fragments.py -q
pytest tests/test_represent_dna.py -q
pytest tests/test_represent_enrich.py -q
pytest tests/test_assemble_global.py -q
```

The `pytest -m "not legacy"` pattern is the default for day-to-day development.
The `legacy` marker is registered in `pyproject.toml` and applied to v1
simulation tests via `tests/conftest.py`.

Conventions:

- **Every stage ships at least one smoke test** that constructs its artifact
  from a tiny in-memory fixture and round-trips it through disk. See
  `tests/test_schemas.py` for the pattern (duck-typed fixtures, `tmp_path`).
- Tests must not require network or a CAVE token. Tests that hit CAVE import
  `caveclient` at module top and so only collect when the `cave` extra is
  installed; keep those isolated and clearly named (`test_cave_*`).
- Keep new tests dependency-honest: if a test needs torch, it's fine (torch is a
  dev dep); if it needs a GPU, skip when unavailable.

## Running the ablations

**Phase 1 — DNA encoder (no data required):**

```bash
# Synthetic ablation: generates a multi-neuron world on the fly
python attic/prior_results/ablate_dna.py --synthetic

# Hard-split ablation (validates the multi-root use case; requires CAVE network access)
python attic/prior_results/half_split_ablation.py --n-neurons 40
```

**Phase 2 — Global GNN (implemented):**

```bash
# Synthetic global GNN ablation (no network required)
python attic/prior_results/global_gnn_ablation.py --synthetic

# Real-data global GNN ablation (requires CAVE auth token)
python attic/prior_results/global_gnn_ablation.py --n-neurons 40
```

Expected Phase 1 results on the hard-split ablation (40 real minnie65 neurons):
- Spatial baseline AUC: ~0.466 (chance)
- DNA AUC random init: ~0.728
- DNA AUC trained (80 epochs): ~0.897 (+0.169)

Expected Phase 2 results on the synthetic ablation:
- DNA AUC random init: ~0.787
- DNA AUC trained: ~0.863
- GNN AUC (DNA → GNN): ~0.914 (+0.051 over DNA alone)

## Stage ownership

The full module-to-stage map and owner assignments are in
[`docs/stage_ownership.md`](docs/stage_ownership.md). Fill in the **Owner**
column as the team forms.

## Branch naming

Use descriptive branch names tied to the phase and feature, e.g.:
- `phase1/dna-encoder-improvements`
- `phase2/global-synapse-graph`
- `fix/schema-validation-edge-case`

## Legacy (v1) code

The original agent/membrane simulation stack (700-walker tracing) is **not part
of the active pipeline** and lives under `neuronauts/legacy/` (`run`, `agent`,
`agent_merge`, `vectorized`, `fields`). `import neuronauts` does not load any of
it. The quarantine history and remaining items are in
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
- Schema changes (`schemas.py`) require a cross-team flag in the PR description
  and a `SCHEMA_VERSION` bump if the on-disk format changes.
