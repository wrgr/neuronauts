# Status

## Phase 0 — COMPLETE
Merged `claude/intelligent-planck-oCPwt` on 2026-06-05.

Delivered:
- `neuronauts/schemas.py` — typed inter-stage contracts (Region, Fragment, NeuronHypothesis, ConnectomeGraph)
- `neuronauts/legacy/` — v1 agent/membrane stack quarantined; `pytest -m 'not legacy'` skips it
- `CONTRIBUTING.md` + `docs/stage_ownership.md` — contributor docs and ownership map
- `tests/test_schemas.py` — smoke tests for all contracts

## Phase 1 — IN PROGRESS
Branch: `claude/tree-dna-phase-1-G1DNn`

**Architecture:** synapse-level global partitioning.  Nodes = synapses; node
features = seg-root DNA (learned from whole kimimaro skeleton tree).  No box
boundary in the global synapse graph.

### Checklist
- [x] Radii saved in skeleton archive (`precompute_self_skeletons_for_cache`)
- [x] `neuronauts/data/fragments.py` — `skeleton_to_fragment`, `extract_fragments_for_region`
- [x] `neuronauts/represent/dna.py` — `TreeDNAEncoder`, `featurize_fragment`, `sample_tree_paths`, `encode_fragments`, `train_dna_encoder`
- [x] `tests/test_data_fragments.py` + `tests/test_represent_dna.py`
- [ ] Ablation: DNA-enriched synapse features AUC vs 6-scalar CellGNN baseline

## Phase 2 — NEXT
Global synapse graph (no box boundary) with DNA node features → CellGNN-style
GNN → neuron hypotheses that span regions.  The box-local CellGNN assembler is
retired; `assemble/fragment_graph.py` replaces it.

## See also
- `docs/roadmap_global_assembly.md` — canonical north-star roadmap
- `docs/stage_ownership.md` — stage→module ownership map
