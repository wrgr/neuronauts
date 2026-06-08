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
- [x] `neuronauts/represent/enrich.py` — `build_synapse_dna_matrix`, `synapse_pair_dna_scores`, `spatial_proximity_scores`, `evaluate_dna_auc`
- [x] `tests/test_represent_enrich.py` — 12 tests including AUC ≥ 0.9 with orthogonal DNA
- [x] `scripts/ablate_dna.py` — end-to-end ablation script (`--synthetic` + `--archive` modes)
- [x] `scripts/fetch_real_skeletons.py` — fetch real proofread v1412 skeletons from CAVE skeleton cache, build Region with uniform-random synapses, run ablation
- [x] **Real-data ablation (30 real minnie65 neurons, v1412 proofread)**:
  - Spatial baseline (uniform synapses): **0.493** (chance — as expected)
  - DNA AUC random init: **1.000** — real skeleton morphology is sufficient for perfect neuron discrimination even with random weights
  - DNA AUC trained (60 epochs): **1.000** (ceiling)
  - Training signal: pos_cos 0.88→0.74, neg_cos 0.79→0.37 — genuine discriminative learning even at AUC ceiling
  - **Interpretation**: 30 diverse real neurons are morphologically so distinct that DNA trivially separates them. Harder evaluation (same-cell-type cohorts, unproofread multi-root data) is needed to see training benefit beyond ceiling.
- [ ] Hard-split ablation: split each neuron skeleton into two halves; test if trained encoder assigns higher cosine similarity to same-neuron halves than cross-neuron pairs (relevant for Phase 2's multi-root matching problem)

## Phase 2 — NEXT
Global synapse graph (no box boundary) with DNA node features → CellGNN-style
GNN → neuron hypotheses that span regions.  The box-local CellGNN assembler is
retired; `assemble/fragment_graph.py` replaces it.

## See also
- `docs/roadmap_global_assembly.md` — canonical north-star roadmap
- `docs/stage_ownership.md` — stage→module ownership map
