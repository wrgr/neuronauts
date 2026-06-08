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
- [x] **Hard-split ablation (40 real minnie65 neurons, each skeleton bisected at balance edge)**:
  - Spatial baseline (uniform synapses): **0.466** (chance)
  - DNA AUC random init: **0.728** — halves of the same neuron are only partially similar without training
  - DNA AUC trained (80 epochs): **0.897** (+0.169) ✓ DNA beats proximity by **+0.431**
  - Training signal: pos_cos 0.95→0.87, neg_cos 0.95→0.64 — encoder learns to align same-neuron halves
  - **Interpretation**: the encoder learns genuine morphological identity from partial skeleton views, directly validating the Phase 2 multi-root matching use case

## Phase 2 — IN PROGRESS
Branch: `claude/tree-dna-phase-1-G1DNn`

**Architecture:** global synapse graph (k-NN, no box boundary) with DNA node
features → CellGNN message-passing → per-synapse embeddings → cluster
assignments.

### Checklist
- [x] `neuronauts/assemble/global_synapse_graph.py` — `GlobalSynapseGraph`, `build_global_synapse_graph`
- [x] `neuronauts/assemble/synapse_gnn.py` — `train_global_gnn`, `run_global_gnn`, `assemble_neurons`
- [x] `tests/test_assemble_global.py` — 10 tests (graph shape, edge invariants, GNN training/inference)
- [x] `scripts/global_gnn_ablation.py` — end-to-end script (`--synthetic` + real-data mode)
- [ ] Real-data GNN ablation on hard-split skeletons (DNA → GNN AUC improvement)

## See also
- `docs/roadmap_global_assembly.md` — canonical north-star roadmap
- `docs/stage_ownership.md` — stage→module ownership map
