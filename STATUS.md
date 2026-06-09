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
- [x] `neuronauts/represent/skeleton_gnn.py` — `SkeletonGNN`, `encode_fragments_gnn`, `train_skeleton_gnn` (data-driven: raw (x,y,z,r) node features, no hand-crafted geometry)
- [x] `tests/test_represent_skeleton_gnn.py` — 20 tests (tensor shapes, bidirectionality, centroid normalisation, isolated vertex, training loop, L2 normalisation)
- [x] `--encoder [path|gnn]` flag added to `ablate_dna.py`, `half_split_ablation.py`, `within_type_ablation.py`

## Within-type evaluation — IN PROGRESS

The honest precision test: all negative pairs are same-cell-type neurons.
Scripts: `scripts/within_type_ablation.py`, `scripts/half_split_ablation.py` (`--n-chunks N`)
Cell type table: `aibs_metamodel_celltypes_v661_merged.csv.gz` (19,735 L2/3 pyramidals at v1412)

### Results so far

**4-chunk within-type (30 × 23P, quarter-skeletons):**
  - Random init AUC: **0.599** (harder start than cross-type — same-type neurons look alike)
  - Trained AUC: **0.687** (+0.089) — below 0.75 threshold; below individual-identity bar
  - Spatial baseline: 0.488 (chance)
  - neg_cos barely separates (0.908 → 0.822 at ep 80, noisy) — limited signal in quarter-skeletons
  - **Interpretation**: quarter-skeletons are too small for reliable individual discrimination with
    current 6-scalar features. Richer features (multi-scale, spine density) likely needed.

**Bisection within-type (40 × 23P, half-skeletons), `--encoder path` (TreeDNAEncoder):**
  - Random init AUC: **0.740** — nearly identical to cross-type bisection (0.728!)
  - Trained AUC: **0.725** (−0.015 — training *hurts*, encoder collapses)
  - Spatial baseline: 0.475
  - Training signal: pos_cos 0.993, neg_cos 0.993 at epoch 80 — fully collapsed, no separation
  - **Interpretation**: the 6-scalar path features carry individual-level signal at random init
    (0.74 is above chance and comparable to the cross-type start). BUT the triplet loss fails
    to discriminate — it collapses all within-type pairs to the same region of embedding space
    because positives and negatives look too similar for the loss to find a gradient direction.

**Bisection within-type, `--encoder gnn` (SkeletonGNN):** *pending*

### Summary table

| Experiment | Encoder | Random init AUC | Trained AUC | Δ | neg_cos (final) |
|---|---|---|---|---|---|
| Cross-type bisection (40 neurons) | path | 0.728 | 0.897 | +0.169 | 0.64 |
| Within-type bisection (40 × 23P) | path | 0.740 | 0.725 | −0.015 | 0.993 (collapsed) |
| Within-type 4-chunk (30 × 23P) | path | 0.599 | 0.687 | +0.089 | 0.822 (partial) |
| Within-type bisection (40 × 23P) | gnn | *pending* | *pending* | — | — |

### Why SkeletonGNN replaces TreeDNAEncoder

`TreeDNAEncoder` samples root-to-leaf paths and encodes each with a Transformer over hand-crafted
features `(dx, dy, dz, step_dist, arc_norm, turn)`.  The direction features `dx/dy/dz` are in
global nm space — all L2/3 pyramidal (23P) neurons have apical dendrites pointing roughly the same
direction, so within-type negatives look identical to positives.

`SkeletonGNN` receives centroid-normalised `(x-cx, y-cy, z-cz, radius)` — orientation-free by
design. The relative geometry and branching topology are emergent in the learned message-passing
representations rather than baked in as fixed features.

### Next steps for the training recipe (both encoders)
- **Hard negative mining**: sample negatives with highest current cosine similarity instead
  of random — the encoder is never forced to push apart the nearby pairs it's confusing
- **InfoNCE / NT-Xent loss** (SimCLR-style): normalised temperature-scaled cross-entropy
  doesn't suffer from the same collapse mode as triplet loss; within-batch negatives provide
  denser gradient signal
- **Larger margin**: current margin=1.0 may be too loose; try margin=2.0 to enforce wider separation
- **More paths** (n_paths=16 instead of 6): larger variance in path samples per fragment
  stabilises the training signal for within-type cases

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
- [ ] Within-type GNN ablation with `--encoder gnn` to compare against path baseline

## See also
- `docs/roadmap_global_assembly.md` — canonical north-star roadmap
- `docs/stage_ownership.md` — stage→module ownership map
