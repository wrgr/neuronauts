# Status

## Phase 0 — COMPLETE
Merged `claude/intelligent-planck-oCPwt` on 2026-06-05.

Delivered:
- `neuronauts/schemas.py` — typed inter-stage contracts (Region, Fragment, NeuronHypothesis, ConnectomeGraph)
- `neuronauts/legacy/` — v1 agent/membrane stack quarantined; `pytest -m 'not legacy'` skips it
- `CONTRIBUTING.md` + `docs/stage_ownership.md` — contributor docs and ownership map
- `tests/test_schemas.py` — smoke tests for all contracts

## Phase 1 — COMPLETE
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

**Bisection within-type (40 × 23P, half-skeletons), `--encoder gnn` (SkeletonGNN):**
  - Random init AUC: **0.768** — higher than path encoder at random init (0.740)
  - Trained AUC: **0.829** (+0.061) ✓ clears the 0.75 individual-identity bar
  - Spatial baseline: 0.475
  - Training signal: pos_cos 0.990→0.730, neg_cos 0.988→0.345 — **no collapse**
  - **Interpretation**: raw centroid-normalised (x,y,z,r) features break the orientation
    degeneracy that collapses the path encoder on within-type negatives. The GNN learns
    individual morphological identity within a cell type. The bottleneck was the features,
    not the loss function.

**Multi-fragment within-volume (50 neurons, 4-way split, volume 300–5000 µm³):**
Both encoders run on the *same* fetched skeletons (seed 42) for an apples-to-apples comparison.

  - `--encoder path` (TreeDNAEncoder):
    - Spatial baseline: **0.515** (chance — uniform synapse positions)
    - Random init AUC: **0.617**; Trained AUC: **0.626** (+0.009)
    - pos_cos 0.943, neg_cos 0.939 at ep 80 — **collapsed**
  - `--encoder gnn` (SkeletonGNN):
    - Random init AUC: **0.600**; Trained AUC: **0.599** (−0.0004)
    - loss pinned at **exactly 1.0 = the triplet margin** from epoch 10; pos_cos = neg_cos ≈ 0.998 — **total collapse to a single point**

  - **Key finding — the bottleneck is fragment *size*, not encoder architecture.** SkeletonGNN
    cleared the within-type bar on *half*-skeletons (0.768→0.829) but collapses just as hard as the
    path encoder on *quarter*-skeletons (4-way split). Quarter-skeletons carry too little
    morphological signal for individual identity: anchor/positive/negative distances all go to ~0,
    so triplet loss saturates at the margin and produces no gradient. This matches the earlier
    path-encoder result on within-type 4-chunk (partial, 0.599→0.687). **Conclusion: fragment
    granularity must stay at half-skeleton scale or coarser; the fix for finer fragments is a
    non-collapsing objective (InfoNCE/NT-Xent), not a different encoder.**

**NT-Xent ablation (GNN encoder, 4-way 50-neuron vol-filter, seed 42):**
  - NT-Xent loss (τ=0.1) also collapses: pos_cos = neg_cos ≈ 0.997, loss pinned at log(2N−1)=4.60
  - Trained AUC: **0.604** (+0.008) — no meaningful improvement over triplet collapse
  - **Root cause**: with all embeddings clustered near the same direction (cos≈0.997), NT-Xent
    gradient ∝ (softmax − y) ≈ 1/N per pair → effectively zero. NT-Xent shares the same
    uniform-collapse fixed point as triplet loss when morphological signal is insufficient.
    This is a *data* limitation (within-type quarter-skeletons are genuinely indistinguishable),
    not a loss-function limitation. NT-Xent does NOT fix data-induced collapse.

**NT-Xent ablation (GNN encoder, 2-way 40-neuron diverse, seed 42):**
  - Trained AUC: **0.740** (+0.001) vs triplet loss's **0.829** (+0.061) — significant regression
  - Loss is driven by the *high initial cosine similarity* of GNN embeddings (cos≈0.97 at init):
    NT-Xent gradient ∝ 1/N is too small to move the network when all embeddings are already
    clustered; triplet loss has constant gradient regardless of embedding similarity.
  - **Conclusion**: NT-Xent is worse than triplet/cosine contrastive for our GNN architecture.
    The GNN emits high-cosine embeddings at initialization; triplet's constant gradient
    is better suited. NT-Xent would benefit from much higher temperature (τ≥1.0) or explicit
    variance regularization (VICReg) to prevent the high-cosine collapse at initialization.

**NT-Xent ablation (path encoder, 2-way 40-neuron diverse, seed 42):**
  - Trained AUC: **0.852** (+0.123) vs triplet loss's **0.897** (+0.169) — small regression
  - The path encoder benefits from random path-sampling diversity at init (lower initial cos than GNN)
    so NT-Xent gradient is stronger, but triplet still wins because triplet runs ~5 optimizer
    steps/epoch vs. NT-Xent's ~2 steps/epoch (same batch_size=32).  Per-step training signal is
    comparable; NT-Xent needs ~200 epochs to match triplet's 80 epochs.

### Summary table

| Experiment | Encoder | Random init AUC | Trained AUC | Δ | neg_cos (final) |
|---|---|---|---|---|---|
| Cross-type bisection (40 neurons) | path | 0.728 | 0.897 | +0.169 | 0.64 |
| Within-type bisection (40 × 23P) | path | 0.740 | 0.725 | −0.015 | 0.993 ✗ collapsed |
| Within-type 4-chunk (30 × 23P) | path | 0.599 | 0.687 | +0.089 | 0.822 (partial) |
| Within-type bisection (40 × 23P) | **gnn** | **0.768** | **0.829** | **+0.061** | **0.345 ✓** |
| Multi-fragment 4-way (50 neurons, vol-filter) | path | 0.617 | 0.626 | +0.009 | 0.939 ✗ collapsed |
| Multi-fragment 4-way (50 neurons, vol-filter) | gnn | 0.600 | 0.599 | −0.000 | 0.998 ✗ collapsed (loss=margin) |

**Reading the table:** GNN beats path only when fragments are large enough (half-skeleton). At
quarter-skeleton granularity *both* encoders collapse under triplet loss — the two
half→quarter rows bracket the size threshold where individual-identity signal runs out.

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
- **VICReg** (variance-invariance-covariance regularization): adds an explicit variance term
  that prevents dimensional collapse; complements triplet loss without replacing it; the right
  lever for fine-grained (quarter-skeleton) fragments where data signal is weak
- **Larger margin**: current margin=1.0 may be too loose; try margin=2.0 to enforce wider separation
- **More paths** (n_paths=16 instead of 6): larger variance in path samples per fragment
  stabilises the training signal for within-type cases
- ~~**InfoNCE / NT-Xent loss**~~ — tried and reverted; see NT-Xent ablation above for diagnosis

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

## Phase 2.1 — IN PROGRESS (Half-Synapse Graph Reformulation)
Branch: `claude/tree-dna-phase-1-G1DNn`

**Problem reformulation:** The correct task is to partition the half-synapse set such that
each partition belongs to a single neuron.  The v117 segmentation is noisy evidence (split +
merge errors) but informative — used as a soft evidence channel, not as ground truth.

**Key design choices:**
- Nodes = half-synapses (pre-side and post-side treated as independent partition problems)
- Node features = concat(normalised position, seg DNA from SkeletonGNN)
  — position captures local context; DNA captures "what kind of neuron am I on"
- Edge type 0 (same-segment): strong topological evidence, may span frankenmerges
- Edge type 1 (spatial k-NN): weak proximity evidence
- Edge feature = [type_onehot(2), cosine_similarity(dna_i, dna_j)] — DNA cos-sim is physically
  motivated: neurons are connected trees, so same-neuron segments share morphological character
- Ground truth = label-version root IDs (supervision only, never in node features)
- Evaluation = ARI (Adjusted Rand Index, partition quality), not cosine AUC

### Checklist
- [x] `neuronauts/assemble/half_synapse_graph.py` — `HalfSynapseGraph`, `build_half_synapse_graph`
- [x] `neuronauts/assemble/partition_gnn.py` — `HalfSynapseGNN`, `train_partition_gnn`, `partition_half_synapses`, `evaluate_partition_ari`
- [x] `tests/test_assemble_half_synapse.py` — 24 tests (graph construction, GNN forward/training, ARI evaluation)
- [x] `scripts/half_synapse_ablation.py` — end-to-end `--synthetic` mode smoke test
- [x] **Synthetic ablation (10 neurons × 2 segs × 8 syn/seg, frankenfraction=0.2)**:
  - ARI random init: **0.585**  (already above chance — same-seg edges provide structure)
  - ARI trained (40 epochs): **0.661** (+0.076) ✓ training improves partition quality
  - Training signal: pos_sim 0.878→0.975, neg_sim 0.601→−0.161 — model genuinely learns
  - **Interpretation**: remaining gap is from missing cross-segment spatial edges (inter-segment
    distances exceed k-NN reach with current skeleton offsets); endpoint-adjacent edges will close this
- [x] `neuronauts/assemble/fragment_graph.py` — `build_fragment_graph`, `assemble_fragments`, `score_edge` (endpoint-proximity stitching, Phase 2.2 foundation)
- [x] `tests/test_assemble_fragment_graph.py` — 22 tests (graph construction, scoring, union-find clustering, cross-region spans, pooled DNA, degree cap)
- [x] `scripts/multi_fragment_ablation.py` — N-way volume-filtered ablation (`--volume-min/max` for same-type proxy, `--n-splits N`, `--encoder [path|gnn]`)
- [x] **Multi-fragment 4-way ablation, both encoders** (see Within-type summary table): isolates
      fragment *size* — not encoder choice — as the collapse driver. GNN beats path only at
      half-skeleton scale; both collapse on quarter-skeletons under triplet loss.
- [x] **NT-Xent ablation** — tried and reverted: NT-Xent collapses identically on 4-way within-type
      (data limitation, not loss limitation) and gives worse AUC on 2-way diverse (GNN: 0.829→0.740,
      path: 0.897→0.852). **Triplet/cosine contrastive is retained** — better gradient signal for our
      initialization regime. VICReg (variance regularization) is the next lever if we need fine-grained
      fragments; see summary above for diagnosis.
- [x] Endpoint-adjacent edges wired into `build_half_synapse_graph` — new edge type 2,
      `endpoint_radius_nm` parameter, `max_endpoint_pairs` cap; `train_partition_gnn` auto-detects
      `n_edge_types` from graph; hard neg pool extended to include type-2 cross-neuron edges;
      4 new tests added (`test_endpoint_adj_absent_by_default`, `test_endpoint_adj_edges_when_close`,
      `test_endpoint_adj_edge_feat_shape`, `test_endpoint_adj_cos_sim_in_feat`,
      `test_gnn_auto_detects_3_edge_types`)
- [ ] Real-data ARI evaluation with CAVE v117 seg IDs

## See also
- `docs/roadmap_global_assembly.md` — canonical north-star roadmap
- `docs/stage_ownership.md` — stage→module ownership map
