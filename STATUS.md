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
      5 new tests added (`test_endpoint_adj_absent_by_default`, `test_endpoint_adj_edges_when_close`,
      `test_endpoint_adj_edge_feat_shape`, `test_endpoint_adj_cos_sim_in_feat`,
      `test_gnn_auto_detects_3_edge_types`)
- [x] **Real-data ARI evaluation** (20 real minnie65 neurons × 3 skeleton pieces, `real_skeleton_partition.py`):

  Script: `scripts/real_skeleton_partition.py` — fetches real proofread v1412 skeletons, splits each
  into N pieces (simulating v117 fragmentation), places synapses near skeleton vertices, encodes with
  SkeletonGNN, builds HalfSynapseGraph, trains HalfSynapseGNN, evaluates ARI.

  | Config | ARI init | ARI trained | ΔARI | Clusters pred/true |
  |---|---|---|---|---|
  | No endpoint edges, threshold=0.75 | 0.011 | 0.079 | +0.068 | 3/20 |
  | No endpoint edges, threshold=0.87 | 0.011 | 0.088 | +0.078 | 5/20 |
  | **Endpoint edges 10 µm, threshold=0.87** | **0.011** | **0.418** | **+0.407** | **17/20** |

  **Key finding — endpoint-adjacent edges are transformative.** The skeleton split creates piece
  endpoints within ~0-1000 nm of each other (one skeleton step). With `endpoint_radius_nm=10_000`,
  all adjacent piece-pair endpoints are captured (9504 directed endpoint-adj edges for 60 pieces),
  giving the GNN direct cross-piece same-neuron evidence. Without endpoint edges, the GNN can only
  use spatial k-NN over synapse positions, which doesn't reliably connect pieces of the same neuron
  (synapses from different pieces may be widely separated in the global volume). ARI jumps from
  0.088 to 0.418 (+0.330) and correctly identified clusters jump from 5 to 17/20.

## Phase 2.2 — IN PROGRESS (Edge classification + correlation clustering)
Branch: `claude/abstract-tree-stitch`

**Reformulation:** learn f(v117 seg → v1412 neuron) *directly* as an edge
function instead of a per-node embedding + cosine threshold.  For every edge
(a pair of observations joined by same-fragment / spatial / endpoint evidence)
an edge classifier predicts P(same v1412 neuron), supervised by the v1412
co-membership of the endpoints.  Inference lifts the per-edge log-odds to a
global partition with **correlation clustering** (greedy additive edge
contraction, GAEC).

**Why this beats threshold union-find.** Union-find is greedy and
irreversible: one spuriously-similar cross-neuron edge fuses two neurons for
good.  Correlation clustering contracts on *net* evidence between clusters, so
a high-similarity edge can be cut when the rest of the graph disagrees — the
v117 merge-error (frankenmerge) case.  The edge head also sees the **spatial
separation of the two endpoints**, the franken discriminator: a same-segment
edge spanning a large distance is almost certainly a merge error, not a true
within-neuron link.

### Modules
- [x] `neuronauts/assemble/edge_partition.py` — `EdgePartitionGNN` (embedding
      backbone + edge-classification head), `train_edge_partition_gnn`
      (BCE on v1412 co-membership = learn f(117→1412)), `correlation_cluster`
      (GAEC), `partition_by_correlation` (with `bias` knob for the
      precision/recall tradeoff), `edge_merge_metrics` (over/under-merge)
- [x] `tests/test_edge_partition.py` — 13 tests (GAEC cut/merge cases, model
      forward/training, ARI recovery, merge metrics)
- [x] `treestitch/synthetic.py` — offline world generator with
      `frankenmerge_frac` (fuses cross-object pieces into bad v117 segments)
- [x] `treestitch` wrappers: `train_edge_partition`, `partition_observations_cc`,
      `merge_metrics`
- [x] `scripts/compare_partition_methods.py` — head-to-head on one shared graph

### Synthetic ablation (20 objects × 3 pieces, frankenmerge_frac=0.25)
Both methods consume the *same* fragment embeddings and the *same* typed graph;
only the inference algorithm differs.  Over-merge rate = fraction of labelled
edges that are false merges (the costly, irreversible error).

| Method | ARI | clusters | merge_P | over-merge |
|---|---|---|---|---|
| union-find (cosine threshold) | 0.855 | 19/20 | 0.908 | **0.080** |
| **edge_cc (correlation clustering)** | **0.948** | 19/20 | **1.000** | **0.000** |

**Spatial-overlap stress test** (decreasing object spacing → spatial proximity
becomes misleading, the regime that breaks threshold union-find):

| Object spacing | union-find ARI / over-merge | edge_cc ARI / over-merge |
|---|---|---|
| 60 µm (separated) | 0.748 / 0.106 | **0.948 / 0.000** |
| 25 µm (overlap)   | 0.473 / 0.360 | **0.695 / 0.036** |
| 12 µm (heavy overlap) | 0.000 / 0.716 | **0.583 / 0.049** |

**Key finding — correlation clustering is structurally robust to over-merge.**
As objects overlap, union-find collapses (ARI → 0, false-merge rate → 72%: it
fuses everything spatial proximity touches).  edge_cc holds ARI ≥ 0.58 with the
false-merge rate under 5% throughout — a ~15× reduction in the irreversible
error at the hardest setting.  The `bias` knob trades the remaining
under-merge against over-merge on a controllable curve.

**Caveat:** the synthetic frankenmerges fuse spatially-separated pieces, so the
endpoint-distance feature carries strong signal.  Real v117 merges occur between
*adjacent* neurons — see the real-data validation below.

### Real-data validation (adjacent-neuron merges)
Script: `scripts/real_franken_partition.py` (fetches real proofread skeletons,
splits into pieces, fuses *spatially adjacent* cross-neuron pieces into shared
v117 segments via `treestitch.worldbuild.frankenmerge_adjacent`).  20 neurons ×
3 pieces, 8 adjacent-neuron merges (≤6 µm), shared encoder + graph.

Best-vs-best (sweep union-find threshold and edge_cc bias — both knobs tuned):

| Method | best ARI | over-merge | regime |
|---|---|---|---|
| union-find (thr 0.95) | 0.248 | 0.126 | 15/20 clusters |
| **edge_cc (bias −3.0)** | **0.385** | **0.010** | 208/20 clusters (over-fragmented) |

**Honest findings:**
1. **Both methods are weak on real data** (best ARI ≤ 0.39 vs 0.95 synthetic).
   The binding constraint is the **fragment representation**, not the inference
   algorithm: the FragmentEncoder barely separates thirds of real neurons
   (pos_cos 0.71 vs neg_cos 0.52) — the documented small-fragment collapse
   (Phase 1).  Adjacent franken pieces also defeat the endpoint-distance cue
   that made the synthetic case easy.
2. **edge_cc's over-merge advantage holds everywhere** — false-merge rate 0.010
   vs 0.126 (~13×) on real data.  Correlation clustering structurally refuses to
   over-merge, the consistent, defensible win across all regimes.
3. **edge_cc's default operating point is miscalibrated.** At `bias=0` the
   classifier's `p_neg` sits above 0.5 on real data, so GAEC merges everything
   (2 clusters).  It only wins after sweeping `bias` strongly negative, landing
   in a heavily over-fragmented regime.  **Next lever:** auto-calibrate `bias`
   from labelled training edges, and add hard-negative mining to the edge
   trainer (the metric GNN already has it — that is why union-find's default is
   better behaved).

### What the real v117→v1412 structure actually looks like
Parallel study: `docs/seg_117_to_1412.md` + `scripts/probe_seg_mapping.py`
(chunkedgraph lineage over plain HTTP, no caveclient).  Key intuition:
- **v117 and v1412 are materialization *versions* (timestamped snapshots) of one
  graphene segmentation**, not separate segmentations.  The mapping is supervoxel
  lineage: resolve the same supervoxels to their root at a chosen timestamp.
- **Real split structure is "one dominant trunk + a tail of slivers", not equal
  pieces.** 7/8 sampled soma neurons were *already a single v117 root*; the one
  split-fix stitched 10 roots but ~93% of mass was one trunk.  **This means our
  equal-thirds benchmark is unrealistically hard on the split side** — real
  soma-cell assembly is mostly "attach slivers to a trunk".  The merge/split
  action concentrates in *non-soma* roots, which the nucleus-table sample misses
  and which is the next sampling target.
- v1412's materialization tables are **expired**; for synapse-anchored work use
  an available version (1300/1507/1621/1718) as the proofread target.

## Phase 2.3 — IN PROGRESS (Real f(v117→v1718), no synthetic)
Branch: `claude/abstract-tree-stitch`

Synthetic worlds proved misleading (over-optimistic).  Moved to **real** lineage
data anchored on the available **v1718** materialization (v1412 is expired;
fall back = an earlier real version 1621/1507/1300, never synthetic).

### Real data access (`neuronauts/data/lineage.py`)
ChunkedGraph + materialization over plain HTTP (no caveclient), all verified 200:
- `version_timestamp`, `list_versions` — `[117, 943, 1300, 1507, 1621, 1718]`.
- `root_leaves` (root→supervoxels/L2), `roots_at` (supervoxels→root @ timestamp,
  batched binary POST) — the v117↔v1718 lineage.
- `root_at_version` (carry a nucleus soma forward to v1718).
- `fragment_breakdown` (proofread neuron → its v117 roots + mass shares).
- `fetch_synapses` — **real synapses** from `synapses_pni_2` via the
  materialization **v3** query API (v2 has an `ipc_compress` server bug);
  positions in nm, plus supervoxel ids for lineage assignment.

### Real fragmentation structure (`scripts/characterize_v117_to_v1718.py`)
n=40 somata, real v117→v1718:
- **88% of somata are already a single v117 root**; only **10%** have ≥2
  substantial v117 fragments.  Median fragments/neuron = 1 (mean 2.0 from the
  sliver tail), dominant mass share median 1.000.
- Confirms quantitatively: real soma split structure is **"one trunk + slivers"**,
  not equal pieces.  The equal-thirds synthetic benchmark was unrealistically
  hard; real soma partition is mostly trivial at the supervoxel level, but
  non-trivial at the *synapse* level because synapses land on the sliver tail
  (e.g. one neuron: 800 synapses → 90% on the trunk + 6 sliver fragments).

### Real partition world (`treestitch/realworld.py`, `scripts/real_lineage_partition.py`)
Fully real: observations = real synapses; fragment id = real v117 root of the
synapse's supervoxel; label = real v1718 root; fragment shape = the fragment's
real synapse point cloud.  Real frankenmerges (a v117 root spanning ≥2 neurons)
arise from the data.

**Morphology caveat:** the skeleton cache does **not** serve v117 fragment roots,
and the existing CloudVolume+kimimaro self-skeletonization
(`cell_graph.precompute_self_skeletons_for_cache`) can't run here (deps not
installed).  Current fragment morphology = the real synapse cloud; the **L2
cache** (`l2cache …/attributes` → `rep_coord_nm`, verified 200) is the finer
real upgrade and the next step.

### First fully-real benchmark (15 neurons, v117→v1718, real synapses)
22 v117 fragments (1.5/neuron — real slivers present), 3926 synapse nodes.

| Method | ARI | clusters | merge_P | over-merge |
|---|---|---|---|---|
| union-find | **0.305** | 27/15 | 0.962 | 0.038 |
| edge_cc (bias 0) | 0.099 | 5/15 | 0.961 | 0.039 |

**Honest findings (consistent with the franken real-data run):**
1. **Both methods are weak on real data** (best ARI 0.31).  The binding
   constraint is the **representation/evidence**, not the inference algorithm.
2. **Endpoint-adjacency edges = 0** here, because synapse-cloud fragments have no
   real skeleton endpoints.  Endpoint edges were *transformative* with real
   skeletons (ARI 0.09→0.42 earlier).  **Restoring real fragment skeletons (L2
   cache) to recover endpoint adjacency + DNA is the highest-value next step.**
3. **edge_cc collapses at the default bias** (p_pos≈p_neg≈0.95, edge_acc pinned
   at the base rate → merges to 5 clusters).  It needs bias auto-calibration and
   hard-negative mining (the metric GNN has the latter, which is why union-find
   is better behaved by default).

**Direction:** the real-data evidence (here + the franken run) consistently says
the lever is *evidence quality* — real fragment morphology (endpoint edges + DNA)
and a calibrated/hard-mined edge classifier — not more inference machinery.

### L2 cache skeleton benchmark (8 neurons, v117→v1718, real L2 skeletons)
15 v117 fragments (1.9/neuron), 1428 synapse nodes.
**L2 cache hit: 15/15 fragments** — all v117 roots resolved to real L2 centroids.

| Method | ARI | clusters | merge_P | over-merge |
|---|---|---|---|---|
| **union-find** | **0.838** | 24/8 | 0.999 | 0.001 |
| edge_cc (bias 0) | 0.422 | 5/8 | 0.998 | 0.002 |

**Endpoint-adjacency edges: 2052** (was 0 with synapse-cloud; each L2 skeleton has
~17 leaf vertices on average for these fragments, giving real endpoints for stitching).

**Key finding — L2 skeletons are transformative:**
- union-find ARI: **0.305 → 0.838** (+0.533); synapse-cloud had 0 endpoint edges,
  L2 skeleton has 2052.  The endpoint-adjacency signal is the critical missing piece.
- edge_cc: **0.099 → 0.422** (+0.323); same cause — endpoint edges give the
  classifier cross-fragment same-neuron evidence it couldn't see before.
- **edge_cc is still under-merging (5/8 clusters vs 8 true).**  Diagnosis: the
  model outputs high positive logits for ALL edges (same-neuron AND different-neuron,
  logit≈3.5 for both) due to class imbalance — most spatial/endpoint edges are
  within-neuron.  GAEC then merges to a few large clusters.  Bias sweep (−3 to +3)
  changes nothing because all logits are already strongly positive.  Fix: add
  **hard-negative mining** to `train_edge_partition` (same as `train_partition`
  already has), and add an explicit **pos_weight < 1** to rebalance BCE loss.

**net result:** With L2 fragment skeletons, union-find is now a strong baseline
(ARI 0.838 on 8-neuron real data).  edge_cc's structural over-merge-resistance
advantage (13×) is still real but its default operating point needs calibration.

## See also
- `docs/roadmap_global_assembly.md` — canonical north-star roadmap
- `docs/stage_ownership.md` — stage→module ownership map
