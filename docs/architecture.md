# System Architecture

This document explains what neuronauts is trying to do, why the previous
approach hit a wall, and how the current design breaks through it. Written
for a contributor who knows PyTorch and connectomics basics but hasn't read
the codebase.

---

## 1. The ceiling problem

The original CellGNN (`neuronauts/cell_graph.py`) builds a K-NN proximity
graph over synapse positions **inside a single 30 µm box**, runs message
passing, and clusters the resulting synapse embeddings into neuron
assignments. It works per box — there is no stitching across box boundaries
anywhere in the codebase.

That choice creates a hard ceiling. Every large pyramidal neuron has an
apical dendrite that spans the full cortical column — hundreds of
micrometers. Box-local partitioning cannot assemble a neuron that is
structurally larger than the box. The consequence is visible in the numbers:

| Metric | Value |
|---|---|
| Pairwise merge accuracy (within-box) | 85%+ |
| Line-graph F1 at column scale | ~0.27 |

The gap is not a model capacity problem. It is an architectural problem: the
model is strong at local decisions and incapable of cross-box decisions
because no cross-box decision is ever made. See `docs/ablation_results.md`
for the per-feature ablation that confirmed the scalar features are largely
redundant and the model is leaving signal on the table.

The fix is to stop partitioning by box entirely. The global pipeline nodes
are synapses drawn from a region with no artificial boundary, and their
primary feature is a learned morphological embedding — tree-DNA — that
travels with each segmentation root regardless of which box that root
happens to intersect.

---

## 2. The three-stage pipeline

```
data/
  Region{synapses, root_ids, bbox, version metadata}
      |
      v  skeleton_to_fragment / extract_fragments_for_region
represent/
  Fragment{vertices_nm, edges, radius_nm, endpoints_nm, dna=None}
      |  train_dna_encoder / encode_fragments
      v  dna filled → Fragment{..., dna=[D]}
assemble/
  GlobalSynapseGraph → CellGNN → NeuronHypothesis[]
      |
      v
connectome/
  ConnectomeGraph
```

Each stage reads and writes **one typed artifact**, defined in
`neuronauts/schemas.py`. A stage owner depends only on the *schema* of the
upstream artifact, not on the upstream code. Stages can be developed,
cached, and tested independently.

The full stage-to-module ownership map is in
[`docs/stage_ownership.md`](stage_ownership.md).

---

## 3. Contracts (`neuronauts/schemas.py`)

All inter-stage artifacts are frozen dataclasses with plain numpy fields,
explicit `validate()`, and pickle-free `.npz` I/O. The current
`SCHEMA_VERSION` is 1. Any change to a field dtype or shape is a
cross-team change — flag it in the PR description and bump `SCHEMA_VERSION`
if the on-disk format changes.

### `Region` — data stage output

Immutable input contract for a spatial tile. Carries all synapses for the
tile in **global nanometers** (not box-relative voxels — the conversion
happens at the `data/` boundary and everything downstream uses global nm).

```python
@dataclass
class Region:
    region_id: str
    bbox_nm: BBox              # (min_xyz, max_xyz) in global nm
    voxel_size_nm: Vec3
    seg_version: int           # base materialization: where skeletons come from
    label_version: int         # target materialization: ground-truth root IDs
    pre_pt_nm: np.ndarray      # [N, 3] float32
    post_pt_nm: np.ndarray     # [N, 3] float32
    pre_root_id: np.ndarray    # [N] int64 @ label_version
    post_root_id: np.ndarray   # [N] int64 @ label_version
    synapse_id: np.ndarray     # [N] int64
    pre_seg_id: np.ndarray | None   # [N] int64 @ seg_version (scaffold)
    post_seg_id: np.ndarray | None
```

`seg_version` and `label_version` must be kept separate. Skeletons must
come from the base materialization — not the target — to avoid leaking
ground-truth labels into node features. The existing
`skeleton_graph.validate_skeleton_graph_config` enforces this and must
remain wired through any new `data/` stage code.

### `Fragment` — represent stage output

One per segmentation root. Wraps the entire kimimaro skeleton tree for that
root. The `dna` field starts as `None` when the fragment is extracted; the
`represent/` stage fills it.

```python
@dataclass
class Fragment:
    fragment_id: int           # globally unique across all regions
    region_id: str
    base_root_id: int          # seg root @ Region.seg_version
    vertices_nm: np.ndarray    # [V, 3] float32, global nm
    edges: np.ndarray          # [E, 2] int64, local vertex indices
    endpoints_nm: np.ndarray   # [T, 3] float32, leaf vertices (seam handles)
    radius_nm: np.ndarray      # [V] float32, inscribed-sphere caliber
    synapse_indices: np.ndarray  # [S] int64, rows of the owning Region
    dna: np.ndarray | None     # [D] float32, filled by represent/
```

`endpoints_nm` are the leaf vertices of the skeleton tree (degree ≤ 1).
These are the "seam-stitch handles" for Phase 2 global assembly — two
fragments in adjacent tiles can be joined when their endpoints are within
ε nm and their tangents align.

`fragment_id` is set to the `base_root_id` (globally unique across
regions). Fragments with fewer than 3 skeleton vertices are discarded by
`skeleton_to_fragment`.

### `NeuronHypothesis` — assemble stage output

A set of fragments asserted to belong to the same neuron. The
`spans_regions` field is direct evidence of cross-box assembly — a
hypothesis listing more than one region is something the box-local pipeline
cannot produce.

```python
@dataclass
class NeuronHypothesis:
    neuron_id: int
    fragment_ids: list[int]
    synapse_indices: np.ndarray    # [S] int64, global synapse rows
    pooled_dna: np.ndarray | None  # [D] float32, mean-pooled DNA
    spans_regions: list[str]
```

### `ConnectomeGraph` — connectome stage output

Directed neuron × neuron graph. `node_features` are intended to hold
pooled tree-DNA per neuron plus connectivity statistics. Currently this is a
target interface; the `experiments/soma_graph/` code runs on placeholder
features today.

```python
@dataclass
class ConnectomeGraph:
    neuron_ids: np.ndarray          # [M] int64
    node_features: np.ndarray       # [M, F] float32
    src: np.ndarray                 # [E] int64
    dst: np.ndarray                 # [E] int64
    edge_synapse_count: np.ndarray  # [E] int64
```

---

## 4. Data stage: skeleton → Fragment

**Modules:** `neuronauts/data/fragments.py`

**Entry points:**
- `skeleton_to_fragment(vertices_nm, edges, radii_nm, base_root_id, region, fragment_id)` — wraps a single kimimaro skeleton tree as a Fragment
- `extract_fragments_for_region(region, skeleton_archive_path)` — loads a `.npz` archive produced by `precompute_self_skeletons_for_cache` and returns one Fragment per valid seg root

The archive is produced by the existing kimimaro pipeline
(`cell_graph.precompute_self_skeletons_for_cache`). Skeleton vertices are
already in global nm in the archive.

**Contaminated roots:** a seg root that maps to more than one
`label_version` root is a false-merge survivor. `extract_fragments_for_region`
does not filter these — it has no access to the label mapping — but
`train_dna_encoder` accepts a `root_label_map` parameter that excludes
contaminated roots from both positive and negative pairs so the encoder
never learns a coherent identity for a merged seg root.

---

## 5. Represent stage: tree-DNA encoder

**Modules:** `neuronauts/represent/dna.py`, `neuronauts/represent/enrich.py`

**The core idea.** Each segmentation root has a skeleton tree. That tree
encodes the neuron's morphological identity: branch caliber, tortuosity,
tangent flow, branching topology. Two fragments from the same neuron (even
partial views of it) should produce similar embeddings. Two fragments from
different neurons should produce dissimilar embeddings. A translation-
invariant embedding of this identity is what allows cross-box matching —
the box coordinate drops out.

### 5.1 Architecture (`TreeDNAEncoder`)

1. **Path sampling.** `sample_tree_paths` finds all leaf vertices (degree ≤ 1)
   in the skeleton tree and samples up to K random leaf-to-leaf paths via BFS.
   For a chain (exactly 2 leaves) there is one path. For a tree with ≥ 3
   leaves, K random leaf pairs are drawn. Default K = 16.

2. **Featurization.** `featurize_fragment` runs `featurize_path_points` on each
   sampled path. Default mode is `"raw_delta3+skeleton"`, which produces 6
   features per step: `(dx, dy, dz, skel_step_dist, skel_norm_arc, skel_turn)`.
   Output per path: `[T-1, 6]`.

3. **Transformer encoding.** A shared `TorchPathEncoder` (from `grammar.py`)
   takes a padded batch of `[N_paths, T_max, D]` sequences and returns
   `[N_paths, output_dim]` embeddings via CLS-style pooling. Default:
   d_model=64, 2 layers, 4 heads, output_dim=64.

4. **Mean pooling.** Path embeddings for one fragment are mean-pooled to a
   single `[D]` vector. This makes the encoder invariant to path order and
   robust to variable-topology trees.

The full forward pass in `TreeDNAEncoder.forward`:
```
path_features_batch: list[list[ndarray[T_k, 6]]]  # B fragments × K paths
→ flatten to [B·K, T_max, 6]
→ TorchPathEncoder → [B·K, D]
→ mean-pool per fragment → [B, D]
```

### 5.2 Training (`train_dna_encoder`)

Triplet contrastive loss with online triplet mining:

- **Positive pairs:** two fragments with the same `base_root_id` that both
  map cleanly to a single `label_version` root. When only one fragment is
  available for a group, path augmentation is used: the same fragment is
  featurized twice with different random seeds to produce two views.
- **Negative pairs:** fragments from different `base_root_id` groups.
- **Contaminated roots** (`base_root_id` → more than one `label_version` root)
  are excluded from both sets.

The `root_label_map` parameter accepts `{base_root_id: set[label_root_ids]}`
from the CAVE base→target mapping. When omitted, `base_root_id` is used
directly as the identity proxy (noisier, but works for synthetic experiments
where there are no false merges).

### 5.3 Ablation evaluation (`represent/enrich.py`)

`evaluate_dna_auc` measures how well trained DNA embeddings predict
same-neuron synapse pairs, compared to a spatial proximity baseline.

The task: given a random pair of synapses, predict whether they belong to
the same neuron. Score = cosine similarity between their fragment DNA vectors.
Baseline score = negative log-distance between synapse positions.

`build_synapse_dna_matrix` broadcasts each fragment's DNA embedding onto all
synapses assigned to that fragment (`Fragment.synapse_indices`).

---

## 6. Assemble stage: global synapse graph + GNN

**Modules:** `neuronauts/assemble/global_synapse_graph.py`,
`neuronauts/assemble/synapse_gnn.py`

**Status:** implemented and tested (Phase 2 is active). The box-local
`build_synapse_graph` in `cell_graph.py` is still present for legacy
compatibility; new assembly code lives in `assemble/`.

### 6.1 Global synapse graph (`GlobalSynapseGraph`)

```
build_global_synapse_graph(region, fragments, k_neighbors=8)
```

Every synapse in the region becomes a graph node. Node features are the DNA
embedding of the fragment that owns that synapse (broadcast from
`build_synapse_dna_matrix`). Edge construction uses a cKDTree over synapse
centroid positions `(pre_pt + post_pt) / 2`:

- K spatial nearest neighbours per synapse (both directions → up to 2K
  edges per node).
- No box boundary in the KDTree query — the whole region is one pool.
- Edge feature: `log(1 + dist / 10_000 nm)` (log-normalised distance,
  one scalar per edge).
- Optional `max_dist_nm` prunes long-range edges.

The output `GlobalSynapseGraph` carries:
```
node_feat   [N, D]   DNA per synapse (float32)
node_pos    [N, 3]   synapse centroid (float32)
edge_src    [E]      directed source indices (int64)
edge_dst    [E]      directed destination indices (int64)
edge_feat   [E, 1]   log-dist feature (float32)
pre_root_id [N]      ground-truth label per synapse (int64)
```

### 6.2 GNN training and inference

`train_global_gnn` wraps the existing `CellGNN` architecture with:
- `node_input_dim` = DNA dimension (e.g. 32)
- `edge_input_dim` = 1 (log-distance)
- `embedding_dim` = 32 (output per synapse)

Training uses cosine similarity contrastive loss with positive pairs
(same-root synapses) and negative pairs (different-root synapses), sampled
per epoch from `graph.pre_root_id`.

`run_global_gnn` returns L2-normalised per-synapse embeddings `[N, D]`.

`assemble_neurons` calls `partition_from_embeddings` (from `cell_graph.py`)
on the GNN embeddings to produce integer neuron labels per synapse.

### 6.3 Connecting GNN output to `NeuronHypothesis`

The current pipeline evaluates via cosine AUC over synapse pairs (see
`scripts/global_gnn_ablation.py`). The path to `NeuronHypothesis` artifacts
is: cluster the GNN synapse embeddings → one cluster = one candidate neuron
→ collect the fragment IDs that overlap each cluster → emit
`NeuronHypothesis(fragment_ids=..., spans_regions=...)`. This wiring is the
next open task in Phase 2.

---

## 7. Validation evidence

### Phase 1: hard-split ablation (primary validation)

The key question for Phase 2 is: can the encoder recognise two halves of the
same neuron as the same neuron? This is exactly the multi-root use case —
where a proofread neuron is covered by multiple segmentation roots.

The hard-split ablation (`scripts/half_split_ablation.py`) bisects each real
proofread neuron's skeleton at its balance edge (the edge that most evenly
splits vertex count), trains the encoder to treat both halves as the same
neuron, and evaluates the same-neuron AUC on held-out pairs.

| Condition | AUC |
|---|---|
| Spatial proximity baseline | 0.466 (chance) |
| DNA, random init | 0.728 |
| DNA, trained (80 epochs) | **0.897** (+0.169) |

Training signal: positive cosine 0.95 → 0.87, negative cosine 0.95 → 0.64.
The encoder learns to align same-neuron halves and push apart different-
neuron halves, despite seeing only a partial skeleton view for each.

This directly validates the multi-root use case: the encoder can recognise
that two fragments from the same neuron (which will have different
`base_root_id`s in unproofread data) belong together.

### Phase 1: 30-neuron real-data ablation

Running the same ablation without the bisection on 30 diverse proofread
neurons reaches DNA AUC 1.000 even at random initialisation, indicating that
the neurons are morphologically so distinct that any features are sufficient
to separate them. Harder evaluation (same-cell-type cohorts) is needed to
see training benefit at this scale.

### Phase 2: synthetic GNN ablation

On synthetic data with 40 neurons (2 fragments per neuron, 10 synapses per
fragment), the GNN adds a further +0.051 AUC over trained DNA alone:

| Condition | AUC |
|---|---|
| DNA (random init) | 0.787 |
| DNA (trained) | 0.863 |
| GNN on trained DNA | **0.914** |

The GNN's message passing propagates identity signal from DNA-rich synapses
to nearby synapses that might share a noisy or zero DNA vector, smoothing
the representation within neuron clusters.

---

## 8. Module map

```
neuronauts/
  schemas.py                    typed contracts (Region, Fragment,
                                NeuronHypothesis, ConnectomeGraph)
  data/
    fragments.py                skeleton_to_fragment,
                                extract_fragments_for_region
  represent/
    dna.py                      TreeDNAEncoder, train_dna_encoder,
                                encode_fragments, sample_tree_paths,
                                featurize_fragment
    enrich.py                   build_synapse_dna_matrix,
                                synapse_pair_dna_scores,
                                spatial_proximity_scores,
                                evaluate_dna_auc
  assemble/
    global_synapse_graph.py     GlobalSynapseGraph,
                                build_global_synapse_graph
    synapse_gnn.py              train_global_gnn, run_global_gnn,
                                assemble_neurons
  cell_graph.py                 CellGNN, partition_from_embeddings
                                (legacy box-local; CellGNN reused by
                                assemble/ as the GNN backbone)
  grammar.py                    TorchPathEncoder, featurize_path_points
                                (shared encoder building block)
  legacy/                       quarantined v1 agent/membrane stack
                                (not loaded by import neuronauts)
```

For the full ownership map (who owns each module) see
[`docs/stage_ownership.md`](stage_ownership.md).

---

## 9. Phase 2 open questions

**Seam stitching.** The `endpoints_nm` field on `Fragment` exists for this
purpose: two fragments from adjacent tiles can be joined when endpoints are
within ε nm, tangents align, and DNA cosine similarity is high. The learned
seam-stitch classifier described in `docs/roadmap_global_assembly.md`
(Section 3.2) is not yet implemented. The `NeuronHypothesis.spans_regions`
field is ready to record which tiles each assembled neuron crosses.

**Global evaluation metric.** The current ablation metric is synapse-pair
AUC — cheap to compute, but it does not measure whole-neuron completeness or
cross-box assembly quality. The column-scale line-graph F1 and per-neuron
completeness/purity metrics described in the roadmap are not yet wired in.

**Overlapping-region tiling.** The global pipeline currently treats each
Region as independent. The seam-stitch mechanism requires overlapping tiles
(core + halo) so that fragments near a seam appear in two adjacent regions
and can be matched. The `experiments/minnie_column/` binning and
`dedup.py` stable-key logic is the intended reuse point for deduplication
when synapses appear in multiple tiles.

**Contamination at scale.** The `root_label_map` contamination filter works
for small real-data experiments. At full Minnie65 scale, the base→target
mapping covers the whole volume and the filter needs to be wired into the
`data/` stage rather than passed per-call into the training loop.

**CellGNN monolith split.** `cell_graph.py` (3,950 lines) does graph
building, GNN, clustering, skeletonization, seg scoring, beam search, and
tangledness checking. The roadmap calls for splitting it into
`assemble/graph.py`, `assemble/gnn.py`, and `assemble/partition.py`. Until
that split lands, `assemble/synapse_gnn.py` imports `CellGNN` and
`partition_from_embeddings` directly from `cell_graph.py`.

---

## 10. Key invariants for new code

- All coordinates in artifacts are **global nanometers**. Convert at the
  `data/` boundary using `Region.from_synapse_table`; never pass box-
  relative voxel coordinates downstream.
- `Fragment.dna` is `None` until the `represent/` stage fills it.
  Downstream code must handle `None` gracefully or assert it is set.
- Skeletons come from `seg_version` (base materialization), not
  `label_version` (target). Ground-truth root IDs come from
  `label_version`. Mixing them leaks labels.
- Contaminated roots (false-merge survivors) must be masked out of positive
  pairs during training. Do not include them in negatives either — a
  contaminated root is not a clean counter-example for any single neuron.
- Tests must not require network access or a CAVE token. Tests that import
  `caveclient` collect only when the `cave` extra is installed.
