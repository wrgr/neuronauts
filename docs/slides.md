---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 28px;
    color: #1a1a2e;
    padding: 48px 64px;
  }
  section.title {
    background: #0f3460;
    color: #eaeaea;
    text-align: center;
    justify-content: center;
  }
  section.title h1 { font-size: 56px; color: #e94560; margin-bottom: 12px; }
  section.title h2 { font-size: 28px; color: #a8dadc; font-weight: 400; }
  section.title p  { color: #a8dadc; font-size: 20px; margin-top: 32px; }
  section.section-header {
    background: #16213e;
    color: #eaeaea;
    justify-content: center;
    text-align: center;
  }
  section.section-header h1 { font-size: 52px; color: #e94560; }
  section.section-header p  { color: #a8dadc; font-size: 24px; }
  h1 { font-size: 40px; color: #0f3460; margin-bottom: 20px; }
  h2 { font-size: 30px; color: #16213e; }
  h3 { color: #e94560; font-size: 24px; margin: 8px 0; }
  strong { color: #e94560; }
  code { background: #f0f4ff; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }
  pre  { background: #f0f4ff; padding: 20px; border-radius: 8px; font-size: 0.75em; }
  table { border-collapse: collapse; width: 100%; font-size: 24px; }
  th { background: #0f3460; color: #eaeaea; padding: 8px 16px; }
  td { border: 1px solid #ddd; padding: 8px 16px; }
  tr:nth-child(even) { background: #f7f9ff; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
  blockquote { border-left: 4px solid #e94560; padding-left: 16px; color: #444; font-style: italic; }
  footer { font-size: 18px; color: #888; }
---

<!-- _class: title -->

# neuronauts

## Assembling neurons at connectome scale

`github.com/wrgr/neuronauts` · branch `claude/tree-dna-phase-1-G1DNn`

---

# The problem we're solving

Connectomics gives us a voxel-segmentation of a brain volume.
Each **seg root** = one contiguous chunk of membrane.
A proofread **neuron** = many seg roots merged together.

**The goal:** given raw segmentation + synapses, automatically recover the neurons.

```
seg root 0011 ──┐
seg root 0047 ──┼──► neuron A   (pyramidal cell, layer 2/3)
seg root 0182 ──┘
seg root 0093 ──────► neuron B  (SST interneuron)
```

At the scale of Minnie65 (1 mm³ of mouse cortex):
- ~130,000 proofread neurons
- ~300 million synapses
- Neurons span **hundreds of micrometers** — far larger than any analysis box

---

# Why the existing approach hits a wall

CellGNN (the prior model) works **per 30 µm box**:

1. Build K-NN proximity graph over synapses inside the box
2. Run message passing → per-synapse embeddings
3. Cluster embeddings → neuron assignments (box-local)

<br>

| Metric | Value |
|---|---|
| Pairwise merge accuracy (within-box) | **85%+** |
| Line-graph F1 at column scale | **~0.27** |

<br>

The gap is **architectural, not capacity**. A large pyramidal neuron has an apical dendrite spanning the full cortical column (~700 µm). Box-local partitioning never sees the whole neuron — cross-box stitching errors compound at every boundary.

---

# The fix: encode identity, then assemble globally

Instead of asking *"which synapses in this box share a neuron?"*
ask *"what does this seg root's skeleton look like?"* then connect globally.

<br>

<div class="columns">
<div>

### Old approach
- Unit: synapse pair within box
- Features: raw position + grammar scores
- Scope: 30 µm box
- Failure mode: cross-box stitching

</div>
<div>

### New approach
- Unit: seg root (whole skeleton)
- Feature: learned morphology embedding (DNA)
- Scope: global (no box boundary)
- Bottleneck: within-type discrimination

</div>
</div>

<br>

> The key insight: a seg root's skeleton is a morphological **fingerprint**. Two fragments of the same neuron should have similar DNA regardless of where they sit in the volume.

---

<!-- _class: section-header -->

# The pipeline

### three stages, typed artifacts

---

# Three stages, three contracts

```
                  Region
                (synapses +
                 seg roots)
                    │
         ┌──────────┘
         │  data/fragments.py
         ▼
      Fragment[]               ← one per seg root, dna=None
         │
         │  represent/dna.py
         ▼
      Fragment[]               ← dna=[D] float32 embedding filled
         │
         │  assemble/synapse_gnn.py
         ▼
  NeuronHypothesis[]           ← set of synapse indices + confidence
         │
         ▼
   ConnectomeGraph
```

Each stage reads and writes `neuronauts/schemas.py` types. No stage depends on upstream *code* — only on the *schema*. Stages can be developed, cached, and tested independently.

---

# Stage 1 — Data

**Input:** kimimaro skeleton archive + CAVE synapse table
**Output:** `Fragment` per seg root

```python
Fragment(
    fragment_id   = root_id,       # seg-version root
    vertices_nm   = [V, 3],        # skeleton vertices in global nm
    edges         = [E, 2],        # tree topology
    radius_nm     = [V],           # neurite calibre
    endpoints_nm  = [L, 3],        # leaf vertices (seam-stitch handles)
    synapse_indices = [S],         # rows in Region owned by this root
    dna           = None,          # filled by represent/
)
```

<br>

**Key invariant:** all coordinates are **global nanometers** — no box-relative voxels ever cross this boundary.

**Contaminated roots** (false-merge survivors: one seg root maps to >1 label-version roots) are kept in the fragment list but masked out of training pairs.

---

<!-- _class: section-header -->

# Stage 2 — Tree-DNA encoder

### learning morphological identity from skeleton trees

---

# Why skeletons, not synapses

Synapses are **noisy** and **version-dependent** (the synapse table changes across proofreading versions). Skeletons are stable — kimimaro runs on the raw segmentation and produces consistent structure.

The skeleton of a seg root encodes:
- Neurite calibre and taper
- Branch topology (axon vs dendrite patterns)
- Tortuosity and curvature
- Overall tree shape

This is the **morphological fingerprint** we want to learn.

---

# The DNA encoder architecture

```
Fragment skeleton (V vertices, tree topology)
        │
        ▼  sample_tree_paths()
  K leaf-to-leaf BFS paths   [T_1, 6], [T_2, 6], ..., [T_K, 6]
  (each vertex: Δx,Δy,Δz + radius + curvature + step_len)
        │
        ▼  TorchPathEncoder (Transformer, shared weights)
  K path embeddings           [K, d_model=64]
        │
        ▼  mean pool across K paths
  one skeleton embedding      [d_model]
        │
        ▼  linear projection
  DNA                         [output_dim=32]
```

`TorchPathEncoder` is reused from the existing grammar module — no new architecture components. The DNA encoder is a **thin wrapper**: sample paths, encode each, pool.

---

# Training: triplet contrastive loss

**Positive pairs:** two fragments from the same neuron
(same `label_version` root ID in `root_label_map`)

**Negative pairs:** fragments from different neurons
(random sampling across groups)

**Loss:** `TripletMarginLoss` on L2-normalised embeddings

<br>

**The hard training task — skeleton bisection:**

Each real proofread neuron is cut at its **balance edge** (the edge whose removal produces the two most equal sub-trees). Both halves share a `label_root`. The encoder is trained to recognise them as the same neuron despite seeing only a partial skeleton.

This directly simulates the Phase 2 use case: a neuron spans multiple unproofread seg roots, each showing only a fragment of the full morphology.

---

# Phase 1 validation results

**Hard-split ablation:** 40 real Minnie65 neurons (v1412 proofread), each bisected at balance edge → 80 half-fragments, 800 synthetic synapses (uniform random → spatial baseline ≈ chance)

<br>

| Metric | AUC |
|---|---|
| Spatial proximity baseline | 0.466 (chance) |
| DNA — random init | 0.728 |
| DNA — trained (80 epochs) | **0.897** |
| Improvement: random → trained | **+0.169** |
| DNA vs spatial | **+0.431** |

<br>

Training signal: `pos_cos` 0.95 → 0.87, `neg_cos` 0.95 → 0.64 — the encoder genuinely learns to align same-neuron halves while pushing different-neuron pairs apart.

---

<!-- _class: section-header -->

# Stage 3 — Global assembly

### k-NN synapse graph + CellGNN message passing

---

# Global synapse graph

```python
GlobalSynapseGraph(
    node_feat   = [N, 32],   # DNA embedding per synapse
    node_pos    = [N, 3],    # nm coordinates
    edge_src    = [E],       # k-NN directed edges (k=8)
    edge_dst    = [E],       #   (no box boundary)
    edge_feat   = [E, 1],    # log-normalised distance
    pre_root_id = [N],       # ground truth
)
```

<br>

<div class="columns">
<div>

**What changes from box-local:**
- No 30 µm boundary
- Node features = DNA (learned), not raw position
- Graph spans the entire region

</div>
<div>

**What stays the same:**
- CellGNN message-passing architecture
- Cosine contrastive loss
- `partition_from_embeddings` clustering

</div>
</div>

---

# Phase 2 pipeline results (synthetic)

40 synthetic neurons, 400 synapses, 8-NN global graph

<br>

| Stage | AUC |
|---|---|
| Spatial baseline | 0.445 (chance) |
| DNA — random init | 0.787 |
| DNA — trained | 0.863 |
| GNN on trained DNA | **0.914** |
| GNN improvement over DNA | **+0.051** |

<br>

The GNN adds signal by aggregating neighbourhood context across the k-NN graph — synapses near same-neuron neighbours inherit their DNA signal through message passing.

Real-data Phase 2 numbers pending (the DNA training on 80 real fragments with large trees is slow; runs as `attic/prior_results/global_gnn_ablation.py --n-neurons 40`).

---

<!-- _class: section-header -->

# Honest gaps

### what we haven't validated

---

# The evaluation is a recall test, not a precision test

**What we measure:**
Can two halves of the same neuron be recognised as identical? *(merge recall)*

**What we don't measure:**
Can two different neurons of the **same type** be kept separate? *(merge precision)*

<br>

The 40 neurons in the hard-split ablation are sampled uniformly at random — almost certainly a mix of pyramidal cells, interneurons, glia, etc. The negative pairs are **cross-type**, which are trivially distinguishable.

**The hard test:** hold out 40 L2/3 pyramidal cells and run the hard-split ablation with **within-type negatives only**. If AUC stays above ~0.75 the encoder is learning individual morphological identity. If it collapses to ~0.5 we need richer features.

The MICrONS dataset has public cell type labels — this evaluation is buildable now.

---

# Other open gaps

| Gap | Status | Blocker |
|---|---|---|
| Within-type cohort evaluation | Not built | Cell type CSV lookup needed |
| Seam stitching across tiles | `endpoints_nm` field ready | Classifier not implemented |
| Global evaluation metric | Synapse-pair AUC only | Need column-scale F1 wiring |
| Overlapping-region tiling | Not designed | Dedup logic needed |
| Contamination filter at scale | Works for 40 neurons | Needs wiring into `data/` stage |
| Real synapse positions | Synthetic/uniform only | Need synapse→root assignment |
| `cell_graph.py` monolith split | 3,950 lines | Refactor planned |
| Training stability | No LR schedule, no val set | Add scheduler + early stopping |

---

<!-- _class: section-header -->

# Where we need help

### specific asks for contributors

---

# Help wanted: evaluation

**Within-type cohort ablation** *(1–2 weeks, ML + connectomics)*
- Fetch `aibs_metamorph_celltypes_v661` cell type labels from GCS
- Filter hard-split cohort to a single cell class (e.g. `"23P"` L2/3 pyramidal)
- Run ablation with within-type negatives
- Report AUC — this is the honest test of the approach

**Column-scale F1 pipeline** *(2–3 weeks, ML + infra)*
- Wire `assemble_neurons` output into `evaluate_sampled` from `line_graph.py`
- Measure per-neuron completeness (what fraction of a neuron's synapses are assigned correctly) and purity (what fraction of a cluster's synapses belong to one neuron)
- Replace synapse-pair AUC as the primary metric

---

# Help wanted: assembly

**Seam stitching** *(2–4 weeks, ML)*
- Each `Fragment.endpoints_nm` marks the leaf vertices where a fragment was cut
- Train a classifier: given two fragments from adjacent tiles, do their endpoints belong to the same neuron?
- Features: endpoint proximity, tangent alignment, DNA cosine similarity
- Output: a merge decision that extends neuron hypotheses across tile boundaries

**Overlapping-region tiling** *(1–2 weeks, infra)*
- The global pipeline currently treats each `Region` as independent
- Design a core+halo tiling scheme so fragments near seams appear in two adjacent regions
- Implement dedup so a synapse assigned in two tiles gets one canonical label

---

# Help wanted: features + scale

**Richer DNA features** *(research, 2–4 weeks)*
- Current: 6 scalar features per skeleton vertex (Δx,Δy,Δz, radius, curvature, step length)
- Ideas: multi-scale path statistics, synapse density profiles, spine/bouton counts, compartment labels (axon vs dendrite)
- Goal: distinguish same-type neurons — the current features may be type-discriminative but not individual-discriminative

**Scale testing on full Minnie65** *(infra, 2–3 weeks)*
- Current ablations: 40 neurons, 800 synapses
- Minnie65: ~130K neurons, ~300M synapses
- Need: tiled pipeline runner, distributed training, checkpoint/resume
- The code is correct at small scale; it hasn't been stress-tested at volume

**Training stability** *(ML, 1 week)*
- Current: fixed LR, no validation set, no early stopping
- Add: cosine LR schedule, hold-out split, early stopping on val AUC
- Prevents overfitting on small cohorts, makes hyperparameter search easier

---

# Getting started

```bash
git clone https://github.com/wrgr/neuronauts
cd neuronauts
uv sync                              # or: pip install -e ".[dev]"
source .venv/bin/activate

# Run the test suite (skips legacy v1 tests)
pytest -m "not legacy" -q

# Phase 1 ablation — no data required
python attic/prior_results/ablate_dna.py --synthetic

# Phase 2 ablation — no data required
python attic/prior_results/global_gnn_ablation.py --synthetic

# Real-data ablation — requires CAVE network access
python attic/prior_results/half_split_ablation.py --n-neurons 40
```

<br>

Read `docs/architecture.md` for a full design walkthrough.
Stage ownership and open tasks: `docs/stage_ownership.md`.

---

<!-- _class: title -->

# The ask

The pipeline from skeleton → DNA → global graph → neuron is working at small scale.

**We need people who want to own a stage.**

- `data/` — synapse + skeleton ingestion at volume
- `represent/` — richer features, better training, within-type validation
- `assemble/` — seam stitching, tiling, scale testing
- `evaluate/` — honest metrics at column scale

Each stage has a typed contract. You can work independently once the interface is agreed on.

`docs/architecture.md` · `CONTRIBUTING.md` · `docs/stage_ownership.md`
