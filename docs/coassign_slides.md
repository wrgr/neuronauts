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

# Synapse Co-Assignment

## Partitioning neurons at connectome scale

`neuronauts/coassign/` · branch `claude/synapse-coassign`

---

# One central insight

Synapses are **physical facts**.
Segment IDs are **bookkeeping**.

<br>

| | Segment IDs | Synapses |
|---|---|---|
| Stable across versions? | ❌ Change with every proofread | ✅ Fixed locations in tissue |
| Unit of measurement? | Version-dependent | Version-independent |
| Ground truth? | v1412 ≠ v117 ≠ v1433 | Same physical event |

<br>

> "We've spent millions of dollars getting v117. Local is mostly not helpful — the segment evidence is the primary signal."

**The task:** cluster synapses into neuron cliques using v117 segments as *evidence*, not as *nodes*.

---

# The problem, precisely

Given N synapses with noisy v117 segment labels, find a **partition** where every synapse in a cluster belongs to the same physical neuron.

<br>

```
Synapse 0  (seg 1047)  ──┐
Synapse 1  (seg 1047)  ──┼──► Neuron A   ← same-seg: strong evidence
Synapse 2  (seg 2303)  ──┘                ← spatial proximity: DNA similar
Synapse 3  (seg 9912)  ──────► Neuron B
Synapse 4  (seg 9912)  ──┐
Synapse 5  (seg 0041)  ──┴──► Neuron C   ← seg 9912 is a frankenmerge!
```

<br>

Two kinds of errors we must survive:

- **Splits** — one neuron, many v117 segments → model must merge them
- **Frankenmerges** — many neurons, one v117 segment → model must ignore same-seg evidence selectively

---

<!-- _class: section-header -->

# Architecture
## SynapseGraph → GNN → Correlation Clustering

---

# Step 1: Build the SynapseGraph

Two edge types, one graph:

<br>

<div class="columns">
<div>

### Same-segment edges
`same_seg = 1.0`

Two synapses sharing a v117 root ID are co-continuous in the automated segmentation.

**Strong but noisy**: correct when v117 is right, broken by splits, poisoned by frankenmerges.

Cap: 200 directed pairs per segment — prevents O(N²) blowup from large frankenmerges.

</div>
<div>

### Spatial k-NN edges
`same_seg = 0.0`

Each synapse connected to its 8 nearest neighbours in nm space.

**Weak but unbiased**: catches cross-segment proximity where a split puts two nearby synapses in different segments.

</div>
</div>

<br>

**Nothing hardcoded.** Position scale and the relative weight of same-seg vs. spatial edges are learned from data.

---

# Step 2: SynapseCoassigner

A GNN that reads the SynapseGraph and outputs **P(same neuron)** per edge.

<br>

```
Input per node: [pos_x, pos_y, pos_z, dna_0, …, dna_63]  ← LayerNorm (learned)
                                                             ↓
                             3 × MessagePassing layers
                     msg = Linear([h_src ∥ same_seg])     ← same_seg is a learned feature
                     update = Linear([h ∥ agg]) → ReLU → LayerNorm
                                                             ↓
               Edge scorer: MLP([h_u ∥ h_v ∥ |h_u−h_v| ∥ same_seg])
                                                             ↓
                                              σ → P(same neuron)
```

<br>

Key design choices:
- `LayerNorm` at input — learned normalisation, no hardcoded position scale
- `same_seg` is a plain input feature — the model learns its weight, not a hard gate
- Scorer sees **both** node embeddings **and** the raw same-seg flag — explicit shortcut for easy cases

---

# Step 3: Correlation Clustering

Given P(same neuron) per edge, find the **best partition** of synapses into neuron clusters.

This is correlation clustering — NP-hard in general.
We use the **greedy pivot algorithm** (O(E), 3-approximation):

<br>

```
Shuffle nodes randomly.
For each unassigned node:
  Look at already-assigned neighbours with P >= threshold.
  If any: join the cluster with the highest mean edge probability.
  Else: start a new cluster.
```

<br>

**K materializations**: run greedy K times with different random orderings.
Each run produces a different (but nearby) partition.
Sort by log-likelihood → return top-K unique results.

<br>

> The true partition should appear in the top-K — **coverage@K** measures this.

---

# Step 4: Training

Binary cross-entropy on edge labels: **1** if both synapses share the same v1412 neuron, **0** otherwise.

<br>

### Hard negative mining

The hard cases are spatially close synapses that belong to *different* interdigitated neurons.
A random negative is usually far away and trivially easy.

```python
# Hard negatives: spatial edges (same_seg=0) that cross neuron boundaries
hard_neg_pool = spatial_edges_where(
    true_label[src] != true_label[dst]   # different neuron
    and true_label[src] > 0              # both labeled
    and true_label[dst] > 0
)
# 50% of the negative sample is drawn from hard_neg_pool
```

<br>

This forces the model to learn what makes two nearby-but-different neurons distinguishable — primarily their DNA.

---

<!-- _class: section-header -->

# Results
## 20 real proofread neurons, 3 pieces each, 60 epochs

---

# Real-data results

Neurons fetched from CAVE (v1412), split into 3 pieces each to simulate v117 splits, trained end-to-end on CPU in ~5 minutes.

<br>

| Metric | Value | Notes |
|---|---|---|
| Pairwise precision | **0.952** | Co-assignments made are almost always correct |
| Pairwise recall | **0.420** | Under-merging — threshold too conservative |
| Pairwise F1 | **0.583** | |
| coverage@5 | **False** | True partition not yet in top-5 |

<br>

**The precision/recall story:**

The model *knows* which edges are confident. It is declining to merge when it's unsure.
The fixed threshold=0.5 is too conservative — a **calibration pass** on held-out data would move recall from 0.42 toward 0.8+ without hurting precision.

---

# The v117 data harness

`neuronauts/data/cave.py` connects the pipeline to real CAVE data:

<br>

```python
from neuronauts.data.cave import fetch_v117_region, encode_seg_dna
from neuronauts.coassign import build_synapse_graph, SynapseCoassigner, train

# 1. Fetch from CAVE
region = fetch_v117_region(
    bbox_nm,              # spatial bounding box in global nm
    token=CAVE_TOKEN,
    min_seg_synapses=2,   # drop tiny segments
    skeleton_cache_dir="/tmp/cache",
)
# → positions_nm [N,3], seg_ids [N], gt_labels [N], skeletons dict

# 2. Encode DNA
seg_dna = encode_seg_dna(region.skeletons, region.seg_ids)

# 3. Build graph and train — identical to the demo
graph = build_synapse_graph(region.positions_nm, region.seg_ids, region.gt_labels, seg_dna)
model = SynapseCoassigner(node_dim=graph.node_dim)
train(model, [graph])
```

<br>

Run end-to-end: `python scripts/v117_coassign.py --token YOUR_TOKEN --cache-dir /tmp/v117_cache`

---

# Why K materializations?

A single partition can be wrong in subtle ways with no signal about *where* the uncertainty is.

<br>

### The workflow

K candidate partitions, ranked by log-likelihood. A proofreader sees the **edges where materializations disagree** — exactly the uncertain decisions.

```
Materialization 1:  [A,B,C | D | E,F]      ← merges synapses 2+3
Materialization 2:  [A,B | C | D | E,F]    ← keeps synapse 3 separate
Materialization 3:  [A,B,C | D | E | F]    ← merges 2+3, splits 5+6

Disagreement on synapse 3: review this merge decision.
Disagreement on synapses 5+6: review this split decision.
```

<br>

**coverage@K** = "is the true partition in the top-K?"

Target: coverage@5 ≥ 0.9 after threshold calibration.

---

<!-- _class: section-header -->

# Next Steps
## Where to contribute

---

# Immediate (this branch)

<div class="columns">
<div>

### 1. Threshold calibration
`cluster.py`

The model learns good edge scores but threshold=0.5 is too conservative.

Add `calibrate_threshold(model, graphs)`:
sweep threshold on held-out graphs, pick the F1-maximizing value.

**Expected impact: recall 0.42 → 0.7+**

</div>
<div>

### 3. Endpoint-adjacent edges
`graph.py`

Add a third edge type: skeleton endpoints (leaf vertices) that are spatially near each other across different segments.

These are the principled bridge sites — where a real neuron was cut by the automated segmentation.

```python
Fragment.endpoints_nm  # already populated
```

</div>
</div>

<br>

### 2. More training / larger model
100-200 epochs, d_model=128-256, 4-6 GNN layers. Neurons span hundreds of microns; 3 layers × 8 k-NN hops = ~24 hops of effective range.

---

# Medium term

### 4. Prototype-based assignment (EM-style)

The hardest case: segment A and C belong to the same neuron, but B between them is a frankenmerge. A→B is weak, B→C is weak — A and C never connect.

Fix: maintain a **running embedding per growing neuron hypothesis** (mean of its confirmed members). Assign unassigned synapses to the closest hypothesis — or mark uncertain. Iterate E/M steps until convergence. The prototype aggregates clean evidence from across the full arbor, bypassing local gaps.

<br>

### 5. Within-type evaluation

```bash
python scripts/coassign_demo.py --n-neurons 30 --n-pieces 3 --cell-type 23P
```

All neurons L2/3 pyramidal — same cell type, similar morphology. Current cross-type precision (0.95) is partly easy cross-type separation. **Within-type is the honest test.**

---

# How to contribute

```
neuronauts/coassign/          ← 4 files, ~500 lines
├── graph.py                  ← SynapseGraph + build_synapse_graph
├── model.py                  ← SynapseCoassigner (GNN encoder + scorer)
├── cluster.py                ← greedy_cluster, materializations, metrics
└── train.py                  ← BCE + hard negative mining

tests/test_coassign.py        ← 25 tests
scripts/coassign_demo.py      ← end-to-end demo (~5 min on CPU)
scripts/v117_coassign.py      ← real v117 data harness
neuronauts/data/cave.py       ← CAVE fetch + DNA encode
```

<br>

**Good first contributions:**

| Task | File | Effort |
|---|---|---|
| Threshold calibration | `cluster.py` | 1–2 days |
| Endpoint edges | `graph.py` | 1 day |
| Within-type eval | run the demo | 30 min |

See `INTRO.md` for the full biology + pipeline walkthrough.
See `NEXT_STEPS.md` for the full roadmap.

---

<!-- _class: title -->

# Let's build the connectome

`python scripts/coassign_demo.py --n-neurons 20 --n-pieces 3`

P=0.952 · R=0.420 · F1=0.583 after 60 epochs

**Next:** threshold calibration → coverage@5 → real v117 evaluation
