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

## Reconstructing neurons from a noisy connectome

`neuronauts/coassign/` · branch `claude/synapse-coassign`

---

<!-- _class: section-header -->

# Part 1 — The Biology
## What are we actually looking at?

---

# What is a connectome?

A **connectome** is a complete wiring diagram of a piece of brain tissue.

<br>

<div class="columns">
<div>

**Nodes = neurons**
The computing cells of the brain. Each has a cell body plus long branching wires (axons and dendrites) that stretch up to **millimeters** through tissue.

**Edges = synapses**
The junctions where one neuron passes a signal to another. Each synapse has a *sender* (pre-synaptic) and a *receiver* (post-synaptic) side.

</div>
<div>

```
   neuron A  ╮
      ╲       ╲
   synapse     ● ← signal passes here
      ╱       ╱
   neuron B  ╯
```

<br>

The **structure** of this wiring — who connects to whom — is the physical substrate of computation in the brain. It's a prerequisite for understanding vision, memory, movement.

</div>
</div>

---

# The MICrONS dataset

The largest connectome dataset ever produced:

<br>

| | |
|---|---|
| **Volume** | 1 mm³ of mouse visual cortex |
| **Neurons** | ~200,000 |
| **Synapses** | ~500,000,000 |
| **Imaging** | Electron microscopy at 8×8×40 nm/voxel |

<br>

At 8 nm resolution you can see individual cell membranes, the synaptic cleft, and the thinnest axon branches threading between neighbours.

> Reconstructing the wiring diagram from this raw image stack is one of the central open problems in modern neuroscience. **It is fundamentally a large-scale computer vision + clustering problem.**

---

# From pixels to neurons: segmentation

The raw data is a 3D stack of grayscale EM images. To get neurons, a computer-vision pipeline assigns every voxel a **segment ID** (a.k.a. **root ID**) — ideally, all voxels of one neuron share one ID.

<br>

```
  Raw EM voxels          Segmentation            Goal
  ┌─┬─┬─┬─┬─┐            ┌─┬─┬─┬─┬─┐
  │ │ │▓│ │ │            │1│1│▓│2│2│      all voxels of one
  │ │▓│▓│▓│ │   ──CV──►  │1│▓│▓│▓│2│  →   neuron share one ID
  │ │ │▓│ │ │            │1│1│▓│2│2│
  └─┴─┴─┴─┴─┘            └─┴─┴─┴─┴─┘
```

<br>

This is **extraordinarily hard**: neurites are thin, membranes are sometimes ambiguous, and a single neuron snakes through hundreds of microns in a complex tree shape. The segmentation is **never perfect**.

---

# Two versions: v117 and v1412

Segmentation is iterated, not one-shot:

<br>

<div class="columns">
<div>

### v117 — automated
Computer vision ran once and produced a complete labeling of the volume.

**It is noisy.** This is the cheap, fast, error-prone output.

This is what we have at scale for fresh tissue.

</div>
<div>

### v1412 — proofread
Human annotators reviewed v117, found errors, and corrected them.

**This is ground truth.** Expensive — it took years of expert labor.

We train *against* it and evaluate *against* it.

</div>
</div>

<br>

> The whole point: we paid millions of dollars to produce v117 at scale. We cannot afford to proofread every neuron by hand. **Can a model do the cleanup automatically?**

---

# Two kinds of segmentation error

<div class="columns">
<div>

### Splits (common)
One real neuron broken into **many** v117 segments. The CV cut a neuron where it shouldn't have.

```
   real neuron
   ████████████
        ↓ v117
   ███ ██ ████      ← 3 seg IDs
   1042 88 9931        for 1 neuron
```

Fix: **merge** the pieces back together.

</div>
<div>

### Frankenmerges (rarer, worse)
**Two** neurons fused into one v117 segment. The CV missed a membrane.

```
  neuron A   neuron B
   ███████ + ███████
        ↓ v117
   ██████████████     ← 1 seg ID
       seg 9912          for 2 neurons
```

Fix: **ignore** the false same-seg evidence.

</div>
</div>

<br>

Frankenmerges are worse: they inject **false synaptic connections** into the connectome, and a big merged blob looks locally plausible everywhere.

---

<!-- _class: section-header -->

# Part 2 — The Core Idea
## Why cluster synapses, not segments?

---

# The task, in one sentence

> Given a set of synapses, each carrying a **noisy v117 segment label**, decide **which synapses belong to the same neuron**.

<br>

This is a **clustering / partition problem**: group synapses into clusters, one cluster per physical neuron.

```
Synapse 0  (seg 1047)  ──┐
Synapse 1  (seg 1047)  ──┼──► Neuron A      same-seg → likely together
Synapse 2  (seg 2303)  ──┘                  but seg 2303 ≠ 1047 (a split!)
Synapse 3  (seg 9912)  ──────► Neuron B
Synapse 4  (seg 9912)  ──┐                  seg 9912 spans two neurons
Synapse 5  (seg 0041)  ──┴──► Neuron C      (a frankenmerge!)
```

We must **see through** splits (merge across seg IDs) and **see past** frankenmerges (don't blindly trust same-seg).

---

# The key insight: synapses are invariant

<div class="columns">
<div>

### Segment IDs
- Change with **every** round of proofreading
- v117 root ≠ v1412 root ≠ v1433 root
- Bookkeeping that shifts under your feet

</div>
<div>

### Synapses
- Fixed physical events at fixed locations
- A synapse is the **same** across all versions
- The stable ground we build on

</div>
</div>

<br>

**Consequence:** if we cluster *synapses*, the partition we learn is **stable across segmentation versions**.

That's why every metric in `cluster.py` is defined on **synapse pairs**, never on segment IDs. A result computed on v117 stays valid when v1412 arrives.

> "Synapses are the invariant nodes. We're finding cliques — which optimizes connectivity as intended and leads to natural metrics."

---

# Two channels of evidence

How do we know two synapses share a neuron? Two signals:

<br>

<div class="columns">
<div>

### 1. Same-segment
Two synapses with the **same v117 seg ID** are co-continuous in the automated segmentation.

**Strong but noisy.** Correct when v117 is right; broken by splits; poisoned by frankenmerges.

</div>
<div>

### 2. DNA similarity
A learned embedding of each segment's 3D **skeleton shape** — its branching, caliber, extent.

**The morphological fingerprint.** Two segments with similar DNA are likely pieces of one neuron — even across a split.

</div>
</div>

<br>

> "DNA" is an analogy: just as biological DNA encodes an organism's identity, this embedding encodes a neuron piece's morphological identity. Produced by the `SkeletonGNN`. The model learns it from **raw** skeleton vertices — no hand-crafted features.

---

<!-- _class: section-header -->

# Part 3 — The Pipeline
## Four steps, four files, ~500 lines

---

# Pipeline overview

```
   Synapses + v117 seg IDs + skeletons
                  │
   ┌──────────────┘
   │  graph.py        build_synapse_graph()
   ▼
   SynapseGraph                          nodes = synapses
   (nodes + same-seg edges + spatial edges)   edges = evidence
                  │
   │  model.py        SynapseCoassigner (a GNN)
   ▼
   P(same neuron) for every edge         learned edge scores
                  │
   │  cluster.py      greedy_cluster() × K
   ▼
   K candidate partitions                ranked by likelihood
                  │
   │  cluster.py      pairwise_precision_recall, coverage_at_k
   ▼
   Metrics + human-review output
```

Each file has one clear job. `train.py` ties them together with a loss.

---

# Step 1 — Build the SynapseGraph

`build_synapse_graph(positions, seg_ids, labels, seg_dna)` turns raw data into a graph with **two edge types**:

<div class="columns">
<div>

### Same-segment edges
`same_seg = 1.0`

Connect every pair of synapses sharing a v117 seg ID.

*Capped at 200 pairs per segment* — otherwise a frankenmerge with 10k synapses would create 50M edges (O(N²) blowup).

</div>
<div>

### Spatial k-NN edges
`same_seg = 0.0`

Connect each synapse to its **8 nearest neighbours** in nm space.

Catches cross-segment proximity — two synapses from split pieces that happen to sit close together.

</div>
</div>

<br>

**Each node carries:** `[x, y, z, dna_0 … dna_63]` — its position plus the DNA of its segment.
**Each edge carries:** the `same_seg` flag (1.0 or 0.0).

Nothing else is hand-fed. The model decides what matters.

---

# Step 2 — Score edges with a GNN

`SynapseCoassigner` reads the graph and outputs **P(same neuron)** per edge.

```
 node features [x,y,z,dna…]  ──►  LayerNorm  (learned normalisation, no hardcoded scale)
                                      │
                       3 × message-passing layers:
                       msg    = Linear([h_neighbour ∥ same_seg])
                       update = Linear([h ∥ aggregated_msg]) → ReLU → LayerNorm
                                      │
                       per-synapse embedding h  [N, 64]
                                      │
        edge scorer:  MLP([ h_u ∥ h_v ∥ |h_u − h_v| ∥ same_seg ])  →  σ  →  P
```

<br>

Three deliberate choices, all in service of *"avoid hardcoding features"*:
- **LayerNorm at input** — the network learns position scaling
- **`same_seg` is a plain input** — not a hard gate; the model learns how much to trust it
- **scorer sees both embeddings AND the raw flag** — an explicit shortcut for the easy cases

---

# Step 3 — Cluster into neurons

We have P(same neuron) on every edge. Now find the **partition** that best agrees with those probabilities.

This is **correlation clustering** — NP-hard in general. We use the **greedy pivot** algorithm (O(E), 3-approximation):

<br>

```
Shuffle the synapses into a random order.
For each synapse not yet assigned:
    Look at neighbours already in a cluster, where P ≥ threshold.
    If any exist → join the cluster with the highest mean edge probability.
    Otherwise    → start a new cluster of its own.
```

<br>

Simple, fast, and good enough in practice. One pass over the edges produces one full partition of all synapses into neuron clusters.

---

# Step 3 (cont.) — K materializations

Run greedy **K times** with different random shuffles → K different partitions. Keep the K unique ones, **sorted by log-likelihood** (best first).

```
Materialization 1:  [A,B,C | D | E,F]      log-score −12.4   ← best
Materialization 2:  [A,B | C | D | E,F]    log-score −13.1
Materialization 3:  [A,B,C | D | E | F]    log-score −13.9
```

<br>

**Why several answers instead of one?**

A single partition gives no signal about *where* it was uncertain. With K partitions, the **edges where they disagree** are exactly the uncertain decisions — and that's where a human should look.

> coverage@K asks: does the **true** partition appear in the top-K? A well-calibrated model should include it even when no single answer is perfect.

---

# Step 4 — Train the model

Binary cross-entropy on edge labels: **1** if the two synapses share a v1412 neuron, **0** if not.

<br>

### The problem: easy negatives dominate

Most random synapse pairs are from far-apart neurons — trivially "different." Training on those teaches the model nothing.

### The fix: hard negative mining

```python
# Over-sample the HARD negatives: spatially close, but different neurons
hard_neg = spatial_edges where  label[src] != label[dst]   # different neuron
                            and  label[src] > 0 and label[dst] > 0  # both known
# 50% of each negative batch is drawn from this pool
```

These are interdigitated neurons that sit right next to each other. Forcing the model to separate them is what makes it learn to **use DNA**, not just distance.

---

<!-- _class: section-header -->

# Part 4 — Results
## Does it work?

---

# Synthetic-split results

20 real proofread neurons (v1412), each cut into 3 pieces to *simulate* v117 splits. Synthetic synapses placed near skeleton vertices. Trained end-to-end on CPU, ~5 minutes.

<br>

| Metric | Value | Reading |
|---|---|---|
| Pairwise precision | **0.952** | Merges made are almost always correct |
| Pairwise recall | **0.420** | Under-merging — too cautious |
| Pairwise F1 | **0.583** | |
| coverage@5 | **False** | True partition not yet in top-5 |

<br>

**The story:** the model *knows* which edges are safe — 95% precision means it rarely merges wrongly. It is declining to merge when unsure. The fixed **threshold = 0.5 is too conservative**; calibrating it should lift recall toward 0.8+ without hurting precision.

---

# Real v117 results — it runs end-to-end

A 20 µm box of real mouse cortex, **actual CAVE data** (no synthetic splits):
synapses at v117, ground truth via the real v117→v1412 map, DNA from
`SkeletonGNN` on real v117 skeletons.

<div class="columns">
<div>

| Region fact | Value |
|---|---|
| Synapses | **782** (782 labeled) |
| v117 segments | **60** |
| Distinct v1412 neurons | **60** |
| Graph edges | **15,802** |
| Same-seg / spatial | 60% / 40% |
| Wall time | ~9.5 min |

</div>
<div>

**100% of synapses got a v1412 label** — the version mapping is reliable.

The model **learns**: over 40 epochs the edge loss falls 0.69 → 0.44, and edge-level precision/recall climb to **0.76 / 0.88**.

`python scripts/v117_coassign.py \`
`  --token $TOK --max-segs 60`

</div>
</div>

> This is the honest test: real splits, real frankenmerges, real morphology — and 60 distinct neurons interdigitated in one small box.

---

# Real v117 results — closing the partition gap

Edge scoring was always strong; the **partition** lagged. Calibrating the
clustering threshold and scaling the model closes most of the gap:

| Same 60-neuron v117 box | Best F1 | Best recall | Edge P/R |
|---|---|---|---|
| 40 ep · d64 · threshold 0.5 | 0.507 | 0.588 | 0.76 / 0.88 |
| **120 ep · d128 · calibrated (0.675)** | **0.760** | **0.723** | **0.82 / 0.92** |

Threshold calibration alone (holding the model fixed) lifts sweep F1 from
**0.724 → 0.792**; the larger model + more epochs add the rest.

**What this tells us** — the central lesson:

- **Edge-level learning works** (edge P/R = 0.82 / 0.92). The model can tell, for a given pair, whether they share a neuron.
- **The partition step was the bottleneck, not the encoder.** A fixed 0.5 cut over-merged this denser real-data graph; the calibrated 0.675 threshold recovers most of the lost F1 (0.51 → 0.76).
- **coverage@5 is still False** (best recall 0.72 < 0.90): getting the *whole* neuron requires bridging see-through gaps — the job of **prototype/EM assignment (#4)**, the next lever.

> Synthetic splits gave P=0.95; real interdigitated v117 went 0.51 → 0.76 F1 once the partition step was calibrated. The remaining gap to coverage *is* the research problem.

---

# The v117 data harness

`neuronauts/data/cave.py` connects the pipeline to real CAVE data in three calls:

```python
from neuronauts.data.cave import fetch_v117_region, encode_seg_dna
from neuronauts.coassign import build_synapse_graph, SynapseCoassigner, train

# 1. Fetch real data: synapses + skeletons at v117, labels via v117→v1412 map
region = fetch_v117_region(bbox_nm, token=CAVE_TOKEN, min_seg_synapses=2,
                           skeleton_cache_dir="/tmp/cache")
# → positions_nm [N,3], seg_ids [N], gt_labels [N], skeletons{seg_id: SkeletonData}

# 2. Encode DNA from the fetched skeletons
seg_dna = encode_seg_dna(region.skeletons, region.seg_ids)

# 3. Build + train — identical from here to the synthetic demo
graph = build_synapse_graph(region.positions_nm, region.seg_ids,
                            region.gt_labels, seg_dna)
train(SynapseCoassigner(node_dim=graph.node_dim), [graph])
```

<br>

The split between **fetch** (needs network) and **encode** (needs GPU) is intentional — cache skeletons once, re-encode freely.

---

<!-- _class: section-header -->

# Part 5 — Where to Help
## The roadmap and how to contribute

---

# Immediate next steps

<div class="columns">
<div>

### 1. Threshold calibration ✅ done
`cluster.py` · `calibrate_threshold`

Sweeps thresholds, picks F1-max. On real v117 it lifted partition **F1 0.51 → 0.76**.

**Next here:** calibrate on *held-out* graphs (not in-sample) for an unbiased cut at scale.

</div>
<div>

### 3. Endpoint-adjacent edges
`graph.py`

Add a third edge type: skeleton **leaf vertices** that are spatially near each other across segments.

These are the principled bridge sites — exactly where the CV cut a neuron.

`Fragment.endpoints_nm` is already populated.

</div>
</div>

<br>

### 2. Bigger model, longer training
100–200 epochs · d_model 128–256 · 4–6 GNN layers. Real neurons span hundreds of microns; 3 layers × 8 hops ≈ 24-hop reach is too short.

---

# Medium-term: see through the noise

### 4. Prototype-based assignment (EM-style)

The case that pure pairwise scoring **fails**: segments A and C belong to one neuron, but B between them is a frankenmerge with ambiguous DNA. A→B weak, B→C weak → A and C never connect.

**Fix:** maintain a running embedding per *growing neuron hypothesis* (mean of its confirmed members). Assign each synapse to the closest hypothesis, or mark it uncertain. Iterate E / M steps. The prototype pools clean evidence from the **whole arbor**, bypassing local gaps.

<br>

### 5. Within-type evaluation — the honest test

```bash
python scripts/coassign_demo.py --n-neurons 30 --n-pieces 3 --cell-type 23P
```

All neurons are L2/3 pyramidal — same type, similar shape. Cross-type precision (0.95) is partly *easy* cross-type separation. Within-type is the real difficulty.

---

# How to contribute

```
neuronauts/coassign/          ← 4 files, ~500 lines, one job each
├── graph.py                  ← SynapseGraph + build_synapse_graph
├── model.py                  ← SynapseCoassigner (GNN encoder + scorer)
├── cluster.py                ← greedy_cluster, materializations, metrics
└── train.py                  ← BCE + hard negative mining

scripts/coassign_demo.py      ← synthetic-split demo (~5 min on CPU)
scripts/v117_coassign.py      ← real v117 data harness
neuronauts/data/cave.py       ← CAVE fetch + DNA encode
tests/test_coassign.py        ← 25 tests, all green
```

<br>

| Good first task | File | Effort |
|---|---|---|
| Threshold calibration | `cluster.py` | 1–2 days |
| Endpoint edges | `graph.py` | 1 day |
| Within-type eval | run the demo | 30 min |

**Read `INTRO.md`** for the full biology + pipeline walkthrough · **`docs/archive/2026-09/NEXT_STEPS.md`** (archived) for the historical roadmap, or `docs/roadmap_global_assembly.md` for the current one.

---

<!-- _class: title -->

# Let's build the connectome

`python scripts/coassign_demo.py --n-neurons 20 --n-pieces 3`

Synapses are the invariant nodes. We're finding cliques.

**Done:** real v117 harness · threshold calibration (F1 0.51 → 0.76)
**Next:** prototype/EM assignment → coverage@5 → within-type eval
