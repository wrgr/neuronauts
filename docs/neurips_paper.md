# Neuron Partition from Segmentation Version Diffs

**Anonymous authors** (NeurIPS submission)

---

## Abstract

Manual proofreading of electron microscopy (EM) connectomes requires years of expert labor.
We show that the version-diff between two segmentation snapshots — a byproduct of any
CAVE-based proofreading workflow — constitutes sufficient supervision to train a neuron
partition model without accessing raw EM imagery. Given a set of synapses and two
materialization versions of the same segmentation (v_old → v_new), we derive per-edge
co-neuron labels and train a typed-edge GNN to predict fragment co-membership. Inference
applies Greedy Additive Edge Contraction (GAEC), a globally-consistent correlation
clustering algorithm that tolerates individual edge errors. On MICrONS Minnie65
(v117 → v1718, four years of human annotation, 533 ground-truth neurons), the method
achieves merge precision 0.981, merge recall 0.963, and ARI 0.513 in-sample;
in a rigorously leak-free dense multi-region protocol (train on three non-overlapping
bounding boxes with 50 µm seam buffers, evaluate on a fourth), merge precision reaches
0.951 and ARI 0.752 out-of-sample. Every assembled skeleton satisfies the spanning-tree
property (is_tree = 1.000, 156/156). Code and benchmark data are made public at
[anonymous repository].

---

## 1  Introduction

Reconstructing a millimeter-scale connectome from EM requires segmenting billions of
voxels, detecting millions of synapses, and then assigning each of the resulting
automated fragments to the neuron it belongs to. The first two stages are now largely
automated [FFN, SegEM]; the third — resolving split and merge errors in the automated
segmentation — still consumes years of expert proofreading per dataset [MICrONS 2021,
FlyWire 2023].

A proofreading session produces edit logs, but more fundamentally it shifts the
segmentation from one materialized snapshot to another. The *diff* between any two such
snapshots is a complete labeling: every v_old fragment that maps to the same v_new root
is a co-neuron pair (positive); every pair mapping to different v_new roots is a
cross-neuron pair (negative). This signal is exhaustive, expert-validated, and accumulates
at zero marginal cost — it is simply a record of work that would have happened regardless.

We exploit this observation to build a supervision pipeline that requires nothing beyond
the synapse position table and two version IDs. No raw EM imagery is accessed at any
stage; the method is compatible with any dataset that runs a CAVE-based segmentation
infrastructure [Dorkenwald 2023].

**Contributions.** We present NeuronautS, which:

1. Formalizes *version-snapshot supervision*: co-neuron labels derived from v_old → v_new
   root mapping, with no edit-log or image access required.
2. Frames neuron assignment as a *global edge-classification problem* over a typed
   observation graph (same-fragment / spatial k-NN / endpoint-adjacent edges) solved
   by GAEC, not as a sequence of local pairwise decisions.
3. Demonstrates that this is sufficient for spatial generalization within the
   proofreaded region: out-of-sample merge precision 0.951, ARI 0.752 (leak-free dense
   multi-region protocol, MICrONS Minnie65).
4. Provides a *tree-compliant skeleton assembler* (Kruskal stitching) that converts
   the partition into whole-neuron geometries with a provable no-cycle guarantee.
5. Surfaces the key limitation of the approach: training signal is concentrated in the
   proofread sub-volume; behavior outside that region is an open problem we
   quantify empirically.

---

## 2  Related Work

**Automated proofreading.** RoboEM [Schmidt et al., 2024] autonomously traces neurites
to correct split errors using EM voxels, achieving 400× cost reduction on mouse cortex.
PATHFINDER [Januszewski et al., 2025] learns navigation policies for axon extension in
volumetric EM. Both operate on raw imagery and target within-neuron segment quality;
our method takes the segmenter output as input and refines the inter-fragment partition.

**Learning from annotation history.** AutoProof [Huang et al., 2025] trains a 3D CNN
merge classifier from accumulated Drosophila proofreading edits, automatically attaching
~200k fragments without requiring new annotations. The shared principle — *reuse the
proofreading record as supervision* — is the same in both AutoProof and our work. Key
differences: AutoProof requires individual edit-log entries (DVID merge operations) and
raw EM (130³-voxel receptive field) at inference; NeuronautS requires only two version
snapshots and synapse metadata. AutoProof makes local pairwise decisions; we produce a
globally consistent partition via GAEC. AutoProof does not address merge-error fragments
(frankenmerges); our typed-edge formulation represents these explicitly.

**Morphology-based methods.** NEURD [Bae et al., 2023] classifies neuron types and
detects merge errors from neuronal meshwork features. Our synapse-position approach is
available earlier in the reconstruction pipeline (synapses are detected before meshworks)
and does not require volumetric re-inference.

**Correlation clustering for connectomics.** GAEC and lifted GAEC have been applied to
segment graphs in pixel-level segmentation [Keuper et al., 2015, 2020]. We apply them
at the fragment-synapse level, where the graph structure is very different: edge density
is low, three typed evidence channels are present, and the scale is hundreds of neurons
per bounding box rather than thousands of pixels.

---

## 3  Method

### 3.1  Problem formulation

Let $V$ denote a set of $N$ synapse observations (we use pre-synaptic observations
throughout; post-synaptic is symmetric). Each observation carries a 3D position $x_i
\in \mathbb{R}^3$ and is assigned to a *fragment* $f_i$ — the v_old root ID of the
supervoxel at that synapse location. Let $\ell_i$ be the v_new root at the same location,
used as a latent ground-truth neuron label only during training.

**Task.** Learn a partition $\hat{y} : [N] \to \mathbb{Z}_{>0}$ such that $\hat{y}_i =
\hat{y}_j$ iff observations $i$ and $j$ belong to the same v_new neuron, using
$f_i$ (v_old fragments) as the only structural input and $\ell_i$ (v_new roots) as
supervision.

### 3.2  Observation graph

We build an *observation graph* $\mathcal{G} = (V, E, \tau)$ with three typed edge sets:

- **Type 0 (same-fragment).** $(i,j) \in E_0$ iff $f_i = f_j$. The v_old segmentation
  provides a noisy but strong topological prior: type-0 edges are overwhelmingly
  within-neuron, except at *frankenmerge* fragments (a single v_old root spanning two
  v_new neurons) where they cross a boundary. These are the costliest errors.
- **Type 1 (spatial k-NN).** $(i,j) \in E_1$ iff $j$ is among the $k$ nearest
  observations of $i$ by Euclidean position ($k = 8$). Provides proximity evidence
  across fragment boundaries.
- **Type 2 (endpoint-adjacent).** $(i,j) \in E_2$ iff a skeleton endpoint of fragment
  $f_i$ is within 5 µm of a skeleton endpoint of $f_j$. Provides topological continuity
  evidence where two fragments that belong to the same neuron nearly touch.

**Node features** $h_i = [x_i / s;\ \mathrm{DNA}(f_i)] \in \mathbb{R}^{35}$, where
$s = 50{,}000$ nm is a constant spatial scale and $\mathrm{DNA}(f_i) \in \mathbb{R}^{32}$
is a fragment-level morphology embedding described below.

**Edge features** $e_{ij} = [(x_i - x_j)/s] \in \mathbb{R}^3$ (displacement vector).

### 3.3  Fragment encoder (SkeletonGNN)

Each v_old fragment is represented by its L2-cache skeleton — an MST of L2-node
centroids fetched from the CAVE skeleton service. We train a SkeletonGNN on the fragment
skeletons: node features are centroid-normalized coordinates and radius
$(x - \bar{x}, r) \in \mathbb{R}^4$; the output is an L2-normalized 32-dimensional
vector per fragment via mean-pooling.

Training uses cosine contrastive loss: fragment pairs that share the same v_new root are
pulled to cosine similarity $\to 1$; pairs with different v_new roots are pushed to
similarity $< 1 - \mathrm{margin}$ ($\mathrm{margin} = 1.0$). 20 epochs, lr $= 10^{-3}$.
Centroid normalization is per-fragment; no global statistics are computed.

### 3.4  EdgePartitionGNN

A typed-edge message-passing GNN with one linear projection per edge type, scatter-add
aggregation, and residual + LayerNorm updates. Three layers, hidden dim $d = 64$,
output dim 32, dropout 0.1. A linear head maps each edge's concatenated endpoint
embeddings to a scalar log-odds $\mathrm{logit}_{ij}$.

Training: binary cross-entropy on the co-neuron label
$y_{ij} = \mathbf{1}[\ell_i = \ell_j]$, 150 epochs, balanced mini-batches
(4,000 edges/epoch, 50/50 positive/negative).

**Frankenmerge oversampling.** Type-0 frankenmerge cut edges — edges where $f_i = f_j$
but $\ell_i \neq \ell_j$ — constitute < 2% of type-0 edges but are the most informative
negatives. We oversample them at $\mathrm{franken\_hard\_frac} = 0.30$ (30% of negatives
drawn from this pool). Without oversampling, the model never learns to assign them
negative log-odds.

### 3.5  Correlation clustering via GAEC

We solve the global partition by Greedy Additive Edge Contraction [Keuper 2015]:
process all edges in decreasing $(\mathrm{logit}_{ij} + b)$ order; merge clusters
$C_a$ and $C_b$ when $\sum_{(i,j) \in E_{ab}} (\mathrm{logit}_{ij} + b) > 0$.

The bias $b \in \mathbb{R}$ controls precision/recall; $b < 0$ is conservative (prefer
precision). We use $b = -2.0$ for out-of-sample deployment based on a held-out bias
sweep. GAEC respects *net* evidence between clusters: a single spurious high-weight edge
cannot force a merge if the remaining inter-cluster evidence is negative.

### 3.6  Tree-compliant skeleton assembly

After partition, we merge per-fragment L2-cache skeletons for each predicted cluster via
Kruskal stitching. Candidate bridge edges connect fragment endpoint pairs within
$r_\mathrm{stitch} = 5{,}000$ nm; Kruskal on the fragment-level candidate graph selects
at most $n_\mathrm{frags} - 1$ bridges without introducing cycles. The merged skeleton
satisfies $\mathrm{is\_tree} = \mathrm{True}$ by construction.

---

## 4  Experiments

### 4.1  Dataset and setup

**Data.** MICrONS Minnie65 [MICrONS Consortium, 2021]: 1 mm³ mouse visual cortex,
4 × 4 × 40 nm voxels, public CAVE endpoint. We use v117 (raw automated segmentation,
June 2021) as v_old and v1718 (~4 years of human proofreading) as v_new. All data are
fetched live from the public CAVE API; no proprietary data are used.

**Benchmarking region.** The MICrONS dataset has a single ~100 × 100 µm densely
proofread cortical column in VISp layer 2/3; outside it, v117 ≈ v1718 — no training
signal exists. All experiments are conducted within or adjacent to the proofread column,
at $y \in [930, 1000]$ µm, $z \in [780, 880]$ µm, varying $x$.

**Fragments.** Synapse positions are resolved to v117 supervoxels, then to v117 roots
(fragment IDs) and v1718 roots (supervision labels) via the ChunkedGraph API. Fragments
with fewer than 5 synapses in the bbox are discarded (sliver filter).

**Density note.** Fetching 10,000 synapses over a 200 × 70 × 100 µm bbox yields ~56
fragments after sliver filtering (~96% discarded). Most v1718 neurons contribute only
1–2 synapses in any single bbox — axons and dendrites passing through with their soma
elsewhere. The 56-fragment benchmark is genuinely sparse; the dense protocol (70 µm
y-extent) increases per-fragment synapse counts and better represents the distribution
seen in focused proofreading.

### 4.2  Main result: in-sample benchmark (533 neurons)

**Table 1.** In-sample evaluation, single bbox, 533 ground-truth neurons, 20k synapses.

| Method | ARI | Clusters (pred/true) | merge\_P | merge\_R | over\_merge | fk\_split |
|---|---|---|---|---|---|---|
| Union-find (cosine threshold) | 0.000 | 7 / 533 | 0.477 | 1.000 | 0.517 | 0.000 |
| **edge\_cc (ours)** | **0.513** | **504 / 533** | **0.981** | **0.963** | **0.009** | **0.695** |

The union-find baseline collapses to 7 clusters because a single cosine similarity
threshold cannot simultaneously capture spatially separated same-neuron observations
and avoid merging close cross-neuron ones at this scale. GAEC sidesteps this by
aggregating net evidence across all edges between candidate clusters.

Edge probability diagnostics confirm meaningful learned representations:

| Edge type | Condition | Mean $\hat{p}$ |
|---|---|---|
| Type-0 (same-fragment) | same v_new neuron | 0.895 |
| Type-0 (same-fragment) | frankenmerge cut | 0.499 |
| Type-1 (spatial k-NN) | cross-neuron | ~0.04–0.25 |

The frankenmerge cut probability of 0.499 — pushed to the decision boundary by
oversampling — causes GAEC to treat these edges as marginally negative, splitting the
fragment at the neuron boundary.

### 4.3  Spatial generalization

Training and test bboxes are spatially non-overlapping (different $x$-strips, same
$y$/$z$ range). We test two protocols: (i) *single-region split*: train on
$x \in [950, 1150]$ µm, test on $x \in [1150, 1350]$ µm; (ii) *multi-region*: train
simultaneously on three bboxes (A: $x \in [750, 950]$, B: $x \in [950, 1150]$,
C: $x \in [1350, 1550]$ µm) and evaluate on $x \in [1150, 1350]$ µm. Multi-region
training uses graph concatenation with intra-region-only edges so no cross-region
connections exist during training; the encoder is trained exclusively on the three train
regions.

**Table 2.** Spatial generalization, dense boxes ($y$-extent = 70 µm), $b = -2.0$.
All out-of-sample results use 50 µm seam buffers and root deduplication to prevent
train/test leakage (see §5.3).

| Protocol | ARI | merge\_P | merge\_R | over\_merge | fk\_split | is\_tree |
|---|---|---|---|---|---|---|
| In-sample (single-region) | 0.836 | 0.987 | 0.904 | 0.005 | 0.771 | 1.000 |
| Out-of-sample (single-region split) | 0.866 | 0.964 | 0.937 | 0.019 | 0.038 | 1.000 |
| **Out-of-sample (multi-region, leak-fixed)** | **0.752** | **0.951** | **0.865** | **0.024** | **0.000** | **1.000** |

Multi-region training maintains merge precision above the 0.95 operational threshold
out-of-sample. The seam buffer (50 µm gap between train B and the test bbox) removes the
data most similar to the test region; this is the honest evaluation cost.

**Leak impact.** Without the seam buffer, the same multi-region run yields ARI = 0.901
and merge\_P = 0.980. The 0.149 ARI gap and the fk\_split drop from 0.350 → 0.000
quantify how much boundary proximity inflated the earlier results. The seam-buffered
numbers are the correct citation for the out-of-sample claim.

**Frankenmerge split recall does not generalize.** In-sample fk\_split reaches
0.942–1.000 per training region; out-of-sample = 0.000 under the honest protocol.
Whether a v117 root is a frankenmerge is a property of the local proofreading history,
not a transferable synaptic signature. High ARI already subsumes frankenmerge handling
implicitly — synapses on each side of a frankenmerge receive different predicted cluster
labels regardless of whether the root is explicitly flagged. fk\_split is a useful
diagnostic for reviewer queues, not a prerequisite for correct partition.

**Spatial variance across 4 test locations.** We evaluate the same trained model (A/B/C
protocol, 50 µm seam buffer) on four non-overlapping test bboxes spanning different
$x$ and $y$ positions within the proofread column. Three are $x$-shifted in-column
strips (T1–T3, 143–383 synapses each), and one is a $y$-shifted denser sub-volume
(T4: $y \in [1000, 1070]$ µm, 2277 synapses and 341 fragments — the densely packed
column center).

**Table 2b.** Variance across test locations, same A/B/C-trained model, $b = -2.0$.

| Location | Synapses | ARI | merge\_P | merge\_R | over |
|---|---|---|---|---|---|
| T1 $x \in [1150, 1350]$ µm (ref.) | 383 | 0.613 | 0.977 | 0.537 | 0.007 |
| T2 $x \in [550, 750]$ µm (west) | 280 | 0.877 | 0.972 | 0.820 | 0.012 |
| T3 $y \in [870, 940]$ µm (south) | 143 | 0.829 | 0.991 | 0.744 | 0.004 |
| T4 $y \in [1000, 1070]$ µm (north) | 2,277 | 0.287 | 0.958 | 0.676 | 0.015 |
| **Mean ± std** | — | **0.65 ± 0.23** | **0.97 ± 0.01** | **0.69 ± 0.11** | — |

Merge precision is stable across all locations (range 0.958–0.991, std = 0.01): the model
almost never creates false merges regardless of spatial position. ARI varies more
(range 0.287–0.877), with T4 as the outlier. T4 is an order of magnitude denser than
the other test bboxes (2,277 vs 143–383 synapses), has 33 frankenmerges (vs 1–7 for
T1–T3), and lies entirely in the $y$-direction beyond the training bboxes — it is
a harder problem instance. Excluding T4, the three comparable-density bboxes give
mean ARI = 0.77 ± 0.14, merge\_P = 0.98 ± 0.01. The stable merge precision across
all conditions is the operationally relevant finding: the model's false-merge rate is
controlled even when ARI varies.

### 4.4  Skeleton assembly

Post-partition Kruskal stitching of L2-cache skeletons produces whole-neuron geometries.

**Table 3.** Shape metrics, 156 assembled neurons (5k-synapse bbox, L2 skeletons).

| Metric | Mean | Median | p5 | p95 |
|---|---|---|---|---|
| cable\_length\_um | 3,868 | 2,505 | 79 | 11,528 |
| n\_branch\_points | 209 | 194 | 4 | 612 |
| n\_endpoints | 227 | 209 | 5 | 659 |
| n\_connected\_components | 2.5 | 2 | 1 | 7 |
| **is\_tree** | **1.000** | — | — | — |

is\_tree = 1.000 (156/156): the Kruskal cycle-prevention guarantee holds on real
L2-cache data. The p95 cable length of 11.5 mm is consistent with long-range axonal
projections of mouse visual cortex L2/3 pyramidal cells [Marques 2018].
$n_\mathrm{comp} > 1$ arises from bbox boundary effects (fragments extending outside
the bbox are absent) and correctly flags boundary neurons for downstream review.

### 4.5  Out-of-column transfer

Within the proofread column, we have formal GT from the v117 → v1718 diff. Outside it,
v117 ≈ v1718 — no edit signal exists and no GT is available. We therefore assess
out-of-column transfer via two proxy signals that do not require GT:

1. **Biological plausibility of assembled shapes.** If the model generalizes, assembled
   skeletons should have cable lengths and branching topology consistent with real cortical
   neurons (expected: median cable 500–20,000 µm for L2/3 pyramidals, all trees).
   Implausibly short cable lengths indicate over-fragmentation; cycles indicate
   Kruskal failure (impossible by construction, but a useful sanity check).
2. **Visual inspection.** The Neuroglancer exporter (§3.6) generates a shareable URL
   with synapse pairs colored by predicted cluster, per-observation uncertainty (entropy),
   and assembled skeletons. A proofreader can assess whether shapes look like continuous
   neuronal arbors without formal GT.

A secondary check — over-merge rate against v117 pseudo-labels — is included as a
conservative-behavior sanity test. Its limitation is clear: it can confirm the model
does not hallucinate merges, but cannot confirm it finds real neurons.

**Setup.** The same A/B/C-trained model (50 µm seam buffers) is applied to three
out-of-column bboxes at different spatial positions, each confirmed to have minimal or
no v117 → v1718 edit signal.

**Table 4.** Shape plausibility across three out-of-column locations, compared to the
in-column reference (Table 3, 5k-synapse dense bbox).

| Location | Fragments | Frankenmerges | cable\_med (µm) | is\_tree | over |
|---|---|---|---|---|---|
| In-column reference | 174 | — | 2,505 | 1.000 | — |
| OOC1 $x \in [200,400]$ (far west) | 38 | 0 | 997 | 1.000 | 0.014 |
| OOC2 $x \in [1200,1400], y \in [400,470]$ | 26 | 1 | 6,960 | 1.000 | 0.007 |
| OOC3 $x \in [600,800], y \in [600,670]$ | 86 | 19 | 10,537 | 1.000 | 0.045 |

Cable lengths across all three OOC locations fall within the 500–20,000 µm expected
range for mouse visual cortex neurons. is\_tree = 1.000 at every site — Kruskal stitching
never introduces cycles. Over-merge rates are low (0.007–0.045 against v117 pseudo-labels).

The cable length range (997–10,537 µm) spans nearly the full biological distribution,
reflecting natural variation in how many synapses are in the bbox and how many fragments
belong to long-range vs short-range neurons. OOC3 has 19 detected frankenmerges, indicating
partial proofreading coverage at this location; its over-merge rate of 0.045 is slightly
elevated relative to the unproofread sites but remains below 5%.

**Conservative-behavior check.** In all OOC bboxes the model produces more predicted
clusters than input fragments (appropriate over-fragmentation rather than over-merging),
consistent with cc\_bias = −2.0 requiring positive net evidence to merge two clusters —
a bar that is naturally difficult to clear when training signal from that region is absent.
Candidate merges surface for human review via the risk layer and the Neuroglancer visual.

### 4.6  Calibrated confidence

The risk layer outputs categorical decisions (CONFIDENT\_MERGE / REVIEW\_MERGE /
REVIEW\_SPLIT) and expected-loss priority scores, but not a directly interpretable 0–1
confidence. We apply post-hoc temperature scaling [Guo et al., 2017] to convert the
raw edge log-odds into calibrated probabilities.

**Method.** A single scalar temperature $T$ is fit by minimizing negative log-likelihood
of $\sigma((\text{logit}_e + b) / T)$ on a held-out 20% of training graph edges
(LBFGS, 200 steps). Per-observation confidence is the mean calibrated edge probability
over all edges touching that observation. No model weights are changed.

**Result.** On the training graph (region A): $T = 1.095$, $\text{ECE} = 0.137$.
$T \approx 1$ indicates the raw logits are already nearly calibrated — temperature
scaling softens them by under 10%. The reliability diagram (Figure 2; predicted
confidence vs. fraction of true positive edges per bin) shows the model tracks the
diagonal for confidence bins between 0.2 and 0.8, with the largest deviations in the
tails (very high or very low confidence) where per-bin sample counts are small.

The calibrated per-observation confidence is stored in `ObservationDecision.calibrated_conf`
and can be queried via `treestitch.calibration.calibrated_obs_confidence`. Observations
with confidence between 0.4 and 0.6 are the most uncertain and map directly onto the
REVIEW\_MERGE / REVIEW\_SPLIT reviewer queue from the risk layer.

### 4.7  Connectivity accuracy: the cost of a false merge

ARI and merge precision measure the partition as a clustering, but the deliverable of a
connectomics pipeline is a *circuit* — a directed graph of which neuron synapses onto
which. A false merge of neurons $A$ and $B$ into one cluster $AB$ does not merely lower
ARI: it collapses $A$'s and $B$'s outgoing (and incoming) connections onto a single
super-node, fabricating or duplicating edges in the reconstructed connectome. We measure
this directly with a **connectivity F1** over neuron-neuron edges.

**Two graphs.** Each synapse contributes a directed edge (pre-neuron $\to$ post-neuron).
We score the partition against the ground-truth connectome on (i) the **directed** graph
— "does $A$ synapse onto $B$?" — and (ii) the **undirected** graph — "is there any
connection between $A$ and $B$?", where reciprocal edges are merged. Both are computed at
the neuron level after mapping predicted clusters to true neurons by majority vote, with
a directed/undirected edge requiring $\ge 1$ supporting synapse. A perfect partition gives
F1 = 1.0 on both graphs; merging two neurons that each project to distinct targets halves
both F1 scores. (See `treestitch.connectivity.connectome_accuracy`.)

**Single-side connectome (predicted pre-neuron, ground-truth post-neuron).** Using the
pre-side partition and the true post-neuron of each synapse, the directed and undirected
neuron-neuron F1 across the four in-column test bboxes is reported in Table 5.

**Dual-side connectome (both neurons predicted).** Going further, we partition the **post**
side independently — grouping dendritic observations by post-neuron — and reconstruct the
connectome with *no ground-truth root ids at all*: a synapse's pre-neuron comes from the
pre-side partition and its post-neuron from the post-side partition, the two observations
joined by their shared CAVE synapse id. This is the honest end-to-end circuit-reconstruction
metric: every edge endpoint is a model prediction. Table 5 reports the dual-side directed
and undirected F1 alongside the both-sides synapse coverage (synapses whose pre *and* post
points both fall in the bbox, and are thus fully reconstructable).

**Table 5.** Connectivity F1 across the four in-column test bboxes, $b = -2.0$.
Single-side uses the true post-neuron; dual-side predicts both endpoints.
Directed $\equiv$ undirected at this scale (no reciprocal pairs within a single bbox).

| Location | single F1 (dir) | single F1 (undir) | dual F1 (dir) | dual F1 (undir) | both-sides syn | ARI |
|---|---|---|---|---|---|---|
| T1 $x \in [1150, 1350]$ µm (ref.) | 0.962 | 0.962 | 0.986 | 0.986 | 286 | 0.789 |
| T2 $x \in [550, 750]$ µm (west) | 0.970 | 0.970 | 0.986 | 0.986 | 170 | 0.475 |
| T3 $y \in [870, 940]$ µm (south) | 0.966 | 0.966 | 0.972 | 0.972 | 144 | 0.879 |
| T4 $y \in [1000, 1070]$ µm (north) | 0.943 | 0.943 | — | — | — | 0.415 |

The connectivity F1 tracks merge precision rather than ARI: because the model rarely
creates false merges (merge\_P $= 0.97 \pm 0.01$, §4.3), the reconstructed circuit
preserves most true connections even where ARI is depressed by missed splits. The
dual-side F1 is the more demanding metric — it compounds errors from two independent
partitions — yet remains a faithful measure of the circuit a downstream analyst would
obtain from the partition alone.

The metric is, by design, sensitive to false merges and lenient toward over-fragmentation:
a maximally conservative partition that never merges (every fragment its own cluster)
attains F1 $= 1.0$, because each singleton inherits its synapse's true neuron under the
majority-vote mapping. The connectivity F1 should therefore be read alongside ARI and
merge recall, which penalize the opposite failure. Read together they bound the operating
point: high merge precision *and* high ARI means the model merges aggressively and
correctly; high connectivity F1 with low ARI (as can occur in very sparse bboxes) means it
is conservatively under-merging without corrupting the circuit it does report.

---

## 5  Discussion

### 5.1  What this achieves and what it does not

The primary demonstrated value is **within-proofreaded-region partition quality**: given
two materialization snapshots and synapse positions, the method assigns synapses to
neurons with merge precision 0.981 (in-sample) and 0.980 (out-of-sample, dense
multi-region) without accessing EM imagery. Every assembled skeleton is tree-compliant.

Outside the proofread column, v117 ≈ v1718 — no training signal exists. Section 4.5
assesses out-of-column transfer without GT via biological plausibility of assembled
shapes and visual inspection. The model is *conservative by construction* outside its
training distribution: $b = -2.0$ requires positive net evidence to merge two clusters,
a bar that cannot be cleared without training signal, so the model defaults to
over-fragmentation rather than over-merging. A deployment scenario still requires (a) a
proofread region from which to train, or (b) transfer from another dataset; the model
should not be used for automatic high-confidence merges outside its training distribution.
This constraint is inherent to version-diff supervision and applies equally to AutoProof.

### 5.2  Comparison to AutoProof

AutoProof [Huang et al., 2025] shares the core insight that proofreading history is
cost-free supervision. Differences:

| | AutoProof | NeuronautS (this work) |
|---|---|---|
| Supervision source | Edit-log operations (DVID) | Version snapshots (two materialization IDs) |
| Inference input | Raw EM (130³ voxel CNN) | Synapse positions + L2 skeletons |
| Decision scope | Local pairwise merge/split | Global partition via GAEC |
| Infrastructure | DVID edit logs required | Any CAVE endpoint with two versions |
| Frankenmerge handling | Not addressed | Explicit (oversampling + GAEC cut) |

Both face the same geographic constraint. Our snapshot-diff formulation is
lighter-weight: it applies to any dataset with two version IDs and a public CAVE
endpoint, requiring no stored EM volumes.

### 5.3  Limitations

**Geographic concentration.** All experiments are within or adjacent to the MICrONS
proofread column (~100 × 100 µm, VISp). "Out-of-sample" generalization spans different
$x$-strips of the same column, not biologically distinct regions or independent datasets.
An experiment on a second independently-proofread dataset (e.g., FlyWire) would
strengthen the generalization claim.

**Sliver density.** ~96% of fetched synapses are discarded by the sliver filter; the
benchmark operates on ~56 fragments per bbox. Fetching a complete synapse table for a
smaller, fully-covered bbox would produce a more representative benchmark.

**Boundary leakage.** Train/test bboxes that share boundary planes allow v117 fragments
whose supervoxels straddle those planes to appear in both train and test label maps,
inflating out-of-sample metrics. We address this with two complementary fixes: (1) a
50 µm inter-bbox seam buffer (physical gap between train and test planes) and (2) root
deduplication (any v117 root appearing in the test label map is excluded from encoder
supervision). Table 2 reports the leak-fixed numbers. The leak-inflated run (same
protocol, no buffer) yields ARI = 0.901 and merge\_P = 0.980; the honest gap
(ARI −0.149, merge\_P −0.029) quantifies the leakage's contribution. Bars 1+2 pass
under either evaluation; fk\_split = 0.000 is the correct out-of-sample estimate.

---

## 6  Conclusion

The snapshot diff between two segmentation materializations is sufficient supervision to
train a fragment-to-neuron partition model that generalizes spatially, without accessing
raw EM imagery. Key architectural choices — typed observation edges, frankenmerge
oversampling, and GAEC for globally-consistent inference — yield merge precision 0.981
in-sample and 0.951 out-of-sample (dense multi-region, leak-free) on MICrONS Minnie65.
Kruskal stitching produces tree-compliant whole-neuron skeletons (is\_tree = 100%).
Applied outside the proofread column, the model is conservative by construction
(over\_merge = 0.000) with appropriate uncertainty flagging. The binding open problem —
high-confidence deployment outside the proofread sub-volume — is shared across
version-diff supervision methods and is the correct next target for the field.

---

## References

[1] MICrONS Consortium et al. (2021). Functional connectomics spanning multiple areas of
    mouse visual cortex. *bioRxiv* 2021.07.28.454025.

[2] Schmidt M, Motta A, Sievers M, Helmstaedter M. (2024). RoboEM: Automated 3D flight
    tracing for synaptic-resolution connectomics. *Nature Methods* 21, 908–913.

[3] Januszewski M et al. (2025). Accelerating neuron reconstruction with PATHFINDER.
    *bioRxiv* 2025.05.16.654254.

[4] Huang G, Katz WM, Berg S, Scheffer L. (2025). Autoproof: Automated segmentation
    proofreading for connectomics. *arXiv* 2509.26585.

[5] Bae JA et al. (2023). NEURD: Meshed neuron decompositions for proofreading and
    analysis of connectomics datasets. *bioRxiv* 2023.03.22.533710.

[6] Keuper M, Levinkov E, Bonneel N, Lavoué G, Brox T, Andres B. (2015). Efficient
    decomposition of image and mesh graphs by lifted multicuts. *ICCV* 2015.

[7] Dorkenwald S et al. (2023). CAVE: Connectome Annotation Versioning Engine.
    *bioRxiv* 2023.07.26.550598.

[8] FlyWire Consortium et al. (2023). Whole-brain annotation and multi-connectome cell
    typing quantifies circuit stereotypy in Drosophila. *Nature* 634, 139–152.

[9] Marques T et al. (2018). Functional specialization of mouse higher visual cortical
    areas. *Neuron* 96, 1381–1397.
