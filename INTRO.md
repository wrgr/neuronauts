# Introduction to Neuronauts

A guide for engineers and ML practitioners with no neuroscience background.

---

## 1. The Biology

**What is a connectome?** A connectome is a complete map of every neuron and every connection in a piece of brain tissue. Think of it as a wiring diagram: nodes are neurons, edges are synapses (the points where one neuron communicates with another). Neuroscientists care about connectomes because the structure of this wiring — which cells connect to which, and how strongly — is the physical substrate of computation in the brain. Knowing the wiring diagram is a prerequisite to understanding how circuits for vision, memory, or movement actually work.

**Neurons and synapses.** Neurons are the computing units of the brain. Each neuron has a cell body and long branching extensions (axons and dendrites) that can stretch millimeters through tissue. Synapses are the directed connections between neurons: each synapse has a **pre-synaptic** side (the sending neuron) and a **post-synaptic** side (the receiving neuron). When a neuron "fires," it releases neurotransmitter at its pre-synaptic terminals; the downstream neurons receive that signal at their post-synaptic sites. In the data, every synapse record contains a 3D position, a pre-synaptic segment ID, and a post-synaptic segment ID.

**The MICrONS dataset.** This project works with the MICrONS dataset: a 1 mm³ volume of mouse visual cortex, containing roughly 200,000 neurons and approximately 500 million synapses, imaged at nanometer resolution using electron microscopy. The imaging resolution is 8×8×40 nm per voxel — fine enough to resolve individual cell membranes, synapse clefts, and the thinnest axon branches as they thread between neighboring cells. No other publicly available dataset at this scale existed before MICrONS. Accurately reconstructing the wiring diagram from this data is one of the central open problems in modern neuroscience.

---

## 2. The Data Pipeline

**Raw images.** The raw data is a 3D stack of grayscale electron microscope images. Each voxel is 8×8×40 nm and records image intensity — roughly, bright pixels are inside cells and dark pixels are the membranes between them.

**Segmentation: turning pixels into neuron IDs.** Computer vision algorithms sweep over this volume and assign every voxel a **segment ID** (also called a **root ID**). The goal: all voxels belonging to the same neuron share one ID. Think of it as a label image where each "color" is a neuron. This is extraordinarily hard in practice — neuronal branches are thin, membranes are sometimes ambiguous, and neurons snake through the volume in complex tree-shaped paths that span hundreds of microns.

**Two versions: v117 and v1412.** The segmentation is not a one-shot process; it is iterated:
- **v117** is the initial automated segmentation. Computer vision ran, assigned IDs, produced a complete labeling. It is noisy.
- **v1412** is the expert-proofread version. Human annotators reviewed the automated output, found errors, and corrected them. This is the ground truth we train against and evaluate against.

**Two types of errors.** The automated v117 segmentation makes two kinds of mistakes:
- **Splits** (the most common error): one real neuron is broken into multiple v117 segments. The algorithm saw an ambiguous region and wrongly cut the neuron in two. In the data, one v1412 root ID maps to many v117 segment IDs.
- **Merges / "frankenmerges"** (rarer but worse): two different neurons are fused into a single v117 segment. The algorithm missed a membrane boundary and welded two cells together. Frankenmerges are especially harmful because they introduce false synaptic connectivity into the final wiring diagram.

**Synapse detection.** Synapses are detected by a separate computer vision pipeline that identifies the characteristic dense spots at junctions between neural processes. Each detected synapse records its 3D position in nm, and the pre- and post-synaptic segment IDs at the segmentation version current at detection time.

**CAVE (Connectome Annotation Versioning Engine).** CAVE is the database backing all of this. It tracks how root IDs evolve as proofreading progresses, maintains the synapse table, skeleton cache, and nucleus table, and serves everything via a versioned REST API. When you query "what is the root ID of this point at materialization v1412?", you are asking CAVE. The code in `neuronauts/data/loaders.py` handles CAVE interactions — fetching the nucleus table, downloading skeletons, and sampling neurons.

---

## 3. The Core Task: Synapse Co-Assignment

**The problem, precisely.** Given a set of synapses, each carrying a noisy v117 segment label, determine which synapses belong to the same neuron. This is a **partition / clustering problem**: group synapses into clusters where each cluster corresponds to one physical neuron. Formally, we want a function from synapse IDs to neuron IDs.

**Why not just trust segment IDs?** Because of splits. A single neuron that was cut into three v117 pieces has three distinct v117 root IDs. If you treat each v117 segment as its own neuron, you report three neurons where there is one — and you miss all the synaptic connections that cross the cut points. The task is to look past the noisy segment labels and recover true neuron identity.

**Why synapses are the right nodes (not segments).** Synapses are invariant physical facts: a synapse sits at a fixed location in the tissue and does not change as the segmentation improves. Segment IDs, by contrast, change with every round of proofreading — a v117 root ID means something different from a v1412 root ID. By clustering synapses (not segments), the partition we learn is stable across segmentation versions. This is why the evaluation metrics in `cluster.py` are defined on synapse pairs, not segment IDs.

**The evidence.** Two types of signals tell us whether two synapses belong to the same neuron:
1. **Same-segment edges**: two synapses sharing a v117 segment ID are co-continuous in the automated segmentation. Strong but noisy evidence: correct when v117 is right, broken by splits, corrupted by frankenmerges.
2. **DNA similarity**: the `SkeletonGNN` (in `neuronauts/represent/skeleton_gnn.py`) produces a learned embedding of each segment's 3D skeleton shape — its branching geometry, caliber profile, and spatial extent. Neurons have characteristic morphologies: a pyramidal cell looks different from an interneuron. Two segments with similar DNA embeddings are likely pieces of the same neuron. The term "DNA" here is a deliberate analogy — just as biological DNA encodes the identity of an organism, this embedding encodes the morphological identity of a neuron piece.

---

## 4. The Pipeline and File Structure

The core co-assignment pipeline lives in `neuronauts/coassign/`:

**`graph.py` — Build the SynapseGraph.**
`build_synapse_graph` takes raw synapse positions (in nm), their v117 segment IDs, ground-truth neuron labels (from v1412, used during training), and a dictionary mapping each segment ID to its DNA embedding. It builds a `SynapseGraph` dataclass with two types of edges:
- **Same-segment edges** (`same_seg=1.0`): pairs of synapses that share a v117 segment ID. These are the "locally co-continuous" edges.
- **Spatial k-NN edges** (`same_seg=0.0`): each synapse connected to its 8 nearest spatial neighbors. This captures cross-segment proximity — two synapses from split pieces of the same neuron that happen to sit close together in space.

To prevent O(N²) blowup from large frankenmerge segments, same-segment edges are capped at 200 directed pairs per segment.

**`model.py` — The SynapseCoassigner GNN.**
A graph neural network that reads the `SynapseGraph` and outputs `P(same neuron)` for each edge. Input features per synapse node: 3D position concatenated with DNA embedding vector. The network runs L message-passing rounds (default L=3, hidden dim=64), then an edge-scoring MLP computes a probability from the pair of node embeddings plus the `same_seg` flag. Nothing is hardcoded: position scales and the relative weight given to same-segment vs. spatial edges are all learned from data.

**`cluster.py` — Partition the graph.**
Given edge probabilities from the model, find a partition of synapses into neuron clusters. This is **correlation clustering**: maximize total log-likelihood of the partition under the edge probability model. Because correlation clustering is NP-hard in general, `cluster.py` uses the greedy pivot algorithm (O(E), 3-approximation). Running greedy K times with different random node orderings produces K distinct candidate partitions — see section 5. Metrics (`pairwise_precision_recall`, `coverage_at_k`) are defined on synapse pairs and are therefore stable across segmentation versions.

**`train.py` — Train the model.**
Binary cross-entropy loss on edge labels (1 if both synapses share the same v1412 neuron, 0 otherwise). To handle class imbalance (most synapse pairs are from different neurons), the loop uses **hard negative mining**: it over-samples spatial edges that cross neuron boundaries, since spatially close synapses from two different interdigitated neurons are the hardest cases.

**Supporting modules:**

`neuronauts/data/loaders.py` fetches real data from MICrONS. `load_nucleus_table()` downloads the list of proofread v1412 root IDs (~4.7 MB public GCS file). `load_skeleton()` fetches a skeleton (a tree of 3D points tracing a neuron's shape) from the CAVE skeleton cache for a given root ID. `sample_neurons()` returns a random sample of proofread neurons, optionally filtered by cell type (e.g., `"E"` for excitatory, `"I"` for inhibitory).

`neuronauts/represent/skeleton_gnn.py` contains the **SkeletonGNN** — the model that maps a segment's skeleton graph to a DNA embedding vector. Input: raw skeleton vertices (x, y, z, radius) and edge lengths. The GNN runs message-passing over the skeleton graph and produces a single `[output_dim]`-dimensional embedding via mean+max pooling, followed by L2 normalization. Training is contrastive: embeddings for skeleton pieces from the same neuron are pulled together (high cosine similarity); embeddings from different neurons are pushed apart past a margin.

---

## 5. Why Multiple Materializations?

A single partition can be wrong in subtle ways — a few merges here, a few splits there — and there is no obvious way to know which decisions were uncertain. Instead of outputting one answer, the pipeline produces **K ranked candidate partitions**, sorted by how well each fits the model's edge probabilities.

The practical workflow: a human proofreader does not inspect the entire result. They look at the edges where the top K materializations **disagree** with each other. Those disagreements are exactly the uncertain decisions — the places where additional evidence (e.g., checking the EM image directly) would resolve the ambiguity. Concentrating review effort on disagreements makes proofreading tractable at scale.

**coverage@K** is the primary quality metric: does any of the K candidate partitions achieve pairwise recall >= 0.9 against the ground-truth labels? A well-calibrated model with K=5 should achieve high coverage even when no single materialization is perfect. Improving coverage@K is the main training objective.

---

## 6. Quick Start

Run the end-to-end demo using real MICrONS skeletons. The demo fetches proofread neurons, splits each skeleton into N pieces (simulating v117 segmentation splits), assigns synthetic synapses near skeleton vertices, trains SkeletonGNN + SynapseCoassigner end-to-end, then reports pairwise precision, recall, and coverage@K:

```bash
# 20 neurons, each split into 3 pieces — takes ~5 min on CPU
python scripts/coassign_demo.py --n-neurons 20 --n-pieces 3

# Harder: all neurons are the same cell type (23P = layer 2/3 pyramidal)
python scripts/coassign_demo.py --n-neurons 30 --n-pieces 3 --cell-type 23P
```

To use the API directly from Python:

```python
from neuronauts.coassign import (
    build_synapse_graph, SynapseCoassigner,
    train, materializations, coverage_at_k,
)
import numpy as np

# positions: [N,3] nm; seg_ids: [N] int64 v117 labels;
# labels: [N] int64 v1412 neuron IDs (0=unknown); seg_dna: dict seg_id→[D]
graph = build_synapse_graph(positions, seg_ids, labels, seg_dna)

model = SynapseCoassigner(node_dim=graph.node_dim)
train(model, [graph], n_epochs=60)

import torch
node_feat  = torch.from_numpy(np.concatenate([graph.node_pos, graph.node_dna], 1)).float()
edge_src_t = torch.from_numpy(graph.edge_src).long()
edge_dst_t = torch.from_numpy(graph.edge_dst).long()
same_seg_t = torch.from_numpy(graph.same_seg).float()
probs = model.edge_probs(node_feat, edge_src_t, edge_dst_t, same_seg_t).numpy()

mats = materializations(graph.n_nodes, graph.edge_src, graph.edge_dst, probs, K=5)
print("coverage@5:", coverage_at_k(mats, graph.labels))
```

---

## 7. Key Terms Glossary

**segment / seg_id / root_id**: The integer ID assigned to a neuron (or a piece of a neuron) by the segmentation pipeline. In this codebase, `seg_id` refers to the v117 noisy label; `root_id` often refers to the v1412 ground-truth label. Always check which `seg_version` or `label_version` a given array was generated at — the same physical location gets different IDs at different versions.

**v117 / v1412 (materialization versions)**: Snapshot version numbers of the CAVE segmentation. v117 is the initial automated computer vision output (noisy). v1412 is the expert-proofread version (ground truth). Higher version numbers are later, more-correct snapshots. CAVE tracks how root IDs map across versions via its lineage APIs.

**pre-synaptic / post-synaptic**: The two sides of a synapse. Pre-synaptic = the sending (upstream) neuron. Post-synaptic = the receiving (downstream) neuron. In the data, `pre_root_id` is the neuron releasing neurotransmitter; `post_root_id` is the neuron receiving it. The synapse graph built by `graph.py` uses pre-synaptic positions and seg IDs as the nodes to cluster.

**DNA (in this codebase)**: Not biological DNA. A learned morphological embedding of a segment's skeleton, produced by the `SkeletonGNN`. Just as biological DNA encodes an organism's identity, the DNA embedding encodes a neuron piece's shape. Two skeleton fragments from the same neuron should have similar (high cosine similarity) DNA; two fragments from different neurons should have dissimilar DNA. Stored in the `Fragment.dna` field (see `neuronauts/schemas.py`).

**frankenmerge**: A segmentation error where two or more distinct neurons are incorrectly fused into a single segment. The resulting chimeric object has processes from multiple real cells — hence "frankenmerge." Frankenmerges are worse than splits because they inject false synaptic connections into the final connectome, and they are harder to detect because a large merged segment looks locally plausible everywhere.

**correlation clustering**: The optimization problem of partitioning a graph into clusters to maximize total edge log-likelihood. Given P(same neuron) for every edge, find the assignment of nodes to clusters that best explains those probabilities. NP-hard in general; the code uses the greedy pivot approximation from `cluster.py`, which is O(E) and achieves a 3-approximation.

**coverage@K**: The metric that asks: does any of the K candidate partitions recover at least 90% of the true co-assignment pairs? Formally, coverage@K is True if `max(recall over top-K materializations) >= 0.9`. This is the primary training target for the probabilistic pipeline: producing K outputs where the correct answer is almost always in the set, even when no single output is perfect.
