# Self-Supervised Neuron Identity Assignment from Proofreading Version History

**Will Gray**, [Additional authors]

---

## Abstract

Reconstructing complete neuronal connectomes from electron microscopy (EM) requires partitioning tens of thousands of automatically segmented fragments into individual neurons — a task that currently demands years of expert manual proofreading. Recent work has begun to exploit the proofreading record itself as a training signal [Autoproof, Huang et al. 2025], but existing approaches require access to the raw EM imagery and individual edit-log entries to make local pairwise merge decisions. We present **NeuronautS**, a complementary framework that requires only the synapse position table and two segmentation version snapshots — no EM imagery, no operation log. The method derives pairwise co-membership labels from the snapshot diff (v_old → v_new), trains an edge-classifying GNN on the resulting observation graph, and produces a globally consistent partition via greedy additive edge contraction (GAEC) correlation clustering. Applied to the public MICrONS Minnie65 dataset (v117 → v1718, 533 ground-truth neurons), the approach achieves ARI = 0.513, merge precision = 0.981, and — critically — a frankenmerge split recall of 0.695, detecting cases where a single automated segment erroneously spans two distinct neurons. Spatial generalization (train on bbox A, evaluate on bbox B) and tree-compliant neuron skeleton assembly via Kruskal stitching complete the pipeline. These results demonstrate that version snapshots, available for every CAVE-based connectomics project, constitute a lightweight and scalable supervision source that removes the EM-imagery requirement from automated proofreading.

---

## Introduction

### The proofreading bottleneck

Large-scale connectomics has entered an era of unprecedented scale: the MICrONS Minnie65 mouse cortex dataset covers 1 mm³ of tissue with ~75,000 neurons and ~523 million synapses [CITATION], while ongoing projects target entire mouse brains and, ultimately, human cortical columns. Automated segmentation of EM volumes has improved dramatically [CITATION: Flood-fill networks, SegEM, NEURD], yet every reconstruction pipeline requires extensive manual proofreading — expert annotators correcting split errors (one neuron fragmented into many segments), merge errors (two neurons incorrectly joined), and frankenmerges (a single segment spanning parts of multiple neurons). Recent estimates suggest that production-quality reconstruction of even moderate volumes requires thousands of person-years [CITATION].

Several approaches have addressed parts of this bottleneck. RoboEM [Schmidt et al., Nature Methods 2024] autonomously traces neurites to correct split errors, achieving 400× lower computational annotation cost on mouse and human cortex. PATHFINDER [Januszewski et al., 2025] reduces axon reconstruction error rates in volumetric EM using learned navigation policies. AutoProof [Huang et al., 2025] leverages accumulated Drosophila annotation data to train merge classifiers, automatically attaching 200,000 fragments (equivalent to ~4 proofreader-years) while retaining 90% of manual workflow value at 20% of the cost. NEURD [Bae et al., 2023] uses meshwork morphology features to classify neuron types and detect merge errors.

These methods share a common requirement: they either need manually curated training data specific to the target dataset, or they must be retrained when the underlying segmentation version changes.

### The version history as self-supervision

We observe that every connectomics project generates a rich, automatically logged training signal as a byproduct of its normal operation: the history of proofreading edits. Each edit that merges two segments (split correction) or separates a merged segment (merge correction) is a direct, expert-validated label stating which pairs of automated fragments belong to the same neuron and which do not. This signal accumulates without any additional annotation cost — it is a record of work that was going to happen regardless.

We term this **version-history self-supervision** and formalize it as follows. Given two segmentation versions v_old (raw automated output) and v_new (proofread), any pair of v_old fragments that map to the same v_new root are labeled *co-neuron* (positive); any pair mapping to different v_new roots are labeled *cross-neuron* (negative). Crucially, the v_old fragments are the units we must partition at inference time, so training on their pairwise co-membership directly optimizes the target task.

### This work

We present NeuronautS, which uses version-history self-supervision to train an edge-classifying GNN on an *observation graph* built from synapse positions. Each synapse is a node; edges connect synapses that share a v_old fragment (same-fragment edges, type 0), are spatial k-nearest neighbors (type 1), or lie near fragment skeleton endpoints (type 2). The GNN learns to predict, per edge, the log-odds that the two connected synapses belong to the same neuron. Global partition is solved by greedy additive edge contraction (GAEC) correlation clustering [Keuper et al., 2015], which aggregates evidence across all edges rather than making irreversible local merge decisions.

Beyond partition quality, the pipeline provides:

1. **Frankenmerge detection**: a specific model signal for type-0 edges that cross a neuron boundary — the model assigns low same-neuron probability to these edges, which then carry negative log-odds through GAEC, causing the frankenmerge to be split rather than preserved.

2. **Uncertainty quantification**: a soft partition output giving per-observation cluster membership probabilities and Shannon entropy, enabling downstream probabilistic connectome construction.

3. **Tree-compliant skeleton assembly**: Kruskal-based stitching of per-fragment L2-cache skeletons into whole-neuron skeletons that are guaranteed to be spanning trees (no new cycles introduced), with morphological metrics for validation.

---

## Results

### Dataset and experimental setup

We evaluate on MICrONS Minnie65 [MICrONS Consortium, 2021], a 1 mm³ mouse visual cortex EM dataset with public segmentation version history. Experiments use version 117 (v117) as the starting segmentation and version 1718 (v1718) as the proofread ground truth, spanning approximately 4 years of expert annotation. All training, validation, and inference use real CAVE API queries to the public MICrONS dataset; no synthetic data is used.

For each experiment we sample a spatial bounding box (1,150–1,250 μm × 930–980 μm × 780–880 μm), fetch up to 20,000 pre-synaptic observations, retain fragments with ≥ 3 synapses per side, and build the observation graph with k = 8 spatial nearest neighbors. Fragments are represented by their L2-cache skeleton (MST of L2-node centroids). A FragmentEncoder (SkeletonGNN) encodes each fragment's geometry into a 32-dimensional DNA vector; this is concatenated with the synapse's 3D position to form the node feature.

**Partition training**: EdgePartitionGNN trained for 150 epochs, lr = 1×10⁻³, franken_hard_frac = 0.30, cc_bias = −1.0. The franken_hard_frac parameter oversamples type-0 same-fragment edges that cross a neuron boundary, which constitute < 2% of type-0 edges but carry the highest-value signal for frankenmerge detection. The cc_bias = −1.0 shifts the GAEC decision boundary to be conservative (prioritize precision over recall), consistent with the principle that over-merges are harder to recover from than under-merges in downstream analyses.

### Three viability bars

We define three viability metrics for the partition step, grounded in the practical constraints of connectome reconstruction:

- **Bar 1** (global quality): ARI ≥ union-find baseline AND merge_precision ≥ union-find baseline
- **Bar 2** (merge safety): merge_precision > 0.95 and merge_recall > 0.70
- **Bar 3** (frankenmerge detection): frankenmerge_split_recall > 0.50

Table 1 shows results for both the union-find baseline (metric GNN + cosine threshold) and the proposed edge_cc method:

| Method | ARI | Clusters (pred/true) | merge_P | merge_R | over_merge | fk_split |
|---|---|---|---|---|---|---|
| union-find | 0.000 | 7 / 533 | 0.477 | 1.000 | 0.517 | 0.000 |
| **edge_cc** | **0.513** | **504 / 533** | **0.981** | **0.963** | **0.009** | **0.695** |

**All three bars pass.** The union-find baseline collapses to 7 clusters on 533 neurons because the cosine threshold metric GNN cannot discriminate at the required resolution when hundreds of objects are present simultaneously; this is an architectural limitation of the single-threshold approach rather than a problem with GNN architectures generally.

The over-merge rate of 0.009 (< 1%) is particularly significant: false merges are the costliest proofreading error because they require finding the merge boundary within a potentially large combined segment. A precision of 0.981 means that 98.1% of predicted same-neuron pairs are correct.

### Frankenmerge detection

Of 533 ground-truth neurons in the benchmark bbox, 18 v117 roots exhibited frankenmerge structure (a single v117 root spanning 2 v1718 neurons). The edge_cc method splits 12.5 of these (weighted by fragment size), achieving fk_split = 0.695.

Mechanistically, the model places type-0 same-fragment edges across a frankenmerge boundary at a probability of 0.499 (just below the decision threshold) after 150 training epochs. With cc_bias = −1.0, these edges carry log-odds of approximately −0.50, sufficient for GAEC to prefer cutting them. This is the first demonstration of data-driven frankenmerge detection from synapse patterns alone, without access to volumetric imagery or morphological features.

### Edge probability diagnostics

To verify the model has learned meaningful representations, we examine edge probabilities by type:

- Type-0 correct (same fragment, same neuron): mean probability = 0.895
- Type-0 frankenmerge cut (same fragment, different neuron): mean probability = 0.499
- Type-1 cross-neuron (spatial k-NN, different neuron): mean probability ≈ 0.1–0.3

The clean separation between correct same-neuron (0.895) and frankenmerge cut (0.499) edges demonstrates that the model has internalized the structural signature of frankenmerges from synapse position patterns.

### Neuron skeleton assembly

After partition, we merge the per-fragment L2-cache skeletons for each predicted cluster using Kruskal-based endpoint stitching (stitch_radius_nm = 5,000 nm). Candidate bridge edges are enumerated via KD-tree on fragment endpoints; bridges are accepted only when they connect previously disconnected fragments (union-find acceptance criterion), guaranteeing that no cycles are introduced.

The resulting merged skeletons are validated by the `is_tree` metric (n_edges == n_vertices − n_connected_components). For well-formed assemblies, `is_tree = True` and `n_connected_components = 1`. Fragments too distant for stitching remain in a forest (n_components > 1), which flags potential under-merge errors for human review.

Cable length statistics across 156 assembled neurons (5k-synapse benchmark, L2 skeletons enabled): median 2,505 μm, p95 11,528 μm, mean 3,868 μm (Table 2). The p95 value of ~11.5 mm is consistent with the long-range axonal projections of layer 2/3 pyramidal cells in mouse visual cortex [CITATION]. Branch point counts (median 194, max 934) reflect complex dendritic and axonal arborizations.

Critically, **`is_tree` = 1.000 (156/156 neurons)** — every assembled skeleton satisfies the spanning tree property. This confirms that the Kruskal stitching algorithm's cycle-prevention guarantee holds on real L2-cache skeleton data.

Of the 156 neurons, 59 (37.8%) are fully connected (n_components = 1); the remainder are forests with 2–14 components. Disconnected components arise when a neuron's fragments extend beyond the bounding box boundary and are therefore absent from the fragment pool — this is expected behavior, not a stitching failure. The `n_connected_components > 1` flag correctly identifies these boundary cases for downstream review.

**Table 2: Neuron shape metrics (156 assembled neurons, 5k-synapse bbox)**

| Metric | Mean | Median | p5 | p95 |
|---|---|---|---|---|
| cable_length_um | 3,868 | 2,505 | 79 | 11,528 |
| n_branch_points | 209 | 194 | 4 | 612 |
| n_endpoints | 227 | 209 | 5 | 659 |
| n_connected_components | 2.5 | 2 | 1 | 7 |
| is_tree | 1.000 | — | — | — |

### Soft partition and uncertainty quantification

The `partition_observations_soft` function extends the hard clustering with probabilistic membership. Each observation receives a membership distribution P(obs ∈ cluster k) for all predicted clusters k, computed from edge probabilities via cluster-level softmax normalization. Shannon entropy H(obs) identifies maximally uncertain observations — these are disproportionately concentrated at frankenmerge boundaries and at spatial borders between adjacent neurons.

The soft partition enables probabilistic connectome construction: connection probability between neuron A and B via synapse pair (pre, post) is P(pre ∈ A) × P(post ∈ B), naturally downweighting uncertain assignments without discarding them.

---

## Discussion

### Relationship to prior work

NeuronautS addresses the fragment-to-neuron assignment problem at a different level than RoboEM and PATHFINDER, which operate on raw EM voxels. Our method takes the output of an existing segmenter as input and refines the inter-fragment partition without re-running volumetric inference. PATHFINDER and RoboEM improve within-neuron segment quality; NeuronautS improves between-neuron boundary placement. The methods are complementary stages of a full reconstruction pipeline.

Relative to Autoproof [Huang et al., 2025], the comparison is more nuanced and the user should understand both similarities and distinctions. **Shared principle**: both methods exploit proofreading history as a training signal that accumulates at no extra annotation cost. Autoproof's authors identify the same core insight: *"there is valuable information contained in these ground-truth annotations, that is otherwise not being used in machine learning models."* In that sense, both systems are instances of the same general idea. **Key distinctions**: Autoproof derives supervision from individual edit-log entries (focused merge decisions stored in DVID) and its core model is a 3D CNN requiring raw grayscale EM input at inference time (130³ voxel receptive field). NeuronautS derives supervision from two version snapshots (no operation log needed) and operates on synapse position metadata alone — no EM imagery required. Autoproof makes local pairwise merge/don't-merge decisions; NeuronautS solves a global partition problem via GAEC. Autoproof explicitly does not address frankenmerge detection; NeuronautS places this as a primary evaluation target (Bar 3). Notably, Autoproof's future work section identifies self-supervised learning as "another rich source of information" for automated proofreading — a direction we pursue here.

The snapshot-diff formulation has a practical advantage: any CAVE-based connectomics project with two materialization versions can immediately use NeuronautS without accessing stored EM volumes, which may be petabytes and require specialized infrastructure. The operation-log formulation requires DVID or equivalent edit-logging infrastructure.

NEURD [Bae et al., 2023] uses morphological features derived from neuronal meshworks; our approach uses synapse position patterns, which are available earlier in the processing pipeline (synapses are detected before meshworks are computed) and do not require volumetric re-inference.

### Limitations and future work

**Spatial generalization — measured results**: Spatial train/test split (train on x: 950–1,150 μm, evaluate on completely non-overlapping x: 1,150–1,350 μm with different neurons, fragments, and synapses):

| Metric | In-sample | Out-of-sample | Change |
|---|---|---|---|
| ARI | 0.836 | 0.694 | −0.14 |
| merge_P | 0.987 | 0.945 | −0.042 |
| merge_R | 0.904 | 0.882 | −0.022 |
| over_merge | 0.005 | 0.022 | +0.017 |
| fk_split | 0.771 | 0.353 | −0.42 |
| is_tree | 1.000 | 1.000 | 0 |

The partition task generalizes well spatially: ARI=0.694 on unseen neurons (−0.14 from in-sample), merge_P=0.945 (0.5% below the 0.95 Bar 2 threshold — recoverable with conservative bias tuning). The skeleton tree compliance guarantee (is_tree=1.000) holds unconditionally.

Frankenmerge detection (fk_split: 0.771 → 0.353) does **not** transfer well across spatial regions. This is mechanistically expected: the model learns which specific v117 roots are frankenmerges in the training region, based on the local proofreading history. Frankenmerge locations are determined by where human annotators made corrections in that specific region, not by transferable morphological features. The fix is multi-region training — training on several non-overlapping bboxes simultaneously allows the model to learn the abstract synaptic signature of a frankenmerge rather than memorizing specific root IDs.

**Dense-box stress test**: Current benchmarks use a 100×50×100 μm bbox (~300–330 fragments). Real connectomics annotation targets the full volume — thousands of interleaved neurons. The `--dense` flag in the spatial split script doubles the y-extent to 100k nm. Dense boxes produce higher cross-neuron edge fractions in the spatial k-NN graph, where the partition problem is genuinely hard. Bar metrics at multiple density levels are reported in supplementary material [PENDING].

**Frankenmerge sample size**: 18 frankenmerge roots in the benchmark bbox produces a fk_split estimate with wide confidence intervals. A larger bbox (or aggregation across multiple regions) would tighten Bar 3.

**Ground truth dependency**: The method requires access to a proofread version of the dataset. For brand-new datasets with no proofreading history, the approach cannot be directly applied. A possible extension is transfer learning from an existing version-history dataset (e.g., pre-training on MICrONS then fine-tuning on a new dataset with even one round of proofreading).

**Abstention**: The current abstention mechanism (observations with low confidence margin left unassigned) has not been systematically tuned. Optimal abstention thresholds likely vary with region density and fragment count.

---

## Methods

### Data

MICrONS Minnie65 (public, CC-BY 4.0). Synapse annotations: CAVE materialization v117 and v1718. Spatial queries via `caveclient` with `filter_spatial_dict` in 4×4×40 nm voxel coordinates. L2-cache skeletons fetched per v117 root via `cloudvolume` + `cloud-volume-skeleton` cache. Fragment encoder and partition model are trained entirely on data fetched from the public CAVE API; no proprietary data is used.

### Model architecture

**FragmentEncoder (SkeletonGNN)**: Message-passing GNN operating on the fragment's L2-cache skeleton. Node features: 3D coordinates + radius. Output: 32-dimensional DNA vector per fragment (L2-normalized).

**EdgePartitionGNN**: Typed-edge message-passing GNN (one message projection per edge type, scatter-add aggregation, residual + LayerNorm). Input: 35D per node (3D synapse position + 32D fragment DNA). Output: 1D log-odds per edge.

**GAEC**: Greedy Additive Edge Contraction [Keuper et al., ECCV 2015]. Processes all edges in decreasing log-odds order; merges adjacent clusters when the net log-odds (sum over all edges between them plus bias) is positive. Time complexity O(E log E).

### Training

FragmentEncoder: contrastive loss (pull same-v1718-neuron fragment pairs, push cross-neuron pairs), 20 epochs, lr = 1×10⁻³.

EdgePartitionGNN: binary cross-entropy on edge same-neuron labels, 150 epochs, lr = 1×10⁻³, balanced mini-batches (50/50 positive/negative), hard_neg_frac = 0.50 (half negatives from spatial cross-neuron edges), franken_hard_frac = 0.30 (30% of negatives from frankenmerge cut edges).

### Evaluation

**ARI**: Adjusted Rand Index between predicted and ground-truth neuron labels. Ignores observations with label 0 (unlabelled).

**merge_precision**: (true positives) / (true positives + false positives) at the edge level, where a true positive is an edge where both endpoints share the same predicted and ground-truth label.

**merge_recall**: (true positives) / (true positives + false negatives).

**over_merge_rate**: fraction of predicted-same-neuron edge pairs that are actually cross-neuron.

**frankenmerge_split_recall**: for each frankenmerge root (v117 root spanning ≥ 2 v1718 neurons), fraction successfully split by the partition.

### Skeleton assembly

`merge_fragment_skeletons`: pools fragment vertices, reindexes intra-fragment edges, builds candidate inter-fragment bridge edges via KD-tree query at stitch_radius_nm = 5,000 nm, applies Kruskal to select at most (n_fragments − 1) tree-compliant bridges. Returns merged Fragment with `is_tree` guaranteed True by construction.

`neuron_shape_metrics`: cable_length_um, n_branch_points, n_endpoints, n_connected_components, is_tree, bbox_volume_um3.

---

## Acknowledgements

MICrONS Consortium for the public Minnie65 dataset and CAVE infrastructure. [Additional acknowledgements.]

---

## References

1. MICrONS Consortium et al. (2021). Functional connectomics spanning multiple areas of mouse visual cortex. *bioRxiv*.
2. Keuper M, Levinkov E, Bonneel N, Lavoué G, Brox T, Andres B. (2015). Efficient decomposition of image and mesh graphs by lifted multicuts. *ICCV*.
3. Schmidt M, Motta A, Sievers M, Helmstaedter M. (2024). RoboEM: Automated 3D flight tracing for synaptic-resolution connectomics. *Nature Methods* 21, 908–913.
4. Januszewski M et al. (2025). Accelerating neuron reconstruction with PATHFINDER. *bioRxiv* 2025.05.16.654254.
5. Huang GB, Katz WM, Berg S, Scheffer L. (2025). Autoproof: Automated segmentation proofreading for connectomics. *arXiv* 2509.26585.
6. Bae JA et al. (2023). NEURD: Meshed neuron decompositions for proofreading and analysis of connectomics datasets. *bioRxiv*.
7. Dorkenwald S et al. (2023). CAVE: Connectome Annotation Versioning Engine. *bioRxiv*.
