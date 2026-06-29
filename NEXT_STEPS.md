# Next Steps

## Where we are

The `neuronauts.coassign` pipeline implements the correct core idea:

- **Synapses are invariant nodes** — stable across segmentation versions (v117, v1412, ...)
- **Task = co-assignment** — partition synapses into neuron cliques via learned P(same neuron) per edge
- **v117 segments supply edge evidence** (same-seg indicator + DNA), not node identity
- **K materializations** — ranked candidate partitions with calibrated probabilities
- **Metrics** — pairwise precision/recall/F1 + coverage@K, all defined on synapse pairs

Current results on 20 real proofread neurons (3 pieces each, 60 training epochs):
- Precision: 0.95 (co-assignments made are almost always correct)
- Recall: 0.42 (under-merging — threshold too conservative)
- coverage@5: False (need threshold tuning + more training)

---

## Immediate (this branch)

### 1. Threshold calibration ✅ DONE — `calibrate_threshold` in `cluster.py`
Sweeps thresholds, clusters at each, returns the F1-maximising cut plus the full threshold→F1 curve. On real v117 (60-neuron box) it lifted best-materialization **F1 from 0.51 → 0.76** (with a larger model + more epochs); calibration alone moved sweep F1 0.724 → 0.792. Wired into `scripts/v117_coassign.py` (default on; `--no-calibrate` to disable).

**Remaining:** the current pass calibrates a single scalar in-sample. At scale, calibrate on held-out graphs and apply the chosen threshold to unseen regions for an unbiased cut. Optionally expose the precision/recall trade-off as a user knob (favour precision for proofreading, recall for discovery).

### 2. Longer training / larger model
60 epochs with d_model=64 is a starting point. Real neurons span hundreds of microns; 3 GNN layers × 8 k-NN hops = ~24 hops of effective range. Likely need:
- More epochs (100-200)
- Larger d_model (128-256) for complex morphologies
- More GNN layers (4-6)

### 3. Endpoint-adjacent edges ← **validated by tree-dna Phase 2.1**
Add a third edge type: skeleton endpoints (leaf vertices) that are spatially near each other across different segments. These are the principled cross-segment bridge sites — where a real neuron was split by the segmentation. Currently the model only has same-seg edges and spatial k-NN; endpoint edges would give it a direct signal for segment merges.

**Real-data result** (`claude/tree-dna-phase-1-G1DNn`, `real_skeleton_partition.py`, 20 v1412 neurons × 3 skeleton pieces):

| Config | ARI trained | Clusters recovered (true=20) |
|---|---|---|
| No endpoint edges, threshold=0.87 | 0.088 | 5/20 |
| **Endpoint edges 10 µm, threshold=0.87** | **0.418** | **17/20** |

Endpoint edges at 10 µm radius alone produced +0.330 ARI and recovered 17/20 clusters correctly. The mechanism: skeleton split creates piece endpoints within 0–1000 nm of each other; with `endpoint_radius_nm=10_000` all adjacent piece-pair endpoints are captured, giving the GNN direct cross-piece same-neuron evidence that spatial k-NN over synapse positions cannot provide (synapses from different pieces may be widely separated). **This is the single most important improvement.**

---

## Findings from related branches

### DNA node features (tree-dna Phase 1)
Branch `claude/tree-dna-phase-1-G1DNn` ran a real-data ablation on 30 real Minnie65 v1412 neurons:
- **Spatial baseline** (uniform synthetic synapses): AUROC **0.493** (chance)
- **DNA AUC random-init** (SkeletonGNN, no training): **1.000** — real skeleton morphology is sufficient for perfect neuron discrimination *even with random weights*. The geometry of different neurons is distinct enough to separate them without learned features.
- **DNA AUC trained** (60 epochs): **1.000** (ceiling)

This validates the DNA node feature: real skeletons carry sufficient morphological identity for co-assignment, independent of the edge structure. The challenge is that v117 atoms are fragments (pieces of neurons), not whole skeletons — they have partial morphology. The endpoint-edge result above is the bridge.

### GAEC correlation clustering (tree-stitch Phase 2.2)
Branch `claude/abstract-tree-stitch` implements `EdgePartitionGNN` + **Greedy Additive Edge Contraction** (GAEC) correlation clustering. Unlike the current greedy-pivot algorithm (which is threshold union-find), GAEC:
- Contracts on *net* evidence between clusters (not per-edge)
- Can cut a high-similarity edge when the rest of the graph disagrees — handles the **frankenmerge** case where one spurious high-similarity cross-neuron edge would irreversibly fuse two cells under union-find
- Exposes `bias` knob for precision/recall trade-off without re-training

Synthetic ablation (20 objects × 3 pieces, frankenmerge_frac=0.25) showed GAEC outperforms threshold union-find at equal input. This is the target inference algorithm when this branch matures.

---

## Medium term

### 4. Prototype-based assignment (EM-style)
The current model does purely pairwise comparison. This fails for "see-through" cases: segment A and segment C belong to the same neuron, but segment B between them is a frankenmerge with ambiguous DNA. A→B is weak, B→C is weak, so A and C never get connected.

Fix: maintain a running embedding per growing neuron hypothesis (mean of its members). Assign each unassigned synapse to the hypothesis it most resembles — or mark it "uncertain" if it doesn't clearly match any. Iterate (E-step / M-step) until convergence. The neuron prototype aggregates all clean evidence from across the whole arbor, bypassing local gaps.

### 5. Within-type evaluation
`python scripts/coassign_demo.py --cell-type 23P`

All neurons are L2/3 pyramidal — same cell type, similar morphology. This is the honest test: the model must distinguish individual neurons, not just cell types. Current cross-type results (precision 0.95) are partly driven by easy cross-type separations. Within-type is harder and more representative of the real use case.

### 6. Tree-topology constraint
Neurons are trees. Merged clusters should be topologically consistent: no cycles, and the spatial arrangement of merged segments should form a plausible arborisation. Add a post-processing step that rejects merges that would create topological impossibilities (e.g., two segments whose combined skeleton has a cycle).

---

## Longer term

### 7. Multiple materializations for human review
The probabilistic output (K materializations) is the key feature for human-in-the-loop proofreading. The UI should show the top-K partitions for a region and highlight the edges where materializations disagree — those are the uncertain merge decisions. A proofreader reviews only those specific decisions, not the whole partition.

### 8. Global context / long-range structure
The current GNN has bounded receptive field (n_layers × k_spatial hops). A neuron that spans 1 mm of tissue may have its soma and distal dendrite tips connected through many hops. Options:
- Hierarchical graph (coarsen the synapse graph at multiple scales)
- Transformer attention over all nodes (quadratic but no range limit)
- Virtual "neuron node" that aggregates all its members (prototype approach, see #4)

### 9. Integration with the proofreading workflow
The output of `materializations()` should connect to the CAVE annotation system: top-1 partition as a proposed merge set, with confidence scores per merge that can be reviewed and overridden. The synapse-level formulation makes this natural — CAVE tracks synapses, and the partition is a labelling of those stable synapse IDs.

---

## How to contribute

The pipeline is in `neuronauts/coassign/` — four files, ~500 lines. Each file has a single clear responsibility. The demo in `scripts/coassign_demo.py` runs end-to-end in ~5 minutes on CPU with 20 neurons.

Good first contributions:
- **Threshold calibration** (#1 above): add `calibrate_threshold(model, graphs)` to `cluster.py`
- **Endpoint edges** (#3): add a new edge type in `graph.py` using skeleton endpoint positions from `Fragment.endpoints_nm`
- **Within-type test** (#5): run `--cell-type 23P` and report precision/recall vs cross-type

See `neuronauts/coassign/README.md` for architecture details and quick-start code.
