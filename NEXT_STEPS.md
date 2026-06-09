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

### 1. Threshold calibration
The model learns good edge scores but the fixed threshold=0.5 in `greedy_cluster` is too conservative. Add a calibration pass: sweep threshold on held-out graphs, pick the value maximising F1. The precision/recall trade-off should be user-configurable.

### 2. Longer training / larger model
60 epochs with d_model=64 is a starting point. Real neurons span hundreds of microns; 3 GNN layers × 8 k-NN hops = ~24 hops of effective range. Likely need:
- More epochs (100-200)
- Larger d_model (128-256) for complex morphologies
- More GNN layers (4-6)

### 3. Endpoint-adjacent edges
Add a third edge type: skeleton endpoints (leaf vertices) that are spatially near each other across different segments. These are the principled cross-segment bridge sites — where a real neuron was split by the segmentation. Currently the model only has same-seg edges and spatial k-NN; endpoint edges would give it a direct signal for segment merges.

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
