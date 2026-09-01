> **Archived 2026-09-01.** Status: superseded. One of five "direction" docs
> that disagreed about the canonical pipeline (`docs/consolidation_plan.md`
> §1.1, §4.4); moved here with `git mv` so history is preserved. This is the
> implementation plan for the box-local CellGNN, which has an architectural F1
> ceiling of ~0.27 (`docs/architecture.md`). Superseded by
> [`docs/roadmap_global_assembly.md`](../../roadmap_global_assembly.md)
> (canonical since 2026-06-05), which extends rather than replaces this plan:
> the CellGNN becomes the within-region assembler of that roadmap's Stage C.
> Content below is unchanged from the original.

---

# Global Topological Merge: Implementation Plan

> **Status: Phase 0 + Sampling Strategy implemented.**
> `neuronauts/cell_graph.py` contains the core CellGNN architecture,
> training loop, and sampling utilities (tangledness scoring, spatial
> train/val/test splitting).  `tests/test_cell_graph.py` has 40 passing
> tests.  `scripts/train.py train-cell-gnn` provides the CLI entry point.
> Remaining phases are tracked below.

---

## Reconciled Approach

The original 4-phase plan and the "graph reachability / GNN" direction are
unified into a single architecture: **the CellGNN**.

**Why this reconciles the two ideas:**

| What was asked for | How CellGNN delivers it |
|--------------------|------------------------|
| Not pairwise — pairwise scores as evidence, not decisions | Grammar scores are one of four *edge features*; the GNN decides in context |
| Global reachability argument | K message-passing rounds = K-hop reachability; synapses on the same arbor converge to similar embeddings |
| Hierarchical substructure → full cell | K=1 → scaffold clusters; K=2 → local branch fragments; K=3+ → full arbor |
| Learnable (GNN) | `CellGNN`: edge-conditioned message passing trained with contrastive loss against CAVE root IDs |
| Probabilistic cell output | Embedding similarity = cell membership probability; `partition_from_embeddings` thresholds it |
| No simulation required for training | Like grammar training, works from cached synapse tables alone |

**How pairwise work is reused:**
The existing grammar produces pairwise merge scores between scaffold groups.
These become the `grammar_score` edge feature in `build_synapse_graph`.
The GNN learns when to trust these scores (e.g., high-quality direct path)
vs. override them (e.g., pairwise score is high but the cluster would be
non-atomic at the cell level).

---

## 1. Problem Statement

Current Neuronauts makes **local** merge decisions: pairwise grammar scores
between nearby agent fragments, resolved by beam search within a small
candidate window. The resulting "neurons" are assemblages of locally compatible
fragments — but no step asks whether the **full cell** (the complete set of
synapses assigned to one neuron on one side) is globally consistent.

**Goal:** Replace greedy bottom-up assembly with a system that reasons about
**full cell hypotheses** and selects the partition of synapses into cells that
maximizes global topological correctness (line-graph F1).

## 2. Key Definitions

| Term | Meaning |
|------|---------|
| **Cell** | All synapses belonging to one neuron on one side (pre or post) within a box |
| **Cell hypothesis** | A proposed set of synapse indices asserted to be one cell |
| **Partition** | A complete assignment of all synapses to cells (one per side) |
| **Ground-truth cell** | Synapses sharing the same CAVE root ID on the same side at target version |
| **Edit tree** | The MICrONS chunkedgraph edit history: sequence of merge/split operations by proofreaders |
| **Topological merge** | A merge decision informed by the topology of the full resulting cell, not just pairwise fragment compatibility |

## 3. What Already Exists

### 3.1 Infrastructure (fully implemented)

| Component | File(s) | Role in global merge |
|-----------|---------|---------------------|
| Scaffold init | `run.py::_scaffold_union_from_seg_ids` | ~10x search-space reduction; forms initial cell proposals |
| Pairwise grammar scoring | `grammar.py`, `run.py::_merge_role_groups` | Generates candidate merge edges between scaffold groups |
| Atomicity head | `shared_grammar_model.py::score_atomicity` | Binary: is this cluster one cell or a merge error? |
| Beam search | `assembly.py::beam_search_merge_groups` | Local merge ordering with atomicity weight |
| GAT edge refinement | `assembly.py::gat_refine_connectivity` | Global context for edge accept/reject |
| GAT soft-F1 training | `shared_grammar_model.py::gat_train_step` | Differentiable partition-aware loss (at edge level) |
| Line-graph F1 evaluation | `line_graph.py::evaluate` | Terminal metric |
| Root-ID remapping | `cave_root_mapping.py` | Cross-version ground truth (v117 -> v1412) |
| Box cache | `dataset_builder.py::BoxCache` | Cached MICrONS boxes with synapse tables |

### 3.2 Ground truth available

- **CAVE root IDs** at materialization v1412: define true cells per box
- **Root-ID remapping tables**: allow comparison across versions
- **Chunkedgraph API**: `client.chunkedgraph.get_latest_roots()` maps old -> new
- **Proofread core**: curated neurons with high-quality root assignments

### 3.3 What's missing (the gap)

1. **Cell-level hypothesis scoring**: No module evaluates a proposed full cell
   for biological plausibility (beyond pairwise merge + binary atomicity).

2. **Partition-level optimization**: No step considers alternative partitions
   and selects the best one. Beam search is over individual merge decisions,
   not over competing whole-partition hypotheses.

3. **Edit-tree supervision**: The chunkedgraph is used only for root-ID
   mapping. The actual edit history (merge/split operations, their order, the
   fragments involved) is not consumed as training data.

4. **Top-down cell proposals**: Cells are only formed bottom-up (agents ->
   scaffold -> pairwise merge). No mechanism proposes cells top-down from
   known root-ID structure or from connectivity patterns.

## 4. Implementation Phases

### Phase 0: CellGNN Core ✓ DONE
**`neuronauts/cell_graph.py` — 29 tests passing**

Implemented:
- `build_synapse_graph` — constructs the evidence graph from scaffold IDs,
  spatial proximity, grammar scores, and shared agent visits
- `CellGNN` — edge-conditioned message-passing GNN with K layers
- `partition_from_embeddings` — agglomerative / greedy clustering from cosine sim
- `cell_graph_train_step` — contrastive pull/push with correct hinge loss
- `train_cell_gnn` — epoch loop over a BoxCache (no simulation needed)
- `infer_cells` — inference returning per-synapse cell labels
- `connectivity_graph_from_cell_labels` — labels → `ConnectivityGraph` for F1 eval
- `save_cell_gnn` / `load_cell_gnn` — persistence

The `F1 roundtrip` test confirms a perfect partition of ground-truth root IDs
yields F1=1.0 through the full pipeline.

### Phase 0.5: Sampling Strategy ✓ DONE
**Tangledness-aware sampling + spatial train/val/test split**

Implemented in `cell_graph.py`:
- `score_box_tangledness` — scores each cached box for root-ID complexity
  (root density, multi-root fraction, composite tangledness score).
  Tangled boxes = many distinct roots sharing synapses in a small volume,
  i.e. the hard cases from the proofread core with edit history.
- `rank_boxes_by_tangledness` — sorts boxes most-tangled-first with
  min_synapses / min_positive_pairs filtering.
- `spatial_train_val_test_split` — bins boxes along a spatial axis
  (quantile-based), assigns bins to splits so nearby boxes (which may
  share neurons) stay together.  Prevents data leakage.
- `select_cell_gnn_training_boxes` — end-to-end pipeline: score → filter
  → spatial split → cap train/val sizes (keeping most tangled).

CLI entry point: `python scripts/train.py train-cell-gnn --cache-dir <dir>`

Workflow:
1. Build proofread-core cache:
   `python experiments/root_neighborhood/dataset.py build-cache --cache-dir data/proofread --version 1718`
2. Train CellGNN:
   `python scripts/train.py train-cell-gnn --cache-dir data/proofread --epochs 50`

11 new tests (40 total) verify tangledness scoring, spatial splitting,
and the full selection pipeline.

---

### Phase 1: Cell-Level Plausibility Scoring
**Effort: Small. Uses existing infrastructure.**

**Goal:** After the current pipeline produces neurons, score each resulting
cell for global plausibility and use that score to trigger re-partitioning.

**Steps:**

1. **Compute per-cell atomicity score.** After `_merge_role_groups` produces
   neuron groups, run `score_atomicity` on each neuron's branch sequences.
   Neurons with low atomicity scores are candidates for splitting.

2. **Add a cell-quality diagnostic.** For each neuron, compare its synapse
   set against ground-truth root IDs. Compute:
   - purity = fraction of synapses from the majority root ID
   - completeness = fraction of the majority root's synapses that are captured
   This is an evaluation diagnostic, not a training signal (it uses labels).

3. **Re-partition low-atomicity cells.** When atomicity < threshold, attempt
   to split the cell using the grammar's pairwise scores as an affinity matrix
   and spectral or agglomerative clustering to find subclusters.

4. **Measure impact on line-graph F1.** Compare F1 before and after
   re-partitioning. This validates whether cell-level reasoning helps.

**Files to modify:** `run.py` (add post-merge cell scoring), `assembly.py`
(add cell-split logic).

### Phase 2: Partition-Aware Merge Optimization
**Effort: Medium. Requires new optimization loop.**

**Goal:** Instead of greedily accepting the beam-search result, generate
multiple competing partitions and select the one with the highest predicted
line-graph F1.

**Steps:**

1. **Generate K partition candidates.** Run beam search with different
   parameters (threshold, beam width, atomicity weight) to produce K distinct
   partitions of synapse-to-cell assignments.

2. **Score each partition.** For each candidate partition:
   - Build the ConnectivityGraph
   - Run the GAT to get edge scores
   - Compute predicted soft-F1 over the partition's edges
   Select the partition with the highest predicted F1.

3. **Train the GAT as a partition evaluator.** Extend `gat_train_step` so
   that the GAT not only scores edges but also provides a partition-level
   quality estimate. The soft-F1 loss already does this implicitly; make it
   explicit by exposing a `score_partition()` method.

4. **Backpropagate through merge decisions (stretch).** Make the merge
   threshold a differentiable parameter so that gradient from soft-F1 flows
   back to the grammar's merge scores. This is the full end-to-end version.

**Files to modify:** `assembly.py` (partition generation, scoring),
`shared_grammar_model.py` (partition evaluator), `run.py` (integration).

### Phase 3: Edit-Tree Supervision
**Effort: Medium. Requires new data pipeline.**

**Goal:** Consume the MICrONS chunkedgraph edit history as additional
training signal — proofreader merges as positive pairs, proofreader splits
as hard negatives.

**Steps:**

1. **Fetch edit history from chunkedgraph.** The CAVE API exposes
   `client.chunkedgraph.get_tabular_changelog()` or similar endpoints that
   return the history of merge/split operations for a given root ID. Build a
   fetcher that, for each root ID in a cached box, retrieves its edit log.

2. **Convert edits to training pairs.**
   - Merge edit: the two pre-merge root IDs are now confirmed same-cell
     -> positive merge pair
   - Split edit: the two post-split root IDs are now confirmed different-cell
     -> hard negative merge pair
   - These are especially valuable because they represent cases that were
     ambiguous enough to require human intervention.

3. **Augment merge training data.** Add edit-derived pairs to the existing
   merge training set in `multitask_train_step`. Weight them higher than
   synthetic pairs (they represent hard cases).

4. **Temporal curriculum.** Train on older edit versions first (more
   errors to learn from), validate against latest version (cleanest labels).
   The existing `--base-version` / `--target-version` infrastructure supports
   this directly.

**Files to create:** `neuronauts/edit_history.py` (edit log fetcher and pair
extraction).
**Files to modify:** `scripts/train.py` (add `--use-edit-history` flag),
`shared_grammar_model.py` (augmented training data).

### Phase 4: Top-Down Cell Proposals
**Effort: Large. New modeling approach.**

**Goal:** Generate cell proposals not just bottom-up from agent merges, but
top-down from known connectivity patterns and root-ID structure.

**Steps:**

1. **Root-ID seeded proposals.** For each root ID present in a box, collect
   all its synapses. This defines the ground-truth cell boundary. Use this as
   a "teacher" proposal during training: the model learns what a correct cell
   looks like, then must reproduce similar cells at inference without labels.

2. **Connectivity-based proposals.** Use the synapse graph structure to
   propose cells: synapses that are densely connected (many shared pre/post
   partners) likely belong to the same cell. This is a graph clustering
   problem over the synapse-level connectivity graph.

3. **Proposal scoring and selection.** Score all proposals (bottom-up from
   merges + top-down from connectivity) with the cell-plausibility model from
   Phase 1. Select a non-overlapping subset that covers all synapses and
   maximizes total cell quality.

4. **Set-prediction loss.** Train the system with a Hungarian-matching or
   set-prediction loss that directly compares proposed cells against
   ground-truth cells, rather than decomposing into pairwise decisions.

**Files to create:** `neuronauts/cell_proposals.py` (proposal generation and
scoring).
**Files to modify:** `assembly.py`, `shared_grammar_model.py` (set-prediction
loss).

## 5. Recommended Execution Order

```
Phase 0 (CellGNN core)      ✓ DONE — cell_graph.py + 29 tests
  |
  v
Phase 0.5 (sampling)        ✓ DONE — tangledness + spatial splits + CLI
  |
  v
Phase 1 (baseline eval)     <- DO THIS NOW
  |                            Train CellGNN on proofread-core cache,
  |                            compare F1 vs current beam-search pipeline
  v
Phase 2 (grammar integration) <- NEXT
  |                            Feed grammar pairwise scores as edge features
  |                            Confirm CellGNN outperforms grammar-only
  v
Phase 3 (edit tree)          <- PARALLEL
  |                            Fetch proofreader merge/split history
  |                            Use as hard training examples
  v
Phase 4 (partition loss)     <- ONLY IF CONTRASTIVE PLATEAUS
                               Replace contrastive with soft line-graph F1 loss
                               Directly optimise the terminal metric
```

## 6. Success Criteria

| Phase | Metric | Target |
|-------|--------|--------|
| 1 | Line-graph F1 improvement from cell re-partitioning | > 0.02 F1 lift |
| 2 | Partition-search F1 vs single beam search | > 0.03 F1 lift |
| 3 | Grammar merge accuracy on edit-derived hard pairs | > 0.80 |
| 4 | Cell-level IoU between proposed and ground-truth cells | > 0.85 |

## 7. Relationship to Existing Docs

- `docs/whitepaper.md`: Describes the full architecture; this plan extends
  Section 2.5 (Global Assembly GAT) with cell-level reasoning.
- `docs/global_validation_layer.md`: Design history for the GAT layer; this
  plan addresses the "partition proposal" and "attachment/identity proposal"
  outputs described there but not yet implemented.
- `program.md`: Lists empirical next steps; this plan is the architectural
  roadmap for the "if downstream quality stalls, diagnose translation failure"
  branch.
