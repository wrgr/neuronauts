# Neuronauts v2: Scaffolded Global Grammar

## Mission

Build a unified system that goes from EM voxels to connectome using a shared
learned representation trained end-to-end against synapse line-graph F1.

**Status: v2 fully implemented.** All five architectural layers are complete
and tested (329 tests). The primary remaining work is training on real MICrONS
data and driving the learned components with real supervision.

## Architecture (five layers)

```
EM volume + CAVE synapses
        │
        ▼
1. Perception          fetch.py · fields.py · vectorized.py · membrane_unet.py
   2.5D membrane U-Net (InstanceNorm2d, context slices)
   700 agents × 450 steps → path arrays + synapse hits

        ▼
2. Scaffold init       run.py → _scaffold_union_from_seg_ids
   CAVE seg-IDs pre-group same-supervoxel agents → ~10× search space reduction
   HeuristicConfig: spatial thresholds become candidate generators, not decisions

        ▼
3. Shared Grammar      grammar.py · shared_grammar_model.py
   TorchPathEncoder: Transformer + [CLS] token, multi-modal
   MergeScorer, BridgeHead (6D trajectory prediction)
   multitask_train_step: merge loss + atomicity loss + bridge loss

        ▼
4. Global Assembly     assembly.py · shared_grammar_model.py
   GlobalAssemblyGAT: sparse multi-head attention over ConnectivityGraph
   gat_train_step: BCE + soft-F1 surrogate, path encoder frozen
   label_graph_edges: per-edge labels from majority-vote root-ID matching

        ▼
5. Evaluation          line_graph.py
   Synapse line-graph F1  ← primary scalar for all training and evaluation
```

## Primary Claim

> A learned coordinate-free path representation shared across local merge
> plausibility, cluster atomicity, bridge prediction, and global graph
> attention — trained end-to-end against synapse line-graph F1.

## Primary Training Path

The recommended workflow uses `scripts/train.py`:

```bash
# Step 1: Build a cache of real MICrONS boxes (network required)
python scripts/train.py build-dataset \
  --cache-dir data/boxes \
  --n-boxes 100 \
  --min-synapses 15

# Step 2: Train grammar + GAT
python scripts/train.py train \
  --cache-dir data/boxes \
  --grammar-output models/shared_grammar.pt \
  --gat-output models/gat.pt \
  --epochs 50 \
  --train-gat
```

**Grammar training** uses cached synapse tables directly — no agent simulation
required. Each box contributes merge examples (same-root positive, nearby-different-root
negative) and topology/atomicity examples. Approximately 0.3 s/box on CPU.

**GAT training** additionally requires agent path simulation (~30 s/box on CPU).
The `--gat-every-n-epochs 5` flag amortizes this cost by training the GAT every
5 grammar epochs rather than every epoch.

### Train from proofread-core v117 cache (comparison to latest)

To build a cache focused on proofread anchors in the proofread core:

```bash
python scripts/train.py build-dataset \
  --cache-dir data/proofread_core_v117 \
  --strategy proofread-core \
  --cave-version 117 \
  --proofread-n-roots 50 \
  --proofread-radius-um 40 \
  --proofread-min-anchor-synapses 50
```

When your box cache was built with `--cave-version 117`, train so that **labels**
are in the latest materialization (1412). That way merge/atomicity/GAT supervision
compares v117 segmentation to the current synapse set.

**1. Check the cache** (optional — confirm roots are v117):

```bash
BASE_CACHE_DIR="data/boxes_v117"   # or wherever you built with --cave-version 117
python -c "
import json, sys
path = sys.argv[1]
idx = json.load(open(path))
vers = {r.get('root_id_version') for r in idx if r.get('root_id_version') is not None}
print('root_id_version in cache:', vers or 'none set (assume 117 if built with --cave-version 117)')
" "$BASE_CACHE_DIR/index.json"
```

**2. Build the root-ID mapping table** (one-time, needs network):

```bash
python scripts/train.py remap-roots \
  --cache-dir "$BASE_CACHE_DIR" \
  --base-version 117 \
  --target-version 1412 \
  --output "$BASE_CACHE_DIR/root_remap_v117_to_v1412.tsv"
```

**3. Train using the v117 cache with labels in latest (1412) space:**

```bash
python scripts/train.py train \
  --cache-dir "$BASE_CACHE_DIR" \
  --base-version 117 \
  --target-version 1412 \
  --root-remap-tsv "$BASE_CACHE_DIR/root_remap_v117_to_v1412.tsv" \
  --grammar-output models/shared_grammar_real.pt \
  --gat-output models/gat_real.pt \
  --epochs 30
```

Training will load each box’s v117 roots, apply the precomputed mapping to 1412,
drop synapses that vanish (mapped to 0), and build merge/atomicity/GAT labels
from the **mapped** root IDs. So you are training against the latest synapse set
while keeping box geometry from the v117 pull.

## Key Files

| File | Purpose |
|---|---|
| `neuronauts/grammar.py` | `TorchPathEncoder`, `MergeScorer`, `PathBatch` |
| `neuronauts/shared_grammar_model.py` | `SharedGrammarModel`, `BridgeHead`, `GlobalAssemblyGAT`, `GATTrainingConfig`, `gat_train_step`, `train_global_assembly_gat`, `multitask_train_step` |
| `neuronauts/assembly.py` | `gat_refine_connectivity`, `label_graph_edges` |
| `neuronauts/run.py` | Runtime runner, `HeuristicConfig`, `_scaffold_union_from_seg_ids`, `simulate_paths_and_hits` |
| `neuronauts/dataset_builder.py` | `BoxCache`, `select_random_boxes`, `build_dataset` |
| `scripts/train.py` | ★ Primary training CLI |
| `neuronauts/line_graph.py` | `evaluate` → line-graph F1 (primary metric) |

## Supervision Sources

### 1. Local merge supervision

- positives: subfragments from the same CAVE root cluster (spatial split at PCA midpoint)
- negatives: nearby fragments from different root IDs
- source: cached synapse tables, no simulation required
- versioning: root IDs are assumed to live at a configurable ``--base-version``
  (default 1412).  When ``--target-version`` differs, root IDs are mapped
  forward via ``chunkedgraph.get_latest_roots`` before supervision is
  constructed so labels always correspond to the target materialization.
- cache alignment: run ``scripts/train.py build-dataset`` with ``--cave-version``
  matching ``--base-version`` so cached ``pre_root_id``/``post_root_id`` are in
  the expected label space.
- recompute connectivity: when you change the root-ID mapping, you should
  rebuild any cached per-box derived supervision (at minimum: mask synapses
  that vanish under the mapping and recompute per-box ``n_positive_pairs``).
  Use ``scripts/train.py remap-cache-roots`` to produce a target-version
  cache where training targets are consistent.

### 2. Global atomicity supervision

- positive: synapse cluster where all synapses share one root on the relevant side
- negative: cluster formed by merging two distinct roots
- source: cached synapse tables, no simulation required
- versioning: the same ``base_version → target_version`` mapping is applied as
  for merge supervision; clusters that involve roots which vanish under the
  mapping (mapped to 0) are currently dropped from training.

### 3. Self-supervised bridge loss

- target: 3D midpoint + 3D tangent between adjacent fragment endpoints
- derived geometrically from synapse positions, no manual labels

### 4. GAT soft-F1 loss

- label: per-edge binary from majority-vote root-ID matching
- loss: `(1−w)·BCE + w·(1 − soft_F1)` with `w=0.5`
- this directly aligns GAT training with the terminal metric

## Objective

Primary scalar: **synapse line-graph F1**

Supporting diagnostics (not training targets):
- local merge accuracy
- atomicity accuracy
- bridge prediction MSE / cosine similarity
- GAT edge precision / recall / F1

## Design Principle: Dataset-Centric Optimization

Unlike most ML, connectomics has **one big experiment** (Minnie65) that is very
painful to produce and make sense of. Generalization to other datasets is nice,
but the primary value is: *can we correctly interpret this connectome?*

- **Overfitting to Minnie65 is desirable.** A model that fits this dataset
  extremely well, even if it does not transfer, is incredibly useful.
- **Train for capacity.** Use larger models, longer training, dataset-specific
  tuning. Don't stop early mainly to avoid overfitting.
- **Validate within Minnie65.** Held-out regions/boxes of the same volume matter;
  cross-dataset transfer is secondary.
- **A limited result is still valuable.** A model that works well only on this
  volume would be a major outcome.
- **The real risk is overfitting to noise or artifacts** in the training split,
  not to the structure of the volume.

## What Is Actually Complete

All items from the global inference roadmap are implemented:

| Component | Status |
|---|---|
| Transformer PathEncoder + [CLS] | ✓ `grammar.py` |
| Multi-modal PathBatch (skeleton, mesh) | ✓ `grammar.py`, `fetch.py` |
| BridgeHead + Dijkstra proposals | ✓ `dijkstra.py`, `shared_grammar_model.py`, `run.py` |
| Scaffold init from CAVE seg-IDs | ✓ `run.py`, `fetch.py` |
| 2.5D MembraneUNet (InstanceNorm2d) | ✓ `membrane_unet.py` |
| GlobalAssemblyGAT + training loop | ✓ `assembly.py`, `shared_grammar_model.py` |
| HeuristicConfig decommissioning | ✓ `run.py` |
| Real-data BoxCache + selection | ✓ `dataset_builder.py` |
| End-to-end training CLI | ✓ `scripts/train.py` |

## What To Work On Next

The remaining work is empirical, not architectural:

1. **Run real-data training.** Execute `scripts/train.py run --cache-dir data/boxes --n-boxes 100 --epochs 50 --train-gat` on MICrONS data and measure val F1.

2. **Diagnosis loop.** If val F1 is not improving, diagnose which component is the bottleneck (merge accuracy? atomicity accuracy? GAT edge precision?) using the per-component diagnostics logged in `run_logs/train_log.tsv`.

3. **Model improvements (ranked by expected impact):**
   - Increase training data: more boxes, longer training, larger `volume_shape` for synthetic GAT data.
   - Richer path features: feed skeleton tortuosity and mesh volume-surface ratio into `PathBatch.skeleton_feat` and `mesh_feat`.
   - Pre-train on CAVE edit decisions if available (accepted/rejected merge decisions as additional merge supervision).
   - Scale the model: larger `d_model`, more Transformer layers, more GAT heads.
   - Multi-scale inference: run at MIP 1 and MIP 3 in addition to MIP 2.

4. **Evaluation rigor.** Add a held-out test set separate from the validation boxes used for checkpointing. Report per-neuron F1 distribution, not only mean.

## What To Avoid

- Feature-spreadsheet morphology engineering
- Optimizing only local merge AUC as the primary target
- Hard-coded spatial thresholds as decision rules (they exist only as candidate generators in `HeuristicConfig.learned()`)
- Splitting the learned representation across disconnected models

## Success Condition

The system is successful when:

- one shared learned representation (`TorchPathEncoder`) supports local merge,
  cluster atomicity, bridge prediction, and global GAT assembly
- terminal synapse line-graph F1 improves measurably over the heuristic baseline
  on a held-out set of real MICrONS boxes
- the improvement is robust across different box locations and neuron types
