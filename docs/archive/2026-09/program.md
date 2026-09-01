> **Archived 2026-09-01.** Status: superseded. One of five "direction" docs
> that disagreed about the canonical pipeline (`docs/consolidation_plan.md`
> §1.1, §4.4); moved here with `git mv` so history is preserved. Superseded by
> [`docs/roadmap_global_assembly.md`](../../roadmap_global_assembly.md),
> declared canonical 2026-06-05. Content below is unchanged from the original.

---

# Neuronauts v2: Scaffolded Global Grammar

> ⚠️ **Historical — describes the v2 (EM voxels + agents + GAT) vision, not the
> pipeline that runs today.** For what currently runs, see [`README.md`](README.md);
> for the project's direction, see
> [`docs/roadmap_global_assembly.md`](../../roadmap_global_assembly.md). Retained
> because the outer-loop research brief and the "shared learned representation"
> thesis below still apply.

## Mission

Build a unified system that goes from EM voxels to connectome using a shared
learned representation trained end-to-end against synapse line-graph F1.

**Status: v2 fully implemented.** All five architectural layers are complete
and tested. The primary remaining work is empirical: train on real MICrONS
data, validate downstream assembly quality, and use the outer optimization loop
only after the end-to-end baseline is stable enough to give a reliable signal.

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
   Synapse line-graph F1  ← terminal metric
   Sampled-pair line-graph F1  ← cheaper diagnostic with same semantics
```

## Primary Claim

> A learned coordinate-free path representation shared across local merge
> plausibility, cluster atomicity, bridge prediction, and global graph
> attention — evaluated by downstream synapse line-graph F1.

## Current Reality Check

The original grammar path over-emphasised hand-authored path summaries
(`edge_len`, `radius`, `curvature`). That representation is now being
deprecated as a primary path.

Current preferred feature path:

- `raw_delta3+skeleton`
- per-step isotropic `(dx, dy, dz)` from ordered path points
- concatenated with `skeleton_stepwise_features`

Legacy compatibility path:

- `legacy_geom3`
- retained only for ablations and old checkpoints

Important limitation:

- Grammar training on cached boxes still uses path-like sequences derived from
  synapse geometry, not full agent traces or full mesh supervision.
- Therefore local grammar losses are still surrogate objectives.
- The real acceptance criterion is downstream assembly quality:
  `val_f1`, `val_sampled_f1`, and eventually held-out real-box evaluation.

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

Preferred grammar invocation now includes explicit feature mode:

```bash
python scripts/train.py train \
  --cache-dir data/boxes_v117 \
  --base-version 117 \
  --target-version 1412 \
  --root-remap-tsv data/boxes_v117/root_remap_v117_to_v1412.tsv \
  --max-synapses 100000 \
  --grammar-output models/shared_grammar_raw_skel.pt \
  --epochs 50 \
  --path-feature-mode raw_delta3+skeleton \
  --val-sim-every-n 5 \
  --val-sampled-max-pairs 10000
```

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
| `docs/minnie_column_paradigm.md` | Minnie Column ROI, spatial bins, tubes, easy/medium/hard |
| `docs/minnie_column_downloads.md` | EM, seg, meshes, skeletons, synapse tables — what to attach |
| `experiments/minnie_column/` | Nucleus manifest @1718+, tube synapse fetch (see README) |

## Supervision Sources

### 1. Local merge supervision

- positives: subfragments from the same CAVE root cluster (spatial split at PCA midpoint)
- negatives: nearby fragments from different root IDs
- source: cached synapse tables, no simulation required
- current preferred path features: `raw_delta3+skeleton`
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

Supporting diagnostics:
- local merge accuracy
- local merge BCE
- atomicity accuracy
- bridge prediction MSE / cosine similarity
- GAT edge precision / recall / F1
- sampled-pair line-graph F1

Clarification:

- Grammar training is **not** directly optimizing full line-graph F1.
- Grammar uses local supervised losses.
- GAT training uses BCE + soft-F1 on graph edges.
- Full line-graph F1 remains the terminal metric for deciding whether a system
  change is actually useful.

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

1. **Finish the current grammar run and then train GAT on top of it.**
   The immediate goal is end-to-end downstream evaluation, not more local-only
   feature work.

2. **Judge success at the system level.**
   Compare:
   - `val_merge_bce`
   - `val_merge_acc`
   - `val_f1`
   - `val_sampled_f1`
   - held-out real-box results

3. **If downstream quality improves, promote this stack to the new baseline.**
   Use that baseline for future GAT tuning and outer-loop optimization.

4. **If downstream quality stalls, diagnose translation failure.**
   Typical failure modes:
   - local merge improves but graph assembly does not
   - GAT is not exploiting the improved encoder
   - full line-graph F1 is too blunt, while sampled-pair F1 shows movement
   - current cached path sources are still too weak

5. **Only after a stable baseline exists, use the outer optimizer.**
   `program.md` is the research brief for `scripts/codex_optimize.py` and
   similar loops. The outer loop should be used to propose small code changes
   only once the training/evaluation stack is stable enough to provide a
   meaningful keep/revert signal.

6. **After grammar + GAT baseline is established, add richer morphology.**
   Next likely modality work:
   - mesh-aligned features as an additional feature mode
   - better path sources than synapse-derived pseudo-paths
   - eventually compare `raw_delta3`, `raw_delta3+skeleton`, and mesh-augmented modes

7. **Evaluation rigor.**
   Add a held-out test set separate from the validation boxes used for
   checkpointing. Report per-box and per-neuron distributions, not only means.

## How The Outer Loop Should Use This File

`program.md` is not part of the normal `scripts/train.py` path. It is the
research brief consumed by `scripts/codex_optimize.py` / Gemini-style outer
loops.

Given the current state, the outer loop should focus on:

- improving downstream `val_f1` / `val_sampled_f1`, not just local merge acc
- editing the featureization or grammar internals in small, auditable steps
- preserving train/infer alignment between:
  - cached grammar batches
  - runtime merge scoring
  - GAT node encoding
- avoiding changes that reintroduce heuristic-only geometry bottlenecks

Likely safe targets for the outer loop:

- `neuronauts/grammar.py`
- small featureization changes in `neuronauts/grammar.py`
- path encoder architecture and regularization
- merge / atomicity head behavior

Likely unsafe or low-signal outer-loop behavior:

- changing multiple files at once
- optimizing only local grammar metrics
- relying on one noisy full-F1 measurement without sampled-pair support
- reintroducing hard-coded morphology heuristics as the primary representation

## What To Avoid

- Feature-spreadsheet morphology engineering
- Optimizing only local merge AUC as the primary target
- Treating `legacy_geom3` as the preferred input representation
- Hard-coded spatial thresholds as decision rules (they exist only as candidate generators in `HeuristicConfig.learned()`)
- Splitting the learned representation across disconnected models

## Success Condition

The system is successful when:

- one shared learned representation (`TorchPathEncoder`) supports local merge,
  cluster atomicity, bridge prediction, and global GAT assembly
- terminal synapse line-graph F1 improves measurably over the heuristic baseline
  on a held-out set of real MICrONS boxes
- the improvement is robust across different box locations and neuron types
