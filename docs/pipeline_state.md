# Pipeline State & Reinitialization Guide

Status: 2026-05-01 (updated).

---

## What We Have

### Box Cache
- **Location:** `data/boxes_30um/` — 247 CAVE boxes, 30 µm windows (fetched 2026-05-01)
- **Root IDs:** v1412 (proofread ground truth)
- **Synapse table:** `synapses_pni_2`, pre + post positions + supervoxel IDs
- **Stats:** 1.68M total synapses, avg 6,814/box, 1.63M positive pairs

### CAVE Edit Pairs (training signal for path encoder)
| File | Merge pairs | Split pairs | Notes |
|------|------------|-------------|-------|
| `data/cave_edit_pairs.tsv` + `cave_edit_chains.npz` | 574 | 0 | v1, early break bug |
| `data/cave_edit_pairs_v2.tsv` + `cave_edit_chains_v2.npz` | 4007 | 0 | v2, still early break |
| `data/cave_edit_pairs_v3.tsv` + `cave_edit_chains_v3.npz` | TBD | TBD | v3, fetching: 2002 roots, 493K svids resolving |

**Why v1 and v2 have 0 split pairs:** the fetch loop broke out early once the merge cap was hit,
so `cur_root_to_sids` only accumulated ~124/1870 roots. The split detection pass (which requires
cross-root overlap) found nothing. Fixed in commit `6316ce6`.

### Trained Checkpoints

| Checkpoint | Architecture | Val acc / F1 | Notes |
|-----------|-------------|-------------|-------|
| `models/grammar_cave_real_50.pt` | Transformer grammar + MergeScorer | 87.23% merge acc | Trained on 40 real boxes |
| `models/path_encoder_cave_v2_ep40.pt` | PathEdgeEncoder d=32 | 87.1% | 4007 merge pairs, 0 split — converged |
| `models/cell_gnn_real.pt` (ep3) | CellGNN scalar K=3 | val F1 ≈ 0.859 (ep3) | Trained on real data, overfit by ep4 |
| `models/cell_gnn_30um_edge.pt` | CellGNN edge-mode | — | Synthetic boxes |
| `models/shared_grammar_real.pt` | Grammar + GAT | — | Full grammar model |

---

## Known Gaps

1. **No split pairs (hard positives)** — model distorts all junctions; calibration is off.
   Fix: fetch v3 (in progress) with `max_false_merges=99999` so all roots are scanned.

2. **Short chain / isolate examples** — `add_edit_history_examples` previously required
   `window_size//2` synapses per side, dropping all 1-synapse isolates.
   Fixed in commit `6316ce6`: now uses max(1, available) with start-padding.

3. ~~**CellGNN → GAT not chained**~~ — **fixed**: CellGNN block (`run.py:1409`) flows
   directly into GAT (`run.py:1452`) with no early return; both run additively when
   both checkpoints are provided.

4. **No head-to-head evaluation** — no numbers comparing beam-search only vs CellGNN vs
   Grammar+CellGNN on the same test split.

5. **O(N²) clustering bottleneck** — `partition_from_embeddings` in `cell_graph.py:422`
   does full pairwise cosine sim; will OOM on cells with >500 synapses.

---

## Reinitialization — Full Pipeline from Scratch

Run each step in order. Each step depends on the previous.

### Step 0: Prerequisites

```bash
# CAVE token
cat ~/.cloudvolume/secrets/cave-secret.json   # must contain {"token": "..."}

# Box cache (252 boxes already present)
ls data/boxes_30um/ | wc -l   # should be ~504 (json + npz per box)
```

### Step 1: Fetch Real CAVE Edit Pairs

Samples 50K delta roots from v117 (2021-06-11), resolves supervoxel→current-root mappings
in batches, yields both false-merge (hard negatives) and false-split (hard positives).

```bash
python scripts/train.py fetch-cave-edits \
  --n-sample 50000 \
  --max-false-merges 99999 \
  --min-synapses-per-root 8 \
  --output-tsv data/cave_edit_pairs_v3.tsv \
  --output-chains data/cave_edit_chains_v3.npz
```

Expected output: `~4000–8000 merge pairs + ~200–1000 split pairs` from ~1870 roots.
Runtime: ~10 minutes (batched svid lookup, 247 batches of 2000).

### Step 2: Train Path Encoder

PathEdgeEncoder learns to discriminate valid path continuations from spliced paths.
Training signal: synthetic splice negatives from box cache + real CAVE merge/split pairs.

```bash
python scripts/train.py train-path-encoder \
  --cache-dir data/boxes_30um \
  --epochs 10 \
  --edit-pairs-tsv data/cave_edit_pairs_v3.tsv \
  --edit-chains-npz data/cave_edit_chains_v3.npz \
  --output models/path_encoder_v3.pt
```

Expected: ~87–88% accuracy. Previous ceiling was 87.3% on merge-only; split pairs
should calibrate the model and may push higher.
Runtime: ~15 min/10 epochs.

### Step 3: Train Grammar Model

Grammar (PathEncoder + MergeScorer + ArborEncoder) trained on box cache synapse pairs.
Uses same path feature mode (`raw_delta3+skeleton`) as run.py.

```bash
python scripts/train.py train \
  --cache-dir data/boxes_30um \
  --epochs 10 \
  --grammar-output models/grammar_v3.pt
```

Expected: ~87% merge accuracy (matches `grammar_cave_real_50.pt` in 10 epochs).
Runtime: ~20 min/10 epochs.

### Step 4: Train CellGNN

K-hop synapse-level GNN trained contrastively against CAVE root IDs.

```bash
python scripts/train.py train-cell-gnn \
  --cache-dir data/boxes_30um \
  --epochs 10 \
  --path-encoder-checkpoint models/path_encoder_v3.pt \
  --pretrained-path-emb-dim 16 \
  --cell-gnn-output models/cell_gnn_v3.pt
```

Expected: val F1 ~0.85–0.87 (ep3 was 0.859 on previous run).
Runtime: ~20 min/10 epochs.

### Step 5: Evaluate Full Pipeline

Runs beam-search baseline AND CellGNN+Grammar on the test split, reports line-graph F1.

```bash
# Baseline (no learned models)
python scripts/train.py evaluate \
  --cache-dir data/boxes_30um \
  --split test

# Grammar only
python scripts/train.py evaluate \
  --cache-dir data/boxes_30um \
  --grammar-checkpoint models/grammar_v3.pt \
  --split test

# Grammar + CellGNN
python scripts/train.py evaluate \
  --cache-dir data/boxes_30um \
  --grammar-checkpoint models/grammar_v3.pt \
  --cell-gnn-checkpoint models/cell_gnn_v3.pt \
  --split test
```

---

## Architecture Quick Reference

```
EM volume + synapse positions
        │
        ▼
[1] Agent simulation (700 agents × 450 steps)
        │
        ▼
[2] Grammar merge scoring  ← PathEdgeEncoder + MergeScorer  (path_encoder_v3.pt)
        │
        ▼
[3] Beam-search merge → MergedNeurons (pre + post)          (grammar_v3.pt)
        │
        ▼
[4] Connectivity assembly → ConnectivityGraph
        │
        ├── [5a] CellGNN re-partition (if ckpt)             (cell_gnn_v3.pt)
        │
        └── [5b] GAT edge refinement (if ckpt)
        │
        ▼
[6] Line-graph F1 evaluation
```

### Transfer function framing

The path encoder is not trained on invented errors. It learns the **transfer function**
from the raw CV segmentation (v117, 2021-06-11) to proofread ground truth (v1412):

- **False merge** (hard negative, label=0): one v117 root → 2+ current roots.
  CV merged two cells; the junction between their synapse chains is a real boundary.
- **False split** (hard positive, label=1): 2+ v117 roots → same current root.
  CV split one cell; the junction across their chains is a valid same-cell path.

Both types are required. Merge-only training makes the model distrust all junctions;
split positives force reliance on actual trajectory features (curvature, step length,
direction coherence).

---

## Key Files

| File | Purpose |
|------|---------|
| `neuronauts/path_dataset.py` | `fetch_cave_false_merge_chains`, `add_edit_history_examples`, `train_path_encoder` |
| `neuronauts/grammar.py` | `PathEdgeEncoder`, `MergeScorer`, `ArborEncoder` |
| `neuronauts/cell_graph.py` | `CellGNN`, `build_synapse_graph`, `partition_from_embeddings` |
| `neuronauts/assembly.py` | `gat_refine_connectivity`, `GlobalAssemblyGAT` |
| `neuronauts/run.py` | Full pipeline orchestration |
| `scripts/train.py` | All training and evaluation CLI commands |
| `data/boxes_30um/` | 252 cached CAVE boxes (synapse tables + root IDs) |

---

## Active Branch

All current work: `claude/review-pipeline-GrimS`

Recent commits:
- `6316ce6` — Fix split pair detection + short chain support in edit history examples
- `f49e799` — Add `min_synapses_per_root` parameter to fetch function
- `d117f7e` — Various CellGNN / path encoder fixes
