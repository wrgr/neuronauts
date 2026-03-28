# CellGNN Training Pipeline — Open Items

Status as of 2026-03-28.  Items are ordered by dependency — complete
earlier items before later ones.

---

## 1. Real Data Acquisition

The CellGNN has been validated on synthetic boxes.  Training on real
MICrONS data is the critical next step.

### 1a. CAVE-only boxes (no auth token required)

The fastest path to real data.  Public `minnie65_public` datastack
does **not** require a CAVE token — only `caveclient` must be installed.

```bash
pip install caveclient

# Fetch 80 synapse-seeded boxes, skip EM volume (~30 min)
python scripts/train.py build-dataset \
    --cache-dir data/boxes_cave \
    --n-boxes 80 \
    --strategy synapse-seeded \
    --no-em \
    --min-positive-pairs 5 \
    --cave-version 1412

# Train CellGNN on cached data
python scripts/train.py train-cell-gnn \
    --cache-dir data/boxes_cave \
    --epochs 50 \
    --cell-gnn-output models/cell_gnn_cave.pt
```

For environments without outbound network, use the helper:

```bash
python scripts/fetch_cave_boxes.py \
    --cache-dir data/boxes_cave \
    --n-boxes 80 \
    --no-em \
    --min-positive-pairs 5
```

This writes a self-contained `BoxCache` that can be copied to an
offline machine for training.

### 1b. Static archived data (fully offline)

Download v1078 materialization CSVs once, then select boxes offline:

```bash
python -m neuronauts.synapse_root_counts_static \
    --version 1078 \
    --static-dir data/microns_static \
    --output run_logs/synapse_root_counts_v1078.tsv

python scripts/train.py build-dataset \
    --cache-dir data/boxes_static \
    --n-boxes 50 \
    --counts-tsv run_logs/synapse_root_counts_v1078.tsv \
    --nucleus-csv data/microns_static/v1078/nucleus_detection_v0.csv
```

### 1c. Proofread-core (highest quality, requires CAVE token)

Proofread regions have the richest structure and most reliable labels.
Requires `CAVE_TOKEN` environment variable or `--cave-token` flag.

```bash
# Sample proofread roots
python scripts/train.py build-dataset \
    --cache-dir data/boxes_proofread \
    --strategy proofread-core \
    --proofread-n-roots 30 \
    --proofread-radius-um 40 \
    --no-em \
    --cave-version 1412 \
    --cave-token "$CAVE_TOKEN"

# Optionally remap to a newer materialization for label refresh
python scripts/train.py remap-roots \
    --cache-dir data/boxes_proofread \
    --base-version 1412 --target-version 1718 \
    --output data/boxes_proofread/root_remap.tsv
```

---

## 2. Edit-History Integration

`edit_history.py` produces contrastive pairs from proofreader merges
and splits.  The wiring into `train_cell_gnn` is complete
(`--edit-pairs-tsv`), but no real edit-pair TSV has been generated yet.

### Next steps

- [ ] Run `python -m neuronauts.edit_history build-pairs` on a real
      cache to produce `edit_pairs.tsv`
- [ ] Train CellGNN with `--edit-pairs-tsv edit_pairs.tsv` and compare
      val F1 with/without edit supervision
- [ ] Tune `--edit-weight` (currently default 2.0)

---

## 3. Evaluation Against Beam-Search Baseline

The `evaluate` subcommand is implemented but hasn't been run on real
data with a grammar baseline.

### Next steps

- [ ] Run `evaluate --cell-gnn-checkpoint ... --grammar-checkpoint ...`
      on test split with real CAVE boxes
- [ ] Compare F1/precision/recall head-to-head
- [ ] If GNN wins on precision but loses on recall (or vice versa),
      consider an ensemble: GNN embeddings as additional edge features
      in the grammar pipeline

---

## 4. Hyperparameter Sweep on Real Data

The `sweep` subcommand is ready.  On synthetic data all configs converge
to similar val F1 due to small box count.  Real data (50+ boxes) should
differentiate.

### Priority sweep axes

| Parameter              | Range to try          | Why                                      |
|------------------------|-----------------------|------------------------------------------|
| `proximity_radius_nm`  | 2000, 5000, 10000     | Controls graph density and K-hop reach   |
| `partition_threshold`   | 0.3, 0.5, 0.7        | Precision/recall tradeoff at clustering  |
| `d_model`              | 32, 64, 128           | Model capacity                           |
| `n_layers`             | 2, 3, 4              | Message-passing depth (K-hop reach)      |

```bash
python scripts/train.py sweep \
    --cache-dir data/boxes_cave \
    --d-models "32,64,128" \
    --n-layers-list "2,3,4" \
    --proximity-radii "2000,5000,10000" \
    --partition-thresholds "0.3,0.5,0.7" \
    --epochs 30 \
    --best-output models/cell_gnn_sweep_best.pt
```

---

## 5. Scale Testing on Larger Boxes

`scale-test` is implemented.  Remaining work:

- [ ] Test on boxes with 200+ synapses (`--box-side-um 15` or higher
      `--min-synapses`)
- [ ] Profile with `--proximity-radius-nm 2000` vs `10000` to measure
      graph density impact
- [ ] Identify the synapse count where O(N^2) cosine-similarity
      clustering in `partition_from_embeddings` becomes a bottleneck
      (likely >500 synapses); consider switching to ANN or sparse
      methods at that point

---

## 6. Test Coverage Gaps

### Critical (no existing tests)

- [ ] `cmd_train_cell_gnn` integration test (full function with mock cache)
- [ ] `cmd_evaluate` integration test (checkpoint load → eval → JSON output)
- [ ] `cmd_sweep` integration test (2-config grid → best model saved)
- [ ] `cmd_scale_test` integration test (tracemalloc profiling)

### Medium priority

- [ ] Edit-pairs TSV loading / malformed input handling
- [ ] `partition_from_embeddings` with threshold=0.0 and threshold=1.0
- [ ] `spatial_train_val_test_split` with 1–2 boxes (degenerate case)
- [ ] `cell_graph_train_step` with out-of-range edit pair indices
- [ ] `train_cell_gnn` with `edit_weight=0.0` (should behave like no
      edit pairs)
