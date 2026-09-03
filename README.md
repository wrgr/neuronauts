# neuronauts

`neuronauts` reconstructs neurons from electron microscopy connectome data. Given CAVE synapse tables and skeleton fragments, it learns to partition synapses by their parent neuron and assembles fragment trees into globally consistent neurons.

> **New here?** [`docs/MAP.md`](docs/MAP.md) is the orientation page: what is live,
> what is superseded but real, and what is archived synthetic-substrate history.
> The numbers in this README predate the registered experiment program in
> [`results/RESULTS.md`](results/RESULTS.md); the MAP says which is which.

Two research tracks are active and have now cleared their initial viability bars on real MICrONS minnie65 data:

| Track | What it learns | Key result |
|---|---|---|
| **treestitch** (global partition) | SkeletonGNN embeds fragments; PartitionGNN clusters synapses by neuron | ARI=0.752, merge_P=0.951, Bars 1+2 **pass** (out-of-sample, leak-fixed) |
| **grammar** (merge scoring) | PathEdgeEncoder + MergeScorer scores pairwise fragment merges | Cross-region holdout AUC=**0.816** [0.754, 0.874] |

The earlier **CellGNN** baseline (synapse graph → cell assignment) is in `neuronauts/cell_graph.py` and remains the default `train-cell-gnn` CLI target; test F1=0.272.

---

## Repository layout

| Directory | What is in it |
|---|---|
| [`neuronauts/`](neuronauts/) | The package. `harness/` is the substrate, `metrics/` the one metric home, `experiments/` the registered program, `report/` the runner and renderer, `data/` the CAVE access layer. |
| [`scripts/`](scripts/README.md) | **Only** what builds or refreshes current data, plus the minimal-repro probes and the `train.py` command line. |
| [`experiments/`](experiments/README.md) | **Only** live research threads: `pcfg/`, `fingerprints/`, `minnie_column/`, `soma_graph/`, `root_neighborhood/`. |
| [`treestitch/`](treestitch/) | The global-partition package behind the top result above. |
| [`tests/`](tests/) | The default suite. |
| [`attic/`](attic/README.md) | **The archive.** Everything superseded, in ten subdirectories, each with a README saying what it was and what replaced it. Excluded from `pytest`; nothing is deleted. |
| [`results/`](results/RESULTS.md) | One directory per experiment. A run without a row in `RESULTS.md` does not exist. |
| [`docs/`](docs/MAP.md) | `MAP.md` first, then `threads/` for per-thread state and `archive/` for dated snapshots. |

---

## Active pipeline: treestitch

```
CAVE skeleton cache  (cache/l2_skeleton/*.npz)
  → [data/fragments.py]   skeleton_to_fragment, extract_fragments_for_region
  → [represent/dna.py]    SkeletonGNN encodes each fragment (centroid-normalised xyz+r)
  → [treestitch/]         global synapse graph (k-NN) → PartitionGNN → assembly
  → [treestitch/risk.py]  risk-weighted decisions: CONFIDENT_MERGE / REVIEW / ABSTAIN
```

Entry point:

```bash
python attic/prior_results/train_l2_partition.py --help
python attic/prior_results/multi_region_train.py --help   # multi-region with seam-buffer leak fix
```

## CellGNN baseline pipeline (box-cache route)

```
CAVE synapse table
  → [Stage 0] Box cache (data/boxes_30um/, 247 × 30 µm boxes)
  → [Stage 1] Edit-pair mining (v117→v1412 lineage)
  → [Stage 2] Path encoder pretraining
  → [Stage 3] CellGNN training
  → [Stage 4] Line-graph F1 evaluation
```

```bash
python scripts/train.py --help   # all stages
```

---

## Current results  *(updated 2026-06-30)*

### treestitch (Phase 2.11 — leak-fixed, out-of-sample)

| Metric | Value |
|--------|-------|
| Out-of-sample ARI | **0.752** |
| Out-of-sample merge_P | **0.951** ✓ Bar 1+2 pass |
| Out-of-sample merge_R | 0.865 |
| is_tree | 1.000 (Kruskal guarantee) |
| cable_um median | 3 579 µm (biologically plausible) |
| Bar 3 (fk_split > 0.50) | ✗ 0.000 — frankenmerge detection not yet transferable OOC |

Protocol: train on 3 in-column regions (A/B/C), test on a spatially disjoint region; 50 µm seam buffer + root dedup (no boundary leakage). See `STATUS.md` for full per-phase progression.

### Grammar cross-region holdout (Phase 2 / vibrant)

> Cross-region ROC-AUC = **0.816**, 95% CI [0.754, 0.874] — recovers 60% of true cross-region merges (F1=0.57 vs 0.00 for baseline). Robust across seeds (0.78–0.82).

See `experiments/pcfg/HOLDOUT_RESULTS.md` for protocol and per-threshold breakdown.

### CellGNN baseline

| Model | Test line-graph F1 |
|-------|--------------------|
| `cell_gnn_seg.pt` (6-feat + seg signal) | **0.272** @ t=0.99 |
| `cell_gnn_5feat.pt` (scalar features) | 0.269 @ t=0.99 |
| `cell_gnn_real.pt` (first no-EM baseline) | 0.264 @ t=0.99 |

---

## Prerequisites

```bash
pip install -r requirements-dev.txt    # or: make setup

# CAVE token (needed for live fetches; not needed to train on an existing cache)
mkdir -p ~/.cloudvolume/secrets
echo '{"token": "YOUR_TOKEN"}' > ~/.cloudvolume/secrets/cave-secret.json
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md#dev-setup) and
[`docs/CAVE_AUTHENTICATION_SETUP.md`](docs/CAVE_AUTHENTICATION_SETUP.md).

---

## CellGNN runbook (stage-by-stage)

### Stage 0 — Box cache

247 boxes are cached at `data/boxes_30um/` (494 files: json + npz per box).

```bash
ls data/boxes_30um/ | wc -l   # expect ~494
```

### Stage 1 — Edit-pair mining (preferred: from cache)

```bash
python scripts/train.py fetch-cave-edits-from-cache \
  --cache-dir data/boxes_30um \
  --min-synapses-per-root 8 \
  --output-tsv data/cave_edit_pairs_v3.tsv \
  --output-chains data/cave_edit_chains_v3.npz
```

### Stage 2 — Path encoder pretraining

```bash
python scripts/train.py train-path-encoder \
  --cache-dir data/boxes_30um \
  --epochs 10 \
  --edit-pairs-tsv data/cave_edit_pairs_v3.tsv \
  --edit-chains-npz data/cave_edit_chains_v3.npz \
  --max-examples-per-epoch 50000 \
  --output models/path_encoder_v3.pt \
  --seed 42
```

### Stage 3 — CellGNN

```bash
python scripts/train.py train-cell-gnn \
  --cache-dir data/boxes_30um \
  --epochs 10 \
  --cell-gnn-output models/cell_gnn_30um_v1.pt \
  --n-layers 2 \
  --seed 42
```

With pretrained path encoder (recommended):

```bash
python scripts/train.py train-cell-gnn \
  --cache-dir data/boxes_30um \
  --epochs 10 \
  --path-encoder-checkpoint models/path_encoder_v3_ep8.pt \
  --pretrained-path-emb-dim 16 \
  --cell-gnn-output models/cell_gnn_v3.pt \
  --n-layers 2 \
  --seed 42
```

### Stage 4 — Evaluate

```bash
python scripts/train.py evaluate \
  --cache-dir data/boxes_30um \
  --cell-gnn-checkpoint models/cell_gnn_v3.pt \
  --split test
```

---

## Run tests

```bash
pytest
pytest -m 'not legacy'   # skip the quarantined v1 agent/membrane stack
```

---

## Saved checkpoints

Curated checkpoints are tracked in `models/` and catalogued in [`models/README.md`](models/README.md).

| File | Thread | Headline metric |
|------|--------|-----------------|
| `neuronauts_l2_partition.pt` | treestitch | in-column ARI ≥ 0.80, merge_P ≥ 0.99; out-of-sample merge_P=0.951 |
| `neuronauts_l2_partition_xregion.pt` | treestitch | cross-region variant |
| `grammar_cave_real_50.pt` | grammar | val merge acc **87.2%** — 40 real CAVE boxes, 50 ep |
| `shared_grammar_raw_skel_gat50e.pt` | grammar | `raw_delta3+skeleton` + GAT, 50 ep |
| `cell_gnn_seg.pt` | cell_assignment | test line-graph F1 **0.272** @ t=0.99 |
| `cell_gnn_real.pt` | cell_assignment | first real-CAVE no-EM baseline |
| `shared_grammar_root_neighborhood_run001.pt` | root_neighborhood | grammar on proofread-anchor cache |

Path-encoder and training-run checkpoints are produced locally; write new runs to `models/scratch/` (git-ignored). Add a curated keeper with `git add -f models/<name>.pt` and a row in `models/README.md`.

---

## Key files

| Module | Purpose |
|--------|---------|
| `treestitch/` | Global tree stitching: embed, partition GNN, assembly, calibration, risk, NGL export |
| `neuronauts/represent/dna.py` | `SkeletonGNN`, `TreeDNAEncoder`, `encode_fragments` |
| `neuronauts/represent/enrich.py` | `build_synapse_dna_matrix`, `evaluate_dna_auc` |
| `neuronauts/data/fragments.py` | `skeleton_to_fragment`, `extract_fragments_for_region` |
| `neuronauts/data/lineage.py` | L2 skeleton cache + provenance |
| `neuronauts/assemble/edge_partition.py` | `train_edge_partition_gnn` (hard-negative mining) |
| `neuronauts/cell_graph.py` | `CellGNN`, `build_synapse_graph`, `train_cell_gnn` |
| `neuronauts/grammar.py` | `PathEdgeEncoder`, `MergeScorer`, `ArborEncoder` |
| `neuronauts/dataset_builder.py` | `BoxCache`, box fetching and caching |
| `neuronauts/fetch.py` | CAVE synapse/skeleton fetch (query + bulk routes) |
| `neuronauts/schemas.py` | Typed contracts: `Region`, `Fragment`, `NeuronHypothesis` |
| `scripts/train.py` | All CellGNN/grammar training and evaluation CLI |
| `attic/prior_results/train_l2_partition.py` | treestitch L2 partition training |
| `attic/prior_results/multi_region_train.py` | Multi-region train with seam-buffer leak fix |
| `attic/prior_results/spatial_variance.py` | Spatial variance + OOC shape study (7 test bboxes) |
| `data/boxes_30um/` | 247 cached CAVE boxes |
| `cache/l2_skeleton/` | L2 skeleton NPZ cache (PROVENANCE.json tracked; NPZs local-only) |

---

## Research threads

The work is organized as research threads; see [`experiments/README.md`](experiments/README.md) for the full index with status, entry points, and checkpoints.

| Thread | Status | Description |
|--------|--------|-------------|
| **tree_dna** | active frontier | Skeleton GNN → global synapse partition (treestitch); Phase 2.11 complete |
| **grammar** | active | Cross-region merge scoring; holdout AUC=0.816 |
| **cell_assignment** | active (baseline) | CellGNN; test F1=0.272 |
| **error_correction** | active | False-merge/split supervision from v117→v1412 lineage |
| **pcfg** | active | Non-neural PCFG synapse partitions |
| **fingerprints** | incubating | Neuron connectivity signatures |
| **root_neighborhood** | incubating | Grammar on proofread-anchor cache |
| **soma_graph** | incubating | Soma-seeded graph assembly |
| **minnie_column** | active (data) | Minnie65 column data pipeline |
| **topology** | optional | Topology validator / atomicity checks |

The longer-range direction (global assembly roadmap) is [`docs/roadmap_global_assembly.md`](docs/roadmap_global_assembly.md).

---

## Data

- **Box cache**: 247 × 30 µm CAVE boxes, v1412 root IDs, ~1.68M synapses total (148/30/49 train/val/test, spatial split, seed=42)
- **Edit pairs v3**: 25 860 pairs from 2002 v117 roots (25 444 false-merge, 416 false-split)
- **L2 skeleton cache**: `cache/l2_skeleton/` — per-fragment NPZ archives with provenance; fetched once via `scripts/warm_cache.py`, reused across runs
- **Cell type table**: `aibs_metamodel_celltypes_v661_merged.csv.gz` (19 735 L2/3 pyramidals at v1412, used for within-type ablations)

---

## Package surface policy

`neuronauts.__init__` re-exports only the active no-EM training/evaluation pipeline APIs. Legacy experimental modules are available via direct imports (`neuronauts.legacy.*`). See `CONTRIBUTING.md` for the active-vs-legacy convention.

**Not used in the default training loop** (present and tested, not wired in):
- EM volume fetching / Sobel membrane fields
- Agent simulation (`legacy/vectorized.py`, `legacy/fields.py`)
- Topology validator (`topology_model.py`, `topology_dataset.py`)
- Neuroglancer inspector (`scripts/inspect_pipeline.py`)
