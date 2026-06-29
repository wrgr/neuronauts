# PCFG Synapse Partition Experiment

> **Status: active** (experiment thread). One of the project's research threads —
> see the index in [`../README.md`](../README.md). A cheap, non-neural baseline
> for the merge decision the [grammar](../grammar/README.md) thread learns.

Applies a Berlin-style bigram grammar to **synapse half-partitions** to learn
`f(v117) → v18xx` merge prediction.  No neural network, no EM volume, no agent simulation.

## Motivation

The Berlin grammar experiment models proofreading sessions as a language with tokens
{N, S, A, O} and achieves AUC 0.95 for expert/student separation using a simple bigram
(first-order Markov) model.  It is data-constrained (~15 labeled users), so expressive
models (HMM, Transformer) collapse.

Neuronauts has the inverse situation: **enormous ground truth** (1.68M synapses across
247 cached 30 µm boxes, v117→v1412 root remap TSV) but the current grammar trains a
TorchPathEncoder that requires expensive agent simulation.

This experiment asks: **can a cheap bigram grammar over synapse half-partitions match
the TorchPathEncoder’s 85.6% merge accuracy?**

## Concept

**Synapse half-partition** — all synapses on *one side* (pre or post) of a given v117
root ID.  Each synapse has two half-sides; treating them independently halves ambiguity.

**Token alphabet** (analogous to Berlin’s N/S/A/O action tokens):

| Token | Meaning |
|-------|---------|
| F | Forward — step projects along PCA1 main axis |
| B | Backward — step projects against PCA1 |
| L | Lateral-left — \|PCA1\| ≤ threshold, PCA2 ≥ 0 |
| R | Lateral-right — \|PCA1\| ≤ threshold, PCA2 < 0 |

Synapses are sorted by PCA1 (main arbor axis); steps between consecutive synapses are
tokenized.  A clean single-cell arbor mostly produces F→F transitions; a false merge
produces direction reversals (F→B).

**Grammar features per half-partition (17-dim)**:
- 16-dim normalized bigram transition matrix P(next\|current)
- 1-dim conditional entropy H(next\|current)

**Pairwise merge-pair features (35-dim)**:
- concat(features\_a, features\_b, \[log1p(centroid\_distance\_nm)\])

**Ground truth (positive pairs)**: two v117 roots that map to the *same* v18xx root are
real false-splits corrected by proofreading.  Negative pairs have different v18xx roots.

## Comparison to Berlin

| Aspect | Berlin grammar | This experiment |
|--------|---------------|----------------|
| Input tokens | N/S/A/O action types | F/B/L/R step directions |
| Features | bigram(16) + cond\_entropy(1) | same |
| Classifier | LogisticRegression + LOO | LogisticRegression + k-fold |
| Data scale | ~15 users (data-constrained) | thousands of partition pairs |
| Target | expert vs. proto-expert | merge vs. no-merge (v18xx GT) |
| AUC | 0.95 | TBD |

The larger dataset means HMM or trigram models may succeed here where they failed in Berlin.

## Usage

### Prerequisites

Build a v117 synapse cache and root remap TSV (network access to CAVE required once):

```bash
# 1. Build synapse-only cache at v117 materialization
python scripts/train.py build-dataset \
  --cache-dir data/boxes_v117 \
  --cave-version 117 \
  --n-boxes 100 \
  --no-em

# 2. Compute root remap v117 → v1412
python scripts/train.py remap-roots \
  --cache-dir data/boxes_v117 \
  --base-version 117 \
  --target-version 1412 \
  --output data/boxes_v117/root_remap_v117_to_v1412.tsv
```

### Run

```bash
python experiments/pcfg/run_experiment.py \
  --cache-dir data/boxes_v117 \
  --root-remap-tsv data/boxes_v117/root_remap_v117_to_v1412.tsv \
  --side both \
  --min-synapses 4 \
  --cv-folds 5 \
  --verbose
```

### Expected output

```
Loading root remap from data/boxes_v117/root_remap_v117_to_v1412.tsv ...
  XXXXX root ID mappings loaded
Processing N boxes from data/boxes_v117 ...
Extracted XXXXX half-partitions from N boxes
Building merge pair dataset ...
  XXXXX pairs  (XXX positive, XX.X%)

PCFG synapse partition grammar -- merge prediction
(n=XXXXX pairs from N boxes, XX.X% positive):
  bigram-syntax (16+16 feats)           CV AUC = 0.XX
  bigram + entropy (17+17 feats)        CV AUC = 0.XX
  bigram + entropy + dist (35 feats)    CV AUC = 0.XX

reference: Berlin proofreader bigram AUC = 0.95  (n=15, LOO)
           Neuronauts PathEncoder merge acc = 0.856  (85 boxes)
```

### Smoke test (no CAVE access needed)

```bash
python -c "
from experiments.pcfg.pcfg_partitions import (
    tokenize, bigram_features, cond_entropy, BIGRAM_DIM
)
import numpy as np
rng = np.random.default_rng(0)
pts = rng.standard_normal((20, 3)) * 1000.0
toks = tokenize(pts)
bg = bigram_features(toks)
assert bg.shape == (BIGRAM_DIM,), bg.shape
assert abs(bg.sum() - 1.0) < 1e-9, bg.sum()
ent = cond_entropy(toks)
assert 0.0 <= ent <= 2.0, ent
print('smoke test passed -- tokens:', toks[:8])
"
```

## Files

| File | Purpose |
|------|---------|
| `pcfg_partitions.py` | Core grammar: tokenize, bigram\_features, cond\_entropy, partition\_features, extract\_partitions, build\_merge\_pairs |
| `run_experiment.py` | CLI: loads BoxCache + remap TSV, builds dataset, runs CV, prints results |
| `README.md` | This file |
