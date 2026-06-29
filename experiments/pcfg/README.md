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

## Synapse-level correction (both directions): `synapse_correction.py`

`pcfg_partitions.py` models only false-splits (merges) at the *half-partition* level. The
`synapse_correction.py` module works at the **synapse level** and learns the full
correction operator `f(v117) -> proofread partition`, capturing **both** merges and splits
from a single label.

**Construction.** Each synapse-side is joined to itself across materializations through its
*immutable supervoxel* (`chunkedgraph.get_roots(supervoxel, timestamp=later)` — single-valued,
unlike `get_latest_roots` on a whole root, which forks exactly when a split happened). That
gives `(root_v117, root_later)` per side. The pairwise label is just **"same later root?"**:

| v117 relation | later relation | correction | stratum |
|---|---|---|---|
| different root | same root | **merge** (false-split corrected) | `merge` (cross-root) |
| same root | different root | **split** (false-merge corrected) | `split` (within-root) |
| same/same or diff/diff | — | none (stable) | — |

So the learned affinity `P(same later root | v117 features)` *is* the correction; XOR with the
v117 relation yields the edit. CV groups are connected components of the
(v117-root ↔ later-root) co-occurrence graph — the physical "cells" — to avoid the
cell-identity leakage that inflated the berlin risk AUC.

```bash
# offline sanity check (no token): injects known false-merges + false-splits
python -m experiments.pcfg_synapse_partitions.run_synapse_correction --synthetic

# real data (needs a CAVE token): v117 synapses in the proofread column -> later roots
python -m experiments.pcfg_synapse_partitions.run_synapse_correction \
    --token $CAVE_TOKEN --later-version 1718 --n-boxes 6 --side-um 30
```

Reports grouped-CV AUC overall and per stratum (`merge` / `split`), each vs a permutation
null. Offline checks live in `tests/test_synapse_correction.py`.

## Files

| File | Purpose |
|------|---------|
| `pcfg_partitions.py` | Core grammar: tokenize, bigram\_features, cond\_entropy, partition\_features, extract\_partitions, build\_merge\_pairs |
| `run_experiment.py` | CLI: loads BoxCache + remap TSV, builds dataset, runs CV (merge-only, half-partition level) |
| `synapse_correction.py` | Synapse-level cross-version join + both-direction (merge **and** split) pair dataset & features |
| `run_synapse_correction.py` | CLI: CAVE fetch (or `--synthetic`), grouped-CV eval per stratum vs permutation null |
| `README.md` | This file |

## Running on live CAVE (`scripts/v117_pcfg.py`) — setup & troubleshooting

This runs **here, against live CAVE** (no laptop needed). Typical invocation:

```bash
python scripts/v117_pcfg.py --token $CAVE_TOKEN --n-boxes 1 --side-um 20 --use-learned
```

### One-time token setup
Save your token once so every CAVE client picks it up automatically:

```python
from caveclient import CAVEclient
CAVEclient(server_address="https://global.daf-apis.com").auth.save_token(token="<YOUR_TOKEN>", overwrite=True)
```

(You can still pass `--token`/`CAVE_TOKEN`; saving it just avoids cold-start auth flakiness.)

### Intermittent `bad_auth_header` on the first CAVE call
You may occasionally see the run fail immediately with:

```
400 Client Error: BAD REQUEST ... /info/api/v2/datastack/full/minnie65_public
{"error": "bad_auth_header", "message": "Authorization header must begin with 'Bearer'"}
```

This is **not** a bad token and not a code bug: the header we send *is* a correct
`Bearer <token>`, and the identical request succeeds on retry — the public
endpoint intermittently rejects a valid request during datastack resolution.
`neuronauts/fetch.py` now resolves the datastack info up front inside a backoff
retry (`_build_caveclient`), so transient `bad_auth`/connection blips
auto-recover. If a run still dies at that step, just **re-run it, or start a
fresh session** — it clears on its own.

### Slow / hanging synapse query
The default fetch is an unfiltered spatial query over the ~337M-row
`synapses_pni_2` table. With stable connectivity that completes; if it hangs,
**re-run in a fresh session**. (An experimental `--lightweight` path exists —
dense seg-cutout root enumeration + root-filtered lookup — but its synapse
**counts are not yet validated** against the spatial query, so don't trust its
numbers until that check is done.)

### Rule of thumb
A failed CAVE run here is almost always transient connectivity, *or* a setup
mistake on our side — never assume the service is "down" or "throttling" without
checking (CAVE returns no `429`/`Retry-After`; a `bad_auth` 400 is the transient
above). A fresh session is the fastest reset.
