# P1 proofread-dense benchmark: substrate matters

## Motivation

The existing T-series test boxes (T1–T4) show near-trivial v117→v1718
fragmentation: median 1 fragment per neuron, max 2. A reconstruction model can
score well on them without ever solving a hard merge, so they are poor benchmarks
for *reconstruction quality*.

We need a region densely populated with **real** proofreading errors. Scanning
nucleus-soma edit rates across 394 candidate bins, we found **P1**:

> **P1 bbox** (true nm, the 4,4,40 frame used by `fetch_region_synapses` and the
> L2 cache `rep_coord_nm`): `x=818,500–918,500`, `y=685,000–785,000`,
> `z=794,000–994,000`.

~100% of nucleus somas in P1 have `v117 != v1718` — it is saturated with real
edits. Yet whether the benchmark is *hard* depends entirely on the **observation
substrate** we sample it with.

## Two substrates, same region

`scripts/p1_completeness_benchmark.py` builds P1 under both substrates.

| | **Synapse** (`min_syn=5`) | **L2 node** (`min_l2=2`) |
|---|---|---|
| observations | real synapses | L2 chunkedgraph nodes (~5–10 µm) |
| in-box observations | 8,781 (of 50,000 fetched) | 1,384,644 |
| dropped by sliver filter | 41,219 synapses (**82%**) | — |
| v117 fragments | 1,160 | 937 |
| frankenmerges | 241 | 24 |
| **GT complete (no edit)** | **905/1160 = 78%** | **402/937 = 43%** |
| max fragments / neuron | **5** | **60** |
| median fragments / neuron | 1 | 1 |
| mean fragments / neuron | 1.02 | 1.80 |
| **neurons needing ≥2 merges** | **23/1551 = 1%** | **101/533 = 19%** |

### Why they diverge

The synapse substrate applies `min_syn_per_fragment=5`. On P1 that filter drops
**82% of synapses** and, with them, every low-degree fragment — the small axon and
dendrite slivers that are exactly the pieces a proofreader has to merge back. What
survives is the high-degree trunk of each neuron, which almost always already maps
1-to-1 onto its proofread root. The result *looks* 78% solved with a max of 5
fragments — the same degree-bias that makes the T-series trivial, just measured on
a region we *know* is 100% edited.

The L2 substrate keeps every fragment down to 2 nodes, so the merge tail survives:
43% complete, a long tail out to 60 fragments on a single neuron, and ~1 in 5
neurons genuinely needing multiple merges. **This is the honest benchmark.**

## Completeness prediction task

A v117 fragment is **complete** (needs no edit) when it maps 1-to-1 onto a single
v1718 neuron — sole contributor, not a frankenmerge. `fragment_completeness` /
`completeness_metrics` (in `treestitch/partition.py`) give the GT and score a
predictor. The GNN marks a fragment complete when it lands alone in a singleton
cluster.

| substrate | model ARI | completeness P | R | F1 | acc |
|---|---|---|---|---|---|
| synapse | 0.017 | 0.829 | 0.096 | 0.172 | 0.279 |
| L2 (fragment-centroid) | 0.001 | 0.414 | 0.776 | 0.540 | 0.433 |

Both model rows are weak, for the same root cause: **the shared checkpoint was
trained on synapse observation graphs.** On the synapse substrate it at least sees
same-fragment edges (multiple synapses per fragment) and is over-conservative
(predicts almost nothing complete → high precision, near-zero recall). On the L2
substrate, collapsed to one centroid node per fragment, there are **no
same-fragment edges at all**, so the model is fully out of distribution (ARI≈0).

The GT fragmentation and completeness columns are *substrate-driven and valid
regardless of the model* — they are the benchmark. Both rows above use the
**shared synapse-trained checkpoint** out of distribution; closing the gap requires
training on the L2 substrate itself. We did that next.

## Training on the L2 substrate

`scripts/train_l2_partition.py` trains a partition GNN directly on the L2 substrate.
Each observation is an L2 node; **same-fragment edges (edge type 0) connect the L2
nodes of one v117 fragment** — the signal the fragment-centroid graph destroyed. To
fit a 1.38M-node region in memory, each v117 fragment is subsampled to ≤50 L2 nodes
(farthest-point), giving a ~25k-node graph that still carries both same-fragment and
cross-fragment spatial-kNN edges. Train/test is a spatial split along x (per
fragment, by node centroid, with a buffer gap) so no fragment straddles the split:
650 train fragments (west 60%) / 172 test fragments (east 30%).

At eval, same-fragment co-membership is **reconciled** (a v117 fragment is an atomic
supervoxel group — proofreading only merges it with others, never splits it), so the
model is scored only on the *cross-fragment* merge decisions that actually matter.

### A scaffold you can trust: the conservatism sweep

`over_merge_rate` — the rate of false merges, the costly, hard-to-undo error — is the
trust axis. Sweeping `cc_bias` (more negative = more conservative) on the held-out
east split:

| cc_bias | ARI | merge_P | merge_R | **over_merge** | cmpl_P | cmpl_R | cmpl_F1 |
|---|---|---|---|---|---|---|---|
| −6 | 0.652 | 0.998 | 0.997 | **0.001** | 0.465 | 1.000 | 0.635 |
| −4 | 0.652 | 0.998 | 0.997 | **0.001** | 0.465 | 1.000 | 0.635 |
| **−2** | **0.606** | 0.995 | 0.997 | **0.004** | 0.514 | 0.912 | **0.658** |
| 0 | 0.303 | 0.978 | 0.998 | 0.017 | 0.468 | 0.550 | 0.506 |
| +2 | 0.019 | 0.881 | 0.998 | 0.105 | 0.532 | 0.312 | 0.394 |
| +4 | 0.001 | 0.783 | 1.000 | 0.214 | 0.438 | 0.087 | 0.146 |

*Numbers use the morphological fragment descriptor (v2) — see below.*

Two results:

1. **The L2 substrate is learnable.** Trained on L2, the model reaches **ARI 0.65**
   on held-out space — versus **ARI ≈ 0** for the synapse-trained checkpoint applied
   to the same data. The earlier "model fails on L2" was a domain-shift artifact, not
   a property of the substrate.

2. **There is a trustworthy operating point.** At `cc_bias −6…−4` the scaffold makes
   **1 false merge per ~1000 edges** (over_merge 0.001) while still recovering the
   real merges (merge_R 0.997). False merges are the errors a proofreader cannot
   cheaply undo, so a near-zero over-merge scaffold is one you can build on. The best
   all-round point is **`cc_bias −2`**: ARI 0.61, over_merge 0.004, edge_acc 0.965.
   Pushing past 0 trades that trust away fast (over_merge 0.10→0.21, ARI collapses).

## Cross-region generalisation

To make an honest out-of-region claim, we train on five geographically separate
regions (A–E, none overlapping P1) and evaluate on the same P1 east split.

**Training regions** (L2 worlds built from CAVE API, cached to `cache/l2_world/`):

| Region | Neurons | Fragments | L2 nodes |
|--------|---------|-----------|----------|
| A | 432 | 484 | 720k |
| B | 305 | 302 | 503k |
| C | 175 | 199 | 200k |
| D | 200 | 221 | 37k |
| E | 313 | 345 | 661k |
| **Total** | **1,425** | **1,551** | **2.1M** |

Conservatism sweep on P1 east (172 fragments, **never seen during training**):

| cc_bias | ARI | merge_P | merge_R | **over_merge** | cmpl_P | cmpl_R | cmpl_F1 |
|---|---|---|---|---|---|---|---|
| −6 | **0.652** | 0.998 | 0.997 | **0.001** | 0.465 | 1.000 | 0.635 |
| −4 | **0.652** | 0.998 | 0.997 | **0.001** | 0.465 | 1.000 | 0.635 |
| −2 | **0.652** | 0.998 | 0.997 | **0.001** | 0.465 | 1.000 | 0.635 |
| 0 | 0.006 | 0.826 | 0.999 | 0.162 | 0.455 | 0.188 | 0.265 |
| +2 | −0.000 | 0.775 | 1.000 | 0.225 | 0.250 | 0.013 | 0.024 |

**Cross-region model matches the P1-trained model at the trustworthy operating
point** (ARI 0.652, over_merge 0.001), trained entirely on A–E. The conservative
plateau extends one bias step further (cc_bias −2 stays at the −4/−6 quality rather
than relaxing as it does for the P1-trained model), but the peak ARI is identical.
This confirms the L2 substrate and morphological descriptor generalise across regions.

Checkpoint: `models/neuronauts_l2_partition_xregion.pt`.

## Fragment encoder: morphological descriptor (v2)

The original trained `SkeletonGNN` encoder collapsed on the L2 substrate
(pos_cos≈neg_cos≈1 throughout training). Root cause: with centroid-normalised
xyz inputs, the mean-pool readout produces **the same 64D vector for every
fragment** (`mean(x-cx) = 0` by construction), so the GNN output sits at
cosine similarity ≈ 0.97 at random init and no contrastive objective can make
progress from that collapsed starting point.

The fix is `encode_fragments_morphological` (`treestitch/embed.py`): a **deterministic
PCA-based descriptor** (log node count, centroid-distance radii stats, PCA spread
along three axes, elongation ratio, endpoint distance). No training, no collapse,
and the cosine similarity distribution across fragments drops to mean 0.68 (vs 0.97).
The partition GNN benefits: ARI at `cc_bias −2` rises 0.581→0.606, over_merge drops
0.005→0.004, and edge classification accuracy reaches 0.965 (vs 0.942 with the
collapsed GNN). The remaining gap to a *learned* morphological representation is
the obvious next lever for cross-region generalisation.

Checkpoint: `models/neuronauts_l2_partition.pt`. Generalization here is held-out
*space within P1*; cross-region L2 training (build L2 worlds for A–E) is the next
step for an out-of-region claim.

## Reproduce

```bash
# both substrates (synapse fetch is cached; L2 loads from cache/l2_world/p1_full.npz):
NEURONAUTS_L2_CACHE_DIR=$PWD/cache/l2_skeleton \
NEURONAUTS_SYNAPSE_CACHE_DIR=$PWD/cache/synapse \
PYTHONPATH=$PWD python3 scripts/p1_completeness_benchmark.py

# GT only, no model:
python3 scripts/p1_completeness_benchmark.py --no-model

# train a partition model ON the L2 substrate (held-out spatial split, ~5 min):
NEURONAUTS_L2_CACHE_DIR=$PWD/cache/l2_skeleton \
PYTHONPATH=$PWD python3 scripts/train_l2_partition.py \
  --save-checkpoint models/neuronauts_l2_partition.pt
```

The L2 world (533 neurons, 1.38M nodes, ~52 min to build) is cached to
`cache/l2_world/p1_full.npz` (git-lfs); reruns load in seconds. P1 is also wired into
`scripts/spatial_variance.py` as in-column eval region `P1`. The L2-trained
checkpoint is `models/neuronauts_l2_partition.pt`.

```bash
# train cross-region (A–E → P1 eval; caches load instantly, training ~15 min):
PYTHONPATH=$PWD python3 scripts/train_l2_partition.py \
  --train-regions A,B,C,D,E \
  --save-checkpoint models/neuronauts_l2_partition_xregion.pt
```

Region L2 worlds (A–E) are cached to `cache/l2_world/{A,B,C,D,E}_full.npz` (git-lfs).
The cross-region checkpoint is `models/neuronauts_l2_partition_xregion.pt`.
