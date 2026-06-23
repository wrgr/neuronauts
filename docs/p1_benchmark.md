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
regardless of the model* — they are the benchmark. Closing the model gap requires
**training a partition model on the L2 substrate itself** (one node per L2 node,
with same-fragment edges restored), which is the recommended next step.

## Reproduce

```bash
# both substrates (synapse fetch is cached; L2 loads from cache/l2_world/p1_full.npz):
NEURONAUTS_L2_CACHE_DIR=$PWD/cache/l2_skeleton \
NEURONAUTS_SYNAPSE_CACHE_DIR=$PWD/cache/synapse \
PYTHONPATH=$PWD python3 scripts/p1_completeness_benchmark.py

# GT only, no model:
python3 scripts/p1_completeness_benchmark.py --no-model
```

The L2 world (533 neurons, 1.38M nodes, ~52 min to build) is cached to
`cache/l2_world/p1_full.npz` (git-lfs); reruns load in seconds. Raw run logs are
in `cache/l2_world/p1_synapse_result.log` and `cache/l2_world/p1_l2_result.log`.
P1 is also wired into `scripts/spatial_variance.py` as in-column eval region `P1`.
