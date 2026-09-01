# bench_v1 baselines — the honest floor

Real MICrONS data, locked split, calibrated on val only, one run on test.
Reproduce with:

```bash
python scripts/baseline_bench_v1.py   # floors + ceiling
python scripts/model_bench_v1.py --epochs 60   # learned model
```

Dataset manifest sha256
`58f59da287b4331d2f01d10357b4f2b66e33aaae6421d13b6f96fed9d4dbc24f`
(base v117 → labels v1718). Machine-readable records are the `*.json` files
beside this one, each stamped via `neuronauts.results_schema`.

> **Corrected 2026-09-01.** An earlier version of this file reported numbers on
> a dataset built with `min_syn_per_fragment=3`, which silently discarded **87%
> of the candidate v117 roots and 68% of the true merge pairs** — the sliver and
> singleton population that *is* the confuser set. Those numbers described an
> easier problem than the real one and are superseded by everything below.
> See "The filter error" at the end. Old manifest hash was
> `f4185886…`; it is void.

## Read the cross-fragment columns, not the aggregate ones

Aggregate pairwise F1 is dominated by observation pairs that already share a
v117 root — correct for free, before any decision is made. EXP-056 called out
exactly this artifact (an atomic baseline scored 0.914 pair-F1 while resolving
nothing). So the numbers that matter are restricted to pairs whose **v117 roots
differ**: the pairs a merge decision actually has to get right.

## Test results (P1c, 20,000 observations, 12,287 v117 fragments, 12,445 true neurons)

| Method | ARI | pair F1 | cross-merge P | cross-merge R | cross joins predicted |
|---|---:|---:|---:|---:|---:|
| untouched v117 (do nothing) | **0.9610** | 0.9610 | n/a | 0.000 | 0 |
| proximity union-find (d=2 µm, val-calibrated) | 0.0002 | — | 4.3e-06 | 0.686 | 112,100,512 |
| EdgePartitionGNN + GAEC (cc_bias=1.0, val-calibrated) | 0.0017 | 0.0019 | 1.4e-05 | 0.441 | 22,211,225 |
| *oracle fragment ceiling (bound, not a method)* | *0.9706* | *0.9706* | *0.928* | *0.619* | *469* |

There are **703** true cross-fragment pairs to find in the test region.

**Nothing yet beats doing nothing.** Both methods that attempt merges score
essentially zero cross-merge precision and destroy ARI. On this dataset, as of
this run, the state of the art is the untouched v117 segmentation.

## What these say

**1. The do-nothing floor is already at ARI 0.9610.** Trusting v117 and making
no merges scores 0.9610 ARI and 0.9610 pair-F1 on test. Any headline ARI near
0.9 on this task means almost nothing on its own — it is what you get for
declining to act. This is the single most important calibration point for
reading past claims in this repo, and the reason the cross-fragment columns
exist.

**2. Naive spatial proximity is catastrophically imprecise.** At the radius the
val sweep selected (2 µm), the method predicts **112 million** cross-fragment
joins to recover 482 of 703 true ones — a precision of 4.3×10⁻⁶, and ARI
collapses to 0.0002. Beyond 3 µm it merges the entire region into a single
cluster. Neuropil is dense: synapses from unrelated neurons routinely sit within
2 µm, so proximity alone carries almost no evidence about identity.

This independently reproduces EXP-052's core failure on a clean, properly split
dataset: that run recovered 13 of 14 true pairs while predicting 496,510 joins.
Same pathology, now measured under a protocol with a real validation set.

**3. There is a real, well-posed gap for a learned model.** The reachable target
without splitting frankenmerged fragments is cross-merge precision 0.928 at
recall 0.619. So a model must move from `(n/a, 0.000)` toward `(0.928, 0.619)`.
That is the scoreboard.

**4. The oracle ceiling is not 1.0, and that is a label-noise result.** Even
with perfect fragment→neuron assignment, recall caps at 0.619 and precision at
0.928, because a frankenmerged v117 root can only receive one label. The
missing 40% of true pairs are unreachable by *any* method that treats v117
fragments as atoms — which is what EXP-056 showed geometry alone cannot fix.
Splitting fragments is a separate problem from joining them, and this bound
quantifies what it costs to ignore it.

## Val sweep (calibration; test was touched once, afterwards)

Proximity union-find on val (11,375 fragments, 136 true cross pairs):

| d (nm) | ARI | cross-merge P | cross-merge R | cross joins | clusters |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.9234 | 0.000 | 0.000 | 458 | 11,247 |
| 1,000 | 0.6193 | 0.000 | 0.000 | 10,463 | 9,885 |
| **2,000** | 0.0002 | 1.2e-06 | 0.552 | 61,307,842 | 3,004 |
| 3,000 | 0.0000 | 1.1e-06 | 1.000 | 126,704,993 | 243 |
| 5,000–12,000 | 0.0000 | 1.0e-06 | 1.000 | 131,915,269 | 1 |

Every radius is a bad operating point. At 500 nm and 1 µm the method makes a few
hundred joins and gets *none* right; from 2 µm up it swallows the region. There
is no threshold on raw proximity that works, which is the finding.

## The learned model, and why it fails here

`EdgePartitionGNN` + GAEC correlation clustering (the Phase 2.11 configuration)
trained on train only, `cc_bias` swept on val only, applied once to test.

The edge classifier **did** learn real signal: 60 epochs took edge accuracy to
0.953 with `p_pos` 0.725 against `p_neg` 0.143 — a genuine separation, not a
collapse. It still produces no usable partition:

| cc_bias (val) | ARI | cross joins | correct | clusters |
|---:|---:|---:|---:|---:|
| −1.0 | 0.8667 | 25 | 0 | 11,966 |
| 0.0 | 0.0141 | 1,302,637 | 4 | 9,875 |
| 1.0 | 0.0018 | 9,768,474 | 38 | 8,944 |
| 2.0 | 0.0007 | 23,340,174 | 60 | 7,341 |
| 3.0 | 0.0000 | 131,915,269 | 136 | 1 |

There is no operating point. At `bias=−1` it makes 25 joins and gets none right;
by `bias=1` it is making ~10 million joins for 38 correct; at `bias=3` it has
merged the region into one cluster. The val sweep picks 1.0 only because a
vanishing F1 beats a zero one.

It is worth being precise about the one positive signal: the model's test
precision (1.4×10⁻⁵) is about **3× better than proximity's** (4.3×10⁻⁶) at
comparable recall, so it is extracting *some* information beyond distance. That
is four orders of magnitude short of usable, and it does not change the verdict.

**The high edge accuracy and the useless partition are consistent, not
contradictory.** 0.953 edge accuracy is dominated by within-fragment edges and
easy distant negatives. The decisions that matter — 703 true cross-fragment
joins among ~10⁸ candidate pairs — are a ~10⁻⁶ minority, and the classifier's
margin there is nowhere near enough. This is the same class-imbalance trap as
aggregate pair-F1, one level down.

**Diagnosis: the substrate, not the algorithm.** Fragment morphology here is the
real synapse point cloud, because no L2 skeleton cache exists for these regions.
STATUS.md measured exactly this difference on the same model family: union-find
ARI **0.305** with synapse-cloud fragments versus **0.838** once real L2
skeletons supplied genuine endpoint adjacency. A synapse-cloud "endpoint" is
just an extreme of a point cloud — and for the 8,823 single-synapse fragments in
this test region, it is the point itself, carrying no information at all about
where a neurite was severed.

A measurement from this run supports that reading: endpoint-adjacency at the
10 µm radius tuned for *skeletons* explodes on this substrate — 1,176,878
endpoint edges against 71,146 base edges on an earlier val build, because
thousands of unrelated cloud extremes fall within 10 µm in dense neuropil. This
run used 2 µm.

## The filter error

The first version of this benchmark was easier than the real task, and I did not
notice until asked how many IDs the boxes actually contained.

`build_bench_v1.py` defaulted to `min_syn_per_fragment=3`. On the test region
that kept **1,617 of 12,287 v117 roots (13.2%)** and **49 of 153 true merge
pairs (32%)**. The discarded 87% is the sliver and singleton population — median
synapses per v117 root is **1**, and 8,823 of 12,287 roots have exactly one
synapse. That population is not noise to be cleaned away; it is the confuser set
that defeated EXP-051, EXP-052 and EXP-053A. EXP-052 records the same structure
directly: 1,023 usable path roots against **10,218 singleton confusers**.

Removing it made cross-merge precision look roughly **4× better than it is**
(proximity: 1.7×10⁻⁵ on the filtered set versus 4.3×10⁻⁶ on the full one), and
measured recall against a third of the true positives.

This is the same failure mode as the incident this whole effort is about — a
benchmark quietly made easier than reality — arriving by a different mechanism.
The parameter was documented; its *cost* was not measured, and an undefended
default is what did the damage.

Fixed three ways:

1. `--min-syn-per-fragment` now defaults to **1**: keep everything.
2. Any value above 1 prints what it costs, in roots and in true positives, at
   build time.
3. Every manifest records `population_unfiltered` alongside the filtered stats,
   so the full candidate population is always visible in the dataset itself.

## A known limitation that is not fixed

The 20,000-synapse cap is a **sample of each region, not full coverage**. Using
EXP-052's documented density (24,573 synapses in a 30 µm box ≈ 0.9 pre-side
synapses/µm³), P1c's 541,670 µm³ would hold on the order of 10⁵ synapses, so
20,000 covers only a few percent of it. I attempted to measure the density
directly and the probe fetches timed out, so that figure is an inference from a
documented number, not my own measurement — treat it as an estimate.

The consequence is that the true confuser population is larger still, and every
precision figure here is correspondingly optimistic. The clean fix is to size
regions to the fetch budget — a ~30 µm box where a single ~20k fetch *is* the
whole population, as EXP-052 used — rather than sampling a large box. That is
the change I would make before `bench_v2`.

## Next

**Build the L2 skeleton cache for the bench_v1 regions and re-run.** That is
the one change the evidence points at, and it is the same lever STATUS.md
already measured as worth +0.533 ARI on the model family being used here.
`neuronauts/data/lineage.py::l2_skeleton` fetches per v117 root and caches to
`cache/l2_skeleton/`; EXP-053B's warning applies (only 27.8% of roots had
bounded L2 coverage there), so **measure coverage before trusting a re-run** —
if coverage is low the result will be uninterpretable rather than negative.

Everything else should wait on that. Tuning the classifier, the bias schedule,
or the clustering while the evidence channel is a point cloud is optimising the
part that is not binding — which is what the honest Phase 2.3 finding already
said: *"the binding constraint is the representation/evidence, not the inference
algorithm."*

Whatever runs next is scored on the same cross-fragment columns, against the
same locked test set, with the threshold picked on val. The bar to clear first
is not the oracle ceiling — it is `untouched v117`.
