# bench_v1 baselines — the honest floor

Real MICrONS data, locked split, calibrated on val only, one run on test.
Reproduce with:

```bash
python scripts/baseline_bench_v1.py   # floors + ceiling
python scripts/model_bench_v1.py --epochs 60   # learned model
```

Dataset manifest sha256
`f4185886e9137c61c0fde2ed26f14a76171d8c8e3a57d2cc7d40b895311c5c87`
(base v117 → labels v1718). Machine-readable records are the `*.json` files
beside this one, each stamped via `neuronauts.results_schema`.

## Read the cross-fragment columns, not the aggregate ones

Aggregate pairwise F1 is dominated by observation pairs that already share a
v117 root — correct for free, before any decision is made. EXP-056 called out
exactly this artifact (an atomic baseline scored 0.914 pair-F1 while resolving
nothing). So the numbers that matter are restricted to pairs whose **v117 roots
differ**: the pairs a merge decision actually has to get right.

## Test results (P1c, 7,483 observations, 1,617 v117 fragments, 1,782 true neurons)

| Method | ARI | pair F1 | cross-merge P | cross-merge R | cross joins predicted |
|---|---:|---:|---:|---:|---:|
| untouched v117 (do nothing) | **0.9654** | 0.9654 | n/a | 0.000 | 0 |
| proximity union-find (d=2 µm, val-calibrated) | 0.0006 | — | 1.7e-05 | 0.775 | 19,862,569 |
| EdgePartitionGNN + GAEC (cc_bias=2.0, val-calibrated) | 0.0013 | 0.0027 | 1.3e-05 | 0.420 | 14,179,549 |
| *oracle fragment ceiling (bound, not a method)* | *0.9711* | *0.9712* | *0.868* | *0.601* | *295* |

There are **426** true cross-fragment pairs to find in the test region.

**Nothing yet beats doing nothing.** Both methods that attempt merges score
essentially zero cross-merge precision and destroy ARI. On this dataset, as of
this run, the state of the art is the untouched v117 segmentation.

## What these say

**1. The do-nothing floor is already at ARI 0.9654.** Trusting v117 and making
no merges scores 0.9654 ARI and 0.9654 pair-F1 on test. Any headline ARI near
0.9 on this task means almost nothing on its own — it is what you get for
declining to act. This is the single most important calibration point for
reading past claims in this repo, and the reason the cross-fragment columns
exist.

**2. Naive spatial proximity is catastrophically imprecise.** At the radius the
val sweep selected (2 µm), the method predicts **19.9 million** cross-fragment
joins to recover 330 of 426 true ones — a precision of 1.7×10⁻⁵, and ARI
collapses to 0.0006. Beyond 3 µm it merges the entire region into a single
cluster. Neuropil is dense: synapses from unrelated neurons routinely sit within
2 µm, so proximity alone carries almost no evidence about identity.

This independently reproduces EXP-052's core failure on a clean, properly split
dataset: that run recovered 13 of 14 true pairs while predicting 496,510 joins.
Same pathology, now measured under a protocol with a real validation set.

**3. There is a real, well-posed gap for a learned model.** The reachable target
without splitting frankenmerged fragments is cross-merge precision 0.868 at
recall 0.601. So a model must move from `(n/a, 0.000)` toward `(0.868, 0.601)`.
That is the scoreboard.

**4. The oracle ceiling is not 1.0, and that is a label-noise result.** Even
with perfect fragment→neuron assignment, recall caps at 0.601 and precision at
0.868, because a frankenmerged v117 root can only receive one label. The
missing 40% of true pairs are unreachable by *any* method that treats v117
fragments as atoms — which is what EXP-056 showed geometry alone cannot fix.
Splitting fragments is a separate problem from joining them, and this bound
quantifies what it costs to ignore it.

## Val sweep (calibration; test was touched once, afterwards)

| d (nm) | ARI | cross-merge P | cross-merge R | cross joins | clusters |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.9233 | 0.000 | 0.000 | 189 | 1,301 |
| 1,000 | 0.6991 | 0.000 | 0.000 | 7,971 | 1,128 |
| **2,000** | 0.0006 | 1.4e-05 | 0.719 | 11,164,219 | 245 |
| 3,000 | 0.0000 | 1.4e-05 | 1.000 | 15,991,102 | 25 |
| 5,000–12,000 | 0.0000 | 1.4e-05 | 1.000 | 16,464,008 | 1 |

The sweep maximises cross-merge F1, which is why it picks 2 µm — but every
radius is a bad operating point. At 500 nm and 1 µm the method makes a handful
of joins and gets *none* of them right; from 2 µm up it over-merges the region.
There is no threshold on raw proximity that works, which is the finding.

## The learned model, and why it fails here

`EdgePartitionGNN` + GAEC correlation clustering (the Phase 2.11 configuration)
trained on train only, `cc_bias` swept on val only, applied once to test.

The edge classifier **did** learn real signal: 60 epochs took edge accuracy to
0.944 with `p_pos` 0.70 against `p_neg` 0.196 — a genuine separation, not a
collapse. It still produces no usable partition:

| cc_bias (val) | ARI | cross joins | correct | clusters |
|---:|---:|---:|---:|---:|
| −1.0 | 0.7379 | 0 | 0 | 2,592 |
| 0.0 | 0.8209 | 3,218 | **0** | 1,264 |
| 1.0 | 0.0090 | 2,153,382 | 240 | 111 |
| 2.0 | 0.0013 | 8,383,838 | 142 | 2 |
| 3.0 | 0.0000 | 16,464,008 | 224 | 1 |

There is no operating point. Below `bias=0` the model declines to merge at all,
which is just the do-nothing baseline; at `bias=0` it makes 3,218 cross-fragment
joins and gets **zero** of them right; above that it collapses the region into a
handful of clusters. The val sweep picks 2.0 only because a vanishing F1 still
beats a zero one.

**The high edge accuracy and the useless partition are consistent, not
contradictory.** 0.944 edge accuracy is dominated by within-fragment edges and
easy distant negatives. The decisions that matter — the handful of true
cross-fragment joins among millions of candidates — are a ~10⁻⁵ minority, and
the classifier's margin there is nowhere near enough. This is the same
class-imbalance trap as the aggregate pair-F1, one level down.

**Diagnosis: the substrate, not the algorithm.** Fragment morphology here is the
real synapse point cloud, because the L2 skeleton cache was not built for these
regions. STATUS.md measured exactly this difference on the same model family:
union-find ARI **0.305** with synapse-cloud fragments versus **0.838** once real
L2 skeletons supplied genuine endpoint adjacency. A synapse-cloud "endpoint" is
just an extreme of a point cloud, so it carries almost no information about
where a neurite was severed.

Two measurements from this run support that reading. First, endpoint-adjacency
edges are pathological on this substrate: at the 10 µm radius tuned for
skeletons, the val region alone produces 1,176,878 endpoint edges against 71,146
base edges — thousands of unrelated cloud extremes fall within 10 µm in dense
neuropil. (This run used 2 µm, which yields a tractable 13,396.) Second, the
proximity baseline fails the same way for the same reason: without real
geometry, "close" does not mean "connected".

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
