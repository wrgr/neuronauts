# bench_v1 baselines — the honest floor

Real MICrONS data, locked split, calibrated on val only, one run on test.
Reproduce with:

```bash
python scripts/baseline_bench_v1.py
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
| untouched v117 (do nothing) | 0.9654 | 0.9654 | n/a | 0.000 | 0 |
| proximity union-find (d=2 µm, val-calibrated) | 0.0006 | — | **1.7e-05** | 0.775 | **19,862,569** |
| *oracle fragment ceiling (bound, not a method)* | *0.9711* | *0.9712* | *0.868* | *0.601* | *295* |

There are **426** true cross-fragment pairs to find in the test region.

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

## Next

The learned model (EdgePartitionGNN + correlation clustering, the Phase 2.11
configuration) runs against these same splits and is scored on the same
cross-fragment columns. It has to beat `(0.868, 0.601)`-normalised progress from
a standing start of zero cross-merge recall, and it must do so with the
threshold picked on val, not test.
