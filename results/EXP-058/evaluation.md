# EXP-058 — the baseline ladder

## Result: passed, and proximity is worth nothing

The ladder is correctly ordered, so the evaluation path works end to end. What
it measures is that **endpoint proximity carries no usable identity signal on
this substrate** — it is not merely weak, it is indistinguishable from random.

Evaluated over the 1,297 tier-10 atoms with a proofread owner (947 owners, 492
true same-owner pairs), with the other 19,529 atoms present as distractors.

| Rung | ARI | pair P | pair R | TP | FP | clusters | largest |
|---|---:|---:|---:|---:|---:|---:|---:|
| do nothing (v117 untouched) | 0.0000 | — | 0.0000 | 0 | 0 | 1,297 | 1 |
| random, matched count | −0.0000 | 0.0006 | 0.9980 | 491 | 838,669 | 2 | 1,296 |
| proximity ≤ 1 µm | 0.0000 | 0.0006 | 1.0000 | 492 | 839,964 | 1 | 1,297 |
| proximity ≤ 2 µm | 0.0000 | 0.0006 | 1.0000 | 492 | 839,964 | 1 | 1,297 |
| proximity ≤ 5 µm | 0.0000 | 0.0006 | 1.0000 | 492 | 839,964 | 1 | 1,297 |
| **oracle** | **1.0000** | **1.0000** | **1.0000** | **492** | **0** | **947** | **6** |

`do nothing` has no precision because it predicts no merges — NaN, not 1.0,
which is one of the conventions `neuronauts/metrics/` fixed on the way in. It
is why "untouched v117" scored a perfect precision in EXP-052.

## What the numbers say

**Proximity collapses the population into a single cluster at every threshold
tested**, including 1 µm. It recovers all 492 true pairs and 839,964 false
ones. Random merging at the same edge count reaches the same precision to four
decimal places. Whatever ordering information the endpoint gap carries, union-
find at a distance threshold cannot extract it.

The reason is visible in the candidate surface itself: of 6,848,187 candidate
atom pairs, **61% are within 1 µm and 99.5% within 2 µm**. In this tissue a
sub-micron gap is not evidence of anything — dozens of unrelated processes pack
within a micron of any given endpoint. This is the same conclusion the tree-
assembly work reached from adjudicated links (0/32), now measured across the
whole substrate with a matched random control beside it.

## The consequence for the program: recall is free, precision is everything

Every later experiment now has its floor: **pair precision 0.0006 at full
recall.** Beating recall is trivial and meaningless here; the entire task is
precision at usable recall. That should change how EXP-064 is judged. An AUROC
bar is a poor fit for a 0.06%-positive problem — a scorer can look strong at
AUROC 0.9 and still be useless at any threshold anyone would deploy. EXP-064
should be judged on **precision at fixed recall**, and on where its abstention
curve crosses a precision a proofreader would accept, with AUROC reported as a
secondary diagnostic rather than the bar.

## Limits of these numbers, stated

- **The ceiling is conditional.** The oracle clusters by proofread owner, so it
  scores 1.0 by construction — but only over pairs the shared candidate panel
  could ever propose. The panel is bounded (k = 8 nearest endpoints within
  5 µm), not all-pairs, so a true ceiling is lower wherever the panel missed a
  pair. It happens to contain all 492 true pairs here; that is a property of
  this substrate, not a guarantee.
- **Trained checkpoints are absent from the ladder.** The plan lists them as a
  rung. The treestitch models expect a different input contract, and wiring
  them in badly would be worse than recording that they are missing. That rung
  is outstanding.
- **1,297 atoms is a thin evaluation set**, a further thinning of EXP-057's
  4,802: only that many proofread-owned atoms have tier-10 geometry. The
  comparison *between rungs* is fair because every rung shares it, but no
  absolute number here should be read as describing the region.

## Reproduce

```bash
uv run python -m neuronauts.experiments.exp058_baseline_ladder
```

The shared candidate panel is cached at
`data/substrate/panels/k10_proximity.npz` (6,848,187 pairs, ~66 s to build,
5.7 GB peak) and is reused by EXP-060.

## A bug worth recording

The first run of this experiment **passed with every precision and recall
NaN**. The metric suite returns `pair_precision`/`pair_recall`; the experiment
asked for `merge_precision`/`merge_recall`, got nothing, and the ordering check
accepted the result because one of its clauses read `== 0.0 or isnan(...)`.
Fixed twice over: the score function now names the keys it requires and raises
if the suite does not return them, and the ordering check refuses to pass on a
non-finite rung. A ladder that cannot be ordered is not a passing ladder.
