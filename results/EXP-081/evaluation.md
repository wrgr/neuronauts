# EXP-081 — The frontier: 2,137 decisions, 34 of them joins

## Result: a soma-seeded grower faces a median of 46 cut ends per cell and should extend at 1. Across 40 cells: 2,137 tips, 34 live extension sites — **1.6%**. Every ranking number in this repository was measured on the 1.6% and ignored the rest.

## What was wrong with the panels

`scripts/build_contact_panels.py` builds one panel per cell, centred on the
**known seed/target contact**. That hands the grower the answer to the question
it actually has to answer — *where should I look* — and then scores it on the
easier question of which candidate is right once you are already in the right
place.

A grower starting at a cell body has no such information. It has a **frontier**:
every cut end of the cable it has claimed so far. It must decide at each one
whether anything continues, and for the overwhelming majority the honest answer
is no.

## The measurement

Tips are found on the mip-5 cloud (adequate at micron scale): a point with no
cable beyond it along the outward direction from the soma, deduplicated at 6 µm
so one ending counts once. A tip is *live* if a fragment of the seeded target
lies within 5 µm.

| | |
|---|---|
| cells | 40 |
| tips (frontier decisions) | **2,137** |
| live extension sites | **34** |
| dead ends | 2,103 |
| tips per cell | median 46 |
| live sites per cell | median 1 |
| **base rate** | **1.6%** |

## Why this changes the arithmetic

The ranking results — true partner at median rank 5 of 2,440, top-1 on 22 of 66
— are conditioned on being at a live site. That condition holds 1.6% of the time.

At the frontier the binding constraint is precision, and it is severe. With 46
tips per cell and one true join, a false-extend rate of 5% per tip yields about
2.3 false joins for every 1 correct one. To make the single true join with a
better-than-even chance of no false join, the per-tip false-positive rate has to
sit below roughly 2%.

This also recasts the abstention work. I treated stopping as a secondary
requirement measured on 21 whole cells, and reported figures from it that swung
0.44–0.64 on one feature. Stopping is not secondary: **it is 98.4% of the
decisions a grower makes.** A stop rule at AUC 0.64 does not approach what 46
consecutive decisions per cell require.

## What to measure next, and what not to

The right experiment is a frontier one: at each of the 2,137 tips, score whether
anything continues, and report precision at the 1.6% base rate — not accuracy,
and not per-panel rank. Sampling only live sites, as every panel experiment here
has done, cannot estimate it.

The frankenmerge question belongs at the same frontier: an extension that joins
two different cells is our own error to detect, and EXP-063's detector (held-out
AUC 0.958) has never been applied to a proposed join, only to existing objects.

## Limits

40 cells, one growth step from the soma fragment only — a real grower's frontier
grows as it claims objects, and later tips are thinner and harder. Tips come
from mip-5 centroids; the count is a micron-scale estimate, and a finer read
would likely find more tips, which makes the base rate lower rather than higher.
Live-site detection uses a 5 µm radius, which is generous.
