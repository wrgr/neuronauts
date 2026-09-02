# EXP-070 — Object vs endpoint distance

## Result: passed its soundness gates; the metric was wrong, but it is not the reason proximity failed

Every proximity measurement in this repo — EXP-060, EXP-060B, EXP-061, and
`CORRECTION.md` — computes distance between **endpoints**: the degree-1 nodes of
each atom's contracted L2 skeleton. That is a skeleton-space distance. "How
close does this object come to that object" is a different question, and the raw
fetch has always held the data to ask it: every L2 node of every atom, not just
the tips. Nothing consumed it until now.

This experiment re-measures EXP-060's own quantity over the object point cloud,
changing the point set and nothing else.

## Why the two numbers are comparable at all

An atom's endpoints are a **strict subset** of its L2 nodes. That is not assumed
here, it is verified id-by-id at build time (`scripts/build_object_geometry.py`,
gate 3) and again per-pair at run time:

| Check | tier ≥10 | all tiers |
|---|---:|---:|
| Endpoint L2 ids absent from their atom's node set | 0 / 5,103,160 | 0 / 6,893,517 |
| Max coordinate disagreement on a shared L2 id | 0.000 nm | 0.000 nm |
| Pairs where object gap **exceeds** endpoint gap | **0** / 492 | **0** / 15,235 |

Nested point sets mean `object gap ≤ endpoint gap` for every pair, always, with
equality when the closest approach genuinely is tip-to-tip. A single violation
would have meant the index was wrong rather than the geometry interesting, so
the run fails on one. There were none.

The control also reproduces exactly: EXP-060's recorded median true-pair gap of
**6,526.2 nm** and within-radius fraction of **0.473577** come back to the digit
from this code path, so the comparison below is measuring the point set and not
a rewrite.

## What changes

Reachability at the 5 µm search radius — the quantity EXP-060 failed on, and an
upper bound on what any proximity proposer can recall. The denominator is the
same pair universe for both columns (every pair whose atoms both have geometry);
a pair the endpoint metric cannot see at all, because an atom has no endpoint
row, counts as *not reached*, not as absent. A first version of this table took
the endpoint column over finite gaps only, which quietly dropped those pairs
from its denominator and flattered it by ~4 points on the full population — the
QA pass caught it, and the corrected numbers are below:

| Substrate | Denominator | Endpoint | **Object** | Gain |
|---|---|---:|---:|---:|
| tier ≥10 | all same-owner pairs | 47.4% | **55.5%** | +8.1 |
| tier ≥10 | MST spanning links | 64.9% | **75.7%** | +10.8 |
| all tiers | all same-owner pairs | 10.9% | **14.5%** | +3.6 |
| all tiers | MST spanning links | 43.3% | **56.8%** | +13.5 |

The MST rows are the ones that matter — `CORRECTION.md` established that
assembly needs a spanning set, not the same-owner clique. EXP-060B's uncapped
tier-10 recall of 64.6% sits right on this table's endpoint reachability of
64.9% (the small difference is its extra filter on endpoints with a non-finite
tangent), which is the consistency check that the two experiments are measuring
the same thing. So **the uncapped proximity ceiling on tier ≥10 moves from
~64.6% to ~75.7%.** On the full population EXP-060B's 47.4% was over its own
universe — an MST built from endpoints, which cannot contain the 289 atoms that
have none — so the like-for-like endpoint figure is 43.3%, and the object metric
takes it to 56.8%.

Two further findings, neither of which is a matter of degree:

**289 labelled atoms in the full population have no endpoint row at all** — 6%
of the labelled set. 280 of them do have object geometry. They are invisible to
every proposer built so far, at any radius, and become proposable for free under
the object metric.

**The answer key itself changes.** The MST is built from the same distances it
is later used to score, so switching metrics moves the ground truth:

| Substrate | Links (object metric) | Shared with endpoint MST | Object-only | Agreement |
|---|---:|---:|---:|---:|
| tier ≥10 | 350 | 348 | 2 | 99.4% |
| all tiers | 3,538 | 3,075 | 463 | 86.9% |

On tier ≥10 this is negligible. On the full population 463 object-metric links
have no counterpart in the endpoint MST, but that number conflates two different
facts, and the QA pass separated them: **325 of the 463 touch an atom with no
endpoint row** — they were never available to the endpoint MST at all — while
**138 object-only plus 187 endpoint-only links (~9%) are genuinely re-routed**
between atoms both metrics see. So EXP-060B's panel was scored against a partly
different answer key than the one the better metric implies; the re-routing is
about a third of what "463 differ" suggests, and it does not overturn
EXP-060B's conclusion, but its recall figures do carry an error bar nobody drew.

Finally, the closest approach is tip-to-tip on **30.1%** of tier-10 pairs and
**48.2%** of full-population pairs. The endpoint model's premise — that a false
split shows up as two tips facing each other — is right about half the time, and
wrong the other half.

## What this does not show

It does not rescue proximity. 75.7% is not 90%, and the panel-size problem
EXP-060B measured (median 3,870 candidate objects for uncapped recall) is
untouched by a change of metric — object distance changes which pairs are
*reachable*, not how many distractors sit inside the same ball. **The failure of
proximity to propose is not an artefact of measuring from skeleton tips.**

The recommendation is therefore narrower than "re-run everything and expect a
different answer": object distance should replace endpoint distance downstream
because it is the correct quantity, it is strictly tighter, it recovers 280
otherwise-unreachable atoms, and it fixes an answer key — not because it moves
the verdict.

## Open, and deliberately not answered here

Why are 43% of full-population spanning links still more than 5 µm apart at
object distance? Two fragments of one neuron that far apart, with a spanning
link between them, means either the intervening material was labelled as some
other owner, or it exists unlabelled, or the neuron genuinely leaves and
re-enters.

One side probe was run and is reported only because it constrains the answer
weakly: walking the straight line between the two closest points of each long
spanning link, the corridor is essentially always full of segmentation (for links beyond
5 µm: median coverage 90.7%, median 29 distinct atoms crossed; coverage falls
from 94.4% at 5–20 µm to 81.2% beyond 50 µm) but **almost never contains an atom
of the same owner** — median 0, and only 2.2% of those links contain even one. That argues against "the gap is unlabelled material of the
same cell". It is weak evidence, by construction: a neurite between two
fragments curves, and a straight line will leave it. The positive reading (the
corridor is dense with *other* cells' material) is the reliable half, and it is
really a restatement of the panel-size problem from a different direction.

Answering this properly needs a curve-tolerant test — adjacency-graph paths
through the object cloud rather than a straight line — which is its own
experiment, not a footnote to this one.
