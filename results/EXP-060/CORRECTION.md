# Correction to EXP-060 and EXP-061 — I used the wrong denominator

*Written 2026-09-02, prompted by the observation that proximity should depend
on which atoms are filtered out. That was right, and following it up exposed
two compounding errors in my analysis. The conclusion "geometry cannot propose
candidates" was too strong and is withdrawn.*

## Error 1: all-pairs recall is the wrong target

EXP-060 measured what fraction of **all same-owner pairs** the candidate panel
proposes: 86 of 492, 17.5%. But assembly does not need every pair. A neuron
with *k* fragments needs *k−1* links to be assembled; the rest follow by
transitive closure. The right denominator is a **spanning set**.

| denominator | pairs | proposed by the panel |
|---|---:|---:|
| all same-owner pairs | 492 | 86 — **17.5%** |
| **minimum-spanning-tree links** | **350** | **86 — 24.6%** |

The same measurement contaminated the reported gap distribution. "Median true
partner 6.5 µm, p90 56 µm" was over all pairs, and is dominated by distant
fragment pairs of the same cell — a soma-adjacent fragment and a distal axon
tip are a "true pair" 90 µm apart that no proposer should ever be asked to
find. The quantity that governs assembly:

| | tier ≥10 | all tiers (≥1 synapse) |
|---|---:|---:|
| nearest same-owner partner, median | **1.3 µm** | 1.8 µm |
| has a partner within 5 µm | 76.7% | 67.3% |
| MST longest link per neuron, median | 1.6 µm | 20.7 µm |
| whole neuron spannable within 5 µm | 70.0% | 25.8% |

**Most atoms have a same-owner partner about a micron away.** Reporting 6.5 µm
as "the" distance was misleading, and the "geometry cannot propose" headline
followed from it.

## Error 2: the panel was under-built, and I blamed the wrong parameter

Of the 264 MST links the panel missed, **141 — 53% — have a gap inside the
5 µm search radius**. They were missed by the `k = 8` nearest-neighbours cap,
not by distance. Capturing every within-radius pair would take panel recall on
spanning links from 24.6% to roughly **65%** at the same radius.

So the binding constraint at 5 µm is `k`, not the radius, which is the opposite
of what EXP-060 concluded. The deeper issue is that k-nearest-neighbours *per
endpoint* is the wrong reduction: two atoms are adjacent if **any** endpoint
pair is close, but an endpoint with hundreds of competitors within a micron
spends its whole k budget on them and never reaches the true partner. Reducing
by atom pair rather than by endpoint neighbour is the correct primitive.

## What survives, and what does not

**Withdrawn:** "geometry alone cannot generate the candidate set." Not
demonstrated. A better-built proximity panel reaches ~65% of the links assembly
needs at 5 µm, and the tier-1 substrate — now complete — has not been tried at
all.

**Stands, unchanged:** EXP-058's finding that proximity cannot *rank*. Accepting
panel pairs by distance collapses the labelled population into one cluster at
every threshold, pair precision 0.0006, indistinguishable from random at
matched edge count. Nothing here touches that. The panel containing the right
pair does not help if the decision rule cannot pick it out.

**Stands, with a caveat:** EXP-061's angular measurement. The tangent carries
3–6× enrichment over chance, and the median angle to the partner is 45.5°. But
that too was computed over all pairs; the angle to the *nearest* partner, which
is the one a proposer must find, has not been measured and could be sharper.

**Revised bottom line.** The problem is better localised than I claimed. It is
not that geometry cannot propose — it is that (a) our panel was built with the
wrong reduction, and (b) proximity cannot rank what it proposes. That makes the
scorer the bottleneck, which is where EXP-064 already points, and it demotes
"abandon geometric proposal for embedding retrieval" from a conclusion to an
option.

## What to run

1. **Rebuild the panel by atom-pair reduction** rather than endpoint k-NN, at
   2 and 5 µm, and re-measure MST-link recall. Expected ~65% at 5 µm.
2. **Repeat on the tier-1 substrate** (277k atoms with geometry, now complete).
   More fragments means more links to make — the MST longest link median rises
   from 1.6 µm to 20.7 µm — so this could go either way, and it is the question
   that prompted the correction.
3. **Add the biological constraints** that shrink the space before scoring:
   synapse polarity (an axonal atom does not continue into a dendritic one),
   the one-soma-per-neuron rule, and caliber continuity across the join. None
   of these is a learned model; all are free, and EXP-060/061 measured raw
   geometry with none of them applied.

## Process note

The gap measurement and the panel-recall measurement were made in the same
experiment and reported together, which is what made the error look like a
finding rather than an artefact. Both were correct computations of the wrong
quantity. The check that would have caught it immediately — "how far is the
*nearest* partner?" — costs one line and was not run because the all-pairs
number already told a clean story.

---

## Error 3: panel sizes were counted in L2/endpoint space, not object space

*Added after the observation that the numbers "get blown up".* Correct, and by
up to 19.6x. EXP-061 reported cone panel sizes as counts of **distractor
endpoints**. The decision unit is the **atom**: a proposer offers candidate
partner objects, and one partner atom contributing forty endpoints to a cone is
one candidate, not forty.

| cone | endpoints (as reported) | distinct atoms (correct) | inflation |
|---|---:|---:|---:|
| 10 µm, 15° | 44 | 26 | 1.7× |
| 10 µm, 45° | 370 | 127 | 2.9× |
| 25 µm, 30° | 2,586 | 469 | 5.5× |
| 50 µm, 30° | 19,258 | 1,454 | 13.2× |
| 50 µm, 45° | **42,640** | **2,174** | **19.6×** |

The inflation grows with cone size, because a bigger cone sweeps more of each
neighbouring arbor rather than more neighbours. The whole tier-10 substrate is
20,826 atoms, so a panel can never exceed that however many endpoints it
touches — a reported "42,160" was larger than twice the entire object
population and should have been caught on sight.

This does not rescue the cone: 2,174 candidate atoms per endpoint is still far
too many to score. But "the panel would be larger than the problem" was wrong
as stated, and the correct figure is ~10% of the substrate rather than 2×.

**Rule for every future panel number: report candidates in object space.**
L2 nodes and endpoints are the geometry the search runs over; atoms are what
gets decided. Mixing them makes a panel look one to two orders of magnitude
worse than it is.
