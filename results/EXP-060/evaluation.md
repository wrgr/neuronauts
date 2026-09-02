# EXP-060 — the endpoint filter, and why it could not have worked


> **⚠️ Corrected 2026-09-02 — see [CORRECTION.md](../EXP-060/CORRECTION.md).**
> This experiment measured recall against *all* same-owner pairs. Assembly
> needs only a spanning set, and against minimum-spanning-tree links the
> panel proposes 24.6%, not 17.5%. The reported gap distribution (median
> 6.5 um) is likewise an all-pairs figure; the *nearest* same-owner partner
> has a median of 1.3 um. And 53% of the missed spanning links lie inside
> the 5 um radius, missed by the k=8 cap rather than by distance. The
> conclusion "geometry cannot propose candidates" is withdrawn.


## Result: failed, and the filter was never the problem

The bar — 90% recall of true continuation pairs, at a median panel of ≤20
partners, keeping ≤1% of endpoints — was **unreachable before any filter was
applied**. The candidate surface itself cannot propose the pairs.

| filter | endpoints kept | panel recall | median panel | p90 panel |
|---|---:|---:|---:|---:|
| none (ceiling) | 99.996% | **17.5%** | 819 | 2,246 |
| leaf ≥1 µm | 68.6% | 15.7% | 613 | — |
| leaf ≥1 µm, cal ≥30 nm | 35.3% | 10.8% | 422 | — |
| leaf ≥1 µm, cal ≥50 nm | 13.1% | 3.3% | 203 | — |
| leaf ≥2 µm, cal ≥30 nm | 17.3% | 5.5% | 221 | — |
| leaf ≥2 µm, cal ≥50 nm | 6.9% | 1.4% | 112 | — |
| leaf ≥2 µm, cal ≥80 nm | 0.40% | 0.2% | 1 | — |
| leaf ≥5 µm, cal ≥30 nm | 2.0% | 0.6% | 17 | — |
| leaf ≥5 µm, cal ≥50 nm | 0.55% | 0.2% | 2 | — |
| leaf ≥5 µm, cal ≥80 nm | 0.095% | 0.0% | 0 | — |

Recall falls at least as fast as the panel shrinks at every step. Leaf length
and tip caliber do not separate split sites from spines — they remove both.

## Why: same-neuron fragments are tens of microns apart

Measured directly, as the minimum endpoint-to-endpoint distance between each
pair of true partner atoms:

| percentile | gap |
|---|---:|
| p10 | 730 nm |
| p25 | 1.6 µm |
| **p50** | **6.5 µm** |
| p75 | 31.8 µm |
| p90 | 56.4 µm |
| p99 | 90.8 µm |

| within | true pairs reachable |
|---|---:|
| 1 µm | 14.6% |
| 2 µm | 30.9% |
| **5 µm (the search radius)** | **47.4%** |
| 10 µm | 57.1% |
| 20 µm | 67.7% |
| 50 µm | 86.6% |
| 100 µm | 99.6% |

**The median true partner is 6.5 µm away.** A 5 µm search radius can reach at
most 47.4% of true pairs no matter how large `k` is; the k = 8 cap then costs
the rest, 47.4% → 17.5%.

Widening the radius does not rescue it. Reaching 90% needs roughly 50 µm, and
endpoint density in this tissue is ~5.1 × 10⁻⁹ per nm³ — a 50 µm ball holds on
the order of **2.7 million** endpoints. The panel would be larger than the
problem. (A first attempt at exactly this sweep, k = 64 at 50 µm, was killed
before producing output, which is the same fact arriving as an out-of-memory.)

## What this settles

EXP-058 showed proximity is useless as a **scorer**: it ranks true pairs no
better than random, at pair precision 0.0006. This shows proximity is also
inadequate as a **proposer**: at any tractable budget it does not put the right
pairs in front of a scorer at all. Those are separate failures and both hold.

Together they retire proximity as a candidate-generation primitive on this
substrate — not as a weak baseline to beat, but as a mechanism that cannot
reach the answer. It is the quantitative form of what the tree-assembly work
concluded from 32 adjudicated links ("swap the channel for directed
continuation + EM texture"), and of hypothesis H3 in the consolidation plan,
now with the distance distribution that says *why*.

## What the program should do instead

Candidate generation has to use something that is not distance:

1. **Directed continuation.** A tangent cone projects along the neurite rather
   than searching a ball, so a partner 30 µm downstream is reachable while the
   thousands of unrelated processes packed within a micron are not. The
   endpoint table already carries the outward tangent for this.
2. **Identity-first proposal.** Tree-DNA / published morphology embeddings
   retrieve by *similarity*, not position — the one channel with a real
   within-type result (AUC 0.829 at half-skeleton scale). Retrieval over
   embeddings has no radius at all.
3. **Lineage-anchored scaffolding.** The certified dendritic scaffold (59.6% of
   synapses at 99.8% purity) proposes by containment, not proximity.

EXP-061 as currently scoped — proximity versus cone, by compartment — is now
the *right* experiment but with the wrong framing: the cone is not an
improvement on the ball, it is the replacement. It should be rewritten to
measure reachability and panel size for a directed cone at 10–50 µm, against
this experiment's numbers as the proximity baseline.

## Honest note on the bar

The 90% recall bar was adopted verbatim from the PCFG report's E4, and neither
that document nor this experiment's author checked first whether a 5 µm
proximity panel could reach 90% of true pairs at all. It cannot — 47.4% is the
ceiling. The bar was therefore unreachable by construction, and a bar that
cannot be met by any setting of the thing it governs is a badly-specified bar,
even when the experiment it fails is informative. What rescued this run is that
the gap distribution was measured in the same run, so the failure explains
itself rather than merely recording a number. Future candidate-generation bars
should state the reachability ceiling alongside the recall target.

## Reproduce

```bash
uv run python -m neuronauts.experiments.exp060_endpoint_filter
```

Ten panels are rebuilt from scratch (not post-filtered), because a proposer
filters *then* searches: dropping a spine changes which endpoints appear in its
neighbours' k-nearest sets, and post-filtering a finished panel would flatter
every row.
