# EXP-061 — the directed cone: informative, but not sharp enough


> **⚠️ Corrected 2026-09-02 — see [CORRECTION.md](../EXP-060/CORRECTION.md).**
> This experiment measured recall against *all* same-owner pairs. Assembly
> needs only a spanning set, and against minimum-spanning-tree links the
> panel proposes 24.6%, not 17.5%. The reported gap distribution (median
> 6.5 um) is likewise an all-pairs figure; the *nearest* same-owner partner
> has a median of 1.3 um. And 53% of the missed spanning links lie inside
> the 5 um radius, missed by the k=8 cap rather than by distance. The
> conclusion "geometry cannot propose candidates" is withdrawn.


## Result: failed, and the reason is not that the tangent is uninformative

> **Corrected 2026-09-02.** The first write-up of this table compared a
> best-of-two-directions statistic (the smaller of A's angle to B and B's angle
> to A) against the chance level for **one** random direction, (1 − cos θ)/2.
> The QA pass measured the real null — random unit tangents through the same
> loop, 20 seeds — at about twice that, so the enrichment was overstated ~2×:
> 3–6× is 2–3×. The reach and panel columns were never affected, and the
> verdict is unchanged. Details: `docs/threads/qa_pass_2026-09-02.md`.

No cone reached 70% of true pairs at a median panel of ≤20. But the tangent is
**not** noise — it carries 2–3× more directional information than chance, and
saying otherwise would be the easy wrong conclusion.

| cone | reach (within distance) | reach by angle | if direction were random (measured) | enrichment | median panel |
|---|---:|---:|---:|---:|---:|
| 10 µm, 15° | 6.1% | 10.2% | 3.4% | **3.0×** | 45 |
| 10 µm, 30° | 15.2% | 28.7% | 13.4% | 2.1× | 175 |
| 10 µm, 45° | 26.2% | 48.6% | 27.7% | 1.8× | 378 |
| 25 µm, 15° | 7.9% | 10.2% | 3.4% | 3.0× | 656 |
| 25 µm, 30° | 19.3% | 28.7% | 13.4% | 2.1× | 2,582 |
| 25 µm, 45° | 32.7% | 48.6% | 27.7% | 1.8× | 5,680 |
| 50 µm, 15° | 8.7% | 10.2% | 3.4% | 3.0× | 4,837 |
| 50 µm, 30° | 24.4% | 28.7% | 13.4% | 2.1× | 19,244 |
| 50 µm, 45° | **40.2%** | 48.6% | 27.7% | 1.8× | 42,160 |

Angle from the outward tangent to the true partner, against the measured null
(the better of two random directions, which is what the statistic takes):

| | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| **observed** | 14.6° | 27.0° | **45.5°** | 69.1° | 94.4° |
| if random (measured) | 25.9° | 42.5° | 64.8° | 90.0° | 112.7° |
| *if random, single direction — as first reported* | *36.9°* | *60.0°* | *90.0°* | *120.0°* | *143.1°* |

The observed distribution is shifted hard toward the tangent at every
percentile. A tip does tend to point at its continuation.

## But the shape of the enrichment is the problem

Enrichment is highest exactly where reach is lowest. A 15° cone is 3× better
than chance and finds 10% of partners. A 45° cone finds 48.6% but is 29% of
the sphere by solid angle — barely a direction at all — and its panel at 50 µm
is 42,160 distractor endpoints.

The trade is unavoidable because the angular signal is broad, not sharp: the
median partner is 45.5° off-tangent and **p90 is 94.4°, meaning roughly one true
partner in ten lies behind the endpoint** relative to the direction it was
pointing. Over gaps of 6–56 µm (EXP-060) a neurite curves enough that a
straight extrapolation from the tip does not land on the partner.

Comparing like for like against EXP-060: proximity reaches 47.4% within 5 µm;
the best cone here reaches 40.2%, and needs 50 µm and a 45° opening to do it.
The cone is not obviously better than the ball. It is a different, similarly
inadequate slice of the same geometry.

## What the three experiments settle together

| | EXP-058 | EXP-060 | EXP-061 |
|---|---|---|---|
| Can geometry **rank** true pairs? | No — pair precision 0.0006, indistinguishable from random | — | — |
| Can geometry **propose** them? | — | No — 17.5% proposed, 47.4% reachable at any k | No — 40.2% at best, panel 42,160 |

**Geometry alone cannot generate the candidate set**, by position or by
direction. That is a stronger and more useful statement than any of the three
alone, and it is consistent with the tree-assembly work's 0/32 adjudicated
links and with hypothesis H3.

Note what is *not* claimed: the tangent still carries 2–3× enrichment, so it
belongs in a **scorer** (EXP-064) as a feature over candidates generated some
other way. What it cannot do is generate them.

## Where candidate generation has to come from

Only one channel in this repo has a real within-type identity result:
**morphology embeddings** (tree-DNA, within-type AUC 0.829 at half-skeleton
scale; H2, the one supported hypothesis). Retrieval over an embedding has no
radius and no cone — it ranks by similarity, so a partner 56 µm away is no
harder to retrieve than one at 2 µm. That makes **EXP-057C** (whether Weis et
al. 2025 released per-root GraphDINO embeddings for this volume) the highest-
value unblocked experiment, not a nice-to-have; and if the embeddings are not
published, training our own on the harness becomes the critical path rather
than a later refinement.

The second candidate is **lineage-anchored scaffolding** — propose by
containment in a certified soma-owned scaffold rather than by any geometric
search (59.6% of synapses at 99.8% purity).

## Limits

- **Reachability is an upper bound, not an achieved recall.** It asks whether
  the partner lies inside the cone at all, using the endpoint of each atom
  closest to the partner. A real proposer would not know which endpoint to
  cast from. Actual recall would be lower.
- **Panel size is sampled** (4,000 labelled endpoints of 318k) while
  reachability uses all 492 pairs. The sample is large enough for a median but
  not for the tail.
- **The tangent spans up to 5 L2 nodes.** A longer or curvature-aware
  extrapolation might sharpen the angle; that is untested and is the obvious
  cheap follow-up before abandoning direction entirely.

## Reproduce

```bash
uv run python -m neuronauts.experiments.exp061_directed_cone
```
