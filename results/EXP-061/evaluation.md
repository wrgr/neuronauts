# EXP-061 — the directed cone: informative, but not sharp enough

## Result: failed, and the reason is not that the tangent is uninformative

No cone reached 70% of true pairs at a median panel of ≤20. But the tangent is
**not** noise — it carries 3–6× more directional information than chance, and
saying otherwise would be the easy wrong conclusion.

| cone | reach (within distance) | reach by angle | if direction were random | enrichment | median panel |
|---|---:|---:|---:|---:|---:|
| 10 µm, 15° | 6.1% | 10.2% | 1.7% | **6.0×** | 45 |
| 10 µm, 30° | 15.2% | 28.7% | 6.7% | 4.3× | 175 |
| 10 µm, 45° | 26.2% | 48.6% | 14.6% | 3.3× | 378 |
| 25 µm, 15° | 7.9% | 10.2% | 1.7% | 6.0× | 656 |
| 25 µm, 30° | 19.3% | 28.7% | 6.7% | 4.3× | 2,582 |
| 25 µm, 45° | 32.7% | 48.6% | 14.6% | 3.3× | 5,680 |
| 50 µm, 15° | 8.7% | 10.2% | 1.7% | 6.0× | 4,837 |
| 50 µm, 30° | 24.4% | 28.7% | 6.7% | 4.3× | 19,244 |
| 50 µm, 45° | **40.2%** | 48.6% | 14.6% | 3.3× | 42,160 |

Angle from the outward tangent to the true partner, against what a uniformly
random direction would give:

| | p10 | p25 | p50 | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| **observed** | 14.6° | 27.0° | **45.5°** | 69.1° | 94.4° |
| if random | 36.9° | 60.0° | 90.0° | 120.0° | 143.1° |

The observed distribution is shifted hard toward the tangent at every
percentile. A tip does tend to point at its continuation.

## But the shape of the enrichment is the problem

Enrichment is highest exactly where reach is lowest. A 15° cone is 6× better
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

Note what is *not* claimed: the tangent still carries 3–6× enrichment, so it
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
