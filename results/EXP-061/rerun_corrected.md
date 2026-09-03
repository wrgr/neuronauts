# EXP-061 rerun — the directed cone, with tangents fitted to complete objects

*Run 2026-09-02. Not committed. Every number below is produced by the two
scripts named at the end, which are preserved with their output in
`data/external/rerun_060b_061/`.*

## First, a correction to the brief for this rerun

The rerun was commissioned on the premise that EXP-060B and EXP-061 measured
proximity and direction on `data/substrate/c100um/object_clouds_mip5.npz`.
**They did not.** Neither module reads that file. Both read
`data/substrate/topology/k10.npz` / `kall.npz`, whose `ep_pos_nm` and
`ep_tangent` come from `scripts/build_atom_topology.py`: one point per **level-2
node** taken from the level-2 cache (`data/substrate/geom/l2_attributes.npz`),
contracted to a skeleton, with the tangent at a leaf tip taken as the chord back
five level-2 nodes.

The premise is wrong about the file and right about the defect. A level-2 chunk
is 2,048 x 2,048 x 20,480 nanometres — **10:1 longer in z** — so the original
tangent is a chord across up to five anisotropic chunks, and the original
"endpoint cloud" is a sparse, direction-distorted sampling of an object rather
than the object. The rerun is therefore still the right thing to do; only the
name of the defective input changes.

---

## AUDIT

| | this rerun | the original EXP-061 |
|---|---|---|
| **substrate** | `data/substrate/c40um_mip2/object_clouds_mip2.npz` — 40 um cube at (663, 591, 860) um, read at **mip 2 = 32 x 32 x 40 nm**, one point per supervoxel (its voxel centroid). 192,474 objects, 22,494,502 points; median **949 points per labelled object**; typical point spacing about 130 nm (a 150 nm grid reduces the cloud only 1.6x) | `data/substrate/topology/k10.npz` — level-2 node skeleton over the 100 um cube, leaf tips only, chunk anisotropy 10:1 |
| **identity resolution** | v117 at timestamp 1623399000, supervoxel to root resolved on **100%** of supervoxels at mip 2. Checked against a direct `CloudVolume(agglomerate=True, timestamp=V117_TS, mip=2)` read of a 3 um box: **99.56%** of sampled cloud points carry the same object id once the file's known uniform half-voxel offset (16/16/20 nm, documented in `scripts/build_object_clouds.py`) is applied; 97.48% without it. The offset is the same for every point, so cloud-to-cloud distances are unaffected | v117 level-2 leaves per atom; no voxel-level check |
| **sample size** | 127 owners with 2 or more labelled fragments, **332 fragments**, **322 same-owner pairs**, of which **205 are spanning (minimum-spanning-tree) links**. 400 seed points for the cone panel; 20 random-tangent seeds for the null | 492 same-owner pairs, 4,000 sampled endpoints for the panel |
| **main confound** | **the 40 um cube.** Fragments of the same cell outside it are absent, so the pair set is biased toward shorter gaps, and no cone longer than 20 um can be measured without truncation — the original's 50 um row cannot be reproduced at all. To separate the scope change from the geometry change, the original's own level-2 substrate is re-measured **in the same 40 um scope**, and that column is the one to compare against | the 100 um cube, tier >= 10 only (20,826 atoms) |
| **second confound** | the point cloud is one point per supervoxel, i.e. supervoxel centroids, not surface voxels. Gaps are therefore biased upward by roughly the local caliber. It is the same bias on both sides of every comparison here | leaf tips only |

---

## Result 1 — the tangent is much sharper than the original reported

Angle from the outward tangent to the true partner, best of the two directions
of the pair (the original's statistic, unchanged). The corrected tangent is the
first principal axis of the object's own cloud within the stated radius of the
contact point, signed outward by the offset of that point from the local
centroid.

| | p10 | p25 | **p50** | p75 | **p90** |
|---|---:|---:|---:|---:|---:|
| **corrected, 1.5 um fit** | 12.8 | 21.3 | **36.5** | 59.1 | **77.0** |
| corrected, 0.75 um fit | 12.9 | 22.9 | 39.5 | 61.6 | 78.0 |
| corrected, 3 um fit | 11.8 | 23.0 | 41.1 | 64.4 | 80.5 |
| original level-2 tangent, **same 40 um scope** | 11.1 | 22.5 | 41.2 | 61.7 | 79.6 |
| original level-2 tangent, as published (100 um) | 14.6 | 27.0 | 45.5 | 69.1 | 94.4 |
| random-tangent null (measured, 20 seeds) | 25.7 | 42.2 | 65.3 | 90.4 | 112.1 |

Restricted to the 205 spanning links only, the corrected 1.5 um fit gives a
median of **34.1 degrees**.

**This overturns one of the original's specific claims.** It said "p90 is 94.4
degrees, meaning roughly one true partner in ten lies behind the endpoint
relative to the direction it was pointing." On corrected geometry **1.9% of true
partners lie behind the tangent**, not 10%.

Attributing that between the two changes, measured rather than assumed:

| | share of true partners behind the tangent |
|---|---:|
| original, as published (100 um, level-2 tangent) | about 10% (p90 = 94.4 deg) |
| level-2 tangent, restricted to this 40 um scope | 6.5% |
| **corrected tangent, same scope** | **1.9%** |

So the smaller cube accounts for part of it and the corrected tangent for a
further 3.4-fold reduction. On the **median** the split is about even: 45.5 deg
published, 41.2 deg for the level-2 tangent in this scope, 36.5 deg corrected.

The 1.5 um fit is the best of the three tested. A **longer** baseline is worse
(3 um: median 41.1), which answers the original's own open question — it
speculated that "a longer or curvature-aware extrapolation might sharpen the
angle." It does not; a neurite curves, and averaging over more of it blurs the
tip direction rather than stabilising it.

**Where you cast from matters more than how far back you fit.** Splitting pairs
by how end-like the contact point is (`end_ratio`, the offset of the point from
its local centroid as a fraction of the fit radius — near 0 for a point in the
middle of a passing cable, near 0.5 at the end of one):

| contact point | n | angle p25 / p50 / p75 | within 30 degrees |
|---|---:|---:|---:|
| a cable **end** (end_ratio > 0.30) | 259 | 20.8 / **34.1** / 55.6 | **40.5%** |
| intermediate (0.15-0.30) | 57 | 28.5 / 47.8 / 73.4 | 29.8% |

---

## Result 2 — enrichment against the original's null: 2.1x to 4.3x, not 2x to 3x

Same empirical null as the corrected original: random unit tangents pushed
through the same loop, same pairs, 20 seeds.

| half-angle | reach by angle | measured chance | enrichment | *original* |
|---|---:|---:|---:|---:|
| 15 deg | 15.2% | 3.5% | **4.34x** | *3.0x* |
| 30 deg | 37.9% | 13.3% | **2.85x** | *2.1x* |
| 45 deg | 57.8% | 27.8% | **2.08x** | *1.8x* |

A real but modest improvement — and, on its own, it would say the original's
verdict stands almost unchanged. **That reading is wrong, because the null is
wrong for the question.**

## Result 3 — the null a proposer actually faces is not isotropic

A random-tangent null asks "how often would a randomly aimed cone contain the
partner." But the objects a cone has to reject are not scattered isotropically
around a neurite: **a cable in dense neuropil is flanked by its neighbours, not
followed by them.** So the operational question is what the cone removes from
the panel versus what it keeps of the truth. Measured directly against the ball
of the same radius, counting distinct candidate **objects**:

**5 um radius** — ball reaches 35.4% of pairs and holds a median of 1,598 objects

| half-angle | reach | share of ball's reach | panel | share of ball's panel | enrichment |
|---|---:|---:|---:|---:|---:|
| 15 deg | 7.5% | 0.21 | 49 | **0.031** | **6.9x** |
| 30 deg | 14.3% | 0.40 | 154 | 0.096 | 4.2x |
| 45 deg | 22.7% | 0.64 | 296 | 0.186 | 3.5x |
| 90 deg | 33.9% | 0.96 | 885 | 0.554 | 1.7x |

**10 um radius** — ball reaches 43.2% and holds a median of 10,584 objects

| half-angle | reach | share of ball's reach | panel | share of ball's panel | enrichment |
|---|---:|---:|---:|---:|---:|
| 15 deg | 9.0% | 0.21 | 286 | **0.027** | **7.7x** |
| 30 deg | 17.7% | 0.41 | 938 | 0.089 | 4.6x |
| 45 deg | 28.3% | 0.65 | 1,887 | 0.178 | 3.7x |

**This overturns the original's "the cone is not obviously better than the
ball."** At a fixed radius the 15 degree cone keeps a fifth of the ball's reach
for a thirtieth of its panel.

## Result 4 — an independent check on the 99 prebuilt contact panels

The 66 cut-centred panels in `data/external/panels/` were built by
`scripts/build_contact_panels.py` on correct identity, independently of this
cube, and carry `along` (the absolute cosine between the seed's local axis and
the direction to a candidate). They give the same statistic on a different
sample, and with the candidate's **closest point** rather than "any point of the
object", which is the stricter and more realistic definition.

Direction separates truth from distractors cleanly:

| `along` (absolute cosine) | p25 | p50 | p75 |
|---|---:|---:|---:|
| the true continuation (n = 66) | 0.43 | **0.865** | 0.975 |
| every other candidate (n = 160,331) | 0.00 | **0.041** | 0.247 |

A double cone on that axis (the panels store an unsigned cosine, so the cone is
two-sided), against a median panel of 2,440 objects:

| half-angle | true-partner recall | share of panel kept | enrichment | median panel | on an arbor terminal (nothing should continue) |
|---|---:|---:|---:|---:|---:|
| 15 deg | 34.8% | 0.78% | **44.7x** | **15** | 13 |
| 30 deg | 50.0% | 3.20% | 15.6x | 66 | 51 |
| 45 deg | 60.6% | 7.23% | 8.4x | 159 | 120 |
| 60 deg | 69.7% | 13.5% | 5.2x | 304 | 231 |
| 90 deg | 98.5% | 62.9% | 1.6x | 1,527 | 902 |

Ranking the panel by gap alone puts the true continuation at a median rank of
66.5; ranking by gap **inside a 30 degree cone** puts it at **rank 9** — for the
half of seeds whose partner survives the cone.

---

## Does the original conclusion survive?

**The headline verdict survives; three of its supporting claims do not.**

**Survives.** No cone reaches 70% of true pairs at a median panel of 20 or fewer
objects. The best point on the whole sweep is the 15 degree double cone on the
contact panels: a median panel of **15 objects**, inside the bar, at **34.8%**
recall — half of what the bar demands. Widening to 60 degrees reaches 69.7% but
costs 304 objects. The trade-off the original identified is real and is still
the reason a cone alone cannot generate the candidate set.

**Does not survive — "roughly one true partner in ten lies behind the
endpoint."** It is 1.9%. The original's p90 of 94.4 degrees was an artefact of a
tangent measured across five 10:1-anisotropic chunks; the corrected p90 is 77.0
degrees.

**Does not survive — "the cone is not obviously better than the ball."** At a
fixed radius the cone is decisively better: 3.1% of the ball's panel for 21% of
its reach at 15 degrees.

**Does not survive as stated — "the tangent carries 2-3x more directional
information than chance."** Against the isotropic null the original chose it is
2.1x to 4.3x. Against the panel a proposer actually faces it is **6.9x to 44.7x**,
because the distractors near a neurite are overwhelmingly perpendicular to it
and an isotropic null does not know that. The original's own framing — "the
tangent belongs in a scorer as a feature over candidates generated some other
way" — understates it: on correct geometry direction is strong enough to be a
*filter*, cutting 97-99% of the panel, and the reason it still cannot stand
alone is recall, not selectivity.

**One claim is scope, not geometry.** The original's "median partner is 6.5 um
away, p90 56 um" cannot be re-tested in a 40 um cube. Within this cube the
median spanning-link gap on correct geometry is **3.5 um**, and its p25 is
**224 nm** — but that number is not comparable to the original's and is not
offered as a correction to it.

## Limits of this rerun

- **A 40 um cube.** Cones of 25 and 50 um cannot have their panels measured; the
  25 um reach figures are reported but the 50 um row of the original's table has
  no counterpart here. Long-gap pairs are under-represented.
- **Reachability is still an upper bound.** As in the original, the angle is
  measured from the point of A closest to B — an oracle a real proposer does not
  have. Result 1's split by `end_ratio` is the closest thing here to measuring
  the cost of that oracle, and it is not the whole cost.
- **Points are supervoxel centroids, not surface voxels**, so every gap is
  biased upward by about the local caliber. Uniform across the comparison.
- **The cube's cone panel counts an object if any of its points lies in the
  cone**, matching the original's endpoint-counting rule; the contact panels use
  the stricter closest-point rule. That difference, not a disagreement about the
  data, is why the cube gives 6.9x where the panels give 44.7x at 15 degrees.
- 322 pairs and 66 contact panels are enough for medians, not for the tail.

## Reproduce

```bash
cd data/external/rerun_060b_061
python rerun061.py          # angles, empirical null, cone panel, level-2 control
python rerun061_extras.py   # cone-versus-ball trade, end_ratio split, contact panels
```
The scripts, their JSON output and `prep.npz` are preserved in
`data/external/rerun_060b_061/` (gitignored). Inputs read:
`data/substrate/c40um_mip2/`, `data/substrate/topology/kall.npz`,
`data/substrate/c100um/labels_v1822.npz` and `data/external/panels/`.
