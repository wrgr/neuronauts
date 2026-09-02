# EXP-076 — Does the seed's own end shape stop the grower, and does polarity pick the partner?

## Result: no to both. The seed's end shape does not separate a cut from a genuine terminal once the two classes are matched for distality (AUC 0.476 on 250 pairs), and the raw separation runs opposite to the hypothesis. Polarity agreement with the seed is worse than useless — it inverts on axons, and half of these joins are axonal. A third box-placement error was found and corrected on the way.

EXP-075 showed that local *candidate* geometry cannot supply a stop rule
(max-score AUC 0.304, anti-correlated). It left open that the negative was
about candidate features, and that the seed alone might carry the answer: a
severed process should hold its caliber right up to the cut face, while a
genuine terminal tapers to a tip or closes into a bouton. `end_ratio` and
`end_drop` measure exactly that, on the seed, with no candidate involved. It
also named polarity consistency as an untried term. Both are tested here.

## Substrate

39 contact panels centred on a real seed/target cut (`scripts/build_contact_panels.py`),
33 panels on already-whole cells as that script places them, and 25 on
already-whole cells with the placement corrected (below). The panel build for
the remaining 28 join-needing cells had not progressed in the hour this ran, so
39 is the cut-class n throughout, not 67.

`end_drop` is **not a second feature**: `end_drop == 1 - end_ratio` holds to
1.2e-07 across all 94 panels, so its AUC is exactly one minus `end_ratio`'s.
Only one statistic is being tested here, reported twice.

## A third placement error, found before it became a finding

EXP-075 caught two box-placement errors on the already-whole class. Here is a
third, in the same place, and it invalidates the premise of question 1 as
posed.

`build_contact_panels.py` centres a whole-cell panel on "the farthest interior
point from the soma", intending an arbor terminal. Measured directly on the
cached mip 5 clouds, over all 35 already-whole cells that have a seed cloud and
an interior point, **the seed's own cloud continues past that point in 28 of
them** — by a median of 2,303 nm and up to 2,972 nm, with dozens of
seed points beyond it. The point is not where the arbor ends; it is where the
interior mask (a full box-width inside every cube face, EXP-075's fix for error
2) clipped the arbor. The seed reaches a median of 8.6 µm further from the soma
than any interior-eligible point does. So the "genuine terminal" class was, for
four fifths of its members, a box sitting mid-cable.

Corrected by searching interior seed points for one that is a genuine end of
its local cable — the seed's local principal axis must run out within 500 nm on
one side — and taking the most distal such point. Of the 34 whole cells the
builder attempts, 25 yield a corrected panel; 8
are skipped because no interior point is a genuine cable end (best margin 528
to 2,183 nm) and 1 has no interior point at all. Corrected panels are in
`data/external/panels_tip/`, with the builder beside them as
`_build_tip_panels.py` (one changed block against `scripts/build_contact_panels.py`;
the repository script is left untouched, since a build was running against it).

Both versions are reported below. The correction does not rescue the result.

## 1. The seed's end shape: no stop rule

Hypothesis: a cut holds its caliber, so `end_ratio` (tip caliber over caliber
1 µm back) should be **higher** at a cut than at a terminal.

| terminal class | cut n | terminal n | cut median | terminal median | AUC (cut > terminal) | pairs | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| as built (farthest interior point) | 39 | 33 | 0.736 | 0.790 | 0.443 | 1,184 | [0.306, 0.581] |
| corrected (genuine interior tip) | 39 | 25 | 0.736 | 1.016 | **0.325** | 925 | [0.198, 0.464] |
| corrected, cuts matched for distality | 10 | 25 | 0.814 | 1.016 | **0.476** | 250 | [0.236, 0.716] |

Read the middle row carefully before it looks like a result. AUC 0.325 is a
separation, but **in the direction opposite to the hypothesis**: the cut faces
taper *more* than the genuine terminals, not less. A mechanism is available —
the segmentation tends to break a process where it is thinnest, so a cut face
sits at a local caliber minimum — but it is not the stated hypothesis and it is
not usable, because of what the third row shows.

### The confound that eats it

The two classes are separated almost perfectly by something that has nothing to
do with end shape:

| control, cut > terminal | AUC | pairs | reading |
|---|---:|---:|---|
| seed caliber `cal_seed` | 0.503 | 975 | classes are caliber-matched — EXP-075's error 1 does **not** recur |
| **distance from soma to box** | **0.054** | 975 | terminals are farther from the soma in 94.6% of pairs |

Terminal boxes sit a median 90 µm from their soma; cut boxes sit at 54 µm. A
class separator at AUC 0.054 means *any* statistic that varies with distality
will appear to separate the classes. Within the cut class alone, `end_ratio`
does trend with distality (Spearman +0.289, p = 0.083) in the same direction as
the between-class difference. Restricting the cuts to those more than 70 µm
from their soma removes the asymmetry, and the separation collapses to 0.476 —
straddling chance, with a confidence interval [0.236, 0.716] far too wide to
support a third digit at 250 pairs.

**The stop rule is not in the seed's end shape.** EXP-075's negative was not
merely about candidate features. Even taking the uncorrected 0.675 for
`end_drop` at face value, it is nowhere near the 0.914 that EXP-063 reached on
a related question, and it does not survive the distality control.

### What is not ruled out

The measurement itself is coarse, and a better one might yet find something. In
`build_contact_panels.py`, `end_ratio` slabs *all* seed voxels within a 3 µm
ball of the centre, so the slab is a disc up to 3 µm across and any other
branch of the seed passing nearby is counted into it; and `edge` is set by a
single extreme voxel, so a bend in the cable within 1.3 µm of the end can empty
the reference slab. Two panels return NaN (the reference slab was empty) and
one returns 27.3. A caliber profile taken along the seed's own centreline,
rather than a ball-and-slab, is a different experiment and is not tested here.
What is established is that *this* statistic, on *these* panels, does not do it.

## 2. Polarity: coverage first

`data/external/object_polarity.npz` holds 279,075 objects, every one with at
least one synapse. Coverage of the cube is 277,081 of 909,888 objects = **30.5%**,
and it is overwhelmingly a size effect:

| object cloud size (supervoxels) | n | with polarity |
|---|---:|---:|
| 1 | 316,144 | 0.026 |
| 2-4 | 210,852 | 0.216 |
| 5-19 | 210,826 | 0.456 |
| 20-99 | 116,566 | 0.643 |
| 100+ | 55,500 | 0.940 |

Inside the panels: **39 of 39 true partners carry polarity (100%), against 63.8%
of 45,811 distractors.** That looks like a strong signal and it is almost
entirely the same size effect. Binned by candidate voxel count:

| candidate n_vox | distractors | coverage | true partners | coverage |
|---|---:|---:|---:|---:|
| < 200 | 12,305 | 0.25 | 0 | — |
| 200-1,600 | 15,760 | 0.61 | 4 | 1.00 |
| 1,600-6,400 | 11,424 | 0.89 | 16 | 1.00 |
| > 6,400 | 6,322 | 0.97 | 19 | 1.00 |

True partners have a median 6,154 voxels against 858 for distractors. Among
distractors of the same size, 93.5-97.6% carry polarity. "The true partner
always has a synapse" is, to a first approximation, "the true partner is big".

## 2b. Polarity agreement is worse than useless

The geometric baseline is EXP-075's best stack, `along x collin x caliber`.
The EXP-075 script was not kept, so the reconstruction was checked against its
published table on a different panel set — collinearity worst, along-axis
middling, the product best — and reproduces that ordering (`along` 20 vs their
14, `along x collin` 10 vs 9, `along x collin x caliber` 8 vs 12,
`collin` 185 vs 220, distance 80 vs 42).

39 panels, median 1,185 candidates each, one true partner per panel. AUCs are
per-panel means. Pooled AUCs over all 1,786,629 positive/negative pairs agree
with them to within 0.015 on every row except `polarity agreement alone`, where
the pooled figure is 0.575 against a per-panel 0.526; both readings are close
enough to chance that the row's verdict is the same either way.

| score | median rank | mean | top-1 | top-5 | top-20 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| distance alone | 80 | 110 | 0 | 0 | 4 | 0.911 |
| n_vox alone | 191 | 227 | 0 | 0 | 1 | 0.801 |
| has any synapse alone | 771 | 751 | 0 | 0 | 0 | 0.681 |
| polarity agreement alone | 822 | 668 | 0 | 0 | 1 | 0.526 |
| polarity purity alone | 519 | 519 | 0 | 0 | 0 | 0.739 |
| **geometry** (along x collin x caliber) | **8** | 116 | 9 | 17 | 24 | 0.901 |
| geometry x polarity agreement | 93 | 186 | 9 | 13 | 14 | **0.843** |
| geometry x polarity purity | 6 | 106 | 9 | 19 | 24 | 0.910 |
| geometry, 0-synapse candidates dropped | **7** | **85** | 9 | 19 | 24 | **0.927** |
| ... same count dropped, smallest first (size control) | 8 | 100 | 9 | 18 | 24 | 0.915 |

Polarity agreement with the seed **costs** 0.058 AUC and moves the median rank
from 8 to 93. The reason is structural, not statistical. Every one of the 39
seeds is postsynaptic-dominant (`frac_pre` 0.00 to 0.22) because a soma
fragment carries the soma and its dendrites. But 20 of the 39 true partners are
purely presynaptic and a 21st is 0.88 — they are the cell's **own axon**. Agreement therefore
scores half the correct answers as the worst possible match:

| true partner's polarity | n | geometry alone | with agreement |
|---|---:|---:|---:|
| presynaptic (axon), `frac_pre` > 0.5 | 21 | median rank 17 | median rank **241** |
| postsynaptic (dendrite), `frac_pre` <= 0.5 | 18 | median rank 4 | median rank **1.5** |

Agreement is not a polarity term at all — it is a "is this a dendrite" term. It
is nearly perfect where the join is dendritic and catastrophic where it is
axonal, and a soma seed's joins are roughly half of each. A term that used the
seed's polarity *local to the box* rather than its whole-fragment aggregate
might behave differently; the panels do not carry that, and it is untested.

## 2c. The one polarity term that helps, and how much of it is size

Dropping candidates that carry no synapse at all raises AUC from 0.901 to
0.927, improves the true partner's rank in 22 of 39 panels, and worsens it in
0 (Wilcoxon p = 9.9e-06). But a control that drops the same number of
candidates per panel, smallest first, reaches 0.915 on its own.

| comparison | AUC gain | 95% CI (20,000 panel bootstraps) |
|---|---:|---|
| synapse filter over geometry alone | +0.026 | [+0.009, +0.047] |
| synapse filter over the size-matched control | **+0.012** | [+0.002, +0.031] |

So roughly half the gain is size, and what remains is +0.012 with a confidence
interval that barely clears zero at n = 39. It moves the median rank from 8 to
7. This is a real but marginal filter, not a selector.

It also never removed the true partner *in this sample* — but only because all
39 true partners happened to carry a synapse, which the coverage table above
shows is a consequence of their being large. The size-matched control did
remove the true partner once (a 215-voxel partner). The evidence is also thin
where it exists: the median true partner carries 7 synapses, 17 of 39 carry
fewer than 5, and one carries exactly 1. On a cell whose continuation is a
small or synapse-free fragment, this filter can discard the answer.

## What this settles

- **The stop rule is not local, and now not in the seed either.** EXP-075 ruled
  out candidate geometry; this rules out the seed's end shape as measured.
  Together they are a stronger case that abstention has to come from tree
  context — whether the join yields a valid arbor — rather than from any
  statistic computed inside an 8 µm box.
- **Polarity's value is in the seed's local identity, not its aggregate.**
  Whole-fragment `frac_pre` on a soma seed is a dendrite detector and inverts on
  axonal joins. The finding that 20 of 39 joins are the cell's own axon, against
  a uniformly dendritic seed, is the useful part: a grammar has to know which
  compartment it is growing before polarity means anything.
- **The already-whole class needs re-checking wherever it is used.** EXP-075's
  0.304 was computed on panels placed the same defective way. The direction of
  that error there is unknown and untested; it is not claimed here that 0.304
  is wrong.

## Limits

- 39 cut panels against 25 corrected terminals. The distality-matched
  comparison rests on 250 pairs and cannot support a third digit; its interval
  [0.236, 0.716] would accommodate a modest real effect in either direction.
- The panel build stalled at 39 of 67 join-needing cells, so the cut class is
  not the full set and was not chosen at random from it — it is the first 39 in
  card order.
- The distality match was made by restricting cuts to > 70 µm from the soma,
  which leaves 10 of them. Matching by resampling or regression on a larger cut
  set is the better test and was not run.
- `end_ratio` is measured by a ball-and-slab that admits nearby branches of the
  same seed, and its reference slab is empty in 2 of 39 cut panels. A
  centreline caliber profile is untested.
- Polarity is whole-object `frac_pre` from a precomputed table. No local or
  per-compartment polarity was available.
