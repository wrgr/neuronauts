# How many seam positives does EXP-057B actually deliver to *train*?

*Opened 2026-09-02, immediately after EXP-057B passed. EXP-057B's headline —
1,116 new candidate atoms against the existing 56, "roughly a 10–20× increase" —
is a count over the **whole cube**. Every downstream model splits that cube.
This thread measures what survives the split, because that, not the raw count,
is what decides whether EXP-062/063 can be run.*

## The short version

EXP-057B is a real unblock, but the margin is thinner than the headline reads,
and **the split design matters more than the label-tier choice.** Under
EXP-057's own split, the strictest defensible cut yields **143 training
positives** — below the 150 at which this repo's seam GNN was net-negative, and
well below the 513 at which it first cleared zero. Re-centring the split on the
positives and narrowing the buffer moves the same strict cut to **264**, and a
one-step-looser cut to **508**.

Nobody should pick a label tier for EXP-062/063 before picking a split.

## The control

The 56 existing seam positives split 15 train / 22 val / 19 buffer under
`assign_split(axis=0, centre=median(population x), buffer=20 µm)` — reproducing
EXP-057's own table exactly, so the numbers below come from the same machinery
that produced the figure they are being compared against.

## Under EXP-057's split, as-is

`data/substrate/c100um/cb2_seam_positives.npz`, 2,067 atoms, all of them in the
population. Tier codes: 3 = the existing 56, 2 = new and independently called
mixed-lineage by our own v1822 crosswalk at the strict threshold, 1 = mixed at
the raw threshold only, 0 = no independent v1822 signal at all.

| Cut | total | **train** | val | buffer |
|---|---:|---:|---:|---:|
| tier 3 (existing 56, re-found) | 37 | 10 | 12 | 15 |
| tier 2 — new, mixed-strict | 730 | 173 | 374 | 183 |
| tier 1 — new, mixed-raw only | 445 | 78 | 228 | 139 |
| tier 0 — new, no v1822 signal | 855 | 218 | 431 | 206 |
| **tier ≥2 AND split_before** (the recommended cut) | 606 | **143** | 305 | 158 |
| tier ≥1 AND split_before | 937 | 197 | 482 | 258 |
| tier ≥1, any decision type | 1,212 | 261 | 614 | 337 |
| all CB2-touched atoms | 2,067 | 479 | 1,045 | 543 |

Reference points, both from this repo's own seam GNN as recorded in
`results/EXP-057/evaluation.md` and `docs/pcfg_global_assembly_report.md`:
**net-negative at 150 training objects, first cleared zero at 513.**

So at the recommended cut the train side lands at 143 — a 9.5× improvement on
15, and still on the wrong side of both reference points. Even taking every
CB2-touched atom in the cube, tier 0 included, train reaches only 479.

**This is the caveat the intake document's "10–20× increase" does not carry.**
The multiplier is right; the absolute number is what gates the model, and it was
never stated against the 513.

## The split is doing more damage than the label tier

Two things about EXP-057's split are choices, not constraints, and both were
made before these positives existed:

**It is centred on the population median, not the label median.** EXP-057's own
evaluation already flagged the consequence for the old labels — "val holds 2.4×
the labelled atoms of train". The same imbalance hits the new ones: 479 train
vs 1,045 val, a 2.2× skew. Centring the split plane on the *positives'* median
instead costs nothing and roughly balances them:

| Cut | centre = population median | **centre = positives median** |
|---|---:|---:|
| tier ≥2 & split_before | 143 tr / 305 va | **219** tr / 210 va |
| tier ≥1 & split_before | 197 tr / 482 va | **317** tr / 311 va |
| tier ≥1, any | 261 tr / 614 va | **420** tr / 402 va |
| all CB2 | 479 tr / 1,045 va | **735** tr / 733 va |

(axis 0. Axes 1 and 2 behave similarly — axis 1 is already near-balanced on the
population median, and axis 2 peaks slightly higher at 792 for the all-CB2 cut.
No axis is dramatically better than the others.)

**The buffer is 20 µm, which is 4× the candidate search radius.** It exists so a
cross-seam pair never trains and tests the same neuron, so it must exceed the
radius at which pairs are proposed — 5 µm. 20 µm is generous against that, and
it costs 26% of the positives. Sweeping it, at axis 0 centred on the positives:

| buffer | tier ≥2 & split_before | tier ≥1 & split_before | tier ≥1 any | all CB2 |
|---:|---:|---:|---:|---:|
| 0 µm | 304 | 462 | 606 | 1,033 |
| 5 µm | 285 | 423 | 556 | 955 |
| **10 µm** | **264** | **386** | **508** | **881** |
| 20 µm (current) | 219 | 317 | 420 | 735 |
| 30 µm | 183 | 259 | 341 | 612 |

A 0 µm buffer is not on the table — it defeats the point. 10 µm is still 2× the
candidate radius and recovers ~20% over 20 µm.

## Where that leaves EXP-062 and EXP-063

Combining both levers (positives-centred, 10 µm buffer):

- **tier ≥2 AND split_before — 264 train.** Above "net-negative at 150", below
  "cleared zero at 513". The strictest, most defensible labels, in the band
  where the model was known to be useless-but-not-harmful.
- **tier ≥1, any decision type — 508 train.** Essentially at the 513 mark,
  without spending the tier-0 labels that carry no independent corroboration.
- **all CB2 — 881 train.** Clears 513 comfortably, but 855 of the 2,067 atoms
  are tier 0: CB2 says a proofreader operated on the object, and our own v1822
  crosswalk sees nothing. That is a real precision/quantity trade and it should
  be made explicitly, with the tier-0 result reported separately, not folded in.

Two caveats that apply to all of the above:

1. **513 is not a law.** It came from one model, on a different label
   distribution, in the PCFG report. It is the best anchor available and it is
   the one EXP-057 used to justify blocking, so it is the right number to be
   measured against — but a different model class could need more or less.
2. **These are not yet *located* seams.** EXP-057B's own caveat, carried in the
   artifact's `meta`: v117 resolution went through one arbitrary supervoxel per
   root, not the decision's `edit_point_nm`. Membership means "this decision's
   operand traces back to this atom", not "this atom's synapses are near this
   decision's edit point". EXP-062 asks *where* the seam is — for that specific
   question the spatial re-check is owed first, and it needs no network
   (`interface_point_nm` is in `data/external/cb2/full_mouse_rows_raw.parquet`).
   EXP-063 only asks *whether* an atom is a frankenmerge, so it can proceed
   without it.

**Recommended order:** EXP-063 first. Its bar is an evaluation metric, so it
spends positives on the test side where there is no shortage, and it does not
need located seams. EXP-062 second, after the `edit_point_nm` re-check and a
deliberate decision on the split.

## Reproducing

All numbers above come from `population.npz`, `cb2_seam_positives.npz` and
`neuronauts.harness.spatial_split.assign_split`; no network. The sweep is a
dozen lines and is not itself a registered experiment — if EXP-062/063 adopt a
different split, they should re-run it and record the counts they actually
trained on.
