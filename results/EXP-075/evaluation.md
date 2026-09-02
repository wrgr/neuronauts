# EXP-075 — Can local geometry tell a grower when to stop?

## Result: no. Max-score AUC 0.304 — the evidence is not merely silent, it argues confidently for the wrong answer. The stop rule cannot be local, which is the strongest argument yet for scoring trees rather than pairs.

EXP-074 established the specification: distance narrows a soma seed's field about
tenfold and then stops, leaving a contact panel of order tens, and the baseline
abstains 0.0% of the time on the 36 of 103 cells that need no joins at all. So
the grammar's stop rule is load-bearing. This asks whether local geometry can
supply it, before building a grammar that assumes it can.

## Substrate

24 contact panels (`scripts/build_contact_panels.py`), each every object with a
voxel in an 8 µm box at mip 2, with its true closest approach and four terms:

- `along` — does the candidate lie along the seed's severed axis
- `collin` — are the two local axes collinear
- caliber agreement, from the level-2 cache at native resolution
- proximity, `exp(-gap/500nm)`

16 panels from cells needing a join, centred on a real seed/target contact.
7 from cells whose soma fragment is already the whole in-box arbor, centred on
an interior arbor terminal — where a grower would ask the question and the
honest answer is that nothing continues.

## Ranking the true partner: two regimes

Among ~1,200 candidates per panel:

| feature | median rank | top-5 | top-20 |
|---|---:|---:|---:|
| distance | 42 | 1/16 | 4/16 |
| **along-axis** | **14** | 4/16 | **10/16** |
| collinearity | 220 | 0/16 | 0/16 |
| along × collin | **9** | 4/16 | 9/16 |
| along × collin × caliber | 12 | **6/16** | 10/16 |

Split by whether the segmentation merely severed a process or left a real gap:

| | touching (n=8, median gap 32 nm) | gapped (n=8, median gap 431 nm) |
|---|---:|---:|
| distance | 18 | 142 |
| along-axis | **6** | 122 |
| full stack | 8 | **33** |

Cut-face geometry works where there is a cut face to find, and not otherwise.
Collinearity alone is worse than distance and earns its place only inside the
product — an eight-cell pilot put it at median 38, which did not survive the
larger sample.

## The stop rule: worse than nothing

| | median | min | max |
|---|---:|---:|---:|
| needs a join — best in panel | 0.603 | 0.345 | 0.886 |
| needs a join — **the true partner** | 0.368 | 0.000 | 0.876 |
| already whole — best in panel | **0.734** | 0.541 | 0.817 |

**Max-score AUC separating the two classes: 0.304.** A whole cell's best false
candidate outscores a genuine partner in **89%** of pairs. Below 0.5 means the
statistic is anti-correlated: thresholding it stops precisely on the cells that
need joining and grows precisely on the ones that are finished.

The mechanism is not mysterious. A genuine terminal ends in dense neuropil among
well-aligned neighbours that look exactly like continuations, while a true
severed partner is often poorly aligned, because the cut fell at a branch or the
local axis estimate is noisy. Local evidence sees a good continuation everywhere,
and there is no local fact that distinguishes "this process continues into that
object" from "this process ends here, next to that object".

## Two placement errors caught before they became findings

Both would have produced a confident wrong answer, and neither was about method.

1. **Soma versus neurite.** The first whole-cell panels were centred on the
   soma, because a whole cell has no target to centre on. That gave them a seed
   caliber of 1,462-2,824 nm against 119-454 nm for the join-needing panels — an
   abstention test any caliber threshold passes without learning anything.
   Re-centred on arbor terminals, the same cells read 166-425 nm.
2. **The cube boundary.** The farthest point from a soma is usually where the
   cell leaves the cube, and there a continuation genuinely exists — in tissue
   never enumerated. Six of the first eight whole-cell panels sat within 1 µm of
   a cube face. That run returned AUC 0.438 and was discarded, not reported.
   Terminals are now required to sit a full box-width inside every face; one
   cell has no such terminal and is skipped rather than fudged.

The clean run is *more* negative than the contaminated one (0.304 against
0.438), so neither error was hiding a positive result.

## What this settles for the grammar

The stop rule cannot be a threshold on local evidence, and this is not a tuning
problem — the ordering is backwards, so no threshold on this statistic helps.
Abstention has to come from something local geometry cannot see:

- **tree context** — whether joining yields a morphologically valid arbor, which
  is a property of the tree, not of the pair. This is the case for scoring trees
  rather than pairs, now with a measurement behind it rather than an aesthetic
  preference.
- **the candidate's far end** — where the object goes after the join, which the
  8 µm box cannot see.
- **polarity consistency**, which EXP-063 measured at AUC 0.914 for a related
  question and which no term here uses.

## Limits

16 versus 7 panels is small, and the 112 pairs behind the AUC carry a standard
error near 0.07 — the direction is solid, the third digit is not. Panels are
centred on the true contact, so the ranking numbers answer "given you are
looking in the right place, can you pick the right object", which is optimistic
about a real grower. Each panel holds exactly one target fragment, so this is a
single join decision; full recovery compounds errors across many.
