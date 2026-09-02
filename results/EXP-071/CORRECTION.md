# Correction to EXP-071 — the connective-object count is scoped to 200 µm, not the cube

*Written 2026-09-02, hours after the run, when the full-cube object enumeration
recovered only 8.6% of the 2,147 objects this experiment said hold the
connective material. That looked like the enumeration failing. It was not.*

## What happened

The enumeration (`scripts/enumerate_region_objects.py`) covers the **100 µm
harness cube**. EXP-071 fetches each proofread cell's level-2 graph with
`region_bounds(centre, 200.0)` — the **200 µm outer box**, chosen to match the
bounds the original atom-geometry fetch used so the caches compose. Proofread
cells are much larger than the cube, so most of the material EXP-071 counted was
never in the enumeration's scope.

Measured directly, by fetching the coordinates of all 39,613 connective level-2
nodes from the l2cache:

| | nodes | share |
|---|---:|---:|
| Inside the **100 µm** cube | 2,664 | **6.7%** |
| Inside the 200 µm box | 37,977 | 95.9% |

And the enumeration, scored only on what was in scope:

| Connective objects | found by the mip-5 enumeration |
|---|---:|
| All 2,147 | 184 — 8.6% |
| **The 146 with a node inside the 100 µm cube** | **141 — 96.6%** |

## What this corrects

**The enumeration is sound; the mip-5 choice stands.** 96.6% recall on in-scope
connective objects, alongside 99.29% recall of the 279,075 population atoms.
The 8.6% was an artifact of scoring it against objects that were mostly outside
the volume it read. The flat recall across every object-size threshold (7–8.6%
from 1 node to 25) was the tell: a resolution failure improves with size, and
this did not move.

**EXP-071's "2,147 objects, 100% absent from the population" is scoped wrongly
and overstates the count.** The set is *every* level-2 node of these 40 cells
that no population atom claims, across the whole 200 µm box — which includes the
cells' distal arbor far outside the cube, material the population was never
supposed to contain. In the cube itself the figure is **146 objects**, still
100% absent, which is the number that should be compared against a 100 µm
population.

**What survives unchanged.** The claim the experiment turns on is about
*nearest siblings*, and both atoms of every such pair are population atoms
inside the cube:

- nearest-sibling distance, median **3 hops** and ~**1.6 µm** — unaffected, the
  labelled atoms are all in-cube;
- **zero** direct atom-to-atom level-2 contacts — structural, scope-independent;
- the clique-vs-nearest denominator gap (55 hops vs 3) — unaffected;
- the connective material being ordinary cable (median 5 level-2 nodes) rather
  than dust — unaffected, it is a property of the objects, not of the region.

So the *conclusion* — the population omits the cable that joins a cell's
fragments, and the proximity experiments were measuring across that hole — is
not in question. The *magnitude* of the omission, as a count of objects, was
inflated about 15× by the scope mismatch.

## Still open

The path-level version of the claim has not been measured: of the objects lying
specifically **on a nearest-sibling path** (rather than anywhere in the cell),
how many are in-cube and how many are in the population? EXP-071 reports the
aggregate, not the per-path breakdown, and the aggregate is what this correction
rescopes. That measurement is cheap — the cell graphs are cached under
`data/external/cell_l2_graphs/` and need no network — and it is the number
EXP-072 actually depends on.

## For anything downstream

EXP-072 builds its panel from the 100 µm enumeration and scores it against
labelled atoms from the 100 µm population, so both sides are consistently
scoped and this correction does not touch it. Anything that wants EXP-071's
object count at 200 µm needs the enumeration re-run at that side length — about
8× the volume, and the supervoxel mapping is the long pole at ~59 minutes for
the 100 µm cube.
