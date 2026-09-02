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

## Closed: the "100% absent" clause was an identity, and the path-level test now stands in its place

A peer review found that bar clause 3 — "at least 80% of the objects holding the
connective material absent from the population" — **could not fail**.
`objgeom_kall` holds every level-2 node of every population atom, so a node it
does not know cannot resolve to a population atom. `frac_objects_missing == 1.0`
was an identity, not a measurement, and the evaluation presented it as a result.
That clause is withdrawn as evidence.

The claim it was standing in for is path-level, and it had not been measured:
on the nearest-sibling paths, is the bridging material (a) real v117 objects the
population omitted, or (b) level-2 ids re-minted by the very merge edit that
joined the two fragments — which never existed at v117 and so could not be in
any population? (b) would make the whole finding circular.

Measured, on the 40 held-out cells, from the cached graphs with one `roots_at`
call:

| | |
|---|---:|
| Nearest-sibling paths of ≤3 hops | 295 |
| Interior nodes on them, unknown to the population | 351 (230 distinct) |
| Interior nodes on them that a population atom owns | **0** |
| …of the 230, **resolving to a real v117 root** | **230 (100%)** |
| …that did not exist at v117 (re-minted by a later edit) | **0** |
| Distinct v117 objects they belong to | 230, **none in the population** |

So (a): every short-path bridge is an ordinary v117 object that existed at v117
and that the synapse-anchored population did not enumerate. The claim holds on
the paths that matter, on evidence that could have come out the other way.

Two honest riders. The zero known-interior count is *forced* — two v117 atoms
never share a level-2 edge, so a short path's interior is unknown by
construction; what was not forced, and is the real result, is that those nodes
resolve at v117 rather than being edit artifacts. And this covers ≤3-hop paths
(295 of 491 fragments); longer paths run substantially through *known* population
atoms, most of them mixed-lineage, which is a different situation and is not
covered by this test.

## What the object count means

The enumeration's raw count — "909,888 v117 objects in the 100 µm cube" — reads
as a stable count of objects. It is not: it is the count of distinct v117 root
ids with at least one voxel at mip 5, and it is dominated by debris.

| threshold | objects | in population | new |
|---|---:|---:|---:|
| ≥1 voxel | 909,888 | 277,081 | 632,807 |
| ≥5 voxels | 413,539 | 236,590 | 176,949 |
| ≥10 voxels | 293,790 | 189,779 | 104,011 |
| ≥100 voxels | 65,437 | 61,153 | 4,284 |

(One mip-5 voxel is 0.0105 µm³.) 287,461 of the 909,888 (31.6%) are
single-voxel; the 496,349 objects under 5 voxels together hold 0.91% of the
segmented volume, while the 279,075-atom population holds 94.4% of it.

The count is also resolution-dependent. At 40 µm, mip 5 finds 63,482 objects
and mip 2 finds 192,474 — a 3× swing from resolution alone — while the counts
that matter for anything downstream barely move: ≥0.1 µm³ gives 5,792 at mip 5
against 6,041 at mip 2, and ≥1 µm³ gives 256 against 263.

**The stable, honest statement:** the enumeration finds ~104,000 objects
≥0.1 µm³ (and ~4,500 ≥1 µm³) that are absent from the population, holding
5.6% of the segmented volume. The raw count of root ids touched (909,888 at
mip 5) is dominated by sub-5-voxel debris and is not a stable property of the
cube.

## For anything downstream

EXP-072 builds its panel from the 100 µm enumeration and scores it against
labelled atoms from the 100 µm population, so both sides are consistently
scoped and this correction does not touch it. Anything that wants EXP-071's
object count at 200 µm needs the enumeration re-run at that side length — about
8× the volume, and the supervoxel mapping is the long pole at ~59 minutes for
the 100 µm cube.
