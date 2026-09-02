# EXP-071 — Contact adjacency and the connective gap

## Result: passed on 40 held-out cells — the fragments are two or three hops apart, and everything in between is missing from the population

Four experiments had measured the distance between two fragments of one cell and
concluded that geometry cannot propose candidates: EXP-060 (47.4% of true pairs
reachable at 5 µm), EXP-060B (12% recall at a usable panel), EXP-061 (the cone
reaches 40% at 42,000 distractors), EXP-070 (the metric was wrong, the ceiling
moves to 75.7%, the verdict does not).

All four measured the distance between two **synapse-anchored** atoms. This
experiment asks whether that was the right measurement.

## The bar, and why it is a real test

This began as a hand probe on the twelve cells with the most labelled fragments.
Declaring a bar after seeing that would be worthless, so those twelve are
**excluded by id** in the module, and the bar — median nearest-sibling hops ≤ 5,
≥ 50% within 3 hops, ≥ 80% of connective objects absent from the population —
was set from the probe and tested on cells the probe never saw. 483 cells were
eligible; the 40 largest of the remainder were used.

## What it measures

| | Held-out result |
|---|---:|
| Cells measured (disjoint from the probe) | 40 |
| Labelled fragments | 491 |
| **Nearest-sibling distance, median** | **3 L2 hops** |
| Fragments within 3 hops | 60.1% |
| **Direct atom-to-atom L2 contacts** | **0** |
| Connective L2 nodes unknown to the population | 39,613 |
| …resolving to a v117 object | 77.2% |
| Distinct objects holding that material | 2,147 |
| ~~…absent from the population~~ (see below) | ~~2,147 (100%)~~ |
| Median L2 nodes per missing object | 5 |
| Share of connective nodes in objects with ≥2 L2 nodes | 98.6% |

**In nanometres, because hops are not a distance.** The level-2 chunk is
2048 × 2048 × 20480 nm — 10:1 longer in z — so a hop count inherits a
direction-dependent scale, and "3 hops" was the wrong unit to lead with.
Measured on the object clouds for the same 40 cells (546 fragments): the
nearest labelled sibling is a median **1,604 nm** away (lateral 1,223 nm, axial
840 nm); 58% are within 2 µm and **72% within 5 µm**. A single level-2 hop
displaces a median 1,485 nm, and only 20% of hops move further in z than in
xy — the tall chunk works *against* the anisotropy, because a neurite running
in z stays inside one node longer. The imaging itself is 8 × 8 × 40 nm; the
units here are nm on all axes, which is a statement about coordinates, not
about the data being isotropic. (Details: the anisotropy check in the
2026-09-02 QA thread.)

Three things follow, in order of how much they change.

**The zero is structural, and it retires a question rather than answering it.**
Not one pair of v117 atoms in 40 cells shares a level-2 edge, and none ever
will: had the chunkedgraph joined two atoms they would *be* one atom. "Are these
two fragments in contact?" is the wrong query. The measurable quantity is how
much cable lies between them — which is what the rest of this table reports.

**The connective material is ordinary neurite, not debris.** A missing object
contributes a median of 5 L2 nodes, and 98.6% of the connective mass sits in
objects with two or more. For comparison, a *population* atom has a median of 4
L2 nodes. These are perfectly ordinary fragments, the same size as what the
harness already carries. They are excluded by one rule and one rule only: they
own no synapse whose centre falls in the cube.

> **Withdrawn, and replaced by a real test.** The struck row above — "2,147
> objects, 100% absent from the population" — is an **identity, not a result**:
> `objgeom_kall` holds every level-2 node of every population atom, so a node it
> does not know cannot resolve to a population atom, and bar clause 3 could not
> fail. A peer review caught it. The claim it was standing in for has since been
> measured properly at the path level: of the 230 distinct bridging nodes on
> ≤3-hop nearest-sibling paths, **230 (100%) resolve to a real v117 object, none
> existed only after a later edit, and none is in the population.** That could
> have come out the other way — had they been level-2 ids minted by the merge
> that joined the two fragments, the finding would have been circular. See
> `CORRECTION.md`.

**The denominator trap repeated itself, on a new substrate.** Measured over the
same-owner clique the median is **54.8 hops**; measured to the nearest sibling
it is **3**. That is the identical error `results/EXP-060/CORRECTION.md` caught
in EXP-060 — a distribution dominated by distal pairs no proposer should ever be
asked to find. It is worth naming twice because it survived one correction and
reappeared the moment the substrate changed.

## What this says about four earlier experiments

They were measuring the width of a hole the substrate made.

The population is built synapse-first: a v117 object enters it by owning a
synapse whose centre lies in the cube. A passing stretch of neurite with no
synapse of its own never enters — and that is exactly the material joining two
fragments that do. Every proximity measurement then had to reach *across* the
missing piece rather than *to* the neighbour.

This also resolves the one EXP-070 result that never fit. Its corridor probe
found that long spanning links are crossed by dense neuropil but almost never by
another fragment of the same cell — 2.2% of links. That was read as evidence the
partner is genuinely distant. It is this same fact seen from the other side: the
connecting cable is present in the segmentation, it simply is not in the
population, so nothing labelled it.

## What it does not say

**It does not show a proposer can find these fragments.** Hops are counted on
the *proofread* cell's own level-2 graph — ground truth, used here to measure and
never to propose. The claim this supports is narrower and sufficient: the
intervening pieces are ordinary v117 objects that the population skipped, not
hidden data, so they are recoverable by enumeration. Whether a panel built over
the widened substrate actually recovers the spanning links is EXP-060B's
question, re-run, and it has not been answered.

**22.8% of connective L2 nodes did not resolve to a v117 root** and are excluded
from the object counts above. The likely reason is that they did not exist at
v117 — proofreading splits create new level-2 ids — but that has not been
verified, and the 2,147 figure is therefore a floor on the number of missing
objects, not an estimate.

**Forty cells, one 100 µm cube, and the largest cells in it.** Cells with few
fragments were not sampled and could behave differently.

## Consequence for the program

The blocker is the **substrate**, one level upstream of where series B was
looking, and upstream of every scorer and solver in the blocked D–F series.
Candidate generation was being asked to bridge a gap the substrate created.

`scripts/enumerate_region_objects.py` is the fix: a label-blind enumeration of
every v117 object with a voxel in the region, synapse-free ones included. Its
resolution was chosen by measurement rather than caution — recall was scored per
object-size bucket against the known population, and objects with two or more
level-2 nodes are recovered at 100% from mip 2 through mip 5, so mip 5 is used
and single-voxel dust (97% recovered) is the only thing knowingly at risk. On a
12 µm cube it finds **1,476 objects absent from the population against 1,915
present**, holding 6.2% of the segmented volume (at mip 4 the same cube gives
4,279 / 1,939 / 6.7% — the difference is single-voxel dust, see CORRECTION.md
§What the object count means).
