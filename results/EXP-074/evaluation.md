# EXP-074 — Soma-seeded growth, distance only

## Result: failed — recovery 0.4%, purity 0.1%, abstention 0.0%. But the sweep did not measure what it was built to measure, and two confounds in the distance itself are now under test.

This is the baseline the grammar was supposed to improve on: start at a soma
fragment, repeatedly absorb the nearest unclaimed object, and see how much of
the seeded target comes back. It fails on all three bars. The interesting part
is why, and what the failure does *not* establish.

## The radius sweep is degenerate

| radius | recovery | purity | abstention | added/cell (median) |
|---|---:|---:|---:|---:|
| 0.5 µm | 0.4% | 0.1% | 0.0% | 200 |
| 1 µm | 0.4% | 0.1% | 0.0% | 200 |
| 2 µm | 0.4% | 0.1% | 0.0% | 200 |
| 3 µm | 0.4% | 0.1% | 0.0% | 200 |

Four radii, identical to the digit. That is not a coincidence and not a bug in
the radius plumbing: **the add-cap binds long before any radius does.** Growth
pops nearest-first, and the 200 nearest objects to a soma fragment are all far
closer than 500 nm, so every radius sees the same 200 objects. The sweep
measured one configuration four times. Any future version has to vary the cap,
not the radius — the radius is not the active constraint at this density.

## Why recovery is 0.4%: the true partner is not nearby in rank

Measured directly, on twelve cells that need at least one join — the rank of
the first true target in the distance-ordered candidate list:

| | |
|---|---|
| cells where the partner is in the top 64 nearest | 4 of 12 |
| rank of the partner when found | 4162, 2052, 2265, 2455 |
| a partner 536 nm away | has **4,162 objects closer to the seed than it is** |

With a 200-add cap the grower never reaches rank 2,000. Uncapped, precision
would be about 1 in 2,500. Recovery 0.4%, purity 0.1% and abstention 0.0% are
all the same fact: at a soma there is always another object within 500 nm, so
the grower never stops and almost nothing it absorbs is the cell's own.

## One clause tested and only weakly supported

A cell body is a large surface touching thousands of passing processes, which
is the worst place to rank neighbors by distance; a real join happens where a
cable was cut. Restricting the query to the sparse decile of each seed's own
points — a skeleton-free proxy for a cut end, since interior and soma-surface
points sit in dense company and an end point does not — moves the median rank
from **2265 to 1182**. A 2x improvement where roughly a thousand-fold is
needed, and the candidate pool actually grows (3,907 to 5,817) because sparse
points sit on the periphery and reach into more neighborhoods. Querying from
cut ends is a real term, not the missing clause.

## Two confounds in the measurement, tested

Two properties of the measurement rather than of the tissue were unaccounted
for, and both inflate exactly this ranking. They affect EXP-070 and EXP-072
identically.

1. **Every point is a supervoxel centroid, not a surface voxel.**
   `build_object_clouds.py` accumulates a coordinate sum per supervoxel, so
   "closest approach" above is centroid-to-centroid. Two objects that
   physically touch can have centroids microns apart.
2. **The clouds are mip 5 — 256 x 256 x 160 nm** — ranking gaps of
   500-3,000 nm.

Both were tested directly: a mip 2 read (32 x 32 x 40 nm, real voxels) in an
8 µm box around each seed/target contact, re-ranking the same competitors both
ways. Same box, same object universe, two ways of measuring.

| cell | centroid, mip 5 | voxel, mip 2 |
|---|---:|---:|
| 91067002 | 405 (gap 256 nm) | **158** (gap 290 nm) |
| 96650282 | 82 (gap 160 nm) | **33** (gap 40 nm) |
| 69764601 | 72 (gap 160 nm) | 184 (gap 534 nm) |
| 75888840 | 206 (gap 302 nm) | **28** (gap 32 nm) |

The centroid objection was right about the geometry and wrong about the
consequence. Real voxels recover true contact gaps of 32 and 40 nm where
centroids reported 160-302 nm — those objects touch, and the mip 5 cloud could
not see it. But the median rank moves only 144 to 95, and one cell gets worse.
About 1.5x, not the thousand-fold that would change the verdict.

## What this establishes

**Distance narrows the field by roughly an order of magnitude and then stops.**
With correct geometry at 32 nm, some 30-180 objects are as close to a seed as
its own continuation is. In dense neuropil dozens of unrelated processes touch
any given one; contact is necessary and nowhere near sufficient. Distance is a
legitimate candidate generator and not a selector, which is the same shape of
result EXP-070 and EXP-072 reached by other routes.

That sets the specification for the grammar. It does not have to search the
cube, and it does not have to fix a broken distance metric. It has to pick the
true continuation out of a contact panel of order tens — and, on the 36 of 103
cells that are already whole, pick nothing at all. Abstention is not a
refinement of this baseline; the baseline abstains 0.0% of the time, so the
stopping rule is load-bearing and has to be part of the grammar rather than a
threshold bolted on afterward.

## Reusable output

Per-cell contact panels at mip 2 are cached under `data/external/panels/`
(object id, true gap in nanometres, target membership, voxel count) so EXP-075
can be developed against real panels without re-reading the volume.
