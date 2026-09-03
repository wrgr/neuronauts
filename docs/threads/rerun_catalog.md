# What to re-run now that the substrate is fixed

Today established that three things this repository measured on were wrong, and
all three inflate distances or destroy local geometry:

1. **`object_clouds_mip5.npz` holds one point per supervoxel *visible at mip 5*
   — roughly 20% of an object — and each point is a supervoxel **centroid**, not
   a surface voxel.** Two objects that physically touch can have centroids
   microns apart. Harmless at micron scale; fatal for contact.
2. **`objects_v117_mip5_svmap.npz` knows 21.5% of the supervoxels present at
   mip 2** (measured: 41,380 of 192,391 in one 8 µm box). Anything labelling a
   finer read through it silently discards the rest.
3. **Box placement**: soma-versus-neurite caliber, cube-boundary contamination,
   and clipped "terminals" each produced a confident wrong answer.

The fix for 1 and 2 is not a better map. Read objects directly with
`agglomerate=True` — measured at the same cost (8.0 s against 8.1 s on a
12.5M-voxel box, agreeing on 100.0000% of voxels).

## Why this matters more than a normal bug

Four registered experiments concluded that proximity-based candidate generation
fails, at around **0.09% precision**: EXP-060, EXP-060B, EXP-061, EXP-072. That
finding redirected the whole program — it is why the work moved to soma-seeded
growth, and nine further experiments are blocked behind it.

**Every one of those measured proximity on the eroded centroid clouds.** On
correct object-level contact, the same geometry puts a cell's own continuation
at **median rank 5 of 2,440 candidates**, first on a third of cells. That is not
a small correction to 0.09%.

I do not know yet whether the negative results survive. They may: ranking one
seed's partner within its own panel is an easier question than generating
candidate pairs across a whole cube, and the panels are centred on the true
contact, which a real proposer would not know. But the finding that redirected
the program has never been measured on correct data.

## Priority

| | experiment | why re-run | expected change |
|---|---|---|---|
| 1 | **EXP-072** object proposal on the widened substrate | failed at 0.09% precision on eroded centroid clouds; its failure blocks nine downstream experiments | large — this is the load-bearing negative |
| 2 | **EXP-070** object vs endpoint distance | "passed", but its distance ordering came from centroids; the ordering it validated may be wrong | moderate |
| 3 | **EXP-060B** object-space atom-pair panel | panel sizes and recall both depend on contact distance | moderate |
| 4 | **EXP-061** directed cone vs proximity ball | cones fitted to centroid clouds; a tangent from 20% of an object's points is not its tangent | moderate — collinearity went from useless to useful under exactly this fix |
| 5 | **EXP-074** soma-seeded growth | already known wrong; the rank-4,162 result is centroid-based | large |
| 6 | EXP-071 connective gap | about the *population*, not the supervoxel map | none expected — verify only |
| — | EXP-063 frankenmerge detection | synapse and shape features, no contact distance | none |

## What must be true before any re-run is believed

Today's five failures were all substrate, never the hypothesis. So each re-run
carries its own audit, printed **before** the result:

- identity: objects read with `agglomerate=True`, never through a supervoxel map
- geometry: real voxels, never supervoxel centroids, for anything contact-scale
- placement: boxes audited for boundary contamination and for the confound the
  comparison is vulnerable to (distality, caliber, size)
- power: state the effective sample. The abstention measurement swung from 0.44
  to 0.64 on one feature with 21 terminals — that is noise, and reporting either
  figure as a result was wrong.
