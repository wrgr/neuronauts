# The join, measured on correct identity — 66 panels

## Result: the full geometric stack puts a cell's own continuation at median rank 5 of 2,440 candidates, first on a third of cells. Every earlier number on this question was measured on objects missing 78.5% of their voxels.

Re-derived with `scripts/build_contact_panels.py` reading objects directly
(`agglomerate=True`) rather than labelling a mip-2 box through a mip-5
supervoxel map. 66 cut-centred panels, median 2,440 candidates each — against
1,000–1,500 when objects were eroded.

## The gapped regime was an artifact, confirmed independently

| | eroded build | correct identity |
|---|---:|---:|
| median true gap | 120 nm | **32 nm** |
| joins with a real gap | 21 of 39 | **10 of 66** |

56 of 66 fragments touch their own continuation. EXP-075's two-regime table
described two amounts of our own erosion, not two kinds of tissue.

## Ranking the true partner

| feature | median rank | top-1 | top-5 | top-20 |
|---|---:|---:|---:|---:|
| distance | 60 | 2/66 | 4/66 | 12/66 |
| along-axis | 56 | 2/66 | 10/66 | 24/66 |
| along × collin | 30 | 11/66 | 18/66 | 29/66 |
| along × collin × proximity | 12 | 12/66 | 25/66 | 35/66 |
| **× caliber** | **5** | **22/66** | **31/66** | **44/66** |

Two things changed from every previous run of this comparison:

**Caliber went from noise to the strongest single addition** — median rank 12 to
5, top-1 12 to 22. Earlier it ranked at median 140 and looked useless. It was
being read from the level-2 cache for objects whose voxels were 79% missing;
computed from each object's own complete cross-section it is the term that
converts near-misses into first place.

**Collinearity is now useful**, cutting median rank 56 to 30 inside the product.
On eroded objects it measured worse than distance (median 220), because a local
axis fitted to a fifth of an object's voxels is not that object's axis.

Restricting to the near set, as EXP-077 did: the partner is within 100 nm on
56 of 66 cells, the near set holds a median of 112 candidates, and the partner
ranks median 8 there, first on 12 of 56.

## What is still hard

Top-1 on 22 of 66 is a third. A soma-seeded grower makes many sequential joins,
so a two-thirds per-decision error compounds away quickly. The residual
difficulty is the one EXP-077 named: many candidates touch the seed at a single
voxel, and distance cannot order a tie — which is why the terms that describe
*how* two objects meet, rather than how near they are, carry the result.

Paired with the abstention measurement (AUC 0.642, weak), the shape of the
problem is now: local geometry ranks well and stops badly.
