# EXP-077 — Why is the gapped regime hard?

## Result: because it is not a regime. The gap is our measurement, not the tissue. Correcting two defects in `build_contact_panels.py` drops the median true gap from 120 nm to 32 nm and leaves 6 of 39 joins gapped rather than 21. Nothing bridges the six that remain: 314 objects touch both fragments and not one belongs to the cell. The task framing is unchanged; the difficulty is not the gap, it is the 83 objects that touch the seed at one voxel.

EXP-075 split soma-seed joins by the measured gap between the seed fragment and
its labelled continuation: under 100 nm the along-axis term reached median rank
6, at a median 431 nm nothing worked, and that second class was about half of
all joins. This asks what the gapped class actually is.

It is an artifact of how the panel measured distance. Both defects are in our
own code, both inflate gaps, and both were verified with reproductions that
could have come out the other way.

## Defect 1 — the panel labels its voxels through a 21% sample of the segmentation

`objects_v117_mip5_svmap.npz` maps supervoxel to v117 root. It was built by
reading the cube at mip 5 — 256 × 256 × 160 nm — and keeping the supervoxel id
of each voxel read, so it holds only the supervoxels that survived a 8 × 8 × 4
downsample. `build_contact_panels.py` labels its mip-2 box through that map and
**drops every voxel the map does not know** (`rt = to_root(svid); ok = rt > 0`).

Measured two independent ways:

| | |
|---|---:|
| supervoxels in an 8 µm mip-2 box the map knows | **21.3%** (median over 39 boxes) |
| …of those, agreeing with `chunkedgraph.get_roots(sv, v117)` | **100%** |
| seed **voxels** in the box that survive the map | 44.3% (median) |
| an object's supervoxels inside the cube, from `get_leaves(root, bounds)`, that the map holds | **19.1%, 20.3%, 19.1%** on three roots |
| …of those, that are genuine leaves of that root | 100% |

The `get_leaves` check involves no volume read and no code of mine beyond a set
intersection, and it agrees with the box measurement. The map is **incomplete,
not wrong**: every entry it has is correct, and it has about a fifth of them.
Each object therefore enters a panel eroded to a scatter of its own voxels, and
a distance between two eroded point sets is an overestimate of the distance
between the objects.

This also traces the first mis-diagnosis of this session. Refetching one box and
resolving it at v1822 showed 41,366 of the cell's 72,630 voxels carrying v117
root 0 — which reads as a large unenumerated object bridging the gap. Asking the
chunkedgraph what those 818 supervoxels actually are returned **two** roots: the
seed and the target. There was no third object; there was a hole in our map.

## Defect 2 — the thinning step truncates instead of subsampling

Both point sets are then capped:

```python
sub = Sv if len(Sv) <= 20000 else Sv[:: len(Sv) // 20000][:20000]
Qs  = Q  if len(Q)  <=  4000 else Q[:: len(Q) // 4000][:4000]
```

`np.nonzero` returns voxels in raster order, so the stride walks along x. When
the count sits between the cap and twice the cap the stride rounds to 1 and
`[:4000]` keeps the **first 4,000 voxels**, a slab at the low-x end of the
object; at larger counts the floor-divided stride still overshoots and the slice
chops the tail. Across the 39 join-needing panels, **24 targets exceed the
4,000 cap**, and the kept fraction of their x-extent runs from 0.53 to 0.97.

## Corrected: the gap collapses

Each box was reread at mip 2 and every supervoxel in it resolved at v117 for
object identity and at v1822 for proofread-cell ownership. Gaps come from an
exact Euclidean distance transform seeded on the seed's voxels — no thinning, no
nearest-neighbour approximation — and calibers from a distance transform of the
label boundary, so an object's caliber is its own inscribed radius.

| | stored panel | corrected |
|---|---:|---:|
| median gap, seed to its labelled continuation | 120 nm | **32 nm** |
| joins with a gap ≥ 100 nm ("gapped") | **21 / 39** | **6 / 39** |
| joins one mip-2 voxel apart | 11 / 39 | **30 / 39** |
| candidate objects in the box | 1,185 | 2,389 |

Fifteen of the twenty-one gapped joins are touching. Some examples, with each
defect also applied on its own:

| cell | stored | thinning only | erosion only | exact |
|---|---:|---:|---:|---:|
| 73296677 | 1,204 | 2,179 | 1,204 | **32** |
| 48130450 | 1,108 | 1,067 | 1,108 | **32** |
| 57669039 | 983 | 32 | 983 | **32** |
| 62922224 | 903 | 101 | 32 | **32** |
| 69764601 | 534 | 32 | 32 | **32** |
| 15352054 | 425 | 32 | 32 | **32** |
| 56920938 | 890 | 878 | 890 | **878** |

Neither defect alone explains every case and they compound in both directions —
57198692 reads 32 nm as built and 520 nm under thinning alone, because the
erosion happened to keep the touching voxel that the truncation would have cut.
The rank correlation between the stored gap and target size, caliber or mapped
fraction is weak (Spearman −0.18 to −0.32): which panels got hurt is close to
arbitrary, which is why the damage looked like a biological regime.

**32 nm is one mip-2 voxel, so it deserved a check at native resolution.** A
2 µm box at mip 0 (8 × 8 × 40 nm) around the contact, on the first five panels:
the seed and its continuation are **8 nm apart — one native voxel — in all
five**. The contact is tissue adjacency, not a downsampling artifact.

## Question 2: nothing sits in the gap, and the framing does not change

For the six joins with a real remaining gap, every object in the box was scored
on its exact distance to the seed **and** to the target, and each object's
proofread owner at v1822 was read off the chunkedgraph.

| cell | gap | compartment | unsegmented in the corridor | objects touching both | of those, owned by the cell |
|---|---:|---|---:|---:|---:|
| 81185697 | 3,066 nm | axon | 0.0% | 122 | **0** |
| 56920938 | 878 nm | axon | 0.0% | 82 | **0** |
| 73516581 | 451 nm | axon | 0.0% | 27 | **0** |
| 76419401 | 374 nm | dendrite | 0.0% | 31 | **0** |
| 56746690 | 160 nm | dendrite | 24.4% | 21 | **0** |
| 67137111 | 120 nm | dendrite | 17.7% | 31 | **0** |

**Zero of 314.** There is no unenumerated object, no third fragment, no missing
tissue that a wider enumeration would recover. The material between the two
fragments is other cells' processes — 8 to 34 distinct objects inside a 250 nm
cylinder — plus, in two cases, genuinely unsegmented voxels. So the join is one
join, and "recover the continuation of this seed" remains the right question.

Two riders, because the test is sharper than the headline. First, in dense
neuropil *something* always touches both: 21 to 122 objects do, in every case.
"Find the object that touches both" is not a usable bridge test, and the fact
that none of them is the cell is the result, not the setup. Second, three of the
six have a **small piece of the same cell hanging off the seed** but not
reaching the target — 439 voxels at 32 nm (76419401), 546 at 40 nm (81185697),
32 voxels at 72 nm (73516581). Those are stubs, not bridges: a grower should
absorb them, and doing so shortens the first step without closing the gap. For
those cells the labelled target is two hops rather than one, which is a
refinement of the target definition, not a change of framing.

## What is actually hard

Distance does not fail because the partner is far. It fails because **83 objects
(range 24–334) touch the seed at exactly one voxel**, and 110 (range 35–401)
come within 100 nm. Distance cannot order a tie.

Ranking the true partner among ~2,389 candidates, average rank on ties, as a
fraction of the field so the two panel versions are comparable:

| | distance | along | collin | along×collin | along×collin×prox |
|---|---:|---:|---:|---:|---:|
| panels as built (n=39) | 0.056 | 0.014 | 0.147 | 0.010 | 0.026 |
| **corrected (n=39)** | 0.024 | 0.020 | 0.137 | 0.012 | **0.005** |
| corrected, gap < 100 nm (n=33) | 0.020 | 0.016 | 0.128 | 0.010 | **0.004** |
| corrected, gap ≥ 100 nm (n=6) | 0.098 | 0.114 | 0.215 | 0.073 | 0.043 |

The genuinely gapped six are still the hard ones — an order of magnitude worse
on every term — but they are 15% of joins, not half, and six panels carry no
weight worth quoting to three digits.

The operational question is what happens **after** distance has done its work.
On the 33 panels where the partner is itself within 100 nm, ranked only against
the other objects within 100 nm (median near-set size 110):

| feature | median rank in the near set | median per-panel AUC |
|---|---:|---:|
| distance | 50 | 0.590 |
| caliber agreement | 32 | 0.733 |
| collinearity | 25 | 0.757 |
| along-axis | 22 | 0.897 |
| **along × collin** | **10** | **0.913** |
| **along × collin × proximity** | **10** | **0.919** |

Local geometry does separate the true continuation from the touching crowd, at
roughly AUC 0.92 — measured on correct voxel identity, exact distances and axes
fitted to full local point sets rather than to a 205-stride sample. That is a
different picture from "nothing works here", and it is a within-panel
discrimination number, not a stop rule.

## What this voids, and what it does not

- **EXP-075's two-regime table does not stand.** Both columns were the same
  regime measured with different amounts of damage, and the 431 nm median that
  defined the gapped column is an artifact. Its *ranking* numbers were computed
  on eroded, truncated geometry over a candidate field missing about half its
  objects.
- **EXP-075's stop-rule AUC of 0.304 is untouched by this and still unverified.**
  It rests on whole-cell panels built by the same code, so it inherits both
  defects on top of the placement error EXP-076 already found. This experiment
  did not re-measure it.
- **EXP-074's second confound was fixed only halfway.** It correctly identified
  the mip-5 centroid clouds as too coarse and moved to "real voxels at mip 2",
  but that mip-2 read was labelled through the mip-5 map. Its four-cell
  comparison (gaps of 290, 40, 534, 32 nm) reads 32, 32, 32, 32 nm here. Its
  conclusion — distance is a generator and not a selector — survives, and gets
  stronger: the tie at one voxel is wider than it looked.
- **EXP-071 is untouched and is not the same finding.** Its connective objects
  are real material the *population* omitted; the enumeration already covers
  them. This is about the supervoxel map, which is downstream of both.
- **Anything measured on `object_clouds_mip5.npz` inherits this.** The clouds
  carry one point per *mapped* supervoxel, so they are a ~20% sample of each
  object's supervoxel decomposition. That is harmless when the question is
  "within 2 µm?" and fatal when it is "do these touch?".

## Limits

- 39 join-needing panels, one 100 µm cube. They are all the panels the builder
  produced; 67 cells need a join and 28 were skipped by its own filters, so this
  is not a random sample of joins.
- The corrected boxes are still centred where the mip-5 clouds put them. That
  can only have widened the measured gaps, so it does not threaten the collapse,
  but for the six residual cases the closest approach is an in-box minimum. The
  proofreaders' own link point lies inside the box in 38 of 39 panels (median
  548 nm from centre), so the box does cover the site where the join was made.
- The native-resolution check covers five panels, not all thirty-nine.
- Ownership at v1822 identifies a bridge only if proofreading merged it into the
  cell. A piece of the true continuation that no one ever merged would read as
  another cell's, which would make the "zero of 314" a floor rather than an
  exact count. Against that: all six gaps sit where a proofreader *did* join the
  two fragments, so the material they judged continuous is in the cell.
- Caliber near the contact is undefined when the gap exceeds 3 µm (no seed voxel
  within 1,500 nm of the midpoint); one panel falls back to the seed's caliber
  over the whole box.

## Reusable output

`scripts/probe_exp077_true_gap.py` writes corrected panels to
`data/external/panels_v2/` (39 panels) — same fields as `data/external/panels/`, plus
`frac_cell` (fraction of the object's voxels the proofread cell owns at v1822),
an exact inscribed-radius caliber, the corridor tally, and the two single-defect
gap variants for attribution. The bridge test of question 2 is cached in
`data/external/exp077_bridge.json`. `build_contact_panels.py` is unchanged; the
fix it needs is a supervoxel map read at mip 2 and a thinning step that
subsamples rather than truncates.
