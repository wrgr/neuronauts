# EXP-072 re-run on the corrected substrate — the failure survives

*2026-09-02. Not a re-registered run: the registered run covers the 100 µm cube
and a mip-2 read of that cube is roughly nineteen hours. This re-measures the
same experiment, same code path, same bar, on the 40 µm sub-cube where both
earlier substrates were already measured, so the three rows differ only in
geometry.*

## Result

**The corrected substrate does not rescue proposal by distance alone.** On real
surface voxels read with `agglomerate=True`, the bar setting gives chained
recall 87.3% at a chained precision of **0.093%** and a median of **575**
reachable labeled atoms. The bar wanted at most 50, and wanted the widened
substrate to beat the synapse-only control by 20 points; it loses to it by 2.9.
Two of the three clauses fail — the same two the registered run failed.

The geometry fix is real and it is visible in the tissue measurements: contacts
that the mip-5 centroid cloud reported at 272 nm are really 60 nm, and the share
of true spanning links that touch within 500 nm doubles. It moves the median
rank of a cell's own continuation from 135 to 107 out of ~978 candidates. That
is not the kind of movement that changes a verdict.

## Audit — read this before the numbers

**Substrate read.** CloudVolume on the v117 chunked-graph path, mip 2
(32 × 32 × 40 nm), `agglomerate=True`, `timestamp=1623399000`. A 40 µm cube
centered at 663 / 591 / 860 µm — the same center and the same box convention
(`floor`/`ceil` of center ± side/2) as `data/substrate/c40um_mip2/`. Read as 180
tiles of 250 × 250 × 200 voxels, 1,565,001,000 voxels, 7.3 minutes on four
threads. Built by `scripts/build_object_clouds_voxel.py`; output kept under
`data/external/c40um_voxel/`.

**How identity was resolved.** From the volume itself. No supervoxel map, so the
21.5%-coverage defect cannot enter. **Verified against the existing build:** the
object set is identical — 192,474 objects on both sides, zero one-sided
difference either way — and per-object voxel counts agree exactly (ratio
mine/existing: median 1.000, p10 1.000, p90 1.000; totals 1.554 × 10⁹ both).

That verification changes what this box can and cannot test. **The eroded-identity
defect is a defect of the 100 µm mip-5 substrate, not of the 40 µm one**: the
40 µm build used its own mip-2 supervoxel map, which was complete. So this box
isolates the *centroid* defect — supervoxel centroid versus real surface voxel —
and the mip-5 arm below carries both defects together.

**Points.** Real boundary voxels (a voxel whose object id differs from a face
neighbor), sub-sampled 1-in-32 with a floor of 24 per object per tile:
**18,290,470 points over 192,474 objects**, against **22,494,502** supervoxel
centroids in the existing cloud. The corrected arm has *fewer* points, so it is
not flattered by density.

**Sample size.** Not a sample of seeds. Every labeled atom in the box is a seed:
**708 labeled atoms** (of 4,801 in the 100 µm cube), **205 minimum-spanning-tree
links**, 192,474 objects of which 35,610 clear the 0.041 µm³ dust floor. The
chained graph is therefore complete *within the box*, which is what the
reachable-set clause of the bar measures.

**Why this box is fair.** It is the box on which both prior arms were already
measured, so the comparison is geometry against geometry with the labels, the
code, the dust floor and the bar held fixed. The check that it is the same
harness: the mip-2 centroid arm reproduces the published probe row to the digit
— 16.1% direct, 0.09% precision, 598 reachable, 89.8% chained.

**Where it is not fair.** It is the central 6.4% of the cube by volume and holds
14.7% of its labeled atoms, so it over-weights the dense center; objects are
clipped at the face; and 55% of the spanning links join atoms more than 2 µm
apart, which no 2 µm proposer can reach in one step (reported separately below).

**The confound I am most vulnerable to.** The corrected cloud is a 1-in-32
sub-sample of boundary voxels, not every voxel. EXP-077 says the signal that
matters lives in the last tens of nanometers — median true gap 32 nm, with many
candidates tying at single-voxel contact. If my sampling blurs that regime, a
neutral result could be my blur rather than the tissue. Measured, against exact
all-boundary-voxel closest approach:

| | median error | p90 error | ordering (Spearman) |
|---|---:|---:|---:|
| this cloud, 1-in-32 boundary voxels | **+72 nm** | +155 nm | **0.998** |
| the mip-5 centroid cloud | +185 nm | +597 nm | 0.847 |

(1,000 object pairs in an 8 µm box; the centroid cloud additionally has no point
at all for 6% of the pairs.) That check is not sufficient, and the sharper one —
run on the near competitors only, the ordering the cap-20 panel actually
consumes — **partly fails**. It is reported in full below under "the near-tie
problem"; the honest limit on this result is stated there and it is not
"exactly neutral".

**Second confound, tested and cleared.** The truth set is derived from the same
geometry it evaluates, so in principle each arm is scored against its own
denominator. Measured: the two mip-2 arms produce **identical** spanning-link
sets — 205 links, 205 shared, 708 labeled atoms shared. Changing centroids to
surface voxels did not change which atoms of a cell are each other's nearest
siblings, so recall is on one denominator after all.

## The bar row

Radius 2 µm, panel cap 20, chained within 3 hops. All three arms, one box, one
code path (`exp072_object_proposal.measure`).

| geometry | direct recall | direct precision¹ | chained recall | chained precision | median reachable labeled |
|---|---:|---:|---:|---:|---:|
| **A** mip-5 supervoxel centroids *(the registered substrate)* | 4.5% | 0.69% | 77.2% | 0.089% | 530 |
| **B** mip-2 supervoxel centroids | 16.1% | 2.63% | 89.8% | 0.091% | 598 |
| **C** mip-2 real surface voxels, `agglomerate=True` | **16.1%** | **2.56%** | **87.3%** | **0.093%** | **575** |

¹ over labeled-atom pairs only, so it is comparable with the chained figure.
Over *all* 14,087 pairs the corrected proposer emits, 33 are true spanning
links — **0.234%**. Most of a panel is unlabeled objects, which cannot be scored
either way.

The synapse-only control, on the same three geometries: chained recall 81.7% /
92.2% / 90.2%. Every arm loses to its own control.

**Verdict against the registered bar, arm C:**

| clause | bar | corrected substrate | |
|---|---|---:|---|
| chained recall at radius 2 µm, cap 20, ≤ 3 hops | > 50% | 87.3% | pass |
| beat the synapse-only control | ≥ +20 points | **−2.9 points** | **fail** |
| median reachable labeled atoms | ≤ 50 | **575** | **fail** |

The registered 100 µm run failed the same two clauses: 63.6% recall, control
71.1%, gain −7.4 points, 1,586 reachable. **The failure survives.**

## What the correction did change

It is not that the geometry fix does nothing. It fixes the geometry.

**True contact becomes visible.** The closest approach of the 205 spanning
links, measured on each geometry:

| | p10 | median | within 500 nm | within 2 µm |
|---|---:|---:|---:|---:|
| mip-5 centroids | 272 nm | 4,019 nm | 18.3% | 42.6% |
| mip-2 surface voxels | **60 nm** | 3,435 nm | **36.6%** | 44.9% |

The share of true links that actually touch doubles. This is the same correction
EXP-077 found on seeded panels (median true gap 120 nm → 32 nm), reproduced here
over a whole sub-cube rather than 66 seeds.

**And the ranking barely moves.** For each of the 404 directed link-ends, where
does the true partner sit in the seed's distance ordering?

| geometry | partner within 2 µm | median rank | top-1 | top-20 | candidates within 2 µm (median) |
|---|---:|---:|---:|---:|---:|
| A mip-5 centroids | 148 / 404 | 135 | 0 | 10 | 824 |
| B mip-2 centroids | 164 / 404 | 110 | 1 | 37 | 930 |
| C mip-2 surface voxels | 167 / 404 | **107** | 2 | 37 | **978** |

(The last column is counted the way the experiment counts, from ≤ 64 seed
points. Counted exactly, from every boundary voxel, it is a median of 1,388 —
see the near-tie section.)

Correcting the resolution (A → B) is worth 25 rank places. Correcting centroids
to surface voxels at that resolution (B → C) is worth 3. Meanwhile the panel it
has to rank *grows*: a labeled atom has a median of 978 objects within 2 µm on
correct geometry, because correct geometry finds contacts the eroded one
missed. That is the whole shape of the result — the fix adds true contacts and
false contacts at the same rate.

**Recall on the reachable subset.** 55% of the spanning links join atoms more
than 2 µm apart — the connective cable between them is unlabeled, or the path
leaves the box — so they are unproposable in one step at the bar radius. Scored
only on the 91 links that *are* reachable: corrected 33/91 = **36.3%**, mip-2
centroids 33/91 = 36.3%, mip-5 centroids 9/91 = 9.9%. Even given a partner it
can reach, distance alone puts it in a panel of 20 barely a third of the time.

## The near-tie problem, and the honest limit on this result

The audit above shows the sampled cloud reproduces exact ordering at Spearman
0.998 over random pairs. Run instead on only the competitors that are *exactly*
within 2 µm of a seed — the set the panel is drawn from — on 24 seeds across
three 8 µm boxes, against every boundary voxel of every object:

| | median error | ordering (Spearman) | **top-20 set shared with exact** | top-5 shared |
|---|---:|---:|---:|---:|
| the cloud's own points | +52 nm | 0.992 | **2 of 20** | 0 of 5 |
| with the experiment's ≤ 64 seed points | +301 nm | 0.918 | 2 of 20 | 0 of 5 |

**The bulk ordering survives; the top of it does not.** A median of **1,388**
objects — more than half of everything in the box — lie within 2 µm of a given
seed, and a large fraction of those are in actual contact, so the first twenty
places are a field of near-ties that 52 nm of blur is enough to reshuffle. The
specific twenty objects in any panel here are not the exact twenty. The second
row says the same is true of the registered method itself: the ≤ 64-point seed
cap costs 301 nm, six times more than the point sampling does, and it applies to
every arm equally.

So this measurement cannot claim "the corrected top-20 panel contains what an
exact top-20 panel would contain". What it can claim is the *rate*, and that
rate has independent corroboration: EXP-077's contact panels, built with much
denser geometry (20,000 seed voxels against 4,000 candidate voxels per object,
no cube-wide budget to respect), rank the true partner in the top 20 by distance
alone on **12 of 66 cells — 18%**. This cube-wide run gets **16.1%**. Two
different geometries, two different framings, the same answer to within a couple
of points. That agreement is the reason I am willing to state the verdict; the
top-20 instability is the reason it is stated as a rate and not as a panel.

## Reading

The negative was not an artifact of the broken substrate. It was already the
right answer about **distance alone**, and it stays the right answer when the
distance is measured between real surfaces at 32 nm.

This is consistent with, not contradicted by, the seeded-panel result that
prompted the re-run. That result — a cell's own continuation at median rank 5 of
2,440 — comes from the *full geometric stack*, along-axis × collinearity ×
proximity × caliber. Distance alone in those same panels ranks the partner at
median 60 (EXP-077's own table), which is the same order as the median 107 found
here over a whole sub-cube. The gap between rank 107 and rank 5 is the scoring
terms, not the substrate.

So the redirect that EXP-072 caused stands, but its stated reason should be
amended. It is not "the substrate was too coarse". It is: **contact is a
generator and not a selector**, at any resolution, and the terms that describe
*how* two objects meet have to do the selecting. EXP-072's own preamble said "if
a substrate fix alone moves recall, that is the finding; if it does not, no
scorer downstream was ever going to rescue it" — the second half of that
sentence is now the part that is wrong. The substrate fix did not move it, and
the scorer terms demonstrably do (EXP-077, top-1 on 22 of 66 against 2 of 66 for
distance). The nine experiments blocked behind EXP-072 are blocked behind a
scorer, not behind a substrate rebuild.

## Reproducing

```
python scripts/build_object_clouds_voxel.py --side-um 40      # 7.3 min, 4 threads
```
then `exp072_object_proposal.measure()` against
`data/external/c40um_voxel/object_clouds_mip2_voxel.npz` with
`objects_v117_mip2_voxel.npz` beside it (its `in_population` flag is set by the
same rule `enumerate_region_objects.py` uses — a v117 root owning a synapse
whose center is in the region). `object_clouds_mip5_clip40.npz` in the same
directory is arm A, the 100 µm mip-5 cloud clipped to this box. The three-arm sweep is 9 minutes of
compute; the cross-truth and rank pass another 4; the two audits above about 20,
most of it volume reads.
