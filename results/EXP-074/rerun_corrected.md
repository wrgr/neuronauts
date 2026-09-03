# EXP-074 re-run on corrected contact geometry

## AUDIT

**Substrate.** The 99 contact panels in `data/external/panels/*.npz`: 8
micrometre boxes read at mip 2 (32 x 32 x 40 nm), 66 centred on a seed/target
contact, 33 on an interior arbor terminal of an already-whole cell. Not
`object_clouds_mip5.npz`, which is what the original ran on.

**Identity resolution.** `CloudVolume(..., mip=2, agglomerate=True,
timestamp=1623399000)` — v117 object identifiers resolved server-side.
`objects_v117_mip5_svmap.npz` is not used. Truth labels for purity come from
`labels_v1822.npz` as before; target membership comes from the cell cards.

**Sample size.** 66 cells that need a join, each with exactly one target
fragment inside its box (65 of the 66 within the 2 micrometre bar radius); 33
already-whole cells. Median 449 candidate objects per panel after the declared
dust floor and the 2 micrometre radius, 909 without the radius. The 66 cells
have 231 seeded target fragments between them, so the box holds 29 percent of
the joins the original was scored on.

**Main confound: scope, and it is large.** The original grew multi-hop across
the whole 100 micrometre cube, choosing among roughly 300,000 objects, and was
scored against all 299 target fragments. This scores a *single hop* inside a
box that is centred on the seed/target contact — so the true partner is
guaranteed present, guaranteed adjacent, and competing with about 450 objects
instead of 300,000. **The recovery figure below is an upper bound on the
corrected-geometry version of the original question, not a like-for-like
replacement for its 0.4 percent.** Everything that follows is reported with
that in mind, and the matched control in the last section is there to separate
what the geometry fix did from what the change of scope did.

**Second confound: ties.** At 32 nm the true partner is usually touching the
seed at the smallest measurable gap, and so are dozens of other objects. The
partner is tied at exactly its own gap with a median of **52** competitors
(maximum 238). Any single ranking number is therefore decided by the tie-break
rather than by the metric, so every table reports three policies: random (mean
of 200 seeds), optimistic (ties resolved in the grower's favour — an oracle, not
a method), and pessimistic.

**As instructed, the cap is swept and the radius is not the axis.** The
original's four radii were identical because the 200-add cap bound first. The
radius still does not bind, even though it now changes the candidate pool by a
factor of five:

| radius | candidates per panel | recovery at cap 64 | recovery at cap 200 | purity at cap 200 |
|---:|---:|---:|---:|---:|
| 0.5 um | 164 | 68.7% | 98.2% | 7.2% |
| 1 um | 258 | 66.0% | 95.2% | 6.2% |
| 2 um | 449 | 65.7% | 95.0% | 5.8% |
| 3 um | 627 | 66.5% | 95.2% | 5.8% |
| unbounded | 909 | 66.5% | 93.7% | 5.8% |

Five-fold more candidates, three points of recovery. The cap is the live axis
and the radius is not, because 95.5 percent of true partners are within 500 nm
and a wider ball only adds distractors.

All numbers below come from one script and one protocol, with 200 tie-break
seeds.

---

## Result: 1 of 3 bars, in a scope generous enough that clearing it proves less than it looks. Purity fails by an order of magnitude, and abstention is now provably out of reach for any distance rule.

Bars, from `docs/threads/exp074_spec.md`: recovery ≥ 0.60, purity ≥ 0.80,
abstention ≥ 0.70. Radius 2 micrometres, dust floor 0.041 cubic micrometres
(1,001 mip-2 voxels), synapse-carriers exempt — the same parameters the
original declared.

### Cap sweep, corrected geometry, random tie-break

| cap | recovery | purity | abstention | labelled adds | unlabelled adds |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.0% | 13.0% | 0.0% | 5 | 60 |
| 2 | 2.0% | 13.0% | 0.0% | 10 | 121 |
| 4 | 4.3% | 13.7% | 0.0% | 20 | 243 |
| 8 | 8.8% | 13.9% | 0.0% | 41 | 486 |
| 16 | 17.6% | 13.7% | 0.0% | 83 | 972 |
| 32 | 38.3% | **14.5%** | 0.0% | 172 | 1,939 |
| **64** | **66.6%** | 12.6% | 0.0% | 344 | 3,879 |
| 128 | 88.0% | 8.3% | 0.0% | 686 | 7,761 |
| 200 | 95.1% | 5.8% | 0.0% | 1,058 | 12,141 |
| all | 100.0% | 3.0% | 0.0% | 2,174 | 29,276 |

Against the original's **recovery 0.4 percent, purity 0.1 percent, abstention
0.0 percent**:

| | original | corrected, cap 64 | corrected, cap 200 | bar | cleared |
|---|---:|---:|---:|---:|:--|
| recovery | 0.4% | **66.6%** | **95.1%** | 0.60 | **yes**, from cap 64 |
| purity | 0.1% | 12.6% | 5.8% | 0.80 | no — best is 14.5% at cap 32 |
| abstention | 0.0% | 0.0% | 0.0% | 0.70 | no, and see below |

Recovery and purity move in opposite directions along the cap, and they never
overlap: at the smallest cap that clears recovery, purity is 12.6 percent
against a bar of 80, and purity's own best anywhere on the sweep is 14.5
percent. There is no cap at which this passes.

The tie policy sets how much of the recovery curve is real:

| cap | recovery, optimistic | random | pessimistic | purity, optimistic |
|---:|---:|---:|---:|---:|
| 1 | 70.8% | 1.0% | 0.0% | 97.9% |
| 8 | 70.8% | 8.8% | 0.0% | 88.5% |
| 32 | 75.4% | 38.3% | 9.2% | 29.9% |
| 64 | 84.6% | 66.6% | 36.9% | 16.9% |
| 200 | 95.4% | 95.1% | 93.8% | 5.9% |

The optimistic column is the interesting one and it is not a result. Taking one
object per cell and resolving ties in the grower's favour recovers 70.8 percent
of partners at 97.9 percent purity — which says the partner **is** at the
minimum gap in about seven cells in ten. Distance simply cannot say *which* of
the tied objects it is.

### Abstention is not a tuning failure

Every one of the 33 already-whole terminal panels has a candidate object
touching the seed at **exactly 32 nm** — median 32, minimum 32, maximum 32.
That is one voxel, the smallest gap the substrate can express. A cut-centred
seed's true partner sits at a median of 32 nm as well (p90 160 nm). The two
populations are not separated at any threshold:

| radius | abstains on already-whole cells | true partners retained |
|---:|---:|---:|
| 32 nm | 0.0% | 69.7% |
| 160 nm | 0.0% | 90.9% |
| 500 nm | 0.0% | 95.5% |
| 2,000 nm | 0.0% | 98.5% |

So **abstention 0.0 percent is a property of the tissue at 32 nm, not a
parameter left untuned.** The original said the stopping rule has to be part of
the grammar; the corrected geometry does not soften that, it closes the door on
the alternative. (16 of the 33 already-whole panels under-list the objects in
their own box — see the note in `../EXP-070/rerun_corrected.md`. That defect
can only *remove* candidates, and all 33 still have one at 32 nm, so restoring
the missing objects would make abstention harder, not easier. The conclusion is
robust to it.)

---

## The matched control: what the geometry fix actually bought

Same panels, same candidate sets, same box, same single-hop rule, same tie
policy — only the point set the distance is measured over changes. "Old" is the
mip-5 supervoxel-centroid cloud the original ranked with; "new" is the mip-2
voxels.

| cap | old recovery | new recovery | old purity | new purity |
|---:|---:|---:|---:|---:|
| 4 | 6.3% | 4.3% | 15.4% | 13.7% |
| 16 | 21.9% | 17.6% | 15.0% | 13.7% |
| 32 | 39.0% | 38.3% | 13.4% | 14.5% |
| 64 | 61.0% | 66.6% | 10.3% | 12.6% |
| 128 | 81.3% | 88.0% | 7.2% | 8.3% |
| 200 | 96.9% | 95.1% | 5.7% | 5.8% |

And the rank of the true partner among the same 469-object pool:

| metric | optimistic | mid-rank | pessimistic | tied at the partner's gap |
|---|---:|---:|---:|---:|
| mip-5 centroid (old) | 19 | **38.0** | 58 | median 13 |
| mip-2 voxel (corrected) | 1 | **47.5** | 82 | median 52 |

**The corrected geometry did not move the result.** It is far more accurate:
the gap to the true partner drops from a median of 205 nm (p90 835) under the
centroid cloud to 32 nm (p90 160), and 46 of the 66 partners turn out to be in
literal contact where the centroid cloud put them a median 160 nm away. But it
is not more *discriminative*, because the same collapse happens to the
distractors — four times as many competitors land in an exact tie with the
partner (median 52 against 13). On expected rank it is marginally worse.

That settles the attribution the original could not make. The original
measured true-partner ranks of 2,052 to 4,162 on a cube-wide pool, median
2,265 — one partner 536 nm away had 4,162 objects closer to the seed. Here the
partner has a mid-rank of 47.5. Decomposing:

* **2,265 to 38** is the change of scope — an 8 micrometre box centred on a cut
  end instead of the whole cube queried from a soma surface. The old centroid
  metric gets there too, so none of this improvement is the geometry's.
* **38 to 47.5** is the change of geometry, and it goes the wrong way.

So the centroid substrate was **not** why EXP-074 failed. It failed because a
soma-seeded, cube-wide, 200-capped nearest-first walk asks distance to do
selection, and distance narrows the field to a few dozen touching objects and
then stops — which is what the original's own follow-up concluded, and what
EXP-070's re-run finds independently from spanning links.

---

## What this establishes, and what it does not

**Establishes.** With correct 32 nm geometry, in a box centred on the join, a
distance-only grower can be made to *contain* the answer — 66.6 percent of true
partners inside 64 adds, 95.1 percent inside 200 — but it cannot *select* it:
purity peaks at 14.5 percent against a bar of 80, and it never declines to grow,
because at 32 nm something always touches. The specification for the grammar is
unchanged and now better supported: pick one continuation out of roughly 50
touching objects, and on an intact arbor pick none.

**Does not establish.** That recovery on the original task is anywhere near 95
percent. This is single-hop inside a box placed on the answer; the original's
299-fragment, multi-hop, cube-wide question is not re-measured here and would
need panels that are not centred on their own targets. Nor does it say anything
about the roughly 3 of 65 partners still missed at cap 200, the 1 partner
beyond the 2 micrometre radius, or the 165 of 231 target fragments that lie
outside these boxes entirely.

**Not attempted.** The brief notes a median true partner rank of 5 in 2,440
"with the full geometric stack" — along-axis, collinearity and caliber. That is
not this experiment. EXP-074 is the distance-only baseline, and distance alone
gives a mid-rank of 47.5. The gap between 47.5 and 5 is exactly the room the
grammar has to work in, and measuring it is the next experiment's job, not
this one's.


---

## Provenance

Cached under `data/external/rerun_corrected/` (gitignored):
`final074.json` (every cap/radius/tie-policy figure above),
`centers.json` (each panel's box centre, recomputed with
`scripts/build_contact_panels.py`'s own logic and verified against three
re-read boxes), `panel_integrity.json` (the read-free completeness check).
The segmentation reads behind the EXP-070 companion are in
`truegap_mst.json`, `mstlinks.json` and `viol_check.json`.
