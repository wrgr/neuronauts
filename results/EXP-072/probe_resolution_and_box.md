# EXP-072 probes — does read resolution matter, and does the box?

*Two probes, 2026-09-02, both on the 40 um sub-cube where a mip-2 and a mip-5
substrate both exist. Recorded here rather than left in a scratchpad. Neither is
a registered run.*

## 1. Read resolution: does it change the answer?

Same code (`exp072_object_proposal.measure`), same labels, same 40 um cube,
same physical dust floor (0.041 um^3 on synapse-free objects). Only the
segmentation read resolution differs. Bar setting: radius 2 um, panel cap 20,
chained within 3 hops.

| read resolution | objects kept | direct recall | **precision** | reachable labelled | chained recall |
|---|---:|---:|---:|---:|---:|
| mip 5 — 256x256x160 nm | 35,783 | 5.0% | **0.090%** | 532 | 78.7% |
| mip 2 — 32x32x40 nm | 35,610 | **16.1%** | **0.091%** | 598 | 89.8% |

The dust floor equalizes the two substrates (35,783 vs 35,610 objects kept from
63,482 and 192,474 raw), so this compares resolution, not object count.

**Precision is unmoved: 0.090% against 0.091%.** The panel collapse -- the thing
blocking proposal -- is resolution-independent. Chained recall rises, but so
does the reachable set (532 -> 598), which is the same collapse with more of it.

**Direct recall triples, 5.0% -> 16.1%.** That is real and is the first figure
in this line of work to beat EXP-060B's 12% at the same cap. Finer clouds place
an object's surface more accurately, so a true partner ranks higher in a
distance-ordered panel. It is a genuine gain on the quantity a *non-chaining*
proposer would use.

### On 48x48x40 nm

It is not a native scale. The volume offers 8/16/32/64/128/256 nm in xy with z
at 40/40/40/40/80/160. 48x48x40 would be the most isotropic option available
(|log2(z/xy)| = 0.263, against 0.322 for 32x32x40 and 0.678 for 64x64x40), but
reaching it means reading mip 2 at 32 nm and downsampling xy by 1.5 -- so it
costs exactly what mip 2 costs and adds a resampling step.

**Measured cost of mip 2 at 100 um:** the volume read completes, but the
supervoxel-to-root mapping faces **350,540,938 distinct supervoxels at
~10,300/s -- about 9 hours**. The mip-5 equivalent is 73.1M supervoxels in 58
minutes. Roughly 10x, for a precision change of 0.001 percentage points.

**Recommendation, given the blocker:** keep mip 5 for the 100 um substrate,
which is already built, and use the existing mip-2 40 um substrate wherever
geometry accuracy is what matters (scorer features, skeleton-level constraints).
Revisit if direct recall rather than the chained collapse becomes the binding
constraint.

## 2. Does the box hold the connecting path?

A proofread cell's axon can leave the cube and re-enter: the gold cell
864691136011850926 has a full skeleton that is **one** connected component, but
clipped to the 100 um cube it becomes **8**, and every one of the 7 floating
pieces reconnects to the soma only by travelling **90-223 um outside** the box
(6 of 7 axon, radius 116-137 nm; the one dendrite piece 385 nm). Within the box
those pieces cannot be joined by any method.

That raises a denominator question of the kind `results/EXP-060/CORRECTION.md`
has caught twice: are we scoring against links whose connecting path is not in
the box at all? Measured on the 40 held-out cells of EXP-071, walking each
nearest-sibling path through the proofread level-2 graph:

| | paths | share |
|---|---:|---:|
| Entirely inside the 100 um cube | 419 | **85.3%** |
| Leave the cube somewhere | 72 | **14.7%** |
| Had an unpositioned node | 0 | 0% |

So there is a hard **~85% ceiling** on any box-local method here, and 14.7% of
the truth set is unreachable by construction. That is worth carrying as a
reported ceiling, but it does not explain the collapse: the failure is a
precision of 0.09%, not a recall of 85%.

### Why a bigger box is not the fix

The full cable of that one cell spans 819 x 669 x 520 um and would need a
**1,285 um cube** to contain -- about 2,120x the volume of the current cube, and
the mapping cost scales with it. A 300 um box holds 54.5% of the cell's skeleton
vertices; 200 um holds 34.6%; the current 100 um box holds 8.5%. No feasible box
contains a pyramidal axon.

The better framing: box size limits *completeness*, not *proposal quality*. The
links a proposer is asked to find are local (nearest sibling ~1.6 um, EXP-071);
the 14.7% that exit should be excluded from the recall denominator or reported
separately, the same fix the clique-vs-nearest-sibling trap needed.
