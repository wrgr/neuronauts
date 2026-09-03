# EXP-070 re-run on corrected contact geometry

## AUDIT

**Substrate.** Two, because one population cannot answer both questions.

1. **99 contact panels**, `data/external/panels/*.npz` — 8 micrometre boxes read
   at mip 2 (32 x 32 x 40 nm), 66 centred on a seed/target contact and 33 on an
   arbor terminal. 64,726 seed-to-candidate pairs in which the candidate also
   carries level-2 geometry, out of 217,087 panel candidates (29.8 percent; the
   rest are objects too small to have been enumerated as atoms).
2. **28 of EXP-070's own 350 tier-10 spanning links**, re-read from the
   segmentation. Four per object-gap stratum, drawn with a fixed seed, box 10
   micrometres on a side centred on the level-2 closest approach.

**Identity resolution.** `CloudVolume(..., mip=2, agglomerate=True,
timestamp=1623399000)`. Voxels carry v117 object identifiers resolved
server-side. `objects_v117_mip5_svmap.npz` is not used anywhere in this re-run.

**Sample size.** 64,726 pairs across 99 panels for the metric comparison;
38,673 of those also have an endpoint gap. 28 spanning links for the
reachability re-measurement.

**Main confound.** The panel pairs all sit inside a box centred on a real
contact, so 97.5 percent of them are within 5 micrometres at true distance.
That population can rank metrics against each other; it *cannot* test the 5
micrometre reachability rate, because its answer is fixed by construction. The
28 link reads exist for that, and they carry their own limit: a gap measured
inside a box is an upper bound on the pair's true closest approach, so "within
5 micrometres" can be under-counted but never over-counted.

**One correction to the premise of this re-run.** EXP-070 never read
`object_clouds_mip5.npz`. Its inputs are `data/substrate/geom/objgeom_k10.npz`
and `objgeom_kall.npz` — level-2 cache *representative coordinates*, one per
level-2 node. Those are real voxels of the object, not supervoxel centroids, so
the specific defect described in the brief does not apply to this experiment.
The general defect does: a level-2 representative coordinate is one point per
level-2 chunk, and objects that touch are reported hundreds of nanometres to
several micrometres apart. Everything below measures that instead.

---

## Result: the pass survives, both gates reproduce, and the conclusion is strengthened — but two of the sentences around it do not survive

### Gate 1 — the control

The tier-10 object-metric spanning tree was rebuilt from scratch here:
**350 links, 75.7143 percent within 5 micrometres, median gap 1,314.4 nm.**
EXP-070 recorded 350, 0.757143, 1314.4. Same to the digit, from separate code.
The code path is sound.

### Gate 2 — the ordering invariant

Object gap must not exceed endpoint gap, because an atom's endpoints are a
subset of its nodes. On 38,673 panel pairs — an entirely different pair
universe from the one EXP-070 ran on — there are **0 violations**.

It is worth saying plainly what that means. The invariant is a consequence of
set nesting, so it cannot fail unless the index is corrupt. EXP-070 passed on a
consistency check, not on a claim about tissue. Nothing here changes that, and
nothing here makes it more informative than it was.

### The substantive ordering — and it is wider than EXP-070 claimed

Each proxy, measured against the true contact gap on the same pairs, in the same
box:

| Point set | pairs | median above truth | p90 | rank agreement with truth (Spearman, per panel) |
|---|---:|---:|---:|---:|
| Endpoints (degree-1 skeleton nodes) | 38,673 | **1,247 nm** | 3,204 | 0.722 |
| Object (all level-2 nodes) — EXP-070's metric | 64,726 | **645 nm** | 1,346 | 0.948 |
| mip-5 supervoxel centroids | 62,826 | **224 nm** | 502 | 0.981 |

Restricted to pairs that are *actually in contact* (true gap at or below 64 nm,
7,692 pairs), the same three point sets report:

| Point set | median reported gap between touching objects | share it places beyond 1 micrometre |
|---|---:|---:|
| Endpoints | 1,638 nm | 73.8% |
| Object (level-2) | 689 nm | 22.6% |
| mip-5 centroids | 205 nm | 1.2% |

So EXP-070's central ordering claim — endpoints are the worse point set,
object nodes the better one — **holds, and the margin is larger than the
reachability table suggested.** The endpoint metric puts three quarters of
genuinely touching object pairs more than a micrometre apart.

The unexpected part is the third row. The mip-5 supervoxel-centroid cloud is
about **three times tighter than EXP-070's object metric** and ranks candidates
closer to truth. The reason is point density, and it is checkable without any
distance at all: over the 277,081 objects that appear in both tables, the
level-2 set carries a median of 4 points per object and the mip-5 cloud a
median of 16 — a median 3.5x more points, and more points on 87.7 percent of
objects. A supervoxel is a smaller unit than a level-2 chunk, so sampling one
point per supervoxel samples the object more finely even at mip 5.

EXP-070's recommendation ("object distance should replace endpoint distance
everywhere downstream") is therefore right and understated: on this evidence
the level-2 node cloud is itself second-best among the coarse substrates
already built.

### Reachability — the number that carried the conclusion

28 tier-10 spanning links, stratified by their object-metric gap, re-read:

| object gap | links (of 350) | weight | sampled | true gap at or below 5 um | true gaps measured (nm) |
|---|---:|---:|---:|---:|---|
| 0 – 500 nm | 63 | 0.180 | 4 | 4/4 | 32, 32, 32, 32 |
| 500 – 1,000 | 75 | 0.214 | 4 | 4/4 | 120, 32, 32, 32 |
| 1,000 – 2,000 | 87 | 0.249 | 4 | 4/4 | 32, 40, 32, 32 |
| 2,000 – 5,000 | 40 | 0.114 | 4 | 4/4 | 32, 32, 32, 32 |
| 5,000 – 10,000 | 16 | 0.046 | 4 | 1/4 | 7,898 · 7,732 · **4,636** · 5,012 |
| 10,000 – 20,000 | 24 | 0.069 | 4 | 0/4 | partner absent from a ±5 um box, all four |
| above 20,000 | 45 | 0.129 | 4 | 0/4 | partner absent from a ±5 um box, all four |

| | within 5 um |
|---|---:|
| endpoint metric says | 64.9% |
| object metric says (EXP-070's headline) | 75.7% |
| **true contact, stratum-weighted from these 28 reads** | **76.9%** |
| bounding the one stratum that splits at 0/4 and at 4/4 | 75.7% – 80.3% |

**EXP-070's 75.7 percent is very nearly the right number.** Its conclusion —
object distance is the correct quantity, it is strictly tighter, and it does
not approach the 90 percent bar, so proximity's failure is not an artefact of
measuring from skeleton tips — **survives intact on real geometry.** The
ceiling is real: roughly a quarter of spanning links are genuinely more than 5
micrometres apart in tissue, and no change of point set reaches them.

---

## What does not survive

**1. The gap numbers are not tissue distances, and the write-up reads as if
they were.** EXP-070 reports a median spanning-link gap of 1,314 nm and
percentiles from 315 nm to 62 micrometres, and reasons about "how far apart"
fragments are. Every one of the 16 sampled links that the object metric placed
under 5 micrometres has a true gap of **32 to 120 nm** — one or two voxels.
They touch. Their object-metric gaps ran 223 to 2,775 nm and their endpoint
gaps 232 to 12,357 nm; none of that spread exists in the tissue. The honest
restatement of the headline is not "75.7 percent of spanning links are within
the 5 micrometre search radius" but **"about three quarters of spanning links
are in contact, and the remaining quarter are genuinely far."** That is a
sharper result than the one EXP-070 wrote down, and it moves the problem
entirely off the radius.

**2. The answer key does not distinguish what it is used to distinguish.** The
spanning tree is built by minimising object-metric distance, and within the
sub-5-micrometre regime that distance is nearly uninformative: 16 for 16, pairs
separated by 223–2,775 nm in the metric are separated by 32–120 nm in fact.
Where a fragment touches several same-owner fragments, which link the tree
picks is therefore close to arbitrary. EXP-070 already flagged that its metric
change re-routed about 9 percent of full-population links and put an error bar
on EXP-060B's recall; the reads say the underlying ordering is weaker than
either metric's numbers imply. This is an inference from 16 links, not a
measured tree-versus-tree comparison — that would need pairwise true gaps among
same-owner fragments, which is a separate read.

## What the reads also settle, that EXP-070 left open

EXP-070 asked why 43 percent of full-population spanning links remain beyond 5
micrometres at object distance, and offered a weak straight-line corridor
probe. Eight links at 10 micrometres and beyond were re-read here, and in all
eight the partner has **no voxel anywhere in a ±5 micrometre box** around the
near fragment's closest point. The long links are long in the tissue, not in
the metric. That does not say what fills the gap, but it removes "the metric
invented the distance" from the list.

## Panel integrity, found while doing this

Recomputing each panel's box and re-reading three of them found one panel
(`panel_19801307`) whose stored candidate list is a strict subset of what the
same box actually contains: 1,342 objects listed against 2,419 present, with
gaps quantised one voxel coarser. A read-free proxy — objects listed per
mip-5-visible object in the box — flags **16 of the 99 panels** the same way
(ratio near 1.05 against a median of 2.07). All 16 are already-whole panels.
Two other panels, one whole and one cut-centred, reproduced exactly (2,653 and
2,365 candidates, gaps identical on 300 sampled objects each).

**The cause is not established.** What is ruled out: it is not my box, because
all 1,342 stored candidates are inside the box I read and the two panels that
reproduce did so from the same reconstruction; it is not the builder changing
under the panels, because `scripts/build_contact_panels.py` is unmodified since
before the earliest panel was written; and it is not the builder's 20,000-point
seed subsample, because recomputing the gap both subsampled and in full from
the same read agrees to within 40 nm on all eight pairs checked. What remains
unexplained is why that panel's read saw 1,342 objects where mine sees 2,419.
I have not found it, and the next thing I would check is whether those panels
were written by a run whose volume fetch returned a partly empty box.

Dropping the 16 moves every number in this file in the same direction and by
too little to matter: median above truth 1,247 to 1,270 nm (endpoints), 645 to
662 (object), 224 to 240 (mip-5); Spearman 0.722 to 0.729, 0.948 to 0.952,
0.981 to 0.984. The ordering and every conclusion drawn from it are unchanged.
All 16 sit in the already-whole population, which contributes nothing to the
reachability measurement.

## Verdict

EXP-070's pass **survives**. Both gates reproduce, the ordering it validated
holds on an independent pair universe, and its reachability figure lands within
1.2 points of the truth measured from the segmentation. Its recommendation
survives and should be strengthened: object distance beats endpoint distance by
more than it claimed, and the mip-5 supervoxel cloud beats both. What should be
retracted is the reading of its gap numbers as separations in tissue — three
quarters of spanning links do not sit 1.3 micrometres apart, they touch — and
with it any confidence that the spanning tree's choice among touching fragments
means anything.


---

## Provenance

Cached under `data/external/rerun_corrected/` (gitignored):
`mstlinks.json` — the 350 tier-10 object-metric spanning links with their
closest-approach coordinates, rebuilt here and matching EXP-070's recorded
counts exactly; `truegap_mst.json` — the 28 segmentation reads, one record per
link with the box centre, per-object voxel counts and the measured gap;
`viol_check.json` — eight full-fidelity panel re-reads; `panel_integrity.json`
and `centers.json` — the panel box audit.
