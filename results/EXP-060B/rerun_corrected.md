# EXP-060B rerun — the recall/panel-size trade-off on corrected object geometry

*Run 2026-09-02. Not committed. Every number below is produced by the three
scripts named at the end, which are preserved with their output in
`data/external/rerun_060b_061/`.*

## First, a correction to the brief for this rerun

The rerun was commissioned on the premise that this experiment measured
proximity on `data/substrate/c100um/object_clouds_mip5.npz`. **It did not.** The
module reads `data/substrate/topology/k10.npz` and `kall.npz`, whose `ep_pos_nm`
comes from `scripts/build_atom_topology.py`: one point per **level-2 node**,
taken from the level-2 cache, contracted to a skeleton, and then **only the leaf
tips are kept**.

The premise is wrong about the file and right about the defect, in two ways that
compound:

1. a level-2 chunk is 2,048 x 2,048 x 20,480 nanometres, **10:1 longer in z**,
   so a distance between level-2 node positions carries a direction-dependent
   scale;
2. only **leaf tips** are kept, so an object's "cloud" is a handful of points at
   the ends of its branches rather than the object. In this 40 um cube a
   labelled object contributes a median of **183 endpoints** against **949**
   points in the corrected cloud, and the endpoints are all at branch ends.

So the rerun is still the right thing to do; only the name of the defective
input changes.

---

## AUDIT

| | this rerun | the original EXP-060B |
|---|---|---|
| **substrate** | `data/substrate/c40um_mip2/object_clouds_mip2.npz` — 40 um cube at (663, 591, 860) um, read at **mip 2 = 32 x 32 x 40 nm**, one point per supervoxel (its voxel centroid). 192,474 objects, 22,494,502 points; median **949 points per labelled object**; typical spacing about 130 nm | `topology/k10.npz` (20,826 atoms) and `kall.npz` (279,075 atoms) over the 100 um cube — level-2 leaf tips only, chunk anisotropy 10:1 |
| **identity resolution** | v117 at timestamp 1623399000, supervoxel to root resolved on **100%** of supervoxels at mip 2. Checked against a direct `CloudVolume(agglomerate=True, timestamp=V117_TS, mip=2)` read of a 3 um box: **99.56%** of sampled cloud points carry the same object id once the file's known uniform half-voxel offset (16/16/20 nm, documented in `scripts/build_object_clouds.py`) is applied; 97.48% without it. The offset is identical for every point, so cloud-to-cloud distances are unaffected | v117 level-2 leaves per atom; no voxel-level check |
| **sample size** | **708** labelled atoms in the cube (pure, owner tier above none), **127** owners with 2 or more fragments, **332** fragments, **205 minimum-spanning-tree links**. Panels are built for the 332 link participants, which is exactly the set that can contribute a spanning pair | 1,297 labelled / 350 links at tier >= 10; 4,511 / 3,260 on the full population |
| **main confound** | **the 40 um cube**, one eighteenth the volume of the original's. Fragments of the same cell outside it are absent, so both the spanning set and the candidate crowd are smaller. To separate that from the geometry change, the original's own level-2 substrate is re-measured **in the same 40 um scope**, and that column — not the published one — is the comparison | the 100 um cube |
| **second confound** | points are supervoxel centroids, not surface voxels, so every gap is biased upward by roughly the local caliber. Uniform across the comparison | leaf tips only |
| **method check** | the fast panel (coarse-grid discovery, then exact refinement on the full cloud) was verified against a brute-force query over all 22.5M points on 8 seed/radius combinations: **identical candidate sets, maximum gap error 0.000 nm** | — |

**A process-error disclosure.** A first attempt at this run was left alive by a
`pkill` pattern that did not match its own target (the interpreter is
`.../Python`, capital P), so two processes wrote the same output file and
interleaved their results. Both were killed by process id, the output deleted,
and the run repeated from scratch with a single process. Nothing below comes
from that contaminated file.

---

## Result 0 — the spanning set itself was partly wrong

The minimum spanning tree is built per owner from nearest-point gaps, so it
depends on the geometry. Rebuilding it on correct geometry changes it:

| | corrected 32 nm clouds | level-2 leaf tips |
|---|---:|---:|
| spanning links found | **205** | 192 |
| links the two agree on | **176** | 176 |

Fourteen percent of the links the original was scoring itself against were not
the links that actually span the owner. And the gaps are very different:

| spanning-link gap | p10 | p25 | **p50** | p75 | p90 |
|---|---:|---:|---:|---:|---:|
| **corrected 32 nm clouds** | **107 nm** | **224 nm** | 3,516 nm | 15,259 nm | 24,172 nm |
| level-2 leaf tips | 743 nm | 1,563 nm | 4,003 nm | 12,755 nm | 18,680 nm |

On the 176 links both substrates find, the level-2 measurement **overstates the
gap on 73.9%** of them, by a median factor of 1.19 and a p90 factor of 13.1. The
near field is where it goes wrong: a quarter of true spanning links are within
**224 nm** on real voxels — objects effectively touching — where the level-2
skeleton reports 1,563 nm. **44.9%** of spanning links are under 2 um on correct
geometry against **36.5%** on level-2 tips.

---

## Result 1 — the corrected curve, in object units (the original's metric)

Candidate universe restricted to population objects, matching the original's:

| panel size (objects) | recall at 2 um | recall at 5 um |
|---:|---:|---:|
| 20 | **16.6%** | **16.6%** |
| 50 | 27.3% | 27.3% |
| 100 | 36.1% | 36.1% |
| 300 | 42.9% | 45.9% |
| **uncapped** | 44.9% (median panel 565) | **53.2%** (median panel **1,490**) |

The 2 um and 5 um columns are identical up to a cap of 100, and the reason is
measured, not assumed: **100% of seeds already have 50 or more population
objects within 2 um, and 98.8% have 100 or more.** Radius stops mattering long
before panel size does.

With **every** object in the cube as a candidate, including the synapse-free
connective cable EXP-071 found:

| panel size | recall at 2 um | recall at 5 um |
|---:|---:|---:|
| 20 | 11.2% | 11.2% |
| 50 | 24.4% | 24.4% |
| 100 | 32.2% | 32.2% |
| 300 | 41.0% | 41.0% |
| uncapped | 44.9% (median 1,311) | 53.2% (median **5,696**) |

The reachable ceiling is unchanged (44.9% / 53.2%) because the spanning partners
are population objects either way; what the extra objects do is **crowd the
panel**, quadrupling it at 5 um and costing 5 points of recall at a cap of 20.

## Result 2 — the same measurement on the original substrate, same scope

This is the comparison that isolates geometry. Same 40 um cube, same owners,
same 205 spanning links; only the point clouds differ.

| panel cap | **corrected 32 nm clouds** | level-2 leaf tips | *published original, 100 um, tier >= 10* |
|---:|---:|---:|---:|
| 20 | **16.6%** | 13.7% | *12.0%* |
| 50 | **27.3%** | 20.0% | *15.7%* |
| 100 | **36.1%** | 24.9% | *22.9%* |
| 300 | **45.9%** | 32.2% | *38.6%* |
| uncapped, 5 um | **53.2%** | 36.6% | *64.6%* |
| uncapped, 2 um | **44.9%** | 22.4% | *42.9%* |

(the level-2 column is scored against the corrected spanning set, so the two are
answering the same question; scored against its own spanning set it reads 14.6 /
21.9 / 27.6 / 34.9 / 39.6 / 24.5 — the same shape.)

**Correct geometry recovers 21% to 45% more spanning links at every capped
panel size**, and the gain is largest exactly in the near field the level-2 tips
were mis-measuring: at 2 um, uncapped reachability doubles, 22.4% to 44.9%.

Where the partner is reachable at all, its rank in the distance-ordered panel:

| substrate, radius | share of links reachable | median rank | p90 rank |
|---|---:|---:|---:|
| corrected, 2 um | **44.9%** | 33 | 151 |
| corrected, 5 um | **53.2%** | 43 | 764 |
| level-2 tips, 2 um | 22.4% | 15.5 | 53.5 |
| level-2 tips, 5 um | 36.6% | 41 | 377 |

The level-2 substrate ranks *better* when it reaches at all — its panels are
smaller because its clouds are sparse — but it reaches only half as often. That
is the shape of a substrate that is missing candidates, not one that is ranking
them well.

## Result 3 — adding EXP-061's direction term moves the curve at a deployable panel

EXP-060B and EXP-061 were always the same trade seen from two sides, so here
they are combined on one substrate: the corrected 2 um proximity panel,
re-ranked by gap **inside a cone** on the seed's local axis (the `along` term of
`scripts/build_contact_panels.py`, computed from the same mip 2 clouds; a local
axis was fittable for 99.8% of 305,465 candidates).

| ranking | cap 20 | cap 50 | cap 100 | cap 300 |
|---|---:|---:|---:|---:|
| gap only | 16.6% | 27.3% | 36.1% | 42.9% |
| gap inside a 15 deg cone | 10.7% (panel 6) | 10.7% | 10.7% | 10.7% |
| gap inside a 30 deg cone | 19.5% | 20.0% | 20.0% | 20.5% |
| **gap inside a 45 deg cone** | **25.4%** | 29.3% | 30.2% | 31.2% |
| gap inside a 60 deg cone | 24.4% | 32.2% | 34.6% | 37.1% |

**At the panel size a scorer would actually deploy, direction raises recall by
half** — 16.6% to 25.4% at 20 candidates. Put together with Result 2, the
deployable number goes 13.7% (original substrate, same scope) to 16.6%
(corrected geometry) to **25.4%** (corrected geometry plus direction), nearly a
doubling.

The cone also imposes a ceiling: inside 45 degrees the most that is reachable at
any cap is 31.2%, against 44.9% for the ball, because a cone that tight loses
partners. Direction buys recall at small panels and costs it at large ones.

---

## Does the original conclusion survive?

**Yes — the headline survives intact, and one of its numbers was pessimistic.**

The original's bottom line was: *"geometry (ball or cone, either tier) gives
partial credit — real but insufficient for a deployable proposer on its own"*,
with 12-23% recall at a panel a scorer could use and 47-65% only at a panel of
thousands. On corrected 32-nanometre object geometry the same measurement gives
**16.6% at a panel of 20** and **53.2% at a median panel of 1,490** — the same
curve, shifted up. Adding direction reaches **25.4% at a panel of 20**. Nothing
here comes close to the recall an assembler needs, and the trade-off between
recall and panel size is still the finding.

**What was wrong, and mattered.**

- **The spanning set.** 14% of the links the original scored against were not
  the true spanning links, because the level-2 skeleton mis-ranked which
  fragments were nearest.
- **The near field.** A quarter of true spanning links are within 224 nm on real
  voxels; the level-2 tips report 1,563 nm. Uncapped reachability at 2 um
  doubles once that is fixed.
- **The size of the panel penalty.** Correct geometry gives 21-45% more recall
  at every fixed panel size than the substrate the conclusion was drawn on. The
  conclusion held anyway, but it was drawn from numbers that were too low.

**What was right for the wrong reason.** The published uncapped figure of 64.6%
at tier >= 10 is *higher* than the corrected 53.2%. That is scope, not accuracy:
the tier >= 10 slice is 1,297 well-sampled large atoms in a 100 um cube whose
partners are mostly other large atoms. In the same 40 um scope the original
substrate reaches 36.6%, not 64.6%.

**Two of the original's other conclusions are untouched by this rerun** and
should not be read as re-tested here: the synapse-floor result (raising a
synapse-count floor lowers recall faster than it shrinks the panel) and the
finding that small 1-9 synapse fragments are genuinely distal. Both were
measured on the 100 um population and neither depends on the fine geometry in a
way this cube could settle.

## Limits of this rerun

- **A 40 um cube**, 1/18 the original's volume: 205 spanning links against
  3,260. Enough for the curve's shape and for a paired comparison against the
  same links on the other substrate; not enough for the tail, and not a
  substitute for re-running the full 100 um cube at mip 2.
- **Points are supervoxel centroids, not surface voxels.** Every gap is biased
  upward by roughly the local caliber, so the true near-field is even tighter
  than Result 0 shows.
- **The tier comparison is not reproduced.** This cube has 708 labelled atoms
  total; splitting them by synapse tier would leave too few links per cell.
- Panels are built only for the 332 spanning-link participants. That is exact
  for recall (no other seed can contribute a spanning pair) but means the panel
  size medians describe those 332, not all 708 labelled atoms.

## Reproduce

```bash
cd data/external/rerun_060b_061
python prep.py               # scope, both spanning-link sets, the gap comparison
python rerun060b.py          # the cap curves on both substrates, same scope
python rerun060b_cone.py     # the proximity-plus-direction curve
```
The scripts, their JSON output and `prep.npz` are preserved in
`data/external/rerun_060b_061/` (gitignored). Inputs read:
`data/substrate/c40um_mip2/`, `data/substrate/topology/kall.npz`,
`data/substrate/c100um/labels_v1822.npz`.
