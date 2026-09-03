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

---

# The older work, and why it can now be run for real

The list above stops at the registered experiments, which is where I first drew
the line and it was the wrong line. Most of this repository's work predates
them — the grammar and PCFG threads, tree-DNA, fingerprints, topology, cell
assignment, treestitch — 41 scripts in `attic/benchmarks_semi_synthetic/` and 28
engines in `attic/morpho_grammar/`, plus 31 files in `experiments/pcfg/`.

Those were all scored the same way: **take an intact skeleton, cut it in
software, and see whether the method puts it back together.** Both halves of
such a split still carry matching geometry, matching caliber and a matching
tangent, so the task is far easier than the real one. 25 of the 26 engines also
contain no checkpoint-loading code, so they were scored at initialization.

That is the standing reason to distrust them. But it is not the only thing that
was missing, and the missing piece is now present.

## What was actually missing: a real task with real seeds

A synthetic benchmark has to invent both the fragments *and* what counts as
success. Every one of those scripts invented both. The real task now has neither
invented:

- **The seeds are real and exhaustive.** They come from `nucleus_detection_v0`,
  a detection table independent of the segmentation — so segmentation errors
  cannot corrupt the seed set. All 332 nuclei in the 100 µm cube resolve to a
  v117 fragment; 76 (23%) are non-neuronal and are simply ignored, which is a
  feature rather than a loss; 103 are evaluable neuron seeds. We know them all,
  and the same table covers the entire volume.
- **The target is real.** `box_truth.seeded_target` is the seed's own in-box
  connected component under v1822 proofreading — what a grower starting at that
  cell body should recover. 36 of 103 need nothing, which is what makes
  abstention a real requirement rather than a stylistic one.
- **The substrate is now correct.** Objects read with `agglomerate=True`, real
  voxels at 32 nm, no supervoxel map in the path.

So the honest statement about the attic is not "these were synthetic, discard
them". It is: **these methods have never been asked the real question.** A
grammar over morphology scored on soma-seeded recovery, with real seeds, a real
target and correct geometry, is an experiment nobody in this repository has run.

## Extended priority

Everything above the line first — the proximity negatives are load-bearing and
cheap to re-measure. Then:

| | thread | what it needs to run for real | note |
|---|---|---|---|
| 7 | `experiments/pcfg/` (31 files) | soma-seeded panels instead of synthetic partitions | the PCFG scores trees rather than pairs, which is exactly what local geometry could not do — abstention and tie-breaking both want tree context |
| 8 | `docs/threads/grammar.md` engines | real fragments, real seeds, and training (25 of 26 were never trained) | its 85–87% is in-sample on synthetic damage; the honest number is unknown, not low |
| 9 | `tree_dna`, `topology`, `fingerprints` | per-object features on complete voxels | caliber went from useless to decisive under exactly this fix; these are the same kind of feature |
| 10 | `cell_assignment`, `treestitch` | real seeds; these assign fragments to cells, which is the soma-seeded task under another name | closest in spirit to the current framing |

## The rule for all of it

Re-running an old experiment on new data is only worth doing if the result is
believable afterward. Today five separate confident answers dissolved on
inspection, every one of them substrate rather than hypothesis. So each re-run
prints its audit before its result, states its effective sample size, and names
the confound it is vulnerable to. An old method that now scores well on correct
data is a real finding; an old method that scores well because the new substrate
leaks is the same mistake in a better disguise.
