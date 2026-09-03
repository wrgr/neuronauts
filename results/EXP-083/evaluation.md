# EXP-083 — whole-cell shape does not police a join, and the reason is measurable

## Result

**No.** A whole-cell shape score does not detect a wrong join by threshold.
Held out by cell, the area under the receiver operating characteristic curve
(AUC) for telling a correctly assembled arbor from the same arbor with one
wrong branch grafted in is **0.505** on 2,062 size-matched pairs, and it stays
at **0.56** when the wrong cable is a *third of the whole neuron*. The reason
is a number, not an opinion: a wrong join of an eighth of the arbor moves the
22-descriptor `shape_geom` vector by **about half of one between-cell standard
deviation**. Real neurons differ from each other far more than a chimera
differs from its host, so no absolute threshold exists to find.

The same descriptors *do* work in the weaker, comparative setting — given the
true piece and a wrong piece **offered at the same site on the same base**,
they pick the true one **0.64** of the time label-free (0.68 on real
segmentation breaks, 0.75 with caliber and compartment labels added). The wrong
piece arrives on the identical stem edge — same gap, same direction, same
parent, same amount of cable — so every local pairwise cue is neutralized by
construction, and that 0.64 is information local geometry cannot supply. It is
also all it is: one wrong candidate rejected about two times in three. And a
control shows it is not recognizing foreign
cable at all: moving **cell A's own** branch to the wrong site is detected
*better* (0.71) than another cell's branch (0.64). What the score reads is
"this piece does not belong *here*", which is a local placement question wearing
a global costume.

## The construction

103 proofread arbors (`data/external/cell_skeletons/*_skv4.npz`, one connected
component each, median 10.6 mm of cable). For each test:

| | |
|---|---|
| base | the arbor with two subtrees removed: `z` (3–15% of the cable, so the correct assembly is itself partial, as a grower's is) and `u`, the branch under test |
| **correct** | base + cell A's own subtree(u), back where it was |
| **corrupted** | base + a subtree of a *different* proofread cell, cable matched to within 15% (median mismatch 7.5%), rigidly translated so its root lands on A's vertex `u` |

The stem edge is A's own in both — same length, same direction, same parent —
so the join looks identical locally and the two assemblies carry the same amount
of cable in the same place. Any separation has to come from the shape of the
whole tree. Construction verified against the raw skeletons on 8 cells: cable
totals, vertex counts, rooting, and the identity of the join edge all check out.
(The first version of that check failed; the error was in my expected-cable
formula, which forgot that the stem edge is A's in both arms, not the code.)

**Arm 2 uses real segmentation breaks.** The 232 `split_challenges` in
`data/external/cell_cards/*.json` record where the v117 segmentation actually
cut each cell, with both sides' coordinates; 219 snap to a skeleton vertex
(median 606 nm) and 214 yield a usable pair. There the wound and both candidate
pieces are places the segmentation really broke, not arbitrary cuts.

32 descriptors: cable, tips, branch points, Strahler order, branch order,
partition asymmetry, geodesic and radial extent, radial cable percentiles,
tortuosity, principal-axis eigenvalue ratios, Sholl peak and its radius,
fraction of cable running back *toward* the soma, self-overlap, caliber mean /
variation / taper slope / distal-to-proximal ratio, and compartment composition,
switches and placement. `shape_geom` is the strict set: topology and geometry
only, no caliber, no compartment label.

Scoring is a conditional (paired) logistic fit — one weight vector, no
intercept, so any per-cell offset cancels — trained on the within-pair
difference and applied as an ordinary score `w·x` to a single assembly. Five
folds, cell-disjoint. A label-shuffle null over the same pipeline lands at
0.456–0.580 across bins and seeds.

## Absolute detection: chance, at every size tested

| what was grafted | pairs | median wrong cable | descriptor shift | **absolute AUC** |
|---|---:|---:|---:|---:|
| arm 1, random cuts | 2,062 | 48 µm (0.4% of arbor) | 0.01 σ | **0.505** |
| arm 2, real breaks | 214 | 105 µm (1.0%) | — | **0.507** |
| frankenmerge scale | 1,057 | 1,534 µm (12.8%) | 0.46 σ | **0.547** |
| … 20–35% of arbor | 259 | — | — | 0.561 |
| … 35–50% of arbor | 101 | — | — | 0.562 |

"Descriptor shift" is the median distance between a pair's two `shape_geom`
vectors (22 descriptors), in units of the between-cell standard deviation of
the same descriptors. It grows as expected — 0.000, 0.001, 0.004, 0.017, 0.043, 0.188,
0.496 σ across the size bins below — and never reaches the ~2 σ a threshold
would need.

Adding caliber, compartment labels, or everything moves absolute AUC to 0.51–0.56.
Size alone is 0.50, so the matching worked.

## Within-site comparison: real, and it peaks at tens of microns

`shape_geom`, fit inside each size bin, cell-disjoint. "Paired" is the fraction
of sites where the assembly built from the true piece scores better than the one
built from the wrong piece.

| wrong cable added | pairs | median | share of arbor | paired [95% CI over cells] |
|---|---:|---:|---:|---|
| 1–3 µm | 309 | 1.9 µm | 0.02% | 0.492 [0.450, 0.534] |
| 3–10 µm | 309 | 6.2 µm | 0.05% | **0.699** [0.644, 0.754] |
| 10–30 µm | 309 | 19.6 µm | 0.18% | **0.738** [0.686, 0.786] |
| 30–100 µm | 309 | 58.0 µm | 0.52% | **0.748** [0.696, 0.796] |
| 100–300 µm | 308 | 166 µm | 1.4% | 0.721 [0.666, 0.775] |
| 300–1000 µm | 292 | 471 µm | 4.2% | 0.620 [0.564, 0.672] |
| >1000 µm | 226 | 1,514 µm | 13.5% | 0.566 [0.484, 0.648] |

**Minimum detectable wrong join: between 3 and 10 µm of cable.** Below 3 µm it
is at chance, and for 31% of those pairs the whole-cell descriptor vector is
*bit-identical* between the correct and the corrupted assembly — the wrong join
is not merely hard to see, it leaves no trace in the description at all.

**It gets worse, not better, above ~300 µm.** That is not a modeling artifact:
the single strongest label-free descriptor, the fraction of cable running back
toward the soma, tracks the same arc with no model at all — 0.534, 0.702,
0.741, 0.731, 0.731, 0.562, 0.442 across the seven bins. A short stretch of
cable either heads away from the soma or it does not, and that is a sharp test;
a 1.5 mm branch is a spatially spread object whose own orientation statistics
resemble any other cell's 1.5 mm branch. The discrimination lives at the scale
of a directional stretch of cable and is lost exactly where the damage is
largest.

## Feature sets, all pairs

| set | arm 1 absolute AUC | arm 1 paired | arm 2 (real breaks) paired |
|---|---:|---:|---:|
| size only | 0.500 | 0.532 [0.511, 0.554] | 0.493 [0.427, 0.557] |
| shape_geom (no caliber, no labels) | 0.505 | 0.642 [0.618, 0.664] | **0.680** [0.616, 0.750] |
| + caliber | 0.507 | 0.699 [0.679, 0.720] | 0.664 [0.604, 0.726] |
| + compartment labels | 0.509 | 0.740 [0.718, 0.760] | **0.724** [0.661, 0.793] |
| all 32 | 0.511 | 0.747 [0.726, 0.768] | 0.710 [0.645, 0.778] |

Single descriptors, model-free (arm 1 / arm 2): compartment switches per mm
0.706 / 0.650, cable running inward 0.643 / 0.680, caliber taper slope 0.624 /
0.556, tortuosity 0.571 / 0.605, cable beyond 100 µm 0.436 / 0.334 (inverted),
median radial percentile 0.449 / 0.364 (inverted). The compartment-switch
feature is the grammar's axon-dendrite chimera veto, and it is the single
strongest thing here — but it reads the label the donor skeleton already
carries, and it fires at the join, so it is not whole-cell shape and it is kept
out of `shape_geom`.

## The control that reframes the whole thing

Replace the foreign donor with **another branch of cell A itself**, cable
matched, moved to the same site (1,995 pairs):

| | absolute AUC | paired |
|---|---:|---:|
| wrong piece from another cell | 0.505 | 0.642 [0.618, 0.664] |
| wrong piece is A's own cable, displaced | 0.516 | **0.710** [0.691, 0.729] |

Displacing a cell's own cable is *more* detectable than importing another
cell's. So the score is not recognizing that two neurons have been merged. It
is asking whether the added cable sits plausibly with respect to *this soma* —
a placement question. That is still information a purely pairwise contact score
does not have (it is referenced to the soma, not to the parent tip), but it is
not the global grammar check the experiment set out to test, and it should not
be described as one.

## What this does and does not license

- **It refutes the extension of EXP-063 to a proposed join.** EXP-063 reaches
  held-out AUC 0.958 flagging a frankenmerge in an existing object, on a
  different substrate (level-2 and synapse clouds of v117 atoms, size-controlled
  only by a tier floor). Here, with size matched *within the pair* and the
  substrate a skeleton, the same question scores 0.56 with a third of the neuron
  foreign. I have not shown EXP-063 is wrong; I have shown its signal does not
  survive exact size matching on this substrate, which is a reason to check what
  carries it there.
- **A grower gets no absolute check from whole-cell shape.** The frontier
  problem of EXP-081 — 2,137 sites, 34 live, 1.6% — needs an accept/reject
  decision at a site with no counterfactual. This gives a comparative score
  only.
- **The comparative score is complementary, and modest.** Because gap,
  direction, stem caliber and added cable are all equalized inside the pair,
  0.64–0.68 is what the whole-tree view adds *after local geometry has been
  zeroed out*. It is not comparable to EXP-081's 0.630, which is a different
  decision (whether a tip continues at all), nor to EXP-076's median rank 5 of
  2,440, which is an unmatched candidate field where most candidates fall to
  distance alone. Whether the two combine is untested here and is the one cheap
  thing worth doing next: the panels of EXP-076/EXP-079 already carry the
  candidates, and the soma-referenced descriptors above can be added to them.

## Limits

- The corrupted piece is translated to the join site with no rotation, so it
  keeps its true orientation in the volume. A real candidate at a frontier site
  is already within a few microns of the cut end, so the translation is small in
  effect, but I have not measured the difference.
- The paired setting holds the base **exactly** fixed. That is the honest model
  of ranking candidates at one site, and it is why a 0.001 σ shift is readable
  at 6 µm; it is not a model of comparing two differently-grown assemblies,
  where such a shift would be swamped.
- The "correct" assembly is a proofread arbor with one branch pruned. A real
  grower's assembly carries its own accumulated errors, which this does not
  simulate.
- Compartment labels come from the donor cell's own skeleton labeling, so
  `shape+compart` is optimistic about how cleanly axon and dendrite can be told
  apart at inference.
- Arm 2 has 214 pairs on 67 cells; its confidence intervals are wide and its
  bins are thin.
- One split, one model family. Feature sets were compared on the same held-out
  folds, so small differences between them should not be read as a ranking.

## Reproduce

```
python scripts/exp083_run.py            # arm 1, 2,062 size-matched pairs
python scripts/exp083_arm2.py           # arm 2, real segmentation breaks
python scripts/exp083_run_samecell.py   # same-cell displacement control
python scripts/exp083_run_big.py        # frankenmerge-scale grafts
python scripts/exp083_score2.py         # absolute AUC and within-site paired
python scripts/exp083_diag.py           # descriptor shift, size-stratified fit
python scripts/exp083_null.py           # label-shuffle null
python scripts/exp083_controls.py       # the two control arms
```
Outputs: `result.json`, `diagnostics.json`, `controls.json`, `features*.npz`.
