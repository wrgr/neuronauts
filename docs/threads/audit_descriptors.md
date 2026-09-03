# Audit: the per-object descriptor threads, and what the erosion actually touched

Scope: `docs/threads/tree_dna.md`, `docs/threads/topology.md`,
`experiments/fingerprints/`, and the tree-descriptor scripts in `scripts/`
(`exp079_tree_features.py`, `exp079_contacts.py`, `exp079_morphology_grammar.py`,
`build_object_geometry.py`, `build_atom_topology.py`, `build_object_clouds.py`).

This is the deep read behind row 9 of
[`rerun_catalog.md`](rerun_catalog.md) — *"`tree_dna`, `topology`,
`fingerprints`: per-object features on complete voxels; caliber went from
useless to decisive under exactly this fix; these are the same kind of
feature."* The question it answers is narrower and more useful than that row:
**which of these assets ever touched the eroded substrate at all.**

## Result

**No descriptor beats the baseline.** Two new ones, measured on the 99 corrected
panels: contact-face area is a clear negative (it makes the baseline worse), and
a signed heading trends positive but does not reach significance (top-1 24 → 28
on the honest subset, better on 27 panels and worse on 15, Wilcoxon p = 0.52).
`along × collin × proximity × caliber` stands at top-1 22 of 66.

**Only one asset in scope was contaminated by the erosion, and it is the one
running right now.** tree-DNA, topology and fingerprints never touched the
eroded clouds — they are built on skeletons and on raw electron-microscopy
voxels. But `scripts/exp079_tree_features.py`, which builds the tree features
for the in-flight EXP-079, takes its **candidate-side** features from
`object_clouds_mip5.npz`. Its `oproj`, `occupy`, `extent` and `dsoma` are
computed from a median of 51 centroid points per object, with 44.5% of
candidates carrying no cloud at all. **If EXP-079 reports the tree grammar as no
better than the geometric baseline, that verdict will have been taken on
rubble.**

**A confound that undercuts the published stop number.** Cut panels carry a
median 2,440 candidates against terminals' 1,726, and the candidate *count*
alone separates the two classes at AUC 0.758 — higher than any stop score
either EXP-076 or this audit has measured. Every max-over-panel stop score,
including EXP-076's 0.642, is inflated by this. It is a fourth construction
confound in a measurement that has already been rebuilt three times.

**So the premise needs splitting.** "Descriptor work was judged on rubble" is
true of the live tree-feature experiment and false of the older threads. Their
dismissals were not substrate artifacts of the erosion, and re-measuring them on
complete voxels will not change them: tree-DNA's problem is that it scores
software-cut skeletons, topology's is that it was never measured at all, and
fingerprints was never dismissed — it is the one descriptor family in scope that
has already beaten a geometry baseline on real errors.

## The erosion, measured rather than assumed

`rerun_catalog.md` states the defect; I re-measured its size on the 99 corrected
panels before using it to judge anything. Over **217,087 candidates across 99
panels**, comparing each object's entry in `object_clouds_mip5.npz` against its
true mip-2 voxel count in the corrected panel:

| | |
|---|---:|
| candidates present in the mip-5 cloud at all | 120,453 of 217,087 (**55.5%**) |
| **true partners** present in the mip-5 cloud | **66 of 66 (100%)** |
| cloud points per real mip-2 voxel, for present objects | median **0.0247** (~1 point per 40 voxels) |
| present objects with < 3 points (`axis_of` returns `None`) | 25,581 (**21.2%** of present) |
| true partners: cloud points per object | median **324** |

So on the eroded substrate roughly **56% of all candidates cannot support a
shape descriptor at all** — they are either absent or too sparse to fit an axis
— while **every single positive can**, with a median of 324 points.

That asymmetry is the thing to hold on to. The erosion did not add uniform noise
to a descriptor. It made the descriptor **well-defined for the positives and
undefined for the majority of the distractors**, so whatever imputation rule
filled the gap was doing more of the ranking than the descriptor was. A feature
scored that way can come out looking useless (its real signal drowned in
imputed values) or spuriously strong (if presence itself leaks). Either way the
number is not about the feature.

### The exposure test

An asset is contaminated if its per-object features come from
`object_clouds_mip5.npz`, or from a fine read labelled through
`objects_v117_mip5_svmap.npz`. It is **not** contaminated if it reads its own
voxels and resolves the identifiers it actually read. That distinction, not the
mip level, is what separates these assets.

## Asset by asset

| asset | what it computed | on what substrate | what it claimed | eroded? |
|---|---|---|---|---|
| `tree_dna` thread + `neuronauts/path_edge_encoder.py` + `attic/prior_results/*_ablation.py` | morphological embedding of local arbor — caliber, branching, tortuosity, tangent flow; Transformer over per-step skeleton path features; a `SkeletonGNN` triplet encoder on PR #17 | **proofread kimimaro skeletons of whole cells, bisected in software** at the balance edge | half-skeleton same-cell AUC **0.829**; path-discrimination accuracy 0.899 | **No** |
| `topology` thread + `neuronauts/topology_model.py`, `topology_dataset.py` | attention arbor validator — is a cluster a merge of two distinct roots | synapse clusters + skeleton path features | **nothing** — no real-data result recorded, no checkpoint tracked | **No** |
| `experiments/fingerprints/` (`cutface/`) | cut-face ultrastructure hash: image patch, bio/art frequency bands, learned encoder, confidence combiner, abstention | **real EM voxels at mip 0/1**, plus a v117 segmentation rebuilt per box by reading the supervoxel layer and calling `cg.get_roots` on **the identifiers actually read** | combiner top-1 **0.767** vs geometry 0.644, 73 real held-out v117 split sites; panel recall 1.000; 92% precision at 51% coverage | **No** |
| `scripts/exp079_tree_features.py` (+ `exp079_contacts.py`) — **in flight** | four tree-level features per candidate: `oproj` (signed projection on the seed's outward heading), `occupy`, `extent`, `dsoma` | productions from 103 real proofread skeletons (correct); **candidate side from `object_clouds_mip5.npz`** | pending — `results/EXP-079/` is empty | **Yes, on the candidate side** |
| `scripts/build_object_geometry.py`, `build_atom_topology.py` | per-atom level-2 node clouds; contracted skeleton topology — endpoints, branches, cable length, mean caliber | CAVE `lvl2_graph` per atom + `l2_attributes`, fetched per object, behind three fatal integrity gates | endpoint/caliber distributions for filter choice | **No** (but level-2 chunks are 10:1 anisotropic, and EXP-070 showed the tip metric was the wrong one) |
| `scripts/build_object_clouds.py` | one point per supervoxel visible at the chosen mip, at its **centroid** | the segmentation volume at mip 5 | — | **it is the source** |

### Reading the table

**Only one asset in scope is contaminated, and it is the one running right
now.** The historical descriptor threads — tree-DNA, topology, fingerprints —
never touched the eroded clouds. They were built on skeletons and on raw EM,
which is a different substrate with different problems. So the premise that
"descriptor work previously judged unhelpful may have been judged on rubble" is
**true of the current tree-feature experiment and false of the older threads.**

**`exp079_tree_features.py` is the live contamination.** Its `panel_rows()`
calls `EC.cloud(a)`, and `exp079_contacts.cloud()` is a reader for
`object_clouds_mip5.npz`. Every candidate-side `oproj`, `occupy`, `extent` and
`dsoma` is therefore computed from a median of 51 centroid points, with 44.5% of
candidates having no cloud at all. `exp079_evaluate.py` already concedes the
shape of this in a comment — *"the mip-5 clouds hold every one of the 66 targets
but only 54% of candidates"* — and imputes the uncovered to the covered median.
My independent count puts it at 55.5%, with a further 21.2% of the covered too
sparse to fit an axis. **If that experiment reports the tree grammar as no
better than the geometric baseline, the verdict will have been taken on
rubble.** The productions are fitted on correct skeletons; it is the candidates
they are scored against that are eroded.

**tree-DNA's problem is not erosion, so re-measuring on complete voxels will not
fix it.** Its ablations bisect an intact proofread skeleton in software. Both
halves of such a cut share caliber and tangent by construction, which is the
regime `rerun_catalog.md` warns about, and the units are proofread whole cells
rather than v117 fragments. The 0.829 also deserves its footnote: random
initialization already scores 0.768, so the *trained* lift is +0.061, and the
same method fails outright at quarter-skeleton granularity. What tree-DNA needs
is real v117 fragments and real seeds — not a better substrate for the same
synthetic task.

**topology was never dismissed; it was never measured.** There is no real-data
result and no tracked checkpoint. Note that EXP-063's frankenmerge detector
answers the same question — is this object a merge of two roots — on real data
at held-out AUC 0.958, and `rerun_catalog.md` lists it as needing no re-run
because it uses no contact distance. EXP-081 observes it has never been applied
to a *proposed* join. That, rather than the attention validator, is the live
path for the atomicity question.

**fingerprints was not judged on rubble, and it is the only descriptor family in
scope that has already beaten a geometry baseline on real errors** (0.767 vs
0.644 on 73 real v117 split sites, via a learned confidence combiner — blunt
fusion and gating both failed). Two caveats keep that from transferring for
free: it was measured on 45–75-candidate panels at real v117 split sites, not on
our 2,440-candidate soma-seeded panels, so its geometry baseline is not our
baseline; and its own residual diagnosis says the remaining misses are
geometrically *distant* partners and degenerate faces, i.e. it wants global tree
context, not a better patch.

### A stale docstring that misdirects exactly this audit

`scripts/build_contact_panels.py`'s module docstring still reads *"Caliber and
axis come from the level-2 cache … Supervoxels in the box map to level-2 nodes
in one batched call."* The code no longer does that — `caliber_span()` computes
caliber from the object's own mip-2 voxels, and its own inner docstring says so.
Since the whole EXP-077 caliber finding turns on **which substrate caliber came
from**, a reader auditing that result from the module docstring would reach the
wrong conclusion. Flagged, not fixed.

## Which dismissals were substrate artifacts

| dismissed descriptor | dismissed as | substrate artifact? |
|---|---|---|
| caliber (EXP-075, median rank 140) | "noise" | **Yes** — pooled over objects missing ~79% of their voxels; now the strongest single term (median 12 → 5, top-1 12 → 22) |
| collinearity (EXP-075, median rank 220) | "worse than distance" | **Yes** — a local axis fitted to a fifth of an object's points is not that object's axis; now cuts median rank 56 → 30 inside the product |
| the two-regime "touching vs gapped" split (EXP-075) | two kinds of tissue | **Yes** — median true gap 120 nm → 32 nm; it described two amounts of our own erosion |
| the stop rule at AUC 0.304 "anti-correlated" (EXP-075) | worse than nothing | **Yes**, plus box placement — corrected to 0.642, weak but real |
| tree-DNA half-skeleton identity | scale-specific, small trained lift | **No** — synthetic software cuts on proofread skeletons; a different and unfixed problem |
| topology / atomicity validator | — | **No** — never measured on real data |
| cut-face fingerprint hash | complementary but not exploitable by blunt fusion | **No** — real EM voxels, complete identity; and a learned combiner *did* exploit it |
| EXP-079 tree features (`oproj`, `occupy`, `extent`, `dsoma`) | pending | **Would be** — measure before believing any negative |

## What I ran: contact-face and signed-heading descriptors

The two descriptors the corrected panels cannot express, both of which the
erosion would have destroyed outright, and both computed from the same box the
panels used so that rows join by object identifier.

**Why these two.** EXP-077 named the residual difficulty exactly: *"many
candidates touch the seed at a single voxel, and distance cannot order a tie."*
`gap_nm` is a minimum over voxel pairs, so a glancing brush and a full severed
cross-section both score 32 nm. What separates them is an **area** — and area is
precisely what a substrate of one centroid per supervoxel cannot represent, so
this descriptor could not have been measured before the identity fix. Second,
the panel's `along` is an **unsigned** `|cos|`, so a candidate lying *behind*
the cut, on the soma side, scores as high as the true continuation lying ahead
of it. Orienting that axis needs the direction the seed's cable was heading when
it ran out — and a local axis fitted to a fifth of an object's voxels is not
that object's axis, which is exactly why collinearity measured useless on the
eroded build.

`scripts/probe_contact_face.py` (new; not committed) re-reads each panel's own
box — mip 2, `agglomerate=True` at the v117 timestamp, identical centre rule —
and emits per candidate:

- `n_touch1`, `n_touch2` — its voxels inside a 1- and 2-voxel dilation of the
  seed (26-connectivity; one iteration reaches 32 nm in x/y, 40 nm in z)
- `touch_rms` — spread of that contact patch about its centroid, nm
- `touch_cc` — how many separate patches it makes against the seed
- `oproj` — signed cosine between the seed's outward heading and the direction
  from the contact to the candidate's local centroid
- `d_ahead` — how far along that heading the candidate sits, nm, signed

### Audit, printed before the result

Per `rerun_catalog.md`'s rule:

- **identity** — objects read with `agglomerate=True` at the v117 timestamp; no
  supervoxel map anywhere in the path
- **geometry** — real 32×32×40 nm voxels; no centroid stands in for a surface
- **placement** — the panels' own centres, reused verbatim, so nothing moved
- **power** — 66 cut panels and 33 terminals is the entire sample; every table
  states its n
- **the confound this is vulnerable to** — the box is centred on the true
  contact, so the partner is *guaranteed* the local coverage a distractor may
  lack. Uncovered candidates are therefore imputed to the covered **median** of
  each multiplier, so having coverage is never itself evidence.

### Two correctness checks before the result

The probe reproduces two numbers it was not fitted to, by a different method
than the one that produced them:

- **56 of 66** true partners are voxel-adjacent to their seed under a 1-voxel
  dilation. EXP-077 reports *"56 of 66 fragments touch their own continuation"*
  from a KD-tree closest approach. Independent route, identical count.
- The stop AUC for `along × collin × proximity` comes out at **0.636** (0.627
  matched) against EXP-076's **0.642**.

I also re-derived each candidate's local length from the panel's own fields,
since `caliber_span` is exactly invertible: `length = n_vox × 40960 / (π ×
cal²)`. The only 1,173 candidates whose recovered length falls below the 32 nm
clamp are precisely the 1,173 with `n_vox < 3` that take the cube-root branch —
an exact self-consistency check on the inversion.

### Result: ranking, 66 cut panels of median 2,440 candidates

| ranker | median rank | top-1 | top-5 | top-20 |
|---|---:|---:|---:|---:|
| `n_touch1` contact area alone | 36.2 | 1 | 5 | 21 |
| contact patch radius alone | 45.5 | 0 | 1 | 14 |
| `oproj` signed heading alone | 31.5 | 5 | 14 | 27 |
| `abs(oproj)` unsigned control | 45.5 | 2 | 8 | 21 |
| **BASELINE** `along × collin × prox × caliber` | **6.0** | **22** | **31** | **44** |
| baseline × log1p(contact area) | 9.5 | 12 | 25 | 39 |
| baseline × contact area | 23.0 | 7 | 16 | 32 |
| baseline × 1/(contact patches) | 6.0 | 22 | 32 | 42 |
| baseline × (1+`oproj`)/2 | 3.5 | 28 | 37 | 42 |
| baseline × max(`oproj`,0) | 3.5 | 26 | 35 | 42 |

**Contact area is a real negative.** The intuition was sound and the measurement
does not support it: a severed process should meet its continuation across a
whole cut face, and the true partner's contact is indeed larger than the field's
(median 204 voxels against a touching-field median of 57). But the panel is full
of large objects — glia and passing dendrites — that share far more surface with
the seed than its own thin continuation does, so the term rewards big neighbours
and multiplying it in costs 10 of the 22 top-1 calls. Do not retry contact area
as a ranking term.

**The signed heading carries real information, but it is nearly all already in
`along`.** The sign is not noise: `oproj` alone ranks at median 31.5 against its
own unsigned control's 45.5, and the true partner sits at median `oproj` +0.66
against a field median of +0.01, ahead of the cut in 46 of 66. But the panel's
existing unsigned `along` alone already ranks at 13.0 on the same subset, and
the gain from adding the sign does not survive a significance test.

### The coverage leak, and the control that removes it

`oproj` is defined only for candidates with voxels within 1.5 µm of the contact
— **6.1% of the panel**, so **93.7% of candidates are imputed**. And the imputed
value is 0.005, effectively zero. That means `baseline × max(oproj,0)` over the
full panel is not really a heading term at all: it is a hard *"must be within
1.5 µm of the box centre"* gate, and the box centre is the true contact. That is
proximity restated with the answer folded in.

The honest test restricts the whole ranking to candidates where the descriptor
is defined, which removes the gate entirely:

| ranker (**covered candidates only**, 65 panels, median 151 candidates) | median rank | top-1 | top-5 | top-20 |
|---|---:|---:|---:|---:|
| **BASELINE** | **3.0** | **24** | **36** | **49** |
| baseline × (1+`oproj`)/2 | 3.0 | **28** | 39 | 46 |
| baseline × max(`oproj`,0) | 3.0 | 26 | 35 | 43 |
| baseline × log1p(contact area) | 6.0 | 13 | 30 | 45 |
| `oproj` alone | 31.0 | 5 | 14 | 27 |
| `abs(oproj)` alone (unsigned control) | 44.0 | 2 | 8 | 21 |
| `along` alone (the panel's unsigned term) | 13.0 | 6 | 18 | 35 |

**Verdict: not significant.** The best variant gains 4 top-1 calls (24 → 28) and
wins on 27 panels against 15, but Wilcoxon signed-rank gives **p = 0.52** and
McNemar on the top-1 flips (6 gained, 2 lost) gives **p = 0.29**. On the full
panel the same term reads better on 26 and worse on 19 at **p = 0.83**. At 66
panels this is a trend, not a result, and it should not be reported as one.

### Stopping — and a confound that undercuts the existing number

EXP-081 makes this the decisive axis: stopping is **98.4%** of a grower's
frontier decisions, not a secondary requirement.

| stop score (max in panel) | AUC all | AUC matched at 5 µm |
|---|---:|---:|
| `along × collin × prox` (EXP-076 reported 0.642) | 0.636 | 0.627 |
| BASELINE (`× caliber`) | 0.531 | 0.538 |
| contact area alone | 0.421 | 0.266 |
| max(`oproj`,0) alone | **0.705** | **0.722** |
| baseline × max(`oproj`,0) | 0.680 | 0.716 |

Two things here, and the second matters more than the first.

**Caliber, the strongest ranking term, actively hurts stopping** — 0.636 down to
0.531. It tells you which candidate, not whether any.

**But the comparison itself is confounded, and not by distality this time.** Cut
panels carry a median **2,440** candidates against terminals' **1,726**
(Mann-Whitney p < 0.001), and 151 covered against 126 (p < 0.001). Any
max-over-panel score rises with the number of draws. Measured directly: the
covered-candidate **count alone** separates the two classes at **AUC 0.758** —
higher than every score in the table above, including mine. A permutation test
that reshuffles `oproj` values across panels while holding panel sizes fixed
does put 0.690 above its null (mean 0.538, p = 0.0055), so the heading is
contributing something beyond size. But a cue that outperforms every score under
test is not a nuisance to note in the limits; it is the leading explanation
until it is matched away.

**So I am not claiming a stop improvement, and EXP-076's 0.642 inherits the same
problem.** This is the fourth construction confound found in this one
measurement, after the soma-versus-neurite caliber confound, the cube-boundary
contamination, and the clipped terminals. The pattern EXP-075 named — *"where
the box sits is doing as much work as what is computed inside it"* — extends to
how many candidates the box happens to contain.

### Where the data went

`data/external/contact_face/*.npz`, 99 files, 1.4 MB, ~12 s per panel to
rebuild via `scripts/probe_contact_face.py`.

### Two concurrent results that converge with this one

Both landed while this audit was running, and both should be read alongside it —
they make the descriptor family's position worse, not better.

- **EXP-082** finds that **30% of human joins connect two v117 objects that do
  not touch near the join.** That independently explains my contact-area
  negative from the other direction: a contact-face descriptor cannot rank what
  never makes contact. My panels are kinder than the real distribution — the
  true partner is voxel-adjacent in 56 of 66 — because they are centred on
  in-box seeded-target contacts rather than drawn from the human edit log.
- **EXP-083** finds whole-cell shape does not police a join: **AUC 0.505** over
  a 22-descriptor `shape_geom` vector, rising only to 0.56 when the wrongly
  grafted cable is a third of the whole neuron. So the shape-descriptor family
  is now measured negative at **both** scales — per-object here, whole-cell
  there.

## Already refuted — do not retry

- **SegCLR embeddings do not select** (EXP-080). As a re-ranker over geometry's
  top 20 they score **0 of 44 top-1 against geometry's 22**, median rank 11.0
  where chance is ~9.5 — below chance, with 44 of 44 true partners carrying an
  embedding, so not a coverage artifact. Multiplying geometry by the embedding
  changes nothing because the term is near-constant across candidates. The only
  variant left untested is the skeleton-local rolling trace July actually used,
  rather than a whole-object mean.
- **A synapse-pattern term (density, spacing, polarity) scores at chance** and
  makes the combination worse — top-1 22 → 16. Polarity remains usable as a
  *veto* rather than a score, and even then only where both sides carry
  synapses: `build_object_polarity.py` records that only 279,075 of 910,888
  objects carry any synapse, so ~69% of candidates have no polarity signal at
  all, and the connective cable between fragments is exactly the population
  synapse-anchored evidence cannot see.
- **Richer cut faces do not help the fingerprint** (`train_depth_bands.py`): a
  3-section depth stack ties the single 16 nm slab (0.757 vs 0.767) and 8 nm is
  worse (0.686). Do not spend effort on finer face representations.
- **Equal-weight fusion and shortlist gating of a hash with geometry** fail
  repeatedly across the fingerprints thread. Only a learned per-site confidence
  combiner converted the complementary signal into a win.

## What I would do next

In priority order, and the first is much more urgent than the rest.

**1. Point EXP-079's candidate features at correct clouds before reading its
verdict.** One line in `scripts/exp079_contacts.py` selects the substrate. The
corrected surface-voxel build already exists —
`scripts/build_object_clouds_voxel.py`, whose own measurements put it at +72 nm
median closest-approach error against exact all-boundary-voxel distance
(Spearman 0.998 on per-seed ordering) where the mip-5 centroid cloud errs
+185 nm at Spearman 0.847. But it has only been run at 40 µm, and I measured its
coverage of the panel population: **88 of 99 panel seeds, but only 8 of 66 true
partners and 14.1% of candidates.** So the fix needs
`build_object_clouds_voxel.py --side-um 100` first. Until then, a negative
result from EXP-079's tree grammar says nothing about the tree grammar.

**2. Match the stop comparison on panel size, then re-measure.** Both classes
need equal candidate counts, or the score needs normalizing for how many draws
the max is taken over (a per-panel rank or a top-*k* mean rather than a max).
Until that is done neither 0.642 nor 0.705 should be quoted. This is cheap — it
is arithmetic on panels that already exist.

**3. The one untried descriptor with a real prior: the cut-face hash on our own
panels.** `experiments/fingerprints/` is the only family in scope that has
already beaten a geometry baseline on real v117 errors (0.767 against 0.644),
its substrate is clean, and it was never tested in the soma-seeded regime. Two
things must carry over from its own hard-won findings, or it will fail the way
it failed there: the win came **only** from a learned per-site confidence
combiner — equal-weight fusion and shortlist gating both lost to geometry alone
— and richer faces do not help, so use the 16 nm single slab and do not spend
time on depth stacks or 8 nm. Its own residual diagnosis also predicts where it
will struggle here: our failures are geometrically distant partners, which is
the case it said needs global context rather than a better patch.

**4. Do not re-measure tree-DNA or the topology validator on corrected voxels
expecting the caliber story to repeat.** Neither was dismissed on eroded
objects. tree-DNA needs a real task — v117 fragments and real seeds — not a
better substrate for software-cut skeletons, and its 0.829 is mostly random
initialization (+0.061 trained lift). For the atomicity question that
`topology.md` owns, EXP-063's frankenmerge detector already answers it on real
data at held-out AUC 0.958; EXP-081 notes it has never been applied to a
*proposed* join, which is the experiment worth running.

**5. Fix the stale docstring in `scripts/build_contact_panels.py`** so the next
person auditing the caliber result is not told it came from the level-2 cache.

