# Compartment-augmented PCFG — findings

Goal: extend the PCFG global grammar with (a) real Google **SegCLR** local
embeddings and (b) compartment production rules (axon↔dendrite crossing,
multi-soma) to detect false merges and propose splits. See the approved plan for
the full design.

---
## ★ HEADLINE FINDING — SegCLR encodes cell *type/compartment*, not cell *identity*

The single fact that explains every result below.  Proven directly:

- Spectral-2 clustering of one clean neuron's SegCLR nodes aligns with **axon/
  dendrite compartment** at ARI **0.69–0.86**.
- Pool two different cells and cluster: it splits by **compartment (ARI 0.68–0.84)**
  and by **cell identity ARI ≈ 0.00–0.03** (10/10 cell pairs).  It literally cannot
  tell the two cells apart; it splits axon-from-dendrite *across* both.

Why: SegCLR is trained contrastively with a **~4 µm receptive field and local
positives**, so it learns *local morphology* — which is dominated by compartment/
type, not by which individual cell you're on.

**Consequences (each validated below):**
- **Multi-soma merges → solved** (that's *topology*, not SegCLR).
- **Splits → mostly geometry; SegCLR helps only as a type/compartment
  disambiguator** (it can pick the right continuation when the wrong candidate is a
  *different type/compartment*; not when it's the same type).
- **Same-type / same-compartment merges → not solvable by SegCLR** (no identity
  signal to separate two same-type cables). Every SegCLR merge attempt (absolute
  step, comparative walk, branch proposer, global spectral) fails for this one
  reason.

---

## Exp 0 — SegCLR-only value probe (the "is this worth building?" test)

**Question (user's framing):** retrieve the MICrONS SegCLR embeddings and, *using
just SegCLR*, see the top-K for split errors — how valuable is SegCLR alone at
localizing the seam of a false merge?

**What was built.** `neuronauts/segclr.py` — a dependency-free loader for the
public SegCLR release (no `connectomics` package, no auth). It reads the CSV-zip
shards over HTTPS byte-range requests (only the ZIP central directory + one
segment's CSV per fetch, not the ~220 MB shard), reproduces the sharding
(`md5_shard`, 10 000 shards, **bytewidth = 64 bytes** — verified against real
shard membership), parses 64-dim embeddings with nm coordinates, caches per
segment, and assigns embeddings to points/vertices spatially.

**Verified facts (not assumed):**
- Sharding scheme confirmed by checking that every member of shards 0/1/2319/777
  hashes back to its shard with `bytewidth=64` (the `..._BYTEWIDTH64` scheme).
  `bytewidth=8` — the naive reading — is wrong and was ruled out empirically.
- CSV row layout: `node_id, x, y, z, e0..e63` (68 cols); coords in nm.
- **"m343" is a 2022 segmentation snapshot, NOT a queryable public
  materialization** (public exposes v1300+). So SegCLR segment ids do not map
  trivially to current CAVE roots — the version footgun is real. Exp 0 therefore
  runs fully inside m343 (SegCLR point clouds only); the m343→current bridge
  (via `seg_m343` volume or chunkedgraph) is the next milestone.

**Setup.** Take real m343 neurons (large embedding point clouds), union pairs
whose arbors actually pass near each other, build a spatial kNN graph, and label
each edge as a *seam* edge (cross-cell contact ≤1.5 µm) or *within* edge. Score
every edge purely by SegCLR discontinuity `1 − cos(emb_i, emb_j)`. Command:

```
python -m experiments.pcfg.run_compartment_grammar --exp0 \
    --shards 0 1 2 3 4 5 --n-neurons 24 --subsample 30000
```

**Result (7 real-contact pairs):**

| metric | value | reading |
|---|---|---|
| AUC (raw per-point) | **0.92 mean / 0.95 median** | SegCLR discontinuity is a *strong ranking signal* for seams |
| best-seam-edge percentile | **~2.8% median** | the first true seam edge sits ~top 3% of all edges |
| edge hit@100 / hit@500 | 0.00 / 0.29 | the exact seam edge is rarely in the absolute top-100 |
| **site hit@1..10** | **0.00** | the seam *neighbourhood* is not among the top ~40 high-discontinuity sites |
| euclid-pooled AUC | 0.22 | euclidean pooling **crosses** the seam and destroys signal — pool geodesically |

**Verdict.** SegCLR alone is a **strong global ranking signal** (AUC ≈ 0.95) but
**not a sufficient stand-alone top-K localizer**: within-cell embedding variation
(compartment transitions, caliber changes, myelination) produces the highest
discontinuities, so the true seam is buried ~3% deep and its site is never in the
top candidate sites. This is exactly the case for the **compartment-augmented
grammar** — the structural rules (A↔D-not-via-soma, multi-soma) and geodesic
(non-seam-crossing) pooling are needed to suppress the within-cell false
positives SegCLR cannot. SegCLR should enter as a strong *corroborating* term,
not the sole detector.

**Caveats (per CLAUDE.md — not over-claimed):** synthetic merges (real neurons,
real contacts, but geometric ground truth rather than a proofread seam); small n
(7 contacting pairs); 30k subsample lowers contact density; no real
proofread-merge validation yet (blocked on the m343→current bridge). The
numbers are indicative of value, not conclusive.

## Exp 1 — SegCLR-only on REAL proofread merges (the number that matters)

Exp 0 used *synthetic* merges. Exp 1 replaces them with **real** false merges via
the m343→current bridge: an m343 SegCLR root whose supervoxels now belong to ≥2
substantial current cells (`chunkedgraph.get_latest_roots`) was a real
false merge that proofreaders split. Label each SegCLR point by its nearest
current-descendant skeleton (= the ground-truth cell it was assigned to), then
run the same seam-localization scoring. Command:

```
python -m experiments.pcfg.run_compartment_grammar --exp1 \
    --shards 0 1 2 3 4 5 6 7 --n-neurons 20 --min-share 0.15
```

Verified along the way: m343 SegCLR coords and current (v1822) skeleton coords
share the same nm frame (point→skeleton median ≈ 970 nm, coverage ≈ 0.92); the
CAVE skeleton service needs `cloud-volume` (without it every skeleton fetch
silently negative-caches to empty — a real footgun).

**Result (7 real merges, from a scan of ~180 large-m343 candidates):**

| metric | **synthetic (Exp 0, n=7)** | **REAL (Exp 1, n=7)** |
|---|---|---|
| AUC (SegCLR discontinuity) | **0.95** | **0.49 mean / 0.56 median** (≈ chance) |
| AUC spread | 0.78–1.00 | 0.27–0.63 (0/7 > 0.7; **3/7 < 0.5**) |
| best-seam-edge percentile | ~2.8% | 1.35% median / 10.0% mean |
| site hit@3 | 0.00 | 0.14 (1/7) |

Per-merge real AUCs: 0.59, 0.56, 0.42, 0.27, 0.34, 0.59, 0.63 (2–3 cells each).

**The synthetic test massively overestimated SegCLR's value.** On real false
merges, SegCLR discontinuity across the true seam is **near chance (AUC ≈ 0.52)**.
This makes biological sense: synthetic merges join two *unrelated* neurons
(different type/location → different embeddings, trivially separable), whereas a
*real* false merge occurs exactly where two cells are locally similar and
touching — that is *why* the segmentation merged them — so their SegCLR
embeddings are barely distinguishable at the join.

**Revised verdict.** SegCLR alone is **not** a reliable stand-alone split-error
detector on realistic merges. Its value as the "fuser" term is real but
secondary; the compartment/structural grammar (A↔D-not-via-soma, multi-soma,
topology) must carry the detection, with SegCLR as a weak corroborator. This
strengthens — not weakens — the case for the compartment-augmented PCFG over a
pure-embedding detector.

**Caveats (per CLAUDE.md):** n = 7 real merges — only ~8% of large-m343
candidates are genuine ≥2-substantial-cell merges (most just shed small
fragments), so real 2-cell merges are the minority and n is modest, but the
pattern is consistent (0/7 above 0.7, mean at chance). Nearest-skeleton labeling
adds noise in the seam region of intertwined merges, which further depresses the
real AUC — so ~0.49 is a floor, and part of the synthetic↔real gap is method, not
only biology. Both effects point the same way: SegCLR-alone does not localize real
seams.

## Exp 2 — SegCLR top-1 retrieval (the RIGHT framing; corrects Exp 1)

Exp 1 asked "is the single most-discontinuous *edge* in the whole neuron exactly
at the seam?" — a harsh framing dominated by within-cell embedding variation, and
it made SegCLR look useless (AUC ≈ 0.52). The **retrieval framing** — for a node,
is its top-1 nearest node *in embedding space* the same cell, and among local
candidates does the same-cell one win — is the right question and tells a very
different, positive story.

```
python -m experiments.pcfg.run_compartment_grammar --exp2 --shards 0 1 2 3 --n-neurons 12
```

Data fact that reshapes the test: SegCLR **nodes are ~1.2 µm apart** (median NN
spacing ~1174 nm), so a 200 nm radius is *tighter than the node spacing* and finds
no candidates. The local test needs ~2–4 µm to have neighbours.

**Metric A — embedding top-1 retrieval (is the nearest-embedding node same-cell?):**

| set | top-1 same-cell | chance |
|---|---|---|
| 12 clean cells pooled | **0.866** | ~0.20 |
| real merges (per-object) | **0.83–0.98** (mean ~0.93) | — |

**Metric B — local top-1 same-cell at contacts (does SegCLR pick the correct
same-cell candidate over the false-merge partner?), on real merges:**

| radius | mean acc | note |
|---|---|---|
| 2 µm | ~0.71 | small n (8–860 discriminative nodes/merge) |
| **4 µm** | **~0.90** (0.76–0.95) | larger n (37–3295), stable |

**Corrected verdict.** SegCLR is a **strong local cell-identity signal**: its top-1
match is same-cell ~87–98% (retrieval), and at a real false-merge seam its top-1
local candidate is the correct same-cell one **~90%** of the time (at the
node-appropriate ~4 µm radius). It genuinely could drive merge/split decisions.
The Exp-1 "near chance" result was an artifact of the edge-discontinuity-ranking
framing, **not** a property of SegCLR. The compartment grammar and SegCLR are
complementary — SegCLR discriminates identity locally, the grammar supplies the
structural rules (A↔D, multi-soma) and the split geometry.

## M3 — leakage-safe column eval (train/eval by tangential PCA split)

### Merge grammar (multi-soma + A↔D), eval region, synthetic merges

negatives = clean proofread neurons; positives = synthetic merges (soma+soma and
axon-graft). Logistic combiner fit on TRAIN region, reported on EVAL region.

| | result |
|---|---|
| overall EVAL AUC | **0.840** |
| clean false-positive @0.5 | **0.05** |
| **multi-soma** (soma+soma) recall | **1.00**  (n_soma alone AUC = **1.000**) |
| **axon-graft** (1-soma) recall | **0.00**  (A↔D score AUC = 0.66) |

**Read:** multi-soma merges are trivially solved; the hard **single-compartment**
merge (no 2nd soma) is **not** reliably caught by the A↔D rule.

**Real-merge A↔D check** (7 real merges from the m343→current bridge, built from
the *real* descendant cells + their real synapses, bridged at contact):
- 2-soma merges (4/7): multi-soma catches all (ad irrelevant).
- 1-soma merges (3/7): A↔D fired on only **1/3** (scores 0.789, 0.0, 0.0). The
  hit was a genuine axon-fragment-onto-dendrite; the misses are **same-compartment**
  merges.
- clean neurons: ad_score max **0.617** (19% > 0.1) — false positives overlap the
  real merges.

**Conclusion (real data, not synthetic):** A↔D's weakness is fundamental, not a
tuning artifact — **not every merge crosses a compartment boundary.** Axon-to-axon
and dendrite-to-dendrite merges have no A↔D transition and are invisible to the
grammar (and to SegCLR-absolute). **Same-compartment single-soma merges are the
genuine open residual** for every method here. Multi-soma and cross-compartment
merges are handled; the rest need a different signal (e.g. SegCLR used
comparatively against a *reference* continuation, not an absolute scan).

### Split stitch (SegCLR top-1), dense 8-neuron column cluster (322 fragments)

| | result |
|---|---|
| top-1 SegCLR stitch (best-join neighbour = same neuron) | **0.73** (179/245) |
| absolute join-score AUC (same vs diff contacts) | 0.66 (same 0.923 / diff 0.890) |
| contacts | 255 same-neuron vs 405 diff-neuron |

The 2-neuron probe's 0.92 was **base-rate inflated** (mostly same-neuron contacts);
in a realistic dense patch SegCLR-alone top-1 is **0.73** (vs ~0.39 chance from the
candidate mix). Real signal, not a clean solve — the stitcher uses only embedding +
contact distance, no directional/tangent continuity, which real fragment stitchers
rely on. Adding geometry should lift this substantially, with SegCLR as tie-breaker.

### Split stitch — endpoint-based (the fix), dense 8-neuron cluster (558 endpoints)

Point-cloud contact stitching (above) was 0.73 because it counts crossing cables.
Switching to **fragment endpoints** (a true split joins two *ends*; crossings do
not co-locate endpoints) transforms the problem:

| gap | endpoints w/ continuation | contested | SegCLR-select | geometry-only |
|---|---|---|---|---|
| ≤4–10 µm | 44–93 | 0–1 | **1.00** | 1.00 |
| ≤15 µm | 150 | 12 | **1.00** (12/12 contested) | 0.98 (**0.75** contested) |

**Roles, now validated:** endpoint *proximity* generates candidates and already
disambiguates almost everything (crossings don't put endpoints together); **SegCLR
is the selector** — on the hard *contested* endpoints (a genuine competitor within
15 µm) SegCLR is **12/12** while colinearity-geometry is **9/12**.  Overall
**150/150**.  So for splits: geometry proposes, SegCLR decides.

### Synthesis (both error types)

The architecture is symmetric — **structure proposes, SegCLR decides**:
- **Merge**: the grammar proposes candidate seams (multi-soma solved AUC 1.0;
  A↔D localizes) and SegCLR corroborates.  Multi-soma merges are solved;
  single-compartment (axon-graft) merges are the open residual (both A↔D and
  SegCLR-absolute are weak — SegCLR has no *comparison* to make on a single walk).
- **Split**: endpoint proximity proposes candidate continuations and SegCLR
  selects — **1.00** on the column cluster, resolving the contested cases that
  geometry alone misses.  This is SegCLR's comparative strength, realised.


## Contiguity + global matching — right idea, wrong substrate (SegCLR)

Insight (correct): the fragments were **one continuous process** that segmentation
cut, so a true split = two fragments that were *contiguous* (coaxial end-to-end,
matching caliber), whereas a false contact = two cells with a membrane between.
And a cut has two faces → each fragment endpoint rejoins **≤1** partner → **matching**
(global), not agglomeration → structurally no blob cascade.

Tested on the same dense patch (265 fragments):
- The **coaxial end-to-end filter is precise**: 18 candidate joins, **17
  same-neuron** — it cleanly rejects the side-by-side different-cell contacts that
  wrecked precision before. *Contiguity geometry is the right discriminator.*
- But **recall ≈ 0** (18 candidates vs ~172 real adjacencies). The coaxial cut-face
  signal cannot be extracted from **sparse (~1 µm), overlapping SegCLR point
  clouds**; m343 fragments interleave rather than tile end-to-end.
- Matching vs agglomeration is moot at 18 candidates; SegCLR adds nothing.

**Takeaway:** contiguity is the right *signal* and matching the right *global
structure*, but the continuity lives in **skeleton geometry / EM cut-faces**
(caliber, cross-section, membrane apposition) — **not** in SegCLR type embeddings.
Split-fixing is a **geometric-continuity** problem (standard skeleton/mesh
agglomeration), and SegCLR is the wrong tool for it, just as it lacks the identity
needed for same-type merges.

## ★ Landmark question: can we JOIN fragments to fix splits WITHOUT introducing mergers, from local info? — NO

The goal in the repo's currency: put each neuron's **half-synapses** back on the
correct neuron. Start = m343 fragments (under-merged). Stitch fragments by contact,
score joins by {distance, geometry=tangent colinearity, SegCLR, combined},
agglomerate, measure **synapse-pair precision/recall/F1** vs true neuron. Dense
8-neuron column patch, 279 fragments, 34 424 half-synapses.

| | precision | recall | F1 |
|---|---|---|---|
| no-stitch (baseline) | 0.96 | 0.76 | 0.848 |
| **ORACLE** — join only *correct* same-neuron contacts | 0.96 | 0.90 | **0.928** |
| distance / geometry / segclr / combined (any threshold) | **0.14** | 0.9+ | **0.24** |

**The good joins exist** (oracle fixes most splits: recall 0.76→0.90 at precision
0.96). But **every local method collapses precision 0.96 → 0.14** — it merges nearly
everything into one blob. Three compounding reasons:
1. In dense tissue, **different-neuron contacts outnumber same-neuron** (247 vs 172):
   most touching fragments are *different* cells.
2. Local grammar/geometry/SegCLR **cannot separate same- from different-neuron
   contacts** (SegCLR = type not identity; colinearity is ambiguous at a contact).
3. **Agglomeration transitivity is catastrophic** — ~7 wrong joins chain all 8 cells
   into one component; every method's accepted set contains far more. One bad join
   fuses two whole cells.

**Answer:** the landmark ("accurate fragment-joining without introducing mergers
from cheap local info") **does not work** in dense tissue. The bottleneck is not
recall (splits are locally fixable) but **precision under agglomeration** — and
that needs cell *identity*, which SegCLR does not carry (below). A viable path is
non-agglomerative / globally-consistent partition with a hard merge-cost, not
greedy local joins.

## Can tweaking open SegCLR recover cell identity? — NO (not decodable)

SegCLR is open code, so: could a tweak make it encode *identity*, not just type?
Test: train a supervised metric head on **frozen** SegCLR to separate same-cell
from **different-cell-SAME-TYPE** pairs (held-out cells; 8 cells, mostly 23P).

| | AUC |
|---|---|
| raw cosine (SegCLR as-is) | **0.513** |
| learned metric head (frozen SegCLR) | **0.509** |

Both at **chance**. Individual identity is **not present** in the frozen embeddings
for same-type cells — so no simple head/metric extracts it. This is a consequence
of the ~4 µm receptive field: two same-type axons are genuinely indistinguishable
in local EM. The only levers that could help are ones that add *information*:
retrain SegCLR with a **larger receptive field + distant same-cell positives**, or
use **EM-native seam features** (membrane continuity at the actual contact) — not a
post-hoc tweak to the released embeddings.

## Split fixer 2×2 ablation (geometry × SegCLR) + merge detector

**Split — contested endpoints** (a same- AND a different-neuron endpoint both
within 15 µm; n=12), top-1 same-neuron continuation:

| | SegCLR off | SegCLR on |
|---|---|---|
| geometry off (distance only) | **0.50** | **1.00** |
| geometry on (colinear tangent) | **0.75** | **1.00** |

(all-with-continuation, n=150: neither 0.96 / geom 0.98 / segclr 1.00 / both 1.00.)

### ⚠ Critical review — do NOT over-read the split "1.00"

Honest caveats (the headline number is real but narrow):
1. **n = 12 contested.** "1.00" is 12/12; the 95% CI is roughly [0.76, 1.0]. Not
   "solved" — directional only.
2. **Lenient metric.** We score "top-1 candidate is *same-neuron*", not "top-1 is
   the *true adjacent continuation*". A neuron's arbor folds back on itself, so a
   *different* branch of the same cell often sits within 15 µm; joining it counts
   as "correct" here but would build wrong topology. The numbers are an **upper
   bound on same-cell selection**, not correct-reconstruction accuracy.
3. **Type-not-identity confound (the big one).** Per the headline finding, SegCLR
   resolves a contested split **only when the wrong candidate is a different
   type/compartment**. The 12 contested wins were almost certainly
   type-distinguishable; a *same-type* confusor (another same-type axon passing by)
   would not be resolved by SegCLR. So "SegCLR decides splits" really means
   "SegCLR decides splits *when the distractor is a different type*."
4. **Task is mostly trivial.** At ≤6 µm even distance-only scores 1.00 — the true
   continuation is usually the only thing nearby. The interesting residual
   (contested) is ~8% of endpoints.
5. **Endpoint/enumeration incompleteness.** Only 2 PC1-extreme endpoints per
   fragment; sparse seg-lookup (150 probes) may miss small fragments — so some true
   continuations are absent from the candidate set and uncounted.

**Honest split verdict:** *geometry* does the heavy lifting (proximity resolves the
large majority); *SegCLR adds value as a type/compartment disambiguator on a small
contested slice*, validated only at n=12 with a lenient metric. Not "SegCLR solves
splitting."

**Merge — standalone detector** (max split-score over a graph-diameter walk,
clean cells vs real merges):

| statistic | clean med | merged med | AUC |
|---|---|---|---|
| absolute step | 0.068 | 0.088 | 0.671 |
| comparative | 0.059 | 0.079 | 0.657 |

Both ~0.66; comparative is **not** better *as a detector*. Clean cells have real
internal identity structure (axon vs dendrite, proximal vs distal) whose
comparative peaks rival a merge seam, so **no SegCLR statistic scans for merges
alone**. (Comparative *is* better at *localizing* a known/proposed candidate cut —
4.82× vs 3.78× — detection ≠ localization.) **The same-compartment residual needs
a candidate-cut *proposer*, not a better scorer.**

## Candidate-cut proposer (branch points + comparative-SegCLR) — NEGATIVE

Attack on the same-compartment residual: propose a cut at every branch point
(degree≥3), score each by comparative split-vs-joined over geodesic windows, rank.
Validated on real merges (reconstructed merged skeleton, bridge = true seam):

| merge | true-seam rank | seam score | top score |
|---|---|---|---|
| …589909259 | **149/149** | −0.001 | 0.058 |
| …847908702 | 28/301 | 0.056 | 0.255 |
| …467909772 | 105/134 | 0.005 | 0.056 |

TOP-1 = **0/3**; clean neurons' max cut score (0.10–0.13) *exceeds* the real seam
scores → **fails, worse than useless as a detector**. Two fundamental reasons:
1. **A same-compartment seam is locally *similar*** — cells merge where they touch
   and look alike, so across-similarity at the contact is high → comparative ≈ 0.
2. **Branch points are intrinsically high-comparative** — a branch is where one
   cell's cable diverges into locally different morphology, so the seam competes
   against the hardest candidate set and loses.

This also explains the earlier walk's 4.82× (its baseline was *random* path points,
not branch points). **Cells are not tight embedding clusters** (identity is a
manifold: axon-region, dendrite-region…), so cluster-separation cannot isolate a
same-compartment seam. **The residual is genuinely hard; SegCLR's power is local
comparison (splits: top-1), not global identity clustering (merges).**

## Comparative merge detector — attacking the same-compartment residual

The same-compartment merge (axon-axon / dendrite-dendrite, one soma) beats the
grammar (no A↔D transition) and the *absolute* SegCLR step (within-cell drift
competes).  Fix: make SegCLR **comparative** — at a candidate cut, ask "are the two
sides better modelled as *one* identity or *two*?"  Statistic =
``within-side cohesion − across-side similarity`` (`comparative_split_score`): a
true identity switch makes within ≫ across; gradual within-cell drift keeps
within ≈ across, so it stays low.

Real merges (2-cell, walk across the seam), seam/away contrast:

| merge | absolute step | **comparative** |
|---|---|---|
| …589909259 | 2.81× | **3.41×** |
| …847908702 | 6.60× | **8.72×** |
| …467909772 | 1.94× (peak **319** nodes off) | 2.33× (peak **36** nodes off) |
| **mean** | **3.78×** | **4.82×** |

The comparative statistic is uniformly higher-contrast and, on the merge where the
absolute step's peak was 319 nodes from the seam, pulls it to 36 — it is robust to
the within-cell drift that defeats the absolute step.  Not a full solve (n=3; peak
still ~36–45 nodes off in 2/3), but the right direction: **grammar localizes
candidate cuts, comparative-SegCLR scores split-vs-joined at each.**

## Walk detector — the merge/split asymmetry (key design finding)

Idea: lay SegCLR embeddings along the skeleton, rolling-average them, and detect
where local identity changes (merge) or should continue (split).

- **Merges (absolute step magnitude): weak.** Walking a synthetic merge, the
  bridge seam scores only **0.036–0.053** while within-cell variation peaks at
  **0.25–0.30** — the seam does not stand out (its max was 246 µm from the true
  bridge). Same lesson as Exp 1: absolute embedding change is not merge-specific.
- **Splits (comparative top-1): strong.** Stitching a neuron's m343 fragments,
  each fragment's *highest*-SegCLR-similarity contacting neighbour is the correct
  **same-neuron** continuation **85/92 = 92%** of the time. But the *absolute*
  join-score barely separates same- vs different-neuron contacts (0.942 vs 0.917,
  AUC **0.659**).

**Conclusion — SegCLR is a comparative signal, not an absolute one.** It excels at
*choosing among candidates* (splits: which fragment continues this cable — top-1
~0.9) and is weak at *flagging a seam by magnitude* (merges). Therefore:

- **Merge detection → the structural grammar** (multi-soma + A↔D crossing) is the
  primary detector; SegCLR only corroborates the *few* candidates the grammar
  localizes (not a standalone scan).
- **Split fixing → SegCLR top-1 continuation matching** (`walk_detector.stitch_fragments`)
  is the primary tool — a comparative decision, which is SegCLR's strength.

This asymmetry is the load-bearing result for the whole design.

## M1 — compartment labeling on real neurons (PASS)

Built `neuronauts/soma_clusters.py` (verified soma routine extracted to core) and
`experiments/pcfg/compartments.py` (`label_compartments`): synapse polarity
(pre→axon, post→dend) snapped to vertices and diffused along the tree, soma from
large-radius clusters (+ optional nucleus table). Run:

```
python -m experiments.pcfg.run_compartment_grammar --m1 --n-neurons 4
```

Result on 4 proofread neurons (5k–12k verts, 2k–8k synapses each):

| root | n_soma | PRE→AXON | POST→DEND | is_tree |
|---|---|---|---|---|
| …686494647 | 1 | 0.99 | 1.00 | ✓ |
| …812081779 | 1 | 0.99 | 1.00 | ✓ |
| …195284556 | 1 | 1.00 | 1.00 | ✓ |
| …975539779 | 1 | 0.98 | 1.00 | ✓ |

Polarity concordance is essentially perfect and each neuron yields exactly one
soma. Useful data fact: the CAVE skeleton service represents the **soma as a
single large-radius vertex** (~5300 nm; all cable ≤ ~425 nm at p99), so the
radius>3000 nm threshold is cleanly separated and multi-soma detection (2 merged
cells → 2 big-radius vertices) is robust. The compartment alphabet is ready to
drive the grammar.

## Next
1. **Grammar productions (M3)**: A↔D-crossing (geodesic windows, soma-mediation
   guard) + multi-soma, locating the offending edge; combine with the PCFG signals.
   SegCLR enters only as a weak corroborator (Exp 1).
2. Split proposal via `skeleton_cut_op`; evaluate on real merges (m343→current)
   and vs the `atomicity_detector` / `skeleton_topology_merge` baselines.
3. Optionally de-noise the Exp 1 ground truth (supervoxel→current-root labels) to
   pin down the real-merge SegCLR AUC precisely.
