# Findings — synapse-level correction model `f(v117) → v1718`

Learning the proofreading correction at the synapse level: join each synapse-side to
itself across materializations via its immutable supervoxel, label every pair by
"same later root?", and learn the affinity. Grouped-by-cell CV (union-find over
v117↔later co-occurrence), permutation nulls on every number.

Data: 7 boxes (40 µm) tiled on the densely-proofread column center, v117 → v1718.
485,988 synapse-sides, 107,996 v117 roots → **391 false-merges (cut)**, **85
false-splits (join)**. Cached SideTable: `data/sidetable_7box.npz`.

## Headline

| stratum | model | AUC | null | n pairs | pos% |
|---|---|---|---|---|---|
| **split** (de-merge, within v117 root) | logreg | **0.847** | 0.73 | 929,209 | 98.8% |
| **merge** (de-split, cross v117 root)  | RF | **0.979** | 0.53 | 1,284 | 25% |

Both beat their permutation null at p < 0.001.

## What the ablation established (`ablate_merge.py`)

Merge-stratum AUC under feature ablations and an A/B order-randomization probe:

| variant | logreg | RF |
|---|---|---|
| full (43 feats) | 0.341 | 0.979 |
| order-random | 0.309 | 0.979 |
| no-grammar (9) | 0.320 | 0.989 |
| geom-only (6) | 0.119 | 0.988 |
| dist-only (1) | 0.013 | 0.689 |
| size-only (2) | 0.502 | 0.560 |

The RF-vs-logreg gap (0.98 vs 0.34) first read as a leakage smell (cf. the berlin
cell-identity-leakage retraction). The battery says otherwise:

- **Not an ordering artifact** — `order-random` is identical (0.979 → 0.979).
- **Not morphological grammar** — `no-grammar` is identical (0.989); the bigram grammar
  adds nothing to merges.
- **Not a "tiny fragment" size cue** — `size-only` ≈ chance (0.56).
- **It is local continuity geometry** — `geom-only` = 0.988, while `dist-only` is only
  0.689. So it is the *combination* of distance + axial/lateral separation + caliber/density
  continuity, not mere proximity. This is the legitimate de-split discriminator: are two
  adjacent cross-root fragments the same neurite?
- logreg scores < 0.5 because the geometry→label relation is **nonlinear** (mid-range
  continuity is positive; both very-near and far are negative) — a linear model inverts it.

For the **split** stratum the story is geometry-led with a *modest, genuine grammar lift*:
logreg geom-only 0.799 → full 0.847; the tangled-arbor bigram signal helps locate the seam.

## Caveat that bounds the merge number (do not over-read 0.98)

Merge **negatives are "any nearby different-cell pair," not adversarial.** The hard,
deployment-relevant negative — two *different* cells that are themselves collinear and
caliber-matched (the false-merge trap) — is under-represented. So 0.98 overstates true
de-split difficulty. A deployment-honest number needs **continuation-like hard negatives**
(nearby cross-root pairs selected to be collinear), which is the next experiment.

Also: the **split** stratum is 98.8% "keep" — AUC is encouraging but the corrections of
interest are the rare ~1%; report precision/recall@k (catch-vs-flag) before calling it
deployable.

## Regime check for the self-supervised grammar (`grammar_regime.py`, cached, label-free)

Can we learn the grammar from raw segmentation and skip the edit labels? Only if the
presegmentation is "good but imperfect" — correct structure must dominate the local
statistics. Measured on the cached 7-box SideTable (v117 = contaminated input,
v1718 = clean reference, same synapses; labels mark seams only, never train):

| measurement | value | reading |
|---|---|---|
| transition-level pollution | **1.2%** of within-arbor steps cross a seam | regime holds — correct structure dominates |
| arbors with ≥1 seam | 1.4% | errors are sparse and local |
| grammar drift KL(clean‖contam) | **0.0002 bits** | contamination is **free** — no labels needed to learn the grammar |
| anchor gate, **bigram-token** surprise | AUC **0.539** (null 0.50, p<0.001) | the F/B/L/R grammar is nearly **blind** to seams |
| anchor gate, **geometry** surprise (2-D Gaussian on log-len + turn-angle) | AUC **0.658** (null 0.50, p<0.001) | keeping continuous geometry recovers signal |
| seam-transition stereotypy | entropy 2.64/4.0; L↔R reversals ~1.8× lift | partly systematic, but tokens overlap clean arbors |

**Conclusion: the bottleneck is the representation, not contamination.** The raw data is
clean enough to self-supervise on (KL≈0), and you do *not* need edit labels to train the
grammar. But the lossy 4-token alphabet discards the geometry where the error signal lives:
a trivial geometry-preserving density already beats it (0.66 vs 0.54), against a supervised
ceiling of 0.85 on the same seams. So the path is a **richer self-supervised geometric
grammar** trained on raw arbors, scored by surprise — label-free, deployable beyond the
proofread column.

Caveat unchanged: seams are partly systematic (stereotyped as lateral reversals), so an
anomaly model will still miss the most systematic presegmentation biases — that residual is
what the scarce edit labels are for.

## Course-correction: anomaly is the wrong framing; gap is trivial; continuation is the prize

Pushing the self-supervised grammar with more features + a kNN density (`selfsup_grammar.py`)
went the *wrong* way — and that is the useful finding:

| scorer (de-merge seams, label-free) | AUC |
|---|---|
| bigram-token surprise | 0.539 |
| 2-D Gaussian (log-len, turn-angle) | 0.658 |
| kNN density, 5 features | **0.587** (worse — more features hurt) |
| **gap-after alone (1 raw feature, no learning)** | **0.813** |
| supervised pairwise (geometry) ceiling | 0.85 |

**A de-merge seam is not a generic anomaly — it is a specific directional event (a spatial
gap between two lobes).** An undirected density/anomaly score dilutes that gap among
irrelevant features and flags the wrong junctions; one directed feature (the gap) gets 0.81
with no labels and no grammar. So:

1. **De-merge (split) is geometrically near-trivial** — gap alone ≈ supervised. An expressive
   grammar earns almost nothing here.
2. **The grammar's real value is de-split (merge)** — there a gap exists in *both* the
   should-join and should-leave cases, so gap cannot discriminate; you need a *directional
   continuation* model (do these fragments continue each other: collinearity, caliber match).
3. **The right self-supervised objective is PREDICTIVE, not density** — next-synapse /
   coherent-vs-spliced continuation (the existing path-encoder Stage-2 objective), which is
   directional, rather than anomaly/density which is omnidirectional.

## Reality check 1 — group-level, do-nothing-protected eval (the headline)

Pairwise AUC was a mirage. Evaluating the supervised pairwise model at the synapse-group
level against the do-nothing baseline (`group_eval.py`, grouped CV):

- splits needed = **11,363 pairs**, merges needed = **321 pairs** (do-nothing recall 0).
- The logreg corrector (the one with split AUC 0.85 / merge AUC 0.98) is **net-NEGATIVE at
  every threshold** — it makes the partition *worse* than doing nothing:

| thr | edits proposed | edit prec | edit recall | split_rec | merge_rec | net_fixed |
|---|---|---|---|---|---|---|
| 0.50 | 210,675 | 0.04 | 0.75 | 0.77 | 0.02 | **−193,169** |
| 0.70 | 365,690 | 0.03 | 0.88 | 0.91 | 0.02 | **−345,014** |

At 1.2% base rate, even a 0.85-AUC ranker floods the partition with false edits
(precision 2–6%); to catch half the real splits it proposes ~100k edits, ~94% wrong.
**Merge recall is ~0.02 — it catches essentially no merges.** This is exactly the
do-nothing trap: AUC looked great, deployment is catastrophic. The real bar is
**precision at useful recall**, and we are nowhere near the break-even (~0.5).

## Reality check 2 — learned end-to-end grammar (raw coords only)

A tiny BiGRU coherence encoder on raw coordinates + displacements (no hand features),
self-supervised coherent-vs-spliced, cells held apart (`learned_grammar_neural.py`):

- **Mastered the proxy task**: val AUC 0.820 (coherent vs spliced), 175 s training.
- **But transfers to real errors at chance**: seam anchor-gate AUC **0.479** (null 0.474),
  de-split/merge AUC **0.456** (null 0.479) — worse than gap (0.81) and even the bigram (0.54).

Why: a synthetic splice introduces an abrupt coordinate jump, so the encoder learned
"discontinuity = incoherent" — the trivial gap signal again. But real false-merges are
*gap-free* (CV wrongly fused two touching processes with no jump), so the learned cue is
the wrong one. **The self-supervised proxy ≠ the real error distribution.** End-to-end
learning did not beat hand features here — not because representation learning is wrong,
but because the *training signal* (synthetic splices) doesn't match real CV mistakes.

## Where this leaves us

The bottleneck moved from representation to **training signal + eval realism**:

1. Stop trusting pairwise AUC; the group-level do-nothing guardrail (precision at recall,
   net_fixed) is the only honest score, and nothing clears it yet.
2. The learned model needs **real-error-shaped negatives** (gap-free adjacency merges that
   mimic CV statistics) or **real edit labels**, not synthetic splices.
3. The merge direction remains the prize and the hardest: ~0.02 recall today.

## Next

1. **Self-supervised continuation model on the merge problem.** Train coherent-vs-spliced
   (path-encoder objective) on raw arbors; evaluate on the de-split (merge) stratum where the
   gap heuristic is useless. This is where learning should pay — point the expressive model
   here, not at de-merge.
2. Clean-column vs un-proofread-**bulk** drift (needs a bulk fetch) → confirm the 1.2%
   regime holds outside the column, where the corrector must actually run.
3. PR@k / catch-vs-flag curves per stratum; then repartition (affinity → components) scored
   against v1718 by edge-F1 — the end-to-end correction.

## Reframe pays off: global whole-object beats local (synapse-cloud proof; skeletons next)

Local methods plateaued and failed the do-nothing guardrail. Reframing to whole-object
global shape (`global_shape_merge.py`, cached synapse clouds, no skeletons -- MICrONS
skeleton service was down 503): classify each v117 root as false-merge (spans >=2 v1718
roots) from GLOBAL shape only (bimodality, 2-means gap, DBSCAN blob count, PCA shape).

objects=9,377  false merges=354 (3.78% base rate); grouped-by-cell CV, null 0.50:

| features | model | AUC | prec@top-2% | recall |
|---|---|---|---|---|
| size-only | logreg | 0.844 | 0.30 | 0.16 |
| shape-only | logreg | 0.860 | 0.35 | 0.18 |
| full | rf | 0.875 | **0.41** | 0.22 |

- **Precision 41% at the object level vs 2-6% for local pairwise** (~11x over base rate) --
  the first operating point that could beat do-nothing.
- **Shape adds beyond size** (0.844 -> 0.878; shape-only 0.86) -- global structure, not just
  "merges are big", carries signal. The global reframe is validated.
- This is the FLOOR: synapse clouds see only spatially-separated merges; intertwined merges
  (two somas/two trunks, spatially overlapping) need skeleton topology. That residual is the
  case for the skeleton-topology model once the MICrONS skeleton service is back up.

## Self-supervised generative grammar (the agreed model) — scale test

After several reframes (learn don't build; generative not discriminative; skeleton+synapse;
self-supervised on noisy data), the model: a denoising PointNet autoencoder over each
object's raw points (skeleton vertices + its synapses, xyz only), Chamfer reconstruction,
trained on real (noisy) neurons. No labels, no hand features, no synthesized merges. Errors =
low grammaticality (high reconstruction error); the reconstruction = the proposed correction.
`skel_ssl_grammar.py`.

| corpus (clean cells) | AUC(merge\|recon err) | base | prec@10% / recall |
|---|---|---|---|
| 176 (40 ep) | 0.738 | 26% | 0.54 / 0.21 |
| **674 (45 ep, ~4×)** | **0.774** | 8.8% | 0.27 / 0.31 (~3× base) |

- **Premise confirmed directionally**: scaling the noisy corpus lifts the grammar
  (0.74 → 0.77; fold-1 0.86) — "noisy data buys scale" holds (clean-vs-noisy KL was ~0).
- **But diminishing returns**: 4× data → +0.036 AUC; scale alone won't reach the hand
  baselines (synapse-cloud 0.88). The limiter is now **model capacity**, not data: a single
  128-d global latent decoding 256 points is too coarse to express the multi-scale structure
  that makes a merge improbable.
- Honest standing: a pure unsupervised generative grammar (zero labels/features/negatives)
  flags real merges at 0.77 / ~3× enrichment, with the reconstruction as the fix. The next
  lever is a richer generative model (multi-scale latent, include radius + synapse type),
  not more data.

## Richer model (v2) + both error types — capacity is NOT the lever; the objective is

v2 (`skel_ssl_grammar_v2.py`): same framing, richer model — multi-scale DGCNN encoder, raw
radius + skeleton-vs-synapse channels (fed in and reconstructed), bigger latent/more points.
And the eval now scores BOTH error types with the one grammar (per the merge+split reminder).

| | v1 PointNet-AE | v2 DGCNN (richer) |
|---|---|---|
| merge vs clean | 0.774 | **0.776** |
| split vs clean | — | 0.727 (rec 0.44@10%) |
| any error vs clean | — | 0.762 (prec@10% 0.36, ~3× base) |

- **One unsupervised grammar catches BOTH merges (too much structure) and splits (too little)**
  — no labels, no split supervision, reconstruction corrects in opposite directions. The
  framing is validated.
- **But the richer model gave ~nothing on merges (0.776 vs 0.774).** Neither scale (0.74→0.77)
  nor capacity (≈0) closes the gap to the supervised hand baseline (0.88).
- **Diagnosis — the ceiling is the OBJECTIVE, not the model.** A reconstruction AE minimizes
  average reconstruction, so a merged object reconstructs only slightly worse than a clean one;
  a bigger encoder reconstructs everything better without *widening the clean-vs-error gap*.
  Reconstruction error is a weak proxy for grammaticality (classic AE-anomaly failure).
- Next lever is a generative objective where errors are explicitly low-LIKELIHOOD, not just
  slightly-higher-recon-error: a density/likelihood model (flow/autoregressive grammar) or a
  predictive/contrastive objective — not a bigger autoencoder.

## Autoregressive connected-synapse grammar — scored the wrong production (edge geom, not branching)

`synapse_grammar_ar.py`: a literal grammar P(next connected synapse | trajectory), over a
branching synapse tree (parent-relative tree-edge displacements; spanning-tree construction
-> no self-loops; branching preserved). Causal transformer, per-step Gaussian NLL.

| | merge | split | any |
|---|---|---|---|
| AR edge-geometry (30 ep) | 0.627 | 0.518 | 0.611 |
| reconstruction AE (ref) | 0.776 | — | — |

- **Underperforms the AE and full training didn't help** (flat from 3-ep). Splits at chance.
- **Diagnosis: the likelihood scores edge GEOMETRY, not branching TOPOLOGY.** A merge's
  anomaly is an extra subtree / improbable node degree / two roots (and the artificial
  nearest-point bridge makes the seam edge *short* -> looks normal); a split's anomaly is
  premature TERMINATION (a cut where the cable should continue). The tree was built as a
  representation but the grammar never scores the branching decisions -- so "allow branching"
  is exactly what the NLL ignores.
- **Fix (the real branching grammar):** per node predict branch DEGREE (0=tip/stop, 1=continue,
  >=2 branch) and termination, not just edge displacement. Then a merge node (improbable degree
  / extra subtree) and a split tip (stop where continuation expected) both become low-probability.
  That makes the grammar score topology, which is where both errors live.

## Branching-degree grammar + peak scoring — autoregressive direction underperforms (conclusion)

Added the diagnosed fix: per-node branch-DEGREE prediction (0=tip,1=continue,2/3+=branch) so
the likelihood scores branching topology, plus peak (90th-pct) per-node NLL scoring to
un-dilute a local seam.

| variant | merge | split | any |
|---|---|---|---|
| edge-only | 0.63 | chance | 0.61 |
| + degree (mean NLL) | 0.64 | 0.52 | 0.62 |
| + degree (peak NLL) | 0.63 | 0.57 | 0.62 |
| reconstruction AE (ref) | **0.78** | — | — |

Neither degree nor peak rescued it; ~0.63 merge / chance split across all variants, well below
the AE. **Conclusion: the autoregressive single-object synapse grammar underperforms**, for
structural reasons, not tuning:

- **Splits are out of scope for a per-object grammar.** A severed fragment is a valid *small*
  neuron in isolation; its wrongness is RELATIONAL (relative to the cell it was cut from). With
  synapses only (no soma/caliber), nothing local flags the cut. Catching splits needs a
  relational model (does this tip want to continue into a neighboring fragment), not a grammar.
- **Merge anomaly is GLOBAL, not sequential.** "Two arbors" is a whole-shape fact the AE reads
  directly (two blobs); the local trajectory grammar can't surface it, and peak-of-sequence
  didn't help.

Standing of the whole generative program: elegant and label-free, validated in concept (one
grammar, both error types), but it has NOT beaten the simpler global whole-object detector
(synapse-cloud 0.88 / 41% precision is still the strongest, and the group-level do-nothing
guardrail remains the unbeaten bar). The strongest signal is GLOBAL + RELATIONAL, not a
per-object autoregressive grammar.

## Closing the loop: detection is NOT the bottleneck -- the CUT OPERATOR is

`close_loop_merge.py`: global detector (RF on whole-object shape, grouped-CV OOF) -> flag ->
CUT flagged object via 2-means on its synapses -> score corrected partition vs v1718 by
Rand-disagreement pair counting (fixing AND breaking both count) vs the do-nothing baseline.

do-nothing within-object pair errors (merges to cut) = 796,390. Net fixed:

| flag thr | flagged | true+ | false+ | net_fixed | % of base |
|---|---|---|---|---|---|
| 0.70 | 85 | 39 | 46 | −1,360,568 | −171% |
| 0.80 | 21 | 11 | 10 | −514,351 | −65% |
| **oracle: cut every TRUE merge** | | 354 | 0 | **−1,875,596** | **−236%** |

**net_fixed is negative at every threshold AND at the oracle.** So detection (~0.88) is fine;
the 2-means CUT makes the partition far worse even applied to perfectly-detected real merges.

**Why — merges are imbalanced.** The 2nd v1718 component is a median **11% of the object**;
**47% of merges have it under 10%** (a big cell + a small embedded fragment), only 18% are
balanced (>30%). A balanced spatial 2-means shreds the big cell instead of peeling the small
fragment, so it introduces ~3× more pair errors than it fixes.

**Verdict (assessment).** The open problem is the **correction operator**, not detection. A
deployable corrector needs a **surgical, connectivity-aware, imbalance-aware cut** — peel the
small fragment along the skeleton at the seam — not a spatial clustering. This is the first
result that clears the do-nothing guardrail's logic: it tells us exactly what must be built
(the cut), and that naive cutting is actively harmful. Detection-first framings (incl. the
whole generative-grammar program) were optimizing the wrong half.

## Connectivity cut on real v117 skeletons -- VIABLE (+79% oracle): the seam is one edge

`close_loop_cut.py`: cut each real over-merged v117 object along its actual skeleton (fetched
v117 skeletons), score vs v1718 by Rand-disagreement, net vs do-nothing. 119 objects with
skeletons, do-nothing pair errors = 348,183.

| cut operator | pair errors | net_fixed | % of base |
|---|---|---|---|
| kmeans (spatial) | 1,175,487 | −827,304 | −238% |
| radius_jump (caliber discontinuity) | 578,633 | −230,450 | −66% |
| min_radius (thin neck) | 395,515 | −47,332 | −14% |
| **ORACLE (best single skeleton edge)** | **71,747** | **+276,436** | **+79%** |

- **First operator all session to beat do-nothing, decisively (+79%).** The best single
  skeleton-edge cut removes 79% of merge pair-errors -- so the false-merge seam is (mostly)
  ONE edge on the real cable topology.
- **Must be the skeleton.** kmeans on synapses is −238%; the seam is a connectivity feature,
  invisible spatially because the merged cells touch.
- **The remaining problem is seam-edge DETECTION, and it is learnable.** Unsupervised
  heuristics (thin-neck −14%, caliber-jump −66%) miss the seam, but the oracle proves the
  signal is there. The gap −14% -> +79% is a per-edge "is this the false-merge seam" classifier
  on the skeleton (we have the labels: v1718 gives the true seam edge per object).

This reframes the deliverable: detection of bad OBJECTS was never the hard part; the hard,
do-nothing-beating part is detecting the seam EDGE and cutting it -- and that is now shown to
be both viable (+79% ceiling) and a well-posed learning problem.

## Learned seam detector -- viable operator, but autonomous detection is data-starved

`seam_detector.py`: GraphSAGE GNN over each v117 over-merged skeleton (raw inputs: vertex
xyz + log radius + local synapse count), trained to regress per-edge CUT BENEFIT, with
abstention (cut only if predicted benefit > tau). Grouped-by-cell CV, 140 merge objects.

| operator | net_fixed | % base |
|---|---|---|
| oracle (best single edge) | +300,677 | +77.9% |
| min_radius heuristic | −61,879 | −16% |
| **learned GNN, cut-always** | **−140,182** | **−36%** |
| learned GNN, abstain (tau=0.5, 41 cuts) | −715 | −0.2% |

- GNN top-1 seam accuracy 25% (>> random; up from 13% undertrained) -- it learns signal, but
  not enough: a near-miss edge shreds the big cell, so wrong picks are catastrophic.
- Abstention works as designed (bounds downside at do-nothing) but never reaches net-positive --
  predicted confidence doesn't track cut quality tightly enough.
- **Binding constraint = autonomous seam-edge detection: hard + data-starved.** 140 merge
  objects (column ceiling ~354) is far too few to learn to pick one correct edge among
  hundreds under a brutal cost asymmetry. Same wall as every learned model this session: high
  oracle, unreachable on column-scale data.

## State of the merge-correction problem (mapped end to end)

- detect bad OBJECT: easy (whole-object shape, ~0.88 / 41% precision).
- CUT operator: viable -- best single skeleton edge is +79% over do-nothing (the seam is one
  edge on the real cable; spatial cuts are −238%).
- find the seam EDGE: the binding constraint -- oracle +79%, learned 25% top-1 / net ~0.
Realistic near-term value is human-assisted (model proposes top-k candidate cut edges; a
proofreader selects) or a much larger training corpus (more proofread boxes), not autonomous
cutting at column scale.

## Clear metrics + counts, and the top-k human-assist measurement

Intuitive re-report of the oracle single-edge cut (`cut_report.py`, 150 real false-merge
objects, 21,119 synapses, median 2 cells each):
- **splits applied = 134, merges (joins) applied = 0** (one cut per object; >2-cell objects
  need >1 cut, not done).
- **86% of synapses placed in the correct cell** after the cut; per-object separation
  accuracy median 0.98 / mean 0.85; **75% of merges cleanly separated by one cut** (>=90%).
- The "+79%" headline is a within-object synapse-PAIR (Rand-disagreement) reduction
  (401,642 -> 87,385 = 78%); quadratic, so big cells dominate -- that's why it differs from
  the plainer 86% / 75%.

**Which detector is in the loop:** none -- the cut experiments (`close_loop_cut`,
`seam_detector`, `cut_report`) run on the KNOWN false-merge objects to isolate the cut/seam
problem (perfect detection assumed). The error DETECTOR is a separate stage: the global
whole-object shape RF (~0.88 AUC / 41% precision, `global_shape_merge.py`). A deployed
pipeline chains detector -> seam-cut, so end-to-end is below these "given-known-merge" numbers.

**Top-k human-assist** (`seam_detector.py`): model proposes top-k candidate cut edges, a
proofreader applies the good ones (skips harmful). 150 objects, oracle ceiling +77%.

| k | best-of-k net (apply all) | + human verify (apply only helpful) |
|---|---|---|
| 1 | −33% | **+14%** |
| 3 | −25% | **+18%** |
| 5 | −21% | **+21%** |
| 10 | −2% | **+26%** |

- **As a human-assist tool it works**: propose cuts, human verifies -> +14% (k=1) to +26%
  (k=10) of merge pair-errors fixed (~1/3 of the +77% ceiling at k=10).
- **Autonomous it does not**: apply-all is net-negative (the model can't self-tell good cuts
  from bad; predicted-benefit abstention tops out at ~0). The lift requires human verification.
- (oracle-edge-in-top-k looks low at 3-14% because many near-seam edges tie on cut quality;
  the meaningful number is the verified best-of-k net.)

## Recursive corrector built (Phases 1-4): connectivity metric + recursion + RF stop

The plan's pipeline, validated on cached data (150 v117 over-merged skeletons, grouped):

| operator | pooled pair net | pre-side acc | post-side acc | connectivity | splits |
|---|---|---|---|---|---|
| single oracle cut | +77.4% | 0.917 | 0.990 | 0.944 | 150 |
| **recursive, pure stop (ceiling)** | **+90.7%** | 0.972 | 0.997 | 0.979 | 216 |
| **recursive, RF detector stop** | **+89.0%** | 0.943 | 0.995 | 0.964 | 171 |

- **Pre/post metric (Phase 1)** reproduces cut_report's +78% exactly and reveals what the old
  pooled metric hid: pre-side (axon) accuracy << post-side (dendrite) -- axons are the hard merges.
- **Recursion (Phase 4)** lifts the oracle ceiling +77% -> +90.7% by peeling >2-cell merges
  (216 cuts over 150 objects); pre-side acc 0.917 -> 0.972.
- **Supervised RF atomicity stop (Phase 3)** reproduces the 0.88 detector and, as the recursion
  STOP test (no labels at the stop), reaches +89.0% -- within 1.7% of the label-purity ceiling.
- **Remaining autonomous gap = the CUT.** All the above use the ORACLE edge (labels pick which
  edge). The deployable cut is the seam GNN (net-0 autonomous / +14-26% human-assist); wiring it
  into the recursion + the bigger data (Track B) is what tests autonomous net-positive. Merge/join
  (false-split) direction via beam_search is the next increment.

## (a) learned cut into recursion + (b) join direction

**(b) Merge/JOIN of false-splits -- synapse adjacency is the wrong candidate generator.**
`join_corrector.py`: candidate joins = spatially-adjacent cross-v117-root synapse pairs. Result:
606,061 candidate root-pairs, **0.0% truly same cell** -- because a synapse IS a pre-cell
touching a post-cell (adjacent, different cells), synaptic partners drown the ~21 real
same-cell fragment pairs. The break in a false-split is in the CABLE, between synapses (median
cross-fragment synapse gap 6,870 nm; only 29% within 4 um). And the 7-box column has only ~21
false-splits with substantial fragments. So the join direction needs SKELETON continuity
between fragment tips (do the cables continue: collinear, caliber-matched), not synapse
proximity -- the relational model -- plus far more data. Honest negative for the synapse-based join.

**(a) learned seam cut wired into the recursion** (`recursive_corrector.py --cut learned`):
seam GNN per-edge benefit (keyed by undirected global edge to bridge the two tree rootings),
run autonomously (abstaining, tau=0.1) and as human-assist (top-5, oracle-verified), reported
in **connectivity** terms. Grouped-by-cell 4-fold CV, 150 merge objects / 147 cells:

| mode | splits | pre side acc | post side acc | pooled net (pair) | connectivity (both sides) |
|---|---|---|---|---|---|
| **autonomous (abstain)** | 621 | 0.880 | 0.979 | **-73.4%** | 0.894 |
| **human-assist (top-5)** | 166 | 0.882 | 0.980 | **+52.1%** | 0.898 |

The deployable-autonomous answer at column scale is **net-negative** (-73% of do-nothing on the
pair guardrail), even though pooled side accuracy reads a healthy 0.95 -- because do-nothing
already places most sides right, so the metric that matters is the pair guardrail, and the
autonomous learner **over-cuts** (621 splits): the damage lands hardest on the post side
(dendrite, -197%, 17,246 sides) where a wrong cut shatters the most pairs. This is the
data-starvation thesis made concrete -- the seam GNN can't yet pick the right edge unaided on
150 objects. **Human-assist is strongly net-positive: +52% pooled pair reduction with only 166
cuts (3.7x fewer), 0.898 connectivity** -- a proofreader picking from the model's top-5 is the
deployable mode today. Closing the autonomous gap is exactly what Track B (bigger data) tests;
the fetch is now checkpointed per-box so it survives proxy/CAVE outages and resumes.

## Track B payoff: data scaling moves the autonomous learned cut across zero

Track B grew the sample from the 7-box column to a 27-box block: **sidetable_big** = 1,808,250
synapse-sides, **915 false-merge roots** (vs 354), 15,107 cells; **513 v117 merge-object skeletons**
cached (vs 150; ~46% of merge roots have no skeleton, so the rest are un-skeletonizable).

Re-running (a) the learned seam-cut recursion on the bigger sample (lighter train: 3 folds, 12
epochs -- a lower bound vs the 4x35 column run), grouped-by-cell CV, connectivity metric:

| pooled pair-error vs do-nothing | 7-box (150 obj) | Big (513 obj) |
|---|---|---|
| **autonomous (abstain)** | **-73.4%** | **+4.6%** |
| human-assist (top-5)         | +52.1% | +42.0% |
| connectivity autonomous      | 0.894 | 0.892 |
| connectivity assist          | 0.898 | 0.914 |

**The autonomous learned cut crossed from net-negative to net-positive** (-73% -> +4.6%): the seam
GNN was harmful purely for lack of training objects; 3.4x the merges makes it deployable-autonomous
in aggregate -- the data-starvation thesis confirmed. The pre/post split shows where:
- **post-side (dendrite): -197% -> +5.6%** autonomous -- flips hard positive, cut cleanly unaided.
- **pre-side (axon): still -60.4%** autonomous -- axons remain over-cut; this is the residual gap
  (why connectivity is 0.892 autonomous vs 0.914 assisted). Axons need assist or yet more data.

**Seam-hash (error-site index) also sharpens with scale.** Same bigger sample (513 objects, 83,518
vertices, 3.9% seam): SimHash cosine-LSH retrieves seams from a seam query at **5.21x base rate /
6.05x over a non-seam location** (was 2.98x / 4.12x on the column). More data -> more separable
merge-seam signature -> a stronger content-addressable error prior.
