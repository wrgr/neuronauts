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
