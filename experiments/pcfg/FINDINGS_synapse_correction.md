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

## Next

1. **Self-supervised continuation model on the merge problem.** Train coherent-vs-spliced
   (path-encoder objective) on raw arbors; evaluate on the de-split (merge) stratum where the
   gap heuristic is useless. This is where learning should pay — point the expressive model
   here, not at de-merge.
2. Clean-column vs un-proofread-**bulk** drift (needs a bulk fetch) → confirm the 1.2%
   regime holds outside the column, where the corrector must actually run.
3. PR@k / catch-vs-flag curves per stratum; then repartition (affinity → components) scored
   against v1718 by edge-F1 — the end-to-end correction.
