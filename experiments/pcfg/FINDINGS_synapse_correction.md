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

## Next

1. Hard-negative mining for the merge stratum (collinear different-cell pairs) → honest AUC.
2. PR@k / catch-vs-flag curves per stratum (berlin `prospective_flagging` style).
3. Turn the affinity into a repartition (threshold → connected components) and score it
   against v1718 by edge-F1 — the end-to-end correction, not just pairwise ranking.
