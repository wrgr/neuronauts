# EXP-063 — Frankenmerge detection

## Result: passed — held-out AUC 0.958; polarity alone beats the published shape detector; CB2's uncorroborated tiers are intermediate, not noise

A frankenmerge is one v117 object spanning two cells. The PCFG report measured
that *detecting* one is comparatively easy (whole-object shape RF, AUC 0.875 /
precision 0.41 at the top 2%, on v117 roots at a 3.78% base rate) and that
*cutting* one is the hard part. This experiment re-asks the detection question
on the harness substrate with three things the report could not do: polarity as
a competitor (H8), the L2 object cloud as a competitor (from EXP-070), and
ConnectomeBench2's corroboration tiers as held-out positives.

## Setup

- **Substrate:** tier ≥10 only (20,826 atoms). This is the size control: on the
  full population a mixed atom has a median 818 L2 nodes and a trustworthy
  negative 28, so "is it big" would be near-perfect there. Tier ≥10 puts a floor
  under both classes and a size-only rung is run anyway. **Corrected after a
  peer review:** the strict-negative set this run actually uses — pure,
  proofread-owned, with CB2-touched atoms excluded — has a median of **382**
  level-2 nodes against 945 for the positives, not the 809 first quoted here.
  That 809 included the 363 CB2-touched atoms (median 2,826) that the run then
  removes. A size gap remains, so the bar must beat the stronger of two size
  rungs.
- **Positives:** atoms our own v1822 tally calls mixed-lineage (`labels.mixed`,
  n_roots ≥ 2): 2,149.
- **Strict negatives:** pure atoms with a proofread owner, **minus any atom a
  CB2 decision touches**: 934. That exclusion removed **363 of 1,297** — 28% of
  the atoms this repo's own overlay calls clean and proofread-owned were the
  operand of a recorded human split or merge. Leaving them in would have been
  label noise of exactly the kind this run measures.
- **Lenient negatives:** every pure atom not CB2-touched (val: 6,357), for a
  realistic 13% base rate.
- **Split:** spatial, axis 0, centred on the positives' median, 10 µm buffer.
  Train 922 pos / 259 strict neg; val 946 / 563. The negative side of train is
  thin, and the split is centred on positives, not on both classes.
- **Models:** the repo's own numpy GBDT (depth-1 stumps) and logistic
  regression, fit on train, scored on val. No new dependency.
- **Global-shape rung:** the ten features of `global_shape_merge.global_features`,
  ported to numpy and validated against scikit-learn on 400 real synapse clouds
  (7 columns exact; the three 2-means columns reach an equal-or-better optimum
  on 99.5% of clouds) — `tests/test_atom_features.py`.

## Held-out AUC, strict negatives

| Feature set | GBDT | Logistic |
|---|---:|---:|
| size only — log n synapses | 0.483 | 0.427 |
| size only — n level-2 nodes | **0.654** | 0.558 |
| **polarity** (pre/post fraction, 5 cols) | **0.914** | **0.914** |
| global shape — the PCFG report's detector | 0.862 | 0.875 |
| global shape without size/extent | 0.729 | 0.742 |
| topology (12 cols from `k10.npz`) | 0.911 | 0.918 |
| object geometry (9 cols from `objgeom_k10`) | 0.916 | 0.910 |
| polarity + global shape | 0.934 | 0.935 |
| **all** | **0.958** | 0.956 |

Three things to read off this:

1. **Size carries less than the stack, but it is not nothing — and the first
   version of this page said it was.** Log of the synapse count scores 0.483,
   below chance, and this document originally read that as "size carries
   nothing on this substrate." The peer review pointed out that log-synapse-
   count is the weakest available size proxy; the honest one is the object's
   level-2 node count, which scores **0.654** on its own. So detection is
   mostly shape rather than scale, but scale is a real signal and the bar now
   has to beat 0.654 rather than 0.483. The verdict is unaffected — the stack
   reaches 0.958 — and the global-shape rung's own strength still comes from
   `log_extent` and the cluster terms rather than `log_n`.
2. **Polarity alone, at 0.914, beats the published shape detector at 0.875.**
   H8 — "a mixed-polarity atom is a frankenmerge flag" — is supported, on held-out
   tissue, with five columns of free synapse counts. The consolidation plan had
   it as untested and Bar 3 at 0.000 in every real run; the two are not in
   conflict, because Bar 3 is a cut metric and this is detection.
3. **The families are complementary.** Polarity + shape reaches 0.935 and all
   four together 0.958; no single family is within 0.04 of the stack.

Under lenient negatives the best AUCs fall to 0.79–0.81. The lenient set holds
pure atoms with *unproofread* owners, some of which are surely frankenmerges the
overlay cannot see — that is the ordinary reason a lenient negative set scores
lower, and why the gate is on strict. Precision at the top 2% under lenient
negatives (13% base rate) is 0.39 (GBDT) / 0.53 (logistic), against the report's
0.41 at 3.78%; the base rates differ enough that this is a reference, not a
comparison.

## CB2's tiers, scored by the trained detector

The detector never saw tiers 1 and 0: they are outside the positive definition
(n_roots < 2), and CB2-touched atoms were excluded from the negatives. So their
scores are a label-validity test with no circularity. Tiers 3 and 2 are inside
the positive definition (37 and 654 of them in the training-eligible set), so
their rows are a consistency check only.

| CB2 tier | n (val) | AUC vs strict neg | mean score | above the positive median |
|---|---:|---:|---:|---:|
| 3 — existing 56 | 12 | 0.955 | +2.66 | — |
| 2 — new, mixed-strict | 298 | 0.969 | +2.60 | — |
| **1 — new, raw-mixed only** | **169** | **0.788** | **+0.09** | 14% |
| **0 — new, no v1822 signal** | **114** | **0.736** | **−0.22** | 12% |
| *(reference: own-mixed val mean +2.41; strict-neg val mean −2.41)* | | | | |

Tiers 1 and 0 sit almost exactly at the midpoint between the two class means.
They are not negatives (AUC 0.74–0.79 against strict negatives, well above 0.5)
and they are not positives (only 12–14% score above the positive median). The
honest reading is **mixed**: some fraction are frankenmerges these features see,
and the rest either are not, or are frankenmerges of a kind that mixed polarity
and object shape cannot detect — a cut inside one compartment, say, where both
halves are dendrite. This experiment cannot separate those two explanations,
and does not try.

What it does settle for EXP-062: **tier 2 is safe to spend, tier 1 is a
maybe, tier 0 is unverified** — which is the recommendation the sample-size
thread made on priors, now with a measurement behind it. The strictest cut
(tier ≥2, split-before) that gives 264 training positives under the
recommended split is the set this detector says looks like the rest.

## What is deliberately not here

**Bar 3.** The registry's first draft of this criterion carried "Bar 3 above
0.5". Bar 3 is frankenmerge *split* recall — the fraction of true frankenmerges
a method cuts — and needs a cut operator this experiment does not have. It moved
to EXP-062 before this run, with the reason recorded in the registry note.
Reporting 0.000 for it here, as every prior run did, would have said nothing
about detection.

**A 3-seed CI.** Both models are deterministic given the split, and the split is
deterministic. The variance that matters is across splits, and 259 training
negatives is too few to sub-sample honestly. The numbers above are one split;
the margin over the bar (0.958 vs 0.875) and over the stronger size rung (0.654) is large
enough that a plausible split-to-split swing does not threaten the verdict, and
small enough differences between feature sets (topology vs object geometry, say)
should not be read as rankings.

## For the program

Detection is not the bottleneck, on this substrate any more than on the PCFG
report's. The cheap signal (polarity) is strong, the object cloud adds to it,
and the stack is at 0.96 held out. What remains is the same thing the report
identified: locating the seam and cutting it — EXP-062, which now has a
positive set this detector agrees with.
