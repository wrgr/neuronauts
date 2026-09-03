# EXP-084 — Cajal's conservation laws hold in this tissue, and a mismatched branch breaks them

## Result: real branch points obey Murray's law (exponent median 3.18 against an ideal 3.0) across 3,781 bifurcations. Caliber mismatch separates from real at AUC 0.675, with zero parameters and no training.

The first tree-level signal in this program that works. Everything measured
before it was pairwise and local, and at a grower's frontier local features
reach 0% precision at the operating point (EXP-081).

## Why this is a different kind of evidence

Murray's law is material conservation at a bifurcation: the mother process's
cross-section is shared out among its daughters, so `r0^3 = r1^3 + r2^3`. It is
biophysics, not a fitted model — no training data, no parameters, and it applies
to any branch point in any cell. It also operates on a **structure**, not a pair:
you cannot evaluate it without a tree.

Nobody in this repository had checked whether it describes this tissue. A law
that does not hold cannot police an assembly.

## It holds

| | |
|---|---|
| bifurcations measured | **3,781**, across 60 cells |
| Murray exponent, solving `r0^p = r1^p + r2^p` | **median 3.18** (ideal 3.0) |
| interquartile range | 2.17 – 4.65 |
| bifurcation angle | median 99° |

The median sits within 6% of the theoretical value. The spread is wide, which
matters for how it can be used: this is a population law, informative across many
branch points, not a sharp test at any single one.

## A mismatched branch breaks it

Taking each branch point's mother radius with another branch point's daughters —
what a wrong join does when it grafts foreign cable onto an arbor:

| | real | mismatched |
|---|---:|---:|
| Murray exponent, median | 3.18 | **1.16** |
| median \|p − 3\| | 1.10 | 1.94 |
| **AUC by \|p − 3\|** | — | **0.675** |
| Cajal bifurcation-angle prior, AUC | — | 0.603 |

## Honest reading

0.675 is a real signal and a weak one, from a single branch point. Its value is
that it **compounds**: an assembled cell has many branch points, and a wrong
join creates one bad branch among many good ones, so the evidence accumulates
over a tree in a way a single pairwise score cannot. That is the property the
program needs and has not had.

The mismatch construction is a proxy for a wrong join, not a wrong join. It
mismatches caliber while holding the rest of the geometry fixed, so it isolates
the conservation term. Scoring real proposed joins is the next step and belongs
with the whole-cell shape work (EXP-083).

## Provenance

`neuronauts/morpho_grammar/cajal_conservation_priors.py` was deleted from the
working tree and survives on the remote. It is 90 lines of parameter-free priors
and carries **no label leakage** — unlike
`neuronauts/global_merge/represent/cloudvolume_em_sampler.py`, which takes
`is_true_continuation` as an argument and branches on that ground-truth label
while its docstring claims to sample voxel intensities. The module still emits a
deprecation warning pointing at the retired attic package; the warning is about
its untrained engines, and the conservation laws measured here are independent
of them.
