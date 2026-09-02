# Semi-synthetic benchmarks — not evidence about real segmentation

**Every script in this directory scores a method against artificially damaged
skeletons, not against v117 segmentation.** 32 of the 34 import the synthetic
world builder directly. A "split" here is an intact skeleton cut in software, so
the two halves still carry matching geometry, matching caliber and a matching
tangent at the cut. Real v117 fragments do not.

That difference is the whole difficulty of the program. Numbers from this
directory are therefore **not comparable** to anything in `results/`, and the
headline figures they produced — including the "~85–87% pairwise merge accuracy"
quoted for a long time in `docs/threads/grammar.md` — did not survive contact
with real data:

| question | on synthetic damage | on real v117 |
|---|---|---|
| pairwise merge proposal | ~85–87% | ~0.09% precision (EXP-060, 060B, 061, 070) |
| candidate generation after widening | — | collapses (EXP-072) |

What *has* held up on real data lives in `results/`: EXP-063 detects a
frankenmerge at held-out AUC 0.958, EXP-071 explains why the synapse-anchored
population omits connective cable, and EXP-075 shows local geometry cannot
supply a stop rule.

Keep these scripts for their mechanics — the damage model, the metric
implementations and the harness plumbing are all reusable. Do not quote their
scores. See `docs/threads/experiment_survey.md` for the evidence grade of every
experiment in the repo.
