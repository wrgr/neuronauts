# The "language of proofreading" — what `wrgr/berlin` actually shows

> Read directly from the source repo (github.com/wrgr/berlin, "Calibrate the
> Humans" talk + Nature Communications draft), not summarized from memory.
> This is prior work on MICrONS/minnie65 proofreader behavior, not morphology,
> and it answers a different question than our PCFG proposal. Recorded here so
> it is not conflated with either.

## What it is

A study of **proofreader behavior telemetry** on minnie65 (NeuVue UI action
logs + CAVE outcome validation, 8 expert + 36 novice annotators, Nov 2021–Aug
2022). The central question is operational: given finite, expensive human
judgment, which decisions need an expert and which can go to a student or a
model, and can you tell before spending the effort? Not: what did an edit
change morphologically.

## The result that is genuinely a "grammar" — and what it is a grammar of

A **first-order Markov model over the UI action-token stream**
(navigate/segment/annotate/other, one token per logged event) recovers
**expert vs proto-expert at LOO AUC 0.95**, matching a 28-feature hand-built
behavioral bank. Quoted directly: *"the 'language of proofreading' is
learnable, not merely a metaphor."*

This is a grammar of **how someone works**, not of **what they decided** or
**what the tissue looks like**. It is a different object from both of the two
this project has been discussing:

| | terminals | learns | corpus |
|---|---|---|---|
| our PCFG proposal | skeleton segments, tips | is this object well-formed; where to cut | gold cells (harness) |
| Berlin's action grammar | UI events (navigate/segment/annotate) | who is an expert, from behavioral style | 44 annotators' session logs |
| ConnectomeBench2 | merge/split decisions | what a proofreader decided | ~400k expert decisions |

The "two grammars" framing added to `docs/consolidation_plan.md` §6.3a
conflated the second and third of these. Corrected: there are properly
**three** things called grammar in this project's orbit, over three different
alphabets, and none of them substitutes for another.

## The finding that generalizes past Berlin: expressive models are data-starved at this scale, three times over now

A **richer latent grammar (HMM) over the same action tokens collapses to AUC
0.39–0.59** at n=15 annotators — explicitly read by the source as data-starved,
not falsified, and flagged as future work once telemetry scales.

This is the same shape as two results already in this repo:

- our seam-locating GNN: net-negative at 150 training objects, first clears
  zero at 513 (`docs/tree_assembly_handoff.md`);
- ConnectomeBench2's own motivation for existing: the prior benchmark had
  "hundreds of samples," theirs has 716k, "large enough to both train and
  evaluate on."

Three independent projects, three different alphabets (skeleton edges, UI
actions, segmentation decisions), the same failure mode: a low-order model
wins and a richer one collapses until the corpus is an order of magnitude
larger. That is now a pattern, not a coincidence, and it is the strongest
argument on record for starting every new corpus with the cheapest model that
could possibly work.

## The morphology test that exists but is inconclusive, not negative

Berlin also tested whether **cell morphology (caliber, branches, size)
predicts per-cell proofreading error** in a 28-cell benchmark. Result: no
predictive relationship (all p > 0.09). But the source itself flags the test
as compromised: **17 of the 28 cells carried stale 2021–22 root ids** at
analysis time, and the ground-truth-free risk signal that *did* work (AUC 0.76)
was traced to annotation-category difficulty, not cell morphology. Quoted
directly: *"CAVE morphological confirmation is inconclusive... A clean test
needs more cells + a morphology-sensitive task."*

This is not evidence against a morphology-difficulty relationship. It is a
stale-root confound of exactly the kind `neuronauts/data/lineage.py` and
`neuronauts/harness/labels.py` already resolve by timestamp (the same problem
this project solved for its own v117→v1822 overlay). **A clean re-run of that
specific test, on our substrate, with correct timestamped lineage, is cheap
and currently unclaimed** — see "possible follow-on" below.

## Correction applied

`docs/consolidation_plan.md` §6.3a is being rewritten from "two grammars" to
name Berlin's action grammar as the third object and cite the data-starvation
convergence above; EXP-057D (learn a first-order model of the edit language)
is retargeted to be explicit that its alphabet is ConnectomeBench2's decision
outcomes, not UI actions, so it is not accidentally read as a re-run of Berlin.

## Possible follow-on (not scheduled, flagged for a decision)

Re-run Berlin's `cave_morphology.py` test — does cell/atom morphological
complexity predict correction difficulty — using this project's real,
timestamp-correct v117/v1822 lineage resolution instead of the stale 2021–22
roots that confounded the original. This would need: (a) a difficulty/error
signal per atom or per correction, which this project does not yet have
outside the seam-locating experiments, and (b) agreement from whoever owns the
Berlin dataset access before pulling its telemetry into this repo. Not
proposed as a numbered EXP until both exist.
