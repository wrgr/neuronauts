# One accurate neuron: the plan

*Opened 2026-09-07. Written against `docs/PROGRAM.md` (the current staged
program), `docs/threads/rerun_catalog.md`, and `results/`. The premise under
discussion: "we've done a ton, numbers haven't moved, and it's a failure of
putting the pieces together." This document says where that premise is right,
where it is not, and what to do in what order.*

---

## 1. The premise, tested

### 1.1 Where it is right, and exactly where

**Nothing in this repository has ever been assembled and scored as a neuron.**

Not "scored badly" — never scored. Every number in `results/` is a proxy for the
task: rank within a panel (EXP-077), area under the curve over tips (EXP-081),
recovery of an in-box component under a distance-only greedy walk (EXP-074),
pairwise separation at one branch point (EXP-084). There is no artifact anywhere
that is *a neuron we built*, and no call site that scores one.

The sharpest form of this: **the metric already exists and was never pointed at
a reconstruction.** `neuronauts/metrics/partition.py` computes cable-length-
weighted pair precision and recall and expected run length, from per-item
weights, with `weight = cable length in µm` documented as the harness default.
That is the proofreading-grade scorecard for a reconstructed cell, written,
tested, and never called on one.

`docs/PROGRAM.md` says `neuronauts/program/` holds the stage entry points.
**That directory does not exist.** Stage 1 (conservation over a whole tree) and
Stage 2 (joint assignment) are both marked READY; neither is written. So there
has been no integration failure, because there has been no integration attempt.
That is the premise, and it is correct.

### 1.2 Where it is not right

"Put the pieces together" has a ceiling that can be stated before doing it.

| piece | strength | what it is a function of |
|---|---:|---|
| objects within 2 µm (best frontier feature, EXP-081) | AUC 0.630 | local geometry |
| Cajal caliber conservation, one branch point (EXP-084) | AUC 0.675 | local geometry |
| whole-cell shape as a pairwise re-ranker (EXP-083) | 64.2% | local geometry, soma-referenced |
| alignment × proximity (EXP-081) | AUC 0.572 | local geometry |

These are not four independent votes. They are four readings of the same local
neighbourhood, and the repo already has the experiment that shows what happens
when you concatenate them: EXP-080 added synapse pattern to the combination and
**top-1 went 22 → 16**. Correlated AUC-0.6 features combined do not reach the
sub-2%-per-tip false-positive rate the frontier arithmetic demands. If
integration means feature concatenation, it will produce a number near 0.65 and
stop.

Two things in the program are *structurally* different rather than more of the
same, and both are unbuilt:

- **Joint assignment with a priced abstain (Stage 2).** "An object continues at
  most one cut end; a cut end takes at most one object" is a constraint no
  pairwise scorer can express. It is free — `scipy.optimize.linear_sum_assignment`,
  no training, no new features — and
  `attic/morpho_grammar/hungarian_bipartite_assembler.py` is already
  scorer-agnostic. It changes precision at a low base rate by construction.
- **Compounding over a tree (Stage 1).** 0.675 at one branch point is weak.
  Summed over the dozens of branch points a whole assembly creates, it is the
  only route in this program by which a weak signal legitimately becomes a
  strong one. Untested.

So: integrate — but integrate these two, as a joint objective, not the pile of
correlated local scores.

### 1.3 A third thing, which neither of us has been saying

**The constraint the whole program is now designed around was measured on the
substrate this repository already declared defective.**

`docs/threads/rerun_catalog.md` item 1: `object_clouds_mip5.npz` holds one point
per supervoxel *visible at mip 5* — roughly 20% of an object — and each point is
a supervoxel centroid, not a surface voxel. "Harmless at micron scale; fatal for
contact." Item 5 lists EXP-074 as "already known wrong" for exactly this reason.

`scripts/measure_frontier_load.py`, which produced EXP-081 — the 1.6% base rate,
the 46-tips-per-cell frontier, and by extension the 0.007 precision translation
that `PROGRAM.md` puts at the top of the page — loads
`data/substrate/c100um/object_clouds_mip5.npz` and nothing else. It was never
re-run on the corrected substrate.

Two biases in it, both pushing the same direction:

1. **Dead ends are overcounted.** A tip is a point with no cable beyond it. On a
   cloud carrying ~20% of an object's supervoxel centroids, a thin neurite is
   sparse, and sparse reads as ended. The tip finder additionally subsamples to
   4,000 points and requires ≥3 neighbours within 3 µm.
2. **Live sites are undercounted.** `live` is defined as a fragment of the
   *seeded target* within 5 µm, where the target is the set of **labelled**
   proofread fragments. EXP-071 established that the connective neurite between
   two labelled fragments — 2,147 objects, median 3 hops — is **absent from the
   population entirely**. A tip whose true continuation is one of those objects
   scores as a dead end.

Both inflate the denominator and shrink the numerator. The base rate that
everything downstream is conditioned on is therefore a lower bound of unknown
looseness. It may survive the correction. Nobody has checked, and it is cheap to
check.

### 1.4 Why the ton of work did not accumulate

| | ledger rows in `results/RESULTS.md` |
|---|---:|
| EXP-057 … EXP-074 (14 experiments) | present, reproducible through `neuronauts.experiments._runner` |
| EXP-075, 076, 077, 080, 081, 082, 083, 084, 085 | **0** |

`results/RESULTS.md` states its own rule: *"A run without a row does not
exist."* Every experiment behind `PROGRAM.md`'s "What is dead, with evidence"
table is in the second group. Nineteen scripts under `scripts/` hard-code
`/Users/wgray13/projects/neuronauts/`, including `measure_frontier_load.py`,
`measure_frontier_discrimination.py`, `audit_frontier_precision.py` and
`test_cajal_conservation.py` — the load-bearing negative *and* the one
positive.

So the repo's most confident claims are its least reproducible ones, and the
rules in `PROGRAM.md` §"Rules for every stage" were written after this happened
and are enforced by nothing. That is the mechanism by which a lot of work
produces no movement: it was done in a mode that cannot compound. It is the same
failure as §1.2, one level up — the pieces do not connect to each other.

---

## 2. What "an accurate neuron" has to mean

This needs deciding before anything is built, because it changes the shape of
the work by an order of magnitude.

The 100 µm harness cube cannot hold a whole pyramidal cell.
`docs/threads/soma_seeded_targets.md` records that the gold cell's axon leaves
through +y, runs 90–223 µm outside, and re-enters; that cell needs a **1,285 µm**
cube. Three candidate targets:

| | target | scored against | cost |
|---|---|---|---|
| **A** | the seeded cell's **in-box connected component** — every object in the cube that its v1822 root claims | v1822 root, cable-weighted completeness / purity / expected run length | achievable on the current substrate |
| **B** | the **whole cell** across the volume, including cross-face matching | same | a different data program: fetch scope, out-of-box enumeration, cell-scale matching |
| **C** | one cell, at whatever scope it reaches, presented with a proofreader-grade scorecard and a rendering | both references side by side | A plus a day of presentation |

**Recommendation: A as the deliverable, C as how it is shown.** B is a real
target and should be named as the thing after, not folded into this. Cross-face
matching is a grammar at the scale of cells and the thread already says so.

---

## 3. The plan

Ordered by what invalidates what. Each phase has a bar and produces a ledger
row. No phase begins before the one above it has one.

### Phase 0 — Make one number exist  *(smallest, do it first)*

The scorecard for a single seeded reconstruction, and the two references it will
always be read against.

- Define the artifact: `NeuronReconstruction { seed_nucleus_id, claimed_object_ids,
  per_join_provenance }`. One dataclass in `schemas.py`, `.npz` I/O like the rest.
- Write `neuronauts/metrics/reconstruction.py` — a **thin wrapper**, not new
  metric mathematics. It takes a reconstruction and a v1822 target, weights every
  object by cable length, and calls the existing
  `partition_metrics` to return cable-weighted precision, recall, and expected
  run length.
- Compute it, over all 103 evaluable seeds, for the two references:
  - **v117 as-is** — claim nothing, join nothing. This is the do-nothing
    baseline, the number any method must beat, and **it does not exist anywhere
    in this repository.**
  - **v1822** — the human answer. The ceiling.

**Bar.** Three numbers, for 103 seeds, for both references, reproducible through
the runner, in the ledger. **What it buys:** the size of the gap that is
actually available. If v117-as-is already recovers most of the in-box cable, the
program's headroom is small and we should know that this week rather than after
Stage 2.

### Phase 1 — Re-measure the frontier on the corrected substrate

Re-run EXP-081 as a registered experiment, changing exactly two things and
nothing else:

1. Read objects with `agglomerate=True, timestamp=V117_TS`, mip 2 — not
   `object_clouds_mip5.npz`.
2. Define `live` against **every object the v1822 root claims**, including the
   connective objects EXP-071 found missing from the population — not only
   labelled target fragments.

**Bar.** The corrected base rate with an interval, and an explicit statement of
whether 1.6% survives. Report the two corrections separately so we know which
one moved it.

**Why here and not later:** every scorer built in Phase 2 will be tuned to an
operating point. Tuning to a possibly-wrong one wastes the phase. This is one
script and a re-read; it is the cheapest thing in the plan that can change the
most.

### Phase 2 — The grower: joint assignment with a priced abstain

Create `neuronauts/program/` and write `stage2_assignment.py` for real.

- Frontier = every cut end of the claimed cable. Candidates per tip from the
  widened enumeration.
- Cost matrix from the geometry that **reproduces**: EXP-077's top-1/5/20 counts
  reproduce exactly on independent recomputation; its distance row does not
  (published 2 of 66, recomputed 0 of 66). Use the former. Do not use the
  latter.
- One dummy column per tip, priced — declining is inside the optimisation, not a
  threshold bolted on afterwards. The functional form comes from the attic's
  calibrated stop rule; its numbers do not.
- Solve with `linear_sum_assignment`. Iterate: claim, re-cut the frontier,
  re-solve.

**Bar.** Phase 0's three numbers, at Phase 1's base rate, beating (a) v117-as-is
and (b) EXP-074's distance-only greedy. `PROGRAM.md`'s stated bar — precision
above 20% at recall 0.3 against greedy's 0% — stands as the secondary read.

### Phase 3 — Compounding conservation as the objective

`stage1_conservation.py`. Sum Cajal evidence (`r0³ = r1³ + r2³`, EXP-084's
measured exponent 3.18 over 3,781 real bifurcations) over **every branch point
the assembly creates**, and fold in EXP-083's placement score — which the
program already argues belongs inside Stage 1 rather than beside it, because the
control shows it reads *"this does not belong here,"* soma-referenced, not
*"this is a chimera"*.

Critically: this becomes the **objective the assignment in Phase 2 optimises**,
not a filter run after it. That is the actual integration the premise is asking
for, and the difference between a joint objective and a feature concatenation is
the difference between §1.2's two structural moves and §1.2's ceiling.

**Bar.** `PROGRAM.md`'s: AUC above 0.85 over whole assemblies, plus a stated
minimum detectable wrong-join size. Then Phase 0's numbers again, against Phase
2's.

### Phase 4 — One neuron, shown

Best-scoring seed. Rendering, and the scorecard: our reconstruction, v117-as-is,
and v1822, three columns, cable-weighted completeness / purity / expected run
length. Cell type from `cell_type_final` with its `cell_type_source`, so a
prediction is never presented as a curated fact.

---

## 4. Cross-cutting, and not optional

Given §1.4, these are conditions on the work, not a phase of it:

1. **Every phase goes through `neuronauts.experiments._runner` and gets a ledger
   row.** No result outside the registry counts, including a negative one.
2. **No absolute paths.** Nineteen scripts carry
   `/Users/wgray13/projects/neuronauts/` and none of them runs anywhere else.
   New work uses repo-relative paths; the load-bearing existing ones
   (`measure_frontier_load.py`, `test_cajal_conservation.py`) get fixed as they
   are re-run.
3. **`PROGRAM.md` marks EXP-075–085 as unregistered** until they are backfilled,
   so its "what is dead" table carries the weight its provenance supports rather
   than the weight its prose implies.
4. **Print the audit before the result** — substrate, identity resolution,
   effective sample size, and the confound this comparison is vulnerable to.
   Already the rule; now enforced by the runner rather than by intention.

---

## 5. Open decisions

1. **Scope**: A, B or C from §2. Recommendation A + C.
2. **Where the work runs.** This session's container has no CAVE token and no
   `data/substrate/` or `data/external/` — both are gitignored and the clone is
   fresh. Nothing in §3 can execute here without a token and a fetch budget for
   the 100 µm cube substrate. Either provision that, or the phases run on the
   machine that already holds the substrate.
3. **Whether Phase 1 gates Phase 2 strictly.** It is ordered that way above. The
   argument for relaxing it is that Stage 2's constraint helps at any base rate;
   the argument against is that the abstain price is exactly the parameter the
   base rate sets.
