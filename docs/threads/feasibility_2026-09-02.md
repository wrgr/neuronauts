# Feasibility assessment — 2026-09-02

*What the evidence in this repo, as it stands after the QA pass of the same
date, says about whether the experiment program (`docs/consolidation_plan.md`
§6: substrate → baselines → propose → cut → score → assemble) and the north
star it serves (`docs/roadmap_global_assembly.md`: learned global assembly by
stitching skeleton fragments across space) can succeed — and what is actually
in the way. Every number below is from a registered experiment's `result.json`
or a QA reproduction; nothing is projected.*

## 1. Where the program stands

17 registered experiments: **6 passed** (057B, 058, 059, 060B, 063, 070),
**3 failed** their bar (057, 060, 061), **2 unwritten** (057C, 062), **6 blocked**
(064–069) — all six behind EXP-060's failing bar.

| Capability | What is established | Where |
|---|---|---|
| Substrate | 279,075 atoms partition every one of the cube's 25,280,966 L2 nodes; 100% id resolution; endpoints a verified subset of nodes | `build_object_geometry.py` gates; EXP-070 |
| Labels | 4,802 pure proofread-owned atoms; 2,444 mixed-lineage; 56 mixed *and* proofread; +2,067 atoms of external seam evidence with zero-disagreement provenance | EXP-057, EXP-057B |
| **Frankenmerge detection** | **Held-out AUC 0.958** on tier ≥10; polarity alone 0.914, beating the published shape detector (0.875); size at chance | EXP-063 |
| Proposal by geometry | Reachability ceiling at 5 µm, object metric: **75.7%** of spanning links (tier ≥10), **56.8%** (all tiers). At a usable panel (≤20 objects) recall is **12%**; 65% needs a median panel of 3,870. The tangent cone adds **2–3×** over chance and reaches no more than 40% at any panel under 42,000 | EXP-060, 060B, 061, 070 |
| Ranking by geometry | Collapses: every distance threshold that reaches the true pairs merges everything; pair precision 0.0006 | EXP-058 |
| Label density | 16.2% of synapse mass on proofread single-lineage atoms (bar was 30%); strictest seam-positive cut yields **264 training atoms** after re-centring the split, vs 513 at which this repo's seam GNN first cleared zero | EXP-057; `seam_positive_sample_size.md` |
| Seam location / cut | **Untested.** EXP-062 unwritten; Bar 3 (split recall) 0.000 in every historical run | — |
| Embedding retrieval (SegCLR) | One 34-atom crude pilot, AUC 0.445, with 6 of 20 downloaded shards truncated. Inconclusive; never a registered run | `data/external/segclr/auc_result.json`; EXP-057C todo |
| Cross-atom adjacency from the segmentation | **Not on disk.** 0 of 7.3M shard edges leave their atom; the chunkedgraph's L2 graph is within-root by construction, so "who touches whom" needs a contact-site or mesh fetch, not `lvl2_graph` | this session's check |

## 2. The structural block, stated plainly

EXP-064 (scorer bake-off) and everything after it require EXP-060 to have
*passed*. EXP-060's bar is 90% recall at a median panel of 20. The measured
reachability ceiling for **any** geometric proposer on this substrate — object
metric, uncapped, tier ≥10 — is 75.7%. The consolidation plan's own protocol
(§6.2) says a proposal bar must sit below the measured reachability; EXP-060's
does not, and no amount of re-running changes that.

So the D/E/F series are blocked behind a bar that is known to be unmeetable,
which is a program-design fault, not a scientific result. Two things follow,
and both are decisions rather than experiments:

1. **Re-declare EXP-060's bar** per §6.2, *before* any re-run, at a target below
   the ceiling and in object units — e.g. "≥60% of spanning links at a median
   panel ≤50, on the object metric", which EXP-060B's curve says is roughly
   where the trade-off sits.
2. **Rewire EXP-064** to `requires_ran=[EXP-060]` and `requires=[EXP-060B,
   EXP-070]`, so scoring is measured on the best available panel with its
   recall ceiling stated in the result, rather than never.

Neither was done in this pass: changing a bar is the user's call.

## 3. Feasibility, honestly

**As assisted proofreading — flag frankenmerges, rank continuation candidates
for a human — feasible now, on this evidence.** Detection is at 0.96 held out
with free features. A proposer that hands a human 50–100 candidates reaches
16–23% of spanning links at tier ≥10 (EXP-060B), and the two per-atom filters
that EXP-063 validated as strong signals (polarity, object geometry) have not
yet been tried as *pair* filters to shrink those panels.

**As autonomous global assembly — the roadmap's north star — not feasible on
current evidence, and the blocker is candidate generation, not scoring, not
assembly, and not labels.** The roadmap's seam-stitch gate (§3.2: "endpoints
within ε **and** tangents align") has now been measured directly on real
fragments, with the geometry computed correctly: fewer than 60% of true spanning
links on the full population are inside any tractable ε, and the tangent adds
2–3× over chance — informative, not decisive. Scoring (series D) and assembly
(series E) can only choose among what the proposer offers; they cannot recover
recall it never had. Building them now would measure the proposer's ceiling
again, expensively.

Three things would change this assessment, in order of leverage:

1. **Adjacency from the segmentation itself.** The corridor probe in EXP-070's
   evaluation found that long spanning links have essentially no same-owner
   material between them — the far partner is not reachable by walking the
   neurite, it is a genuinely separate fragment. That argues the right proposer
   target is *touching* atoms, chained transitively by assembly, not partners at
   40 µm. Whether touching-pair recall is high is **the single most important
   unmeasured quantity in the program**, and it needs a contact-site fetch (a
   bounded CAVE/mesh job for the 4,802 labelled atoms), not more geometry on
   what is already on disk.
2. **An embedding channel with no radius.** The intake work established SegCLR
   is fragment-native and trained on v117; the pilot was too small and partly
   corrupt to say anything. A registered EXP-057C — hundreds of atoms, complete
   shards, the crosswalk checked — with the intake doc's decision rule (~0.7
   AUC or deprioritise) would settle it in a day.
3. **A cut operator.** Detection is ahead of cutting, and the PCFG report's
   finding that the second cell is a median 11% of a frankenmerge means even
   perfect detection yields little without a locator. EXP-062 on the 264-atom
   strict set is a shallow-model experiment, not a GNN one, and should be
   written that way.

On sample size: 264 strict training positives (508 at tier ≥1, 881 with tier 0)
supports shallow models. EXP-063 measured tier 0 as a mixed bag (AUC 0.74
against negatives — not noise, not positives), so spending it should be a
deliberate, reported choice.

## 4. Recommended order

1. Decide the EXP-060 bar and the EXP-064 wiring (§2). Program decision.
2. **EXP-071 — contact adjacency.** Fetch cross-root contacts for the labelled
   atoms; measure what fraction of object-MST spanning links are direct
   contacts, and the touching-pair panel size. If most are, the proposer
   problem changes shape entirely.
3. EXP-062 on the tier ≥2 split-before set, shallow cut model, Bar 3 reported
   honestly and located seams re-checked against CB2's `edit_point_nm` first.
4. EXP-057C as a real run, or retire it.
5. Series D/E on whatever panel 2 and 4 produce — not before.

## 5. Corrections to the record made in this pass

Two claims were overstated and are fixed with re-runs: EXP-061's enrichment
over chance (was 3–6×, is 2–3× — the null was single-direction for a
best-of-two statistic) and EXP-070's answer-key effect (463 → 325 re-routed).
One was understated: EXP-070's object-metric gain on the full population (+9.1
→ +13.5 points; the endpoint column had dropped unreachable pairs from its
denominator). Details and the reproductions: `docs/threads/qa_pass_2026-09-02.md`.
