# Handoff: tree assembly / tiered identity resolution

> Written 2026-08-21 at end of session, branch
> `claude/tree-assembly-algorithm-wbtae0`. Read this before re-engaging —
> it's the fast path back to full context. Companion doc:
> `docs/tree_assembly_algorithm.md` (the full design + every run's numbers,
> in chronological/argumentative order — this file is the distilled map).

## The one-paragraph state of the world

We do **not** have a working global assembly algorithm. We have two solid,
separately-useful results and one clean, well-characterized failure that
tells us exactly what to build next. (1) Tile-based stitching of dendritic
fragments works and scales correctly once tiles are sized right. (2) A
**verified scaffold** — the dendritic/soma side of the graph — can be
certified to ~100% precision today with one exact lineage lookup, no
learning. (3) Growing that scaffold outward into the axon tail by
*proximity* is falsified, cleanly, with real numbers: axons don't stay
near their soma, and nearest-fragment distance carries zero identity
signal. The fix is not "try harder at proximity" — it's swap the channel
for directed continuation + EM texture, which a separate, older experiment
in this repo (cut-face fingerprints) already proves works on a structurally
similar problem, just not yet adapted to this one.

## What's confirmed (keep, build on)

| # | Result | Evidence |
|---|---|---|
| 1 | Two-level tile stitching improves cross-tile assembly at near-zero precision cost via the exact shared-observation channel. | 200 µm box, 3 independent runs: ΔARI +0.10 to +0.11, Δmerge_P ≈ 0, assembly 0%→22-24%. |
| 2 | Assembly scales by adding right-sized tiles, not enlarging them. | Same 600 µm volume: 2×2 (~300 µm tiles) → ΔARI +0.003 (dead). 6×2 (~100 µm tiles, the proven size) → ΔARI +0.07 to +0.12. 25-40× improvement from tiling alone. |
| 3 | **The dendritic scaffold is real and cheap.** One exact lineage test (nucleus supervoxel → v117 root, `roots_at`, one batched call) certifies most of the post/dendrite side. | 200 µm dual-side census: post-side NAMED (exactly 1 contained soma) = 59.6% of ALL synapses at 99.8% mass purity. On the soma-seeded L2 substrate specifically: 79-99.8% of dendritic mass at ~100% purity, 94-100% frankenmerge exclusion. |
| 4 | Multi-nucleus roots are not automatically catastrophic. Glia carry no synapses, so most MULTI-tier mass is a benign rider (glia/duplicate nucleus detection), not a true merge error — separable by the v1718 oracle today, and by a served cell-type table (`nucleus_ref_neuron_svm`) without training. | 200 µm: 50-67% of MULTI-tier roots resolve to a single v1718 target (benign); catastrophic caseload is bounded and enumerable (142 roots, ~2.2k syn, per 200 µm box). |
| 5 | Combining geometry + EM texture, with a calibrated abstention threshold, reaches near-perfect precision on a real "confirm this reconnection" task. Pure texture alone fails; pure geometry alone is decent; the trained combination beats both and abstention buys the rest. | Cut-face fingerprints (`experiments/fingerprints/`), real v117 split sites: pure texture top-1 = 0.0 (early attempt). Geometry alone = 0.67-0.64. Combiner = 0.767, and its abstention curve reaches **precision 1.0 at 11% coverage** (8/73 held-out sites). Panel recall (true partner present in the candidate set) = 1.0. |

## What's falsified (don't retry as-is)

| # | Claim tested | Result | Why |
|---|---|---|---|
| 1 | Star attribution: attach an orphan axon fragment to its nearest NAMED (single-soma) anchor. | **Structurally wrong.** Even at near-full synapse density, only 9 of 1,063 proofreader-adjudicated axon fragments have their true neuron among in-box soma anchors. | Axons travel millimeters; a local box essentially never contains the soma an axon fragment belongs to. |
| 2 | Link attribution: connect an orphan axon fragment to its nearest *other* axon fragment by synapse-cloud proximity, gated by a decoy margin + morphology cosine. | **Zero measured precision at every threshold swept** (0/32 doubly-adjudicated links; the whole G/M/C grid). | Nearest-neighbour gaps are sub-micron everywhere and decoy margins collapse to ~1 — dozens of unrelated processes pack within a micron of any given fragment. Proximity alone carries no identity information in this regime. |
| 3 | Blanket skeleton-oddness atomization ("shatter every long-bridge fragment") as a general real-data preprocessing step. | Collapses level-0 partition quality (ARI 0.22→0.06) because it deletes the same-parent lineage prior almost everywhere (85% of real fragments flagged odd at the naive threshold). | The oddness heuristic needs per-substrate recalibration (point-cloud vs skeleton MST) and should demote the parent link to a soft edge, never delete it outright. |
| 4 | 2×2 tiling on a 600 µm box at the same epoch budget as 200 µm runs. | Level-0 partition quality collapses (merge_P 0.75→0.26) at 3.4× the neuron density. | Confirms result #2 above — evidence-quality-first framing, not more assembly machinery. |

## The synthesis: what to build next

**Certify the scaffold, then extend it by directed continuation + EM texture
verification — never by proximity.** Concretely, in priority order:

1. **Adapt the cut-face combiner to the link-certification problem.**
   `experiments/fingerprints/cutface/` already has the pieces:
   `candidates.py` (proximity-ball + **direction cone** panel — the missing
   ingredient in this session's link test), `patches.py` (masked cut-face
   patch extraction), `features.py` (contrastive encoder + combiner),
   `evaluate.py` (panel recall / top-1 / abstention curve — reusable
   as-is). The adaptation: candidates aren't "the two faces of one known
   segmentation break" anymore, they're "does this ANON fragment's proximal
   endpoint continue into a NAMED scaffold's distal endpoint" — same shape
   (endpoint pair, direction cone, EM patch), different source of
   candidates (skeleton endpoints from `treestitch.stitch.candidate_stitch_edges`
   instead of segmentation-break sites). Reuse `evaluate.py`'s three
   metrics unchanged; that's the honest yardstick.
2. **Wire the exact-lineage soma-containment scaffold as the anchor set**,
   not the fragile oddness/proximity gates. `count_contained_somata` +
   the NAMED tier from `scripts/tier_census.py` are already the right
   primitive — use them directly to seed the certified set before any
   learned step runs.
3. **Re-run `scripts/verify_attribution.py --links` with the cutface-style
   directed-continuation + patch-texture score in place of the current
   `(gap, margin, cos)` battery.** Same script structure, same oracle
   grading (gradeable = both ends proofreader-adjudicated), same sweep —
   just a better candidate generator and a real verifier channel. This is
   the single next experiment; everything else is scaffolding for it.
4. **Once a link channel clears a non-trivial coverage at zero measured
   false-accept**, chain accepted links transitively from NAMED anchors —
   identity propagates along certified chains (§5 of the design doc), and
   the catastrophic-editor pricing (never touch a NAMED claim; price a
   scaffold split/merge near-infinite) governs contradictions.
5. **Recalibrate oddness for atomization** (result #3 above) before
   reusing it anywhere: MST-based for point-cloud substrates (already
   fixed in `treestitch/atomize.py::mst_odd`), stricter threshold, and
   demote to a soft edge feature rather than a hard cut.

## Key files (all on `claude/tree-assembly-algorithm-wbtae0`, all pushed)

| File | What it is |
|---|---|
| `docs/tree_assembly_algorithm.md` | Full design doc + every run's numbers, chronological. Read for depth. |
| `docs/tree_assembly_handoff.md` | This file. |
| `treestitch/stitch.py` | Level-1 seam stitching (super-fragments, exact channels, constrained Kruskal). Confirmed result #1-2. |
| `treestitch/atomize.py` | Level −1 atomization + oddness detection (`mst_odd`, `oddness_scores`). Falsified result #3 lives here; the fix (soft edges) does not yet. |
| `treestitch/realworld.py::count_contained_somata` | The exact lineage scaffold gate. Confirmed result #3. |
| `scripts/tier_census.py` | Day-one tier matrix + MULTI benign/catastrophic split. Confirmed results #3-4. |
| `scripts/verify_attribution.py` | Star + link attribution verifier (`--links` flag). Falsified results #1-2 come from here; **next experiment extends this file**. |
| `scripts/scaffold_census.py` | Earlier (superseded by tier_census.py) scaffold purity measurement; kept for the census-v1→v3 gate-calibration history. |
| `treestitch/stitch_viz.py` | Neuroglancer viz of stitch products (super-fragments, stitch decisions, odd fragments, frankenmerges). |
| `experiments/fingerprints/cutface/` | The cut-face verifier — confirmed result #5, and the source of the adaptation in next-step #1. Read `README.md` first. |
| Artifact: results dashboard | https://claude.ai/code/artifact/8cc0dd5b-8a92-4c74-be82-c951a903184e — charts for confirmed results #1-2 and the falsified verification curve. Not yet updated with the scaffold/cutface numbers from this final exchange. |

## Honest gaps / caveats to carry forward

- All numbers are from a **single 200 µm box** (occasionally 600 µm)
  around `x∈[950k,1150k] y∈[930k,1000k] z∈[780k,880k]` nm. No
  cross-region generalization test has been run on the tiered-identity
  frame yet (the *old* pipeline's cross-region results, Phase 2.8-2.12,
  don't transfer to this reframing — different unit of certification).
- The scaffold purity numbers (79-99.8%) came from a **soma-seeded L2
  substrate** — biased toward dendrites by construction. The **dual
  synapse-level census** (unbiased) is the honest number: 59.6%
  half-edge, 7.2% full-edge, day one. Don't conflate the two when citing.
- Ground truth for the axon tail is sparse: 93% of ANON fragments were
  never proofreader-adjudicated (self-labeled at v1718). Every precision
  number on the axon side is measured on the adjudicated ~7% and
  transferred — flag this whenever citing axon-side numbers to anyone.
- The cut-face combiner's 0.767/1.0-at-11% result is on **known
  segmentation-break sites** with a tight proximity+cone panel, n=73 held
  out. It has not been run on the open-field axon-attribution geometry.
  Next-step #1 is exactly closing that gap, not assuming it's already closed.
