# docs/archive/2026-09 — the retired "direction" docs

Moved here 2026-09-01 per [`docs/consolidation_plan.md`](../../consolidation_plan.md)
§4.4. Nothing is deleted: git history is preserved (everything arrived via
`git mv`), and each file carries a header recording when it was archived and
what replaced it. Content is otherwise unchanged.

## Why these moved

The consolidation survey found **three architectural regimes still coexisting**
in the working tree (`docs/consolidation_plan.md` §1.2): the v1 agent/membrane
simulation (quarantined, dead), the v2 shared grammar + GAT, and the box-local
CellGNN (architectural F1 ceiling of ~0.27, per `docs/architecture.md`) —
plus the treestitch global partition and harness substrate that actually hold
up on real data. Five documents in the repo root and `docs/` each described a
different one of these as "the" pipeline and disagreed with each other about
it. That disagreement is a named cause of the confusion the consolidation plan
is fixing (§1.1: "Five 'direction' docs, three of which disagree on the
canonical pipeline").

[`docs/roadmap_global_assembly.md`](../../roadmap_global_assembly.md) resolved
this by declaring itself canonical on 2026-06-05: "This document is now the
canonical direction for the project. It supersedes the 'primary pipeline'
framing in `README.md`, `program.md`, and `pipeline_state.md` where they
conflict." The docs below are the losing side of that reconciliation, moved
out of the live doc set so a new contributor has one place to start rather
than five.

## Contents

| File | What it claimed | Superseded by |
|---|---|---|
| [`program.md`](program.md) | "Neuronauts v2: Scaffolded Global Grammar" — the EM-voxels-to-connectome vision via one shared learned representation (agent/grammar/GAT), self-described as "v2 fully implemented" across five architectural layers. `CLAUDE.md` §2 pointed here as the pipeline overview. | [`docs/roadmap_global_assembly.md`](../../roadmap_global_assembly.md), canonical since 2026-06-05. `CLAUDE.md` §2 now points there. |
| [`NEXT_STEPS.md`](NEXT_STEPS.md) | The `neuronauts.coassign` design — synapses as invariant nodes, co-assignment via calibrated P(same neuron), K ranked materializations — reporting precision 0.95 / recall 0.42 on 20 real proofread neurons. Consolidation plan §4.1 calls this "the correct core idea." | The idea survives: folded into `assemble/` + `represent/` per `docs/consolidation_plan.md` §4.1/§3. The document itself is superseded as a roadmap by `docs/roadmap_global_assembly.md`. |
| [`STATUS.md`](STATUS.md) | A running phase-by-phase status log (Phase 0 onward) — schemas, legacy quarantine, then tree-DNA identity results (within-type AUC 0.829 at half-skeleton scale) cited as one of the few results that has held up on real data (§1.5). | `docs/consolidation_plan.md` (current survey + §8b execution log) as narrative status; `results/EXP-*/provenance.json` as the source of truth for any specific claim ("every claim is a command," §2 principle 4). |
| [`docs/pipeline_state.md`](pipeline_state.md) | "Pipeline State & Reinitialization Guide," a 2026-05-01 snapshot mixing the v1 agent/simulation framing with the no-EM pipeline and referencing checkpoints since curated away. It already carried its own historical banner. | `README.md` (what runs today), `models/README.md` (checkpoints), `docs/roadmap_global_assembly.md` (direction) — its own stated replacements. |
| [`docs/model.md`](model.md) | "Neuronauts Model & Architecture Note" — a captured design discussion covering the agent/grammar/GAT pipeline, box-scale vs. global scale, and hierarchical assembly. Already flagged itself historical. | [`docs/architecture.md`](../../architecture.md) (current system architecture) and `docs/roadmap_global_assembly.md` (current direction). |
| [`docs/global_inference_roadmap.md`](global_inference_roadmap.md) | Tracked the planned transition from box-scale grammar/heuristic assembly to scaffold-aware, globally optimized connectome inference (transformer path encoding, trajectory bridge head, Dijkstra proposals). Already self-labeled "Superseded by `roadmap_global_assembly.md`." | `docs/roadmap_global_assembly.md`, per its own banner. |
| [`docs/global_topological_merge_plan.md`](global_topological_merge_plan.md) | The implementation plan for the box-local CellGNN: tangledness scoring, spatial train/val/test splitting, the training loop in `cell_graph.py`. | `docs/roadmap_global_assembly.md`, which extends rather than replaces it — the CellGNN becomes the within-region assembler of that roadmap's Stage C. The CellGNN's architectural ceiling (~0.27 F1) is documented in `docs/architecture.md`. |
| [`docs/TODO.md`](TODO.md) | "CellGNN Pipeline — Open Items" (2026-04-28): the K-hop sweep and per-feature ablation findings for the box-local CellGNN, still cited by `models/README.md` and two `experiments/*/README.md` files. | `docs/architecture.md` (the ceiling analysis that explains why these ablations plateaued) and the experiment program in `docs/consolidation_plan.md` §6, where open items are now tracked as registered EXPs. |

## What this is not

This is not a claim that the ideas in these documents were wrong. `NEXT_STEPS.md`'s
coassign design and `docs/global_topological_merge_plan.md`'s CellGNN both
carry forward into the current direction (as `represent/`/`assemble/` code and
as the Stage C within-region assembler, respectively). What is retired is each
document's claim to be *the* canonical description of the pipeline — a claim
only one live document (`docs/roadmap_global_assembly.md`) now makes.

## Inbound references left alone

A few live documents mention these files by name in historical or
prose-citation context rather than as a navigable link, and were left
unchanged because editing them is out of this cleanup's scope (they are on the
consolidation plan's do-not-touch list, or the mention isn't a path that
resolves anywhere): `README.md` ("See `STATUS.md` for full per-phase
progression"), `docs/roadmap_global_assembly.md`, `docs/grammar_harness_handoff.md`,
`docs/grammar_literature_directions.md`, `docs/tree_assembly_algorithm.md`,
`docs/consolidation_plan.md`, and the stale-reference note in
`docs/stage_ownership.md` §"Legacy quarantine plan." Every reference that was
an actual relative link into one of these files (in `CLAUDE.md`,
`neuronauts/coassign/README.md`, `experiments/cell_assignment/README.md`,
`experiments/tree_dna/README.md`, `models/README.md`,
`docs/coassign_slides.md`, and a docstring in
`neuronauts/represent/enrich.py`) was repointed here.

Note also that each archived file's *own* internal relative links (e.g.
`docs/pipeline_state.md` linking to `../README.md`) were written for the old
`docs/` location and are now one directory level short — they were left as
originally written per the rule that only a header may be added, not the body
edited. Treat the tables above as the authoritative "superseded by" pointers
for these files, not the links inside them.
