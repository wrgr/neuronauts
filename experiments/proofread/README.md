# Two-cue abstaining auto-proofreader

Error **detection + correction** (false merges/splits) framed the way trained human
proofreaders work: every candidate edit must jointly satisfy a **global shape
*grammar*** ("does the resulting neuron parse as valid?") and a **local
ultrastructure** cue ("does the raw EM support this cut/join?"), fused by a
**calibrated, abstaining** combiner trained on the proofreading edit log, with the
residual deferred to a **ranked human queue**.

Positioning vs SOTA: **AutoProof** (shape+synapse, edit-log-trained, workflow) is
the closest and our *baseline*; **NEURD** (mesh shape only), **RoboEM** (local EM
flight only), **PathFinder** (black-box FOV plausibility). None does the explicit
two-cue + abstention decomposition. Our novel contribution is the **second cue
AutoProof omits — local EM ultrastructure** — and the test is **complementarity**:
does it resolve the residual ambiguous edits shape/grammar alone cannot?

## Pillars

- **Pillar 1 — `grammar_energy.py`** (done): global shape grammar as an energy over
  morphology (one soma; A↔D only via soma; caliber continuity; single tree). An
  edit returns **ΔEnergy** (`cut_delta_energy`, `join_delta_energy`) — the global
  cue for whether a cut/join is correct. Reuses `neuronauts.soma_clusters` and
  `experiments.pcfg.compartment_grammar`. *Validated:* correctly rejects
  soma-merging joins (ΔE −1); **neutral on same-compartment merges (ΔE 0) — the
  residual that Pillar 2 must resolve** (the complementarity boundary, visible on
  turn one).
- **Pillar 2 — `local_evidence.py`** (todo): local EM cues at the edit site —
  membrane/cross-section/cytoplasm match (committed `cutface_encoder*.pt` via
  `neuronauts.em_corridor`) + a short continuation score.
- **Pillar 3 — `combiner.py`** (todo): calibrated abstaining combiner over
  [grammar ΔEnergy, local evidence], trained on real edits
  (`edit_history` / `synapse_correction`), applied via **matching** (not
  agglomeration), residual → `treestitch.risk` queue + `ngl_export` links.

## Metric
Synapse-pair line-graph F1 (`neuronauts.line_graph.evaluate_suite`) before/after
auto-correction, leakage-safe on the proofread column; precision reported *with*
coverage (abstention). Anchors: greedy-agglo failure F1 0.14; oracle F1 0.928.
