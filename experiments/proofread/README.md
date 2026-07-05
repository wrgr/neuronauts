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
- **Pillar 2 — `local_evidence.py`** (done): local EM cues at the edit site —
  cut-face cross-section/cytoplasm match (committed `cutface_encoder*.pt` via
  `neuronauts.em_corridor.cross_section_patch`) + a membrane-barrier score along
  the connecting axis, from **one** bulk EM+seg fetch. `LocalEvidence(cutface_sim,
  barrier, ...)`; `continuation` summarises "one process". *Validated on real
  MICrONS mip-1 EM:* same-neurite z-gap pairs mean cut-face sim **+0.58** vs
  cross-neurite **+0.30** (separable), barrier ≈0 on continuous cytoplasm. Barrier
  is a labelled *first-cut* intensity approximation, not the RoboEM flight method.
- **Pillar 3 — `complementarity.py` / `queue.py` / `pipeline.py`** (built + tested):
  leakage-safe GroupKFold combiner over [shape/grammar, local evidence] on real
  column edits (`synapse_correction`), and a ranked abstaining CUT/JOIN/ABSTAIN
  queue with Neuroglancer site links (`treestitch.ngl_export`). Driver:
  `run_complementarity.py`.

## Result so far — see `FINDINGS.md` (honest: not yet a win)
Pillars 1 and 2 are **validated** as cues on real data (grammar rejects soma-merges
ΔE<0; local cut-face separates same/different process +0.58 vs +0.30 at proper
cross-section sites). But the **end-to-end complementarity does not yet clear the
bar**: on one 24 µm column box (24 local cut-errors, 0 join candidates) shape≈chance
(AUC 0.487), local best-single (0.601), joint 0.536, and confident auto-edits run
**below base rate** (0.125 vs 0.30). Diagnosed cause: candidates are sampled at
**synapse-cleft positions, not on the neurite at the edit seam** (26/80 sites have no
cross-section) — a mechanical fix (geometric seam localization), not a dead end.

## Metric
Synapse-pair line-graph F1 (`neuronauts.line_graph.evaluate_suite`) before/after
auto-correction, leakage-safe on the proofread column; precision reported *with*
coverage (abstention). Anchors: greedy-agglo failure F1 0.14; oracle F1 0.928.
*(F1 before/after is gated on placing the cues at the real edit site — FINDINGS step 1.)*
