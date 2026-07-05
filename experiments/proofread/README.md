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

## Result — see `FINDINGS.md` (honest: the local cue does not deliver)
**Global grammar (Pillar 1) is the cue that carries deployable signal** — it rejects
real multi-soma merge seams (ΔE −1) and is blind only to same-compartment seams (the
complementarity boundary). **The local cut-face ultrastructure cue (Pillar 2) does
not.** Tested directly at real merge seams *on the neurite* (`seam_test.py`), cut-face
similarity is anti-predictive (seam 0.627 vs continuation 0.552, AUC 0.40) because
cut-face/SegCLR embeddings encode **cell type, not identity**, and the two processes
at a seam are adjacent same-type — the earlier +0.58/+0.30 re-ID used easy random
distractors, not the hard seam negatives. So the two-cue thesis is **not supported by
the local cue we have**; a useful second cue must read membrane continuity / topology
across the seam, not cross-section appearance. The infrastructure (three pillars,
combiner, abstaining queue, ground truth) is built and tested; the conclusion is that
grammar, not local ultrastructure, is the signal.

**Following by inference works (`follow_test.py`).** The constructive counterpart:
cut a real neurite, open a gap, and rank competing processes by *trajectory* (no EM).
Proximity is useless (top-1 0.46 ≈ chance) but trajectory+caliber follows the true
continuation at **0.96 top-1**, recovering **94%** of proximity-failures. On the hard
parallel-process (fascicle) cases, direction alone gets 0.56, trajectory+caliber 0.72,
and adding **bidirectional consistency** (does the far end's cable point back through
the gap at the cut — inference by consequence) lifts it to **0.84** (verified across 4
seeds; driver = reciprocal trajectory). So identity lives in *geometry/trajectory*,
not cross-section appearance, and each layer of *logical* consistency closes more of
the residual — the learnable backbone of "follow like a human," trainable on the edit
log, no EM.

## Metric
Synapse-pair line-graph F1 (`neuronauts.line_graph.evaluate_suite`) before/after
auto-correction, leakage-safe on the proofread column; precision reported *with*
coverage (abstention). Anchors: greedy-agglo failure F1 0.14; oracle F1 0.928.
*(F1 before/after is gated on placing the cues at the real edit site — FINDINGS step 1.)*
