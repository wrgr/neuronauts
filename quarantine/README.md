# Quarantine — synthetic-derived and label-leaking code

**Nothing in this directory may be used to produce a reported metric.**

This code is retained as a historical record of the synthetic-data quality
incident, not as a working component. It is excluded from imports, tests, and
CI. Most of it will not even run in a clean checkout (many files hardcode
`sys.path.insert(0, "/Users/wgray13/projects/neuronauts")`).

Full analysis: [`docs/synthetic_data_audit_and_dataset_plan.md`](../docs/synthetic_data_audit_and_dataset_plan.md).

## The quarantine rule

Code lands here if it **manufactures segmentation fragments or synapses and
then reports assembly / connectome / proofreading metrics from them**, or if it
**passes a ground-truth label into a scorer as an input**.

Applied mechanically, by what the code does — not by whether the surrounding
documentation happened to disclose it. Some quarantined scripts *were*
honestly captioned in `STATUS.md` (noted below); they are still quarantined,
because the artifact itself carries no such caption and gets cited without it.

**Deliberately NOT quarantined:** encoder-representation ablations
(`scripts/half_split_ablation.py`, `within_type_ablation.py`,
`multi_fragment_ablation.py`). These bisect skeletons as the *stated
experimental task* (within-neuron re-identification) and report an encoder AUC,
not a v117→proofread assembly claim. Their uniform-random synapse placement is
an explicitly reported chance-level control. They remain subject to the
provenance lint and must carry a synthetic declaration.

## What is in here and why

### 1. The five root defects

| Defect | Location (now under `quarantine/`) |
|---|---|
| **D3** — leaked partner ID used as a scoring feature with weight 3.0 | `neuronauts/morpho_grammar/tree_grammar_infiller.py:117-140` |
| **D4** — "Micro-EM verifier" that never reads EM; returns a Gaussian conditioned on the ground-truth label | `neuronauts/global_merge/represent/cloudvolume_em_sampler.py:29-66`, `local_em_verifier.py` |
| **D4** — related oracles returning the answer directly | `morpho_grammar/active_gap_oracle.py:86-89` (p=0.99), `geodesic_em_tracer.py:43-46` (0.98/0.05) |
| **D5** — "Tree-Grammar Transformer" is untrained random matrices; no checkpoint ever loaded | `morpho_grammar/tree_grammar_infiller.py:20-45` |
| **RNG "baselines"** — the published-SOTA comparison rows compare against noise | `morpho_grammar/autoproof_baseline.py:29`, `neurd_baseline.py:29` |
| **RNG "texture prior"** | `morpho_grammar/ultrastructural_texture_prior.py:27` |

**D1** (fragments manufactured by bisecting proofread skeletons) and **D2**
(synapses fabricated with `partner_base = obj_counter * 100`, making partner
overlap a deterministic function of neuron identity) live in the benchmark
scripts below and in `treestitch/data.py::_split_skeleton_n_pieces`, which
stays in the main tree because live code depends on it — it now carries an
explicit warning and is enforced by the provenance lint.

### 2. Benchmark scripts (`quarantine/scripts/`)

All share one copy-pasted data section: real skeletons → bisected into thirds →
fabricated synapses with identity-encoding partner IDs → injected frankenmerges
→ an ordinal 60/20/20 "split" whose **validation set is never materialized**
(`n_val` only offsets the test slice; no `val_pieces` variable exists anywhere).
The same `sample_neurons(250, seed=42)` population and its "held-out" test slice
were reused across ~28 experiments.

- `benchmark_exp021_3d.py` … `benchmark_exp050_interneuron_stratified.py`
- `benchmark_{dual_engine,pcfg_infiller,asymmetric_relational,volumetric_em_inductive}.py` (EXP-022–025)
- `benchmark_{bar3_breakthrough,definitive_large_scale,multimodal_synapse_dna,synapse_membership_box,em_and_confidence_sweep,multi_region_dense}.py`
- `test_global_merge_franken.py` (EXP-001–007; no train/test split at all — 15 neurons, evaluated on the same 15)

Worst individual offenders:

- **`benchmark_exp049_dense_subvolume.py:278-279`** — docstring claims real v117
  data with v1412 proofread labels; the code calls
  `generate_dense_subvolume_fallback(...)` **unconditionally**. The real-fetch
  import is never called. Random-walk skeletons are named `v117_seg_NNNN` to
  look real. This is the archetype of the incident.
- **`benchmark_exp050_interneuron_stratified.py:239-268`** — a "stratified
  PV+/SST+/VIP cell-type benchmark" that generates its own cell morphologies.
  No real cells at all.
- **`benchmark_exp033_end_to_end_synapse_typing.py`** — "synapse typing
  accuracy" measures recovery of a rule the script itself wrote (`syn_types` is
  assigned by piece index).
- **`benchmark_exp045_full_spectrum_evaluation.py:133-137`** — synapse polarity
  drawn from hand-picked priors.

### 3. Artifact generators

- **`scripts/generate_dashboard.py:129-132`** — the displayed "prediction" is
  the ground-truth label ± a fixed offset (`+0.40` if same cell, `−0.35`
  otherwise); headline KPIs are hardcoded HTML strings.
- **`scripts/export_viz_data.py`** — produced `viz/sample_connectome_viz.json`
  and the `.swc` "community validation" exports from manufactured fragments,
  fabricated synapses, and injected frankenmerges.

### 4. Simulated-v117 partition scripts

Real skeletons fetched, then split into pieces to *simulate* v117 fragmentation,
with synapses placed synthetically:

- `scripts/real_skeleton_partition.py` — source of the ARI 0.088 → 0.418
  endpoint-edge result cited in `NEXT_STEPS.md` as "the single most important
  improvement". **The conclusion survived independent replication on real L2
  skeletons** (Phase 2.3: ARI 0.305 → 0.838 with genuine endpoint edges), but
  the originally cited evidence was synthetic-derived.
- `scripts/real_franken_partition.py` — Phase 2.2 adjacent-neuron franken
  validation. `STATUS.md` captioned this one honestly.

## What was NOT quarantined and remains trustworthy

- **Phase 2.3–2.12** (`STATUS.md`): real v117→v1718 lineage supervision via
  `neuronauts/data/lineage.py`, region sampling, and the Phase 2.11 leak-fixed
  protocol (50 µm seam buffer + root dedup) — honest out-of-sample ARI 0.752,
  merge_P 0.951, fk_split 0.000.
- **EXP-051–056** (`results/exp05*.md`): fail-closed real benchmarks with
  pre-registered gates and label-blind inference. All honest negatives.
- **`experiments/pcfg/HOLDOUT_RESULTS.md`**: three spatially disjoint 60 µm
  boxes ≥90 µm apart with epoch selection on a third region.
- **`experiments/fingerprints`**: synthetic pretraining declared up front,
  evaluated only on held-out real sites.
