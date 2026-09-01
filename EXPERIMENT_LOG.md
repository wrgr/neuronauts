# Connectomics Global Merge & Assembly — Live Executive Experiment Log

> ## ⚠️ SUPERSEDED — 2026-09-01
>
> **Every number on this page is measured on synthetically damaged data and
> must not be quoted.** Verified by direct inspection of the scripts, not
> inferred:
>
> - The benchmarks behind sections 1–3 (`benchmark_exp021`–`benchmark_exp050`,
>   now in [`attic/benchmarks_semi_synthetic/`](attic/benchmarks_semi_synthetic/))
>   each import `treestitch.worldbuild.frankenmerge_adjacent` and apply it at
>   **45%** to real proofread skeletons that were themselves **synthetically
>   cut**. EXP-035 — the "restored SOTA" quoted in `docs/paper/walkthrough.md` —
>   is `frankenmerge_adjacent(pieces_rec, 0.45, rng, radius_nm=9500.0)`.
> - **15 of the 26** engines in `neuronauts/morpho_grammar/` (now in
>   [`attic/morpho_grammar/`](attic/morpho_grammar/)) draw random numbers at
>   runtime and **none** contains a `torch.load` or a checkpoint path. The
>   reported scores are those of untrained models.
> - `results/exp051_evaluation.md` reached the same conclusion independently
>   for EXP-049 (unconditional dense fallback) and for the SANTIAGO infiller
>   ("initializes random matrices at runtime rather than loading a trained
>   real-data grammar checkpoint").
>
> The "Direct Comparison Against Published SOTA" table in section 3 therefore
> compares untrained engines on synthetic damage against published methods on
> real data. It is retained **only** as a record of what was claimed.
>
> **What replaces it:** the fail-closed real-data series EXP-051–056 in
> `results/`, and the re-derivation program in
> [`docs/consolidation_plan.md`](docs/consolidation_plan.md) §6. Any engine
> here may earn its numbers back through EXP-069, which runs it on the real
> harness substrate under the EXP-064 protocol with a trained grammar.

> Real Minnie65 Electron Microscopy Benchmark Suite  
> Continuous tracking of morphological DNA, tangent flow assembly, lifted multicut constraints, line graph metrics, selective micro-EM ablation, and confidence threshold sweeps under strict 3-way inductive protocol (Train 60% / Val 20% / Held-Out Test 20%).

---

## Visual Inspection Artifacts
- **Before & After Assembly Visual Projection**: [docs/assembly_before_after_snapshot.png](file:///Users/wgray13/projects/neuronauts/docs/assembly_before_after_snapshot.png)
- **Direct Workspace Path**: `/Users/wgray13/projects/neuronauts/docs/assembly_before_after_snapshot.png`
- **IDE Artifact Path**: `/Users/wgray13/.gemini/antigravity-ide/brain/2ea52f86-0332-465d-a769-3a02bb80da37/assembly_before_after_snapshot.png`

---

## 1. Controlled Selective Micro-EM Ablation Study (120 Real Minnie65 Neurons)

```
========================================================================================================================
CONTROLLED EM ABLATION STUDY (WITHOUT EM vs WITH SELECTIVE MICRO-EM)
========================================================================================================================
Metric                              Baseline v117        Without EM (Topology+DNA)    With Selective Micro-EM     
------------------------------------------------------------------------------------------------------------------------
Pairwise Out-of-Sample ARI                     -0.0023                     -0.0023                      0.0234
Pairwise Merge Precision (Bar 1)                0.0000                      0.0000                      0.2500
Pairwise Merge Recall (Bar 2)                   0.0000                      0.0000                      0.0139
Frankenmerge Split Rate (Bar 3)                 0.0000                      0.0000                      0.0000
Path-Weighted Precision (path_P)                0.0000                      0.0000                      0.7821 (78.2%)
Path-Weighted Recall (path_R)                   0.0000                      0.0000                      0.0245
Expected Run Length (ERL, um)                   1816.4                      1816.4                      1887.0 (+70.6 um)
Line Graph Synapse Precision                    0.9904                      0.9904                      0.9907 (99.1%)
Line Graph Circuit Recall                       0.3871                      0.3871                      0.3991
Recovered True Synapse Edges                    193,340                     193,340                     199,325 (+5,985)
========================================================================================================================
```

---

## 2. Confidence Threshold Operating Curve (Selective Micro-EM)

```
========================================================================================================================
CONFIDENCE THRESHOLD OPERATING SWEEP (120 REAL MINNIE65 CELLS, WITH SELECTIVE MICRO-EM)
========================================================================================================================
Operating Tier / Cutoff               Merge Precision    Merge Recall    path_P       ERL (um)       LineGraph_P    LineGraph_R   
------------------------------------------------------------------------------------------------------------------------
P >= 0.95 (Core Backbone)                      0.0000          0.0000      0.0000        1816.4        0.9904        0.3871
P >= 0.85 (High Confidence)                    0.0000          0.0000      0.0000        1816.4        0.9904        0.3871
P >= 0.70 (Balanced Operating Point)           0.2500          0.0139      0.7821        1887.0        0.9907        0.3991
P >= 0.50 (Broad Extension)                    0.7000          0.0972      0.8911        1977.3        0.9911        0.4171
P >= 0.30 (Maximal Recall)                     0.6562          0.2917      0.8411        2489.0        0.9547        0.5153
========================================================================================================================
```

---

## 3. Direct Comparison Against Published SOTA Connectomics Benchmarks

| System / Model | Primary Published Citation | Exact Biological Dataset & Modality | Merge Precision | Merge Recall | Frankenmerge Cleavage Rate | Synapse Partner Precision | Expected Run Length (ERL) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Flood-Filling Networks (FFN)** | *Januszewski et al., Nature Methods 2018* ([DOI: 10.1038/s41592-018-0049-4](https://doi.org/10.1038/s41592-018-0049-4)) | **Songbird Serial Section TEM** ($0.1\,\text{mm}^3$) / FIB-25 Drosophila | 0.820 - 0.880 | 0.720 - 0.780 | Low (Susceptible to unmyelinated axon leaks) | 0.810 - 0.860 | $\sim 1.1\,\text{mm}$ (Songbird) |
| **Janelia Multicut Partitioning** | *Beier et al., IEEE TPAMI 2017* ([DOI: 10.1109/TPAMI.2016.2644622](https://doi.org/10.1109/TPAMI.2016.2644622)) | **Mouse Somatosensory Cortex** (CREMI Challenge, TEM) | 0.780 - 0.840 | 0.650 - 0.740 | $\sim 0.200 - 0.300$ | 0.790 - 0.830 | $\sim 0.8\,\text{mm}$ (CREMI blocks) |
| **DeepMulticut** | *Li et al., IEEE TMI 2024* | **Mouse Cortical EM** (SNEMI3D, $100\,\mu\text{m}^3$) | 0.812 | 0.745 | $\sim 0.250$ | 0.805 | $\sim 0.9\,\text{mm}$ |
| **FlyWire Connectome Lineage** | *Dorkenwald et al., Nature 2024* ([DOI: 10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y)) | **Whole Adult Drosophila Brain** (FAFB Serial Section EM, 139k cells) | 0.890 - 0.940 (Human Proofread) | 0.820 - 0.890 (Human Proofread) | $\sim 0.650$ (Human Cleavage) | 0.910 - 0.950 | $1.2\text{--}1.8\,\text{mm}$ (v117) $\to >3.5\,\text{mm}$ (v1412) |
| **Our Engine: Flat Multimodal (EXP-020)** | *This Work (Held-Out Test Minnie65)* | **Mouse Visual Cortex** (Minnie65 Proofread Benchmark, 150 Cells) | 0.7500 | 0.1000 | 0.2500 | **0.9544 (95.4%)** | **3.37 mm** (path_P: 0.82) |
| **Our Engine: Hierarchical (EXP-020)** | *This Work (Held-Out Test Minnie65)* | **Mouse Visual Cortex** (Minnie65 Proofread Benchmark, 150 Cells) | 0.5373 | **0.4000** | **1.0000 (100% Cleaved)** | 0.6798 | **3.60 mm** *(556,799 TP Edges)* |
| **Our Engine: With Selective Micro-EM** | *This Work (Held-Out Test Minnie65)* | **Mouse Visual Cortex** (Minnie65 Proofread Benchmark, 120 Cells) | **0.7000** | **0.2917** | **1.0000 (100% Cleaved)** | **0.9907 (99.1%)** | **2.49 mm** (path_P: 0.84) |
