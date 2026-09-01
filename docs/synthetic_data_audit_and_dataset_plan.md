# Synthetic-data audit and canonical train/test dataset plan

> Written on branch `claude/synthetic-data-quality-review-9ihg5x`, 2026-09-01.
> Scope: full review of the ~60-experiment history following the discovery that
> synthetic data was used without authorization and results were overstated;
> proposal for a canonical train/val/test dataset as the reset point.
>
> Every defect cited below was verified directly in this checkout (file:line).
> Where a claim comes from a doc rather than code, the doc is cited.

---

## Part 1 — Audit findings

### 1.1 Three tracks, opposite quality

| Track | Where | Verdict |
|---|---|---|
| **"Phase" track** (Phases 0–2.12) | `STATUS.md`, `treestitch/`, `scripts/spatial_train_test_split.py`, `multi_region_train.py`, `train_l2_partition.py` | Mostly **real** CAVE lineage data, honestly labeled, culminating in a leak-fixed seam-buffered protocol (Phase 2.11). **Salvageable foundation.** |
| **"EXP-NNN" track** (EXP-001–050) | `EXPERIMENT_LOG.md`, `scripts/benchmark_exp0*.py`, `docs/paper/*`, `dashboard/`, `viz/` | **Synthetic-derived with ground-truth leakage.** Source of every headline claim. **Retract.** |
| **Cleanup track** (EXP-051–056) | `results/exp05*.md`, `scripts/benchmark_exp05[1-6]*.py` | Real, fail-closed, all honest negatives. Correct practice; currently **blocked on data**, not methodology. |

### 1.2 The five root defects (shared library code → contaminate dozens of experiments at once)

**D1 — Fragments are manufactured, not real.**
`treestitch/data.py:94-103` (`_split_skeleton_n_pieces`) bisects real proofread
skeletons into equal thirds and calls the thirds "v117 fragments". The repo's
own measurement (`STATUS.md:373-382`, `characterize_v117_to_v1718.py`) shows
real v117 structure is "one trunk + slivers" — 88% of somata are already a
single v117 root. Used by 31 scripts.

**D2 — Synapses are fabricated and their partner IDs encode the answer.**
The idiom `partner_base = obj_counter * 100` appears in **31 scripts**
(e.g. `scripts/benchmark_exp026_enhanced_dual_engine.py:58-66`). Partner IDs
are drawn from a per-neuron pool, so partner overlap is a deterministic
function of the ground-truth neuron identity.

**D3 — The leaked partner ID is a scoring feature with weight 3.0.**
`neuronauts/morpho_grammar/tree_grammar_infiller.py:117-140`: synaptic
co-targeting Jaccard between fabricated partner sets enters `composite_logit`
with coefficient 3.0. This label-as-feature term drives the Top-1/Top-3
"infilling accuracy" numbers.

**D4 — The "Micro-EM verifier" never touches EM.**
`neuronauts/global_merge/represent/cloudvolume_em_sampler.py:29-66` takes
`is_true_continuation` (the ground-truth label) as an argument and returns a
Gaussian sample from one of two distributions depending on it. No CloudVolume
call, no voxels. The label is piped in from `gt_map`/`gt_target_id` at every
call site (e.g. `benchmark_exp021_3d.py:196,201`,
`morpho_grammar/dual_engine_infiller.py:78`). Related oracles:
`active_gap_oracle.py:86-89` (returns the answer at p=0.99),
`geodesic_em_tracer.py:43-46` (0.98 if true else 0.05). This is the sole
source of the "Selective Micro-EM Ablation Study" table in
`EXPERIMENT_LOG.md`.

**D5 — The "Tree-Grammar Transformer" is untrained random matrices.**
`tree_grammar_infiller.py:20-45` initializes all weight matrices from
`rng.normal` at construction; no checkpoint is ever loaded by any
`benchmark_exp0[1-5]0` script. Independently confirmed by the EXP-051 audit
(`results/exp051_evaluation.md`).

### 1.3 The claimed "strict 3-way inductive protocol" is not real

`EXPERIMENT_LOG.md:4` and `docs/paper/MICCAI_2026_Neuronauts.tex` claim a
Train 60% / Val 20% / Held-Out Test 20% protocol. In code
(`benchmark_exp026_enhanced_dual_engine.py:94-97` and ~28 identical copies):

- The **validation set is never materialized** — `n_val` only offsets the test
  slice; no `val_pieces` variable exists anywhere. "Hyperparameter tuning on
  Val" as claimed in the paper did not happen; tuning was done against test.
- The split is an **ordinal cut on fetch order** over *synthetic pieces*, with
  no spatial stratification and no seam buffer. Phase 2.11 measured that seam
  leakage alone is worth ΔARI −0.149 and Δfk_split −0.350.
- The same `sample_neurons(250, seed=42)` population and its "held-out" test
  slice were **reused across ~28 experiments** — the test set was effectively
  a development set.
- `frankenmerge_adjacent` corruption is applied **before** the split, so
  injected errors straddle the train/test boundary.

### 1.4 Additional integrity findings

- **EXP-049** (`benchmark_exp049_dense_subvolume.py:278-279`): docstring
  claims real v117 data with v1412 labels; in fact the code calls
  `generate_dense_subvolume_fallback(...)` **unconditionally** (verified —
  the CAVE fetch import is never called) and names random-walk skeletons
  `v117_seg_NNNN` to look real. 100% synthetic.
- **EXP-050** (`benchmark_exp050_interneuron_stratified.py:239-268`): the
  "interneuron-stratified benchmark" generates its own pyramidal / basket /
  Martinotti / VIP skeletons. No real cells at all.
- **EXP-044**: the "published-method baselines" (AutoProof, NEURD proxies) are
  RNG stubs (`morpho_grammar/autoproof_baseline.py:29`,
  `neurd_baseline.py:29`) — the SOTA-comparison rows compare against noise.
- **Dashboard** (`scripts/generate_dashboard.py:129-132`): the displayed
  "prediction" is the ground-truth label ± a fixed offset; headline KPIs
  ("3,595.4 μm", "95.44%") are hardcoded HTML strings.
- **Viz/SWC exports** (`scripts/export_viz_data.py:33-58`): synthetic
  fragments + fabricated synapses + injected frankenmerges exported as
  `viz/sample_connectome_viz.json` and `.swc` "for community validation".
- Neither paper (`docs/paper/MICCAI_2026_Neuronauts.tex`,
  `TreeGrammar_Connectomics_2026.tex`) discloses any of this; the word
  "synthetic" does not appear.
- ~30 benchmark scripts hardcode `sys.path.insert(0, "/Users/wgray13/...")` —
  the EXP-017–050 results are not reproducible in a clean checkout as written.
- **A CAVE bearer token is committed in plaintext** in ≥10 files
  (`neuronauts/data/loaders.py:43`, `scripts/coassign_demo.py`,
  `scripts/probe_seg_mapping.py`, `docs/seg_117_to_1412.md`, …).
  **Rotate it** and move to env-only per `CLAUDE.md` §1.

### 1.5 Tainted claims that must be retracted or re-derived

All of the following trace exclusively to D1–D5 pipelines:

- The `EXPERIMENT_LOG.md` SOTA comparison table ("Our Engine" rows: merge_P
  0.75, synapse precision 95.4%/99.1%, ERL 3.37–3.60 mm, 556,799 TP edges).
- The Selective Micro-EM ablation and confidence-sweep tables (LineGraph_P
  0.9904, +5,985 recovered edges, etc.).
- "88.33% Top-3 / 46.67% Top-1 / 60.55% circuit recall / 482k synapses"
  (EXP-026) and "75% Top-3 / 81.46% path precision" (EXP-025).
- Both paper drafts, `docs/whitepaper.md`, `docs/slides.html`, `dashboard/`,
  `viz/sample_connectome_viz.json`, the `.swc` community exports.

### 1.6 What is clean and worth keeping

- **Phase 2.3–2.12** (`STATUS.md`): real v117→v1718 lineage supervision
  (`neuronauts/data/lineage.py`), region-based sampling, and the **Phase 2.11
  leak-fixed protocol** (50 µm seam buffer + cross-boundary root dedup) with
  honest out-of-sample numbers: ARI 0.752, merge_P 0.951, fk_split 0.000.
- **EXP-051–056**: fail-closed real benchmarks with pre-registered gates,
  checkpoint SHA recording, and label-blind inference. All honest negatives.
- **`experiments/pcfg/HOLDOUT_RESULTS.md`**: the best split in the repo —
  three spatially disjoint 60 µm boxes ≥90 µm apart, epoch selection on a
  third region (AUC 0.816 [0.754, 0.874]).
- **`experiments/fingerprints`**: synthetic pretraining honestly declared,
  evaluated only on held-out real sites, and reports its own synthetic
  over-promise.
- `models/*.pt` checkpoints that `models/README.md` documents as real-data
  trained (with `grammar_synthetic.pt` correctly labeled as synthetic).

### 1.7 Why the EXP-051–056 cleanup stalled — the data problem

Every gate failure has the same root cause, and it is **data, not models**:

1. **Ordinary spatial boxes carry no edit signal.** Verified in
   `docs/dataset_seeding_for_edit_pairs.md`: 0% v117≠current divergence in
   all three box caches vs 28% in proofread-root neighborhoods. EXP-051's box
   had **1** true merge pair among 21,175 candidate joins; EXP-052's
   soma-anchored box had 14 among 29,985.
2. **No checkpoint was ever trained for the dense-confuser regime** the gates
   test (EXP-053A: all four real checkpoints fail identically).
3. **L2 substrate coverage is inadequate as fetched** (EXP-053B: 27.8% of
   roots covered; 1/14 true pairs had geometry on both sides; unbounded
   retrieval didn't complete).
4. **Label noise floor exists**: 116/11,241 v117 roots in one box are
   mixed-lineage (frankenmerges), and EXP-056 showed geometry-only atomization
   can't cleave them. "v117 fragment = atom" is an approximation, not a fact.

Conclusion: **a curated, edit-bearing, spatially split, provenance-stamped
dataset is the prerequisite for every next step.** That is Part 2.

### 1.8 Current data reality (as of this checkout)

- **No labeled data exists on disk.** `cache/synapse/` and
  `cache/l2_skeleton/` contain only `PROVENANCE.json` manifests, zero `.npz`
  files; `cache/l2_world/` (required by `train_l2_partition.py` and
  `p1_completeness_benchmark.py`) does not exist. Everything must be
  re-fetched.
- **Materialization versions are inconsistent** across modules: defaults of
  117 / 1412 / 1718 are scattered (`edit_history.py`→1718,
  `cave_synapse*.py`→1412, `bulk_synapses.py`/`path_dataset.py`→117), and
  `lineage.py:495` warns **v1412 is expired**. Available versions per
  `lineage.py`: 117, 943, 1300, 1507, 1621, 1718.
- **Over-limit region fetches are non-deterministic** (`lineage.py:595-598`):
  the server returns a different subset each call, so an uncached re-fetch
  silently changes the dataset.
- The default grammar training split (`scripts/train.py:568-572`) is a flat
  random shuffle over box records with **no test set** — since arbors span
  many boxes, the same neuron lands in train and val. Avoid it.
- The strongest existing split machinery: region registry A–E + P1/T1–T4/OOC
  (`scripts/train_l2_partition.py:52-58`, `scripts/spatial_variance.py`),
  seam-buffered bbox split (`scripts/spatial_train_test_split.py`), and the
  fragment-centroid-with-buffer split (`train_l2_partition.py:244-254`).

---

## Part 2 — Plan: `neuronauts-bench v1`, a canonical train/val/test dataset

Design principles, each one earned by a specific failure above:

1. **The split unit is the neuron / spatial region — never the box** (fixes
   §1.8 random-shuffle leakage; arbors span hundreds of µm).
2. **Labels come only from real proofreading lineage** — v117 roots labeled by
   the v117→target-version supervoxel lineage (`fragment_breakdown`,
   `root_label_map`). No generated fragments, no fabricated synapses, ever.
3. **Sampling must be edit-anchored** (fixes §1.7.1): boxes/regions selected
   from proofread-soma anchors and edit-rate scans, with a pre-registered
   edit-signal gate before any training or eval.
4. **Fail-closed everywhere** (adopt the EXP-051–056 harness): missing data,
   missing checkpoint, or failed gate ⇒ hard exit recording the failure —
   never a synthetic fallback. Synthetic data is opt-in (`--synthetic`),
   clearly labeled, and stamped into every output artifact.
5. **Test set is locked**: written once, evaluated rarely, never tuned
   against (fixes §1.3 test-reuse).

### Step 0 — Freeze and quarantine (no science; ~1 day)

- Add a taint banner to `EXPERIMENT_LOG.md`, `docs/whitepaper.md`, both
  papers, `dashboard/`, `viz/` stating which numbers are synthetic-derived
  and leaked, linking to this audit. Do **not** delete history — the record
  of what happened is part of the fix.
- Move `scripts/benchmark_exp0{17..50}*.py`, the shared synthetic benchmark
  helpers (`benchmark_{bar3_breakthrough,definitive_large_scale,dual_engine,
  pcfg_infiller,multimodal_synapse_dna,asymmetric_relational,
  em_and_confidence_sweep,volumetric_em_inductive,multi_region_dense,
  synapse_membership_box}.py`, `export_viz_data.py`, `generate_dashboard.py`)
  and the oracle modules (`morpho_grammar/`,
  `global_merge/represent/cloudvolume_em_sampler.py`, `local_em_verifier`)
  into a `quarantine/` tree that is excluded from imports and CI.
- **Rotate the committed CAVE token**; purge it from source (env-only).
- Decide (owner call): whether the papers/exports shared externally need an
  explicit correction notice.

### Step 1 — Pin the coordinate/version contract (~half day)

One place (`neuronauts/data/versions.py` or similar), imported everywhere:

- `BASE_VERSION = 117` (the segmentation being corrected).
- `LABEL_VERSION` = **one** verified-available materialization. Candidates:
  1300 (recommended by `lineage.py` as the v1412 replacement) or 1718 (most
  proofreading signal, used by `edit_history.py` and the Phase 2.x work).
  **Recommendation: 1718** — more accumulated proofreading = more positive
  pairs, and it's what the clean Phase 2.x results already use. Verify with
  `list_versions()` at build time and hard-fail on drift.
- One coordinate frame declaration (nm, (4,4,40) synapse-table voxels) with
  converters — the (8,8,40) bug that once put 93% of box centers outside the
  volume argues for a single audited module.
- Kill the scattered per-module defaults (117/1412/1718) in favor of explicit
  arguments.

### Step 2 — Region inventory and edit-signal survey (~1–2 days, network)

Before choosing splits, measure where the label signal actually is:

- Scan the proofread column: for a grid of candidate regions (start from the
  known registry — P1, A–E, T1–T4, plus the validated pcfg region at
  (733.6k, 513.6k, 595.6k)), compute per-region: synapse count, v117 root
  count, **true merge-pair count** (v117 roots sharing a LABEL_VERSION root),
  **mixed-lineage root count** (frankenmerges), soma count, and L2 coverage
  fraction. All via `fragment_breakdown` / `roots_at` — the same primitives
  EXP-051/052 used.
- This is cheap relative to its value: EXP-051 vs EXP-052 showed a 14×
  difference in positive pairs between an ordinary and a soma-anchored box.
  We need regions with **hundreds** of positives, which means larger or
  denser regions than 30 µm cubes — P1 (100×100×200 µm, ~100% soma edit
  rate) is the anchor candidate.
- Deliverable: `docs/region_inventory.md` + machine-readable
  `cache/region_inventory.json` ranking regions by label richness.

### Step 3 — Define and materialize the canonical splits (~2–3 days)

**Split design** (Phase 2.11 protocol, promoted to the standard):

- **Train**: 3–5 disjoint regions from the inventory (start from A–E where
  they score well), each with ≥50 true merge pairs and ≥20 frankenmerges.
- **Val**: 1–2 regions, spatially disjoint from train with **≥50 µm seam
  buffers**, used for all model selection, threshold calibration, and epoch
  choice. (This is the set the EXP-0xx track never had.)
- **Test**: P1 east (or the top inventory region not used above), locked.
  Also keep 1–2 **out-of-column** regions as a transfer probe (pseudo-labels
  only; reported separately, never headline).
- **Leakage rules**, enforced by code, not convention:
  - 50 µm seam buffer between any two splits on every shared face;
  - **root dedup**: any v117 root (or LABEL_VERSION root) appearing in two
    splits is dropped from the *training* side;
  - fragments with centroids in buffer bands dropped from both sides
    (the `train_l2_partition.py:244-254` mechanism);
  - a `verify_split.py` gate asserting zero shared roots and printing the
    buffer distances — run in CI against the committed manifests.

**Materialization**:

- Fetch synapses (tiled, under the server limit so fetches are deterministic —
  never the unordered over-limit path), L2 skeletons per v117 root, lineage
  maps, soma/nucleus anchors, and edit pairs
  (`edit_history.edits_to_synapse_pairs`) per region into
  `cache/{synapse,l2_skeleton,l2_world}/` `.npz` with the existing
  `__provenance__` stamping (code version, versions, bbox, seed, sha).
- Commit to the repo: the **manifests** (bbox specs, root-ID lists per split,
  seeds, checksums, fetch timestamps) — small JSON/TSV. The bulk `.npz` stays
  local/LFS with a one-command rebuild script (`make dataset` /
  `scripts/build_bench_v1.py`).

**Acceptance gates — the dataset is not "done" until all pass** (per
`CLAUDE.md`: counts validated against a trusted query):

1. Edit-signal gate: every train/val/test region has nonzero merge pairs
   **and** split pairs; counts match an independent recomputation via the
   raw-HTTP lineage path vs the caveclient path (the two stacks cross-check
   each other).
2. Count validation: per-region synapse counts within tolerance of a direct
   `synapses_pni_2` query for the same bbox.
3. Leakage gate: `verify_split.py` clean.
4. Coverage gate: L2 coverage ≥ an explicit threshold (EXP-053B's 27.8% is
   the failure mode; if bounded retrieval can't clear it, precompute the L2
   cache per root unbounded — budgeted, cached, resumable — before locking).
5. Label-noise report: mixed-lineage (frankenmerge) rate per region recorded
   in the manifest; these roots get a `mixed=True` flag so evaluations can
   report with/without them rather than pretending atomicity.

### Step 4 — Honest re-baseline on the canonical splits (~1 week)

Purpose: replace the retracted table with real numbers, however modest.

- **Trivial baselines first** (these define the floor): untouched-v117
  (predict no merges — EXP-051 showed this already gives ERL 81 µm / circuit
  F1 0.987 in-box), spatial-proximity union-find, same-L2-component.
- **Clean-track models next**: EdgePartitionGNN + edge_cc (Phase 2.11
  config), union-find over SkeletonGNN embeddings — trained on Train, tuned
  on Val only, one run on Test.
- Report the honest metric suite: ARI, merge_P/merge_R at the val-calibrated
  operating point, fk_split, ERL delta vs untouched-v117, and circuit F1 —
  each with the split, versions, dataset manifest hash, and checkpoint SHA
  stamped in a machine-readable `results/*.json` (EXP-053A already
  established this pattern).
- Rewrite `EXPERIMENT_LOG.md` from these results; the old log moves to
  `docs/history/experiment_log_retracted.md` with the taint banner.
- Then, and only then, resume the EXP-054/055 sequence (scorer bake-off,
  conservative forest) — their prerequisite gates should now be satisfiable
  because the panel comes from label-rich regions with adequate L2 coverage.

### Step 5 — Guardrails so this can't recur (~1–2 days, parallel)

- **CI provenance lint**: fail any PR whose non-quarantined code (a) passes a
  ground-truth label into a scorer/verifier signature
  (`is_true_continuation`, `gt_target_id`, `is_same_cell` as a
  feature/input), (b) contains a fallback that generates data on fetch
  failure without `--synthetic`, or (c) writes a results file without a
  provenance stamp.
- **Results schema**: a tiny `neuronauts/results_schema.py` that every
  benchmark uses — refuses to write results lacking
  `{data_manifest_sha, base_version, label_version, split, checkpoint_sha,
  synthetic: bool}`.
- **Split immutability**: test-region manifests carry a checksum; CI fails if
  they change without a version bump (`bench_v2`).
- Standing rule in `CLAUDE.md`: synthetic data is opt-in, labeled in every
  table/figure it touches, and never mixed into a headline metric.

### Sequencing and decision points

```
Step 0 (freeze/quarantine)  ──►  Step 1 (version pin) ──► Step 2 (survey)
                                                            │
Step 5 (guardrails) ── parallel ──────────────────────────► Step 3 (build + gates)
                                                            │
                                                          Step 4 (re-baseline)
```

Decisions needed from the project owner:

1. **LABEL_VERSION**: 1718 (recommended) vs 1300.
2. **External corrections**: whether anything derived from EXP-020/023/026/035
   (papers, viz exports, dashboard) was shared externally and needs a notice.
3. **Token rotation**: the committed CAVE token must be rotated outside this
   repo.
4. **Compute/storage budget** for the unbounded L2 precompute if the coverage
   gate fails on bounded retrieval (EXP-053B suggests it will).

---

## Part 3 — Execution record (2026-09-01)

Steps 0, 1, 2, 3 and 5 are done. Step 4 (honest re-baseline) is the next task
and now has a dataset to run on.

### What shipped

| Step | Status | Artifacts |
|---|---|---|
| 0 — freeze & quarantine | **done** | `quarantine/` (40 scripts + `morpho_grammar/` + 2 EM-oracle modules), retraction notices on `EXPERIMENT_LOG.md`, both manuscripts, LaTeX sources, whitepaper, slides, `dashboard/`, `viz/`; CAVE token purged from 11 files |
| 1 — version contract | **done** | `neuronauts/data/versions.py`, verified live |
| 2 — region survey | **done** | `scripts/survey_regions.py`, `docs/region_inventory.md`, `results/region_inventory.json` |
| 3 — dataset build | **done** | `scripts/build_bench_v1.py`, `scripts/verify_split.py`, `data/bench_v1/` |
| 5 — guardrails | **done** | `scripts/lint_provenance.py`, `neuronauts/results_schema.py`, `.github/workflows/provenance.yml`, 27 tests |
| 4 — re-baseline | **next** | — |

### Verified facts established during execution

- **v1412 is genuinely gone.** The server reports
  `[117, 943, 1300, 1507, 1621, 1718, 1822]`. The code only *warned* about this;
  it is now enforced by `verify_version_contract()`. A newer 1822 exists and is
  the natural `bench_v2` bump.
- **The 20,000 operating point is empirical; its cause is NOT isolated.**
  Same bbox: `limit=20,000` → 20,000 rows in 50.7s; `limit=50,000` → 260.9s;
  `limit=200,000` → nothing, exceeding `lineage.py`'s own 300s timeout. The
  tiled fetch defaulting to `per_tile_limit=200_000` is therefore incompatible
  with the module's own timeout — that part was our misconfiguration, not a
  server fault, and it is fixed.
  **Correction:** I initially wrote that "request time tracks `limit`, not bbox
  size". That was generalised from two observations and is wrong — a 10 µm cube
  later failed at `limit=20,000` while a 40 µm-wide tile succeeded at the same
  limit. A raw request for that small box returned HTTP 200 in 85.6s on one
  attempt and died with `ProxyError(RemoteDisconnected)` on another: long
  requests are unreliable through this session's egress proxy, which itself
  reports healthy with no recent relay failures. What is established is only
  that `limit=20,000` completes reliably here. A related hazard:
  `fetch_region_synapses` collapses non-200s, parse errors and proxy
  disconnects alike into `None`, so callers cannot tell "empty region" from
  "request died" — which is why the builder fails closed on `None`.
- **Merge-pair counts are sampling-density dependent.** Each P1 z-third sampled
  at 20,000 synapses yields as many or more true merge pairs (16 / 32 / 22) than
  all of P1 sampled at 20,000 (16). A pair is observable only when *both*
  fragments land in the sample. **Therefore the zeros in
  `docs/region_inventory.md` (B, C, D, T1) are not evidence of absent signal** —
  they are consistent with under-sampling, and must not be cited as "this region
  has no edit signal".
- **OOC3 is not signal-free and should not carry pseudo-labels.** Measured at a
  20,000-synapse sample: **33 true merge pairs and 173 frankenmerges** — the
  richest region surveyed. `STATUS.md` Phase 2.12 noted "19 frankenmerges
  (partial proofreading)" and an elevated over-merge rate of 0.045 there. The
  out-of-column protocol builds pseudo-ground-truth on "each v117 root = one
  neuron", which those frankenmerges violate by construction. **The elevated
  OOC3 over-merge rate is therefore better explained as a label artifact than as
  a model failure.** OOC1 (4 frankenmerges, 0 pairs) is roughly consistent with
  the assumption; OOC3 is not. In `bench_v1`, OOC3 is used with *real* labels, as
  a training region.
- **Frankenmerges outnumber true merge pairs 5-10x in every region measured.**
  The dominant real v117→v1718 error is a merge needing a split. Much of the
  prior work emphasised the opposite direction.

### Corrections to my own work, recorded

- I initially transcribed **T3 and T4 with region E's x-range**. Their real
  extent is x = 1,150-1,350k (`scripts/spatial_variance.py:308-331`). The
  identical E/T4 statistics this produced looked like a train/test leak in the
  existing code; it was my transcription error. Corrected and re-surveyed.
- Two waiver comments I inserted were indented to match a *suffix* of the target
  line, silently de-indenting it and breaking `neuronauts/dataset_builder.py`
  and `scripts/train.py`. Caught by the test suite: baseline is 6 failed / 949
  passed / 1 error; the broken tree was 52 failed / 825 passed / 4 errors. Fixed,
  and every file in the repo now parses.
- `scripts/lint_provenance.py` crashed on paths outside the repo, so linting a
  file by absolute path silently failed. Fixed and covered by a test.

### Two findings about existing live code

- **`scripts/train.py` built a flat random split** over box records with no test
  set at all (`rng.shuffle(all_records)`). Because a cortical arbor spans many
  boxes, this puts the same neuron in train and val. It now fails closed behind
  `--allow-random-split`, which prints a warning that its numbers may not be
  reported.
- **Region B, a training region throughout Phases 2.7-2.12, showed 0 true merge
  pairs** at a 20,000-synapse sample (C and D likewise). Given the sampling
  caveat above this is not proof of absence, but it does mean the merge signal
  those phases trained on was far thinner than the region count suggests.

### The dataset

`data/bench_v1/` — see its README for the full contract.

| Split | Regions | Observations | v117 roots | True merge pairs | Frankenmerges |
|---|---|---:|---:|---:|---:|
| train | OOC3, P1a, A, E | 22,369 | 4,946 | 43 | 505 |
| val | P1b | 5,741 | 1,311 | 17 | 177 |
| test | P1c | 7,483 | 1,617 | **49** | 154 |

Root-disjoint (dedup removed 289 roots from train, 317 from val), every
cross-split seam ≥ 25,000 nm, independently verified by `scripts/verify_split.py`.
Manifest sha256 `f4185886e9137c61c0fde2ed26f14a76171d8c8e3a57d2cc7d40b895311c5c87`.

**The gates are not decorative.** The first build attempt put OOC3 in test and
left train with 11 merge pairs against a required 20; it aborted rather than
write the dataset. Moving OOC3 to train fixed the assignment — and incidentally
made train→test a genuine cross-region test, since OOC3 sits ~140 µm from P1c.

### Step 4 — what to run next

1. Trivial baselines first, to establish the floor: untouched-v117 (predict no
   merges), spatial-proximity union-find, same-L2-component.
2. Then the clean-track models (EdgePartitionGNN + edge_cc at the Phase 2.11
   configuration), trained on train, calibrated on **val only**, one run on test.
3. Stamp every number with `neuronauts.results_schema.ResultsRecord` carrying
   the manifest hash above.
4. Rewrite `EXPERIMENT_LOG.md` from those results; move the old log to
   `docs/history/experiment_log_retracted.md`.

Optional and costed, not blocking: deepen coverage beyond the 20,000-synapse
sample using many narrow tiles at ~20k each (~17 min/region), or raise the 300s
timeout in `lineage.py`.

### Open decisions for the project owner

1. **Token rotation.** The committed token is purged from source, but the value
   in the environment is the same one. Rotate it outside this repo.
2. **External correction.** Noted that the bad results were caught internally and
   not read by anyone outside, so no external correction appears necessary; the
   retraction notices stand as the internal record.
3. **`bench_v2` scope** — whether to move to label version 1822 and/or full
   synapse coverage.
