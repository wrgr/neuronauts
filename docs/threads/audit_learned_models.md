# Audit — every learned model in the repository, against the frontier task

*Written 2026-09-02 against `f781db4d6`. Every claim below is either (a)
something I ran and can show, or (b) quoted from a file with its path, and
labeled as quoted. Where I could not verify a number I say so rather than
repeating it as fact. The repository advanced during this audit — EXP-082–084
and the frontier-precision conversion in `docs/PROGRAM.md` all landed while it
was running — and §1, §3 and §5 are reconciled against them.*

## 0. The one-minute answer

**Sixteen checkpoint files sit on disk across roughly two dozen model classes.
Eight of those checkpoints were fitted on real segmentation — one of them
carrying a frozen encoder that was itself fitted on software cuts — and eight
on skeletons cut in software. Two trained models carry a real-substrate number that survives
scrutiny — the EXP-063 frankenmerge detector and the level-2 partition
checkpoints — and neither is aimed at the frontier task. Nothing in the
repository, old or new, beats hand-built geometry on it.**

I trained a new one to find out. On the 99 corrected panels, under
cross-validation held out by cell, a small graph neural network over the
panel's candidate set scores **top-1 on 17–18 of 66** against the hand-built geometry
product's **22 of 66** (Wilcoxon on ranks p = 0.125 — the two are not
distinguishable at this sample size, and neither is a logistic regression at
17). No learned model beat geometry. Two diagnostics say why, and they point in
an unwelcome direction:

- **The model can express a better ranking.** Fit on all 99 panels and scored
  in sample, the same network reaches top-1 **46 of 66**, median rank 1.0. It
  is not capacity-limited.
- **But more labels do not close the gap.** Quadrupling the training set from
  ~15 joins per fold to ~60 moves held-out top-1 by 19 → 19 → 18. The learning
  curve is **flat**.

Together those say the network memorizes each cell and transfers nothing. The
fourteen hand-built geometric features appear to carry approximately all the
transferable signal that is in them, and the hand-built product already
extracts it about as well as a fitted model does. **The lever is a new feature
or a new substrate, not a better architecture and not more panels of this kind.**

The highest-expected-value learned asset for the frontier is one that has never
been pointed at it: **the EXP-063 frankenmerge detector** (held-out area under the receiver operating
characteristic curve of 0.958 on real v117 objects). EXP-081 already named this gap — the detector "has never
been applied to a proposed join, only to existing objects." It is a precision
instrument and precision is what binds. It is also cheap: it refits in two
minutes and its features do not come from the eroded mip-5 clouds.

---

## 1. What I verified myself

| Check | How | Result |
|---|---|---|
| Hand-built geometry baseline reproduces | Re-derived `along × collin × exp(−gap/500) × min/max caliber` straight from `data/external/panels/*.npz` | **Reproduces**: median rank 6.0, top-1 22/66, top-5 31/66, top-20 44/66 vs the reported 5 / 22 / 31 / 44. The median differs by one rank through tie handling; every top-k count is exact. My distance row also recomputes to 0 of 66 top-1 against a published 2 of 66. **Independently corroborated**: `scripts/audit_frontier_precision.py`, committed in `f781db4d6` while this audit was running, reports the same two corrections from a separate recomputation. |
| My scoring harness is correct | Scored the panels using only the hand-built product as a single feature column | Identical to the baseline, to the count. The harness is not the source of any negative below. |
| `attic/morpho_grammar` engines load no checkpoint | `grep -c "torch.load\|load_state_dict\|joblib.load\|open("` over all 26 files | **Zero hits in every file.** Not one engine performs any file I/O at all. |
| Two engines read the answer | Read `geodesic_em_tracer.py` and `active_gap_oracle.py` directly | Confirmed, see §4. |
| The two best checkpoints are half untrained | Read `attic/prior_results/train_l2_partition.py:284` | Confirmed: `encoder = FragmentEncoder(**enc_kwargs)  # kept for checkpoint compatibility`, and `train_fragment_encoder` is imported and never called. |
| Checkpoint → class mapping | `torch.load` on `gat_skeleton_50e.pt`, `neuronauts_l2_partition.pt`, `cell_gnn_seg.pt` | Confirmed shapes, parameter counts and stored `init_kwargs`. |
| EXP-063 is not built on the eroded clouds | Traced its feature sources | Confirmed: polarity from synapse counts, shape from synapse centers, topology and geometry from **level-2 node positions** (`neuronauts/harness/objgeom.py`), never from `object_clouds_mip5.npz`. **The substrate warning does not void EXP-063.** |

The claim I was asked to check — "25 of 26 engines contain no checkpoint-loading
code" — is **true but understated**. The correct statement is 26 of 26 files
(25 engines plus an empty `__init__.py`); the phrasing implies one engine does
load a checkpoint and none does. The intended exception is
`pcfg_morphology.py:79 fit_from_skeletons()`, which counts forks and terminals
in process — real fitting, but not a checkpoint.

---

## 2. Per-model table

Substrate column: **REAL** = fragments of the actual MICrONS segmentation.
**CUT** = skeletons cut in software, principal-component half-splits, spliced paths or
same-object z-cuts, where both halves of a "split" share geometry, caliber and
tangent by construction. **NONE** = no learned parameters.

### 2A. Real substrate, trained, number stands

| Model | Where | Checkpoint | Substrate | n | Headline claim | Reproduces? | Recommendation |
|---|---|---|---|---|---|---|---|
| **Frankenmerge detector** (gradient-boosted stumps + logistic regression) | `neuronauts/harness/baselines.py:210,285`, run by `neuronauts/experiments/exp063_frankenmerge_detection.py` | **No — refits every run** (2 min) | REAL v117 objects, spatial train/validation split with a 10 µm buffer | 922 train positives, 259 strict negatives, 20,826 objects | held-out area under the receiver operating characteristic curve (AUC) **0.958**; polarity alone 0.914; size-only control 0.654 | Not re-run here; the stamped `results/EXP-063/result.json` matches the claim and the split is leak-guarded | **Rank 1. Apply it to proposed joins.** See §5. |
| **EdgePartitionGNN** (+ `HalfSynapseGNN` backbone) | `neuronauts/assemble/edge_partition.py:48`, `partition_gnn.py:136` | `models/neuronauts_l2_partition.pt`, `_xregion.pt` — 90,081 trained params, plus **66,784 deliberately untrained** encoder params | REAL v117 → v1718 lineage labels | 3–5 cortical regions, 50 µm seam buffers | in-column merge precision ≥ 0.997, adjusted Rand index ≥ 0.80; out-of-sample merge precision 0.951 (quoted from `models/README.md`) | Not re-run. **No live trainer** — the only trainer is `attic/prior_results/train_l2_partition.py` | Keep as a reference implementation. Not applicable to the frontier: it consumes a synapse or level-2 world, not a tip's candidate set. |
| **EdgePartitionGNN, bench_v1 evaluation** | branch `origin/claude/synthetic-data-quality-review-9ihg5x`, `results/bench_v1/RESULTS.md` | None produced (`"checkpoint_sha": null`) | REAL — the stamped record says `"synthetic": false`, base v117 → labels v1718 | test region 20,000 observations, 12,287 fragments, 703 true cross-fragment pairs | **adjusted Rand index 0.0017 against a do-nothing floor of 0.9610**; 22.2 M joins for 310 correct | Numbers read from the branch's stamped JSON | **This is the honest verdict on the family: it loses to the untouched segmentation by three orders of magnitude.** Do not revive without a precision mechanism. |
| **CellGNN** | `neuronauts/cell_graph.py:754` | `cell_gnn_seg.pt` (64,992 params), `cell_gnn_5feat.pt`, `cell_gnn_real.pt` | REAL synapses, real v1412 root labels | **37** cached 30 µm boxes, 50 epochs | test line-graph F1 (harmonic mean of precision and recall) **0.272** at t = 0.99 | Not re-run; the 651-file `data/boxes_30um` cache is on disk, so it is reproducible | Different task (synapse-to-cell assignment inside one box) and structurally capped by the box. Not applicable to the frontier. |
| **Seam detector** (`SAGE`), **atomicity detector** (`EdgeConv` + RandomForest) | `experiments/pcfg/seam_detector.py:104`, `atomicity_detector.py:57,93` | **None persisted** | REAL v117; v1718 used only to define the seam, never as input | seam: 140 merge objects, grouped-by-cell cross-validation; atomicity: 9,377 objects, 354 false merges (3.78%) | — | Not run here | Same shape of problem as EXP-063 and less well measured. Low priority while EXP-063 exists. |
| **Cut-face encoders, real fine-tunes** | `experiments/fingerprints/cutface/train_real_cutface.py`, `train_band_encoders.py` | `cutface_encoder_real.pt`, `_bio.pt`, `_art.pt` (25,827 params each) | REAL v117 split sites | **5 training pairs**; **14 training pairs / 12 test sites** | precision 1.0 at 11% coverage (quoted secondhand in `docs/tree_assembly_handoff.md`) | Not verified — and `docs/threads/experiment_survey.md` already flags the thread's own README reporting different numbers | Sample sizes of 5 and 14 cannot support a claim. **Treat as a pilot, not a result.** Two of these checkpoints were written by a source version that no longer exists. |

### 2B. Cut substrate — the numbers measure an easier task

Every checkpoint in this block was trained on halves of one neuron. Both halves
share caliber, tangent and arbor statistics because they were one object a
moment earlier. `docs/threads/grammar.md` already carries this warning for the
grammar thread; the code confirms it and extends it to the path encoder, the
graph attention head and the main cut-face encoder.

| Model | Checkpoints | The cut, in code | Headline claim | Status |
|---|---|---|---|---|
| **SharedGrammarModel** (Transformer path encoder + merge scorer) | `grammar_cave_real_50.pt`, `grammar_synthetic.pt`, `shared_grammar_real.pt`, `shared_grammar_raw_skel_50e.pt`, `shared_grammar_raw_skel_gat50e.pt`, `shared_grammar_root_neighborhood_run001.pt` (133,992 params each) | `neuronauts/merge_dataset.py:56 _split_same_root` — one real root's synapse cluster sorted along its principal component axis and cut at the midpoint | val merge accuracy **87.2%** | **Void for this task.** `docs/threads/grammar.md` marks it superseded; EXP-060/060B/061/070 measured the real-substrate version at ~0.09% precision. |
| **GlobalAssemblyGAT** | `gat_skeleton_50e.pt` (23,041 params, `node_dim=32`, 2 layers) | trained over the grammar embeddings above; its second trainer, `train_global_assembly_gat`, says "synthetic ConnectivityGraph examples" in its own docstring | **none — the metric column is a dash in both `models/README.md` and `docs/threads/grammar.md`** | **Never evaluated.** There is no `results/` record and no metrics sidecar. It never had a number to supersede. The closest measured relative, EXP-053A on `shared_grammar_raw_skel_gat50e.pt`, found precision 0.000026 at the last high-recall threshold and recall 0 at the first non-collapsing one. |
| **PathEdgeEncoder** | folded into `cell_gnn_real.pt` | `neuronauts/path_dataset.py:1-11` — negatives are "splice negatives … concatenating chain segments from two different cells" | — | Cut substrate. The CAVE edit-history augmentation that would have made it real is documented as returning zero synapses on this dataset. |
| **SkeletonSynapseNet** | none | `experiments/pcfg/_fragment_neuron` cuts; `HOLDOUT_RESULTS.md:33`: "No proofreading is needed — the supervision is the synthetic cut itself" | — | Cut substrate, no checkpoint. |
| **CutFaceEncoder** (base) | `cutface_encoder.pt` | positives are two z-planes of the **same** segment id | — | Its own successor script says it "is trained on artificial z-cuts and does not transfer to the oblique, messy cross-sections at real split errors." |
| **SynapseCoassigner**, **SkeletonGNN/FragmentEncoder** | none tracked | `treestitch/data.py:8-10` splits each real skeleton into N pieces; `scripts/coassign_demo.py` additionally invents synapses by jittering skeleton vertices | — | Cut substrate in the demo path. A real path exists (`scripts/v117_coassign.py`) but produced no tracked checkpoint. |

An independent negative worth recording, from
`origin/claude/segclr-fuser-grammar-x8ba3x`: a learned identity discriminator
trained with hard negatives on **real** confusable joins (n ≈ 194) scored an area under the curve of
**0.447** — below chance — against geometry's 0.636, with the sanity control
passing. That branch's own reading is that "training extracted no identity
structure." It agrees with EXP-080's finding that SegCLR embeddings select the
true continuation on 0 of 44 panels where geometry selects 22.

### 2C. Named like models, not learned

| Thing | Where | What it actually is |
|---|---|---|
| **`AsymmetricRelationalModel`** | `neuronauts/global_merge/represent/asymmetric_relational_gnn.py:24` | 36 bilinear matrices from an orthogonal draw plus hand-set diagonal biases (0.8/0.7/0.6) and a fixed weighted sum. **No `fit`, no optimizer, no checkpoint.** Its only callers are in `attic/`. |
| **`attic/morpho_grammar/` — 25 engines** | | **Zero file I/O of any kind in any file.** `tree_grammar_infiller.py:36-43` builds a random embedding table, a random projection and two random-orthogonal attention matrices in `__init__` and nothing ever updates them; **eleven engines inherit those matrices**. Their scores measure the task, not the method. The grep for `nn.Module` finds nothing because this is a hand-rolled numpy transformer. |
| **`VICRegSkeletonModel`, `AttentionArborValidator`, `TreeDNAEncoder`, `PathGrammarReranker`** | `global_merge/represent/vicreg_gnn.py`, `topology_model.py`, `represent/dna.py`, `represent/path_grammar.py` | Real architectures, **no live trainer and no checkpoint**. Every caller is in `attic/` or the test suite; `PathGrammarReranker` has no trainer anywhere. |

---

## 3. The new measurement — a model trained on the 99 corrected panels

**Setup.** 99 panels, one per cell: 66 with a continuation (exactly one correct
object each, median 2,440 candidates) and 33 genuine interior arbor terminals
with no correct answer. Eleven folds over panels, so no cell is ever in both
the fit and the score. Fourteen per-candidate features, all derived from the
panel and all available to a grower: gap, proximity, along-axis, collinearity,
log voxel count, candidate caliber, the caliber ratio, the panel-relative gap
rank, two products including the hand-built product itself, and two panel-level
terms (seed caliber, the seed's own taper `end_ratio`).

**The model.** One round of message passing over the panel's candidate set:
each candidate encoded, the panel summarized by mean and max pooling over its
candidates, every candidate rescored with that summary attached — a fully
connected panel graph, one hop. A learned STOP slot scores "extend nothing
here" from the panel summary and the seed's own end shape. The loss is a
listwise softmax over the candidates plus STOP: for a cut panel the target is
the true partner, for a terminal panel it is STOP. That is the decision a
soma-seeded grower actually makes. Training samples 400 competitors per step
(the 200 hardest by the hand-built product plus 200 at random); **evaluation
always scores the full panel**.

### Ranking, held out by cell — 66 panels, 1 correct in ~2,440

| score | median rank | top-1 | top-5 | top-20 |
|---|---:|---:|---:|---:|
| **hand-built geometry** | **6.0** | **22/66** | 31/66 | **44/66** |
| logistic regression | 6.0 | 17/66 | 31/66 | 44/66 |
| gradient boosting | 12.5 | 5/66 | 25/66 | 37/66 |
| panel network (1 hop) | 8.0 | 17/66 | 31/66 | 39/66 |
| *panel network, in sample* | *1.0* | *46/66* | *60/66* | *65/66* |

Wilcoxon on the paired ranks: panel network vs geometry p = 0.125, logistic
regression vs geometry p = 0.533, gradient boosting vs geometry p = 0.075.
**None of the learned scorers is distinguishable from hand-built geometry at
this sample size, and none beats it.** Re-running the panel network under a
different random path gave top-1 18 rather than 17, so read these counts with
about ±1–2 of run-to-run noise — which is smaller than the gap to 22 but not by
much.

### Commit or abstain — all 99 panels

A panel is *committed* when its top score clears a threshold; it counts correct
only if the panel has a continuation **and** the top object is that
continuation. Best F1 over all thresholds:

| score | best F1 | precision | recall | panels committed |
|---|---:|---:|---:|---:|
| **hand-built geometry** | **0.299** | **0.294** | 0.303 | 68 |
| panel network + STOP slot | 0.254 | 0.267 | 0.242 | 60 |
| panel network (1 hop) | 0.254 | 0.250 | 0.258 | 68 |
| logistic regression | 0.225 | 0.211 | 0.242 | 76 |
| gradient boosting | 0.127 | 0.159 | 0.106 | 44 |

The learned STOP slot buys a little precision over the same network without it
(0.267 vs 0.250) at a cost in recall. It does not reach geometry.

### Why it does not generalize

| training joins per fold | median rank | top-1 | top-5 | top-20 |
|---:|---:|---:|---:|---:|
| ~15 (a quarter) | 8.0 | 19/66 | 30/66 | 40/66 |
| ~30 (a half) | 10.0 | 19/66 | 29/66 | 39/66 |
| ~60 (all) | 7.5 | 18/66 | 31/66 | 42/66 |

**Flat.** Four times the labels moves nothing. Set beside the in-sample top-1
of 46/66, the diagnosis is not ambiguity: the network fits each cell and
transfers none of it. That is the signature of features that carry the
information a fitted model needs to memorize a panel but not the information
that is common across cells. **More panels of this kind will not fix it.**

### Translated to the frontier

`docs/PROGRAM.md`, updated in `f781db4d6`, gives the conversion this section
needs: at the real frontier composition of 46 cut ends per cell with one live
site, geometry's top-1 of 22 of 66 is **0.33 correct joins against roughly 45
false ones per cell — about 135 false per correct, a precision of 0.007.**
Applying the same conversion, the panel network's 17–18 of 66 lands at 0.26–0.27
correct joins per cell against the same ~45 false, which is worse, not better.

**No model in this table — hand-built or learned — has an operating point.** The
question the ranking table answers is which candidate is right once you are
already at a live site; at the frontier that condition holds 1.6% of the time,
and the panels sample only the 1.6%.

### What this measurement is not

The 99 panels are centered on the known seed/target contact, which is exactly
the conditioning EXP-081 says "cannot estimate" the frontier. The 33 terminal
panels make the commit/abstain test a roughly 2:1 mix of live sites and dead
ends; the frontier is about 1:62. **Every number in §3 is therefore an upper
bound on frontier performance, mine included.** Reproduce with the two scripts
recorded in §7.

---

## 4. Three things that inflate old numbers upward

The prior audits stopped at "untrained," which pushes scores *down*. These push
them *up*, and I read all three directly:

1. **`attic/morpho_grammar/geodesic_em_tracer.py` fabricates its evidence from
   the answer.** Its docstring claims it traces three-dimensional electron-microscopy voxel volumes; it receives
   no image data. It takes `is_true_continuation: bool` — the ground-truth
   label — and branches on it to draw the "measured" gradient from two
   different distributions (true → `rng.uniform(0.70, 0.95)`, false →
   `rng.uniform(0.15, 0.40)`). `cajal_geodesic_dual_engine.py:86` and
   `dual_engine_infiller.py:78` both feed the ground-truth id into it.
2. **`active_gap_oracle.py:86-89` returns the ground truth directly**
   (`return gt_target_id, 0.99, 1`) with 98% probability, wired to a real
   ground-truth map by `hierarchical_dual_santiago.py:167`.
3. **`ultrastructural_texture_prior.py` measures no texture** — its "vesicle
   density / mitochondrial cristae" signature is a lookup keyed on the
   already-known cell type.

Also: `autoproof_baseline.py` and `neurd_baseline.py` are roughly 50-line
homemade heuristics, not the published AutoProof or NEURD software whose names
they carry. Any comparison against "AutoProof" or "NEURD" in this repository is
a comparison against those 50 lines.

---

## 5. Ranked by expected value on the frontier

**1. Apply the EXP-063 frankenmerge detector to *proposed* joins.** *(high
value, ~1 day)* It is the one trained model in this repository whose number is
real, held out, on real segmentation, and not built on the eroded mip-5 clouds.
Its features are computable on a *hypothetical* object: take the seed's
synapse cloud, polarity, level-2 topology and level-2 node geometry, union in
the candidate's, and score the union. A join that would create a frankenmerge
gets vetoed. The frontier's binding constraint is precision — a 5% per-tip
false-extend rate yields ~2.3 false joins per correct one — and this is the
only precision instrument on the shelf. EXP-081 named the gap; nobody has
closed it. **First, persist the model**: it currently refits every run and
saves nothing, so nothing downstream can call it.

**2. Stop trying to learn a better ranker on these panels.** *(saves time)* The
flat learning curve plus the in-sample 46/66 is a clear enough signal. A
learned scorer over these fourteen features lands where the hand-built product
already is. Retire this line unless a new feature arrives.

**3. If a learned scorer is attempted again, change the input, not the model.**
The features that would plausibly transfer and are absent: the seed's own taper
profile as a curve rather than one `end_ratio` scalar; the candidate's shape
*beyond* the contact rather than at it; and cross-sectional shape agreement at
the contact face. Note that the third has been tried on real sites and failed
(area under the curve 0.447–0.58) — but at n = 194 and with 5–14 training pairs, which is not a
sample size that settles anything.

**4. Do not revive the grammar, graph-attention or edge-partition checkpoints
for this task.** *(negative recommendation)* Six grammar checkpoints and the
graph attention head were fitted on software cuts; the graph attention head has
no evaluation at all; the edge-partition family's one honest real-substrate
evaluation loses to doing nothing by three orders of magnitude. None of them
consumes anything a grower has at a frontier tip.

**5. Retire or relabel `attic/morpho_grammar/`.** *(hygiene, hours)* Twenty-five
engines with no parameters, three of which read the label. `neuronauts/morpho_grammar/__init__.py`
is a live import shim pointing at them, so they are still reachable from the
package. At minimum the three label-reading engines should carry a header
saying so, since their scores appear in `EXPERIMENT_LOG.md`'s tables.

**6. Fix the two documentation contradictions.** `cell_gnn_real.pt`'s stored
config (`edge_scoring=True`, a pretrained path-encoder transformer in the state
dict) contradicts its own metrics sidecar, which reports contrastive metrics and
calls it a baseline that uses no electron-microscopy imagery. And `models/README.md`'s pointer for its two
best-performing checkpoints is a trainer that lives in `attic/`.

---

## 6. Limits of this audit

- I re-ran the geometry baseline and trained the new model; I did **not** re-run
  CellGNN, the level-2 partition checkpoints, EXP-063, or any fingerprint
  training. Those rows quote stamped artifacts.
- The bench_v1 numbers come from a branch, read through `git show`. I did not
  execute that code. Its RESULTS.md prose reports training figures (edge
  accuracy 0.953) that appear in no stamped artifact and are therefore not
  independently checkable.
- The `origin/codex/…` branch carries large trained-model claims (EXP-020
  synapse precision 0.9544, EXP-026 88.33% top-3). **I did not verify any of
  them**; they belong to the semi-synthetic era that `docs/threads/grammar.md`
  and the EXP-060–072 series already mark superseded.
- My panel result rests on 66 positives. A difference of 22 vs 18 at top-1 is
  not significant at that size, so "geometry wins" should be read as "nothing
  learned has beaten geometry," not as "geometry is better."

## 7. Reproducing §3

Everything is in `data/external/audit_learned_models/` (gitignored, not
committed), with the log of each run beside it. No network access; the scripts
read only `data/external/panels/*.npz`.

```bash
.venv/bin/python data/external/audit_learned_models/baseline.py   # ~5 s   the geometry ladder
.venv/bin/python data/external/audit_learned_models/train_panel.py # ~4 min  the cross-validated table
.venv/bin/python data/external/audit_learned_models/diag.py        # ~3 min  sanity + in-sample capacity
.venv/bin/python data/external/audit_learned_models/curve.py       # ~12 min the learning curve
```

The baseline formula, `along × collin × exp(−gap/500) × min(cal)/max(cal)`, is
taken verbatim from `scripts/exp079_evaluate.py` so the comparison is against
the incumbent as written, not a re-derivation of it.
