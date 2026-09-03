# Audit: treestitch and cell assignment, against a 1.6%-base-rate frontier

*Opened 2026-09-02. Read-only over `treestitch/`, `docs/threads/cell_assignment.md`,
`neuronauts/cell_graph.py`, `neuronauts/assembly.py`, `neuronauts/line_graph.py`,
`scripts/*coassign*`, `docs/tree_assembly_algorithm.md`, `docs/tree_assembly_handoff.md`,
`docs/p1_benchmark.md`, `docs/lineage_approach.md`, plus new measurements in
`scripts/audit_frontier_precision.py`. Nothing was committed.*

Treestitch and cell assignment both put fragments into cells, which is the
soma-seeded task under another name. They are the closest prior art we have.
This asks one question of each asset: **does its claim survive contact with the
task as EXP-081 measured it?** Answers are marked **(verified here)** when I
recomputed them in this session and **(read)** when I am reporting a document's
own number.

## The bar

`results/EXP-081/evaluation.md`: a soma-seeded grower faces a **frontier** —
every cut end of the cable it has claimed. Over 40 cells: 2,137 tips, 34 live
extension sites, **base rate 1.6%**. Precision binds. A 5% per-tip false-extend
rate gives about 2.3 false joins per correct one; for a better-than-even chance
of no false join on a cell, the per-tip false-positive rate must sit **below
roughly 2%**.

Three consequences, and they are the audit's test:

1. **Accuracy, AUC and F1 do not answer it.** They are balanced-set or
   prevalence-independent statistics. Precision at a realistic candidate
   prevalence does.
2. **A metric measured only where a join exists does not answer it.** That is
   the 1.6%; the other 98.4% is the decision "nothing continues here".
3. **A candidate set the experimenter constructed does not answer it.** If the
   denominator is a hand-built graph's edges or a curated panel, the base rate
   is a design choice, not the tissue's.

## 0. The structural fact that frames everything below

**The treestitch/assignment code and the corrected substrate are disjoint code
lineages.** (verified here) Grepping `object_clouds_mip5|objects_v117_mip5_svmap|
substrate/c100um` across `neuronauts/`, `treestitch/` and `scripts/` returns 29
files — every one of them an `exp05x`–`exp07x` experiment or a `build_*`/`exp079*`
script. **Zero hits** in `neuronauts/cell_graph.py`, `assembly.py`,
`line_graph.py`, `scripts/*coassign*.py`, or anywhere in `treestitch/`.

The prior art runs on a different data plane (CAVE `BoxCache`, live chunkedgraph)
than the panels the substrate fix was applied to. So no treestitch or
cell-assignment number was ever measured on corrected geometry — not because
anyone chose the wrong substrate, but because these two bodies of work have
never touched.

## 1. Inventory

`compat` = is the claim comparable to a 1.6%-base-rate frontier decision?

### 1a. Treestitch proper

| asset / claim | substrate | n | compat |
|---|---|---|---|
| Two-level stitch, ΔARI **+0.031**, Δmerge_P **+0.004**, assembly 0→100% | **synthetic worlds**, self-labelled "→ RUN, synthetic worlds" | 24 objects × 4 pieces, 2×2 tiles, 3 seeds | **no** — generated data; ΔARI has no decision site |
| Same on real data, ΔARI **+0.132**, Δmerge_P −0.105, assembly 0→50.6% | **real** v117→v1718, 200 µm box | 288 fragments, 312 neurons, 19 real frankenmerges | **no** — whole-graph delta; the doc itself says this table's merge_P is all-pairs and "not directly comparable" to the rest |
| Atomization lifts stitched ARI **0.670 → 0.957**, frankenmerge separation 0.429 → 1.000 | **synthetic** | 7 synthetic franken fragments | **no** — and **reversed on real data**: the same treatment takes the real 600 µm box from ARI 0.221 → **0.062** and merge_P 0.75 → **0.26** |
| **Geometry-only endpoint stitch precision 0.17–0.67** | **synthetic** | 24 objects × 4 pieces | the only candidate-level precision in the treestitch line — and it is on generated data. Superseded by §3 |
| Stitching improves frankenmerge separation | real | 96 real frankenmerges | **contradicted**: fk_sep 0.54–0.57 baseline → **0.42 stitched** |
| `merge_P` **0.95–0.999** across edge_cc runs | **real**, one bbox | 15 / 110 / 24 neurons; 0 / 5 / 0 frankenmerges | **no.** The denominator is the constructed observation graph, whose type-0 component is *all pairs inside one v117 root* — and `lineage_approach.md` records those as **99.2% true-merge by construction**. High precision on a denominator that is 99% pre-answered |
| `fk_split` in-sample **0.73–1.00**, leak-fixed out-of-sample **0.000** | real | 64 fragments, 386 synapses, **4** frankenmerges | the negative is real and worth keeping; n=4 |
| Union-find collapses at 110+ neurons (ARI 0.007) | real | 110 neurons | **no**, but a useful structural fact |
| "false merge ≈ 5–10× a missed merge" | **not measured** — attributed to community consensus | — | n/a |

### 1b. The p1 benchmark — the strongest claim in the corpus

| asset / claim | substrate | n | compat |
|---|---|---|---|
| Cross-region held out: ARI **0.652**, merge_P **0.998**, over_merge **0.001** | **real**, trained on 5 regions disjoint from P1 | **172 held-out fragments** | **partly.** Alone in this corpus it *reconciles same-fragment co-membership away*, so the trivially-positive type-0 edges leave the denominator and only cross-fragment decisions are scored. Still a constructed spatial k-NN candidate set with no "nothing continues" decision — but it is the only merge_P here not inflated by construction |
| Encoder v2 lifts ARI **0.581 → 0.606** | same | 172 fragments, **single seed** | **no** — the same body of work records fixed-seed run-to-run variance of 310 vs 694 clusters on one tile and concludes multi-seed reporting is needed for small deltas |
| On L2 nodes, **19%** of neurons need ≥2 merges (vs 1% on synapses) | **real**, ground truth only | 937 / 1,160 fragments | **yes, and load-bearing** — the honest statement of how much merging the task requires, and why the synapse substrate hides it |

### 1c. Cell assignment — the default pipeline

| asset / claim | substrate | n | compat |
|---|---|---|---|
| CellGNN held-out test **F1 0.272** @ t=0.99, "the pipeline that runs by default" | **real** CAVE root ids, 30 µm boxes | **37** boxes | **no.** The unit is a synapse pair in a line graph, not a tip; and the thread page states the ceiling is structural — a neuron larger than a 30 µm box cannot be assembled. It cannot express a frontier decision. See §2 for two things wrong with the number itself |
| `cell_gnn_5feat` 0.269, `cell_gnn_real` 0.264 | same | same | **no** — same unit; the 0.264–0.272 spread is inside the noise of a 37-box sample |
| Per-feature ablation: "all 6 deltas within ±0.01 mean F1 — scalar features largely redundant" | same | same | the ablation checkpoints were curated out, so this is unverifiable — but it says the hand-built edge features carry little signal |
| `scripts/v117_coassign.py`, `coassign_demo.py` (correlation-clustering frontier) | `coassign_demo` is **semi-synthetic** — real skeletons bisected in software, synapses scattered with Gaussian jitter | 20 neurons × 3 pieces default | **no.** Neither script contains a quantitative claim; both need live CAVE. Branch `claude/synapse-coassign` is 24 commits behind and flagged "needs rebase" in a table stamped 2026-06-29 |

### 1d. Panel-era claims quoted as if they settled the frontier

| asset / claim | substrate | n | compat |
|---|---|---|---|
| "true partner at **median rank 5 of 2,440**, top-1 **22 of 66**" (EXP-077) | **real, corrected** (`agglomerate=True`, mip 2) | 66 cut panels | **no, and the source says so** — panels are centred on the true contact, which a real proposer would not know. §3 converts it |
| Cut-face fingerprint verifier: **panel recall 1.000**, top-1 **0.767**, precision 1.0 at 11% coverage | **real** v117 split sites | 233 panels; **73** held-out sites | **no — the one explicit true-join-sites-only evaluation in the corpus, and it labels itself so.** It drops any site whose true partner is absent, and **172 "not-a-split" sites were flagged and discarded rather than counted as "nothing continues"**. Those 172 are exactly the frontier's negatives, thrown away |
| PCFG cross-region holdout **AUC 0.816**, recovers 60% of merges | **semi-synthetic** — real skeletons, breaks introduced in software at chosen weak points | 260 pairs, 65 positive | **no** — 25% positive rate; its own source says "do nothing" scores 75% accuracy |
| Tree-DNA half-skeleton identity **AUC 0.829** | **semi-synthetic** — real skeletons bisected in software | 40 cells, one type | **no** — balanced pair AUC, and random init already scores **0.768**, so the trained lift is **+0.061**. Quoted beside the real ARI 0.752 without the substrate distinction |
| Grammar "pairwise merge accuracy ~85–87%" | in-sample cross-validation on a **curated candidate panel** | 85 boxes | **no** — already flagged not-evidence in `attic/README.md`; and the number itself disagrees across files (§2) |
| EXP-063 frankenmerge detection, held-out **AUC 0.958** | **real** | — | **not yet** — a detector on *existing* objects. EXP-081 is right that it has never been applied to a *proposed* join, which is where the frontier needs it |
| Proximity candidate generation fails at ~**0.09%** precision (EXP-060/060B/061/072) | **real data, broken geometry** — all four on eroded mip-5 centroid clouds | — | the *shape* survives (realistic imbalanced precision, every proposed pair scored); the number needed re-running. §3 re-runs one part of it |

## 2. What does not reproduce

### 2a. The distance row of EXP-077's ranking table (verified here)

Recomputed directly from `data/external/panels/*.npz`:

| feature | published (EXP-077) | recomputed |
|---|---|---|
| **distance** | 60, **2/66**, **4/66**, **12/66** | 60.5, **0/66**, **0/66**, **4/66** |
| along-axis | 56, 2/66, 10/66, 24/66 | 57.5, 2/66, 10/66, 24/66 |
| along × collin | 30, 11/66, 18/66, 29/66 | 30.5, 11/66, 18/66, 29/66 |
| along × collin × proximity | 12, 12/66, 25/66, 35/66 | 12.5, 12/66, 25/66, 35/66 |
| **× caliber** | **5**, 22/66, 31/66, 44/66 | **6.0**, 22/66, 31/66, 44/66 |

Every geometry row reproduces exactly on top-1/5/20. The medians sit
consistently 0.5–1.5 higher, and the headline **median rank 5 is 6.0** on the
panel files. Both point the same way: the published table used a smaller
candidate pool than the panels hold — `scripts/exp079_evaluate.py` describes its
input as "~160,000 candidates" while the 99 panel files hold **217,087**, and
dropping candidates can only lower ranks. **I have not proven that is the
cause.** I could not check: `scripts/exp079_evaluate.py` reads
`data/external/exp079_panel_tree.json`, **that file does not exist**, the script
raises `FileNotFoundError`, and `results/EXP-079/` is an empty directory.

The distance row diverges most because distance is the row most sensitive to
pool size — for a reason in §3.

### 2b. F1 0.272 is the number at a tuned threshold, not the default (verified here)

`models/cell_gnn_seg.metrics.json` reports both:

```
test_f1_mean_at_t0.99 : 0.2722   <- the number quoted everywhere
test_f1_mean_at_t0.5  : 0.1976   <- the same model at the code's default threshold
```

The advertised figure is **38% higher** than the same checkpoint at
`CellGNNConfig.partition_threshold = 0.5`. Both are in the sidecar; only one is
in `docs/threads/cell_assignment.md` and `models/README.md`. It also used **37
of the 325 boxes** now in `data/boxes_30um/`, on checkpoints dated May 1.

### 2c. The PathEncoder merge accuracy disagrees with itself

`models/README.md`: "val merge acc **87.2%** … 40 real CAVE boxes".
`experiments/pcfg/README.md`: "PathEncoder merge acc = **0.856** (85 boxes)".
Different number *and* different box count for the same claim, which is already
flagged elsewhere as in-sample. It should not be cited in either form.

### 2d. Ledger gap (verified here)

`results/RESULTS.md` says "a run without a row does not exist." Its last row is
EXP-074. **EXP-075, 076, 077, 079, 080 and 081 all have `evaluation.md` files,
none has a ledger row, and none appears in `neuronauts/experiments/registry.py`.**
Every number this audit and EXP-081 lean on is off-ledger.

## 3. The tests I ran, and the numbers

`scripts/audit_frontier_precision.py`. All on the **corrected** panels.

### 3a. The treestitch scoring rule at the frontier

**Asset.** The decision rule at the heart of the whole treestitch line,
`treestitch/stitch.py::candidate_stitch_edges`:

```
score = max(0, 1 − gap/10_000 nm) × (dna_cos + 1)/2
```

No pooled embedding exists for these objects, so the DNA factor is 1 and the
rule reduces to **"nearest object"**. Run head-to-head with the current best
stack (`along × collin × exp(−gap/500) × caliber-agreement`) as a **joint**
decision — extend or decline, and if extend, to which object — with the
per-decision rates converted to EXP-081's composition of 46 tips per cell, 1
live. Negatives are 33 already-whole panels plus the 25 EXP-076-corrected
terminal panels, 58 in all. Thresholds are picked on half the panels and scored
on the other half.

**Ranking, 66 cut panels, median 2,440 candidates:**

| scorer | median rank | top-1 | top-5 | top-20 |
|---|---:|---:|---:|---:|
| treestitch (`1 − gap/10000`) | 60.5 | **0/66** | 0/66 | 4/66 |
| geometry stack | 6.0 | 22/66 | 31/66 | 44/66 |
| random floor | 1,220 | 0.027/66 | — | — |

**The treestitch rule never once picks the true continuation across 66 panels.**
Zero is indistinguishable from the random floor of 0.027 expected hits.

**Joint decision at the frontier composition:**

| scorer | threshold | joins | right | negatives fire | false joins/cell | frontier precision |
|---|---|---:|---:|---:|---:|---:|
| treestitch | any | 66 | **0** | **1.000** | 45.0 | **0.000** |
| geometry | −∞ (always extend) | 66 | 22 | 1.000 | 45.0 | **0.007** |
| geometry | 0.71 | 20 | 11 | 0.190 | 8.5 | 0.019 |
| geometry | 0.79 | 11 | 6 | 0.034 | 1.6 | 0.055 |

**The headline "top-1 on 22 of 66" is a frontier precision of 0.007** — about
135 false joins per correct one, because a scorer with no threshold also fires
at all 45 dead tips.

**Held out, at the criterion EXP-081 pre-registers (per-tip false rate < 2%):**

- **treestitch: no threshold reaches a 2% fire rate at all.** Its negatives fire
  29/29 at every threshold.
- **geometry, t = 0.805: 3 of 33 held-out cut panels joined correctly** (0.09
  correct joins per cell), **0 of 29 negatives fired.** Point-estimate precision
  1.000 — but 0 events in 29 trials gives a 95% upper bound on the fire rate of
  **9.8%**, which is **4.4 false joins per cell** and precision **0.020**. The
  1.000 is not a result.

**Why the treestitch rule cannot decline, mechanically** (verified here). The
panel's minimum gap is **32 nm — one voxel — in the median panel, and a median
of 85 candidates (max 298) tie at exactly that minimum.** The true partner sits
at that minimum in **46 of 66** panels. So distance is not weak because the
partner is far; it is weak because 85 other objects are exactly as near. And
`1 − gap/10000` maps the entire 32–100 nm band where all the action is onto
0.9968–0.9900, so the score saturates: there is always a touching object, the
max score is always ≈0.997, and no threshold on it can ever say "nothing
continues". This is substrate-independent for any monotone function of a gap.

### 3b. Is the corpus big enough to answer the question? No (verified here)

Zero false fires in *n* negatives gives a 95% upper bound near 3/*n*.
Certifying the 2% criterion needs **n ≥ 150** negative decision sites; we have
**58**. The best bound 58 clean negatives support is 5.0%, i.e. **2.3 false
joins per cell** — still a failing grower. EXP-081 already located **2,103 dead
tips**. That is the corpus this needs, and building it is cheaper than any model.

### 3c. EXP-072's dust-floor negative survives the substrate fix (verified here)

EXP-072 found that removing 87% of objects by a physical size floor moved
precision by nothing — on the broken centroid clouds, so `rerun_catalog.md`
flagged it for re-run. Re-run on corrected panels, it holds, and more sharply:

| n_vox floor | candidates kept | true partners kept | top-1 | median rank |
|---:|---:|---:|---:|---:|
| 0 | 100% | 66/66 | 22/66 | 6.0 |
| 100 | 51% | 66/66 | 22/66 | 6.0 |
| **1,000** | **32%** | **66/66** | **22/66** | **6.0** |
| 3,000 | 22% | 63/66 | 22/66 | 5.0 |

A 1,000-voxel dust floor deletes **68% of the panel at zero cost and zero
benefit** — every true partner survives and not one ranking statistic moves.
The reason: candidates that outrank the true partner have a **median 5,386
voxels, and 68.8% of them clear the floor**, against a median of 121 voxels and
32.4% for the panel at large. **The competition is real processes, not dust.**
The dust floor buys compute, never discrimination — and that is now measured on
correct geometry rather than inherited.

### 3d. The one-soma cannot-link, measured (verified here)

`treestitch.stitch.stitch_super_fragments` enforces at most one soma per
assembled neuron. Taking the 103 soma-bearing v117 fragments from
`data/external/cell_cards/`, it vetoes **3 of 44 wrong picks (6.8%)** on cut
panels and **1 of 58 (1.7%)** on negative panels; only **0.56%** of a median
panel is soma-bearing, which is the ceiling for the direct pairwise form. Two
qualifications, both pushing up: the cards cover 103 of the 332 nuclei in the
cube, and in treestitch the constraint bites *transitively*, after a cluster is
assembled, not on a single pair. **6.8% is a lower bound on its value, not its
value.**

### 3e. A units bug that makes the default pipeline's distance gate inert (verified here)

`neuronauts/cell_graph.py::build_synapse_graph` builds the synapse graph as

```python
iso_positions = positions * PATH_ISO[np.newaxis, :]   # PATH_ISO = [1, 1, 40/32]
...
if d <= proximity_radius_nm:                          # default 5000.0
```

`positions` is `SynapseTable.pre_pt`, which `neuronauts/fetch.py` documents as
"box-relative **voxel** coords" — and which three other functions in the *same
file* multiply by `_MIP2_VOX = [32, 32, 40]` to obtain nanometres.
`PATH_ISO` only isotropises z; it does not convert to nm. So `d` is in 32 nm
voxel units and the threshold of 5,000 corresponds to **160 µm**.

Minimal repro on a real cached box (`data/boxes_30um/0057f989543006b6.npz`,
19,588 synapses):

```
pre_pt extent                : [958.0, 955.2, 766.0]  (raw units)
iso_positions diagonal       : 1657.4                 <- compared against 5000.0
same array x [32,32,40] nm   : [30656, 30568, 30640] nm  = the 30 um box
32-NN pairs passing the gate : 313259 / 313259  (100.0%)
```

The whole box is a third the size of the threshold. **The proximity filter can
never fire**; edge topology is set entirely by `max_edges_per_node = 32`. The
same mislabelling propagates through `CellGNNConfig.proximity_radius_nm`,
`infer_cells_edge`, `cell_gnn_assembly` and the `--proximity-radius-nm` CLI flag.
Every "distance-gated" description of this graph is really a 32-nearest-neighbour
description. This does not by itself invalidate F1 0.272 — the graph is still a
graph — but no sweep of that flag ever varied anything.

### 3f. The cell-assignment training cache contains no segmentation errors (read, corroborated)

`docs/dataset_seeding_for_edit_pairs.md`, recorded 2026-05-01: on `boxes_30um`,
`fetch-cave-edits-from-cache` reported **"26,455 chains, 0 merge pairs, 0 split
pairs"**, and **0%** of 1,000 probed supervoxels have v117 ≠ current root — as
against 28% for `root_neighborhoods_v1718_run001`. `experiments/root_neighborhood/`
exists because of this, and its output directory `data/proofread_core_v117/`
was never built.

`data/boxes_30um/index.json` confirms all 325 boxes carry
`root_id_version: 1412` (verified here). **The default pipeline is trained on
boxes that contain no instance of the error it is meant to fix.** F1 0.272 is
therefore a co-assignment number on already-correct segmentation, which is a
different task from the frontier and easier than it in the one way that matters.

### 3g. The one existing frontier-wide pass, run (verified here)

`scripts/measure_frontier_discrimination.py` already scores every tip at the
real base rate. It has no `evaluation.md`, no ledger row, and its own docstring
disclaims the number because it runs on the mip-5 centroid clouds. I ran it:

```
tips 2137   live 34   base rate 1.6%
                     feature     AUC
nearest object (closer=live)   0.535
   best along-axis alignment   0.564
 alignment of best candidate   0.497
          objects within 2um   0.630
       alignment x proximity   0.572
  top  34 by that score: 0 live -> precision 0.0%  (base 1.6%)
  top  68 by that score: 2 live -> precision 2.9%  (base 1.6%)
  top 170 by that score: 3 live -> precision 1.8%  (base 1.6%)
```

**At the operating point that matters — take the top 34, one per cell — the
precision is zero.** Across the whole curve nothing separates from the 1.6%
base rate. Nearest-object, which is the treestitch selection rule, reaches
AUC 0.535.

This is on broken geometry, so it is a lower bound and should not be quoted as
a final number. Its value is that it is the only measurement in the repository
taken at the true composition rather than converted to it, and it points the
same way as §3a from the other direction: **the corrected-substrate panel work
gets some lift at a known decision site; the coarse frontier-wide pass gets none
when it also has to find the site.** Re-running this on mip-2 geometry is the
single highest-value next measurement, and it needs only the panel builder
pointed at tips instead of contacts.

## 4. What is reusable now

**Reusable, with the number measured.**

- **`neuronauts/harness/box_truth.py`** — the honest target. It already refuses
  to put out-of-box links in a denominator (72 of 491 nearest-sibling paths,
  14.7%, leave the cube). Any frontier experiment should score against
  `seeded_target`, not all same-cell pairs.
- **`scripts/measure_frontier_load.py`** and
  **`scripts/measure_frontier_discrimination.py`** — the first produces the 46/1
  composition every precision number above converts through; the second scores
  every tip at that composition (§3g). **Caveat it carries:** it runs on
  `object_clouds_mip5.npz`, the broken centroid substrate, and EXP-081 says so.
  The tip count is a micron-scale estimate; a finer read finds *more* tips, which
  makes the base rate lower, so the conclusions err conservative.
- **The 58 negative panels** (`data/external/panels/` already-whole +
  `data/external/panels_tip/`). Underpowered by 2.6×, but the only
  "nothing continues" sites in the repository on corrected geometry.
- **A 1,000-voxel dust floor** — free, and now verified free on correct
  geometry (§3c). Take it for the compute; expect nothing from it for accuracy.

**Reusable in principle, measured here as weak.**

- **`treestitch.stitch.stitch_super_fragments`'s structural constraints** —
  cycle rejection, each endpoint used at most once, at most one soma per
  assembled neuron, monotone conservative merging ("under-merge is recoverable
  at the next level, over-merge is not"). These sit *above* the per-pair score,
  which is exactly the layer a per-pair scorer at 1.6% cannot supply for itself.
  The module imports and its tests pass today (45 passed). Measured value: §3d.
- **`treestitch/risk.py`** — an explicit ABSTAIN action under asymmetric
  merge/split costs. The right shape for a precision-bound frontier. It consumes
  soft-partition output from a trained model, so it is not runnable on panels
  today, but the interface is the one a grower wants.
- **`docs/p1_benchmark.md`'s protocol** (not its numbers) — reconciling
  same-fragment co-membership before scoring. It is the only evaluation here
  that removes the trivially-positive component from the denominator, and any
  future merge_P should copy it.

**Not reusable.**

- **The treestitch scoring rule itself.** Both factors are now measured on this
  substrate and both are dead: proximity ranks the partner at median 60 with
  **0/66** top-1 (§3a), and the embedding channel scores **0 of 44** top-1 as a
  selector where geometry scores 22 (EXP-080), below the ~2/44 chance rate. A
  product of two dead terms is dead.
- **Every synthetic and semi-synthetic number.** The synthetic atomization
  result (ARI 0.670 → 0.957) is *reversed in sign* on the real 600 µm box
  (0.221 → 0.062). This body of work has a documented instance of synthetic and
  real disagreeing in direction on the same treatment.
- **CellGNN's F1 0.272 as a frontier claim.** Box-local by construction; tuned
  threshold (§2b); 37 of 325 boxes; and trained on a cache with zero
  segmentation errors in it (§3f).

## 5. Limits of this audit

- The negative panels are **already-whole arbor points, not frontier tips.** A
  dead tip on a partially-grown cell is a plausible but unproven proxy. EXP-076
  further showed these panels usually sit *mid-cable* rather than at a genuine
  ending. That does not corrupt the label used here — no candidate belongs to
  the cell either way, so any extension is a real false join — but these sites
  do not test end shape.
- I could not run treestitch's scorer in its **native endpoint-to-endpoint**
  form; the panels carry closest-approach gaps, not skeleton endpoints. The
  saturation argument is specific to closest approach. What is substrate-free is
  that the rule is a monotone function of a gap times an embedding cosine, and
  both terms are independently measured negative here.
- **58 negatives, 66 positives.** Every precision figure carries the intervals
  shown and no more. The 1.000 point estimates are not results.
- I did **not** re-run the EXP-060/060B/061 proximity negatives on corrected
  geometry, so I claim nothing about their 0.09% either way. Only EXP-072's
  dust-floor component was re-run (§3c).
- I did **not** establish why EXP-077's distance row differs. The pool-size
  explanation fits the direction of every discrepancy but is unproven, and the
  file needed to check it is missing.
- The units bug in §3e is verified as *inert*, not as *harmful*. I have not
  measured what CellGNN's F1 would be with a working distance gate.

## 6. What to do next

1. **Build the frontier negative corpus on corrected geometry.** EXP-081 has
   already located 2,103 dead tips; point `scripts/build_contact_panels.py` at
   ~150 of them instead of at known contacts, and both the 2% criterion (§3b)
   and a real re-run of §3g become possible for the first time. Nothing here is
   blocked on a model.
2. **Point EXP-063's frankenmerge detector at a proposed join**, not an existing
   object. It is the only detector here with a real held-out AUC (0.958) and it
   has never been asked the frontier's question.
3. **Lift treestitch's structural constraints out of `stitch.py`** and apply
   them above a frontier scorer. They are the one asset in the prior art whose
   value is not substrate-dependent, and 6.8% is a floor on it, not a ceiling.
4. **Restore the ledger** (§2d), and either rebuild
   `data/external/exp079_panel_tree.json` or retire `scripts/exp079_evaluate.py`.
