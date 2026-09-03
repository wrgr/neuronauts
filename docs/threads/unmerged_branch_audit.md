# Audit: two unmerged branches nobody has revisited

*Opened 2026-09-02. Read-only mining of `origin/claude/segclr-fuser-grammar-x8ba3x`
(50 commits, 2026-07-02 → 2026-07-07) and
`origin/claude/synthetic-data-quality-review-9ihg5x` (8 commits). Nothing was
merged, checked out, or modified.*

Merge bases: `db1a9ae70` (July branch), `1517b78eb` (bench branch). The other two
remote branches hold nothing: `claude/tree-assembly-algorithm-wbtae0` is 0 commits
ahead of HEAD, and `codex/fix-evaluation-paradigm-and-interneuron-validation` is a
single dependency fix.

**Reading rule used throughout.** Every number is tagged by where it comes from:
**(json)** a committed result file on the branch, **(prose)** the branch's markdown
only, or **(verified here)** something I recomputed in this session. A large
fraction of the loudest July claims are **(prose)** with no code and no artifact,
and the gitignored caches they were computed from are gone.

---

## 0. The short answer

**No, 0.73 top-1 is not comparable to our 33%, and it was retracted on its own
branch three days after it was written.**

The controlling number is chance. Our panel task ranks one true object among a
median of 2,440 candidates, with **exactly one** correct answer per panel — chance
top-1 = 4.098 × 10⁻⁴ (verified here, over all 99 panels: 33 `already_whole` with
zero targets, 66 cut panels with exactly one each). The July numbers were measured
against chance baselines between 0.39 and 0.43.

| claim | unit ranked | pool | chance | score | **score ÷ chance** |
|---|---|---|---|---|---|
| July "split-stitch 0.73" | fragment *contact* | contacting neighbours | ~0.39 (prose) | 0.73 | **1.9×** |
| July "follow 0.977" | skeleton *vertex* | 6.88 mean, 2.29 already true (json) | 0.425 (json) | 0.977 | **2.3×** |
| July "cut-face 0.995 @ 40 nm" | slice *footprint* | 33.7 mean (json) | 0.041 (json) | 0.995 | 24× |
| **our panels** | **object** | **2,440 median** | **4.10e-4** | **0.333** | **≈810×** |

Relative to its own chance baseline, the July 0.73 is the weakest ranking result of
the four. It is also a different unit (a contact, not an object), on a different
substrate (m343, a 2022 snapshot the branch itself calls *"NOT a queryable public
materialization"*), scored leniently (top-1 is *same neuron*, so a different branch
of the same cell folding back counts as correct), and it has **no committed code
and no result file** — `git show --stat 5d98dff76` shows the commit touched exactly
one markdown file.

The branch retracted it. `experiments/pcfg/SEGCLR_GRAMMAR_README.md` on that same
branch: *"the earlier per-endpoint 'contested 1.00' was a lenient metric on n=12
and did **not** survive the synapse-level agglomeration test."*

### The three highest-value assets

They are not the headline claims.

1. **The held-precision + oracle-ceiling harness** (`v117_baseline.py`,
   `v117_proofread.py`). It runs on the *same v117 object substrate we use today*
   and asks the question our panels structurally cannot: what happens when you
   commit the join. Its geometry was eroded in the exact way `rerun_catalog.md`
   describes, so its negative must be re-run, not inherited.
2. **The reciprocal-trajectory feature** (`follow_test.py::_features`), with a
   clean single-feature ablation isolating it as the driver on the parallel-neighbour
   failure mode — the one our panels hit. Two lines of code.
3. **The measured negatives**, chiefly that the cut-face appearance encoder now
   sitting in `neuronauts/em_corridor.py` was tested on real confusable joins and
   scored **at or below chance**, while a live thread (`experiments/fingerprints/`)
   is preparing to deploy exactly that encoder into a line-graph F1 run.

### The one sentence that connects the two branches to today

Our 33% top-1 is a **ranking** result. Committed as joins, 33% top-1 is 33% join
precision. The July branch measured, three separate ways, that a partition metric
needs **~0.99** join precision before it beats doing nothing, because one wrong join
fuses two whole neurons through the union-find. The bench branch then measured the
same wall from the other side: on its substrate, **nothing beats the do-nothing
floor** (ARI 0.9610). Our headline is a good ranker and, as of today, an
unmeasured committer.

---

## 1. What our panels actually measure (established first, so the comparisons are honest)

Verified here by loading all 99 `.npz` files, and cross-read against
`scripts/build_contact_panels.py`, `scripts/build_cell_cards.py`,
`neuronauts/harness/box_truth.py`, and `results/EXP-077/join_on_correct_identity.md`.

- **Substrate: v117 agglomerated chunkedgraph roots**, read at mip 2
  (32 × 32 × 40 nm) with `agglomerate=True` at `V117_TS = 1623399000`
  (2021-06-11). Not level-2, not a proofread root. **This is the same base
  version the July v117 joiner used** — the two are directly comparable on
  substrate, and differ only in geometry quality.
- **Truth for `in_target`:** the object is a v1822-proofread fragment of the seed's
  cell lying in the seed's in-box connected component. Stricter than "same cell".
- **Panel shape:** median 2,440 candidates (min 1,812, max 3,020) in an 8 µm cube,
  no dust floor. Exactly one `in_target` per cut panel — so "top-1 is `in_target`"
  and "top-1 is the correct continuation" are the same statement here. Not lenient.
- **66 vs 99:** 66 cut cells with a successfully built panel, 33 `already_whole`
  cells whose in-box component is a singleton and which therefore have zero targets
  by construction.

Three caveats that matter for every comparison below:

1. **The box is oracle-placed.** For a cut cell, the cube is centered on the
   midpoint of the closest approach between the seed and its true partner. EXP-075
   says this plainly: the ranking answers *"given you are looking in the right
   place, can you pick the right object."* A grower does not get that box. So our
   task excludes candidate generation, which is exactly the half the July joiner
   included.
2. **It is one join decision per cell.** No commit stage, no cascade, no
   union-find, no stop rule.
3. **Feature selection happened on the reported set.** The
   `along × collin × proximity × caliber` product was chosen by comparing five
   scores on the same 66 panels the result is quoted from. There is no held-out
   split.

Also worth carrying: 46 of 66 true partners sit at `gap_nm = 32` (one voxel), and a
median of 85.5 non-target candidates are tied at that same gap — which is why the
distance baseline is a tie-break artifact and why `along`/`collin` carry the result.

---

## 2. The July branch, claim by claim

### 2.1 "M3 split-stitch: 0.73 top-1 in dense cluster" — **do not revive**

*Claimed (prose):* dense 8-neuron column cluster, 322 fragments, top-1 SegCLR
stitch = 0.73 (179/245); absolute join AUC 0.66; 255 same-neuron vs 405
different-neuron contacts; chance ~0.39.

*Actually measured:* the substrate is **m343**, a 2022 snapshot. The unit is a
*contact between fragment point clouds*, not an object. The metric is "same
neuron", not "the correct adjacent continuation".

*Evidence:* **prose only.** `run_compartment_grammar.py` has subcommands
`--exp0 --exp1 --exp2 --m1` and **no `--m3`**. No eval driver, no result JSON, and
`21a5583dd` gitignored the m343 fragment and SegCLR caches it was computed from.
The same is true of `a66dd6c49` (merge AUC 0.840 / multi-soma 1.0), `1376dfd2a`
(the 2×2 ablation), `e58e4c25a`, and `981bb4db7` — **every one of those commits
changed markdown and nothing else.**

*Comparability:* none, in either direction. 1.9× chance against our 810×.

### 2.2 "Split fixer: endpoint stitching + SegCLR = 1.00 on column cluster" — **do not revive**

n = 12 contested endpoints; 95% interval roughly [0.76, 1.0]. The branch's own
five caveats are all correct: the metric scores a different branch of the same cell
folding back within 15 µm as correct, while that join builds wrong topology; at
≤6 µm even distance-only scores 1.00, so the task is mostly trivial; only two
PC1-extreme endpoints per fragment were enumerated. `f09a416d6` touched the
markdown plus 8 lines of `walk_detector.py`. Retracted on-branch by `397a9091a`.

### 2.3 "M1: compartment labeling from synapse polarity + soma caliber (PASS)" — **revive the soma part only**

*Claimed (prose):* 4 proofread neurons, PRE→AXON 0.98–1.00, POST→DEND 1.00.

*Actually measured:* **the metric is circular.** `label_compartments` builds the
AXON label by snapping pre-synapses to vertices and diffusing them, and DEND from
post-synapses; `run_m1` then asks whether pre-synapses land on AXON vertices — the
same synapses that created the label. It measures whether the diffusion and the
0.60 dominance threshold survive at the source vertex. n = 4. This is not a
validated compartment labeler and should not be cited as one.

*What is real:* the soma observation is independent and useful — the CAVE skeleton
service represents a soma as a **single large-radius vertex (~5,300 nm)** while all
cable is ≤ ~425 nm at p99, so a 3,000 nm threshold separates cleanly and a
two-soma object shows two big-radius vertices. `neuronauts/soma_clusters.py`
(79 lines, offline-tested) is worth taking as a merge guard.

### 2.4 "M3 column merge eval: multi-soma solved (AUC 1.0), axon-graft recall 0" — **revive the negative only**

The positives were **synthetic merges** built by gluing clean proofread neurons.
Multi-soma AUC 1.0 is near-tautological: a synthetic soma+soma merge has two somas
and the detector counts somas.

The negative was checked on real data and is the part to keep: on 7 real merges via
the m343→current bridge, the axon↔dendrite rule fired on only **1 of the 3
single-soma merges**, and clean neurons reached ad-scores up to 0.617, overlapping
the real merges. *Not every merge crosses a compartment boundary* — axon-to-axon
and dendrite-to-dendrite merges are invisible to the grammar. Prose, n = 7.

### 2.5 "SegCLR = type, not identity" — **revive; the branch's most valuable positive-in-the-negative**

*Claimed (prose):* clustering one neuron's SegCLR nodes recovers axon/dendrite at
ARI 0.69–0.86; pooling two cells splits by compartment (ARI 0.68–0.84) and by cell
identity at **ARI ≈ 0.00–0.03 on 10/10 pairs**. A supervised metric head on frozen
embeddings, trained to separate same-cell from different-cell-**same-type** pairs,
scores AUC **0.509** against raw cosine **0.513** — both chance.

*It replicates independently in our own cache.* `data/external/segclr/auc_result.json`,
written 2026-09-02 by the current thread's own probe: 34 atoms, 12 owners, 33
same-owner vs 528 different-owner pairs, same_mean 0.823, diff_mean 0.842,
**AUC 0.445**. Same-object embeddings are not more similar than different-object
ones. The mechanism is credible — SegCLR trains contrastively with a ~4 µm
receptive field and local positives, so it encodes local morphology, which is
compartment and type.

*And it now replicates a third time, on our own panels.* **EXP-080 landed at HEAD
(`dd49b5cbb`) while this audit was being written**, testing exactly the July claim
that "geometry proposes, SegCLR decides." Geometry's top 20 candidates re-ranked by
embedding, on the 44 panels whose true partner is in that top 20:

| | median rank | top-1 | top-3 |
|---|---|---|---|
| geometry alone | 0.5 | **22/44** | 30/44 |
| SegCLR selects | 11.0 | **0/44** | 3/44 |
| geometry × SegCLR | 0.5 | 22/44 | 30/44 |

Chance for a selector over 20 candidates is median rank ~9.5 and ~2/44 top-1;
SegCLR reaches 11.0 and 0. **Not a coverage artifact** — 44 of 44 true partners
carry an embedding. Multiplying changes nothing because the embedding term is
essentially constant across candidates. That settles the selector question on our
substrate and supersedes recommendation 10 below.

EXP-080 also names the one variant it did not control, and it is the right one: it
used a **whole-object mean** embedding, whereas the July code laid embeddings
**along a skeleton** and compared rolling-averaged local traces either side of a
gap. A local trace could carry signal a mean washes out. That is the only remaining
live SegCLR question, and it is narrow.

*Direct consequence for a live plan.* `docs/threads/embedding_availability.md`
(2026-09-01) recommends retargeting EXP-057C at SegCLR because *"retrieval over a
morphology embedding has no radius, so it is the remaining option."* The July
branch **built the crosswalk that document says is needed** —
`neuronauts/segclr.py`, a dependency-free byte-range shard reader that reproduces
the sharding with the non-obvious `bytewidth=64` scheme verified against real shard
membership, plus the m343→current bridge (`chunkedgraph.get_latest_roots` on the
m343 root) — **and then measured the payoff and found it at chance for identity.**

That does not kill retrieval as a *candidate generator*: type-level retrieval with
no radius could still cut a 2,440-candidate panel down, and that is a legitimate
thing to test. It does mean **do not expect SegCLR to rank a cell's own
continuation above a same-type neighbour**, which is precisely the discrimination
our panels need. Budget it as a coarse prefilter, never as the discriminator, and
budget the m343 bridge cost knowing that.

### 2.6 "Exp 1: SegCLR near-chance on REAL merges (0.49) vs synthetic (0.95)" — **revive as the methodology lesson**

Per-merge real AUCs 0.59, 0.56, 0.42, 0.27, 0.34, 0.59, 0.63; 0/7 above 0.7;
synthetic 0.95 on the same n = 7. The stated explanation is the transferable part:
a synthetic merge joins two *unrelated* neurons, trivially separable; a real false
merge happens exactly where two cells are locally similar and touching, *and that
similarity is why the segmentation merged them.* **Constructed damage
systematically manufactures easy negatives and overstates every local cue.**

*Do not adopt Exp 2, which the branch published as a "correction" of this.*
`_embedding_top1_same_cell` runs a k=2 nearest-neighbour query in embedding space
and drops only the query itself — it does not exclude the query's *spatial*
neighbours. SegCLR nodes are ~1.2 µm apart, well inside the ~4 µm receptive field,
so the nearest embedding is almost always the node's own cable neighbour, same-cell
by construction. Exp 2's 0.866 is an artifact, and the branch's own later metric-head
result (0.509) supersedes it.

### 2.7 The v117-object joiner — **the most relevant asset; revive the harness, re-run the conclusion**

`experiments/proofread/v117_baseline.py` + `v117_proofread.py` (+ offline tests).
Numbers are (prose), but the code is committed and runnable.

**The baseline reframing is worth adopting outright.** Whole proofread column,
1,355 cells, 5.5 M synapse halves, v117 predictions against v1507 truth:

| region | neurons | v117 objects/neuron | P | R | F1 | **axon-OUT recall (median/neuron)** | dend-IN recall |
|---|---|---|---|---|---|---|---|
| overall | 1,355 | 68 | 0.999 | 0.857 | 0.923 | **0.095** (87% of cells < 0.5) | 0.959 |
| eval (held out) | 307 | 67 | 1.000 | 0.841 | 0.914 | 0.089 | 0.952 |

Precision is ~1.0 with **zero catastrophic merges** in every region. The error is
splits and it is almost entirely axonal — a median of 35 objects is needed to cover
90% of a cell's outputs. **Inputs are 83% of halves and nearly free, so any
aggregate synapse metric is dominated by the easy half:** the 0.923 overall F1
hides 0.095 axon-output recall. Report axon and dendrite separately or the number
means nothing.

**The joiner result.** Candidates built globally with a KDTree over object point
clouds — no oracle box — scored by
`0.5·colinearity + 0.3·(1 − gap/max_gap) + 0.2·caliber_match`, labelled correct iff
the two objects share a v1507 truth:

| candidate gap | oracle median axon-out recall | achievable @ P ≥ 0.999 | greedy max recall (precision) |
|---|---|---|---|
| 2 µm | 0.113 | 0.098 (no lift) | 0.65 (P 0.03) |
| 3 µm | 0.217 | 0.098 (no lift) | 0.73 (P 0.03) |
| 5 µm | 0.436 | 0.098 (no lift) | 0.89 (P 0.03) |
| 8 µm | **0.754** | 0.098 (no lift) | 0.94 (P 0.03) |

Held-out eval reproduces it (oracle 0.155 → 0.432 → 0.735 at 3/5/8 µm). Candidate
generation is not the wall — the correct joins are in the set, and an oracle
recovers axon-output recall from 0.10 to 0.75 within 8 µm. **The discriminator is
the wall:** join edge-precision tops out at 0.5–0.58, and committing at that
precision collapses neuron-level precision to 0.03.

**Why the conclusion is not yet earned.** `neuron_objects()` builds every object's
geometry from `cg.get_leaves(root, stop_layer=2)` → `l2cache` **`rep_coord_nm`** —
one representative coordinate per level-2 chunk (roughly 4 × 4 × 2.5 µm),
subsampled to ≤300 points per object. So every `gap` in that table is a distance
between *chunk representative points*, not between surfaces. **This is the exact
error class `docs/threads/rerun_catalog.md` item 1 describes** — *"two objects that
physically touch can have centroids microns apart… harmless at micron scale, fatal
for contact"* — the error that inflated EXP-060/060B/061/072 to 0.09% precision and
that reading with `agglomerate=True` fixed. Under that geometry, caliber went from
"median rank 140, useless" to the strongest single term and collinearity from rank
220 to 30. The July joiner's features are the same family, measured on the eroded
version.

**Recommendation: revive with caveats, and re-run.** The harness is the instrument
we are missing — oracle ceiling, held-precision sweep, matched per-neuron
synapse-half confusion split by side. The *conclusion* is an open question, not a
settled negative, and re-running it on the corrected mip-2 `agglomerate=True`
geometry is the single highest-value experiment on either branch.

### 2.8 The "follow" model — **revive one feature; discount the accuracy**

`experiments/proofread/follow_test.py`, backed by `out/follow_test.json` — real
committed evidence, unlike §2.1–2.4.

| scorer | top-1 | hard (n=1,175) | confusable (n=295) |
|---|---|---|---|
| nearest (proximity) | 0.464 | 0.000 | 0.305 |
| align (direction only) | 0.941 | 0.921 | 0.563 |
| learned (trajectory + caliber) | 0.962 | 0.939 | 0.722 |
| + consequence (reciprocal trajectory) | **0.977** | 0.970 | **0.841** |

**What the JSON says and the prose underplays:** `mean_candidates` 6.88,
`mean_true_per_inst` 2.29, `chance_top1` **0.425**. `build_instances` gathers
candidate *vertices* in a 1.5–3.5 µm annulus over 189 scattered clean skeletons. So
0.977 is top-1 over a ~7-candidate pool in which 42.5% of the pool is already
correct — 2.3× chance. Not comparable to 2,440 candidates at 810× chance. The
branch states the caveat honestly.

**What transfers, and it is the second-best asset on the branch:** the
**reciprocal-trajectory feature**, with its ablation recorded in the source:

```
# Ablation (confusable top-1): base 0.72; +recip 0.89 (the driver); +tan_agree 0.79;
# +ray_gap 0.75 (near-useless — a tight parallel fascicle has a *small* line-gap, so
# it favours the distractor).
```

```python
u = rel / (dist[:, None] + 1e-9)          # unit vector: cut endpoint -> candidate
recip = np.abs((tan * (-u)).sum(axis=1))  # candidate's own cable aims back at the cut
tan_agree = np.abs(tan @ d)               # both cables lie on the same line
```

This is *bidirectional* consistency: a severed cable is collinear **from both
ends**, whereas a parallel fascicle member is laterally offset and its own cable
does not extrapolate back to the cut. Our panels' `along` (seed axis vs seed→
candidate direction) and `collin` (seed axis vs candidate axis) are both symmetric
in a way that does not capture this — neither asks whether the *candidate's* cable
points back through the gap at the seed's tip. On the evidence, `recip` is the
cheapest untried feature available, aimed at exactly our failure mode (a parallel
same-type neighbour), and computable from data the panel builder already has.

Two attached warnings: **`ray_gap` is actively harmful** (a tight fascicle has a
small line-gap, so it favours the distractor), and 37.3% of our candidates already
get `along = collin = 0.0` from the "<3 local points" degeneracy rule, so any new
directional feature needs the same degeneracy audit.

`follow_learned.py` (MLP on 17 raw canonical coordinates, 0.875 vs 0.841 on the
confusable subset under 5-fold GroupKFold) is a nice existence proof that the
follow function is learnable from raw geometry, on the same 7-candidate pool. Low
priority.

### 2.9 The cut-face follower — **revive the mechanism; discount the 0.995**

`experiments/proofread/cutface_slices.py` (726 lines), backed by
`out/cutface_slices.json`, `out/follow_fused.json`, `out/follow_matching.json`.

*Claimed (json):* re-linking a segmented object's 2-D cut face to the following
slices by **footprint geometry only** (IoU/centroid/area, no appearance, seg id
never used as a feature) gives top-1 **0.995 at 40 nm** among ~33.7 candidates
(chance 0.041), 0.769 at 120 nm, 0.430 at 240 nm.

*What it actually measures.* Reading `evaluate_cutfaces`, the loop skips any object
not present on the later slice (`if oid not in nextf: continue`), and truth is
"same seg id". So it measures **how z-consistent the existing segmentation already
is** — whether an object the segmenter *already linked* across 40 nm can be
re-found by footprint overlap. A real false split is precisely where that
continuity broke, usually from misalignment or damage. The branch flags this
("artificial cuts of continuous objects; a real false split broke for a reason…
an upper bound") and then proves it in §2.10.

*What genuinely transfers:*

- **Global one-to-one matching beats greedy top-1 everywhere**, and is the
  mechanism that makes auto-apply safe: under greedy, a terminal tip whose
  neighbour overlaps it becomes a false merge; under matching the neighbour is
  claimed by its own true continuation and the tip is left unmatched. Measured
  (mip-2, one box): coverage @ P ≥ 0.99 rises 0.900 → 0.949; precision @ coverage
  ≥ 0.85 rises 0.860 → 0.929 mixed-gap.
- **Connect-vs-abstain is a measurable signal**: with terminal cut faces included
  as all-negative instances, the top-candidate fused score separates "has a true
  continuation" from "is a tip" at **AUC 0.836** (`out/follow_ultra.json`,
  `connect_auc_geom` 0.8358, n = 20,662).
- **Learned fusion beats every single cue at every gap**: 0.859 vs 0.775 best
  single at 120 nm.
- The **grammar veto** fires on ~19% of top picks and kills false connects 2.2 : 1,
  lifting precision 0.69 → 0.73 at matched coverage.

*Cross-branch consequence, and this one is time-sensitive.* Our live
`experiments/fingerprints/cutface/` thread trains an **appearance**-based cut-face
encoder. Its own held-out numbers (`experiments/fingerprints/README.md`,
`learned_metrics.json`) are top-1 **0.190** at a 40 nm gap among 165 candidates,
where the spatial baseline gets 0.599; and its stated next step is *"feed its
scores into `cell_graph.build_synapse_graph` as an edge feature and measure
line-graph F1."* The July branch tested that exact committed encoder on real
confusable joins and got **AUC 0.58, then 0.456** — at or below chance, and below
the geometry residual (0.636). Its deployment hook, `batch_cutface_similarity`, is
already merged into `neuronauts/em_corridor.py`. **Do not spend the line-graph F1
run on the appearance encoder.** If cut faces are used at all, use the
footprint-geometry version.

### 2.10 The goal-metric negatives — **revive: this is the "do not repeat" list**

The strongest work on the branch: committed JSON, the metric that matters, real
fragments.

**"Landmark does NOT work"** (`397a9091a`, prose) — dense 8-neuron patch, 279
fragments, 34,424 half-synapses:

| | precision | recall | F1 |
|---|---|---|---|
| no-stitch baseline | 0.96 | 0.76 | 0.848 |
| **oracle** (join only correct contacts) | 0.96 | 0.90 | **0.928** |
| distance / geometry / SegCLR / combined, any threshold | **0.14** | 0.9+ | **0.24** |

Three compounding reasons, all stated: different-neuron contacts outnumber
same-neuron in dense tissue (247 vs 172); local cues cannot separate them; and
**agglomeration transitivity is catastrophic** — about 7 wrong joins chain all 8
cells into one component.

**The corrected fragment baseline** (`3272d69da`) — worth reading as a worked
example of this repo's recurring failure. An earlier run reported F1 = 0.991 for
"v117 vs proofread truth". The branch caught it: the column's v117 is *already
proofread* (604 v117 roots vs 609 v1822 roots), so that measured
proofread-against-proofread — **and it impossibly exceeded the repo's own oracle
ceiling of 0.928, which is the tell.** The real fragmented baseline, at L2 chunk
level, is **P = 1.000, R = 0.132, F1 = 0.234**.

**The goal-metric result** (`17c3b5866`, `out/synapse_f1_join.json`, verified here
by reading the JSON):

| threshold | joins | join precision | recall | F1 |
|---|---|---|---|---|
| before (L2) | — | — | 0.132 | **0.2336** |
| 0.10 (join freely) | 56,999 | 0.14 | 0.983 | **0.0068** |
| 0.59 (best F1) | 1,325 | 0.61 | 0.148 | **0.2568** |
| 0.80 (high precision) | 372 | 0.81 | 0.139 | 0.245 |

A 0.02 gain. Join edge-precision never reaches 0.90. **A partition F1 needs
near-perfect join precision, because one wrong join merges two whole neurons and
destroys many synapse pairs at once.**

**Merge-aware vetoes don't lift synapse-F1** (`ec04aacc5`,
`out/merge_aware_join.json`) — confident-first union-find with axon↔dendrite,
two-soma, caliber and quarantine vetoes:

```
BEFORE (L2):        F1 = 0.234
GREEDY best:        F1 = 0.257   join_P = 0.61
MERGE-AWARE best:   F1 = 0.250   join_P = 0.48
ablation: no_ad / no_soma / no_quarantine all identical to full
typing: axon = 64  dend = 79  contaminated = 1  soma_frags = 0
```

**The diagnosis is the value.** Only ~143 of 1,000+ L2 fragments could be typed at
all — an L2 fragment is so small it carries at most one synapse, so polarity typing
covers almost nothing, and there were zero soma-scale fragments in the box. The
merge signature grammar exploits lives at the **neuron** scale; the errors that
cascade live at the **fragment** scale, where that signal is absent. Removing the
axon↔dendrite veto is bit-identical to keeping it. Chicken-and-egg: vetoes might
work on partially-assembled components, but the damaging joins happen before
components grow large enough to type.

**The cut proposer negative** (`54b8ea3f0`, prose, n = 3) — propose a cut at every
branch point, score by comparative SegCLR. True-seam ranks were 149/149, 28/301 and
105/134; top-1 was 0/3; clean neurons' max cut score (0.10–0.13) *exceeded* the real
seam scores. Two structural reasons: a same-compartment seam is locally similar, and
branch points are intrinsically high-comparative, so the seam competes against the
hardest possible candidate set and loses.

**The local-EM identity discriminator** (`7d99271c1`, `4e8e5284f`) — on real
confusable candidate joins with geometry held roughly flat, isolating each cue's
marginal identity signal:

| cue at the two contact faces | AUC |
|---|---|
| geometry score (residual) | 0.636 |
| identity embedding (mip-1, hard-negative trained) | 0.447 |
| committed type encoder | 0.456 |
| raw mip-1 patch cosine | 0.492 |

Properly sanity-gated: a first attempt on masked mip-2 was **discarded because it
failed its own control** (every masked face is a centred blob → same-object
retrieval at chance 0.50); the unmasked mip-1 representation reaches 0.68–0.70
same-object adjacent-z retrieval, so the negative is not a representation artifact.
Honest caveats stated (small CPU encoder, mip-1 not mip-0, ceiling only ~0.70).

### 2.11 Evidence-quality summary for the July branch

| block | code | result artifact | substrate | verdict |
|---|---|---|---|---|
| Exp 0/1/2, M1 | yes | no | m343 + CAVE | runnable with a token; Exp 2 is an artifact, M1 is circular |
| **M3 (0.73, 1.00, AUC 0.840, 2×2 ablation, type-not-identity ARI, landmark)** | **no** | **no** | m343, caches gitignored | **prose only — not reproducible** |
| follow / cut-face / fusion / matching / pipeline | yes + offline tests | yes | clean skeletons, mip-2/3 cutouts | reproducible; pools small and sparse |
| v117 baseline + joiner | yes + offline tests | no | v117 objects via L2 `rep_coord_nm` — **eroded** | harness good; conclusion needs re-running |
| synapse-F1 join, merge-aware join, real splits | yes + offline tests | yes | real v117→v1822, L2 level | **strongest evidence on the branch** |

---

## 3. neuronauts-bench v1 (`origin/claude/synthetic-data-quality-review-9ihg5x`)

### 3.1 What it is

**One item = one presynaptic synapse observation.** The record is
`positions_nm[3]`, `supervoxel_id`, `synapse_id`, `base_roots` (the **v117** root —
the input) and `label_roots` (the **v1718** root — the label). That is all.
`model_bench_v1.py` states the substrate: `"fragment_substrate": "synapse_cloud"`.
**There is no segmentation geometry** — no voxels, no meshes, no level-2 skeletons.

Labels are **real proofreading**, not synthetic damage: `synapses_pni_2` on
`minnie65_public`, `pre` side only, each supervoxel resolved at both
materializations. A *true merge pair* = two v117 roots → one v1718 root (a real
false split a proofreader repaired); a *frankenmerge* = one v117 root → two v1718
roots.

| split | regions | observations | v117 roots | v1718 roots | true merge pairs | frankenmerges |
|---|---|---|---|---|---|---|
| train | OOC3, P1a, A, E | 76,429 | 52,261 | 53,049 | 351 | 756 |
| val | P1b | 16,244 | 11,375 | 11,543 | 59 | 181 |
| test | P1c | 20,000 | 12,287 | 12,445 | **153** | 203 |

**"Root-disjoint"** is enforced and verified: `scripts/verify_split.py` re-derives,
from a separate code path, that `(base ∪ label)` root sets share nothing across
splits, that cross-split bbox gaps meet the declared seam buffer, that versions are
pinned and `synthetic` is `false`. Disjointness comes from priority dedup
(test > val > train). **"Gated"** is a build-time refusal:
`MIN_MERGE_PAIRS = {train: 20, val: 5, test: 10}` and `MIN_FRANKENMERGES`, with
`SystemExit("Refusing to write a dataset that fails its own acceptance gates…")` —
and the README records it firing for real.

### 3.2 The task, the floor, and the learned attempt

The task is **unseeded global partition**: cluster all N observations into neurons,
starting from the v117 assignment. Metric is exact pairwise contingency, reported
both aggregate and — the headline — **cross-fragment only**, because *"an atomic
baseline scored 0.914 pair-F1 while resolving nothing."*

Test = P1c, 20,000 observations, 703 true cross-fragment pairs:

| method | ARI | cross-merge P | cross-merge R | cross joins predicted |
|---|---|---|---|---|
| **untouched v117 (do nothing)** | **0.9610** | n/a | 0.000 | 0 |
| proximity union-find (2 µm, val-calibrated) | 0.0002 | 4.3e-06 | 0.686 | 112,100,512 |
| EdgePartitionGNN + GAEC (val-calibrated) | 0.0017 | 1.4e-05 | 0.441 | 22,211,225 |
| *oracle fragment ceiling* | *0.9706* | *0.928* | *0.619* | *469* |

**Nothing beats doing nothing.** The GNN's edge classifier trained fine (accuracy
0.953, p_pos 0.725 vs p_neg 0.143) and still has no usable operating point. Its
precision, 1.4e-05, is about 3× proximity's and four orders of magnitude short of
usable. The branch's own diagnosis is **"the substrate, not the algorithm"** — no
L2 cache exists for those regions, so a fragment's morphology *is* its synapse
point cloud, and 8,823 of 12,287 fragments have exactly one synapse.

Commit `0b536b15a` fixed a real bug: `min_syn_per_fragment` defaulted to 3, which
discarded **87% of candidate roots and 68% of positives** — and the discarded
population is exactly the singleton-confuser set. Everything was genuinely re-run
(distinct `data_manifest_sha` and `created_utc` on the pre- and post-fix JSONs, not
a text edit). Commit `4be4b2ded` retracts a causal claim in the CLAUDE.md §0 manner
and is worth reading as a model: *"I earlier wrote that 'response time tracks
`limit`, not bbox size'; that claim was wrong, generalised from two observations."*

### 3.3 Is it a better evaluation substrate than our panels? **No — but steal its process**

**It is not.** Four structural reasons:

1. **No segmentation geometry.** Its fragments are pre-side synapse point clouds;
   our panels read real v117 objects at mip 2 with `agglomerate=True`. The branch's
   own conclusion — *the binding constraint is the representation, not the
   inference algorithm* — is an argument for our substrate. Its stated next step is
   to build the L2 cache; we already went past that to object voxels.
2. **`side="pre"` only**, so any object without a presynaptic terminal in the box
   does not exist: severed dendritic tips, passing axon fragments, and the whole
   dust population are invisible — exactly what fills our 2,440 candidates.
3. **The error direction is inverted.** Frankenmerges outnumber merge pairs 5–10×
   in every region (test: 203 vs 52 repaired neurons). Our task is joining.
4. **It is not a bigger evaluation.** Its 153 test merge pairs come from **52**
   proofread neurons; we have 66 panels with one positive each.

**It is also not reproducible here.** The observation arrays live in
`data/bench_v1/regions/*.npz`, which is gitignored; only the manifests (bboxes,
stats, root id lists) are committed. Every region has `limit_reached: true`, and
`lineage.py` states the server-side `limit` has no stable order — determinism came
entirely from `cache/synapse/*.npz`. **Verified here: `cache/` contains zero `.npz`
files.** A rebuild would fetch a different 20,000 rows per region, break the
recorded checksums, and reproduce none of the published numbers.

**What is worth taking, and it is real:**

- **The do-nothing floor discipline.** ARI 0.9610 for declining to act is the most
  useful number on that branch. We report "median rank 5 of 2,440, top-1 22/66"
  with **no floor and no ceiling** beside it.
- **A genuine train/val/test split.** Our 99 panels sit in one 100 µm cube with no
  split and no root disjointness, and the winning feature product was selected on
  the same 66 panels it is reported on. bench_v1's protocol would not have allowed
  that. This is the cheapest correctness win available.
- **Provenance guardrails:** `scripts/lint_provenance.py` (7 rules including
  LEAK001 label-into-scorer), `neuronauts/results_schema.py` (refuses to write a
  result without `data_manifest_sha`, both versions, and a `synthetic` flag), and
  `.github/workflows/provenance.yml`. None of this exists at HEAD.

### 3.4 The synthetic-data audit, and what is still live at HEAD

`docs/synthetic_data_audit_and_dataset_plan.md` (365 lines) documents five defects
with file:line, and `956ec9e81` quarantined **70 files**. We independently retired
most of it — `neuronauts/morpho_grammar/` at HEAD is a deprecation shim pointing at
`attic/`, and the dashboard/viz scripts are in `attic/`. Two things are not:

1. **`neuronauts/global_merge/represent/cloudvolume_em_sampler.py` and
   `local_em_verifier.py` are still live in the importable package.** Verified here
   by reading the file: `sample_bridge_volume(self, src_coord_nm, dst_coord_nm,
   is_true_continuation: bool, rng)` takes the ground-truth label as an argument
   and branches on it (`if is_true_continuation:` → `radial_coherence = 0.88 *
   dist_attenuation`), returning a Gaussian, under a docstring that claims it
   *"Samples a 3D cylindrical voxel patch."* Nothing outside `attic/` imports it and
   `docs/consolidation_plan.md:232` already marks it ATTIC — it is dead but
   reachable, and it should be moved.
2. **`treestitch/data.py::_split_skeleton_n_pieces`** (bisecting real proofread
   skeletons into thirds and calling them "v117 fragments") is live and called from
   `treestitch/synthetic.py:131` and `treestitch/data.py:168,258`, with no warning
   and no lint to catch a result built on it.

The audit also records that a CAVE bearer token was committed in plaintext in ≥10
files. `neuronauts/auth.py` at HEAD fixed the code path and states *"rotating the
credential is the only thing that undoes that."* **If it has not been rotated, that
is still open.**

---

## 4. Ranked recommendations

Expected value = (chance it changes a decision) × (cost of learning it otherwise),
divided by effort. Nothing here is a merge; each is a port of specific code or a
specific experiment.

| rank | action | source | effort | why |
|---|---|---|---|---|
| **1** | **Re-run the v117 joiner's held-precision + oracle-ceiling sweep on the corrected mip-2 `agglomerate=True` geometry** | `v117_baseline.py`, `v117_proofread.py` | medium (CAVE fetch of synapse halves per seed) | Same v117 substrate we use. Answers the question our panels cannot: at what precision can these ranks be *committed*. Its negative was measured on eroded `rep_coord_nm` geometry — the exact error `rerun_catalog.md` item 1 voids — so it is an open question, not a settled one. |
| **2** | **Add the reciprocal-trajectory feature to the panel score and re-rank** | `follow_test.py::_features` | low (two lines + candidate tangents) | The one untried feature with a clean ablation isolating it as the driver (confusable 0.72 → 0.89) on our exact failure mode. Skip `ray_gap`; it is harmful. |
| **3** | **Put a do-nothing floor and an oracle ceiling beside the panel headline, and split the panels into val/test** | bench_v1 protocol | low | "Median rank 5, top-1 33%" is currently reported with no floor, no ceiling, and a feature product selected on the reported set. Cheapest correctness win available. |
| **4** | **Attach a commit stage to the panel ranker and measure the cascade** | `merge_aware_join.py::constrained_union_find` + `cutface_slices.py::evaluate_follow_matching` | medium | Global one-to-one matching beats greedy everywhere and is the safety mechanism. 33% top-1 = 33% join precision; the branch measured that ~0.99 is the bar. Until this is measured we do not know what our ranker is worth. |
| **5** | **Stop the fingerprints line-graph F1 run on the appearance cut-face encoder** | §2.9 | none (a decision) | That encoder was tested on real confusable joins at AUC 0.58 → 0.456, below the geometry residual. Redirect to footprint-geometry cut faces. |
| **6** | **Re-frame the panel metric to separate axon from dendrite** | `v117_baseline.py::matched_confusion` | low | Inputs are 83% of halves and ~0.95 recall already; an aggregate number hides the axon collapse (0.923 F1 vs 0.095 axon-output recall). |
| **7** | **Port the abstention design: terminals as all-negative instances, connect-vs-abstain as a measured AUC** | `cutface_slices.py` (0.836) | medium | Our abstention is 0.642 [0.54, 0.75] on an effective n of 21 terminals, and drops to 0.531 once the caliber term is added — the stack that ranks best stops worst. The July design measures the same quantity on 20,662 instances. |
| **8** | **Land the provenance guardrails; move the two label-leaking modules to `attic/`** | `lint_provenance.py`, `results_schema.py`, CI | low | `cloudvolume_em_sampler.sample_bridge_volume` takes `is_true_continuation` and branches on it, in the live package namespace. |
| **9** | Port `soma_clusters.py` as a two-soma merge guard | `neuronauts/soma_clusters.py` | low | Non-circular, offline-tested, cheap. Not the circular "M1 PASS". |
| **10** | ~~Use SegCLR as a coarse type prefilter~~ — **superseded by EXP-080 (`dd49b5cbb`)**, which measured it as a selector on our own panels: 0/44 top-1 against geometry's 22/44, below chance, with full coverage. The only live variant left is the skeleton-local rolling trace (not a whole-object mean) that July actually used | `neuronauts/segclr.py` | medium | Identity is now at chance three times over: 0.509 on-branch, 0.445 in our cache, 0/44 on our panels. Do not spend EXP-057C on a discriminator. |
| — | Do **not** revive: the walk detector / comparative-SegCLR merge scan; the branch-point cut proposer; any local-EM appearance cue; the M3 split-stitch and endpoint-stitch numbers | §2.1, 2.2, 2.10 | — | Retracted on-branch, or measured at/below chance three independent ways. |
| — | Do **not** adopt bench_v1 as our evaluation substrate | §3.3 | — | Geometry-free, pre-side only, inverted error direction, not reproducible without a cache we do not have. |

---

## 5. Negative results to not repeat

1. **A partition metric needs ~0.99 join precision, not 0.9.** One wrong join fuses
   two neurons through the union-find. Measured three ways: precision 0.96 → 0.14
   at the contact level; F1 0.234 → 0.007 when joining freely; neuron-level
   precision 0.03 at edge precision 0.5–0.58. Evaluate a joiner with the commit
   stage attached, never as a ranker alone.
2. **SegCLR encodes type, not identity.** Metric head AUC 0.509 vs raw 0.513 on
   same-type pairs; independently replicated by our own
   `data/external/segclr/auc_result.json` at 0.445.
3. **Local EM appearance carries no identity signal at the confusions.** Three
   attempts: the re-ID cut-face cosine (AUC 0.40 at real seams, wrong direction); a
   straight-corridor learned EM feature (made it *worse*: confusable 0.732 →
   0.643); a purpose-trained hard-negative identity embedding (0.447, sanity-gated,
   loss never converged). The tested encoder is the one in
   `neuronauts/em_corridor.py` today.
4. **Merge-aware vetoes do not lift synapse-F1 at fragment scale**, because
   fragments cannot be typed (143 of 1,000+; zero soma fragments). Ablating the
   vetoes is bit-identical to keeping them.
5. **The branch-point cut proposer is worse than useless** — top-1 0/3, and clean
   cells score above real seams.
6. **`ray_gap` is a harmful follow feature** — a tight fascicle has a small
   line-gap, so it favours the distractor.
7. **Constructed damage overstates every local cue** — AUC 0.95 synthetic vs 0.49
   real on the same n = 7. A real error occurs where two processes are locally
   similar; a manufactured one usually does not.
8. **Any aggregate synapse metric is dominated by dendritic inputs** (83% of
   halves, already ~0.95 recall). v117's 0.923 overall F1 hides 0.095 axon-output
   recall.
9. **An unseeded global partition on a geometry-free substrate cannot beat doing
   nothing** — bench_v1's floor is ARI 0.9610 and neither proximity nor a trained
   graph neural network gets within four orders of magnitude on precision.
10. **A result above the oracle ceiling is a bug, not a win.** The July branch
    caught its own F1 = 0.991 that way (it exceeded the 0.928 oracle). Worth
    adopting as a standing check.

---

## 6. Open questions this audit did not settle

- Whether the July joiner's "no lift at held precision" survives corrected
  geometry. **Not ruled out either way** — that is recommendation 1.
- Whether a **skeleton-local rolling SegCLR trace** (July's actual method) carries
  signal that EXP-080's whole-object mean washed out. This is the only live SegCLR
  question; identity and object-level selection are both settled negative.
- Whether type-level SegCLR retrieval is a useful radius-free *candidate generator*
  for a 2,440-candidate panel. Selection is settled; generation is not tested.
- Whether the CAVE token from the audit was ever rotated.
- Whether our abstention's collapse to 0.531 when the caliber term is added is a
  real tension between ranking and stopping, or an artifact of the 21-terminal
  sample. The effective sample is too small to tell.
