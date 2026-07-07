# Honest v117 baseline: what the pre-proofreading agglomeration actually gets wrong

*(`v117_baseline.py`, `tests/test_v117_baseline.py`; metric verified on the whole
proofread column, not boxes.)*

**Question.** How well does the pre-proofreading **v117** agglomeration place each
neuron's synapses onto a single object, vs the proofread **v1507** truth? Scored only
on rooted-soma cells (those with ground truth); the rest of the volume is present as
realistic confusables. Metric = **matched per-neuron confusion**: match each true
neuron to the v117 object holding the most of its synapse *halves*; TP = its halves
there, FN = its halves split into other objects, FP = other neurons' halves in that
object. All 1355 proofread column cells (5.5 M halves), split leakage-safe into
train/eval/gap slabs. Truth v1507 (supported release > 1400); prediction v117 via
`get_roots` at the v117 timestamp, anchored on supervoxel ids.

| region | neurons | v117 objs/neuron | P | R | F1 | **axon-OUT recall** (median/neuron) | dend-IN recall |
|---|---|---|---|---|---|---|---|
| overall | 1355 | 68 | 0.999 | 0.857 | 0.923 | **0.095** (87 % of cells < 0.5) | 0.959 |
| train | 744 | 69 | 0.999 | 0.861 | 0.925 | 0.100 | 0.960 |
| eval (held-out) | 307 | 67 | 1.000 | 0.841 | 0.914 | 0.089 | 0.952 |
| gap | 304 | 63 | 1.000 | 0.863 | 0.926 | 0.093 | 0.964 |

**Three verified conclusions:**

1. **Precision ≈ 1.0, even at full density.** With all 1355 proofread cells packed
   into the column, FP is 0.12 % of halves and there are **0 catastrophic merges**
   (no v117 object is the primary home of two proofread cells) in every region. v117
   essentially never merges two *proofread* cells. (It does merge cells with
   *untruthed* fragments — unscorable against truth, but a real risk for a joiner.)

2. **The error is splits, and it is almost entirely axonal.** Dendritic **input**
   synapses are ~0.95 recall (one v117 object already holds them; median 1 object
   covers 90 %). Axonal **output** synapses are shattered: **median 0.09 recall per
   neuron** — a typical cell has only ~9 % of its outputs on its largest piece, the
   rest scattered across dozens of the ~68 objects (median 35 objects to cover 90 %
   of outputs).

3. **The 0.92 overall F1 is misleading.** Inputs are 83 % of halves and nearly free,
   so they dominate the count. "Synapses on the correct neuron" is ~0.85 by raw count
   but **~0.09 for the outputs that carry the connectome.** So the task is **axon
   reconstruction — joining shattered axonal output pieces without merging cells —
   not merge-cutting.** This is why earlier work at the L2/supervoxel level and in
   tiny boxes was mis-framed. Everything downstream is measured as **axon-output
   recall lift at held precision**, axon and dendrite reported separately.

---

# The v117-object joiner: candidate generation isn't the wall, the discriminator is

*(`v117_proofread.py`, `tests/test_v117_proofread.py`.)*

Given the baseline, the joiner's job is precise: **reconnect a neuron's shattered
axonal output pieces to its arbor without merging cells.** Each v117 object gets a
geometry (its L2 rep-coord point cloud + `max_dt_nm` caliber, via `get_leaves` →
`l2cache`) and its output/input synapse counts. Candidate joins are built **globally**
across all sampled neurons (KDTree over object point clouds within a gap) so
wrong-neuron distractors are real; each is scored by proximity + trajectory
colinearity + caliber match and labelled correct iff the two objects share a v1507
truth. Joins are committed greedy vs merge-aware (`merge_aware_join`, polarity A↔D +
caliber + soma vetoes), sweeping the accept threshold. Metric = **matched
axon-output recall at held precision** vs the oracle (correct-joins-only) ceiling.

**Gap sweep, 30 train neurons (2826 objects; base axon-output recall median 0.098):**

| candidate gap | oracle median | **achievable @ P≥0.999** | greedy max-recall (P) |
|---|---|---|---|
| 2 µm | 0.113 | 0.098 (no lift) | 0.65 (P 0.03) |
| 3 µm | 0.217 | 0.098 (no lift) | 0.73 (P 0.03) |
| 5 µm | 0.436 | 0.098 (no lift) | 0.89 (P 0.03) |
| 8 µm | **0.754** | 0.098 (no lift) | 0.94 (P 0.03) |

The held-out **eval** region reproduces this exactly (oracle-median 0.155→0.432→**0.735**
at 3/5/8 µm; achievable-at-held-P flat at the base; greedy 0.98 at P 0.03) — the
pattern is leakage-proof, not a train artifact.

**The decisive finding (honest negative that localizes the problem):**

1. **Candidate generation is *not* the bottleneck.** The oracle ceiling rises steeply
   with gap — correct proximity joins within 8 µm recover axon-output recall from 0.10
   to **0.75**. The information to reassemble the axon *is* largely present in local
   proximity; the correct joins exist in the candidate set.

2. **The discriminator is the wall.** No hand-crafted geometric cue
   (proximity + trajectory + caliber, with merge-aware polarity/caliber/soma vetoes)
   commits *any* net-positive joins at held precision — achievable recall stays pinned
   at the 0.098 baseline at every gap. Join edge-precision tops out ~0.5–0.58, and
   committing at that precision **collapses neuron-level precision to 0.03** (one wrong
   join fuses two cells through the union-find — the cascade this repo keeps hitting).

3. **Why:** the surviving candidates are *same-compartment* — a true axonal
   continuation vs an adjacent **parallel axon** a couple of µm away. They are
   collinear, caliber-matched and both axon-typed, so geometry, caliber and the A↔D
   polarity veto are all blind to the distinction. This is the identity/same-type
   residual, now quantified at the correct (v117-object) granularity with the correct
   (axon-output recall) metric.

**So the well-posed open problem is a same-type identity discriminator** — the joins
are there (oracle 0.75), but selecting them needs a cue that separates a process from
its parallel neighbour, which no local geometry provides. That is exactly what the
deferred hard-negative identity-embedding plan targets, and this result is the clean
motivation for it (and the honest reason not to claim an F1 lift from geometry alone).

### Local-EM identity discriminator: existing cues fail; a learned one is unproven

On the **real confusable candidate joins** (n≈300, geometry held ~flat in a narrow
high-score band, so this isolates each cue's *marginal* identity signal):

| cue at the two contact faces | AUC (correct vs wrong join) |
|---|---|
| geometry score (residual) | 0.60 |
| **committed cut-face encoder** (type-level) | **0.58** |
| raw 48×48 patch cosine | 0.49 |
| cut-face footprint IoU | 0.49 |

So the *existing* local-EM cues carry no usable identity signal on the confusions —
the type-trained encoder (0.58) barely edges chance and doesn't beat the geometry
residual. That is a valid negative for the committed cues.

A purpose-trained hard-negative **identity** embedding (the plan's untried lever) was
attempted (EM + v117-seg + v1507-seg blocks, leakage-safe train/eval, NT-Xent with
in-window adjacent-process negatives). **Result: inconclusive, not a clean negative** —
the masked mip2 mean-EM "face" representation **fails its own sanity control**: faces of
the *same v117 object* at adjacent z (which must be near-identical) do not retrieve each
other above chance (same-object AUC 0.50), because every masked mean-EM patch is a
centred filled blob whose raw correlation saturates ~1.0 for *all* pairs, and the
type-encoder collapses all thin-axon faces to ~one point. The contrastive loss barely
moved. So whether a learned embedding can decode single-section identity **remains open**
— it needs a representation that preserves within-process texture (unmasked **mip1**
patches, proper augmentation), not the masked mip2 mean-EM used here. Honest status:
the discriminator is unsolved; the confounded attempt is not evidence either way.

---

# Two-cue abstaining auto-proofreader — findings

Error **detection + correction** framed like a trained proofreader: every candidate
edit must satisfy a **global shape grammar** (Pillar 1) *and* a **local EM
ultrastructure** cue (Pillar 2), fused by a **calibrated abstaining combiner**
(Pillar 3), with the residual deferred to a ranked queue. This documents what is
built, what is validated, and — honestly (CLAUDE.md) — what does **not** yet work.

## What is built and tested (offline, no network)

| Pillar | Module | Tests |
|---|---|---|
| 1 global grammar energy | `grammar_energy.py` | `tests/test_grammar_energy.py` (4) |
| 2 local EM ultrastructure | `local_evidence.py` | `tests/test_local_evidence.py` (5) |
| 3 combiner + candidates + queue | `complementarity.py`, `queue.py`, `pipeline.py` | `tests/test_proofread_queue.py` (3) |

## Pillar 1 — grammar energy (validated on real merges)

`grammar_energy` scores ungrammaticality (multi-soma, A↔D-not-via-soma, caliber
jump, disconnection); `cut_delta_energy` / `join_delta_energy` give per-edit ΔE.
On real m343 false-merges that split into two current roots, joining the two pieces
scores **ΔE < 0 (rejected)** when it fuses two somata; clean neurons score E≈0.6–1.6.
**Crucially, same-compartment merges score ΔE ≈ 0 — grammar is blind to them.** That
blind spot is the whole reason for Pillar 2.

## Pillar 2 — local EM cue (validated on real EM at proper cut-face sites)

`local_evidence(pos_a, pos_b, embed_fn)` returns `cutface_sim` (committed contrastive
cut-face encoder over the two cross-sections) and `barrier` (dark-membrane profile
along the connecting axis) from one bulk mip-1 EM+seg fetch.

Re-ID sanity on a real MICrONS box (`experiments/proofread/val_local_evidence.py`): two
z-separated faces of the **same** neurite score mean `cutface_sim` **+0.58** vs
**+0.30** for **different** neurites (separable); `barrier ≈ 0` on continuous
cytoplasm. **The cue carries same/different-process signal at genuine cross-section
sites.**

## Ground truth (Task 23)

`experiments.pcfg.run_synapse_correction.fetch_side_table` over a single 24 µm box at
the proofread column center (v117 → v1718), cached at
`cache/sidetable/col_n1_v1718.npz` (gitignored; ~15 min to refetch). `summarize_edits`:

```
sides=17874  v117_roots=5362  later_roots=5376
split_roots=12   (false MERGES to cut)
merge_targets=2  (false SPLITS to join)
```

## Complementarity result — HONEST, and it does **not** yet clear the bar

Unified "do these two synapse-sides belong to the same cell?" over both strata
(within-root y=0 = false merge/CUT; cross-root y=1 = false split/JOIN), local-site
filter ≤6 µm, leakage-safe GroupKFold OOF logistic. Command:

```
PYTHONPATH=. python -m experiments.proofread.run_complementarity \
  --sidetable cache/sidetable/col_n1_v1718.npz --max-candidates 80 --max-pair-nm 6000 \
  --out out/complementarity_n1.json
```

**80 pairs (24 local cut-errors, 0 join candidates), 57 cell groups:**

| cue | AUC (OOF) |
|---|---|
| shape / grammar (point-cloud, AutoProof-style baseline) | **0.487** (≈ chance) |
| local EM ultrastructure (Pillar 2) | **0.601** |
| joint combiner | **0.536** |

Cue direction is correct among the 54/80 sites with a valid cross-section:
same-cell `cutface_sim` 0.502 vs different 0.405; `barrier` 0.041 vs 0.115. The
ranked queue proposes 8 confident CUTs at **precision 0.125 — below the 0.30 base
rate.** Abstention correctly defers 72/80, but the confident auto-edits are **not
trustworthy on this substrate.**

**Read this straight:** the local-EM cue is the single most informative stream
(0.601 > shape 0.487), which is directionally consistent with the two-cue thesis,
but the joint combiner does **not** beat local-alone and end-to-end auto-correction
is **not** deployable here. This is a negative/underpowered result, not a win.

### Why it fails here (diagnosed, not hand-waved)

1. **Wrong sample point.** Candidates are sampled at **synapse-cleft positions**, not
   on the neurite. 26/80 sites have no cross-section footprint at all (`ok=0`), and a
   cleft cross-section is not the neurite cross-section the encoder was trained on.
2. **Wrong site for a cut.** A false-merge's error is the **seam** between two lobes;
   our within-root pairs are two arbitrary synapse sides (median 11 µm apart), so the
   local cue is not read at the place the human would cut.
3. **Underpowered.** One box gives 24 local cut-errors and **0** usable join
   candidates — too few, and the join direction (where Pillar 2 should shine) is
   untested. Contrast the clean re-ID result above, which *does* separate at proper
   cut-face sites.

The failure traces to **site placement and sample size, not the cue** (which
separates at proper sites) — so the fix is mechanical, not a dead end.

## Update (Task 26): seam-localized test — the site fix does **not** rescue the local cue

I acted on step 1 above and tested it directly (`seam_test.py`,
`run_seam_test.py`): put both cues **on the neurite at real merge seams** and
compare against real continuations.

* **SEAM (should cut):** a real m343 false-merge = two *current* roots wrongly
  joined then split; their closest-approach skeleton points are the historical seam.
* **CONTINUATION (should keep):** two vertices of one clean neuron ~2–6 µm apart
  along the cable.

Result (`out/seam_test.json`; **3 usable seams**, 24 continuations — seam side is
small, read the sign not the third digit):

| cue | seam mean | continuation mean | AUC ("is a seam?") |
|---|---|---|---|
| cut-face sim | **0.627** | 0.552 | **0.40** (wrong direction) |
| membrane barrier | 0.017 | 0.038 | 0.40 (wrong direction) |
| grammar join ΔE | **−0.67** (2/3 rejected) | n/a | — |

**The local cut-face cue does not separate seams from continuations even when
sampled correctly on the neurite — it is at/below chance.** The reason is the
repo's own established fact: cut-face / SegCLR embeddings encode **cell type, not
identity**, and the two processes at a merge seam are *adjacent and usually the
same type* (that similarity is why they got merged), so they look alike. The
earlier +0.58/+0.30 re-ID separation used **easy random distractors**, not the hard
adjacent-same-type negatives that actually occur at seams — so it overstated the
cue. The membrane-barrier proxy likewise carries no seam signal here.

**Grammar (the global cue) is the one that works:** it rejects 2/3 real seams
(the multi-soma ones, ΔE −1) and is blind only to the same-compartment seam — the
known complementarity boundary. So the deployable signal in this project is the
**global shape grammar**, not the local ultrastructure cue as implemented.

### Revised conclusion

The two-cue thesis is **not supported by the local cue we have**. A local cue that
helps must read *identity-bearing* ultrastructure that distinguishes adjacent
same-type processes — e.g. **membrane continuity / topology across the seam** (does
one bounded membrane pass through, or do two membranes appose?) or a RoboEM-style
local trace — **not cross-section appearance**, which is type-confounded. The
cut-face encoder is the wrong instrument for the second cue.

## Update 2: *following* by trajectory inference — this **works** (`follow_test.py`)

If appearance is the wrong instrument, what do humans actually use? Trajectory
momentum + logical reconnection. Tested directly, **offline on skeletons, no EM**
(the appearance trap avoided by construction): cut a real interior point, open a
realistic gap (delete the neuron's own cable within `gap_nm`), and rank every
vertex from *every* loaded neuron in the surrounding annulus — the real pool of
competing processes. True continuation = far side of the gap (same neuron); the
rest are distractors. 189 clean neurons, 2191 cut instances, gap 2 µm.

| scorer | top-1 | hard¹ | confusable² |
|---|---|---|---|
| nearest (proximity) | 0.464 | 0.000 | 0.305 |
| align (direction only) | 0.941 | 0.921 | 0.563 |
| learned (trajectory + caliber) | 0.962 | 0.939 | 0.722 |
| **+ consequence (reciprocal trajectory)** | **0.977** | **0.970** | **0.841** |

¹ *hard* = the 1175 cases where the nearest vertex is a distractor (proximity
misleads). ² *confusable* = the 295 cases with a **parallel, similarly-aligned
distractor** (a fascicle) — the genuinely hard inference. Chance top-1 = 0.425.

**Reading:** proximity is useless (≈ chance); **you can follow a process by
trajectory alone at 0.96 top-1, recovering 94% of the cases where proximity
fails.** Where a parallel process competes (the confusable set), direction alone
drops to 0.56 and trajectory+caliber to 0.72 — the residual is the hard case.

**The consequence layer closes most of that residual.** Adding **bidirectional
consistency** — does the candidate's own cable *point back through the gap at the
cut* (a severed cable is collinear from both ends; a parallel fascicle member is
laterally offset, so its cable does not extrapolate to the cut) — lifts the
confusable case **0.72 → 0.84** and the hard case 0.94 → 0.97. Verified: robust
across 4 seeds (confusable +0.07–0.12 each); a single-feature ablation isolates the
driver as **reciprocal trajectory** (adds +0.16 alone; `tan_agree` adds a little;
`ray_gap` is *useless* — a tight fascicle has a small line-gap, so it favours the
distractor — and was dropped). This is exactly "inference by consequence": for two
ends to be one process, *each* must be consistent with the other, not just locally
plausible. And it needs **no appearance** — pure geometry, no identity, no soma.

The still-open residual (~16% of confusable) is where same-caliber, mutually-collinear
processes genuinely need *global* consequence (does rejecting this strand a fragment /
orphan synapses / fail to reach a soma?) and look-ahead — the next layer.

### Learned, not engineered (`follow_learned.py`) — hand features aren't the point

Hand-crafted align/reciprocal/caliber were only a *proof of signal*; humans don't
compute them, they learn. So we gave a small MLP **only raw coordinates** — the
candidate point, four of its own neighbour offsets (its local cable, unlabelled), and
the two radii, in a canonical frame (cut at origin, incoming trajectory → +x) — **17
raw numbers, no hand-derived feature** — under a *weaker* 5-fold GroupKFold protocol.

| model | overall | hard | confusable |
|---|---|---|---|
| hand features (align+caliber+reciprocal), LOO | 0.977 | 0.970 | 0.841 |
| **MLP on 17 raw coords, no hand features, 5-fold** | **0.975** | **0.969** | **0.875** |

It **matches, and beats on the hard case** (0.875 vs 0.841; robust across seeds
0.84–0.91) — with less training data and nothing engineered. The net *discovers*
"continue straight", "the far end must point back", and "caliber matches" from raw
geometry, and finds nonlinear structure (candidate-cable curvature) the six linear
features missed. **The follow function is learnable end-to-end; hand features were
scaffolding.** Next step is the natural one: a learned tracer over raw geometry **and
raw EM**, supervised by the edit log — see the correction below.

### Correction: "local EM is useless" was overstated

The seam test (Update 1) showed a *specific* pretrained appearance-cosine — the
contrastive cut-face encoder, trained for z-gap **re-ID** — is type-confounded at
merge seams, on **n=3** seams. That does **not** establish that local ultrastructure
carries no identity signal; three seams prove little, and that encoder was optimised
for the wrong objective (whatever separates faces across a z-gap: type, staining,
context), not the fine membrane/ultrastructure a proofreader reads to tell two apposed
processes apart. Humans demonstrably use local EM, so the signal exists. The honest
statement is **"the re-ID appearance cosine is the wrong local model," not "local EM
does not help."** The right test mirrors the geometry result: **learn the local cue
from raw EM, supervised by human edits** (and read *membrane continuity / topology*,
not cross-section appearance) — not a hand-picked pretrained cosine. That is an open,
promising direction, not a closed door.

### First attempt at learned local EM (`follow_em.py`) — the cheap version does *not* help

Took the honest next step: add **raw EM sampled along each candidate corridor** (mip-1
intensity mean + membrane-catching min over a small ball, K=16 steps → 32 raw numbers,
no seg mask, no pretrained cosine, no hand threshold) as extra input channels to the
learned geometry model, on an EM-fetched confusable-enriched subset (110 instances,
56 confusable, one fetch each).

| model | overall | hard | confusable |
|---|---|---|---|
| geometry (raw coords) | 0.845 | 0.862 | 0.732 |
| **geometry + raw-EM corridor** | 0.791 | 0.793 | **0.643** |

**Adding the EM corridor made it worse, not better.** Two honest reasons, verified:
(1) *the corridor is the wrong reader* — a straight chord between two skeleton points
on a thin (~200 nm), curving neurite **leaves the process even for a true
continuation**, so corridor intensity does not cleanly separate true from distractor
(a minimal repro on 5 confusable instances showed near-identical true/distractor
profiles: mean 0.53 vs 0.53); (2) 32 noisy features on 110 instances overfit. Note the
earlier "membranes < 0.30" eyeballing was *my* mis-calibration — MICrONS EM is
low-contrast (bulk 115–144/255, membranes ~100), fixed here by raw min-over-ball
sampling; the learned model still found no usable signal in the straight corridor.

**What this does and does not show (learning from the earlier overclaim):** it shows
the *cheap straight-corridor* learned-EM feature does not add signal on this task — it
does **not** show local EM is useless. The field's working local-EM method (**RoboEM**)
is a learned local *flight* that a 3D CNN steers to *follow the process*, not a straight
chord — precisely because the naive corridor fails. So the honest conclusion stands:
extracting the local-ultrastructure cue needs a **learned tracer that follows the
neurite** (RoboEM-style), a substantial 3D-CNN build, not a corridor feature. The
geometry follow model remains the solid, cheap, positive result; local EM is a real
but *harder* second cue that this first cheap attempt does not unlock.

### The right local cue: real CUT FACES on the following slices (`cutface_slices.py`)

The corridor was a **blind projection** — the mistake was ignoring that we *have the
segmentation objects*: a fragment ends in a real 2-D **cut face** (its segmented
cross-section on its terminal slice) and the continuation is a real object
cross-section on the *following* slices.  Following = linking those real footprints
slice-to-slice, which the straight line threw away.

Honest, non-circular test: cut a real seg object at a z-slice and re-link its cut face
to the true continuation on slice ``z0+gap`` against every other object's footprint
there — matching by **footprint geometry only** (IoU / centroid / area); the seg id
defines truth and segments footprints but is never a matching feature. One mip-2 seg
box (250×250×200 vox, ~34 candidates per cut, chance 0.04):

| z-gap | IoU top-1 | centroid top-1 | IoU top-1 (confusable) |
|---|---|---|---|
| 40 nm  | **0.995** | 0.90 | 0.994 |
| 120 nm | **0.769** | 0.62 | 0.759 |
| 240 nm | 0.430 | 0.37 | 0.411 |

**The real cut face links the process almost perfectly slice-to-slice (0.995 @ 40 nm),
holds up on confusable cuts (a distractor footprint near the cut face), and beats
centroid — pure footprint geometry, no appearance, no type-confounding.** It decays
over big gaps (240 nm) as the cross-section drifts/thins — exactly where the
*trajectory* model carries you. The two cues are complementary in the obvious way:
**follow contiguously by cut-face overlap where slices are close; extrapolate by
trajectory across the gap where they aren't.** This is the working local cue the
appearance-cosine and blind-corridor attempts were groping for — and it was in the
segmentation the whole time.

### The combined follower: learned fusion of cut-face + trajectory (`evaluate_follow_fused`)

Fused the two cues per candidate: raw cut-face IoU, **motion-compensated IoU** (shift
the cut face by the fragment's own extrapolated z-drift, then match shape),
trajectory-position distance, and area — a leakage-safe logistic (GroupKFold by cut
slice) weights them.  Top-1 vs each single cue, by gap (mip-2 seg box, ~46 candidates
per cut, chance ~0.02):

| z-gap | cut-face | trajectory | motion-comp | **learned fusion** |
|---|---|---|---|---|
| 120 nm | 0.767 | 0.697 | 0.775 | **0.859** |
| 240 nm | 0.420 | 0.487 | 0.518 | **0.576** |
| 400 nm | 0.233 | 0.319 | 0.329 | **0.371** |
| 600 nm | 0.138 | 0.210 | 0.211 | **0.238** |
| 800 nm | 0.099 | 0.160 | 0.142 | **0.177** |

**The fused follower is best at every gap** — the single cues trade off (cut-face wins
short, trajectory wins long, motion-comp bridges the middle) and the learned combiner
takes the best of each automatically.  At the operational gap (120 nm ≈ 3 sections) it
reaches **0.859** vs 0.775 for the best single cue.  Accuracy falls with gap because
re-linking across 800 nm among ~46 candidates is genuinely hard (chance 0.02) — but the
message is the shape: **cut-face contiguity ⊕ trajectory extrapolation, fused, beats
either alone across the whole range, using only geometry — no appearance, no
type-confounding.** That is "follow like a human," and every ingredient (cut faces,
trajectory, the fusion) is real and learnable.

### Ultrastructure channel + the connect/abstain decision (`evaluate_follow_ultra`)

Two things we were *not* doing: (1) reading the **ultrastructure inside** the cross-section
(only the silhouette); (2) deciding **whether to connect at all** (forced top-1 → a real
tip becomes a false merge).  Added both.

**Ultrastructure** — interior EM content correlation (motion-compensated over the
footprint overlap) + a myelin/membrane ring feature, appended to the geometry fusion,
at **mip-1** (16 nm, organelles resolved), 20 662 candidates:

| z-gap | fused (geometry) | + ultrastructure |
|---|---|---|
| 120 nm | 0.957 | 0.957 |
| 240 nm | 0.767 | 0.774 |
| 400 nm | 0.472 | 0.480 |

**No measurable lift** (±0.01).  Honest reading — and *not* "ultrastructure is useless"
(third time a content cue hasn't beaten geometry, so state the caveats precisely): (a)
geometry already *saturates* at small gaps (0.957 @ 120 nm) — no headroom, and that is
where footprints overlap; (b) my interior-correlation is computed **over the footprint
overlap**, so it is redundant with IoU and, worse, **cannot be computed at large gaps
where there is no overlap** — exactly where ultrastructure *should* matter; (c) cortical
neuropil is largely unmyelinated, so the ring feature is mostly uninformative.  The real
test of ultrastructure is a **learned organelle/texture matcher at large gaps** (track a
specific mitochondrion across the break, no overlap required) — not a hand-correlation
where geometry already wins.  That remains open; this cheap version does not move it.

**Connect vs. abstain (the cost of failing to connect).** Terminal cut faces (the object
does *not* continue) are included as all-negative instances; the follower's top-candidate
fused score separates "has a true continuation" from "is a tip" at **AUC 0.836**.  So the
follower *knows when not to connect* — the basis for the asymmetric-cost decision
(`treestitch.risk`, merge cost ≫ split cost): set a high connect threshold, auto-fix only
confident continuations, **abstain** on the rest (accept the cheap split, never the
expensive merge).  This is the honest deployable posture — precision via abstention — now
grounded in a real connect-confidence signal, not a forced top-1.

**Grammar consistency (still a gap).** The follower ranks by geometry; it does **not** yet
veto *ungrammatical* connections (axon↔dendrite, caliber jump, would-create-two-soma).
Wiring Pillar-1 `grammar_energy` as a **hard veto** on the chosen candidate (reject if the
join raises the grammar energy) is the missing consistency filter — cheap to add on top of
the ranker, and the natural next step alongside the abstention threshold.

### The three-stage follower, end-to-end (`evaluate_follow_pipeline`)

Wired all three: **rank** (fused geometry) → **grammar veto** (reject the top pick on a
caliber jump — cross-section area ratio > 2.5 — or a two-soma join; Pillar-1's caliber/soma
terms on the real cross-section areas) → **abstain/commit** (commit only above a score
threshold, else leave the gap).  Terminal cut faces (objects that don't continue) are
included, so committing to one is a false merge.  Precision (commits that are the correct
true continuation) vs coverage (real continuations fixed), sweeping the threshold, mip-2,
24 246 cuts (21% terminal):

- **The grammar veto helps.** It fires on ~19% of top picks and kills **false connects 2.2:1
  over true** (3173 vs 1439); at matched coverage it lifts precision **0.69 → 0.73**.
- **Precision ceiling is ~0.73–0.84, below the P≥0.95 bar.** Restricting to the cleanest
  single-jump regime (gap 120 nm) reaches P=0.84 @ 0.75 coverage; the mixed (120+240 nm)
  config tops ~0.73.  The residual is terminals whose nearest neighbour has high footprint
  overlap (an unavoidable local false merge) and multi-section-gap ranking errors.

**Honest reading.** The architecture works and each stage pulls its weight — but this
config is **not yet deployable-precision**.  The gap between here and the 0.995 contiguous
linking (`evaluate_cutfaces`, 40 nm) is the design lesson: **follow section-by-section
(gap 1) where linking is near-perfect, and only invoke the harder gap-jump at a real
break.**  Reaching high precision then needs (a) contiguous tracking not section-skipping,
(b) **global assignment / matching** so each cut face claims ≤1 partner (kills the terminal
false-merges a greedy top-1 makes), and (c) the grammar veto with **real compartment
labels** (synapse polarity), not just caliber.  All three are known, scoped next steps —
the end-to-end skeleton (rank → veto → abstain) is now built and measured.

### Global matching — the safety mechanism, and a deployable operating point (`evaluate_follow_matching`)

Greedy top-1 lets a candidate be claimed by many cut faces, so a terminal tip whose
neighbour overlaps it becomes a false merge.  **Global one-to-one matching** (assign
confident edges first, each face on *either* side claims ≤1 partner) fixes exactly that:
the neighbour is taken by its own true continuation, the tip is left unmatched.  Bipartite
weight = motion-compensated cut-face IoU; sweep the accept threshold; precision (committed
matches that are same-object) vs coverage (true continuations recovered).  mip-2, one box:

| regime | metric | greedy | **global** |
|---|---|---|---|
| **gap 40 nm (contiguous)** | coverage @ P≥0.99 | 0.900 | **0.949** |
| gap 40 nm | coverage @ P≥0.95 | 0.952 | **0.972** |
| gap 40 nm | precision @ cov≥0.85 | 0.996 | **0.998** |
| gap 120 nm | precision @ cov≥0.70 | 0.820 | **0.892** |
| mixed 40+120 nm | precision @ cov≥0.85 | 0.860 | **0.929** |

Two results.  **(1) Global > greedy everywhere**, most where it matters — high coverage,
where greedy's reuse conflicts bite (+0.05–0.07 precision at cov≥0.85; +0.02–0.05 coverage
at P≥0.95).  One-partner-per-face is the mechanism that makes auto-apply safe.  **(2) The
contiguous regime is deployable:** following **section-to-section (40 nm)** with global
matching auto-links **94.9% of continuations at P≥0.99**.  That is the landmark posture —
joining fragments at scale without introducing mergers — and it confirms the design:
*follow contiguously where linking is near-perfect; invoke the harder gap-jump only at a
real break.*

**Honest caveats.** These are *artificial* cuts of continuous objects; a **real** false
split broke for a reason (misalignment, damage), so real breaks are harder than a clean
40 nm cut — the 0.99@0.95 is an upper bound, not the real-split number.  Still mip-2, one
box, truth defined by seg id.  The next step is to run the matcher on **real** false
splits (the reconnections proofreaders made) and report synapse-pair F1 before/after — but
the mechanism (global matching ≫ greedy, contiguous ≫ gap-jump) is real and measured.

### The real-split test — done with CAVE v117→v1822 (`real_splits.py`)

*(Correction of a prior wrong note that claimed this wasn't doable: it is — fetch the
graphene cutout `agglomerate=True` at the v117 timestamp for fragments and at v1822 for
the proofread grouping, the same v117↔later machinery used throughout this repo.  There was
no data wall; the earlier reasoning was simply mistaken.)*

Dense voxel ground truth: v117 id = fragment, v1822 id = proofread neuron; a **real false
split** = a v1822 root gathering ≥2 v117 fragments.  18 µm box, mip-3: **11 803 fragments,
15 real splits.**  Re-linked v117 fragments by motion-compensated cut-face IoU with global
matching; a link is correct iff the fragments share a v1822 root.

**Re-anchor on the goal: synapses on the correct neuron.**  The point of split-fixing is
the synapse-pair line-graph F1 (`line_graph.compute_line_graph_f1`), so a fragment matters
exactly insofar as it carries synapses — a tiny synapse-bearing fragment is as valuable as
a dendrite.  Computed the real metric on the cached column SideTable, grouping synapse
endpoints by v117 (fragmented) vs the proofread truth (later version):

```
synapse endpoints = 17 874
F1_before (v117 vs proofread truth):  P=0.989  R=0.993  F1=0.991
false-split neurons = 2;  synapse endpoints on them = 147 (0.8%)
recall gap closeable by fixing splits: 0.007  (R 0.993 -> 1.0 oracle)
```

**CORRECTION — that 0.991 was the wrong baseline.**  v117 in this column is *already
proofread* (v117 roots 604 ≈ v1822 roots 609; `summarize_edits` = 14 changed roots of
5362), so "v117 vs v1822" measured proofread-vs-proofread ≈ 1.0 — and 0.991 impossibly
*exceeds* this repo's oracle ceiling (0.928), the tell that it is not a real "before."
The real fragmented baseline is the **L2 chunk level** (the fragment unit behind the
oracle-0.928 / greedy-0.14 anchors).  Mapping the same synapse supervoxels to L2 fragments
vs v1822 roots (10 µm box, 1264 endpoints, 1042 L2 fragments, 609 neurons):

```
L2-fragment baseline vs v1822 truth:  P=1.000  R=0.132  F1=0.234
```

**So the column is NOT solved — at the fragment level F1 is 0.234, with enormous headroom
(→ 0.928 oracle).**  Precision is perfect (L2 fragments never wrongly merge); recall is
13% because each neuron is shattered into ~1.7 fragments per synapse.  **Every point of
recall = a synapse-bearing fragment correctly joined** — which is exactly why tiny
synapse-carrying splits are the whole game.  This is the correct substrate the whole effort
should have been on: **join L2 fragments to raise synapse-pair recall toward 0.928 without
dropping precision** (greedy joining collapses P to 0.14 — the trap).  The geometry follower
(cut-face + trajectory + global matching + grammar veto + abstention) is precisely a
precision-preserving joiner; the goal-aligned test is its **synapse-pair F1 before (0.234) →
after, at held precision**, on L2 fragments with v1822-root truth.

### The goal-metric result (`synapse_f1_join.py`) — the follower does NOT crack it at scale

Ran exactly that: L2 volume (10 µm box, 632 synapses), join L2 fragments by global
cut-face matching, sweep the accept threshold, synapse-pair F1 vs v1822 truth.

| threshold | joins | join precision | recall | **F1** |
|---|---|---|---|---|
| before (L2) | — | — | 0.132 | **0.234** |
| 0.10 (join freely) | 56 999 | 0.14 | 0.983 | **0.007** |
| 0.52 (best F1) | 2 101 | 0.49 | 0.148 | **0.256** |
| 0.80 (high-precision) | 372 | 0.81 | 0.139 | **0.245** |

**The follower does not deliver a deployable synapse-F1 lift on real fragments at scale.**
Best F1 is 0.256 vs 0.234 — a 0.02 gain — and it reproduces the field's known wall: join
freely and recall hits 0.98 but **F1 collapses to 0.007** (the greedy-agglomeration
precision cascade, matching this repo's 0.14 anchor); tighten to precise joins and coverage
vanishes (F1 barely moves).  **Join edge-precision never reaches 0.90.**

**Why — the honest mechanism.** Split-fixing is *cascade-sensitive*: a partition F1 needs
near-perfect join precision because one wrong join merges two whole neurons and destroys
many synapse pairs at once (union-find is unforgiving).  The local cut-face follower gives
0.6–0.8 edge precision on dense, adversarial-by-construction L2 fragments; that is nowhere
near the ~0.99 needed to avoid catastrophic cascades.  The 0.99 @ 0.95-coverage headline was
on *artificial single cuts of isolated continuous objects with curated candidates* — the
prior `FINDINGS_synapse_correction.md` (de-split geom AUC 0.988) had the same optimism and
explicitly flagged that its non-adversarial negatives overstate the case.  At real scale,
with every nearby fragment a candidate, the local geometry cue is not precise enough.

### Honest bottom line of the whole two-cue effort

- **Global grammar** (multi-soma, A↔D, caliber): real but narrow — catches gross errors.
- **Local geometry** (cut-face + trajectory + global matching): strong on curated/artificial
  cuts; **at scale on real fragments it hits the precision/recall wall** (F1 0.234 → 0.245 at
  usable precision; collapses if pushed for recall).
- **Local appearance / ultrastructure** (cut-face cosine, EM corridor, interior content):
  type-confounded or redundant — no measured lift.

The landmark — join fragments at scale without introducing mergers, for a large synapse-F1
gain — is **not achieved by these local cues**.  That is consistent with why automated
proofreading is an open problem: the deployable bar is ~0.99 join precision to beat the
cascade, and local geometry alone does not reach it.  A real advance needs either a much
stronger (likely learned, context-rich) join classifier or a fundamentally non-greedy global
formulation with calibrated abstention that commits *only* the ~near-certain joins — the
honest next direction, now measured against the correct baseline (F1 0.234) and metric.

### Merge-aware constrained joining (`merge_aware_join.py`) — vetoes don't lift F1, and *why*

Tried the "recognize merges and defer that subtree" idea: confident-first union-find that
commits a join only if the resulting component stays grammatical — **A↔D veto** (reject an
axon-typed↔dendrite-typed join with no soma, type from synapse polarity: pre side = axon,
post side = dendrite), **2-soma veto**, **caliber veto**, **quarantine** of
merge-contaminated fragments.  10 µm box, greedy vs merge-aware, per-veto ablation:

```
BEFORE (L2):            F1=0.234
GREEDY best:            F1=0.257  join_P=0.61
MERGE-AWARE best:       F1=0.250  join_P=0.48   (no better; the vetoes barely fire)
ablation: no_ad / no_soma / no_quarantine  all ≡ full;  no_vetoes = greedy 0.257
typing:  axon=64  dend=79  contaminated=1  soma_frags=0
```

**The vetoes do not help — and the reason is the useful finding.** Only ~143 of ~1000+ L2
fragments could be **typed at all**: an L2 fragment is so small it carries ≤ 1 synapse, so
synapse-polarity typing covers almost nothing, `soma_frags=0` (no soma-scale L2 in the box),
and `contaminated=1`.  With almost no typed fragments, the A↔D veto has nothing to act on
(removing it is identical to keeping it).  The dominant join errors are therefore between
**untyped, same-compartment** fragments — two *different* axons whose cut-faces overlap in
dense neuropil (the "confusable" parallel-process case) — which **no grammar/polarity veto
can catch**.

**Precise diagnosis (the value of the negative):** the merge signature others might exploit
(polarity / grammar) is real but lives at the *neuron* scale, while the errors that cascade
live at the *L2-fragment* scale where that signal is absent.  Two honest implications:
(1) the veto idea could still work if applied to **partially-assembled components** once they
accumulate enough synapses to be typed — but the damaging joins happen *before* components
grow that large, so it's chicken-and-egg; (2) the residual wrong joins are same-type
parallel processes, exactly the case local appearance is type-confounded on — so the missing
ingredient is genuinely a **finer identity discriminator between same-type neighbors**, not
another grammar constraint.  That is the crisp statement of what an advance must provide,
measured on the correct baseline (F1 0.234) and metric.

**What is genuinely proven / in hand:** the follower cues (cut-face contiguity, trajectory,
global matching) are real and strong on controlled cuts; the **real v117→v1822 split ground
truth is now assembled** (`fetch_agglo_volume` + `split_truth`, 15 real splits with sizes
and synapse counts).  The honest next build is a **synapse-weighted** split-fixing demo:
take a rawer starting grouping (or a less-proofread region with proofread truth), fix
splits with the follower, and report **synapse-pair F1 before/after at held precision** —
the metric that was the goal all along.  Not the geometric top-1 numbers this thread
drifted into.

Honest caveats: 189 scattered clean neurons are far sparser than real neuropil, so
the confusable fraction (13%) and the numbers on it are **optimistic** — dense tissue
has more parallel distractors. Clean skeletons, not messy fragments. The models are
leave-one-neuron-out logistics over 4–6 geometric features — a proof of signal, not a
system. But the direction is clear and the opposite of the appearance result:
**geometry/trajectory (both forward and reciprocal) carries identity signal;
cross-section appearance does not.**

## Honest next steps (revised)

1. **Build the "follow" model out** — this is the promising direction. Bidirectional
   trajectory already closed most of the confusable residual (0.72 → 0.84). Next add
   the *global* consequence features (does a candidate reconnect toward a soma? does
   rejecting it strand a fragment / orphan synapses?) and a short look-ahead down each
   option; measure how much of the remaining ~16% they close. Then train it as a
   ranker on the real edit log (v117→later), supervised by the human's actual
   reconnections — appearance excluded. Needs a *fragment-level* setup (cut cells into
   pieces) so soma/stranding features are non-circular, and clean radius/soma-table
   data (CAVE nucleus table), not the patchy skeleton radius.
2. **Lean on grammar** for merge-side deployment now: it carries real signal (rejects
   ungrammatical merges); wire confident grammar-rejected merges through **matching**
   (not agglomeration) and report synapse-pair F1 before/after vs oracle 0.928 /
   greedy 0.14.
3. Denser distractor sets (more neurons / a reconstructed sub-block) to replace the
   optimistic sparse-skeleton floor in the follow test with real neuropil density.

## Positioning

Baseline = shape/synapse (AutoProof-style); our added cue = local EM ultrastructure.
The decomposition and abstaining queue are built and tested; the *complementarity
claim is not yet supported by data* and is gated on step 1 above.
