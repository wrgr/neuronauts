# What is actually in the first forty experiments — a concept audit of the attic

**Scope.** The 25 early benchmarks `attic/benchmarks_semi_synthetic/benchmark_exp0{21,26..50}*.py`
and the 26 engines they drive in `attic/morpho_grammar/`.

**How it was checked.** All 26 engines were read in full. The 25 harnesses were
read at the four points that decide what a benchmark means — where the world is
built, where the candidate pool is chosen, what enters the scoring call, and how
credit is assigned — supplemented by a grep across all 25 for hardcoded metric
rows and asserted rates. Where §1 states a count or quotes a line, that line was
opened and read; the four claims carrying the most weight (§1.2, §1.3, §1.4,
§1.9) were each re-checked with a grep across all 25 before being written down.
Line numbers cited for individual harnesses in §1.6 and §1.7 come from a
file-by-file pass over eleven of them and were spot-checked, not each reopened.

**This audit is about concepts, not scores.** The scores are not evidence and
this document does not repeat them. Two independent reasons, both already
established in `docs/threads/experiment_survey.md` §1C and re-confirmed here:
the benchmarks cut intact skeletons in software, so the two halves of a "split"
still share geometry that real v117 fragments do not; and 25 of the 26 engines
contain no checkpoint-loading code, so they were scored at random
initialization. Nine further findings are added below, in §1. Together they
account for the distance between this era's headline — "~85–87% pairwise merge
accuracy" — and the ~0.09% precision measured on real v117, more completely than
the synthetic-damage argument does on its own.

**What the audit is for.** To decide, concept by concept, what is worth staging
as a real experiment and what can leave the tree. Verdicts are one of three:

| verdict | meaning |
|---|---|
| **REVIVE** | the idea and enough of the code survive; it can be pointed at real soma-seeded data with work measured in hours, not weeks |
| **REVIVE-WITH-TRAINING** | the idea survives; the implementation is a hand-set guess that was never fitted, so it needs real labels before it means anything |
| **DEAD** | either refuted at the real operating point, or the implementation does not do what its name says and the idea is better served elsewhere |

Nothing here recommends anything already refuted. The refutations taken as
settled: pairwise local geometry at a grower's frontier (`results/EXP-081/`),
SegCLR as a selector (`results/EXP-080/`), synapse density/spacing/polarity as a
*pairwise* score, and the `object_clouds_mip5.npz` substrate. The one positive
template taken as settled: Cajal/Murray caliber conservation validated on 3,781
real bifurcations (`results/EXP-084/`), which scores a **tree**, has **zero
parameters**, and whose evidence **compounds** over many branch points.

---

## 1. What the harnesses actually measured

Nine mechanical findings about the benchmark scripts themselves. These are not
opinions about the era; they are what the code does, and together they explain
the gap between "~85–87% pairwise merge accuracy" here and ~0.09% precision on
real data far better than the synthetic-damage argument alone.

### 1.1 "Top-1 correct" meant "hit either sibling," at every endpoint

Each cell is bisected into three pieces. The harness then walks **every leaf
endpoint of the soma piece** and asks the engine for a continuation. Credit is
assigned like this (`benchmark_exp035_restored_dual_engine.py:293`, and the same
shape in exp026–exp048):

```python
child_ids = [t['fragment_id'] for t in toks if t is not soma_tok]
...
if res["predicted_id"] in child_ids:
    top1_correct += 1
```

`child_ids` holds the cell's other **two** pieces. So a prediction counts as
correct if it lands on *either* sibling, at *any* endpoint, regardless of which
endpoint the true continuation actually attaches to. A soma piece with forty
leaf tips contributes forty scored decisions with the same two acceptable
answers, and thirty-eight of those tips have no true continuation at all.

That is not a join decision. It is "is the right cell somewhere in the pool,"
scored forty times. `results/EXP-081/` measured the real version of the same
question: 2,137 frontier tips across 40 cells, 34 of them live — a 1.6% base
rate, where the honest answer at 2,103 tips is *nothing continues*. The old
harness never posed that question, so no engine in the attic was ever asked to
abstain and none was penalized for failing to.

### 1.2 The synaptic partner identifiers are a one-hot encoding of the answer

This is the single most consequential finding in the audit. Synapses are not
fetched; they are generated per fragment, and the partner identifiers are
derived from the cell counter — in **all 25** scripts
(`benchmark_exp031_calibrated_precision.py:119`; exp049 and exp050 use their own
cell index instead of `obj_counter`, with a block size of 10 and 12 rather than
15, but the construction is the same):

```python
partner_base = obj_counter * 100
partner_ids = np.array([partner_base + rng.integers(0, 15) for _ in range(n_syn)])
```

Every fragment of cell *N* draws its synaptic partners from the fifteen-element
block `[100N, 100N+14]`, and no other cell ever touches that block. So the
"Bipartite Synaptic Partner Jaccard Overlap" — the feature the era called
synaptic fingerprinting — is exactly zero for every cross-cell pair and
reliably positive for every same-cell pair. It is a disjoint label encoding
wearing a biological name.

It is not a small term. `tree_grammar_infiller.compute_candidate_score` gives it
`+ 3.0 * jaccard_syn` in the composite logit — the largest non-attention
coefficient there — and `mcts_handshake_engine` weights its log-odds at 1.2 to
2.0. Any engine using the Jaccard term had a near-perfect oracle available to
it.

The polarity signal is fabricated the same way and is even more degenerate
(`:118`):

```python
syn_types = np.zeros(n_syn) if is_axon else np.ones(n_syn)
```

`is_axon` is just `piece_index == n_pieces - 1`. So a fragment's presynaptic /
postsynaptic balance is a deterministic function of its position in the split,
with no noise at all — which means the compartment typer (`type_segment_v2`) and
every polarity veto were reading the answer key too.

This reframes the "synapse density/spacing/polarity scores at chance" result on
real data. It is not that the synthetic world was optimistic about a real
signal; it is that the synthetic world contained a *different* signal, one that
does not exist in tissue.

### 1.3 The corpus silently falls back to fabricated cells

Every harness that claims "150 real proofread Minnie65 neurons" does this
(exp031:102-104, and the same three lines in exp032, exp033, exp038, exp041,
exp042, exp043, exp045, exp046, exp048):

```python
skel = load_skeleton(root_id)
if skel is None:
    skel = get_biofidelic_skeleton(root_id, rng)
```

`load_skeleton` needs a CAVE bearer token and returns `None` on any failure
(`neuronauts/data/loaders.py:258-271`). With no token or no network, one hundred
percent of the corpus becomes the in-file random-walk generator, while the
banner still prints "real proofread neurons." `get_biofidelic_skeleton` takes
`root_id` and never uses it, so the fabricated cells are not even a function of
which real cells were sampled. `benchmark_exp047_hungarian_bipartite.py:180`
does not call `load_skeleton` at all — it calls the generator unconditionally,
while still importing `load_skeleton` at line 17.

There is no assertion, no warning, and no recorded count of how many cells came
from each source. A run's output cannot be distinguished from a run with no
network access.

### 1.4 The "strict 3-way inductive split" fits nothing

The banner is printed by every harness. The mechanism
(`benchmark_exp031_calibrated_precision.py:146-152`):

```python
n_train = int(round(0.60 * obj_counter))
n_val   = int(round(0.20 * obj_counter))
train_pieces = [p for p in pieces_rec if p['obj_id'] <= n_train]
test_pieces  = [p for p in pieces_rec if p['obj_id'] >  (n_train + n_val)]
print(f"[3/4] Strict 3-Way Inductive Split: {len(train_pieces)} Train Frags ...")
```

Two things. First, `obj_id` is a sequential counter assigned in the load loop —
the root identifier never enters `pieces_rec` — so the split is over an
enumeration order, not over cells, and carries no spatial or biological
disjointness. Second, `train_pieces` is referenced exactly twice in the entire
file: the assignment and the `len()` inside that print. Nothing is fitted on it,
no threshold is chosen on validation, and the engines were already untrained.
"Inductive protocol" is a label over a list comprehension whose output is
discarded.

### 1.5 The "micro-electron-microscopy verifier" never reads a voxel

`neuronauts/global_merge/represent/cloudvolume_em_sampler.py` is named for
CloudVolume and imports it nowhere. `VolumetricEMSampler.sample_bridge_volume`
takes `is_true_continuation: bool` and branches on it:

```python
if is_true_continuation:
    radial_coherence = 0.88 * dist_attenuation ...
else:
    radial_coherence = 0.15 ...
```

It returns a random number drawn from one of two distributions selected by the
ground-truth label, under a docstring that says it computes "3D directional
intensity gradient tensors." Three engines consume it —
`dual_engine_infiller.py:78`, `cajal_geodesic_dual_engine.py:86`,
`geodesic_em_tracer.py:74` — each passing `gt_target_id` down as
`is_true_continuation`. Any benchmark using them (exp021, exp026, exp027) was
handed the answer.

`blind_geodesic_em_tracer.py` is the honest sibling: it removes the label. What
remains is a Hermite spline plus a product of cone alignment, tortuosity,
distance decay and caliber ratio. It reads no image either, and every term in it
is the local pairwise geometry that `results/EXP-081/` measures at 0% precision
in the top 34.

### 1.6 Ground-truth compartment labels ride on the scoring tokens

Nine harnesses write `gt_type` and `is_glia` onto the very token dictionaries
they then pass in as the candidate pool: exp033 (`:239` → pool at `:302`),
exp038 (`:273`, `:282` → `:303`), exp041 (`:253`, `:262` → `:277`), exp042,
exp043, exp045, exp047 (`:309`, `:318` → the open pool at `:327`), exp048
(`:386`, `:395` → `:482`). Whether an engine reads those keys is a per-engine
question — `frankenmerge_resolver.py:54-56` demonstrably does — but the labels
were in the room.

`benchmark_exp046_hierarchical_dual_santiago.py:405-409` is the unambiguous
case: it passes `gt_map=gt_map` directly into `assemble_hierarchical_connectome`,
which uses it at `hierarchical_dual_santiago.py:155-161` to emit "active oracle"
joins. Ground truth enters the link-producing path, not the evaluation.

### 1.7 Comparison rows and legality rates are printed as literals

The Lifted Multicut / SegCLR / RoboEM columns in exp040, exp044, exp047 and
exp049 are dictionaries of constants copied verbatim between four files:

```python
res_segclr = {"name": "SegCLR (2021)", "ari": 0.2640, "precision": 0.5230, ...}
```

No SegCLR model is run. exp047 and exp049 mark them with a dagger and a footnote
saying they are published numbers on a different sample; exp040 and exp044 do
not. The practice goes further than the literature rows:

- **exp040** prints its entire closing scorecard as string literals, including
  `Axon-Dendrite Chimera Rate: 0.00%` and `Glial Non-Synaptic Exclusion Purity:
  100.00%` (`:596-611`).
- **exp031** (`:409`), **exp032** (`:427`), **exp041** (`:410-411`), **exp042**
  (`:423-424`) and **exp043** (`:424-425`) print their syntax-violation, glial-
  merge and chimera rates as literal `0.00%` — the headline safety claims of the
  whole era, asserted rather than measured.
- **exp038** (`:494`) prints `Glial False Merges into Neurons: 0 (100.0% Glial
  Exclusion)` while fifteen fabricated glial distractors sit in the candidate
  pool.
- **exp045** passes literal `cleaved_franken=12` and `=15` for its baselines
  (`:352`, `:363`); **exp046** computes 2 of its 7 scorecard rows and hardcodes
  5 (`:414-418`); **exp048** computes 1 of 7 and hardcodes 6 (`:495-500`), and
  divides its latency by a hardcoded `1573.0` rather than a counted number of
  cuts (`:486`).
- **exp041**, **exp042** and **exp043** make an automated KEEP/REVERT decision
  against hardcoded prior values (`prev_f1=0.7456, prev_ari=0.4556,
  prev_erl=3828.4`), so the optimization loop compares each round against a
  constant rather than against a re-measured predecessor.

The AutoProof and NEURD "baselines" (`autoproof_baseline.py`,
`neurd_baseline.py`, 83 and 85 lines) *are* computed, but they are
nearest-neighbor-within-a-cone heuristics sharing no code, parameter or
published behavior with the systems they are named for. Only exp047 and exp049
label them as proxies in the output.

### 1.8 Two benchmarks that name real data and fabricate it

- **exp049** ("DENSE SPATIALLY-DISJOINT v117 SUBVOLUME", "Ground-truth
  evaluation via v1412 proofread labels") imports `fetch_v117_region` and never
  calls it. Its only world comes from `generate_dense_subvolume_fallback`, which
  draws random walks. No v1412 label is read anywhere in the file.
- **exp050** ("150 real Minnie65 neurons", stratified by subtype) builds every
  cell in-file: `generate_pyramidal_skeleton`, `generate_basket_skeleton`,
  `generate_martinotti_skeleton`, `generate_vip_bipolar_skeleton`.

Both are already recorded as synthetic in `experiment_survey.md` §1C. What is
added here is that both name a real substrate in their own docstrings, and that
under §1.3 the difference between these two and the other twenty-three may be
nothing more than an absent token.

### 1.9 The one Cajal term that works was never used

`cajal_conservation_priors.py` exposes three priors. `results/EXP-084/`
validated the caliber law on 3,781 real bifurcations and measured the
bifurcation-angle prior at AUC 0.603. Grepping every caller in the repository:

```
compute_bifurcation_angle_prior    → called only by a test and scripts/test_cajal_conservation.py
compute_santiago_spine_shaft_score → never called
compute_conduction_time_prior      → called by all 6 engines that use Cajal at all
```

So the engines used only the **unvalidated** term. All six call it the same
wrong way:

```python
p_time = self.cajal.compute_conduction_time_prior(
    centrifugal_order=2,          # hardcoded in 6 of 6 callers
    dist_from_soma_nm=d_nm,       # d_nm is the candidate GAP, not distance from soma
    is_axon=is_axon)
```

`d_nm` is `norm(candidate_coord - cut_coord)` in every one. The "Cajal law of
conduction time" term in every engine is therefore an exponential decay in gap
length — a second distance penalty wearing a biophysical name, duplicating the
`distance_factor` already inside the geodesic score. The centrifugal order,
which is the term's only structural input, is a constant.

### 1.10 None of the twenty-five currently runs

`neuronauts/morpho_grammar/` now contains `__init__.py` and
`cajal_conservation_priors.py` only; the engines moved to `attic/morpho_grammar/`
and no import path was updated. Every one of the 25 fails at import. Separately,
`benchmark_exp032_endpoint_matching.py:87` annotates a signature
`-> List[Dict[str, Any]]` with no `typing` import anywhere in the file, so it
would raise `NameError` at import even with the engines in place — a script that
cannot ever have run in its committed form. And
`benchmark_exp048_grand_unified_engine.py:540-541` writes a viewer to a
hardcoded absolute path under `~/.gemini/antigravity-ide/`.

This matters for the staging question: reviving anything here is porting a
concept, never re-running a script.

---

## 2. Concept families

### Group I — Proposal and gating (local, pairwise)

#### I.1 The tree grammar as a type system (SANTIAGO / probabilistic context-free grammar)
`pcfg_morphology.py`, `blind_pcfg_morphology.py`, `santiago_v2_grammar.py`,
`tree_grammar_infiller.py` · exp026–exp029, exp031–exp035, exp038, exp040

**Concept.** A neuron's arbor is a derivation in a grammar over compartments, so
a continuation is only admissible if it is a legal expansion of whatever
non-terminal the parent currently is.

**Mechanism.** Each fragment is typed from observables — maximum radius, mean
radius, presynaptic/postsynaptic balance, path length — into `<Soma>`,
`<ApicalTree>`, `<BasalTree>`, `<AxonArbor>` or `<Glia>`
(`type_segment_v2`). The parent's symbol yields a list of admissible left-hand
sides (`derive_expected_lhs_from_parent`), and candidates outside it score
`-1e9`. `apply_hard_biological_veto` adds absolute prohibitions: two somas never
fuse, glia never fuses with a neuron, an axon never fuses with a dendrite
without a soma between them.

**To run on real soma-seeded data today.** The typing thresholds (radius 550 nm
for soma, presynaptic ratio 0.60 for axon, and so on) are hand-set numbers that
were never fitted to v117 objects; they need to be fitted and their accuracy
measured on real objects before the admissibility rules mean anything. The rule
probabilities in `pcfg_morphology.fit_from_skeletons` are fitted from a
suspiciously coarse proxy (`fork_count // 3`) that does not parse an arbor into
compartments at all.

**Verdict: REVIVE-WITH-TRAINING** for the type system and the hard veto,
**DEAD** for `EnhancedTreeGrammarInfiller`'s pointer attention. The attention
matrices are random orthogonal `QR` factors that no training ever touched, so
`raw_dot * 3.0` — the largest term in the composite logit — is noise; the
remaining terms are the refuted cone/caliber/distance stack. The type system is
different in kind: it is a constraint on a *structure* (one soma per cell,
polarity coherence within an object) with no free parameters at decision time,
and `results/EXP-063/` shows the object-level version of exactly this signal is
real — its polarity-only feature set reaches held-out AUC 0.914 and its global-
shape set 0.875. That is not a contradiction of the pairwise-polarity
refutation: polarity as a property of one whole object carries signal; polarity
as a match score between two fragments does not.

#### I.2 Directional forward-cone and caliber gating
`tree_grammar_infiller.compute_candidate_score`, `blind_geodesic_em_tracer`,
`blind_precision_engine.score_single_pair`, `best_of_the_best_dual_engine` ·
exp026, exp029–exp032, exp034, exp035, exp043

**Concept.** A neurite continues roughly straight, so a true continuation lies
inside a forward cone about the exit tangent, at a compatible caliber.

**Mechanism.** `cos_align = dot(exit_tangent, ray)` thresholded (values from
0.15 to 0.82 across engines), times `exp(-k * |r_p - r_c| / max(r))`, times an
exponential distance decay.

**Verdict: DEAD.** This is the single most repeated idea in the attic and it is
the one the program has most directly refuted: `results/EXP-061/` (directed cone
against proximity ball) fails, and `results/EXP-081/` puts local features at 0%
precision in the top 34 at a 1.6% base rate. The specific angular veto in
`mcts_handshake_engine` — reject any axon-axon join deflecting more than about
28° — is a hard gate on a quantity now known not to separate.

#### I.3 Geodesic tracing and electron-microscopy voxel verification
`geodesic_em_tracer.py`, `blind_geodesic_em_tracer.py`,
`neuronauts/global_merge/represent/local_em_verifier.py`,
`cloudvolume_em_sampler.py` · exp021, exp026, exp027, exp031–exp037

**Concept.** Instead of a straight line between two cut faces, follow a curved
minimum-cost path through the image volume and ask whether it stays inside one
membrane-bounded tube.

**Mechanism.** A Hermite cubic spline with the two tangents as boundary
conditions; tortuosity as a curvature penalty; then, in the leaking variant, a
"lumen continuity" score drawn from a label-selected distribution (§1.5), and in
the blind variant, no image evidence at all.

**Verdict: DEAD as code.** But mark the distinction carefully: reading actual
voxels along a candidate bridge is an **untested** idea in this repository, not
a refuted one, and the real version already exists elsewhere —
`experiments/fingerprints/` matches cut-face cross-sections from real imagery
and grades REAL in the survey. The Hermite spline itself is 30 lines of
reusable geometry if a curved path is ever needed.

#### I.4 Ultrastructural texture priors
`ultrastructural_texture_prior.py` · exp042

**Concept.** Cytoplasm content differs by compartment — vesicle clusters in axon
terminals, mitochondrial cristae in dendritic shafts, microtubule fasciculation
in shafts — so texture identifies type and should match across a break.

**Mechanism.** It does not measure texture. `extract_texture_signature` switches
on `fragment_token["inferred_type"]` and returns hardcoded constants
(`vesicle_density = 0.85` if the type is already "Axon", `0.05` if "Dendrite"),
then takes a cosine between two such lookups. The resulting "texture
compatibility" is an obfuscated restatement of "the two fragments have the same
inferred type," and the `n_syn <= 1` gate that supposedly makes it useful for
sparse-synapse stubs multiplies that restatement by 1.4.

**Verdict: DEAD.** The idea is sound and is the fingerprints thread's territory;
this implementation contributes nothing but a laundered label.

#### I.5 Synaptic fingerprinting — partner-set overlap
`mcts_handshake_engine.compute_synaptic_jaccard`,
`tree_grammar_infiller`'s `jaccard_syn` term · exp026, exp033, exp037, exp047

**Concept.** Two fragments of the same axon should contact overlapping sets of
postsynaptic partner cells, so partner-identity overlap identifies siblings
independently of geometry.

**Mechanism.** Jaccard index over `syn_partners`, rescaled
`clip(0.50 + 2.5*(J - 0.10), 0.10, 0.95)`, with a neutral 0.50 returned whenever
either side has no synapses.

**The measurement that produced it was circular.** §1.2: in all 25 scripts
the partner identifiers are `cell_index * 100 + rng.integers(0, block)`,
so same-cell fragments share a private fifteen-element block and cross-cell
Jaccard is identically zero. The feature was a one-hot encoding of the answer,
weighted `+3.0` in the infiller's logit and 1.2–2.0 in the handshake engine.
Whatever this concept looked like in the attic, that was not a measurement of
it.

**Verdict: DEAD at the frontier, REVIVE-WITH-TRAINING at cell scale.** At a
frontier the neutral branch fires almost always, and `results/EXP-071/` explains
why structurally: the connective cable between two fragments of one cell is
missing from the synapse-anchored population *because* it carries no local
synapses. Partner overlap cannot speak where the decision is. As a whole-cell
coherence term — does this assembled axon's target set look like one axon's? —
it is a structure-level score, it uses real partner identifiers from the synapse
table rather than fabricated ones, and it has never been measured.

---

### Group II — Structure-level scoring (the shape that works)

#### II.1 Cajal and Murray conservation
`cajal_conservation_priors.py` · exp027, exp034, exp035, and as `p_cajal` in six engines

**Concept.** Branching obeys conservation: cross-sectional area is shared among
daughters (`r0^3 = r1^3 + r2^3`), and the optimal bifurcation angle follows from
the same radii. Neither has a free parameter, and neither can be evaluated
without a tree.

**Verdict: REVIVE** — and it is already the program's one working template
(`results/EXP-084/`: median exponent 3.18 against an ideal 3.0 across 3,781 real
bifurcations, separating real from mismatched branches at AUC 0.675). §1.9 above
is the audit finding attached to it: the engines used only the unvalidated
conduction-time term, called it with the gap distance in the
distance-from-soma slot, and never called the validated angle prior at all. The
file is 90 lines, carries no label leakage, and the two useful functions in it
are `compute_bifurcation_angle_prior` and the caliber law.

#### II.2 Whole-cell shape regularization
`global_morpho_regularizer.py` · exp041

**Concept.** A cortical cell has global shape constraints a pairwise score
cannot see: an apical trunk ascends toward the pia; basal dendrites diverge
radially rather than ascending; caliber decreases monotonically outward from the
soma; a branch does not double back on itself.

**Mechanism.** Four additive terms — pial alignment (only for thick dendritic
parents), a soft penalty when a child's radius exceeds 1.45× its parent's, a
retrograde penalty when `cos(tangent, displacement) < -0.5`, and a
synapse-polarity reward.

**To run on real soma-seeded data today.** The pial axis is hardcoded to
`(0, -1, 0)`, which is a claim about MICrONS coordinates that should be measured
rather than assumed; the four weights are hand-set and were never fitted; and
critically the regularizer is applied **per candidate pair**, not per tree,
which throws away the only property that makes it interesting.

**Verdict: REVIVE-WITH-TRAINING.** Monotone taper along a root-to-tip path and
absence of retrograde segments are properties of an assembled arbor, they
compound over the arbor exactly as Murray's law does, and they belong with the
whole-cell shape work rather than with a pairwise scorer.

#### II.3 Global multi-hypothesis tree search and whole-cell energy
`global_hypothesis_search.py`, `hierarchical_dual_santiago.py` · exp044, exp046

**Concept.** Do not commit at an ambiguous branch point. Split decisions into
confident ones and ambiguous ones by a calibrated margin, enumerate K competing
whole-cell trees over the ambiguous set, score each tree on whole-cell physical
invariants, and take the best tree rather than the best local step.

**Mechanism.** `compute_decision_confidence` turns a log-odds score into a
posterior and a top-1-minus-top-2 margin; `score_global_tree_hypothesis` sums
conduction cost, Murray caliber-step penalty and synaptic consistency over all
links in a candidate tree; `assemble_global_optimal_tree` builds the
hypotheses and picks the maximum.

**To run on real soma-seeded data today.** The hypothesis enumeration is the
weak part: K = 4 trees built by taking the k-th ranked choice at *every*
ambiguous branch simultaneously, which samples four points from an exponential
space along a diagonal and is not a search. The energy needs the validated
Murray term rather than the ratio-threshold proxy, and the synaptic term is the
one refuted at the frontier.

**Verdict: REVIVE-WITH-TRAINING.** This is the correct shape — a score over a
tree whose evidence compounds — and it is the closest thing in the attic to the
user's stated direction. It needs a real search and a validated energy.

---

### Group III — How the decision is made (joint versus greedy)

#### III.1 Joint bipartite assignment with multi-round soma-seeded growth
`hungarian_bipartite_assembler.py` · exp047, exp048, exp049, exp050

**Concept.** Frontier cut ends should not each greedily grab their best
candidate. In each round, all live cut ends across all growing cells compete for
a shared pool of objects, and the assignment is solved **globally and one-to-one**,
with an explicit option for every cut to take nothing.

**Mechanism.** `scipy.optimize.linear_sum_assignment` over an
`N_cuts × (M_candidates + N_cuts)` cost matrix: the left block holds negated
scores, and the right block is a diagonal of slack columns priced at the
acceptance threshold, so a cut whose best real candidate scores below threshold
is matched to its own slack column and abstains. Union-find over claimed objects
prevents a cell from re-absorbing itself. Somas seed the first frontier; each
round's newly claimed objects become the next frontier; growth stops when a
round adds no links.

**To run on real soma-seeded data today.** Swap the scorer (its
`evaluate_bidirectional_handshake` is the refuted local stack), swap the piece
representation for object clouds from `scripts/build_object_clouds_voxel.py`,
and seed from real somas. The assignment layer, the slack mechanism, the
union-find and the round structure are all scorer-agnostic and need no change.

**Verdict: REVIVE.** This is the most directly reusable code in the attic and
the only place in the entire era where a decision is made jointly rather than
greedily. It is also, structurally, exactly the grower `results/EXP-081/`
describes: a frontier of cut ends, most of which must abstain.

#### III.2 Bidirectional handshake and reciprocal consistency
`mcts_handshake_engine.evaluate_bidirectional_handshake`,
`blind_precision_engine`'s Gate 3 · exp030, exp037, exp043

**Concept.** A join should be accepted only if each side independently prefers
the other — `P_handshake(A,B) = sqrt(P(B|A, t_A) · P(A|B, -t_B))`, or explicitly,
the backward search from the candidate must return to this cut.

**Verdict: DEAD standalone, subsumed by III.1.** Mutual-nearest is a genuinely
joint constraint and its intent is right, but it is a symmetrization of a base
score, and symmetrizing a score that is at 0% precision does not rescue it. The
Hungarian assignment enforces the same exclusivity property more strongly and
over the whole frontier at once. The `sqrt` geometric-mean form is worth
remembering as the right way to combine two directional scores if one is ever
built.

#### III.3 Calibration, margin gating, and the terminal (null) hypothesis
`blind_precision_engine.compute_terminal_score`,
`global_hypothesis_search.compute_decision_confidence` · exp030, exp031, exp043, exp044

**Concept.** "This branch simply ends here" is a competing hypothesis with its
own prior, not the absence of a hypothesis. A join is accepted only when it
beats termination *and* beats the runner-up by a margin.

**Mechanism.** `p_term = sigmoid((r - r0)/s) * (1 - exp(-d/D))` — thin branches
far from the soma are likely terminals — with separate constants for axons
(r0 = 70 nm, D = 150 µm) and dendrites (r0 = 120 nm, D = 100 µm). Acceptance
requires `score(top1) - score(top2) >= tau` and `score(top1) > log-odds(p_term)`.

**To run on real soma-seeded data today.** Every constant here is a guess and
none was ever fitted; the same `d_nm`-for-distance-from-soma confusion of §1.9
applies wherever the caller supplies a gap. What it needs is labels, and those
now exist: `results/EXP-081/` has 2,137 real frontier tips with live/dead labels
at a 1.6% base rate.

**Verdict: REVIVE-WITH-TRAINING**, and it is the concept the current program is
most conspicuously missing. `results/EXP-075/` and `EXP-076` refuted the seed's
own *local end shape* as a stop signal (AUC 0.476, matched for distality); they
did not test a calibrated prior over caliber and path distance from the soma
against the real base rate, which is a different claim. No experiment in the
repository currently has a fitted stop rule, and at 1.6% the stop rule is most
of the problem.

#### III.4 Actor-critic iterative refinement
`agentic_actor_critic.py` · exp036

**Concept.** Propose, verify against invariants, and on rejection add the
candidate to an exclusion set and re-propose — a rejection loop with memory.

**Mechanism.** `MorphoActor` is the grammar proposer; `MorphoCriticJudge` runs
five checks (backward ray, excessive distance, caliber disparity, geodesic
membrane breach, synapse polarity) and a log-odds value function; up to three
turns, falling back to the best rejected hypothesis at a soft floor of −2.0.

**Verdict: DEAD.** The critic's five invariants are the refuted local stack plus
the simulated geodesic, and the loop's only effect is to walk down the same
ranked list. The fallback that accepts the best *rejected* candidate at a soft
floor undoes the point of having a critic. The one durable idea inside it — an
explicit reject-and-abstain decision — is better carried by III.3 and by the
slack columns in III.1.

---

### Group IV — Two-sided proofreading and human history

#### IV.1 Split before merge: frankenmerge cleaving as a first pass
`frankenmerge_resolver.py`, `grand_unified_engine.py` · exp045, exp048

**Concept.** Proofreading is two-sided. Cleave the false merges already present
in the segmentation *before* healing false splits, because growing outward from
a segment that is already two cells propagates the error. False merges are
detectable from topological invariants inside one object: two somas, glia fused
to a neuron, an axon fused to a dendrite with no soma between them.

**Mechanism.** `detect_and_cleave_frankenmerges` inspects each multi-fragment
segment for those three violations and, on a hit, cleaves it into singletons.

**Verdict: DEAD as code, and the concept has already been revived properly.**
The attic implementation reads `f.get("gt_type")` at
`frankenmerge_resolver.py:54-56` — it detects violations using the ground-truth
compartment label, so it measures nothing. But `results/EXP-063/` did the real
version and reached held-out AUC 0.958 on real objects with no leakage. What is
worth keeping from the attic is the **ordering** — the two-pass structure in
`grand_unified_engine.execute_grand_unified_proofreading`, split then merge —
which is a pipeline decision the current program has not made explicit, and the
observation that the invariants are *whole-object* properties.

#### IV.2 The oracle: budgeted queries on the decisions the method cannot make
`active_gap_oracle.py` · exp046

**Concept.** Reserve a small budget of expensive, authoritative answers — at
most five per cell — for the long-range tears (>20 µm) that no automatic score
resolves, and spend them where the leverage is highest.

**Mechanism as written.** `query_long_range_gap` projects rays along the exit
tangent, keeps candidates 20–65 µm out that pass a polarity check and a cone
test, takes the top three by ray score, and then: `if gt_target_id in
top_cand_ids: return gt_target_id` with a 98% coin flip. The "oracle" is the
answer key. Any measurement through it reports the recall of the ray gate and
nothing else, and `hierarchical_dual_santiago.py:155` gates the entire stage on
`gt_map is not None`, so exp046's oracle numbers are unavailable without ground
truth by construction.

**Verdict: REVIVE — with a real oracle, which now exists.**
`neuronauts/edit_history.py::fetch_edit_log` was broken for an unknown period
(it called a caveclient method that no longer exists, inside a bare `except`)
and is now fixed: **1,039 real logged operations for one gold-proofread cell**,
each with an operation id, a millisecond timestamp, before/after root ids, an
`is_merge` flag and the proofreader's name
(`docs/threads/edit_history_ground_truth.md`). Every join a human ever made in
this volume is recorded and free. That converts the simulated oracle into a real
one and, more importantly, converts "which objects belong to this cell" from a
geometric proxy into an observed fact.

#### IV.3 Selective micro-electron-microscopy: spend on the contested minority
`local_em_verifier.py`'s stated contract, `dual_engine_infiller` · exp021, exp026

**Concept.** A cheap proposer ranks; an expensive verifier is queried **only** on
the ambiguous band (the docstring says 0.30 ≤ P ≤ 0.70) and only on the top K,
so the expensive evidence is affordable.

**Verdict: REVIVE-WITH-TRAINING.** The implementation fabricates the expensive
evidence (§1.5), but the cascade is sound and `results/EXP-081/`'s arithmetic
makes it attractive rather than merely elegant: 2,137 frontier decisions of
which 34 are live means a real image query at every contested site is
affordable. It needs a verifier that reads pixels — `experiments/fingerprints/`
is that verifier, and it is graded REAL.

---

### Group V — Representation and plumbing

#### V.1 Contrastive skeleton embedding (tree-DNA)
`neuronauts/global_merge/represent/vicreg_gnn.py` · exp021 only

**Concept.** Learn a translation-invariant morphological signature per fragment
by contrastive training, so fragments of one cell embed near each other.

exp021 is **the only script in the entire scope that trains anything**
(`VICRegSkeletonModel(in_dim=4, emb_dim=64, proj_dim=128)`, 40 epochs, at
`benchmark_exp021_3d.py:105-106`). It then contaminates its own evaluation by
passing `is_same_cell` into the EM verifier at line 201.

**Verdict: REVIVE-WITH-TRAINING, at cell or half-cell scale only.** This is the
roadmap's central idea (`docs/roadmap_global_assembly.md` §2) and the survey
grades the half-skeleton result REAL — trained AUC 0.829 against a random-init
0.768, so a genuine but modest +0.061 lift — while recording that the same
method *fails* at quarter-skeleton granularity. That scale dependence is already
established and should not be re-litigated at fragment scale.

#### V.2 Endpoint extraction and closest-surface contact geometry
`hungarian_bipartite_assembler.extract_piece_endpoints`,
`get_closest_vertex_and_tangent`,
`best_of_the_best_dual_engine.find_closest_candidate_vertex` · exp032, exp034, exp047

**Concept.** A join is between specific *endpoints*, not object centroids: the
contact point on a candidate is its closest vertex to the cut, and the tangent
must be measured there, not from the object's overall displacement.

**Verdict: REVIVE as utilities.** `results/EXP-070/` already confirms the
endpoint-versus-object distinction matters on real data. These are three small,
correct, leak-free functions (degree computation, leaf enumeration, local
tangent from the incident edge) and they save re-deriving the same geometry.
Note that the era's own token construction contradicts them:
`pcfg_morphology.serialize_to_grammar_tokens` sets a fragment's tangent from
`vertices[-1] - vertices[0]`, the whole-fragment displacement, which for a
branched fragment is meaningless.

---

### Group VI — Measurement

#### VI.1 Structure-level metrics and legality counters
`synapse_segment_typer.compute_full_pairwise_confusion_matrix`,
`evaluate_grammar_violations_under_mistyping`, `neuronauts/line_graph.py`,
expected run length · exp033, exp035, exp045, exp048

**Concept.** Score an assembly on what a connectome is *for* — recovered
synaptic edges (line-graph circuit precision/recall/F1) and expected run length
— and on biological legality (multi-soma rate, axon-dendrite chimera rate, glia
intrusion rate), not only on pairwise partition agreement.

**Verdict: REVIVE as evaluation.** These read ground truth, which is exactly
right for a metric and exactly wrong for a scorer; the attic's sin was using
`gt_type` inside scorers (§IV.1), not inside these. Expected run length and
line-graph circuit F1 are structure-level and the current program does not
compute them. `results/EXP-059/` (metric agreement, pass) is the place they
should be cross-checked before use.

#### VI.2 Failure-mode forensics
`santiago_v2_grammar.ForensicErrorAnalyzer` · exp038, exp048

**Concept.** Every false positive and false negative is assigned a named
physical cause — `AXON_DENDRITE_SYNTAX_CHIMERA`,
`LONG_GAP_DISTANCE_EXCEEDING_RECEPTIVE_FIELD`,
`ACUTE_BRANCH_ANGLE_OCCLUSION`, `SPARSE_SYNAPSE_SIGNAL_AMBIGUITY` — so an error
rate becomes a distribution over diagnosable modes.

**Verdict: REVIVE.** Cheap, honest, and the right instinct: an aggregate score
tells you nothing about what to fix next. The taxonomy itself is tuned to the
refuted geometry and will need rewriting, and the `if/elif` cascade assigns each
error to exactly one cause by priority order rather than recording all
applicable ones, which will bias the distribution. But 80 lines of "explain
every error" is worth carrying forward into the current experiments.

#### VI.3 Protocol: spatial disjointness, open-world pools, stratification
`benchmark_exp049.check_spatial_disjointness`, exp047's open pool, exp050's strata

**Concept.** Three protocol commitments, each of which is right:
(i) train and test in spatially disjoint bounding boxes with an enforced buffer,
so nothing leaks through proximity; (ii) score against **all** objects in the
test box, not a curated pool; (iii) report metrics stratified by cell type so a
pyramidal-only result is not reported as general.

**Verdict: REVIVE the protocol, DEAD the scripts.**
`check_spatial_disjointness` is 15 correct lines. The open-world pool is already
what the era did (§1.1) and is to its credit. Stratification is the right answer
to a real risk. Everything around them in exp049 and exp050 is fabricated
(§1.8) — and under §1.3, so is much of the rest of the era.

#### VI.4 External baselines and the comparison table
`autoproof_baseline.py`, `neurd_baseline.py`, the hardcoded rows · exp040, exp044, exp047, exp049

**Verdict: DEAD**, and §1.7 above should be treated as a retraction note. What
survives is the *idea* of a baseline ladder, which the current program already
implements honestly as `results/EXP-058/` (oracle 1.0, best proximity 0.0,
random ~0).

---

## 3. Index — all 25 benchmarks

| Benchmark | Primary concept family | Verdict |
|---|---|---|
| exp021 3D + selective micro-EM | V.1 tree-DNA, IV.3 cascade | REVIVE-WITH-TRAINING (only trained script; leaks `is_same_cell` at line 201) |
| exp026 enhanced dual engine | I.1, I.2, I.3 (leaking) | DEAD |
| exp027 Cajal-geodesic | II.1, I.3 (leaking) | DEAD harness; II.1 REVIVE |
| exp028 blind rigorous | I.1 leak-free audit | DEAD scores; the leak audit itself is a good habit |
| exp029 blind directional | I.2 | DEAD |
| exp030 SOTA precision | III.2, III.3 | III.3 REVIVE-WITH-TRAINING |
| exp031 calibrated precision | III.3, I.2 | III.3 REVIVE-WITH-TRAINING |
| exp032 endpoint matching | V.2 | REVIVE (utilities); the script itself cannot import (§1.10) |
| exp033 end-to-end synapse typing | I.1 typing, VI.1 | VI.1 REVIVE; harness writes `gt_type` into the scoring pool (§1.6) |
| exp034 best-of-the-best | V.2, II.1, I.5 | mixed; V.2 REVIVE |
| exp035 restored dual engine | I.1, VI.1 | DEAD (the source of the "~85–87%" figure; see §1.1) |
| exp036 agentic actor-critic | III.4 | DEAD |
| exp037 MCTS handshake | III.2, I.5 | DEAD |
| exp038 SANTIAGO-v2 forensics | I.1, VI.2 | VI.2 REVIVE |
| exp040 SOTA + external volume | VI.4, and a fabricated external volume | DEAD (scorecard printed as literals) |
| exp041 morphology regularizer | II.2 | REVIVE-WITH-TRAINING |
| exp042 multimodal texture | I.4 | DEAD |
| exp043 calibrated ensemble | III.3 | REVIVE-WITH-TRAINING |
| exp044 global multi-hypothesis | II.3, VI.4 | II.3 REVIVE-WITH-TRAINING |
| exp045 full-spectrum evaluation | VI.1, IV.1 | VI.1 REVIVE |
| exp046 hierarchical dual-scale | II.3, IV.2 | II.3 and IV.2 REVIVE-WITH-TRAINING; harness passes `gt_map` into the link-producing path (§1.6) |
| exp047 Hungarian bipartite | III.1 | **REVIVE**; note its corpus is fabricated unconditionally (§1.3) and its pool is the only genuinely population-wide one |
| exp048 grand unified | IV.1 ordering, III.1 | ordering REVIVE; scoring DEAD |
| exp049 dense subvolume | VI.3, III.1 | protocol REVIVE; script DEAD (fabricated, §1.8) |
| exp050 interneuron stratified | VI.3 | protocol REVIVE; script DEAD (fabricated, §1.8) |

---

## 4. The five worth reviving, ranked

Ranked by the stated direction: score whole structures, exploit human
proofreading history, decide jointly rather than greedily.

### 1. Joint frontier assignment with an explicit abstain — the grower's decision layer

**What it is.** Every live cut end across every growing cell competes in one
round for a shared pool of objects; the round is solved as a one-to-one
assignment in which "take nothing" is a real, priced option.

**Why first.** It is the only joint decision mechanism in the attic, it is
already soma-seeded and iterative, its cost matrix is completely decoupled from
whatever scorer you plug in, and its slack columns are the natural home for the
stop rule that item 2 supplies. `results/EXP-081/` describes precisely this
object — a frontier where 2,103 of 2,137 tips must abstain — and no current
experiment implements one.

**Code it needs.** `attic/morpho_grammar/hungarian_bipartite_assembler.py`:
keep `assemble_volume_bipartite` (the round loop, the union-find, the
`N_cuts × (M + N_cuts)` cost matrix with slack priced at the acceptance
threshold), `extract_piece_endpoints` and `get_closest_vertex_and_tangent`.
Delete the `TreeBeamMCTSAssembler` dependency entirely and take a scorer
callable instead. Roughly 180 lines survive of 287.

**Data it needs.** Object clouds built the correct way —
`scripts/build_object_clouds_voxel.py` (CloudVolume, `agglomerate=True`,
`timestamp=V117_TS = 1623399000`, mip 2), never `object_clouds_mip5.npz`. Soma
seeds from `scripts/seed_census.py`. The frontier tips and live/dead labels from
`results/EXP-081/` as the evaluation set.

---

### 2. A calibrated stop rule fitted on the real frontier

**What it is.** Model "this branch ends here" as a hypothesis with its own fitted
prior over caliber and path distance from the soma, and require a join to beat
both termination and its runner-up by a margin.

**Why second.** At a 1.6% base rate the stop decision *is* the problem, and the
repository has no fitted stop rule. This is also the piece that makes item 1
mean anything: an assignment with a badly priced abstain column is just greedy
matching with extra steps. It must be measured against the honest negative —
EXP-075/076 refuted the seed's local *end shape* (AUC 0.476, matched for
distality), which is a different quantity from a prior over caliber and
distance-from-soma, and the distinction should be stated in the writeup so the
result is not mistaken for a re-run of a refuted test.

**Code it needs.** `blind_precision_engine.compute_terminal_score` as a
*functional form to fit*, not to use — its six constants (70 nm / 15 / 150 µm
for axons, 120 nm / 25 / 100 µm for dendrites) are guesses. Plus
`global_hypothesis_search.compute_decision_confidence` for the margin. Fit with
anything standard; the model is two-parameter per compartment class.

**Data it needs.** `results/EXP-081/`'s 2,137 tips with live/dead labels across
40 cells, split spatially rather than randomly (use
`benchmark_exp049.check_spatial_disjointness`, 15 lines, and a buffer). Caliber
at each tip and geodesic path distance from the soma — the latter is not in the
EXP-081 artifact and has to be computed on the claimed cable.

---

### 3. A whole-cell conservation energy that scores a tree, not a pair

**What it is.** Take an assembled candidate arbor and score it on
parameter-free conservation laws at every branch point it contains — Murray's
caliber law, the Cajal bifurcation-angle prior, monotone taper along
root-to-tip paths, absence of retrograde segments — then use that energy to
choose among competing whole-cell hypotheses.

**Why third.** It is the direct generalization of the one thing that works
(`results/EXP-084/`, AUC 0.675 from a *single* branch point) into the regime
where its weakness stops mattering: a cell has dozens of bifurcations, a wrong
join creates one bad branch among many good ones, and the evidence accumulates.
It also fixes the audit finding in §1.9 — the engines used the unvalidated Cajal
term with the wrong argument and never called the validated one.

**Code it needs.** `attic/morpho_grammar/cajal_conservation_priors.py`
(90 lines, no leakage; use `compute_bifurcation_angle_prior` and the caliber law,
discard `compute_conduction_time_prior` until it is validated),
`global_morpho_regularizer.py`'s taper-monotonicity and retrograde terms lifted
out of their per-pair wrapper, and
`global_hypothesis_search.score_global_tree_hypothesis` restructured to sum over
branch points rather than links. The hypothesis *enumeration* in
`assemble_global_optimal_tree` should be discarded — K=4 samples along a diagonal
of an exponential space is not a search; the K candidate trees should come from
item 1's assignment at varying acceptance thresholds instead.

**Data it needs.** `results/EXP-084/`'s 3,781 real bifurcations as the
calibration set for what a legal branch looks like, and real assembled candidate
arbors from item 1 as the thing to be scored. Radii at mip 2 from the same
CloudVolume route as item 1.

---

### 4. The proofreading change log as oracle and as ground truth

**What it is.** Replace the simulated 98%-accurate oracle with the real,
free, recorded history: every merge and split a human performed in this volume,
with timestamps, before/after root ids and the proofreader's name.

**Why fourth.** "Exploit human proofreading history"
has no representative anywhere in the attic — `active_gap_oracle.py` is the only
attempt and it queries the answer key. Meanwhile the real thing became available
when `fetch_edit_log` was fixed: **1,039 operations for a single gold cell**,
378 merges and 661 splits. This is the observed join topology the program has
been approximating with geometric proxies, and it also supplies a budgeted-query
protocol that is honest: rank the frontier decisions by leverage, "ask" the
change log for the top few per cell, and measure how much of a cell you recover
per query.

**Code it needs.** `neuronauts/edit_history.py::fetch_edit_log` (fixed;
`get_tabular_change_log`, note the second underscore) and `roots_at` /
`root_at_version` for resolving historical roots to v117, as used in
`neuronauts/experiments/exp071_connective_gap.py`.
`attic/morpho_grammar/active_gap_oracle.py`'s ray-projection and query-budget
structure is reusable as the *policy* for which decisions to spend a query on;
delete its `gt_target_id` branch entirely.

**Data it needs.** CAVE change logs for the 40 seeded cells of
`results/EXP-081/` — one network call per cell. The open question recorded in
`docs/threads/edit_history_ground_truth.md` should be closed first: of 460 leaf
roots in one cell's merge history, only 64 matched a known atom, and the
remaining 396 were not distinguished between "genuinely inside our box with no
local synapses" and "outside the box." That distinction is a prerequisite, not
an afterthought.

---

### 5. The grammar as a whole-object legality constraint

**What it is.** A small set of parameter-free structural prohibitions applied to
an assembled object rather than to a candidate pair: one soma per cell, no
glia fused to a neuron, no axon fused to a dendrite without a soma between them,
and polarity coherence within an object.

**Why fifth.** It scores a structure, it has no free parameters at decision
time, and unlike everything else in the grammar family it has direct real-data
support: `results/EXP-063/` detects frankenmerges on real objects at held-out
AUC 0.958, with the **polarity-only** feature set alone at 0.914 and the
global-shape set at 0.875. That is the object-level version of exactly these
invariants. It also gives item 1 a veto that costs nothing to evaluate, and it
supplies the split-before-merge ordering that `grand_unified_engine` had right.

**The distinction that must be preserved.** Polarity as a property of one whole
object carries real signal; polarity as a *pairwise match score* between two
fragments is at chance and degrades the combination. This item is strictly the
former. Do not let it drift back into a pairwise term.

**Code it needs.** `santiago_v2_grammar.apply_hard_biological_veto` and
`type_segment_v2` (about 90 lines, no leakage), and
`frankenmerge_resolver.detect_and_cleave_frankenmerges` **rewritten** — its
current form reads `f.get("gt_type")` at lines 54–56 and must be replaced with
the EXP-063 detector. `synapse_segment_typer.evaluate_grammar_violations_under_mistyping`
becomes the evaluation side (it reads ground truth, which is correct for a
metric).

**Data it needs.** `type_segment_v2`'s thresholds fitted and its accuracy
measured on real v117 objects before any veto is trusted — the current numbers
(550 nm soma radius, 0.60 presynaptic ratio, 4 µm zero-synapse glia floor) are
hand-set, and §1.2 means they were never tested against anything: in the
benchmarks a fragment's presynaptic/postsynaptic balance was a deterministic
function of its index in the split, so the typer's reported accuracy measured
the split, not the tissue. `results/EXP-063/`'s trained detector and its atom
feature tables (`neuronauts/harness/atom_features.py`) supply the rest.

**Runner-up, for the record:** selective expensive verification (§IV.3) paired
with the real cut-face model in `experiments/fingerprints/`. The arithmetic is
compelling — 34 live sites among 2,137 means you can afford to read pixels at
every contested one — but it depends on items 1 and 2 existing first to identify
which sites are contested.

---

## 5. What can leave

Everything in `attic/morpho_grammar/` except these five files can go, with no
loss of concept:

| keep | why |
|---|---|
| `cajal_conservation_priors.py` | validated by EXP-084; no leakage; shortlist item 3 |
| `hungarian_bipartite_assembler.py` | the joint assignment layer; shortlist item 1 |
| `santiago_v2_grammar.py` | the veto, the typer and the forensic analyzer; shortlist item 5 |
| `blind_precision_engine.py` | for `compute_terminal_score` alone, as a form to fit; shortlist item 2 |
| `global_morpho_regularizer.py` | for the taper and retrograde terms; feeds item 3 |

Two files should carry an explicit warning rather than being quietly deleted,
because their names promise a measurement they do not make and both still live
in `neuronauts/`, not the attic:

- `neuronauts/global_merge/represent/cloudvolume_em_sampler.py` — branches on
  `is_true_continuation` while claiming to sample voxel intensities; imports no
  CloudVolume.
- `neuronauts/global_merge/represent/local_em_verifier.py` — its only method
  forwards the ground-truth label to the above.

And `docs/threads/experiment_survey.md` should gain the findings in §1.1,
§1.2, §1.3, §1.4, §1.7 and §1.9, none of which it currently carries: the lenient
credit rule that produced the "~85–87%" figure; the synaptic partner identifiers
that encode the ground-truth cell; the silent fallback from real skeletons to a
generator; the "inductive split" that fits nothing; the hardcoded comparison rows
and legality rates; and the fact that the one Cajal term with a real-data
validation was never called by any engine.

The survey's own verdict for this era — SEMI-SYNTHETIC, not evidence — does not
change. What changes is the reason: it is not only that the damage was
synthetic, but that the observables were too, and that in the case of the
synaptic partner identifiers they were synthesized *from the label*.

---

## 6. Staging — what each of the five becomes, in order

The registry already reserves **EXP-069, "Attic re-derivation"** as the only
route out of `attic/morpho_grammar/`, with the criterion "beats the stacked
EXP-064 scorer on the same panel with a trained grammar; else
`EXPERIMENT_LOG.md` stays superseded"
(`neuronauts/experiments/registry.py`). That criterion is written for a
*scorer*, and §1 above is the reason it should be retired rather than attempted:
there is no attic scorer whose numbers can be earned back, because none of them
were measured on observables that exist. The five items below replace it with
five separable questions, none of which is a re-run.

They are ordered by dependency, not by appeal. Items A and B are the pair that
must land together — an assignment without a priced abstain column is greedy
matching, and a stop rule with nothing to gate is a curve with no decision.

| stage | question | depends on | bar it should be held to |
|---|---|---|---|
| **A** — joint frontier assignment | Does solving each round of a soma-seeded grower as one global assignment, with abstention priced, beat per-cut greedy selection on the same scorer? | EXP-081's frontier; the mip-2 object clouds | recovery at equal purity, against the same scorer run greedily. If the two are equal, the assignment layer is not earning its complexity and should be said so. |
| **B** — calibrated stop rule | Can a prior over caliber and path distance from the soma separate the 34 live tips from the 2,103 dead ones? | EXP-081's labels; spatially disjoint split | held-out area under the curve against a distality-matched negative, stated beside EXP-075/076's 0.476 so the two are not confused |
| **C** — whole-cell conservation energy | Does conservation evidence summed over an assembled arbor's branch points separate a correct assembly from one with a wrong join? | A (to produce arbors); EXP-084's 3,781 bifurcations | separation at the *arbor* level, reported next to EXP-084's single-branch 0.675 — the whole claim is that it compounds, so a result that does not beat 0.675 refutes the premise |
| **D** — the change log as ground truth | Of the objects a human actually merged into a cell, how many can any automatic method reach, and how many are outside the synapse-anchored population entirely? | `fetch_edit_log`; the box/no-synapse question in `edit_history_ground_truth.md` closed first | a reachable-set size reported beside every recall number, per the standing convention |
| **E** — whole-object legality veto | Do the structural prohibitions reject wrong assemblies at a rate worth the recall they cost? | A; EXP-063's detector; a fitted `type_segment_v2` | precision gain against recall lost, with the typer's own accuracy on real v117 objects reported first — an unmeasured typer makes the veto unfalsifiable |

Three things that should be true of all five and were true of none of the
twenty-five:

1. **The population is named and counted before the run**, with the count of
   cells that came from a real loader recorded separately from any fallback. A
   run that cannot say where its cells came from is not a measurement (§1.3).
2. **Nothing derived from a label enters the scoring call.** Not `gt_type`, not
   `gt_map`, and not an observable synthesized from the cell index (§1.2, §1.6).
3. **Every printed number is computed in the same run.** No comparison row, no
   safety rate, no prior-round baseline is a literal (§1.7).
