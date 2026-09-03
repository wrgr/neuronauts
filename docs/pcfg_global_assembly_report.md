# PCFGs for global assembly and validation: a deep dive

> **Status: design report (2026-09-01), no code changes.** Written against the
> `feat/global-merge-assembly` branch, the untracked grammar harness
> (`docs/grammar_harness_handoff.md`), the tree-assembly design
> (`docs/tree_assembly_algorithm.md`), and the grammar experiment record
> (`experiments/pcfg/FINDINGS_synapse_correction.md`). Every number below is
> quoted from a file in this repo; nothing here has been run.
>
> Companion: `docs/grammar_literature_directions.md` (wider grammar, structured
> inference, and transformer literature; ranked directions). The same-day
> `docs/consolidation_plan.md` §6.4 maps E0–E5 below onto its EXP-057–069
> series and adopts these bars (E1 → 063, E2 and E3 → 062, E4 → 060,
> E5 → 066, E0 as the prerequisite of series D). Its Phase 2 has moved
> `neuronauts/morpho_grammar/` to `attic/morpho_grammar/` and the EXP-021–050
> benchmarks, plus six unnumbered siblings, to `attic/benchmarks_semi_synthetic/`;
> paths below use the attic locations. A shim keeps `neuronauts.morpho_grammar`
> importable with a `DeprecationWarning`.

## TL;DR

**Yes, a PCFG earns a place here, but as a verifier and seam locator over the
observed skeleton tree, not as a proposer or an identity oracle.** The clearest
wins, in order of expected value per day of work:

1. **Seam location inside a false merge.** The repo has already shown that the
   binding constraint on merge correction is not detecting a bad object (whole-
   object shape gets AUC 0.875) but finding *which single skeleton edge* to cut
   (oracle +79% net pair-error reduction; the learned seam GNN reaches only 25%
   top-1 and was net-negative until 513 training objects). A grammar scores
   exactly that quantity, `log P(A) + log P(B) - log P(A∪B)` for every edge, in one
   linear pass, with a few hundred parameters learned from *correct* neurons
   rather than from the rare error objects. This directly attacks the
   data-starvation wall.
2. **Validation of assembled hypotheses.** The tree-assembly design already
   wants a "topology/biology" step in its verification battery and a set of
   cannot-link priors (one soma, polarity, caliber). A PCFG is the principled
   form of that checklist: hard-zero productions for the impossible, learned
   probabilities for the implausible, and a parse (compartment labels on every
   segment) that a proofreader can read.
3. **Endpoint typing for candidate generation.** The harness has 245 degree-1
   endpoints per atom, and the open question is which of the 5.1M are real
   split sites. A grammar gives each tip a posterior over {biological terminal,
   cut, bounds-clipped} from caliber, leaf length, and compartment context,
   label-free.
4. **A decomposable global term in constrained Kruskal assembly.** Because the
   grammar is context-free, the change in whole-neuron log-likelihood from
   attaching fragment B at an endpoint of A factors into cached inside/outside
   quantities at the junction. Global validation at local cost, which is what
   makes it usable over 20,826 atoms and 5.1M endpoints.

**Where it will not help, and the repo's evidence says so:** fine-scale merge
discrimination between two facing tips (continuity geometry gets AUC 0.988 and
the bigram grammar adds nothing), axon identity across millimetres
(grammaticality is not identity; proximity was measured at 0/32), and anything
trained on synthetic splices (the BiGRU mastered the proxy at 0.82 and
transferred at chance).

**None of the four things called "grammar" in this repo is the object proposed
here.** Two are neural sequence encoders, one is a bigram Markov model over
PCA-ordered synapses, and the one true PCFG (`attic/morpho_grammar/`, formerly
`neuronauts/morpho_grammar/`) has
hand-set rule probabilities, no parser, an infiller built on untrained random
projections, and a benchmark that is both synthetic and non-importable. So the
existing negative results do not falsify the proposal, but two of them
constrain it hard: score *typed branching structure* on real L2 adjacency, not
edge geometry on synapse sequences; and learn from real gold cells, evaluate on
real false merges.

---

## 1. What a PCFG would actually be here

### 1.1 The tree is observed, not latent

In language, the string is observed and the parse tree is latent, so parsing is
an O(n³) search over bracketings. Here the situation is inverted. The
contracted L2 topology from `neuronauts/harness/topology.py` (`AtomTopo`,
`contract`) *is* the derivation tree: junctions are internal nodes, segments are
edges carrying observations (cable length, caliber profile, tortuosity, pre and
post synapse counts), and degree-1 nodes are leaves. What is latent is only

- the **compartment label** of every segment (soma, axon trunk, collateral,
  dendritic trunk, branch, spine, glial process, ...), and
- the **attachments across atoms** (which endpoint continues into which).

So "parsing an atom" is labeling an observed tree, an exact bottom-up dynamic
program that is linear in the number of segments. There is no CKY, no chart.
The formalism is a PCFG whose derivation is known; equivalently a stochastic
typed branching process (a bottom-up probabilistic tree automaton). The
`MorphologicalPCFG.serialize_to_grammar_tokens` bracketed-string idea in
`pcfg_morphology.py` is the same object written as a string; parsing that
string would be O(n³) for no gain.

This inversion is also why the earlier autoregressive "grammar"
(`attic/pcfg_one_offs/synapse_grammar_ar.py`) scored 0.63 on merges and chance on
splits: it linearised a synapse tree into a trajectory and scored per-step
displacement NLL, so "the likelihood scores edge geometry, not branching
topology" (its own diagnosis). Adding a degree head did not rescue it because
the tokens were synapses on a nearest-vertex bridge, not typed cable segments on
real adjacency, and there was no compartment type at all.

### 1.2 The grammar, v0

Nonterminals (labels on segments), deliberately coarse:

| Symbol | Meaning | Free observable |
|---|---|---|
| `SOMA` | soma-bearing segment | nucleus table containment (`count_contained_somata`) |
| `AX_TRUNK`, `AX_COLL`, `AX_TERM` | axon trunk, collateral, terminal arbor | pre-synapse-dominated polarity |
| `DN_TRUNK`, `DN_BRANCH`, `DN_TERM` | apical/primary trunk, branch, terminal | post-synapse-dominated polarity, thicker caliber |
| `SPINE` | short leaf off a dendrite | leaf length ≲ 2 µm, one post synapse |
| `GLIA` | non-neuronal process | no synapses, distinct branching |
| `NEURON`, `NEURITE`, `VOLUME` | start symbols | see below |

Productions, in branching-process form (this keeps the rule set small and the
DP cheap): at each junction, `P(k children | X)` and, independently per child,
`P(Y | X)`. A leaf ends with a **stop type** `∈ {TERMINAL, CUT, CLIPPED}` drawn
from `P(stop | X, tip observations)`. Continuous **emissions** per label:
log-normal segment length, log-normal mean caliber, a caliber-slope term
(centrifugal tapering; Murray's law residual at forks,
`cajal_conservation_priors.py` already has the formula), tortuosity, and
Poisson or negative-binomial pre/post counts per µm.

Hard zeros (these are the cannot-links of `tree_assembly_algorithm.md` §2.3,
now as grammar structure rather than checks bolted onto a solver):

- `NEURON → SOMA …` exactly once; a second `SOMA` anywhere in a `NEURON`
  derivation has probability 0. A `MULTI`-tier root (≥2 nuclei) is therefore
  ungrammatical by construction, which is the census result (multi-soma roots
  are the catastrophic caseload) restated.
- `AX_* → DN_*` and `DN_* → AX_*` are 0 except at the soma; polarity conflicts
  become structural, not a −8.0 hand weight
  (`constrained_multicut.py::compute_edge_weight`).
- `GLIA` never carries a synapse emission; a glial subtree under `NEURON` is 0.
- Each endpoint participates in at most one attachment (biological continuation,
  not a hub), which is the existing Kruskal degree cap.

Soft, learned: caliber jumps, branch angles, collateral density, spine density,
leaf-length distributions per label, and the class mixture
`NEURON → PYR | INTERNEURON | …` (the `SANTIAGOv2PCFG` top-level idea, kept).

Start symbols matter for fragments. A soma-less atom is not a defective
`NEURON`; it is a partial derivation from `NEURITE`, and its root label is
marginalised. This is how the grammar types the `ANON` tier without pretending
to know its identity, and how a severed fragment stops being "a valid small
neuron in isolation" (the split-scoring failure noted in the findings): under
`NEURITE`, a thick, long, untapered leaf that stops is a `CUT`, and the stop
type carries the expected continuation.

Bounds matter too. 4.31 components per atom and many tips at the region face
mean every tip needs a `CLIPPED` option scored by distance to the fetch bounds,
or the grammar will read the box edge as a false split everywhere.

### 1.3 The three quantities the DP gives you

Write `β(s, X)` for the inside probability of the subtree rooted at segment `s`
given label `X`, and `α(s, X)` for the outside probability. With
branching-process productions:

```
β(s, X) = e_X(o_s) · P(k_s | X) · Π_{c ∈ children(s)} Σ_Y P(Y | X) β(c, Y)
          (leaf: k_s = 0, times P(stop_s | X, o_tip))
log P(T) = log Σ_X π(X) β(root, X)
```

One pass each way per atom, `O(|segments| · |N|²)`. Everything below is a
read-out of cached `α`, `β`.

**Attachment delta** (assembly). Attach fragment `B` at a leaf `s` of `A`: the
leaf's stop factor is replaced by a continuation into `B`, with a junction
emission `g(o_join)` for the gap, tangent alignment, and caliber match:

```
Δ_attach(A, s, B) = log Σ_X α_A(s,X) e_X(o_s) P(k=1|X) Σ_Y P(Y|X) g(o_join) β_B(r_B, Y)
                    − log P(A) − log P(B)
```

This is the "global validation at local cost" property: the whole-neuron
likelihood change is a junction-local computation once `α`, `β` are cached.
`Δ = −∞` (an inadmissible union) falls out for two somas, polarity conflicts,
and glia, without a separate rule engine.

**Seam score** (correction). For every edge `e` of an object `O`, with the two
sides re-emitted as `CUT` at the severed tips:

```
S(e) = log P(A_e) + log P(B_e) − log P(O)
```

The argmax is the predicted seam; `max_e S(e)` (or its log-sum-exp) is a
false-merge test statistic. Computable for all edges of an object in one more
pass from `α`, `β` and the sibling products.

**Tip posterior** (candidate generation).
`P(CUT | tip) ∝ Σ_X α(s,X) e_X(o_s) P(k=0|X) P(CUT | X, o_tip)`. This turns the
harness's descriptive endpoint filter (leaf ≥ 5 µm and caliber ≥ 80 nm keeps
4,861 of 5.1M, recall unknown) into a calibrated one.

---

## 2. What the repo has already established, and what it constrains

The grammar program in this repo has produced an unusually clean set of
positive and negative results. They should be read as design constraints, not
as verdicts on PCFGs, because none of them tested a parsed, typed tree grammar.

| # | Result | Where | Consequence for a PCFG |
|---|---|---|---|
| 1 | Merge decisions transfer spatially (merge_P 0.95–0.98, std 0.01); *cut* decisions do not (out-of-sample fk_split = 0.000). | STATUS Phases 2.8–2.12; `tree_assembly_algorithm.md` §1 | Put the grammar on the *cut* side as a label-light prior, since learned cut detectors are the thing that fails to transfer. |
| 2 | Detecting a bad object is easy (whole-object shape RF AUC 0.875, precision 0.41 at top-2%, base rate 3.78%). Cutting it is the problem: 2-means on synapses is −238% vs do-nothing even with oracle detection, because the second cell is a median 11% of the object. | `FINDINGS` "Closing the loop" | The validator's job is *seam location*, not object flagging. E1 must report seam top-1, not just AUC. |
| 3 | The seam is one edge on the real cable: oracle single-edge cut +79% (recursive +90.7%). Learned seam GNN: 25% top-1, −73.4% autonomous on 150 objects, +4.6% on 513; human-assist top-5 +52%. Axons: oracle +91.9%, learned −60%. | `FINDINGS` "Connectivity cut", "Track B", "Axon frontier" | The oracle proves the signal is in the topology. The GNN's problem is 150–513 labelled error objects. A grammar learns from thousands of *correct* subtrees instead. |
| 4 | Contamination is cheap: only 1.2% of within-arbor steps cross a seam, and the bigram grammar of v117 vs v1718 differs by KL = 0.0002 bits. | `grammar_regime.py` | Rule and emission statistics can be estimated on the label-blind atom population by EM; the noisy substrate does not need labels to learn what neurons look like. |
| 5 | The 4-token F/B/L/R bigram is nearly blind to seams (AUC 0.539) while one raw feature, gap-after, gets 0.813; continuity geometry alone gets 0.988 on de-splits and the "no-grammar" ablation matches "full". | `FINDINGS` ablations | Keep continuous emissions; never discretise geometry into tokens. Do not expect the grammar to raise pairwise merge AUC. Its value is elsewhere. |
| 6 | A BiGRU trained on synthetic splices mastered the proxy (0.82) and transferred at chance (0.479); SSL-pretrained seam nets are net-negative zero-shot. Real false merges are gap-free. | `learned_grammar_neural.py`, `seam_ssl.py` | Estimate the grammar on real gold cells; evaluate only on real false merges (the 116 mixed-lineage roots of EXP-052/056, the 354/915 column and block objects). No synthetic positives. |
| 7 | The AR synapse grammar scored 0.63 merge / chance split; "merge anomaly is GLOBAL, not sequential", "splits are RELATIONAL". | `synapse_grammar_ar.py` | Score the tree, not a walk. Give fragments a `NEURITE` start symbol and a `CUT` stop type so "should continue" is representable. |
| 8 | One exact lineage test (nucleus supervoxel → v117 root) certifies 78.9% of dendritic mass at 1.000 mass purity and excludes 93.8% of frankenmerges. | census v3, `tree_assembly_algorithm.md` §5b | The `SOMA` symbol and the one-soma zero are exact; the grammar inherits the strongest global prior for free. |
| 9 | Proximity carries no identity for axons: 0/32 doubly-adjudicated nearest-neighbour links, 9/1,063 axon fragments have their soma in-box. Cut-face EM fingerprints reach precision 1.0 at 11% coverage. | `tree_assembly_handoff.md` | Grammaticality is not identity. For the axon tail the grammar is a veto and a prior on continuation, and the carrying channels stay directed continuation + EM texture. |
| 10 | EXP-053A/B, 054, 055 all failed fail-closed gates: no checkpoint separates 14 true continuations from 29,985 confusers (precision 0.000026 at recall 0.93); only 1/14 true pairs had L2 geometry on both sides. EXP-056: no geometry-only atomiser met ≥0.90 pair recall and ≥0.50 split recall. | `results/exp05*_evaluation.md` | The substrate was the limit (MST over synapse endpoints, unbounded roots). The harness fixes that. EXP-056's predeclared bar is the natural E3 bar for a grammar atomiser. |
| 11 | Real 200 µm two-level stitch: exact shared-observation channel +0.10 ARI at Δmerge_P −0.007; adding shared-atom links reaches 50.6% assembly but costs −0.105 merge_P, and re-glues 4 of 7 separated frankenmerges (vs 1). | `tree_assembly_algorithm.md` §4.1 | The precision bill of the atom channel is frankenmerge re-gluing. A grammar admissibility check on the union is the cheapest thing that could keep atom-link recall at obs-only precision. E5. |

Two housekeeping facts from the survey belong here so nobody re-runs them:
`attic/benchmarks_semi_synthetic/benchmark_pcfg_infiller.py` (formerly in
`scripts/`) imports `TreeGrammarInfiller`, which does
not exist (the module defines `EnhancedTreeGrammarInfiller`), and builds its
"synapses" and "partners" with `rng.choice` / `rng.integers`; and
`results/exp051_evaluation.md` records that the SANTIAGO infiller "initializes
random matrices at runtime rather than loading a trained real-data grammar
checkpoint". Two attic benchmarks are worse than semi-synthetic, per
`attic/README.md`: EXP-049 calls its dense-subvolume fallback unconditionally,
and EXP-050 generates whole neurons from random walks. The
`attic/morpho_grammar/` PCFG vocabulary is a fine starting list of
symbols; nothing else from it should be reused.

---

## 3. Where the grammar earns its place

### 3.1 Validation: the verifier in the battery

`tree_assembly_algorithm.md` §5 defines propose-then-verify: candidate
generators may be sloppy, and identity enters the certified graph only through
a battery of (1) arithmetic, (2) topology/biology, (3) EM at the join, (4) a
decoy panel, (5) independence conjunction. Step (2) is currently a checklist
("union skeleton a single tree; polarity, cable, synapse-count plausibility;
prototype/DNA homogeneity", and `docs/lineage_approach.md` §"This looks like a
neuron"). The grammar replaces it with:

- `Δ_attach > −∞` (admissible) and `Δ_attach > τ` (plausible), calibrated on the
  spatial validation split;
- the parse itself: every segment of the union gets a Viterbi compartment
  label, which is the review overlay the Neuroglancer harness backlog item asks
  for;
- a free decoy margin: `Δ_attach` for the accepted candidate minus the best
  `Δ_attach` among its decoys, using the same cached `α`, `β`.

The independence argument in §5 needs channels that fail independently.
Geometry, EM texture, and lineage arithmetic are one such set; a typed-branching
prior is a fourth that fails differently from all three (it does not look at
the join image, the gap, or the edit history).

### 3.2 Correction: the seam locator

This is the highest-value use, because the repo has isolated the problem so
precisely. Detection is solved well enough (0.875 / 0.41). The cut operator is
solved in principle (best single skeleton edge, +79%). The missing piece is
picking that edge, and every learned attempt has been starved of error objects.
A grammar estimates its ~400 parameters from every correct subtree in the gold
cells and, by EM, from the 20,826 label-blind atoms; the seam score `S(e)`
needs no error labels at all. Error labels are spent only on calibration and
evaluation.

Why it might work where the AR grammar did not: `S(e)` asks "are the two sides,
each with a `CUT` tip, more probable than the whole?", scored on typed cable
segments with caliber and polarity emissions on real L2 adjacency. A seam
between a thick post-dominated subtree and a thin pre-dominated one, or between
two `SOMA`-bearing subtrees, is exactly what typed productions penalise and what
per-step displacement NLL cannot see. Whether it clears the seam GNN is E2's
question, not a claim.

Human-assist first. The findings show top-k proposal plus a proofreader is the
deployable mode today (+52% at k=5 on 150 objects). A ranked `S(e)` list is that
product with an interpretable score.

### 3.3 Candidate generation: endpoint typing

The harness measured 245 endpoints per atom, most of them spines and small
protrusions, and flagged calibration of the descriptive filter as "the live
question". `P(CUT | tip)` is that calibration, and it needs no GT to compute,
only GT to evaluate (E4). It also makes the proposer directional in the way the
findings demand: a `CUT` tip of label `AX_COLL` expects an `AX_*` continuation
of similar caliber along its tangent, which is the cone-plus-caliber panel of
`l2_candidate_panel.py` with a learned prior on it.

### 3.4 Assembly: the grammar term in constrained Kruskal

The design's level-k inference is a maximum-weight spanning forest with cycle
rejection, endpoint-degree cap, and cannot-links. The grammar slots in as

```
w(e) = logit_local(e)  +  λ · Δ_attach(e)          (−∞ if inadmissible)
```

with `logit_local` from the learned stitch scorer (geometry, EM, pooled DNA).
Two practical points:

- `Δ_attach` depends on the current super-fragment, so after each accepted
  merge the union's `α`, `β` must be recomputed (linear in the union) and the
  candidate edges incident to its endpoints re-scored. This is lazy
  re-evaluation with a priority queue, the same net-evidence-between-clusters
  mechanism that makes GAEC refuse over-merges where union-find collapses. It
  keeps the hierarchy's `O(N log N)` shape.
- The see-through problem (weak middle fragment B blocking A–C) is handled
  because the union's likelihood is joint over A, B, C; no separate prototype
  loop is needed.

This is a term in the scorer, not the scorer. The findings' ablation result
(geometry alone 0.988, grammar adds nothing on de-splits) should be expected to
reproduce for the *local* decision; the grammar's contribution in E5 is measured
on frankenmerge re-gluing and merge_P, not on pairwise AUC.

### 3.5 The tiered-identity frame, restated as start symbols

| Tier (`tier_census.py`) | Grammar reading |
|---|---|
| `NAMED` (exactly one nucleus) | derivation from `NEURON`; identity = that soma |
| `MULTI` (≥2 nuclei) | probability 0 under `NEURON`; seam locator runs on the soma-to-soma path |
| `BIG-NOSOMA` | derivation from `NEURITE`, high mass; deferred as its own node |
| `ANON` | derivation from `NEURITE`; typed, not identified |
| `GLIA` | derivation from `VOLUME → GLIA`; label-free rejection complementing the served cell-type table |

---

## 4. Where it does not help

- **Identity.** A grammar says whether a union is a well-formed neuron, never
  whose neuron it is. The `ANON` axon tail stays anonymous until a directed
  continuation plus EM verifier certifies a link. Any plan that uses grammar
  likelihood to *attribute* an orphan will repeat the 0/32 proximity result.
- **The fine merge decision.** Two facing tips within a micron are decided by
  continuity geometry and, ultimately, EM at the cut face. Do not spend E-time
  trying to beat 0.988 with a grammar.
- **Context dependence.** Real neurons are not context-free: tuft depth depends
  on soma layer, collateral density on cell class, total cable on a budget.
  Mitigations: a class mixture at the top of the grammar (marginalised when the
  class is unknown), and global budgets (synapse-count cap, spatial
  compactness) kept as cannot-links outside the grammar. Accept the
  approximation; measure its cost in E1 by stratifying on class.
- **Skeleton quality.** L2 caliber at tips is thin and noisy (p50 26 nm) and the
  handoff chose kimimaro for real radii when morphology matters. Emissions must
  be robust (heavy-tailed) and the caliber terms should be validated on the
  kimimaro pass before they carry weight. Bad skeletons poison a grammar exactly
  as they poison tree-DNA.
- **Rooting.** Centrifugal direction is defined for `NEURON` (root at the
  soma). For `NEURITE` fragments choose the max-caliber junction as root and
  marginalise the root label; report sensitivity to that choice in E0.

---

## 5. Learning the grammar

1. **Supervised counts on gold cells.** Take the gold-tier proofread cells from
   `scripts/fetch_proofread_manifest.py` (the substrate module cites roughly 250
   in its region), contract their skeletons with `harness/topology.py`, and
   label segments: `SOMA` from nucleus containment; axon vs dendrite from
   synapse polarity (the harness measured ~95% pure at atom level); apical vs
   basal from caliber and direction to pia (label noise; flag it); `SPINE` from
   leaf length and a single post synapse. Rule and degree probabilities are
   normalised counts; emissions are fitted per label. Hundreds of cells give
   tens of thousands of junctions and segments for ~400 parameters.
2. **EM on the label-blind population.** Run inside-outside over the 20,826
   tier-10 atoms (then tiers 5 and 1) to refine emissions on the actual v117
   substrate. Result #4 (KL ≈ 0, 1.2% pollution) is the licence to do this
   without labels. Keep the gold cells as held-out for checking that EM does
   not drift the labels.
3. **Calibration.** Temperature-scale `S(e)` and `Δ_attach` on the spatial
   validation split with the seam buffer (`treestitch/calibration.py`), so the
   abstention thresholds in E2 and E5 mean what they say.
4. **No synthetic training.** Positives are real gold trees. Negatives, where
   needed for calibration, are the real mixed-lineage roots.

---

## 6. Integration map

| Piece | Where | Notes |
|---|---|---|
| Grammar spec and parameters | new `neuronauts/harness/grammar.py` | symbols, productions, emissions, hard zeros; serialisable |
| Inside/outside on `AtomTopo` | new `neuronauts/harness/parse.py` | vectorised over segments; returns `log P`, Viterbi labels, `α`, `β`, tip posteriors |
| Fit | new `scripts/fit_grammar.py` | gold counts + EM; writes a versioned parameter file |
| Seam locator / validator | new `scripts/probe_grammar_seams.py` | E1, E2 on the EXP-052/056 mixed roots |
| Endpoint typing | new `scripts/probe_grammar_endpoints.py` | E4; extends the endpoint table in `data/substrate/topology/k10.npz` |
| Atomiser | `treestitch/atomize.py` | E3: cut at `S(e)` seams above threshold instead of the 10 µm rule; keep the same-parent soft edge |
| Stitch scorer | `treestitch/stitch.py` constrained Kruskal; `neuronauts/assemble/fragment_graph.py::score_edge` | E5: add `λ·Δ_attach` and admissibility; lazy re-scoring on accept |
| Hand-weighted solver | `neuronauts/global_merge/solver/constrained_multicut.py::compute_edge_weight` | replace the −8.0 polarity and −5.0 DNA repulsions with log-probability deltas; also note `NeuronHypothesis.is_valid_tree` is hard-coded `True` there |
| Review product | `treestitch/stitch_viz.py` | parse labels as a compartment layer; review queue ranked by `−log P` per hypothesis |
| Retired (done) | `attic/morpho_grammar/{pcfg_morphology, blind_pcfg_morphology, santiago_v2_grammar, tree_grammar_infiller}.py`, `attic/benchmarks_semi_synthetic/benchmark_pcfg_infiller.py` | moved by the consolidation plan's Phase 2 on 2026-09-01; keep the symbol vocabulary as a starting list; `neuronauts.morpho_grammar` still imports through a deprecation shim |

The handoff's plan item 6 ("PCFG vs learned scorer on identical candidates")
should become three arms on identical candidates: learned alone, grammar alone,
learned + grammar. The interesting number is the third minus the first.

---

## 7. Experiments, smallest first, each with a predeclared bar

All on the 100 µm harness cube, spatial train/val split with seam buffer,
grouped by cell. Costs assume the tier-10 topology and the gold overlay
(handoff steps 2 and 4) exist. Both now do: `results/atom_labels_v1822.json`
(built by `scripts/build_atom_labels.py`) reports 279,075 atoms labelled at
v1822, of which 2,444 are mixed-lineage (56 among proofread cells), 2,357 are
pure gold, and 474 gold owner roots. E1 and E2 therefore have 2,444 real
mixed-lineage objects in the cube rather than EXP-056's 116, with the caveat
that only the proofread subset is adjudicated; report the two strata
separately.

| ID | Question | Data | Bar (fail-closed) | Cost |
|---|---|---|---|---|
| **E0** | Does the grammar fit and parse sensibly? | gold cells | held-out gold per-µm log-lik within 10% of train; Viterbi axon/dendrite labels agree with polarity ≥ 0.9 on segments with ≥ 5 synapses; root-choice sensitivity < 5% of log-lik on `NEURITE` fragments | 1 day |
| **E1** | Does `max_e S(e)` detect real false merges? | 116 mixed-lineage roots (EXP-056) + matched single-lineage roots, later the 354 column objects | AUC ≥ 0.875 and precision@top-2% ≥ 0.41 (beat `global_shape_merge.py`); report per class | 1 day |
| **E2** | Does `argmax_e S(e)` locate the seam? | same objects with true seam edge from lineage | top-1 seam accuracy > 25%; human-assist top-5 net ≥ +52% (150-object regime); autonomous with abstention net > 0; axon-side reported separately against the +91.9% oracle | 1–2 days |
| **E3** | Grammar atomiser vs EXP-056 rules | EXP-056 protocol, target lineage withheld | ≥ 0.90 same-lineage pair recall **and** ≥ 0.50 cross-lineage split recall (no geometry rule met both); perfect roots > 25/116 | 1 day |
| **E4** | Are `CUT`-typed tips the real split sites? | gold-overlaid atoms; a tip is a true split site if the atom continues into the same proofread cell across it | recall ≥ 0.9 of true split sites while keeping ≤ 1% of endpoints (vs 0.10% kept by the descriptive filter at unknown recall) | 1 day |
| **E5** | Does `Δ_attach` keep atom-link recall at obs-only precision? | two-level stitch, real 200 µm box, 3 seeds | frankenmerges re-glued ≤ 1 of 7 (obs-only) at assembly ≥ 50% (atom-links); Δmerge_P ≥ −0.01 | 2 days |

E1 and E2 are the same run. If E1 fails the 0.875 bar, stop and write it up;
the global-shape RF stays the detector and the grammar is demoted to E4/E5. If
E2 clears top-1 > 25% but autonomous net stays negative, the deliverable is the
human-assist ranked list, which the findings already show is the deployable
mode.

---

## 8. Risks

- **It is a prior, and priors saturate.** Once EM and typed emissions are in,
  the marginal value over the shape RF may be small for detection. That is why
  E2 (seam) and E4 (tips), where no comparable label-light tool exists, are the
  load-bearing experiments, not E1.
- **Pyramidal-centric rule tables.** The existing symbol set is pyramidal
  (apical/basal). Interneurons, chandelier axons, and glia need their own
  productions or the grammar will flag them as errors. The class mixture and a
  `GLIA` branch are not optional.
- **Bounds artefacts.** Without `CLIPPED`, every region face becomes a seam.
  The census already found oddness flags dominated by bbox clipping; the same
  failure awaits a grammar that ignores it.
- **Metric mirages.** Pairwise AUC was "a mirage" in the findings, and the
  correction work moved to net pair-error vs do-nothing and connectivity. Every
  grammar experiment reports those, not AUC alone.
- **Provenance.** `neuronauts/report/provenance.py` audits EXP-05x blocks as
  partial (no input hashes, no dirty flag). Grammar results should stamp the
  parameter file hash and the harness tier.

---

## 9. Recommendation

Build the parser on `AtomTopo` and run E0 → E1/E2 first. It is one to three
days, it reuses the substrate that the harness spent this week building, and
it targets the single problem the repo has isolated most sharply: choosing the
cut edge without a large corpus of labelled errors. If it works, the same
cached `α`, `β` feed the verifier (3.1), the tip filter (3.3), and the stitch
term (3.4) with no new machinery. If it does not, the negative is cheap and
specific, and the morpho-grammar directory can be retired with a clear
epitaph either way.
