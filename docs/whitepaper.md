# Neuronauts: A Connectome Grammar for EM-to-Connectome Inference

## Abstract

Modern connectomics pipelines are usually fragmented across three loosely
coupled stages: EM perception, local reconstruction or proofreading, and
downstream connectome extraction. This decomposition creates an optimization
gap. Learned components are often trained on proxy tasks such as membrane
segmentation, local merge AUC, or edit imitation, while the scientific target
is the connectome itself.

`neuronauts` is a unified research repo that tries to close that gap. The repo
contains the perception path from MICrONS/CAVE boxes, a shared learned grammar
over neurite-path structure, box-scale hypothesis assembly, and terminal
connectome evaluation using synapse line-graph F1. The central claim is that a
single coordinate-free path representation can be shared across local merge
plausibility, cluster atomicity, and box-scale assembly selection. The current
repo implements the first nontrivial version of that claim: multitask training
of a shared `PathEncoder`, runtime use of learned merge and atomicity scoring,
beam-style box assembly, and hypothesis reranking against box-level connectome
quality. Larger-scale whole-neuron and millimeter-scale inference remain future
work.

## 1. Motivation

The practical goal in connectomics is not merely to produce an acceptable voxel
segmentation. The actual scientific target is object identity at the synapse
level: which synapses belong to the same neuronal process, and therefore which
connections exist in the final connectome.

Segmentation errors matter only insofar as they alter that induced topology.
This suggests a different decomposition:

1. produce local fragment evidence from EM
2. learn a shared grammar for neurite compatibility
3. evaluate candidate assemblies by their effect on connectome correctness

That shift has two consequences:

- local metrics such as merge AUC remain useful, but become internal health
  signals rather than the primary target
- the terminal scalar should be line-graph F1 over synapses, because that is
  the closest box-scale proxy for downstream connectome quality

## 2. System Overview

The current repo is organized into four layers rather than a single monolith.

### 2.1 Perception layer

Implemented in:

- `neuronauts/fetch.py`
- `neuronauts/fields.py`
- `neuronauts/vectorized.py`
- `neuronauts/membrane_unet.py`

Responsibilities:

- fetch MICrONS/CAVE boxes and synapses
- fetch or cache EM and membrane-like fields
- run local agent exploration
- produce path traces and synapse hits
- expose synthetic and real-box evaluation modes

This layer is intentionally local. It produces evidence, not final identity.

### 2.2 Shared grammar layer

Implemented in:

- `neuronauts/grammar.py`
- `neuronauts/shared_grammar_model.py`
- `neuronauts/topology_model.py`

Responsibilities:

- encode variable-length path descriptors with a shared `PathEncoder`
- score pairwise fragment compatibility with `MergeScorer`
- summarize fragment sets with `ArborEncoder`
- train a shared grammar against both merge and atomicity supervision
- train an attention-based atomicity head over padded branch embeddings

This is the learned center of the system.

### 2.3 Assembly layer

Implemented in:

- `neuronauts/assembly.py`
- `neuronauts/assembly_dataset.py`
- `neuronauts/hypothesis_reranker.py`
- `neuronauts/run.py`

Responsibilities:

- construct candidate local merges
- search over a limited set of box-scale merge decisions
- use learned merge scores and atomicity scores during assembly
- export multiple box-level hypotheses
- train a reranker to prefer hypotheses with better line-graph F1

This layer is where the repo starts to move beyond purely local heuristics.

### 2.4 Research-loop layer

Implemented in:

- `neuronauts/experiment_driver.py`
- `scripts/run_research_cycle.py`
- `scripts/codex_optimize.py`
- `scripts/gemini_researcher.py`
- `scripts/view_research_ledger.py`

Responsibilities:

- define one canonical end-to-end experiment cycle
- support both Codex and Gemini as outer-loop proposal sources
- evaluate on selection and holdout box sets
- make keep or revert decisions from structured metrics
- record all runs in a shared ledger and leaderboard

This is how the repo operationalizes an `autoresearch`-style loop without
tying itself to one external model provider.

## 3. The Shared Grammar Hypothesis

The core hypothesis is that a single path representation can support three
closely related but distinct decisions:

1. whether two local fragments should merge
2. whether a set of synapse-side fragments is atomic or contains a merge error
3. whether one box-level assembly hypothesis is better than another

The representation should be:

- coordinate-free
- reusable across volumes
- shared across local and global tasks
- trainable from real supervision signals, not only synthetic ones

In the current repo this hypothesis is instantiated as follows:

- local path sequences are converted to `(edge_len, radius, curvature)` style
  descriptors
- `TorchPathEncoder` produces shared fragment embeddings
- `TorchMergeScorer` scores pairwise merge plausibility
- `TorchArborEncoder` summarizes fragment sets for global atomicity
- an attention-based topology head is trained on padded branch sets

This is still a compact baseline, but it is no longer just a hand-coded
feature spreadsheet. The key point is that the same encoder weights are now
updated by both local merge and global atomicity supervision.

## 4. Why Line-Graph F1 Is the Primary Scalar

The terminal object of interest is the synapse line graph. Two synapses should
be connected if and only if they share a presynaptic or postsynaptic rooted
object. That means the most relevant box-scale metric is not local path
similarity in isolation, but whether the induced connectome topology is correct.

Line-graph F1 is therefore the primary reported scalar in `neuronauts`.

This does not eliminate local metrics. The repo still tracks:

- merge accuracy
- atomicity accuracy
- reranker correlation
- precision and recall for the induced line graph

But these are supporting diagnostics. The keep-or-revert decision in the outer
loop is anchored on selection-set line-graph F1 and guarded by holdout-set
performance.

## 5. Data Sources and Supervision

The current system uses three complementary supervision paths.

### 5.1 Local merge supervision

Implemented by:

- `neuronauts/merge_dataset.py`
- `scripts/export_merge_dataset.py`

Construction:

- positives are subfragments split from one rooted cluster
- negatives are nearby fragments from different roots
- each side is exported as a variable-length path sequence plus mask

This trains local merge plausibility.

### 5.2 Global atomicity supervision

Implemented by:

- `neuronauts/topology_dataset.py`
- `scripts/export_topology_dataset.py`
- `scripts/train_topology_model.py`

Construction:

- candidate pre-side and post-side clusters are built from real MICrONS boxes
- atomic examples share one true root on the relevant side
- non-atomic examples mix roots
- export preserves padded branch embeddings and raw branch sequences

This trains cluster validity.

### 5.3 Assembly ranking supervision

Implemented by:

- `neuronauts/assembly_dataset.py`
- `scripts/export_assembly_ranking_dataset.py`
- `scripts/train_assembly_ranker.py`

Construction:

- generate multiple box-level assembly hypotheses
- compute true line-graph F1 for each
- train a reranker to predict which hypotheses are better

This is the first direct bridge from local reasoning to terminal box-level
connectome quality.

## 6. Current Training Architecture

The current training path is multitask, but still modular.

### 6.1 Shared grammar training

Implemented by:

- `neuronauts/shared_grammar_model.py`
- `scripts/train_shared_grammar.py`

The shared trainer updates one `TorchPathEncoder` using:

- local merge loss from merge examples
- global atomicity loss from topology examples

This is the first point in the repo where the phrase "shared grammar" is true
in code rather than just in intent.

### 6.2 Atomicity-specific training

`scripts/train_topology_model.py` remains available as a narrower training path
for the attention-based atomicity validator. That path is still useful for
isolated experiments, even though the shared grammar trainer is the more
strategic route.

### 6.3 Hypothesis reranker training

The reranker is trained on hypothesis-level features derived from candidate
assemblies:

- number of neurons
- number of edges
- resolved and unresolved synapse counts
- estimated line-graph edge count
- mean and max synapses per neuron

This feature set is still lightweight. It is useful, but it is also one of the
clearest near-term areas for improvement.

## 7. Current Runtime Architecture

The runtime in `neuronauts/run.py` is now a hybrid learned system.

### 7.1 Perception and trace generation

- fetch or synthesize a box
- compute membrane and exploration fields
- run vectorized agents
- collect path arrays and synapse hits

### 7.2 Candidate local merge generation

Geometry still proposes candidate merges. This is deliberate. Spatial rules are
currently used for candidate generation, not as the sole decision rule.

### 7.3 Learned merge scoring

When a shared grammar checkpoint is provided:

- path sequences are derived from traces
- `PathEncoder` embeds fragments
- `MergeScorer` scores candidate joins
- those learned scores can override old overlap heuristics

### 7.4 Box-scale search

The runtime can now perform limited beam-style assembly over role-group merges:

- accept or reject candidate joins
- score states with local merge evidence
- apply atomicity weighting during search

This is still box-local search rather than full explicit whole-neuron search,
but it is a real global reasoning step relative to the original greedy merge
baseline.

### 7.5 Hypothesis selection

When an assembly reranker checkpoint is provided:

- multiple hypotheses are generated from threshold and beam-width sweeps
- hypothesis features are extracted
- the reranker chooses the best predicted hypothesis
- final line-graph evaluation is computed on that chosen graph

This closes the current runtime loop as:

perception -> shared grammar -> box search -> reranker selection -> connectome
evaluation

## 8. Research Loop Architecture

The repo now supports a concrete iterative research loop.

### 8.1 Canonical cycle

The canonical driver in `scripts/run_research_cycle.py` runs:

1. merge dataset export
2. topology dataset export
3. shared grammar training
4. assembly ranking dataset export
5. reranker training
6. selection-set validation
7. holdout-set validation

### 8.2 Outer-loop options

Two outer-loop environments are currently supported:

- `scripts/codex_optimize.py`
- `scripts/gemini_researcher.py`

These are intentionally thin wrappers over the same experiment driver. The
proposal mechanism can vary by environment, but the evaluation logic is shared.

### 8.3 Keep or revert logic

The current rule is:

- prioritize selection-set line-graph F1
- reject improvements that come with meaningful holdout regression
- in tie regions, allow coherent improvements in merge, atomicity, or reranker
  metrics

This is not the final scientific solution, but it is a reasonable and explicit
research controller.

### 8.4 Ledger and leaderboard

All outer-loop runs can write to one shared research ledger:

- `run_logs/research_ledger.jsonl`
- `run_logs/research_ledger.leaderboard.tsv`

This keeps Codex and Gemini experiments comparable rather than split across
incompatible ad hoc logs.

## 9. What Is Actually Implemented Now

The current repo implements:

- MICrONS/CAVE fetch and synthetic box generation
- optional cached membrane preprocessing
- vectorized agent exploration
- line-graph F1 evaluation
- shared torch grammar modules
- local merge dataset export
- global atomicity dataset export
- multitask shared grammar training
- attention-based topology validation
- learned merge scoring in runtime
- box-scale beam-style merge search
- assembly hypothesis export and reranking
- shared Codex/Gemini experiment driver
- selection and holdout evaluation splits
- persistent experiment ledger and leaderboard

This is enough to call the repo a real research platform rather than a loose
collection of sketches.

## 10. What Is Still Future Work

The most important remaining items are:

- richer hypothesis features or a learned graph-level reranker
- stricter split discipline with a true final held-out test tier
- explicit hierarchical whole-neuron assembly beyond local role-group search
- sparse large-scale inference for millimeter-scale neurons
- stronger coupling between hypothesis ranking and final connectome metrics

The current repo is best understood as a box-scale connectome-grammar platform,
not yet a full whole-neuron or whole-volume system.

## 11. Relationship To Existing Work

### RoboEM

RoboEM is close in spirit at the local tracing level. The distinction here is
that `neuronauts` is organized around synapse-object topology and connectome
correctness rather than local flight alone.

### Auto-proof and edit-imitation systems

Those systems provide important supervision ideas for local decisions. The
additional move here is to unify them with global root-consistency supervision
and box-level assembly selection.

### NEURD and morphology-centric systems

NEURD is highly relevant, especially for morphology-rich validation. The main
difference here is that the center of gravity is topological correctness of the
induced connectome rather than morphology annotation as an end in itself.

### Graph-based global inference

`neuronauts` is also adjacent to graph-based inference systems. The distinction
is that the graph is not the whole story: learned path and cluster
representations are treated as the core modeling surface, with search and graph
structure acting as the scaffold around them.

## 12. Conclusion

`neuronauts` now has a coherent architecture:

- local EM perception
- shared neurite grammar
- box-scale learned assembly
- terminal connectome evaluation
- an explicit iterative research loop

The current system is still limited in scale and expressivity, but it is now a
substantive platform for testing the central thesis: that connectome inference
should be driven by a shared grammar of neurite structure and judged by the
correctness of the induced connectome, not only by local segmentation proxies.

## References

1. Karpathy, A. `autoresearch`. GitHub repository. <https://github.com/karpathy/autoresearch>
2. MICrONS Consortium et al. Functional connectomics spanning multiple areas of mouse visual cortex. *Nature* 2021. <https://www.nature.com/articles/s41586-021-03778-x>
3. Silversmith, W. `cloud-volume`. GitHub repository. <https://github.com/seung-lab/cloud-volume>
4. CAVEconnectome. `CAVEclient`. GitHub repository. <https://github.com/CAVEconnectome/CAVEclient>
5. Bae, J. A. et al. Digital museum of retinal ganglion cells with dense anatomy and physiology. *Cell* 2024. <https://www.cell.com/cell/fulltext/S0092-8674(24)00308-4>
6. Li, P. H. et al. RoboEM: neurite reconstruction from 3D EM by AI-based direct image-to-trace translation. *Nature Methods* 2024. <https://www.nature.com/articles/s41592-024-02226-5>
7. Grady, L. Random walks for image segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 2006. <https://pubmed.ncbi.nlm.nih.gov/17063682/>
8. NEURD paper. *Nature* 2025. <https://www.nature.com/articles/s41586-025-08660-5>
