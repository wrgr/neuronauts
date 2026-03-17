# Neuronauts: A Connectome Grammar for EM-to-Connectome Inference

## Abstract

Current connectomics systems are typically split across three mismatched
layers: perception from electron microscopy (EM), local reconstruction or
proofreading, and downstream connectome extraction. This separation creates a
persistent optimization gap: the learned components are usually trained on
proxy tasks such as merge AUC, membrane masks, or local edit imitation, while
the scientific target is the connectome itself. We propose a unified
architecture that closes that gap inside a single repo. `neuronauts` provides
the EM perception layer, the shared grammar layer, and the terminal connectome
metric: membrane-field inference, agent-based fragment proposals, learned path
representation (`PathEncoder`), pairwise fragment compatibility
(`MergeScorer`), cluster atomicity and arbor encoding (`ArborEncoder`),
beam-search assembly, and line-graph F1 on real MICrONS/CAVE data. In the
unified system, the central editable model file is
`neuronauts/grammar.py`, and all learned components are pulled by the same
terminal objective: line-graph F1, not local AUC alone. The core claim is a
connectome grammar: a transferable learned representation of neurite path
structure that jointly supports local merge plausibility, global cluster
atomicity, and full connectome assembly from EM voxels to synaptic topology.

## 1. Motivation

The practical bottleneck in connectomics is not merely voxel segmentation. The
end goal is to recover which synapses belong to the same rooted neuronal object
and therefore which synaptic connections exist in the final connectome.
Segmentation merges and splits matter only insofar as they corrupt this
topology. Existing systems often spend most of their learning capacity on local
proofreading or dense image-space repair, then hope the connectome falls out as
a byproduct.

That is backwards for the present use case. The primary target is the synapse
line graph: two synapses should be connected if and only if they share a
presynaptic or postsynaptic rooted object. This suggests a different system
decomposition:

1. learn how to perceive local neuronal structure from EM,
2. learn a global grammar for fragment and arbor compatibility,
3. optimize the whole stack against connectome correctness.

The relevant ingredients previously lived in separate codebases. The right move
is not to keep them split. The right move is to absorb the useful grammar,
fetch, and search patterns into `neuronauts` and make one runtime home with one
terminal metric.

## 2. Unified System Overview

The architecture has three layers.

### Layer 1: EM perception

This layer converts local MICrONS EM boxes into structured candidate evidence.

- EM volume fetch from MICRONS/minnie65 through CAVE and CloudVolume
- optional learned membrane preprocessing (small U-Net cache)
- membrane or boundary field construction
- agent navigation through the local volume
- fragment or path proposal generation
- synapse capture and pre/post cluster candidate generation

This layer is intentionally local and evidence-generating. Its role is not to
decide final neuron identity. Its role is to create plausible fragments and
candidate synapse-side clusters from real EM.

### Layer 2: shared grammar

This is the learned center of the system.

- `PathEncoder`: raw path sequence `(edge_len, radius, curvature) -> 32D`
- `MergeScorer`: pairwise fragment compatibility
- `ArborEncoder` / `topology_atomicity`: global cluster atomicity and arbor
  validity
- beam search: global hypothesis assembly over fragments and joins
- optional LLM oracle: sparse hypothesis-level coherence scoring

The key point is that all of these should be driven by shared learned
representation weights. `PathEncoder` is not only a local merge prior. It is
the learned coordinate-free grammar of neurite structure.

### Layer 3: connectome extraction

This layer converts assembled objects into the thing we actually care about.

- role-separated pre and post object assignment
- line-graph construction over synapses
- connectome evaluation using precision, recall, and line-graph F1

In the unified view, this layer is not merely a report generator. It is the
terminal objective that should pull the entire learned grammar.

## 3. The Shared Grammar Hypothesis

The central hypothesis of this paper is that there exists a single transferable
representation of local neurite path structure that can support three different
but related decisions:

1. local merge plausibility
2. cluster atomicity vs internal merge error
3. global arbor grammaticality under beam-search assembly

The representation should be:

- coordinate-free
- reusable across volumes
- shared across local and global tasks
- trainable from both edit history and connectome consistency

The current `neuronauts` implementation contains only a lightweight baseline in
`neuronauts/grammar.py`. The proposal is to evolve that shared encoder family
upward into cluster and arbor validation rather than train isolated local
models for each subproblem.

## 4. Why Line-Graph F1 Should Replace Local AUC

Local merge AUC is useful as a ranking metric for merge plausibility, but it is
still a proxy. A merge scorer can improve local AUC while leaving the actual
connectome unchanged or even worse after global assembly. The desired terminal
behavior is not "rank local join decisions well in isolation." The desired
behavior is "assemble rooted objects whose synapses induce the correct
connectome."

Therefore the primary scalar in the unified system should be line-graph F1.

This does not mean local diagnostics disappear. They remain useful for
debugging and ablation. But they should become secondary metrics under the
terminal connectome objective.

Concretely:

- `results.tsv` should center line-graph F1
- `program.md` should state connectome correctness as the primary target
- local merge AUC should be treated as an internal health signal

This change is both scientific and engineering discipline: it forces the
editable model file to optimize the actual task.

## 5. Data Sources and Supervision

The system has two complementary supervision sources.

### 5.1 CAVE edit decisions

These supervise pairwise merge quality.

- examples of accepted or rejected joins
- hard reversals and difficult proofreading cases
- local path-pair compatibility labels

This is naturally suited to `PathEncoder + MergeScorer`.

### 5.2 Synapse root consistency

These supervise global atomicity.

- candidate pre-side or post-side synapse clusters from real MICrONS boxes
- label `atomic` if all synapses in the cluster share one true root on the
  relevant side
- label `non_atomic` otherwise

This is naturally suited to `ArborEncoder` or `topology_atomicity`.

These two supervision sources are not redundant. They pull on different scales
of the same grammar:

- edit decisions supervise local compatibility
- root consistency supervises global topological validity

Both should update the same encoder weights.

## 6. From Local Fragments to Global Assembly

The unified pipeline can be read as a sequence of increasingly global
decisions.

### Step 1: local perception

`neuronauts` fetches a MICrONS volume and synapses, optionally runs a membrane
U-Net cache, and generates agent-based fragment proposals.

### Step 2: local compatibility

`neuronauts` scores pairwise fragment joins using `PathEncoder + MergeScorer`.

### Step 3: cluster atomicity

Candidate pre-side or post-side synapse clusters are scored for atomicity. A
non-atomic cluster should fragment before root identity is assigned.

### Step 4: beam-search assembly

Global hypotheses are assembled from local fragments and bridge proposals. Beam
search explores coherent join sequences rather than committing greedily.

### Step 5: oracle re-ranking

An optional LLM oracle scores high-level coherence and identity of assembled
hypotheses. This is not the core learned representation; it is a sparse global
reasoning layer above it.

### Step 6: connectome extraction

The accepted global hypothesis induces rooted objects and therefore a synapse
line graph, which is evaluated directly.

## 7. What Is Actually New

The strongest claim is not:

- "we built another local merge model"
- "we built another tracing agent"
- "we added an LLM to proofreading"

The stronger claim is:

> A connectome grammar: a learned coordinate-free representation of neurite path
> structure that simultaneously predicts local merge plausibility, cluster
> atomicity, and global arbor grammaticality, and is optimized against terminal
> connectome correctness.

That claim requires both kinds of machinery:

- EM perception and real-data connectome evaluation
- shared grammar and global assembly machinery

The point of the unified repo is to stop treating those as separate systems.

## 8. Relationship To Existing Work

### RoboEM

RoboEM is a learned local neurite flight/tracing system. It is adjacent in
spirit, but its center of gravity is local steering and reconstruction support.
The present system is centered on synapse-object topology and downstream
connectome correctness rather than local flight alone.

### Auto-proof / edit-imitation systems

These learn proofreading decisions from manual edits. That is an important part
of the supervision story here, but not the whole story. The unified system also
learns global atomicity from synapse root consistency and optimizes the final
line graph.

### NEURD

NEURD is morphology-rich and highly relevant. The overlap is real. The key
distinction in the present framing is the emphasis on topology over morphology:

- whether a synapse-linked cluster is one rooted object at all
- whether it should fragment
- whether the final line graph is correct

The scientific center is therefore not only neurite morphology or proofreading,
but connectome topology.

### Random-walk / graph methods

The present formulation is also close in spirit to graph-based global inference,
including random-walk style reasoning. But simple reachability or harmonic
extension alone remains heuristic. The learning contribution here is in the
shared edge and fragment representation, with graph structure acting as the
inductive bias.

## 9. Training And Optimization Loop

The outer optimization loop should be simple:

1. edit the shared model file
2. train on both edit decisions and topology atomicity
3. evaluate local diagnostics and terminal line-graph F1
4. keep or revert

This means the true "single editable file" in the unified system is:

- `neuronauts/grammar.py`

The outer Codex loop should optimize that file because:

- local merge quality depends on it
- cluster atomicity depends on it
- global assembly quality depends on it
- terminal connectome F1 depends on all of the above

In this view, the LLM is not the learned model. The LLM is the research agent
that modifies the learned model and its training loop.

## 10. Immediate Implementation Path

The practical next steps are clear.

1. Keep `neuronauts` as the EM perception and connectome-evaluation layer.
2. Keep the grammar and global-search layer inside `neuronauts/grammar.py`.
3. Continue exporting real MICRONS/CAVE atomicity examples from `neuronauts`.
4. Feed those examples into the shared `PathEncoder` training path.
5. Replace local AUC as the primary reported scalar with line-graph F1.
6. Use beam search over fragment and cluster hypotheses, not only local merge
   scores.

The recent `export_topology_dataset.py` script in `neuronauts` already shows the
correct direction: it builds real training examples from CAVE root assignments
on MICRONS boxes. Those data can be unified with edit-decision supervision so
both forms of training update the same grammar.

## Conclusion

The perception, grammar, and evaluation layers should live in one codebase.
`neuronauts` should be that codebase. It already has the real-data fetch path,
the topology supervision export, and the terminal metric. With the grammar
layer absorbed locally, it becomes a credible path to a real EM-to-connectome
learner instead of a collection of adjacent prototypes.

## References

1. Karpathy, A. `autoresearch`. GitHub repository. <https://github.com/karpathy/autoresearch>
2. MICrONS Consortium et al. Functional connectomics spanning multiple areas of mouse visual cortex. *Nature* 2021. <https://www.nature.com/articles/s41586-021-03778-x>
3. Silversmith, W. `cloud-volume`. GitHub repository. <https://github.com/seung-lab/cloud-volume>
4. CAVEconnectome. `CAVEclient`. GitHub repository. <https://github.com/CAVEconnectome/CAVEclient>
5. Bae, J. A. et al. Digital museum of retinal ganglion cells with dense anatomy and physiology. *Cell* 2024. <https://www.cell.com/cell/fulltext/S0092-8674(24)00308-4>
6. Li, P. H. et al. RoboEM: neurite reconstruction from 3D EM by AI-based direct image-to-trace translation. *Nature Methods* 2024. <https://www.nature.com/articles/s41592-024-02226-5>
7. Grady, L. Random walks for image segmentation. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 2006. <https://pubmed.ncbi.nlm.nih.gov/17063682/>
8. NEURD paper. *Nature* 2025. <https://www.nature.com/articles/s41586-025-08660-5>
