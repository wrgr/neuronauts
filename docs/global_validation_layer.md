# Global Validation Layer

## Motivation

The current `neuronauts` pipeline is mostly local:

- local membrane or exploration field
- local synapse capture
- local role-specific merge logic
- local edge assignment into a synapse line graph

This is useful, but it misses a more interesting source of signal:

- global neuron consistency
- whole-arbor plausibility
- cluster atomicity vs internal merge error
- downstream connectome correctness at the level of neuron-scale structure

The core idea is to add a learned global validation layer above local tracing.

## Problem Statement

Given:

- a local or mesoscale EM subvolume
- a segmentation graph or edit history
- synapses and their pre/post root assignments
- candidate neuron fragments, branch fragments, or synapse clusters

learn a model that answers:

1. Is this candidate cluster atomic?
   Meaning: does it correspond to one neuron or does it contain an internal merge error?

2. If not atomic, how should it be partitioned?
   Meaning: split the candidate into a small number of globally consistent neuron fragments.

3. If atomic, what larger neuron/root/arbor should it attach to?
   Meaning: infer global neuron identity or nearest compatible continuation from morphology and connectivity.

This is a global structured inference problem, not only a local tracing problem.

## Why This Is Interesting

For a fixed collection of branch fragments, the hypothesis space is often much
smaller than it appears locally. Large-scale neuron structure imposes strong
constraints:

- branch continuity
- polarity consistency
- arbor geometry
- synapse distribution
- impossible self-crossings or duplicate ownership
- graph-level parsimony

The novelty is not "another local proofreading model". The novelty is learning
global neuron validity and cluster atomicity directly against downstream
connectomic correctness.

## Proposed Objects

There are three natural object types the model could operate on.

### 1. Synapse cluster

A candidate set of synapses believed to belong to one neuron side
(pre-side or post-side).

Use case:

- validate whether the cluster is atomic
- split if it contains an internal merge

### 2. Branch fragment graph

A graph of candidate branch fragments or agent-generated path fragments, with
edges indicating possible same-neuron attachment.

Use case:

- global assembly of neuron fragments
- attachment or exclusion decisions

### 3. Edit neighborhood

A local subgraph around historical proofreading edits in an existing
segmentation/reconstruction.

Use case:

- learn from accepted human edits
- predict whether a similar region should be split, merged, or left alone

## Inputs

The validation layer should avoid hand-coded morphology heuristics as the main
decision rule. Instead, it should learn from structured inputs such as:

- local EM embeddings
- cached membrane or affinity fields
- branch-fragment geometry
- synapse positions and densities
- polarity labels or pre/post role
- segmentation graph neighborhoods
- edit history around the candidate
- optional mesh or skeleton-derived representations

These inputs can be converted into a graph representation:

- node = fragment, synapse cluster, or segment piece
- edge = candidate same-neuron attachment or incompatibility relation
- label = atomic / non-atomic, split assignment, or attachment identity

## Outputs

The model should support three related predictions.

### Atomicity score

Binary or probabilistic:

- atomic
- contains merge error

### Partition proposal

For non-atomic candidates:

- assign nodes/fragments to subclusters
- produce a split proposal

### Attachment / identity proposal

For atomic candidates:

- predict best matching continuation
- predict root identity or compatible larger arbor

## Supervision

This can learn largely from existing connectomics artifacts.

Potential labels:

- accepted proofreading edits
- final proofread root assignments
- pre/post synapse root consistency
- line-graph correctness
- segmentation graph corrections

Derived supervision:

- synthetic merge errors inserted into proofread neurons
- synthetic split errors
- cluster atomicity labels from known root assignments
- pairwise fragment same-root vs different-root labels

## Model Families

Three plausible first versions:

### 1. Graph classifier

Input:

- candidate fragment graph or synapse-cluster graph

Output:

- atomic vs non-atomic

Strength:

- simplest starting point

### 2. Graph partitioner

Input:

- candidate fragment graph

Output:

- partition of nodes into atomic subclusters

Strength:

- directly solves split decisions

### 3. Fragment compatibility scorer

Input:

- two candidate fragments plus context

Output:

- same-neuron compatibility score

Strength:

- can be composed into global assembly by optimization

## Evaluation

Primary evaluation should remain downstream connectomic quality, not only local
classification accuracy.

Recommended metrics:

- line-graph F1
- precision / recall for connectome edges
- atomicity classification AUROC
- split proposal accuracy
- attachment accuracy
- number of false merges removed
- number of false splits introduced

## Relationship To Current `neuronauts`

Current `neuronauts` can be viewed as:

1. local perception and tracing
2. weak global assembly through merge thresholds
3. line-graph construction

The proposed validation layer would strengthen step 2.

The likely medium-term architecture is:

1. local perception
   - EM, membrane cache, affinities, or learned local embeddings
2. fragment proposal
   - agent traces or segmentation fragments
3. global validation layer
   - atomicity, partition, attachment
4. connectome extraction
   - assign synapses and build the line graph

## Relationship To Existing Work

### Similar to NEURD

- post hoc proofreading
- morphology-aware reasoning
- graph-based reasoning over reconstructed neurons
- merge-error detection and decomposition

### Different from NEURD

- centered on synapse-cluster atomicity and downstream line-graph correctness
- aims to validate whether a candidate cluster is one neuron at all
- potentially uses synapse line graph as the central object, not only mesh-derived morphology graphs
- can learn from edit history and connectome targets jointly

### Similar to Autoproof-style systems

- learns from proofreading outcomes or edit decisions
- operates over segmentation/reconstruction errors

### Different from Autoproof-style systems

- not only imitating local proofreading edits
- emphasizes global neuron validity and fragmentation/attachment over local accept/reject
- objective is connectomic correctness, not only edit classification

## First Concrete Version

The first realistic version should be narrow:

1. Build candidate pre-side and post-side synapse clusters from current `neuronauts`.
2. Label each cluster as atomic or non-atomic using root assignments.
3. Train a graph classifier to predict cluster atomicity from:
   - fragment geometry
   - synapse positions
   - local EM or membrane embeddings
4. Use the classifier as a veto or split trigger before final graph extraction.
5. Measure downstream line-graph F1.

This keeps the problem tractable while still moving from local heuristics toward
global validation.
