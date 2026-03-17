# Global Validation Dataset And Learning Loop

## Core Position

The interesting learned object is not the LLM.

The LLM is only an outer research optimizer:

- proposes code changes
- launches experiments
- keeps or reverts

The actual trainable model should be a global validation model over candidate
synapse clusters or fragment graphs.

That gives us a real inner learning loop, which is what makes the project feel
less hand-tuned and more publishable.

## What Is New Here

This project should emphasize **topology and atomicity**, not only morphology.

NEURD is strongly morphology-centric:

- mesh decomposition
- geometric and structural annotations
- proofreading of merge errors
- downstream analyses on neurite morphology

That is adjacent, but the proposed Neuronauts layer is different if we focus on:

- whether a candidate synapse-linked cluster is topologically atomic
- whether it should split into multiple neurons
- which global neuron identity a fragment belongs to
- downstream line-graph/connectome correctness as the objective

In short:

- NEURD: morphology-rich proofreading and annotation
- Neuronauts global layer: topological validation and atomicity over synapse/fragment graphs

This is also why the work feels closer in spirit to graph-based methods such as
Leo Grady's random walker tradition than to pure morphology pipelines: the
central object is a structured graph decision problem, not only geometric
feature extraction.

## First Concrete Learning Task

Start with the smallest useful task:

### Task

Binary classification:

- input: one candidate synapse cluster
- output: `atomic` or `non_atomic`

Meaning:

- `atomic`: the cluster corresponds to exactly one neuron side
- `non_atomic`: the cluster internally contains at least one merge error and
  should be fragmented

This is the cleanest first version of the global validation layer.

## What Is One Training Example?

A training example should be one candidate cluster constructed from existing
pipeline outputs.

Recommended object:

- one pre-side or post-side cluster from the current role-separated merge stage

Each example contains:

- cluster id
- role: `pre` or `post`
- synapse ids in the cluster
- synapse 3D coordinates
- fragment/path geometry from agents or segments
- optional local membrane/EM embeddings
- optional local segmentation neighborhood

The label is derived from ground-truth root IDs:

- `atomic = 1` if all synapses in the cluster map to the same true root on the
  relevant side
- `atomic = 0` otherwise

For pre-side clusters:

- use `pre_root_id`

For post-side clusters:

- use `post_root_id`

## Why This Is Good

This gives:

- exact labels from existing data
- no need for new human annotation
- direct supervision on the real failure mode we care about
- a topological target, not only a geometric proxy

## Feature Sets

The first model should not depend on a giant feature stack.

Use a narrow, layered feature design.

### Tier 1: topology and graph statistics

- number of synapses in cluster
- pairwise distance statistics
- cluster diameter
- graph density of local compatibility edges
- number of connected components in candidate fragment graph
- role purity is fixed by construction

### Tier 2: branch or path geometry

- path fragment lengths
- local branch count
- turn-angle statistics
- path overlap and separation statistics
- distance of synapses to cluster backbone

### Tier 3: EM or membrane context

- pooled membrane probabilities near synapses
- pooled membrane probabilities along candidate paths
- simple learned patch embeddings around synapses or path nodes

The key is:

- topology first
- local appearance second

That keeps the first paper from becoming "yet another membrane network".

## First Model Families

The first model should be simple and trainable.

### Option A: MLP on cluster features

Input:

- hand-constructed topology + geometry summary vector

Output:

- atomic probability

Pros:

- easy baseline
- fast iteration

Cons:

- limited expressiveness

### Option B: GNN on cluster graph

Input:

- node = synapse or path fragment
- edge = candidate same-neuron compatibility relation

Output:

- graph-level atomic probability

Pros:

- better fit to the structure of the problem

Cons:

- slightly more engineering

### Recommendation

Start with:

1. MLP baseline
2. then GNN if the baseline is too weak

## Negative Examples

There are two strong sources of negatives:

### 1. Natural negatives

Clusters produced by the current pipeline that contain multiple true roots.

### 2. Synthetic negatives

Construct a false cluster by combining two atomic clusters that are:

- spatially nearby
- locally plausible
- globally different roots

This is useful because:

- it makes the decision boundary sharper
- it teaches the model to reject realistic merge errors

## Validation Split

Split at the box or neuron level, not at the individual-cluster level.

Otherwise:

- train and validation leak the same local arbor structure

Preferred split:

- fixed set of validation boxes
- held-out set of robustness boxes

## Baselines

The first paper needs clear baselines.

### Baseline 1

Current heuristic atomicity rule:

- cluster is accepted if produced by current merge thresholds

### Baseline 2

Simple hand-built summary classifier:

- logistic regression or MLP on cluster statistics

### Baseline 3

NEURD-style morphology-only approximation, if feasible

- morphology features without the explicit topology/line-graph objective

This would help demonstrate that the new signal is not only morphology.

## Downstream Use

The first deployment path is conservative:

1. current pipeline produces candidate clusters
2. atomicity model scores each cluster
3. low-confidence or predicted non-atomic clusters are:
   - split
   - vetoed
   - or sent to a secondary partition stage
4. final line graph is rebuilt

This means the model can initially act as a validator, not a full replacement
for the current graph construction logic.

## Evaluation

The main metric should stay downstream:

- line-graph F1

Also report:

- atomicity AUROC
- precision/recall for non-atomic detection
- number of false merges removed
- number of true atomic clusters incorrectly split

## The Inner Learning Loop

This is the missing "real training loop" in the project.

It should look like:

1. Build a cluster dataset from fixed boxes.
2. Train the atomicity model.
3. Evaluate cluster classification.
4. Plug the validator into the pipeline.
5. Evaluate downstream line-graph F1.

This is the true inner optimization loop.

The LLM outer loop can then optimize:

- feature construction
- model choice
- training hyperparameters
- integration policy

That is a better match to the original autoresearch idea than repeatedly
rerunning an unchanged inference benchmark.

## Minimal Next Build

The minimum viable implementation is:

1. Add cluster-dataset export from current `run.py` outputs.
2. Write labels from root consistency.
3. Train an MLP atomicity classifier.
4. Reinsert the classifier as a veto on non-atomic clusters.
5. Measure downstream F1.

That is small enough to build, but substantial enough to reveal whether the
global validation idea is real.
