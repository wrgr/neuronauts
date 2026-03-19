# Synapse-Pair Architecture — Future Direction

> **Status: design notes only — not yet implemented.**
> Captured here for future reference.  The current system trains a merge-head
> grammar using pairwise BCE on cached synapse tables; this document sketches a
> deeper reframing that keeps that BCE objective but clarifies what the model
> should really be predicting and how to condition it on path evidence.

---

## Core reframing

Synapses are the *anchors*.  Fragments are *hidden wires*.

The model's job is **not** to reconstruct every wire.  It is to infer:

> Which anchors are likely connected by a biologically plausible hidden
> wiring pattern?

That is a more tractable and more relevant learning problem than fragment
completion or morphology reconstruction.

---

## Compact problem formulation

Let S = {s₁, …, sₙ} be synapses on a single neuron.

The substrate is a noisy pre-segmentation graph G = (V, E), where V are
fragments/supervoxels and E are observed adjacencies or continuation edges.
Each synapse sᵢ attaches to one or more substrate locations A(sᵢ) ⊆ V.

**Target**: a synapse relation y_ij ∈ {0, 1}, where y_ij = 1 means sᵢ, s_j
are topologically connected in the relevant sense.

The substrate graph matters only because it provides **candidate supporting
paths**.

---

## Model

Learn:

    p_ij = P(y_ij = 1 | sᵢ, s_j, P_ij)

where P_ij is a small set of candidate paths in G connecting A(sᵢ) to A(s_j).

    s_ij = f_θ(hᵢ, h_j, u_ij, ϕ_ij)
    p_ij = σ(s_ij)

Components:
- **hᵢ, h_j**: synapse embeddings
- **u_ij**: aggregated path evidence
- **ϕ_ij**: extra pair features (distance, coarse compartment flag, etc.)

### 1. Synapse embeddings

    hᵢ = g_θ(xᵢ)

xᵢ can include: synapse type/features, local image context, local substrate
neighbourhood, nearby synapse density, fragment attachment confidence.
No global structure needed — just the synapse and its local anchor.

### 2. Candidate pair set

Do not train over all n² pairs.  Build a sparse candidate set C ⊆ S × S using:

- **Local/topological**: pairs reachable within a relaxed radius in G; pairs
  within an approximate geodesic threshold; pairs with limited allowed gaps.
- **Long-range hard candidates**: pairs with high embedding similarity; pairs
  connected by thin-axon continuation hypotheses; hard negatives from current
  model errors.

|C| ≪ n².

### 3. Candidate path proposals

For each (i, j) ∈ C, generate K plausible paths P_ij = {P_ij^(1), …, P_ij^(K)}
connecting A(sᵢ) to A(s_j) in G (allowing controlled relaxation for broken
preseg):

- Shortest observed path
- Shortest relaxed path with gap penalties
- Top-K beam paths
- Bidirectional search with continuation priors

Fragments without synapses enter the problem *only* as elements of a
supporting path.

### 4. Path encoding

Simple baseline: hand-defined path feature vector ψ(P):

    [path_length, gap_count, total_gap_cost, min_edge_conf, mean_edge_conf,
     branch_count, alt_path_margin, supporting_synapse_density,
     endpoint_direction_agreement]

    u_ij^(k) = MLP(ψ(P_ij^(k)))

Richer encoder (later): small transformer, recurrent encoder, or path-GNN.

### 5. Path aggregation

- **Max pooling**: u_ij = max_k u_ij^(k)  — one strong path may suffice.
- **Attention pooling**: soft-weight paths by learned quality.
- **Top-2 margin**: include best score, second-best, and their gap.

### 6. Pair score

    s_ij = MLP([hᵢ, h_j, |hᵢ − h_j|, hᵢ ⊙ h_j, u_ij, ϕ_ij])
    p_ij = σ(s_ij)

### 7. BCE objective

    L_BCE = -Σ_{(i,j)∈C} w_ij [y_ij log p_ij + (1 − y_ij) log(1 − p_ij)]

Weights w_ij matter for:
- Class imbalance (more positives weight)
- Uncertain labels
- Long-range thin-axon emphasis
- Hard negatives

---

## Useful extensions (not baseline)

### Focal variant
Replace BCE with focal loss to down-weight easy negatives:

    L_focal = -Σ w_ij [y_ij (1−p_ij)^γ log p_ij + (1−y_ij) p_ij^γ log(1−p_ij)]

### Bridge / transitivity regulariser
For synapses i, k, j where k lies on a plausible support route:

    L_bridge = Σ max(0, p_ik · p_kj − p_ij − m)

Encourages indirect support to propagate.

---

## What labels should mean

Pick one consistent definition:

| Option | y_ij = 1 means … |
|--------|------------------|
| Path neighbourhood | sᵢ, s_j lie on the same local-to-midrange pathway region |
| Branch/subtree | path between them does not cross a major topological barrier |
| Bounded complexity | ∃ a within-neuron path satisfying complexity constraints |

The label must align with the downstream use; too broad → BCE learns something
less useful.

---

## Inference after training

Given p_ij:

- **Retrieval / ranking**: for a query synapse sᵢ, rank other synapses by p_ij.
- **Sparse affinity graph**: G_S = (S, E_S) with edges for high-probability pairs.
- **Global path recovery**: clustering, component extraction, maximum spanning
  forest, or Steiner-style subtree inference over synapses.

BCE gives the primitive score; downstream graph algorithms impose final
structure.

---

## Minimal first implementation

Inputs per candidate pair (i, j):
- synapse features xᵢ, x_j
- top 1–3 relaxed substrate paths
- summarized path features ψ(P)

Model:
    hᵢ = MLP(xᵢ),  h_j = MLP(x_j)
    u_ij^(k) = MLP(ψ(P_ij^(k)))
    u_ij = max_k u_ij^(k)
    p_ij = σ(MLP([hᵢ, h_j, |hᵢ−h_j|, hᵢ⊙h_j, u_ij]))

Loss: weighted BCE.

---

## Why this is better than fragment-centric learning

Fragment-centric supervision risks spending model capacity on:
- Fragment continuity with no synaptic consequence
- Detailed morphology unrelated to the downstream objective
- Generic neurite completion rather than synapse topology

Synapse-pair BCE keeps supervision aligned to the end goal.

**In one sentence:**
> Learn synapse-pair connectivity probabilities conditioned on a small set of
> candidate supporting paths through the imperfect substrate graph.

---

## Strong next-step upgrade

After the baseline, the most valuable upgrade is:

- Contextual synapse embeddings from a **GNN over the substrate neighbourhood**
  around each synapse
- Plus path summaries

Not a full morphology model — a local context encoder that improves the pair
head.
