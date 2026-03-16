# neuronauts: Agent-Based Connectome Recovery for Short Iterative Research Loops

## Abstract

`neuronauts` is a compact Python research package for recovering neuron connectivity from volumetric electron microscopy using autonomous agents. The system is designed around a practical objective: enable short, repeated local improvement cycles in the style of `karpathy/autoresearch`, where each loop edits a narrow experiment surface, runs a reproducible benchmark, and optimizes a single scalar metric. In `neuronauts`, that metric is line-graph F1: a comparison between ground-truth synapse co-incidence relationships and the relationships inferred by agent trajectories, path merging, and polarity assignment. The package emphasizes deterministic preprocessing, explicit graph construction, and transparent failure analysis over large framework overhead.

## 1. Motivation

Connectomics pipelines usually combine several difficult subproblems: boundary detection, segmentation, proofreading, synapse assignment, and graph reconstruction. `neuronauts` narrows the problem to a specific research question: how far can a lightweight agent system recover connectivity structure directly from local fields and synapse waypoints before heavier segmentation machinery is needed?

This framing is useful for at least three reasons. First, it isolates policy and merge behavior from learned perception systems by using fixed field computations. Second, it exposes a simple optimization target that can drive fast iteration. Third, it fits the `autoresearch` paradigm well: a human or coding agent can repeatedly modify one experiment file, run a bounded benchmark, and retain only changes that improve the score or resolve a known error mode.

## 2. System Overview

The package contains six core components.

`neuronauts.fields` computes membrane and exploration fields from the volume. The membrane signal is a Sobel-gradient magnitude over the raw EM volume, and membrane vectors point away from strong boundaries.

`neuronauts.vectorized` steps all agents simultaneously with NumPy and SciPy. Each agent combines membrane repulsion, wall-following, exploration pressure, synapse attraction, inertia, and noise into a velocity update. Positions and synapse hits are recorded in preallocated arrays so runs remain cheap enough for repeated local loops.

`neuronauts.run` is the primary experiment entrypoint. It assembles fields, launches the vectorized simulation, merges agent paths into neuron hypotheses, builds a directed connectivity graph, and evaluates the result.

`neuronauts.line_graph` defines the benchmark metric. Ground truth is derived from synapse root IDs: two synapses share a line-graph edge when they share a presynaptic neuron or a postsynaptic neuron. Estimated edges are derived from merged neuron assignments. Precision, recall, and F1 are then computed over those two graphs.

`neuronauts.fetch` supports future transfer to MICrONS-like data through `cloud-volume` and `CAVEclient`.

`tests/test_run.py` holds a regression test for the main merge bug found so far.

## 3. The Key Graph Construction Fix

The most important structural correction in the current version concerns how agent hits are merged into neuron hypotheses. A naive implementation can incorrectly combine all synapse-waypoint evidence into a single merged neuron whenever agent paths overlap spatially. That is wrong for line-graph recovery because presynaptic and postsynaptic evidence play different roles. If both are pooled before polarity is assigned, the system can hallucinate synapse co-incidence relationships that do not exist in the ground truth.

`neuronauts` now handles this explicitly. In `neuronauts.run`, synapse hits are split into two matrices: hits on presynaptic sites and hits on postsynaptic sites. Agents are merged separately inside each role-specific set. The result is two families of merged neuron candidates, one for pre roles and one for post roles. Only after those role-specific neuron groups are formed does the system connect them across synapse edges using nearest-path assignment. This mirrors the logic used by the ground-truth line graph and eliminates the oracle-revealed false-positive class where pre and post evidence from unrelated neurons collapse into one merged node.

This design is important beyond the immediate bug fix. It states a broader principle for agentic connectomics systems: spatial overlap alone is not enough to define biological identity when the evidence has a directional role.

## 4. Why the Package Fits `autoresearch`

The `karpathy/autoresearch` pattern works best when a project has four properties:

1. A narrow edit surface.
2. A clear executable benchmark.
3. A scalar metric to optimize.
4. A short loop time.

`neuronauts` is organized around those constraints. The main experiment surface is `neuronauts/run.py`, especially the configuration block that controls sensor weights, merge radius, capture radius, and swarm size. The benchmark is `python -m neuronauts.run`. The scalar metric is `val_f1`. The supporting script `scripts/iterative_loop.py` repeats runs for a fixed time budget, which makes five-minute local loops straightforward.

This structure is deliberate. It keeps the package small enough for a coding agent to understand end to end, while still exposing enough genuine algorithmic structure to support meaningful search rather than parameter thrashing.

## 5. Near-Term Research Plan

The immediate work should remain on synthetic validation. That means:

- tightening regression coverage around merge and polarity logic,
- measuring score variance across seeds,
- separating “parameter search” changes from “algorithmic” changes,
- logging best-of-budget results for fixed five-minute windows, and
- only then transferring to MICrONS fetch paths.

The likely next bottlenecks are under-merging, poor synapse coverage, and sensitivity to merge radius. A disciplined five-minute loop should therefore record not just F1, but also unresolved synapse count and the number of merged neurons and edges produced in each run.

## References

1. Karpathy, A. `autoresearch`. GitHub repository. <https://github.com/karpathy/autoresearch>
2. MICrONS Consortium et al. Functional connectomics spanning multiple areas of mouse visual cortex. *Nature* 2021. <https://www.nature.com/articles/s41586-021-03778-x>
3. Silversmith, W. `cloud-volume`. GitHub repository. <https://github.com/seung-lab/cloud-volume>
4. CAVEconnectome. `CAVEclient`. GitHub repository. <https://github.com/CAVEconnectome/CAVEclient>
5. Bae, J. A. et al. Digital museum of retinal ganglion cells with dense anatomy and physiology. *Cell* 2024. <https://www.cell.com/cell/fulltext/S0092-8674(24)00308-4>
