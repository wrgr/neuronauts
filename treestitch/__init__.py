"""treestitch — domain-agnostic global tree stitching from fragments.

The global tree problem
-----------------------
You have a set of **fragments** — sub-trees of an unknown global forest.
Each fragment carries:
  - a tree skeleton (vertices, edges, radii)
  - an embedding vector (learned by FragmentEncoder)
  - endpoint positions (leaf vertices — the stitch handles)
  - a list of observations attached to it (points/events on the fragment)

The goal is to:
  1. Embed each fragment so that fragments from the same parent tree are
     close in embedding space.
  2. Build an observation graph: one node per observation, typed edges
     connecting observations on the same fragment (same-fragment, type 0),
     spatially nearby observations (spatial k-NN, type 1), and observations
     on endpoint-adjacent fragments (endpoint-adj, type 2).
  3. Train a PartitionGNN to cluster observations by their parent tree.
  4. Evaluate cluster quality (ARI) against ground-truth labels.

Neuro mapping
-------------
  fragment     ↔  skeleton segment (v117 seg root)
  observation  ↔  synapse (pre-side or post-side)
  parent tree  ↔  neuron (v1412 proofread root ID)

Other domains
-------------
  fragment     ↔  road / cable / cell-lineage branch
  observation  ↔  GPS ping / end-to-end signal / cell division event
  parent tree  ↔  route / cable run / lineage tree
"""

from treestitch.schemas import Fragment, ObservationGraph, ObjectHypothesis
from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
from treestitch.graph import build_observation_graph
from treestitch.partition import (
    PartitionGNN,
    train_partition,
    partition_observations,
    evaluate_partition,
    train_edge_partition,
    partition_observations_cc,
    merge_metrics,
)
from treestitch.pipeline import run_pipeline, optimize

__all__ = [
    "Fragment",
    "ObservationGraph",
    "ObjectHypothesis",
    "FragmentEncoder",
    "encode_fragments",
    "train_fragment_encoder",
    "build_observation_graph",
    "PartitionGNN",
    "train_partition",
    "partition_observations",
    "evaluate_partition",
    "train_edge_partition",
    "partition_observations_cc",
    "merge_metrics",
    "run_pipeline",
    "optimize",
]
