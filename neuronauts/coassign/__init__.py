"""Synapse co-assignment pipeline.

The problem
-----------
Synapses are invariant physical nodes — they exist independent of any
segmentation version. The task is to partition them into neuron cliques:
every synapse on the same neuron belongs to the same cluster.

The evidence
------------
- v117 segment IDs: expensive, mostly-right local continuity signal.
  Two synapses sharing a segment very likely share a neuron.
- DNA embeddings (SkeletonGNN): morphological identity of the segment
  each synapse belongs to. Similar DNA → likely same neuron.
- Spatial proximity: weak positional prior.

The pipeline
------------
1. Build a synapse graph (same-seg edges + spatial k-NN edges).
2. Train SynapseCoassigner: GNN → per-synapse embeddings → P(same neuron) per edge.
3. Partition via correlation clustering.
4. Generate K materializations — ranked candidate partitions.
   The true partition should appear in the top-K (coverage@K).

Nothing is hardcoded. Position normalisation and edge-type weighting
are learned from data via LayerNorm and attention.

Quick start
-----------
    from neuronauts.coassign import (
        build_synapse_graph,
        SynapseCoassigner,
        train,
        materializations,
        pairwise_precision_recall,
        coverage_at_k,
    )
"""

from .graph import SynapseGraph, build_synapse_graph
from .cluster import (
    calibrate_threshold,
    coverage_at_k,
    greedy_cluster,
    materializations,
    pairwise_precision_recall,
)

# SynapseCoassigner and train require torch. Import them eagerly when torch is
# present; leave them absent (ImportError at use time) otherwise so that
# graph/cluster helpers stay importable in torch-free environments.
try:
    from .model import SynapseCoassigner
    from .train import train
except ImportError:
    pass

__all__ = [
    "SynapseGraph",
    "build_synapse_graph",
    "SynapseCoassigner",
    "train",
    "greedy_cluster",
    "materializations",
    "pairwise_precision_recall",
    "coverage_at_k",
    "calibrate_threshold",
]
