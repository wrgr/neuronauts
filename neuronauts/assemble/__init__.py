"""Phase 2: global synapse graph + GNN assembly, and fragment-proximity graph."""

from .global_synapse_graph import GlobalSynapseGraph, build_global_synapse_graph
from .synapse_gnn import assemble_neurons, run_global_gnn, train_global_gnn
from .half_synapse_graph import HalfSynapseGraph, build_half_synapse_graph
from .partition_gnn import (
    HalfSynapseGNN,
    evaluate_partition_ari,
    partition_half_synapses,
    train_partition_gnn,
)
from .fragment_graph import (
    assemble_fragments,
    build_fragment_graph,
    score_edge,
)

__all__ = [
    # Global synapse graph + GNN assembly
    "GlobalSynapseGraph",
    "build_global_synapse_graph",
    "train_global_gnn",
    "run_global_gnn",
    "assemble_neurons",
    # Half-synapse graph + partition GNN (Phase 2.1)
    "HalfSynapseGraph",
    "build_half_synapse_graph",
    "HalfSynapseGNN",
    "evaluate_partition_ari",
    "partition_half_synapses",
    "train_partition_gnn",
    # Fragment-proximity graph (endpoint-based stitching)
    "assemble_fragments",
    "build_fragment_graph",
    "score_edge",
]
