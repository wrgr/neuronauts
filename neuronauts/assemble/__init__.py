"""Phase 2: global synapse graph + GNN assembly."""

from .global_synapse_graph import GlobalSynapseGraph, build_global_synapse_graph
from .synapse_gnn import assemble_neurons, run_global_gnn, train_global_gnn
from .half_synapse_graph import HalfSynapseGraph, build_half_synapse_graph
from .partition_gnn import (
    HalfSynapseGNN,
    evaluate_partition_ari,
    partition_half_synapses,
    train_partition_gnn,
)

__all__ = [
    "GlobalSynapseGraph",
    "build_global_synapse_graph",
    "train_global_gnn",
    "run_global_gnn",
    "assemble_neurons",
    "HalfSynapseGraph",
    "build_half_synapse_graph",
    "HalfSynapseGNN",
    "evaluate_partition_ari",
    "partition_half_synapses",
    "train_partition_gnn",
]
