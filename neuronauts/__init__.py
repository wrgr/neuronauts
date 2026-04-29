"""neuronauts — end-to-end connectome inference from electron microscopy data."""

from .agent import Agent, AgentConfig
from .assembly import CandidateMerge, logit_to_probability, probability_to_log_odds
from .cell_graph import (
    CellGNN,
    CellGNNConfig,
    EDGE_FEATURE_NAMES,
    SynapseEdge,
    SynapseGraph,
    build_synapse_chain_paths,
    build_synapse_graph,
    cell_graph_train_step,
    cell_gnn_assembly,
    infer_cells,
    infer_cells_two_pass,
    load_seg_score_cache,
    load_self_skeleton_archive,
    load_skeleton_path_cache,
    partition_from_embeddings,
    precompute_seg_scores_fast,
    precompute_self_skeletons_for_cache,
    precompute_skeleton_paths_for_cache,
    save_seg_score_cache,
    save_skeleton_path_cache,
    subdivide_synapse_graph,
    train_cell_gnn,
)
from .dijkstra import BridgeGraph, BridgePath
from .helpers import UnionFind, pairwise_edges, safe_normalize
from .line_graph import LineGraphMetrics, evaluate, evaluate_sampled
from .merge import ConnectivityGraph, MergedNeuron, merge_agents
from .path_edge_encoder import PathEdgeEncoder, pad_path_sequences

__all__ = [
    # agents
    "Agent",
    "AgentConfig",
    # graph & merge
    "ConnectivityGraph",
    "MergedNeuron",
    "merge_agents",
    # CellGNN architecture & training
    "CellGNN",
    "CellGNNConfig",
    "EDGE_FEATURE_NAMES",
    "SynapseEdge",
    "SynapseGraph",
    "build_synapse_graph",
    "cell_graph_train_step",
    "cell_gnn_assembly",
    "build_synapse_chain_paths",
    "subdivide_synapse_graph",
    "infer_cells",
    "infer_cells_two_pass",
    "partition_from_embeddings",
    "train_cell_gnn",
    # Path-edge encoder (Option 2)
    "PathEdgeEncoder",
    "pad_path_sequences",
    # Edge-feature caches (precompute)
    "load_seg_score_cache",
    "load_self_skeleton_archive",
    "load_skeleton_path_cache",
    "precompute_seg_scores_fast",
    "precompute_self_skeletons_for_cache",
    "precompute_skeleton_paths_for_cache",
    "save_seg_score_cache",
    "save_skeleton_path_cache",
    # bridge search
    "BridgeGraph",
    "BridgePath",
    # evaluation
    "LineGraphMetrics",
    "evaluate",
    "evaluate_sampled",
    # probability
    "CandidateMerge",
    "logit_to_probability",
    "probability_to_log_odds",
    # helpers
    "UnionFind",
    "safe_normalize",
    "pairwise_edges",
]
