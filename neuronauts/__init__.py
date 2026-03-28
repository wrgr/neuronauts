"""neuronauts — end-to-end connectome inference from electron microscopy data."""

from .agent import Agent, AgentConfig
from .dijkstra import BridgeGraph, BridgePath
from .helpers import UnionFind, pairwise_edges, safe_normalize
from .line_graph import LineGraphMetrics, evaluate, evaluate_sampled
from .merge import ConnectivityGraph, MergedNeuron, merge_agents

__all__ = [
    # agents
    "Agent",
    "AgentConfig",
    # graph & merge
    "ConnectivityGraph",
    "MergedNeuron",
    "merge_agents",
    # bridge search
    "BridgeGraph",
    "BridgePath",
    # evaluation
    "LineGraphMetrics",
    "evaluate",
    "evaluate_sampled",
    # helpers
    "UnionFind",
    "safe_normalize",
    "pairwise_edges",
]
