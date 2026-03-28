"""Neuron merge and graph datatypes."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ._scipy_compat import cKDTree

from .agent import Agent
from .helpers import UnionFind


@dataclass
class MergedNeuron:
    neuron_id: int
    agent_ids: List[int]
    path_points: np.ndarray
    synapse_indices: List[int]
    role: str = "mixed"


@dataclass
class ConnectivityGraph:
    neurons: Dict[int, MergedNeuron]
    edges: List[Tuple[int, int, int]]
    unresolved_synapse_indices: List[int]
    metadata: dict[str, object] = field(default_factory=dict)


def merge_agents(
    agents: List[Agent],
    merge_radius: float = 5.0,
    min_path_length: int = 5,
) -> Dict[int, MergedNeuron]:
    valid = [a for a in agents if len(a.path) >= min_path_length]
    if not valid:
        return {}

    n = len(valid)
    uf = UnionFind(n)

    all_points = []
    point_to_agent = []
    for i, agent in enumerate(valid):
        pts = np.array(agent.path)
        all_points.append(pts)
        point_to_agent.extend([i] * len(pts))

    all_points_arr = np.vstack(all_points)
    point_to_agent_arr = np.array(point_to_agent)
    tree = cKDTree(all_points_arr)

    for i, agent in enumerate(valid):
        for pt_neighbors in tree.query_ball_point(np.array(agent.path), r=merge_radius):
            for nb_idx in pt_neighbors:
                j = point_to_agent_arr[nb_idx]
                if j != i:
                    uf.union(i, j)

    neurons = {}
    for neuron_id, agent_indices in enumerate(uf.groups()):
        member_agents = [valid[i] for i in agent_indices]
        all_pts = np.vstack([np.array(a.path) for a in member_agents])
        all_synapses = []
        for a in member_agents:
            all_synapses.extend(a.visited_synapses)
        neurons[neuron_id] = MergedNeuron(
            neuron_id=neuron_id,
            agent_ids=[a.agent_id for a in member_agents],
            path_points=all_pts,
            synapse_indices=sorted(set(all_synapses)),
        )

    return neurons
