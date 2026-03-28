"""Neuron merge and graph datatypes."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    class cKDTree:  # type: ignore[override]
        """Minimal fallback with the subset of scipy.spatial.cKDTree used here."""

        def __init__(self, data: np.ndarray) -> None:
            self.data = np.asarray(data, dtype=np.float32)

        def query_ball_point(self, points: np.ndarray, r: float):
            pts = np.asarray(points, dtype=np.float32)
            scalar_input = pts.ndim == 1
            if scalar_input:
                pts = pts[None, :]

            result = []
            for point in pts:
                dist = np.linalg.norm(self.data - point, axis=1)
                result.append(np.flatnonzero(dist <= r).tolist())
            return result[0] if scalar_input else result

        def query_pairs(self, r: float, output_type: str = "set"):
            pairs = []
            for i in range(len(self.data)):
                dist = np.linalg.norm(self.data[i + 1 :] - self.data[i], axis=1)
                neighbors = np.flatnonzero(dist <= r)
                for offset in neighbors.tolist():
                    pairs.append((i, i + 1 + offset))
            if output_type == "ndarray":
                return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
            return {tuple(pair) for pair in pairs}

        def query(self, point: np.ndarray):
            point_arr = np.asarray(point, dtype=np.float32)
            dist = np.linalg.norm(self.data - point_arr, axis=1)
            idx = int(np.argmin(dist))
            return float(dist[idx]), idx

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
