"""Agent-trace merging — part of the legacy v1 agent/membrane pipeline.

``merge_agents`` collapses overlapping agent traces (from the 700-walker
simulation) into :class:`~neuronauts.merge.MergedNeuron` groups via a spatial
union-find. It is **not** used by the active assembly pipeline; it is retained
for the v1 ``run.py`` orchestrator and its tests. See
``docs/stage_ownership.md`` for the legacy quarantine plan.

It lives apart from ``merge.py`` — which now holds only the active graph
datatypes ``MergedNeuron`` / ``ConnectivityGraph`` — so that importing those
types does not transitively pull in the agent-simulation stack (``agent.py``).
"""

from typing import Dict, List

import numpy as np

from .._scipy_compat import cKDTree
from .agent import Agent
from ..helpers import UnionFind
from ..merge import MergedNeuron


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
