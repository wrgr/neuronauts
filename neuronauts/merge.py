"""Active graph datatypes for neuron assembly.

``MergedNeuron`` and ``ConnectivityGraph`` are the connectivity-graph types
consumed across the active pipeline (``cell_graph``, ``skeleton_graph``,
``assembly``, ``line_graph``). They are deliberately dependency-light: importing
this module must NOT pull in the legacy agent-simulation stack. The agent-trace
merge step that used to live here now lives in ``agent_merge.py`` (v1); see
``docs/stage_ownership.md``.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


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
