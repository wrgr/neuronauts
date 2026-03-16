"""Core agent datatypes."""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class AgentConfig:
    w_membrane_repulsion: float
    w_wall_follow: float
    w_exploration: float
    w_synapse_attraction: float
    w_inertia: float
    max_speed: float
    noise_scale: float
    membrane_threshold: float
    exploration_decay: float
    exploration_decay_end: float
    exploration_radius: int
    synapse_capture_radius: float
    spawn_jitter_scale: float
    max_steps: int
    respawn_on_boundary: bool
    kill_on_membrane: bool


@dataclass
class Agent:
    agent_id: int
    path: List[np.ndarray] = field(default_factory=list)
    visited_synapses: List[int] = field(default_factory=list)
