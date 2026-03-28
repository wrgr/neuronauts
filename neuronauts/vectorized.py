"""Vectorized agent stepping."""

from typing import Tuple

import numpy as np

from ._scipy_compat import cdist

from .agent import AgentConfig
from .helpers import safe_normalize


def run_agents_vectorized(
    volume_shape: np.ndarray,
    n_agents: int,
    synapse_pts: np.ndarray,
    membrane_field: np.ndarray,
    membrane_vectors: np.ndarray,
    exploration_field: np.ndarray,
    config: AgentConfig,
    rng: np.random.Generator,
    synapse_fraction: float = 0.5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = n_agents
    S = len(synapse_pts)
    T = config.max_steps + 1
    shape = volume_shape.astype(np.float32)
    shape_int = volume_shape.astype(np.int32)

    n_at_syn = int(N * synapse_fraction)
    n_rand = N - n_at_syn

    positions = np.empty((N, 3), dtype=np.float32)
    if S > 0 and n_at_syn > 0:
        idx = rng.choice(S, size=n_at_syn, replace=True)
        jitter = rng.uniform(
            -config.spawn_jitter_scale,
            config.spawn_jitter_scale,
            (n_at_syn, 3),
        ).astype(np.float32)
        positions[:n_at_syn] = synapse_pts[idx] + jitter
    positions[n_at_syn:] = rng.uniform(0, shape, (n_rand, 3)).astype(np.float32)
    positions = np.clip(positions, 0, shape - 1)

    velocities = rng.uniform(-0.5, 0.5, (N, 3)).astype(np.float32)
    alive = np.ones(N, dtype=bool)
    synapse_hits = np.zeros((N, S), dtype=bool)

    path_arr = np.zeros((N, T, 3), dtype=np.float32)
    path_arr[:, 0, :] = positions

    for step in range(config.max_steps):
        if not alive.any():
            break

        alive_idx = np.where(alive)[0]
        pos = positions[alive_idx]
        vel = velocities[alive_idx]
        pi = np.clip(pos.astype(np.int32), 0, shape_int - 1)

        mem_val = membrane_field[pi[:, 0], pi[:, 1], pi[:, 2]]
        mem_vec = membrane_vectors[pi[:, 0], pi[:, 1], pi[:, 2], :]
        scale = np.minimum(mem_val / (config.membrane_threshold + 1e-6), 1.0)
        repulsion = mem_vec * scale[:, None]

        vel_norm = safe_normalize(vel, axis=1)
        dot = np.einsum("ij,ij->i", vel_norm, mem_vec)[:, None]
        wall_follow = vel_norm - dot * mem_vec
        wall_follow = safe_normalize(wall_follow, axis=1)

        pix = np.clip(pi, 1, shape_int - 2)
        exp_grad = np.stack(
            [
                exploration_field[pix[:, 0] + 1, pix[:, 1], pix[:, 2]]
                - exploration_field[pix[:, 0] - 1, pix[:, 1], pix[:, 2]],
                exploration_field[pix[:, 0], pix[:, 1] + 1, pix[:, 2]]
                - exploration_field[pix[:, 0], pix[:, 1] - 1, pix[:, 2]],
                exploration_field[pix[:, 0], pix[:, 1], pix[:, 2] + 1]
                - exploration_field[pix[:, 0], pix[:, 1], pix[:, 2] - 1],
            ],
            axis=1,
        ).astype(np.float32) * 0.5
        exp_dir = safe_normalize(exp_grad, axis=1)

        if S > 0:
            dists = cdist(pos, synapse_pts)
            nearest = np.argmin(dists, axis=1)
            nd = dists[np.arange(len(pos)), nearest]
            syn_dir = safe_normalize(synapse_pts[nearest] - pos, axis=1)
            syn_weight = np.clip(config.synapse_capture_radius / (nd + 1e-8), 0, 1)[:, None]
        else:
            dists = None
            syn_dir = np.zeros((len(pos), 3), dtype=np.float32)
            syn_weight = np.zeros((len(pos), 1), dtype=np.float32)

        noise = safe_normalize(
            rng.standard_normal((len(pos), 3)).astype(np.float32), axis=1,
        )

        delta_v = (
            config.w_membrane_repulsion * repulsion
            + config.w_wall_follow * wall_follow
            + config.w_exploration * exp_dir
            + config.w_synapse_attraction * syn_dir * syn_weight
            + config.noise_scale * noise
        )
        new_vel = config.w_inertia * vel + delta_v
        speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
        new_vel = np.where(speed > config.max_speed, new_vel / speed * config.max_speed, new_vel)

        new_pos = pos + new_vel
        oob = (new_pos < 0) | (new_pos >= shape[None, :])
        new_pos = np.clip(new_pos, 0, shape[None, :] - 1)
        new_vel = np.where(oob, -new_vel * 0.5, new_vel)

        positions[alive_idx] = new_pos.astype(np.float32)
        velocities[alive_idx] = new_vel.astype(np.float32)
        path_arr[alive_idx, step + 1, :] = new_pos.astype(np.float32)

        new_pi = np.clip(new_pos.astype(np.int32), 0, shape_int - 1)
        decay = (
            config.exploration_decay
            + (config.exploration_decay_end - config.exploration_decay) * (step / max(1, config.max_steps - 1))
        )
        exploration_field[new_pi[:, 0], new_pi[:, 1], new_pi[:, 2]] *= 1.0 - decay

        if S > 0 and dists is not None:
            synapse_hits[alive_idx] |= dists < config.synapse_capture_radius

        if verbose and step % 200 == 0:
            claimed = synapse_hits.any(axis=0).sum()
            print(f"  Step {step:4d}: {alive.sum()} alive, {claimed}/{S} synapses hit")

    return path_arr, synapse_hits, alive
