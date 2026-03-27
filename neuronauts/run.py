"""Main experiment runner."""

import argparse
import time
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np

from .assembly_dataset import hypothesis_features
from .agent import AgentConfig
from .assembly import CandidateMerge, beam_search_merge_groups, gat_refine_connectivity
from .dijkstra import BridgeGraph
from .fields import compute_exploration_field, compute_membrane_field, compute_membrane_vectors
from .membrane_unet import load_model as _load_membrane_model, predict_membranes as _predict_membranes
from .fetch import (
    RealBoxSpec,
    SyntheticBenchmarkConfig,
    fetch_synapses,
    fetch_volume,
    load_cached_membrane,
    make_test_volume,
)
from .grammar import DEFAULT_PATH_FEATURE_MODE, PATH_ISO, featurize_path_points
from .line_graph import LineGraphMetrics, evaluate
from .merge import ConnectivityGraph, MergedNeuron, cKDTree
from .vectorized import run_agents_vectorized

# Lazy imports for CellGNN (only needed when --cell-gnn-checkpoint is set)
CELL_GNN_CHECKPOINT = None

# ============================================================
# EXPERIMENT CONFIG -- autoresearch edits this block
# ============================================================

# Tuned baseline after local sweep:
# fixed_validation F1 ~= 0.558
# random-batch mean F1 ~= 0.468 over short confirmation loops
AGENT_CONFIG = AgentConfig(
    w_membrane_repulsion=2.0,
    w_wall_follow=0.3,
    w_exploration=1.0,
    w_synapse_attraction=1.0,
    w_inertia=0.6,
    max_speed=1.5,
    noise_scale=0.1,
    membrane_threshold=0.4,
    exploration_decay=0.15,
    exploration_decay_end=0.05,
    exploration_radius=2,
    synapse_capture_radius=1.30,
    spawn_jitter_scale=1.0,
    max_steps=450,
    respawn_on_boundary=True,
    kill_on_membrane=False,
)

# Swarm / merge controls
N_AGENTS = 700
SYNAPSE_SPAWN_FRACTION = 0.25
MERGE_RADIUS = 3.0
MERGE_OVERLAP_THRESHOLD = 0.65
ROLE_MERGE_MIN_SHARED_HITS = 1
MAX_SYNAPSES_PER_NEURON = 32
MIN_PATH_LENGTH = 5
WAYPOINTS_PER_AGENT = 20

# Field / assignment controls
MEMBRANE_SIGMA = 1.0
MEMBRANE_VECTOR_SIGMA = 1.5
POLARITY_CAPTURE_R = 3.5
PRE_POST_OWNER_TOPK = 3
OWNER_MARGIN = 0.0

# Synthetic benchmark policy
BENCHMARK_CONFIG = SyntheticBenchmarkConfig(
    shape=(96, 96, 96),
    n_synapses=30,
    membrane_planes=10,
    min_neuron_groups=6,
    max_neuron_groups=15,
    anchor_margin=12,
    pre_cluster_std=4.0,
    post_cluster_std=4.0,
)
BENCHMARK_CASES = 5
BENCHMARK_MODE = "random"

# Real-data benchmark policy
REAL_MIN_SYNAPSES = 50
REAL_BOXES = [
    RealBoxSpec(center_nm=(1_153_592, 793_592, 655_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(733_592, 513_592, 595_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(1_213_592, 333_592, 975_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(473_592, 433_592, 1_095_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(893_592, 973_592, 915_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(1_333_592, 633_592, 975_640), side_um=6.0, mip=2),
    RealBoxSpec(center_nm=(773_592, 533_592, 795_640), side_um=6.0, mip=2),
]
REAL_BOXES_PER_EVAL = 3
MEMBRANE_CACHE_DIR = "cache/membranes"
SHARED_GRAMMAR_CHECKPOINT = None
ASSEMBLY_RERANKER_CHECKPOINT = None
LEARNED_MERGE_SCORE_THRESHOLD = 0.0
BEAM_WIDTH = 1
BEAM_MAX_CANDIDATES = 24
ATOMICITY_SCORE_WEIGHT = 0.25
RERANKER_THRESHOLDS = "-0.5,0.0,0.5"
RERANKER_BEAM_WIDTHS = "1,2,4"

# ============================================================
# END CONFIG
# ============================================================


@dataclass(frozen=True)
class HeuristicConfig:
    """Controls whether spatial thresholds act as hard decisions or candidate generators.

    Philosophy
    ----------
    In the v1 heuristic-only pipeline, spatial thresholds were *decisions*:
    two agents had to overlap within ``merge_radius`` **and** share ≥
    ``merge_overlap_threshold`` of their synapse hits before being merged.
    Synapse assignments only succeeded when the nearest neuron was within
    ``polarity_capture_r`` voxels.

    In the v2 learned pipeline (PR 4 + 5) these same thresholds become
    *candidate generators*:

    - ``merge_radius`` still restricts which agent-pairs the KD-tree considers
      (a reasonable proximity pre-filter), but the final merge/reject call is
      made by ``learned_merge_score_fn`` or the downstream GAT — not by the
      overlap fraction.
    - ``polarity_capture_r`` is relaxed to ``inf`` so every plausible
      (pre-neuron, post-neuron) pair becomes a candidate edge for the GAT to
      score.  Hard geometric pruning happens **after** learning, not before.
    - ``max_synapses_per_neuron`` is raised to a large limit; again, the
      learned scorer performs the final culling.

    Use ``HeuristicConfig.legacy()`` to reproduce pre-PR5 behaviour exactly
    (useful for ablation runs and regression tests).  Use
    ``HeuristicConfig.learned()`` when a grammar / GAT checkpoint is present.

    Attributes
    ----------
    merge_radius:
        KD-tree radius for generating candidate agent pairs. Kept in both
        modes as a fast spatial pre-filter.
    merge_overlap_threshold:
        Minimum shared-hit fraction for heuristic merging.  Bypassed in
        learned mode.
    role_merge_min_shared_hits:
        Minimum number of shared synapse hits for any pair to be considered.
        Kept in both modes as a data-quality guard.
    polarity_capture_r:
        Maximum voxel distance for synapse-to-neuron assignment.  Set to
        ``inf`` in learned mode so all plausible pairs reach the GAT.
    max_synapses_per_neuron:
        Hard cap on synapse count per neuron.  Raised to a large limit in
        learned mode.
    use_learned_decisions:
        When ``True`` the overlap-threshold check is skipped; all pairs
        within ``merge_radius`` that satisfy ``role_merge_min_shared_hits``
        are forwarded to the learned scorer (or kept as GAT candidate edges).
    """

    merge_radius: float = MERGE_RADIUS
    merge_overlap_threshold: float = MERGE_OVERLAP_THRESHOLD
    role_merge_min_shared_hits: int = ROLE_MERGE_MIN_SHARED_HITS
    polarity_capture_r: float = POLARITY_CAPTURE_R
    max_synapses_per_neuron: int = MAX_SYNAPSES_PER_NEURON
    use_learned_decisions: bool = False

    @classmethod
    def legacy(cls) -> "HeuristicConfig":
        """Reproduce the original pre-PR5 heuristic-only behaviour."""
        return cls(use_learned_decisions=False)

    @classmethod
    def learned(cls) -> "HeuristicConfig":
        """Permissive candidate-generation mode for learned grammar pipelines.

        Thresholds become wide-open so the path encoder, bridge head, and
        GAT make all real decisions.  Only the KD-tree radius and the
        minimum-shared-hits data-quality guard are retained.
        """
        return cls(
            merge_overlap_threshold=0.0,
            polarity_capture_r=float("inf"),
            max_synapses_per_neuron=1024,
            use_learned_decisions=True,
        )


def _valid_agent_indices(path_arr: np.ndarray) -> np.ndarray:
    if path_arr.shape[1] <= MIN_PATH_LENGTH:
        return np.array([], dtype=np.int32)
    valid_mask = (path_arr[:, MIN_PATH_LENGTH, :] != 0).any(axis=1)
    return np.where(valid_mask)[0].astype(np.int32)


def _agent_points(path_arr: np.ndarray, agent_idx: int) -> np.ndarray:
    path = path_arr[agent_idx]
    return path[np.any(path != 0, axis=1)]


def _subsample_points(path: np.ndarray) -> np.ndarray:
    if len(path) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    step_idx = np.linspace(0, len(path) - 1, min(WAYPOINTS_PER_AGENT, len(path)), dtype=int)
    return path[step_idx].astype(np.float32)


def _path_sequence_from_points(
    points: np.ndarray,
    *,
    path_feature_mode: str = DEFAULT_PATH_FEATURE_MODE,
) -> np.ndarray:
    """Compute isotropically-scaled raw step vectors from path points."""
    return featurize_path_points(points, mode=path_feature_mode, iso_scale=PATH_ISO)


@lru_cache(maxsize=4)
def _load_shared_merge_score_fn(checkpoint_path: str):
    from .shared_grammar_model import load_shared_grammar_model

    import torch

    model = load_shared_grammar_model(checkpoint_path)
    model.eval()
    path_feature_mode = getattr(model, "path_feature_mode", DEFAULT_PATH_FEATURE_MODE)

    def score_fn(left_sequence: np.ndarray, right_sequence: np.ndarray) -> float:
        left = torch.from_numpy(left_sequence[None, ...]).float()
        right = torch.from_numpy(right_sequence[None, ...]).float()
        left_mask = torch.zeros((1, left.shape[1]), dtype=torch.bool)
        right_mask = torch.zeros((1, right.shape[1]), dtype=torch.bool)
        with torch.no_grad():
            logits = model.score_merge(left, left_mask, right, right_mask)
        return float(logits.squeeze().cpu())

    score_fn.path_feature_mode = path_feature_mode
    return score_fn


@lru_cache(maxsize=4)
def _load_shared_atomicity_score_fn(checkpoint_path: str):
    from .shared_grammar_model import load_shared_grammar_model
    from .training_batches import pad_nested_path_sequences

    import torch

    model = load_shared_grammar_model(checkpoint_path)
    model.eval()

    def score_fn(branch_sequences: tuple[np.ndarray, ...]) -> float:
        feature_dim = int(branch_sequences[0].shape[1]) if branch_sequences else 3
        nested = pad_nested_path_sequences(
            [list(branch_sequences)],
            max_items=len(branch_sequences),
            feature_dim=feature_dim,
        )
        branch_x = torch.from_numpy(nested.x).float()
        branch_sequence_mask = torch.from_numpy(nested.sequence_mask)
        branch_mask = torch.from_numpy(nested.item_mask)
        with torch.no_grad():
            logits = model.score_atomicity(branch_x, branch_sequence_mask, branch_mask)
        return float(logits.squeeze().cpu())

    score_fn.path_feature_mode = getattr(model, "path_feature_mode", DEFAULT_PATH_FEATURE_MODE)
    return score_fn


@lru_cache(maxsize=4)
def _load_assembly_reranker(checkpoint_path: str):
    from .hypothesis_reranker import load_linear_reranker

    return load_linear_reranker(checkpoint_path)


def _scaffold_union_from_seg_ids(
    role_agent_ids: np.ndarray,
    role_hits: np.ndarray,
    role_seg_ids: np.ndarray | None,
    parent: dict[int, int],
) -> None:
    """Pre-merge agents whose hit synapses all belong to the same scaffold seg_id.

    When CAVE segment IDs are available, agents that only ever hit synapses
    from a single segment are very likely part of the same physical neurite
    fragment.  Pre-unioning them before the more expensive geometry- or
    grammar-based passes collapses the search space and prevents the learned
    scorer from needing to compare pairs that should trivially merge.

    An agent is assigned to a segment if:
    - it hit at least one synapse, AND
    - every synapse it hit has the same non-zero seg_id.

    Agents that hit synapses across multiple seg_ids are left to the
    downstream geometry / grammar pass.

    Parameters
    ----------
    role_agent_ids:
        Integer array of agent indices that have at least one hit.
    role_hits:
        Bool array ``[n_agents, n_synapses]``.
    role_seg_ids:
        Int64 array ``[n_synapses]`` of segment IDs, or ``None`` if not
        available (in which case this function is a no-op).
    parent:
        Union-find parent dict, mutated in place.
    """
    if role_seg_ids is None:
        return

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Map seg_id → first agent that is purely within that segment.
    seg_representative: dict[int, int] = {}

    for agent_id in role_agent_ids.tolist():
        hit_indices = np.flatnonzero(role_hits[agent_id])
        if len(hit_indices) == 0:
            continue
        agent_seg_ids = role_seg_ids[hit_indices]
        unique_segs = np.unique(agent_seg_ids)
        if len(unique_segs) != 1 or int(unique_segs[0]) == 0:
            # Agent spans multiple segments or has an unknown segment — skip.
            continue
        seg_id = int(unique_segs[0])
        if seg_id in seg_representative:
            union(agent_id, seg_representative[seg_id])
        else:
            seg_representative[seg_id] = agent_id


def _merge_role_groups(
    path_arr: np.ndarray,
    role_hits: np.ndarray,
    role_name: str,
    next_neuron_id: int,
    learned_merge_score_fn=None,
    learned_merge_score_threshold: float = LEARNED_MERGE_SCORE_THRESHOLD,
    atomicity_score_fn=None,
    beam_width: int = BEAM_WIDTH,
    beam_max_candidates: int = BEAM_MAX_CANDIDATES,
    atomicity_score_weight: float = ATOMICITY_SCORE_WEIGHT,
    role_seg_ids: np.ndarray | None = None,
    heuristic_config: "HeuristicConfig | None" = None,
) -> tuple[dict[int, MergedNeuron], dict[int, list[int]], dict[int, cKDTree], int]:
    """Merge agent paths into neuron groups, optionally seeded by scaffold seg_ids.

    When ``role_seg_ids`` is provided the union-find structure is pre-seeded
    by ``_scaffold_union_from_seg_ids`` before the geometry- and grammar-based
    passes run.  This means agents that trivially belong to the same CAVE
    segment are merged for free, and the remaining expensive pairwise scoring
    only considers cross-segment candidates — significantly reducing the
    search space in scaffold-aware mode.

    In ``HeuristicConfig.learned()`` mode (PR 5) the overlap-threshold check
    is bypassed: all candidate pairs within ``merge_radius`` that satisfy the
    minimum-shared-hits data-quality guard are forwarded to the learned scorer.
    When no scorer is present in learned mode, pairs are unioned optimistically
    and the GAT refinement step (PR 4) performs the real pruning.
    """
    hcfg = heuristic_config or HeuristicConfig.legacy()
    path_feature_mode = (
        getattr(learned_merge_score_fn, "path_feature_mode", None)
        or getattr(atomicity_score_fn, "path_feature_mode", None)
        or DEFAULT_PATH_FEATURE_MODE
    )
    role_agent_ids = np.where(role_hits.any(axis=1))[0].astype(np.int32)
    if len(role_agent_ids) == 0:
        return {}, {}, {}, next_neuron_id

    parent = {int(agent_id): int(agent_id) for agent_id in role_agent_ids.tolist()}

    # --- Scaffold pre-initialization (PR 3) ---
    _scaffold_union_from_seg_ids(role_agent_ids, role_hits, role_seg_ids, parent)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    sub_pts_list = []
    sub_labels = []
    for agent_id in role_agent_ids.tolist():
        pts = _subsample_points(_agent_points(path_arr, agent_id))
        if len(pts) == 0:
            continue
        sub_pts_list.append(pts)
        sub_labels.extend([agent_id] * len(pts))

    if not sub_pts_list:
        return {}, {}, {}, next_neuron_id

    sub_pts = np.vstack(sub_pts_list)
    sub_labels_arr = np.array(sub_labels, dtype=np.int32)
    # MERGE_RADIUS is always a candidate generator; acceptance is decided below.
    pairs = cKDTree(sub_pts).query_pairs(r=hcfg.merge_radius, output_type="ndarray")
    candidate_merges: list[CandidateMerge] = []
    seen_pairs: set[tuple[int, int]] = set()
    agent_sequences: dict[int, np.ndarray] = {}

    for a, b in pairs:
        agent_a = int(sub_labels_arr[a])
        agent_b = int(sub_labels_arr[b])
        pair_key = (min(agent_a, agent_b), max(agent_a, agent_b))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        hits_a = role_hits[agent_a]
        hits_b = role_hits[agent_b]
        shared_count = int(np.count_nonzero(hits_a & hits_b))
        # Data-quality guard: kept in both legacy and learned modes.
        if shared_count < hcfg.role_merge_min_shared_hits:
            continue
        if learned_merge_score_fn is not None:
            seq_a = agent_sequences.setdefault(
                agent_a,
                _path_sequence_from_points(_agent_points(path_arr, agent_a), path_feature_mode=path_feature_mode),
            )
            seq_b = agent_sequences.setdefault(
                agent_b,
                _path_sequence_from_points(_agent_points(path_arr, agent_b), path_feature_mode=path_feature_mode),
            )
            if len(seq_a) == 0 or len(seq_b) == 0:
                continue
            score = float(learned_merge_score_fn(seq_a, seq_b))
            candidate_merges.append(CandidateMerge(left_agent=agent_a, right_agent=agent_b, score=score))
        elif hcfg.use_learned_decisions:
            # Learned mode but no scorer yet: union optimistically so downstream
            # GAT refinement (PR 4) can make the real acceptance/rejection call.
            union(agent_a, agent_b)
        else:
            # Legacy mode: overlap fraction is the decision criterion.
            overlap = shared_count / max(1, min(int(np.count_nonzero(hits_a)), int(np.count_nonzero(hits_b))))
            if overlap >= hcfg.merge_overlap_threshold:
                union(agent_a, agent_b)

    if learned_merge_score_fn is not None:
        candidate_merges = [candidate for candidate in candidate_merges if candidate.score >= learned_merge_score_threshold]
        candidate_merges.sort(key=lambda candidate: candidate.score, reverse=True)
        candidate_merges = candidate_merges[: max(0, beam_max_candidates)]

        if beam_width > 1 and candidate_merges:
            def beam_atomicity(group_members: tuple[int, ...]) -> float:
                if atomicity_score_fn is None:
                    return 0.0
                sequences = tuple(
                    agent_sequences.setdefault(
                        agent_id,
                        _path_sequence_from_points(
                            _agent_points(path_arr, agent_id),
                            path_feature_mode=path_feature_mode,
                        ),
                    )
                    for agent_id in group_members
                    if len(
                        agent_sequences.setdefault(
                            agent_id,
                            _path_sequence_from_points(
                                _agent_points(path_arr, agent_id),
                                path_feature_mode=path_feature_mode,
                            ),
                        )
                    ) > 0
                )
                if not sequences:
                    return 0.0
                return float(atomicity_score_fn(sequences))

            final_groups = beam_search_merge_groups(
                role_agent_ids.tolist(),
                candidate_merges,
                beam_width=beam_width,
                atomicity_score_fn=beam_atomicity if atomicity_score_fn is not None else None,
                atomicity_weight=atomicity_score_weight,
            )
            grouped_agents = {idx: sorted(group) for idx, group in enumerate(final_groups)}
        else:
            for candidate in candidate_merges:
                union(candidate.left_agent, candidate.right_agent)
            grouped_agents = {}
    else:
        grouped_agents = {}

    if not grouped_agents:
        for agent_id in role_agent_ids.tolist():
            grouped_agents.setdefault(find(int(agent_id)), []).append(int(agent_id))

    neurons = {}
    synapse_owner = {}
    trees = {}
    for members in grouped_agents.values():
        pts = np.vstack([_agent_points(path_arr, agent_id) for agent_id in members if len(_agent_points(path_arr, agent_id)) > 0])
        synapse_indices = sorted(np.flatnonzero(role_hits[members, :].any(axis=0)).tolist())
        neuron_id = next_neuron_id
        next_neuron_id += 1
        neurons[neuron_id] = MergedNeuron(
            neuron_id=neuron_id,
            agent_ids=members,
            path_points=pts,
            synapse_indices=synapse_indices,
            role=role_name,
        )
        trees[neuron_id] = cKDTree(pts)
        for syn_idx in synapse_indices:
            synapse_owner.setdefault(int(syn_idx), []).append(neuron_id)

    return neurons, synapse_owner, trees, next_neuron_id


def _nearest_owner(
    syn_idx: int,
    pt: np.ndarray,
    owners: dict[int, list[int]],
    trees: dict[int, cKDTree],
    owner_margin: float = OWNER_MARGIN,
) -> tuple[int | None, float]:
    candidates = []
    for neuron_id in owners.get(syn_idx, [])[:PRE_POST_OWNER_TOPK]:
        dist, _ = trees[neuron_id].query(pt)
        candidates.append((neuron_id, float(dist)))

    if not candidates:
        return None, float("inf")

    candidates.sort(key=lambda item: item[1])
    best_id, best_dist = candidates[0]
    if len(candidates) > 1 and (candidates[1][1] - best_dist) < owner_margin:
        return None, float("inf")
    return best_id, best_dist


def _build_bridge_graph(
    neurons: dict,
    *,
    bridge_score_fn=None,
    max_bridge_cost: float | None = None,
) -> BridgeGraph:
    """Construct a ``BridgeGraph`` over neuron endpoint nodes.

    Each neuron contributes two nodes: one for its first path point
    (``node_id = 2 * neuron_id``) and one for its last path point
    (``node_id = 2 * neuron_id + 1``).  Intra-neuron edges connect the two
    endpoints with zero cost so the whole neuron acts as a free waypoint.
    Inter-neuron edges are added between every endpoint pair whose Euclidean
    distance is below ``max_bridge_cost`` (or unconditionally when
    ``max_bridge_cost`` is ``None``).  When a ``bridge_score_fn`` is provided
    it is used to override the Euclidean cost with a learned value.

    Parameters
    ----------
    neurons:
        Mapping from ``neuron_id`` to ``MergedNeuron`` objects as returned by
        ``_merge_role_groups``.
    bridge_score_fn:
        Optional callable ``(seq_a, seq_b) -> float`` returning a cost
        (lower is better).  When ``None``, Euclidean endpoint distance is used.
    max_bridge_cost:
        Prune candidate inter-neuron edges whose cost exceeds this threshold.

    Returns
    -------
    BridgeGraph
        Graph ready for ``BridgeGraph.best_bridge`` or ``BridgeGraph.dijkstra``.
    """
    graph = BridgeGraph()
    endpoint_pos: dict[int, np.ndarray] = {}
    endpoint_seq: dict[int, np.ndarray] = {}
    path_feature_mode = (
        getattr(bridge_score_fn, "path_feature_mode", DEFAULT_PATH_FEATURE_MODE)
        if bridge_score_fn is not None else DEFAULT_PATH_FEATURE_MODE
    )

    for neuron_id, neuron in neurons.items():
        pts = neuron.path_points
        if len(pts) == 0:
            continue
        nid_start = 2 * neuron_id
        nid_end = 2 * neuron_id + 1
        endpoint_pos[nid_start] = pts[0].astype(np.float32)
        endpoint_pos[nid_end] = pts[-1].astype(np.float32)
        # Free intra-neuron edge so the full neuron is traversable.
        graph.add_edge(nid_start, nid_end, 0.0)
        if bridge_score_fn is not None:
            seq = _path_sequence_from_points(pts, path_feature_mode=path_feature_mode)
            endpoint_seq[nid_start] = seq
            endpoint_seq[nid_end] = seq

    node_ids = list(endpoint_pos.keys())
    for i in range(len(node_ids)):
        u = node_ids[i]
        for j in range(i + 1, len(node_ids)):
            v = node_ids[j]
            # Skip the self-loop added above (same neuron's two endpoints).
            if u // 2 == v // 2:
                continue
            if bridge_score_fn is not None:
                seq_u = endpoint_seq.get(u)
                seq_v = endpoint_seq.get(v)
                if seq_u is None or seq_v is None or len(seq_u) == 0 or len(seq_v) == 0:
                    cost = float(np.linalg.norm(endpoint_pos[u] - endpoint_pos[v]))
                else:
                    # Bridge score fn returns a logit; convert to non-negative cost.
                    raw = float(bridge_score_fn(seq_u, seq_v))
                    cost = max(0.0, -raw)
            else:
                cost = float(np.linalg.norm(endpoint_pos[u] - endpoint_pos[v]))

            if max_bridge_cost is not None and cost > max_bridge_cost:
                continue
            graph.add_edge(u, v, cost)

    return graph


def _propose_bridges(
    neurons: dict,
    bridge_graph: BridgeGraph,
    *,
    max_bridge_cost: float | None = None,
    top_k: int = 8,
) -> list[tuple[int, int, float]]:
    """Return up to ``top_k`` cross-neuron bridge proposals sorted by cost.

    Each proposal is a tuple ``(neuron_id_a, neuron_id_b, cost)``.  The
    bridge is proposed between the two neurons whose endpoint pair is cheapest
    in the ``BridgeGraph``.  Only one proposal per unordered neuron pair is
    returned (the cheapest endpoint combination).

    Parameters
    ----------
    neurons:
        Same mapping passed to ``_build_bridge_graph``.
    bridge_graph:
        Pre-built ``BridgeGraph`` from ``_build_bridge_graph``.
    max_bridge_cost:
        Hard cost ceiling; proposals above this are discarded.
    top_k:
        Maximum number of proposals to return.

    Returns
    -------
    list of (neuron_id_a, neuron_id_b, cost)
        Sorted ascending by cost.
    """
    neuron_ids = [nid for nid, n in neurons.items() if len(n.path_points) > 0]
    proposals: dict[tuple[int, int], float] = {}

    for nid_a in neuron_ids:
        sources = [2 * nid_a, 2 * nid_a + 1]
        targets = [ep for nid_b in neuron_ids if nid_b != nid_a for ep in (2 * nid_b, 2 * nid_b + 1)]
        if not targets:
            continue
        result = bridge_graph.dijkstra(sources=sources, targets=targets, max_cost=max_bridge_cost)
        for node, path in result.items():
            nid_b = node // 2
            if nid_b == nid_a:
                continue
            pair = (min(nid_a, nid_b), max(nid_a, nid_b))
            if pair not in proposals or path.cost < proposals[pair]:
                proposals[pair] = path.cost

    sorted_proposals = sorted(
        ((a, b, cost) for (a, b), cost in proposals.items()),
        key=lambda item: item[2],
    )
    return sorted_proposals[:top_k]


def _build_graph(
    path_arr: np.ndarray,
    path_lengths: np.ndarray,
    synapse_hits: np.ndarray,
    pre_pts: np.ndarray,
    post_pts: np.ndarray,
    learned_merge_score_fn=None,
    learned_merge_score_threshold: float = LEARNED_MERGE_SCORE_THRESHOLD,
    atomicity_score_fn=None,
    beam_width: int = BEAM_WIDTH,
    beam_max_candidates: int = BEAM_MAX_CANDIDATES,
    atomicity_score_weight: float = ATOMICITY_SCORE_WEIGHT,
    pre_seg_ids: np.ndarray | None = None,
    post_seg_ids: np.ndarray | None = None,
    heuristic_config: "HeuristicConfig | None" = None,
) -> ConnectivityGraph:
    del path_lengths
    hcfg = heuristic_config or HeuristicConfig.legacy()
    n_syn = len(pre_pts)
    valid_idx = _valid_agent_indices(path_arr)
    if len(valid_idx) == 0:
        return ConnectivityGraph(neurons={}, edges=[], unresolved_synapse_indices=list(range(n_syn)))

    role_hits = synapse_hits[valid_idx]
    pre_hits = np.zeros_like(synapse_hits[:, :n_syn], dtype=bool)
    post_hits = np.zeros_like(synapse_hits[:, n_syn:], dtype=bool)
    pre_hits[valid_idx] = role_hits[:, :n_syn]
    post_hits[valid_idx] = role_hits[:, n_syn:]

    pre_neurons, pre_owners, pre_trees, next_id = _merge_role_groups(
        path_arr,
        pre_hits,
        "pre",
        0,
        learned_merge_score_fn=learned_merge_score_fn,
        learned_merge_score_threshold=learned_merge_score_threshold,
        atomicity_score_fn=atomicity_score_fn,
        beam_width=beam_width,
        beam_max_candidates=beam_max_candidates,
        atomicity_score_weight=atomicity_score_weight,
        role_seg_ids=pre_seg_ids,
        heuristic_config=hcfg,
    )
    post_neurons, post_owners, post_trees, next_id = _merge_role_groups(
        path_arr,
        post_hits,
        "post",
        next_id,
        learned_merge_score_fn=learned_merge_score_fn,
        learned_merge_score_threshold=learned_merge_score_threshold,
        atomicity_score_fn=atomicity_score_fn,
        beam_width=beam_width,
        beam_max_candidates=beam_max_candidates,
        atomicity_score_weight=atomicity_score_weight,
        role_seg_ids=post_seg_ids,
        heuristic_config=hcfg,
    )
    del next_id

    neurons = {}
    neurons.update(pre_neurons)
    neurons.update(post_neurons)
    assigned_synapses = {neuron_id: [] for neuron_id in neurons}

    edges = []
    unresolved = []
    _pcr = hcfg.polarity_capture_r
    _max_syn = hcfg.max_synapses_per_neuron
    for syn_idx in range(n_syn):
        pre_neuron, pre_dist = _nearest_owner(syn_idx, pre_pts[syn_idx], pre_owners, pre_trees)
        post_neuron, post_dist = _nearest_owner(syn_idx, post_pts[syn_idx], post_owners, post_trees)

        if (
            pre_neuron is not None
            and post_neuron is not None
            and pre_dist < _pcr
            and post_dist < _pcr
            and len(assigned_synapses[pre_neuron]) < _max_syn
            and len(assigned_synapses[post_neuron]) < _max_syn
        ):
            edges.append((pre_neuron, post_neuron, syn_idx))
            assigned_synapses[pre_neuron].append(syn_idx)
            assigned_synapses[post_neuron].append(syn_idx)
        else:
            unresolved.append(syn_idx)

    for neuron_id, neuron in neurons.items():
        neuron.synapse_indices = sorted(set(assigned_synapses[neuron_id]))

    return ConnectivityGraph(neurons=neurons, edges=edges, unresolved_synapse_indices=unresolved)


def simulate_paths_and_hits(
    volume: np.ndarray,
    pre_pts: np.ndarray,
    post_pts: np.ndarray,
    *,
    seed: int = 42,
    verbose: bool = True,
    membrane_field_override: np.ndarray | None = None,
):
    rng = np.random.default_rng(seed)
    volume_shape = np.array(volume.shape)
    all_syn_pts = np.vstack([pre_pts, post_pts])

    if membrane_field_override is not None:
        mf = membrane_field_override.astype(np.float32, copy=False)
    else:
        mf = compute_membrane_field(volume, sigma=MEMBRANE_SIGMA)
    mv = compute_membrane_vectors(mf, sigma=MEMBRANE_VECTOR_SIGMA)
    ef = compute_exploration_field(volume.shape)

    path_arr, synapse_hits, alive = run_agents_vectorized(
        volume_shape=volume_shape,
        n_agents=N_AGENTS,
        synapse_pts=all_syn_pts,
        membrane_field=mf,
        membrane_vectors=mv,
        exploration_field=ef,
        config=AGENT_CONFIG,
        rng=rng,
        synapse_fraction=SYNAPSE_SPAWN_FRACTION,
        verbose=verbose,
    )
    path_lengths = np.array([AGENT_CONFIG.max_steps] * N_AGENTS)
    return path_arr, synapse_hits, path_lengths, alive


def build_graph_hypotheses(
    path_arr: np.ndarray,
    path_lengths: np.ndarray,
    synapse_hits: np.ndarray,
    pre_pts: np.ndarray,
    post_pts: np.ndarray,
    *,
    thresholds: list[float],
    beam_widths: list[int],
    shared_grammar_checkpoint: str | None = SHARED_GRAMMAR_CHECKPOINT,
):
    learned_merge_score_fn = None
    atomicity_score_fn = None
    if shared_grammar_checkpoint:
        learned_merge_score_fn = _load_shared_merge_score_fn(shared_grammar_checkpoint)
        atomicity_score_fn = _load_shared_atomicity_score_fn(shared_grammar_checkpoint)

    hypotheses = []
    for threshold in thresholds:
        for beam_width in beam_widths:
            graph = _build_graph(
                path_arr,
                path_lengths,
                synapse_hits,
                pre_pts,
                post_pts,
                learned_merge_score_fn=learned_merge_score_fn,
                learned_merge_score_threshold=threshold,
                atomicity_score_fn=atomicity_score_fn,
                beam_width=beam_width,
                beam_max_candidates=BEAM_MAX_CANDIDATES,
                atomicity_score_weight=ATOMICITY_SCORE_WEIGHT,
            )
            hypotheses.append((threshold, beam_width, graph))
    return hypotheses


def select_hypothesis_with_reranker(
    hypotheses: list[tuple[float, int, ConnectivityGraph]],
    *,
    reranker_checkpoint: str,
    n_synapses: int,
) -> tuple[float, int, ConnectivityGraph]:
    reranker = _load_assembly_reranker(reranker_checkpoint)
    features = np.stack(
        [
            hypothesis_features(
                graph,
                merge_threshold=threshold,
                beam_width=beam_width,
                n_synapses=n_synapses,
            )
            for threshold, beam_width, graph in hypotheses
        ],
        axis=0,
    ).astype(np.float32)
    scores = reranker.predict(features)
    best_idx = int(np.argmax(scores))
    return hypotheses[best_idx]


def run(
    volume: np.ndarray,
    pre_pts: np.ndarray,
    post_pts: np.ndarray,
    pre_root_ids: np.ndarray,
    post_root_ids: np.ndarray,
    seed: int = 42,
    verbose: bool = True,
    membrane_field_override: np.ndarray | None = None,
    shared_grammar_checkpoint: str | None = SHARED_GRAMMAR_CHECKPOINT,
    assembly_reranker_checkpoint: str | None = ASSEMBLY_RERANKER_CHECKPOINT,
    learned_merge_score_fn=None,
    learned_merge_score_threshold: float = LEARNED_MERGE_SCORE_THRESHOLD,
    beam_width: int = BEAM_WIDTH,
    beam_max_candidates: int = BEAM_MAX_CANDIDATES,
    atomicity_score_weight: float = ATOMICITY_SCORE_WEIGHT,
    reranker_thresholds: str = RERANKER_THRESHOLDS,
    reranker_beam_widths: str = RERANKER_BEAM_WIDTHS,
    pre_seg_ids: np.ndarray | None = None,
    post_seg_ids: np.ndarray | None = None,
    membrane_unet_checkpoint: str | None = None,
    gat_assembly_checkpoint: str | None = None,
    gat_edge_threshold: float = 0.5,
    cell_gnn_checkpoint: str | None = CELL_GNN_CHECKPOINT,
    cell_gnn_partition_threshold: float = 0.5,
    cell_gnn_proximity_radius_nm: float = 5000.0,
) -> LineGraphMetrics:
    t0 = time.time()
    all_syn_pts = np.vstack([pre_pts, post_pts])

    if verbose:
        print("Computing membrane fields...")
    if membrane_field_override is not None:
        mf = membrane_field_override.astype(np.float32, copy=False)
        source = "override"
    elif membrane_unet_checkpoint is not None:
        _unet_model, _unet_device = _load_membrane_model(membrane_unet_checkpoint)
        mf = _predict_membranes(_unet_model, volume, device=_unet_device)
        source = f"UNet({membrane_unet_checkpoint})"
    else:
        mf = compute_membrane_field(volume, sigma=MEMBRANE_SIGMA)
        source = "Sobel"
    if verbose:
        print(f"  {time.time() - t0:.2f}s | {source} | vol={volume.shape} synapses={len(pre_pts)}")

    if verbose:
        print(f"Running {N_AGENTS} agents x {AGENT_CONFIG.max_steps} steps...")
    t1 = time.time()
    path_arr, synapse_hits, path_lengths, alive = simulate_paths_and_hits(
        volume,
        pre_pts,
        post_pts,
        seed=seed,
        verbose=verbose,
        membrane_field_override=mf,
    )
    if verbose:
        hit_count = synapse_hits.any(axis=0).sum()
        print(f"  {time.time() - t1:.2f}s | {hit_count}/{len(all_syn_pts)} sites hit, {alive.sum()} alive")
    t2 = time.time()

    # --- CellGNN assembly path (alternative to beam search) ---
    if cell_gnn_checkpoint is not None:
        from .cell_graph import cell_gnn_assembly, load_cell_gnn

        if verbose:
            print("CellGNN assembly …")
        cell_gnn_model = load_cell_gnn(cell_gnn_checkpoint)
        grammar_fn = None
        if shared_grammar_checkpoint:
            grammar_fn = _load_shared_merge_score_fn(shared_grammar_checkpoint)
        # Build a minimal SynapseTable-like object for cell_gnn_assembly
        from .fetch import SynapseTable
        syn_table = SynapseTable(
            pre_pt=pre_pts,
            post_pt=post_pts,
            pre_root_id=pre_root_ids,
            post_root_id=post_root_ids,
            synapse_id=np.arange(len(pre_pts), dtype=np.int64),
            pre_seg_id=pre_seg_ids,
            post_seg_id=post_seg_ids,
        )
        graph = cell_gnn_assembly(
            syn_table,
            cell_gnn_model,
            grammar_score_fn=grammar_fn,
            synapse_hits=synapse_hits,
            proximity_radius_nm=cell_gnn_proximity_radius_nm,
            partition_threshold=cell_gnn_partition_threshold,
            verbose=verbose,
        )
        if verbose:
            print(
                f"  CellGNN assembly {time.time() - t2:.2f}s | "
                f"{len(graph.neurons)} neurons, {len(graph.edges)} edges"
            )
        metrics = evaluate(graph, pre_root_ids, post_root_ids)
        if verbose:
            print(f"\nTotal: {time.time() - t0:.2f}s")
            print(f"Result: {metrics}")
        return metrics

    score_fn = learned_merge_score_fn
    atomicity_fn = None
    if score_fn is None and shared_grammar_checkpoint:
        score_fn = _load_shared_merge_score_fn(shared_grammar_checkpoint)
    if shared_grammar_checkpoint:
        atomicity_fn = _load_shared_atomicity_score_fn(shared_grammar_checkpoint)

    # --- PR 5: auto-select heuristic policy ---
    # When any learned component is available (grammar, GAT, or explicit scorer),
    # switch to permissive candidate-generation mode so that spatial thresholds
    # become pre-filters rather than hard decisions.
    _use_learned = bool(
        shared_grammar_checkpoint
        or gat_assembly_checkpoint
        or learned_merge_score_fn is not None
    )
    _hcfg = HeuristicConfig.learned() if _use_learned else HeuristicConfig.legacy()
    if verbose and _use_learned:
        print(f"  heuristic_mode=learned | polarity_capture_r=∞ | max_syn={_hcfg.max_synapses_per_neuron}")

    if assembly_reranker_checkpoint:
        thresholds = [float(item.strip()) for item in reranker_thresholds.split(",") if item.strip()]
        beam_widths = [int(item.strip()) for item in reranker_beam_widths.split(",") if item.strip()]
        hypotheses = build_graph_hypotheses(
            path_arr,
            path_lengths,
            synapse_hits,
            pre_pts,
            post_pts,
            thresholds=thresholds,
            beam_widths=beam_widths,
            shared_grammar_checkpoint=shared_grammar_checkpoint,
        )
        _, _, graph = select_hypothesis_with_reranker(
            hypotheses,
            reranker_checkpoint=assembly_reranker_checkpoint,
            n_synapses=len(pre_pts),
        )
    else:
        graph = _build_graph(
            path_arr,
            path_lengths,
            synapse_hits,
            pre_pts,
            post_pts,
            learned_merge_score_fn=score_fn,
            learned_merge_score_threshold=learned_merge_score_threshold,
            atomicity_score_fn=atomicity_fn,
            beam_width=beam_width,
            beam_max_candidates=beam_max_candidates,
            atomicity_score_weight=atomicity_score_weight,
            pre_seg_ids=pre_seg_ids,
            post_seg_ids=post_seg_ids,
            heuristic_config=_hcfg,
        )
    if verbose:
        print(
            f"  merge+graph {time.time() - t2:.2f}s | "
            f"{len(graph.neurons)} neurons, {len(graph.edges)} edges, "
            f"{len(graph.unresolved_synapse_indices)} unresolved"
        )

    # --- PR 4: Global GAT refinement ---
    if gat_assembly_checkpoint is not None:
        from .shared_grammar_model import load_global_assembly_gat
        t3 = time.time()
        gat_model = load_global_assembly_gat(gat_assembly_checkpoint)
        # Reuse the path encoder from the shared grammar checkpoint when available,
        # otherwise build a minimal encoder consistent with default dimensions.
        if shared_grammar_checkpoint:
            from .shared_grammar_model import load_shared_grammar_model
            _sgm = load_shared_grammar_model(shared_grammar_checkpoint)
            _enc = _sgm.path_encoder
        else:
            from .shared_grammar_model import SharedGrammarModel
            _enc = SharedGrammarModel(
                input_dim=6,
                path_feature_mode=DEFAULT_PATH_FEATURE_MODE,
            ).path_encoder
        graph = gat_refine_connectivity(
            graph, _enc, gat_model, threshold=gat_edge_threshold
        )
        if verbose:
            print(
                f"  GAT refine {time.time() - t3:.2f}s | "
                f"{len(graph.edges)} edges after threshold={gat_edge_threshold}"
            )

    metrics = evaluate(graph, pre_root_ids, post_root_ids)
    if verbose:
        print(f"\nTotal: {time.time() - t0:.2f}s")
        print(f"Result: {metrics}")
    return metrics


def evaluate_synthetic_case(
    benchmark_config: SyntheticBenchmarkConfig,
    volume_seed: int | None = None,
    run_seed: int | None = None,
    verbose: bool = True,
    use_scaffold: bool = True,
    membrane_unet_checkpoint: str | None = None,
    gat_assembly_checkpoint: str | None = None,
    gat_edge_threshold: float = 0.5,
    cell_gnn_checkpoint: str | None = None,
    cell_gnn_partition_threshold: float = 0.5,
    cell_gnn_proximity_radius_nm: float = 5000.0,
) -> LineGraphMetrics:
    chunk, synapses = make_test_volume(config=benchmark_config, seed=volume_seed)
    return run(
        volume=chunk.data,
        pre_pts=synapses.pre_pt,
        post_pts=synapses.post_pt,
        pre_root_ids=synapses.pre_root_id,
        post_root_ids=synapses.post_root_id,
        seed=42 if run_seed is None else run_seed,
        verbose=verbose,
        pre_seg_ids=synapses.pre_seg_id if use_scaffold else None,
        post_seg_ids=synapses.post_seg_id if use_scaffold else None,
        membrane_unet_checkpoint=membrane_unet_checkpoint,
        gat_assembly_checkpoint=gat_assembly_checkpoint,
        gat_edge_threshold=gat_edge_threshold,
        cell_gnn_checkpoint=cell_gnn_checkpoint,
        cell_gnn_partition_threshold=cell_gnn_partition_threshold,
        cell_gnn_proximity_radius_nm=cell_gnn_proximity_radius_nm,
    )


def evaluate_synthetic_batch(
    benchmark_config: SyntheticBenchmarkConfig,
    cases: int,
    mode: str,
    base_seed: int | None = None,
    verbose: bool = True,
) -> tuple[LineGraphMetrics, list[dict[str, float | int | None]]]:
    if cases < 1:
        raise ValueError("cases must be >= 1")

    batch_rng = np.random.default_rng(base_seed)
    case_summaries = []
    metrics_list = []

    for case_idx in range(cases):
        if mode == "fixed_validation":
            volume_seed = case_idx
            run_seed = case_idx
        elif mode == "random":
            volume_seed = int(batch_rng.integers(0, 2**31 - 1))
            run_seed = int(batch_rng.integers(0, 2**31 - 1))
        else:
            raise ValueError(f"unsupported benchmark mode: {mode}")

        if verbose:
            print(f"\n--- Case {case_idx + 1}/{cases} volume_seed={volume_seed} run_seed={run_seed} ---")
        metrics = evaluate_synthetic_case(
            benchmark_config=benchmark_config,
            volume_seed=volume_seed,
            run_seed=run_seed,
            verbose=verbose,
        )
        metrics_list.append(metrics)
        case_summaries.append(
            {
                "case": case_idx + 1,
                "volume_seed": volume_seed,
                "run_seed": run_seed,
                "f1": metrics.f1,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "tp": metrics.tp,
                "fp": metrics.fp,
                "fn": metrics.fn,
            }
        )

    agg = LineGraphMetrics(
        tp=sum(m.tp for m in metrics_list),
        fp=sum(m.fp for m in metrics_list),
        fn=sum(m.fn for m in metrics_list),
        precision=float(np.mean([m.precision for m in metrics_list])),
        recall=float(np.mean([m.recall for m in metrics_list])),
        f1=float(np.mean([m.f1 for m in metrics_list])),
        n_true_edges=sum(m.n_true_edges for m in metrics_list),
        n_estimated_edges=sum(m.n_estimated_edges for m in metrics_list),
        n_synapses=sum(m.n_synapses for m in metrics_list),
    )
    return agg, case_summaries


def evaluate_real_box(
    box: RealBoxSpec,
    min_synapses: int = REAL_MIN_SYNAPSES,
    seed: int = 42,
    verbose: bool = True,
    membrane_source: str = "auto",
    membrane_cache_dir: str = MEMBRANE_CACHE_DIR,
    shared_grammar_checkpoint: str | None = SHARED_GRAMMAR_CHECKPOINT,
    assembly_reranker_checkpoint: str | None = ASSEMBLY_RERANKER_CHECKPOINT,
    learned_merge_score_threshold: float = LEARNED_MERGE_SCORE_THRESHOLD,
    beam_width: int = BEAM_WIDTH,
    beam_max_candidates: int = BEAM_MAX_CANDIDATES,
    atomicity_score_weight: float = ATOMICITY_SCORE_WEIGHT,
    reranker_thresholds: str = RERANKER_THRESHOLDS,
    reranker_beam_widths: str = RERANKER_BEAM_WIDTHS,
    cell_gnn_checkpoint: str | None = None,
    cell_gnn_partition_threshold: float = 0.5,
    cell_gnn_proximity_radius_nm: float = 5000.0,
) -> tuple[LineGraphMetrics | None, dict[str, int | float | tuple]]:
    synapses = fetch_synapses(box.bbox_nm, mip=box.mip)
    summary = {
        "center_nm": box.center_nm,
        "side_um": box.side_um,
        "mip": box.mip,
        "synapses": int(len(synapses.pre_pt)),
    }
    if len(synapses.pre_pt) < min_synapses:
        return None, summary

    chunk = fetch_volume(box.bbox_nm, mip=box.mip)
    membrane = None
    membrane_status = "sobel"
    if membrane_source in {"auto", "cache"}:
        membrane = load_cached_membrane(box, membrane_cache_dir)
        if membrane is not None:
            membrane_status = "cache"
        elif membrane_source == "cache":
            raise FileNotFoundError(f"missing cached membrane for {box.center_nm} in {membrane_cache_dir}")
    summary["membrane_source"] = membrane_status
    metrics = run(
        volume=chunk.data,
        pre_pts=synapses.pre_pt,
        post_pts=synapses.post_pt,
        pre_root_ids=synapses.pre_root_id,
        post_root_ids=synapses.post_root_id,
        seed=seed,
        verbose=verbose,
        membrane_field_override=membrane,
        shared_grammar_checkpoint=shared_grammar_checkpoint,
        assembly_reranker_checkpoint=assembly_reranker_checkpoint,
        learned_merge_score_threshold=learned_merge_score_threshold,
        beam_width=beam_width,
        beam_max_candidates=beam_max_candidates,
        atomicity_score_weight=atomicity_score_weight,
        reranker_thresholds=reranker_thresholds,
        reranker_beam_widths=reranker_beam_widths,
        cell_gnn_checkpoint=cell_gnn_checkpoint,
        cell_gnn_partition_threshold=cell_gnn_partition_threshold,
        cell_gnn_proximity_radius_nm=cell_gnn_proximity_radius_nm,
    )
    return metrics, summary


def evaluate_real_box_set(
    boxes: list[RealBoxSpec],
    boxes_per_eval: int,
    min_synapses: int = REAL_MIN_SYNAPSES,
    seed: int = 42,
    verbose: bool = True,
    membrane_source: str = "auto",
    membrane_cache_dir: str = MEMBRANE_CACHE_DIR,
    shared_grammar_checkpoint: str | None = SHARED_GRAMMAR_CHECKPOINT,
    assembly_reranker_checkpoint: str | None = ASSEMBLY_RERANKER_CHECKPOINT,
    learned_merge_score_threshold: float = LEARNED_MERGE_SCORE_THRESHOLD,
    beam_width: int = BEAM_WIDTH,
    beam_max_candidates: int = BEAM_MAX_CANDIDATES,
    atomicity_score_weight: float = ATOMICITY_SCORE_WEIGHT,
    reranker_thresholds: str = RERANKER_THRESHOLDS,
    reranker_beam_widths: str = RERANKER_BEAM_WIDTHS,
    cell_gnn_checkpoint: str | None = None,
    cell_gnn_partition_threshold: float = 0.5,
    cell_gnn_proximity_radius_nm: float = 5000.0,
) -> tuple[LineGraphMetrics, list[dict[str, int | float | tuple]]]:
    summaries = []
    metrics_list = []

    for box in boxes:
        metrics, summary = evaluate_real_box(
            box=box,
            min_synapses=min_synapses,
            seed=seed,
            verbose=verbose,
            membrane_source=membrane_source,
            membrane_cache_dir=membrane_cache_dir,
            shared_grammar_checkpoint=shared_grammar_checkpoint,
            assembly_reranker_checkpoint=assembly_reranker_checkpoint,
            learned_merge_score_threshold=learned_merge_score_threshold,
            beam_width=beam_width,
            beam_max_candidates=beam_max_candidates,
            atomicity_score_weight=atomicity_score_weight,
            reranker_thresholds=reranker_thresholds,
            reranker_beam_widths=reranker_beam_widths,
            cell_gnn_checkpoint=cell_gnn_checkpoint,
            cell_gnn_partition_threshold=cell_gnn_partition_threshold,
            cell_gnn_proximity_radius_nm=cell_gnn_proximity_radius_nm,
        )
        if metrics is None:
            summaries.append({**summary, "status": "skip_low_synapses"})
            continue
        summaries.append({**summary, "status": "used", "f1": metrics.f1})
        metrics_list.append(metrics)
        if len(metrics_list) >= boxes_per_eval:
            break

    if len(metrics_list) < boxes_per_eval:
        raise RuntimeError(
            f"only found {len(metrics_list)} real boxes with >= {min_synapses} synapses; "
            f"need {boxes_per_eval}"
        )

    agg = LineGraphMetrics(
        tp=sum(m.tp for m in metrics_list),
        fp=sum(m.fp for m in metrics_list),
        fn=sum(m.fn for m in metrics_list),
        precision=float(np.mean([m.precision for m in metrics_list])),
        recall=float(np.mean([m.recall for m in metrics_list])),
        f1=float(np.mean([m.f1 for m in metrics_list])),
        n_true_edges=sum(m.n_true_edges for m in metrics_list),
        n_estimated_edges=sum(m.n_estimated_edges for m in metrics_list),
        n_synapses=sum(m.n_synapses for m in metrics_list),
    )
    return agg, summaries


def _parse_box_indices(indices_text: str | None) -> list[RealBoxSpec]:
    if indices_text is None or not indices_text.strip():
        return list(REAL_BOXES)
    indices = [int(item.strip()) for item in indices_text.split(",") if item.strip()]
    return [REAL_BOXES[idx] for idx in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-mode",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Whether to evaluate on synthetic benchmark cases or real MICrONS boxes.",
    )
    parser.add_argument("--cases", type=int, default=BENCHMARK_CASES, help="Synthetic cases per evaluation.")
    parser.add_argument(
        "--benchmark-mode",
        choices=["random", "fixed_validation"],
        default=BENCHMARK_MODE,
        help="Whether to sample fresh synthetic cases or use a fixed validation set.",
    )
    parser.add_argument("--eval-seed", type=int, default=None, help="Optional seed for reproducible batch sampling.")
    parser.add_argument("--volume-seed", type=int, default=None, help="Optional debug override for a single synthetic case.")
    parser.add_argument("--run-seed", type=int, default=None, help="Optional debug override for a single agent simulation.")
    parser.add_argument("--real-boxes-per-eval", type=int, default=REAL_BOXES_PER_EVAL, help="Real boxes to average per evaluation.")
    parser.add_argument("--real-box-indices", default=None, help="Optional comma-separated subset of REAL_BOXES to evaluate.")
    parser.add_argument("--real-min-synapses", type=int, default=REAL_MIN_SYNAPSES, help="Minimum synapses required for a real box to be used.")
    parser.add_argument(
        "--membrane-source",
        choices=["auto", "cache", "sobel"],
        default="auto",
        help="Use cached learned membranes when available, require them, or always use Sobel.",
    )
    parser.add_argument(
        "--membrane-cache-dir",
        default=MEMBRANE_CACHE_DIR,
        help="Directory containing cached membrane .npy volumes for real boxes.",
    )
    parser.add_argument("--shared-grammar-checkpoint", default=SHARED_GRAMMAR_CHECKPOINT, help="Optional shared grammar checkpoint for learned merge scoring.")
    parser.add_argument("--assembly-reranker-checkpoint", default=ASSEMBLY_RERANKER_CHECKPOINT, help="Optional assembly reranker checkpoint for hypothesis selection.")
    parser.add_argument("--learned-merge-score-threshold", type=float, default=LEARNED_MERGE_SCORE_THRESHOLD, help="Decision threshold over learned merge logits.")
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH, help="Beam width for box-scale learned assembly search.")
    parser.add_argument("--beam-max-candidates", type=int, default=BEAM_MAX_CANDIDATES, help="Maximum learned merge candidates to consider in beam search.")
    parser.add_argument("--atomicity-score-weight", type=float, default=ATOMICITY_SCORE_WEIGHT, help="Weight for atomicity reranking inside beam search.")
    parser.add_argument("--reranker-thresholds", default=RERANKER_THRESHOLDS, help="Threshold sweep used when reranker-driven hypothesis selection is enabled.")
    parser.add_argument("--reranker-beam-widths", default=RERANKER_BEAM_WIDTHS, help="Beam-width sweep used when reranker-driven hypothesis selection is enabled.")
    parser.add_argument("--cell-gnn-checkpoint", default=CELL_GNN_CHECKPOINT, help="CellGNN checkpoint for global topological merge (bypasses beam search).")
    parser.add_argument("--cell-gnn-partition-threshold", type=float, default=0.5, help="Cosine similarity threshold for CellGNN partition clustering.")
    parser.add_argument("--cell-gnn-proximity-radius-nm", type=float, default=5000.0, help="Spatial radius for CellGNN evidence graph construction.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step benchmark logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_mode == "synthetic":
        print("=== Synthetic benchmark ===")
        print(
            f"mode={args.benchmark_mode} cases={args.cases} "
            f"eval_seed={args.eval_seed} benchmark_config={asdict(BENCHMARK_CONFIG)}"
        )

        if args.volume_seed is not None or args.run_seed is not None:
            print(f"debug_replay volume_seed={args.volume_seed} run_seed={args.run_seed}")
            metrics = evaluate_synthetic_case(
                benchmark_config=BENCHMARK_CONFIG,
                volume_seed=args.volume_seed,
                run_seed=args.run_seed,
                verbose=not args.quiet,
                cell_gnn_checkpoint=getattr(args, "cell_gnn_checkpoint", None),
                cell_gnn_partition_threshold=getattr(args, "cell_gnn_partition_threshold", 0.5),
                cell_gnn_proximity_radius_nm=getattr(args, "cell_gnn_proximity_radius_nm", 5000.0),
            )
            print(f"Result: {metrics}")
            print(f"\nval_f1 = {metrics.f1:.4f}")
            return

        metrics, case_summaries = evaluate_synthetic_batch(
            benchmark_config=BENCHMARK_CONFIG,
            cases=args.cases,
            mode=args.benchmark_mode,
            base_seed=args.eval_seed,
            verbose=not args.quiet,
        )
        for summary in case_summaries:
            print(
                "case_result "
                f"case={summary['case']} "
                f"volume_seed={summary['volume_seed']} "
                f"run_seed={summary['run_seed']} "
                f"f1={summary['f1']:.4f} "
                f"p={summary['precision']:.3f} "
                f"r={summary['recall']:.3f}"
            )
        print(f"Result: {metrics}")
        print(f"\nval_f1 = {metrics.f1:.4f}")
        return

    print("=== Real MICrONS benchmark ===")
    selected_boxes = _parse_box_indices(args.real_box_indices)
    print(
        f"boxes_per_eval={args.real_boxes_per_eval} "
        f"real_min_synapses={args.real_min_synapses} "
        f"candidate_boxes={len(selected_boxes)}"
    )
    metrics, box_summaries = evaluate_real_box_set(
        boxes=selected_boxes,
        boxes_per_eval=args.real_boxes_per_eval,
        min_synapses=args.real_min_synapses,
        seed=42 if args.run_seed is None else args.run_seed,
        verbose=not args.quiet,
        membrane_source=args.membrane_source,
        membrane_cache_dir=args.membrane_cache_dir,
        shared_grammar_checkpoint=args.shared_grammar_checkpoint,
        assembly_reranker_checkpoint=args.assembly_reranker_checkpoint,
        learned_merge_score_threshold=args.learned_merge_score_threshold,
        beam_width=args.beam_width,
        beam_max_candidates=args.beam_max_candidates,
        atomicity_score_weight=args.atomicity_score_weight,
        reranker_thresholds=args.reranker_thresholds,
        reranker_beam_widths=args.reranker_beam_widths,
        cell_gnn_checkpoint=getattr(args, "cell_gnn_checkpoint", None),
        cell_gnn_partition_threshold=getattr(args, "cell_gnn_partition_threshold", 0.5),
        cell_gnn_proximity_radius_nm=getattr(args, "cell_gnn_proximity_radius_nm", 5000.0),
    )
    for idx, summary in enumerate(box_summaries, start=1):
        print(
            "box_result "
            f"idx={idx} center_nm={summary['center_nm']} side_um={summary['side_um']} "
            f"synapses={summary['synapses']} status={summary['status']} membrane={summary.get('membrane_source', 'n/a')}"
            + (f" f1={summary['f1']:.4f}" if 'f1' in summary else "")
        )
    print(f"Result: {metrics}")
    print(f"\nval_f1 = {metrics.f1:.4f}")


if __name__ == "__main__":
    main()
