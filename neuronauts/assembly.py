"""Box-scale assembly search: beam search and GAT-based global refinement."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from .grammar import DEFAULT_PATH_FEATURE_MODE, featurize_path_points

if TYPE_CHECKING:
    from .merge import ConnectivityGraph


@dataclass(frozen=True)
class CandidateMerge:
    left_agent: int
    right_agent: int
    score: float


@dataclass(frozen=True)
class BeamState:
    groups: tuple[frozenset[int], ...]
    score: float


def _canonical_groups(groups: tuple[frozenset[int], ...]) -> tuple[frozenset[int], ...]:
    return tuple(sorted(groups, key=lambda group: (min(group), len(group), tuple(sorted(group)))))


def _merge_groups(groups: tuple[frozenset[int], ...], left_agent: int, right_agent: int) -> tuple[frozenset[int], ...]:
    left_idx = None
    right_idx = None
    for idx, group in enumerate(groups):
        if left_agent in group:
            left_idx = idx
        if right_agent in group:
            right_idx = idx
    if left_idx is None or right_idx is None or left_idx == right_idx:
        return groups

    merged = frozenset(set(groups[left_idx]) | set(groups[right_idx]))
    new_groups = [group for idx, group in enumerate(groups) if idx not in {left_idx, right_idx}]
    new_groups.append(merged)
    return _canonical_groups(tuple(new_groups))


def beam_search_merge_groups(
    agent_ids: list[int],
    candidates: list[CandidateMerge],
    *,
    beam_width: int = 4,
    atomicity_score_fn=None,
    atomicity_weight: float = 0.25,
) -> tuple[frozenset[int], ...]:
    """Explore a small beam of accept/reject decisions over candidate merges."""
    initial = BeamState(groups=_canonical_groups(tuple(frozenset({agent_id}) for agent_id in agent_ids)), score=0.0)
    beam = [initial]

    for candidate in candidates:
        expanded: list[BeamState] = []
        for state in beam:
            expanded.append(state)
            merged_groups = _merge_groups(state.groups, candidate.left_agent, candidate.right_agent)
            if merged_groups == state.groups:
                continue

            accept_score = state.score + float(candidate.score)
            if atomicity_score_fn is not None:
                target_group = next(group for group in merged_groups if candidate.left_agent in group and candidate.right_agent in group)
                accept_score += float(atomicity_weight) * float(atomicity_score_fn(tuple(sorted(target_group))))
            expanded.append(BeamState(groups=merged_groups, score=accept_score))

        dedup: dict[tuple[frozenset[int], ...], BeamState] = {}
        for state in expanded:
            prev = dedup.get(state.groups)
            if prev is None or state.score > prev.score:
                dedup[state.groups] = state
        beam = sorted(dedup.values(), key=lambda state: state.score, reverse=True)[: max(1, beam_width)]

    return beam[0].groups if beam else initial.groups


def _score_to_affinity(score: float) -> float:
    clipped = float(np.clip(score, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _fallback_bipartition(affinity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_items = int(affinity.shape[0])
    if n_items <= 1:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty
    if n_items == 2:
        return np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)

    tri_upper = np.triu_indices(n_items, k=1)
    edge_weights = affinity[tri_upper]
    weakest_idx = int(np.argmin(edge_weights)) if len(edge_weights) > 0 else 0
    seed_left = int(tri_upper[0][weakest_idx]) if len(edge_weights) > 0 else 0
    seed_right = int(tri_upper[1][weakest_idx]) if len(edge_weights) > 0 else 1

    left = [seed_left]
    right = [seed_right]
    for idx in range(n_items):
        if idx in {seed_left, seed_right}:
            continue
        if affinity[idx, seed_left] >= affinity[idx, seed_right]:
            left.append(idx)
        else:
            right.append(idx)

    if not left or not right:
        split_at = max(1, n_items // 2)
        order = np.arange(n_items, dtype=np.int32)
        return order[:split_at], order[split_at:]
    return np.array(sorted(left), dtype=np.int32), np.array(sorted(right), dtype=np.int32)


def _spectral_bipartition(affinity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_items = int(affinity.shape[0])
    if n_items <= 1:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty
    if n_items == 2:
        return np.array([0], dtype=np.int32), np.array([1], dtype=np.int32)

    degree = affinity.sum(axis=1)
    if np.allclose(degree, 0.0):
        return _fallback_bipartition(affinity)

    laplacian = np.diag(degree) - affinity
    try:
        _, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        return _fallback_bipartition(affinity)
    if eigenvectors.shape[1] < 2:
        return _fallback_bipartition(affinity)

    fiedler = eigenvectors[:, 1]
    pivot = float(np.median(fiedler))
    left = np.flatnonzero(fiedler <= pivot).astype(np.int32)
    right = np.flatnonzero(fiedler > pivot).astype(np.int32)

    if len(left) == 0 or len(right) == 0:
        order = np.argsort(fiedler).astype(np.int32)
        split_at = max(1, n_items // 2)
        left = order[:split_at]
        right = order[split_at:]
    if len(left) == 0 or len(right) == 0:
        return _fallback_bipartition(affinity)
    return left, right


def repartition_low_atomicity_group(
    members: tuple[int, ...] | list[int],
    *,
    pair_score_fn,
    atomicity_score_fn,
    atomicity_threshold: float = 0.0,
    min_group_size: int = 3,
    max_rounds: int = 2,
) -> tuple[tuple[int, ...], ...]:
    """Split a low-atomicity cell into subgroups using pairwise merge affinities.

    The split is intentionally simple and dependency-light:

    1. Convert pairwise merge scores into a symmetric affinity matrix.
    2. Take a spectral two-way cut of that matrix.
    3. Accept the split only if the mean child atomicity improves over the
       parent. Recurse on any child that is still below the threshold.
    """
    ordered_members = tuple(sorted(int(member) for member in members))
    if len(ordered_members) < max(2, int(min_group_size)) or max_rounds <= 0:
        return (ordered_members,)

    parent_score = float(atomicity_score_fn(ordered_members))
    if parent_score >= float(atomicity_threshold):
        return (ordered_members,)

    n_items = len(ordered_members)
    affinity = np.zeros((n_items, n_items), dtype=np.float32)
    score_cache: dict[tuple[int, int], float] = {}

    for i in range(n_items):
        affinity[i, i] = 1.0
        for j in range(i + 1, n_items):
            pair = (ordered_members[i], ordered_members[j])
            score = score_cache.get(pair)
            if score is None:
                score = float(pair_score_fn(*pair))
                score_cache[pair] = score
            weight = _score_to_affinity(score)
            affinity[i, j] = weight
            affinity[j, i] = weight

    left_idx, right_idx = _spectral_bipartition(affinity)
    if len(left_idx) == 0 or len(right_idx) == 0:
        return (ordered_members,)

    left_group = tuple(sorted(ordered_members[int(idx)] for idx in left_idx.tolist()))
    right_group = tuple(sorted(ordered_members[int(idx)] for idx in right_idx.tolist()))
    if not left_group or not right_group:
        return (ordered_members,)

    child_scores = [
        float(atomicity_score_fn(left_group)),
        float(atomicity_score_fn(right_group)),
    ]
    if float(np.mean(child_scores)) <= parent_score:
        return (ordered_members,)

    partitions: list[tuple[int, ...]] = []
    for child_group, child_score in zip((left_group, right_group), child_scores):
        if len(child_group) >= max(2, int(min_group_size)) and child_score < float(atomicity_threshold):
            partitions.extend(
                repartition_low_atomicity_group(
                    child_group,
                    pair_score_fn=pair_score_fn,
                    atomicity_score_fn=atomicity_score_fn,
                    atomicity_threshold=atomicity_threshold,
                    min_group_size=min_group_size,
                    max_rounds=max_rounds - 1,
                )
            )
        else:
            partitions.append(child_group)

    return tuple(sorted(partitions, key=lambda group: (group[0], len(group), group)))


# ---------------------------------------------------------------------------
# PR 4: Global GAT Assembly
# ---------------------------------------------------------------------------

# Isotropic scaling: convert MIP-2 voxel coords to 32-nm units (1 unit = 32 nm).
# Keeps feature values in the same numerical range as raw voxel coords (~1–60)
# while correctly weighting the Z axis at 40/32 = 1.25× relative to XY.
_PATH_ISO = np.array([1.0, 1.0, 40.0 / 32.0], dtype=np.float32)


def _path_seq_from_pts(points: np.ndarray) -> np.ndarray:
    """Convert an (K, 3) path-point array to the default shared feature mode."""
    return featurize_path_points(points, mode=DEFAULT_PATH_FEATURE_MODE, iso_scale=_PATH_ISO)


def _encode_neurons(neurons: dict, path_encoder) -> "tuple[list[int], object]":
    """Encode each neuron's path sequence and return ``(node_ids, h)`` where
    ``h`` is a ``[N, embedding_dim]`` float32 tensor.

    Neurons with no usable path points receive an all-zero embedding.
    """
    import torch

    node_ids = sorted(neurons.keys())
    if not node_ids:
        return [], torch.zeros((0, path_encoder.output_dim if hasattr(path_encoder, "output_dim") else 32))

    # Respect the path encoder's positional-encoding budget.
    enc_max_len: int = int(getattr(path_encoder, "max_len", 512))

    path_feature_mode = getattr(path_encoder, "path_feature_mode", DEFAULT_PATH_FEATURE_MODE)
    seqs = [
        featurize_path_points(
            neurons[nid].path_points,
            mode=path_feature_mode,
            iso_scale=_PATH_ISO,
        )
        for nid in node_ids
    ]
    # Subsample long paths to stay within the encoder's positional budget.
    seqs = [s[:enc_max_len] if len(s) > enc_max_len else s for s in seqs]
    max_len = max((len(s) for s in seqs), default=1)
    if max_len == 0:
        max_len = 1

    feat_dim = seqs[0].shape[1] if seqs[0].ndim == 2 and seqs[0].shape[0] > 0 else 3
    x = np.zeros((len(node_ids), max_len, feat_dim), dtype=np.float32)
    mask = np.ones((len(node_ids), max_len), dtype=bool)  # True = PAD

    for i, seq in enumerate(seqs):
        T = len(seq)
        if T > 0:
            x[i, :T] = seq
            mask[i, :T] = False

    x_t = torch.from_numpy(x).float()
    mask_t = torch.from_numpy(mask)

    with torch.no_grad():
        h = path_encoder(x_t, mask=mask_t)  # [N, embedding_dim]

    return node_ids, h


def _build_gat_edges(
    node_ids: list[int],
    graph: "ConnectivityGraph",
) -> "tuple[object, object, list[tuple[int, int]]]":
    """Build directed edge tensors for the GAT from a ConnectivityGraph.

    Each (pre_neuron, post_neuron) synapse edge is added in both directions,
    and self-loops are appended so every node attends to itself.

    Returns ``(src_tensor, dst_tensor, edge_pairs)`` where ``edge_pairs`` is
    the list of directed unique ``(src_idx, dst_idx)`` index pairs.
    """
    import torch

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    N = len(node_ids)
    edge_set: set[tuple[int, int]] = set()

    # Self-loops — ensure every node can attend to itself.
    for i in range(N):
        edge_set.add((i, i))

    # Bidirectional edges from synapse connections.
    for pre_nid, post_nid, _ in graph.edges:
        src_i = id_to_idx.get(pre_nid)
        dst_i = id_to_idx.get(post_nid)
        if src_i is not None and dst_i is not None and src_i != dst_i:
            edge_set.add((src_i, dst_i))
            edge_set.add((dst_i, src_i))

    if not edge_set:
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty, []

    pairs = sorted(edge_set)
    src = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    dst = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    return src, dst, pairs


def label_graph_edges(
    graph: "ConnectivityGraph",
    pre_root_ids: np.ndarray,
    post_root_ids: np.ndarray,
) -> np.ndarray:
    """Assign a binary ground-truth label to every edge in a ConnectivityGraph.

    An edge ``(pre_neuron, post_neuron, syn_idx)`` is labelled **1 (correct)**
    when:

    - The pre-neuron's majority-vote root_id matches ``pre_root_ids[syn_idx]``.
    - The post-neuron's majority-vote root_id matches ``post_root_ids[syn_idx]``.

    Majority-vote root_id is computed from the synapse indices already
    assigned to each neuron (``neuron.synapse_indices``).  Neurons with no
    assigned synapses receive no majority root, so their edges are labelled 0.

    This labelling is the direct per-edge analogue of the line-graph F1 metric
    and is used to supervise the GAT edge scorer during training.

    Parameters
    ----------
    graph:
        ConnectivityGraph from ``_build_graph``.
    pre_root_ids, post_root_ids:
        Int64 arrays ``[N_synapses]`` of ground-truth root IDs.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``[len(graph.edges)]``.  Values are 0.0 or 1.0.
    """
    # Compute majority root_id per neuron (pre and post sides separately).
    neuron_majority_pre: dict[int, int] = {}
    neuron_majority_post: dict[int, int] = {}

    for nid, neuron in graph.neurons.items():
        syn_idxs = neuron.synapse_indices
        if not syn_idxs:
            continue
        if neuron.role == "pre":
            counts = Counter(int(pre_root_ids[i]) for i in syn_idxs)
            neuron_majority_pre[nid] = counts.most_common(1)[0][0]
        elif neuron.role == "post":
            counts = Counter(int(post_root_ids[i]) for i in syn_idxs)
            neuron_majority_post[nid] = counts.most_common(1)[0][0]
        else:
            # "mixed" role neurons: check both sides independently.
            pre_counts = Counter(int(pre_root_ids[i]) for i in syn_idxs)
            post_counts = Counter(int(post_root_ids[i]) for i in syn_idxs)
            if pre_counts:
                neuron_majority_pre[nid] = pre_counts.most_common(1)[0][0]
            if post_counts:
                neuron_majority_post[nid] = post_counts.most_common(1)[0][0]

    labels = np.zeros(len(graph.edges), dtype=np.float32)
    for i, (pre_nid, post_nid, syn_idx) in enumerate(graph.edges):
        pre_correct = (
            neuron_majority_pre.get(pre_nid) == int(pre_root_ids[syn_idx])
        )
        post_correct = (
            neuron_majority_post.get(post_nid) == int(post_root_ids[syn_idx])
        )
        if pre_correct and post_correct:
            labels[i] = 1.0
    return labels


def gat_refine_connectivity(
    graph: "ConnectivityGraph",
    path_encoder,
    gat_model,
    *,
    threshold: float = 0.5,
) -> "ConnectivityGraph":
    """Re-score connectivity edges using a ``GlobalAssemblyGAT``.

    The GAT sees all fragment nodes simultaneously, so each fragment's merge
    decision is informed by the global topology of the box — not just its
    local pairwise grammar score.

    Pipeline
    --------
    1. Encode every ``MergedNeuron``'s path_points through ``path_encoder``.
    2. Build a bidirectional edge list (+ self-loops) from the existing
       synapse edges in ``graph``.
    3. Run ``gat_model.forward(x, src, dst)`` to get globally-aware
       embeddings ``h``.
    4. Score each *synapse* edge ``(pre_neuron, post_neuron)`` with
       ``gat_model.score_edges(h, src, dst)``.
    5. Retain only edges whose sigmoid score ≥ ``threshold``; drop the rest
       into ``unresolved_synapse_indices``.

    Parameters
    ----------
    graph:
        ConnectivityGraph produced by ``_build_graph`` (local merge pass).
    path_encoder:
        A ``TorchPathEncoder`` (or any module accepting ``(x, mask)`` and
        returning ``[N, embedding_dim]``).
    gat_model:
        A ``GlobalAssemblyGAT`` module.
    threshold:
        Sigmoid-probability threshold for keeping an edge.  Edges with
        score < threshold are moved to ``unresolved_synapse_indices``.

    Returns
    -------
    ConnectivityGraph
        A new graph with refined edges and updated unresolved list.
        ``neurons`` is unchanged — the GAT refines *connections*, not the
        fragment identity.
    """
    import torch
    import torch.nn.functional as F
    from .merge import ConnectivityGraph

    if not graph.neurons or not graph.edges:
        return graph

    gat_model.eval()
    path_encoder.eval()

    # 1. Encode node features.
    node_ids, h = _encode_neurons(graph.neurons, path_encoder)
    if len(node_ids) == 0:
        return graph

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # 2. Build GAT edges (self-loops + bidirectional synapse edges).
    src, dst, _ = _build_gat_edges(node_ids, graph)

    # 3. GAT message passing.
    with torch.no_grad():
        h_gat = gat_model(h, src, dst)  # [N, gat_dim]

    # 4. Score each synapse edge.
    refined_edges = []
    unresolved = list(graph.unresolved_synapse_indices)

    for pre_nid, post_nid, syn_idx in graph.edges:
        src_i = id_to_idx.get(pre_nid)
        dst_i = id_to_idx.get(post_nid)
        if src_i is None or dst_i is None:
            unresolved.append(syn_idx)
            continue

        with torch.no_grad():
            edge_src = torch.tensor([src_i], dtype=torch.long)
            edge_dst = torch.tensor([dst_i], dtype=torch.long)
            logit = gat_model.score_edges(h_gat, edge_src, edge_dst)
            prob = float(torch.sigmoid(logit).item())

        if prob >= threshold:
            refined_edges.append((pre_nid, post_nid, syn_idx))
        else:
            unresolved.append(syn_idx)

    return ConnectivityGraph(
        neurons=graph.neurons,
        edges=refined_edges,
        unresolved_synapse_indices=sorted(set(unresolved)),
        metadata=dict(graph.metadata),
    )
