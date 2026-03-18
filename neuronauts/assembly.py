"""Box-scale assembly search: beam search and GAT-based global refinement."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

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


# ---------------------------------------------------------------------------
# PR 4: Global GAT Assembly
# ---------------------------------------------------------------------------

# Isotropic scaling: convert MIP-2 voxel coords to 32-nm units (1 unit = 32 nm).
# Keeps feature values in the same numerical range as raw voxel coords (~1–60)
# while correctly weighting the Z axis at 40/32 = 1.25× relative to XY.
_PATH_ISO = np.array([1.0, 1.0, 40.0 / 32.0], dtype=np.float32)


def _path_seq_from_pts(points: np.ndarray) -> np.ndarray:
    """Convert an (K, 3) path-point array to a (K-1, 3) feature sequence.

    The three features per step are edge length (nm), radius from centroid
    (nm), and cumulative turning angle — matching ``_path_sequence_from_points``
    in ``run.py`` so that node features are compatible with the shared
    path encoder.  All length-based features are in physical nanometres so
    that Z-axis anisotropy (40 nm/vox vs 32 nm/vox in XY) does not distort
    the learned geometry.
    """
    if len(points) < 2:
        return np.zeros((0, 3), dtype=np.float32)
    pts_nm = points.astype(np.float32) * _PATH_ISO
    diffs = np.diff(pts_nm, axis=0)
    edge_len = np.linalg.norm(diffs, axis=1)
    centroid = pts_nm.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(pts_nm[1:] - centroid, axis=1)
    if len(pts_nm) < 3:
        curvature = np.zeros(len(edge_len), dtype=np.float32)
    else:
        unit = diffs / np.clip(np.linalg.norm(diffs, axis=1, keepdims=True), 1e-6, None)
        turn = np.linalg.norm(np.diff(unit, axis=0), axis=1)
        curvature = np.zeros(len(edge_len), dtype=np.float32)
        curvature[1:] = turn.astype(np.float32)
    return np.stack([edge_len, radius, curvature], axis=-1).astype(np.float32)


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

    seqs = [_path_seq_from_pts(neurons[nid].path_points) for nid in node_ids]
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
    )
