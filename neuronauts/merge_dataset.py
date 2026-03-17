"""Dataset construction for local merge supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fetch import RealBoxSpec, SynapseTable, fetch_synapses
from .training_batches import pad_path_sequences

MERGE_FEATURE_NAMES = ("edge_len", "radius", "curvature")


@dataclass(frozen=True)
class MergeExample:
    role: str
    label: int
    left_synapse_indices: tuple[int, ...]
    right_synapse_indices: tuple[int, ...]
    left_root_ids: tuple[int, ...]
    right_root_ids: tuple[int, ...]
    left_sequence: np.ndarray
    right_sequence: np.ndarray


def _root_groups(root_ids: np.ndarray) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, root_id in enumerate(root_ids.tolist()):
        groups.setdefault(int(root_id), []).append(idx)
    return groups


def _ordered_points(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return points.astype(np.float32, copy=False)
    pts = points.astype(np.float32)
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    primary_axis = vh[0]
    order = np.argsort(centered @ primary_axis)
    return pts[order].astype(np.float32, copy=False)


def _curvature_from_points(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(max(0, len(points) - 1), dtype=np.float32)
    segments = np.diff(points, axis=0).astype(np.float32)
    seg_norm = np.linalg.norm(segments, axis=1)
    unit = segments / np.clip(seg_norm[:, None], 1e-6, None)
    turn = np.linalg.norm(np.diff(unit, axis=0), axis=1)
    curvature = np.zeros(len(segments), dtype=np.float32)
    curvature[1:] = turn.astype(np.float32, copy=False)
    return curvature


def _sequence_from_points(points: np.ndarray) -> np.ndarray:
    ordered = _ordered_points(points)
    if len(ordered) < 2:
        return np.zeros((0, 3), dtype=np.float32)
    diffs = np.diff(ordered, axis=0)
    edge_len = np.linalg.norm(diffs, axis=1).astype(np.float32, copy=False)
    centroid = ordered.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(ordered[1:] - centroid, axis=1).astype(np.float32, copy=False)
    curvature = _curvature_from_points(ordered)
    return np.stack([edge_len, radius, curvature], axis=-1).astype(np.float32, copy=False)


def _split_same_root(indices: list[int], points: np.ndarray) -> tuple[list[int], list[int]] | None:
    if len(indices) < 4:
        return None
    ordered = _ordered_points(points[indices])
    midpoint = len(ordered) // 2
    if midpoint < 2 or (len(ordered) - midpoint) < 2:
        return None

    centered = points[indices].astype(np.float32) - points[indices].astype(np.float32).mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    primary_axis = vh[0]
    ranked = np.argsort(centered @ primary_axis)
    left = [indices[i] for i in ranked[:midpoint].tolist()]
    right = [indices[i] for i in ranked[midpoint:].tolist()]
    return left, right


def build_merge_examples(
    synapses: SynapseTable,
    *,
    min_fragment_size: int = 2,
    max_negative_pairs_per_role: int = 32,
) -> list[MergeExample]:
    examples: list[MergeExample] = []
    role_specs = [
        ("pre", synapses.pre_pt, synapses.pre_root_id),
        ("post", synapses.post_pt, synapses.post_root_id),
    ]

    for role, points, root_ids in role_specs:
        groups = _root_groups(root_ids)

        # Positives: two subfragments from the same rooted cluster.
        for root_id, indices in groups.items():
            if len(indices) < max(4, min_fragment_size * 2):
                continue
            split = _split_same_root(indices, points)
            if split is None:
                continue
            left_idx, right_idx = split
            left_seq = _sequence_from_points(points[left_idx])
            right_seq = _sequence_from_points(points[right_idx])
            if len(left_seq) == 0 or len(right_seq) == 0:
                continue
            examples.append(
                MergeExample(
                    role=role,
                    label=1,
                    left_synapse_indices=tuple(int(i) for i in left_idx),
                    right_synapse_indices=tuple(int(i) for i in right_idx),
                    left_root_ids=(int(root_id),),
                    right_root_ids=(int(root_id),),
                    left_sequence=left_seq,
                    right_sequence=right_seq,
                )
            )

        # Negatives: nearby but distinct rooted clusters.
        candidate_groups = [(int(root_id), indices) for root_id, indices in groups.items() if len(indices) >= min_fragment_size]
        pairs = []
        for i in range(len(candidate_groups)):
            for j in range(i + 1, len(candidate_groups)):
                root_a, idx_a = candidate_groups[i]
                root_b, idx_b = candidate_groups[j]
                centroid_a = points[idx_a].mean(axis=0)
                centroid_b = points[idx_b].mean(axis=0)
                dist = float(np.linalg.norm(centroid_a - centroid_b))
                pairs.append((dist, root_a, root_b, idx_a, idx_b))
        pairs.sort(key=lambda item: item[0])
        if max_negative_pairs_per_role > 0:
            pairs = pairs[:max_negative_pairs_per_role]

        for _, root_a, root_b, idx_a, idx_b in pairs:
            left_idx = idx_a[: max(min_fragment_size, min(len(idx_a), len(idx_a)))]
            right_idx = idx_b[: max(min_fragment_size, min(len(idx_b), len(idx_b)))]
            left_seq = _sequence_from_points(points[left_idx])
            right_seq = _sequence_from_points(points[right_idx])
            if len(left_seq) == 0 or len(right_seq) == 0:
                continue
            examples.append(
                MergeExample(
                    role=role,
                    label=0,
                    left_synapse_indices=tuple(int(i) for i in left_idx),
                    right_synapse_indices=tuple(int(i) for i in right_idx),
                    left_root_ids=(int(root_a),),
                    right_root_ids=(int(root_b),),
                    left_sequence=left_seq,
                    right_sequence=right_seq,
                )
            )

    return examples


def build_merge_examples_for_box(
    box: RealBoxSpec,
    *,
    min_fragment_size: int = 2,
    max_negative_pairs_per_role: int = 32,
) -> list[MergeExample]:
    synapses = fetch_synapses(box.bbox_nm, mip=box.mip)
    return build_merge_examples(
        synapses,
        min_fragment_size=min_fragment_size,
        max_negative_pairs_per_role=max_negative_pairs_per_role,
    )


def examples_to_arrays(
    examples: list[MergeExample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_batch = pad_path_sequences([example.left_sequence for example in examples], feature_dim=3)
    right_batch = pad_path_sequences([example.right_sequence for example in examples], feature_dim=3)
    y = np.array([example.label for example in examples], dtype=np.int64)
    return left_batch.x, left_batch.mask, right_batch.x, right_batch.mask, y


def save_merge_examples_npz(path: str | Path, examples: list[MergeExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    left_x, left_mask, right_x, right_mask, y = examples_to_arrays(examples)
    roles = np.array([example.role for example in examples], dtype=object)
    left_synapse_indices = np.array([np.array(example.left_synapse_indices, dtype=np.int64) for example in examples], dtype=object)
    right_synapse_indices = np.array([np.array(example.right_synapse_indices, dtype=np.int64) for example in examples], dtype=object)
    left_root_ids = np.array([np.array(example.left_root_ids, dtype=np.int64) for example in examples], dtype=object)
    right_root_ids = np.array([np.array(example.right_root_ids, dtype=np.int64) for example in examples], dtype=object)
    np.savez(
        path,
        left_x=left_x,
        left_mask=left_mask,
        right_x=right_x,
        right_mask=right_mask,
        y=y,
        roles=roles,
        left_synapse_indices=left_synapse_indices,
        right_synapse_indices=right_synapse_indices,
        left_root_ids=left_root_ids,
        right_root_ids=right_root_ids,
        feature_names=np.array(MERGE_FEATURE_NAMES, dtype=object),
    )
