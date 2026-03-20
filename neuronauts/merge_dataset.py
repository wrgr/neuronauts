"""Dataset construction for local merge supervision."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fetch import RealBoxSpec, SynapseTable, fetch_synapses
from .grammar import (
    DEFAULT_PATH_FEATURE_MODE,
    featurize_path_points,
    path_feature_names,
)
from .training_batches import pad_path_sequences

MERGE_FEATURE_NAMES = path_feature_names(DEFAULT_PATH_FEATURE_MODE)


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


def _sequence_from_points(points: np.ndarray) -> np.ndarray:
    ordered = _ordered_points(points)
    return featurize_path_points(ordered, mode=DEFAULT_PATH_FEATURE_MODE)


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
    path_feature_mode: str = DEFAULT_PATH_FEATURE_MODE,
    min_fragment_size: int = 2,
    max_negative_pairs_per_role: int | None = None,
    max_positives_per_role: int = 512,
    max_roots_for_negatives: int = 100,
    seed: int = 0,
) -> list[MergeExample]:
    """Build merge (positive) and non-merge (negative) synapse-pair examples.

    Parameters
    ----------
    max_positives_per_role:
        Cap on same-root positive examples per role (pre/post) before
        generating negatives.  Prevents wasted work in large boxes with
        thousands of root-id groups.
    max_roots_for_negatives:
        Cap the candidate-group pool for negative-pair generation to this
        many randomly-sampled root groups before the O(n²) distance sort.
        Keeps negative generation O(max_roots²) regardless of box size.
        With 100 groups the loop runs ≤ 4,950 iterations instead of millions.
    """
    rng = np.random.default_rng(seed)
    examples: list[MergeExample] = []
    role_specs = [
        ("pre", synapses.pre_pt, synapses.pre_root_id),
        ("post", synapses.post_pt, synapses.post_root_id),
    ]

    for role, points, root_ids in role_specs:
        groups = _root_groups(root_ids)
        group_items = list(groups.items())

        # Positives: one split per same-root cluster, capped early.
        n_pos = 0
        for root_id, indices in group_items:
            if n_pos >= max_positives_per_role:
                break
            if len(indices) < max(4, min_fragment_size * 2):
                continue
            split = _split_same_root(indices, points)
            if split is None:
                continue
            left_idx, right_idx = split
            left_seq = featurize_path_points(_ordered_points(points[left_idx]), mode=path_feature_mode)
            right_seq = featurize_path_points(_ordered_points(points[right_idx]), mode=path_feature_mode)
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
            n_pos += 1

        # Negatives: nearby but distinct rooted clusters.
        #
        # IMPORTANT: keep negative sampling roughly balanced with positives.
        # A hard-coded small cap (e.g. 32) yields a degenerate dataset for large
        # boxes (hundreds of positives per role but only a few negatives), which
        # can produce misleadingly perfect accuracy/BCE.
        neg_cap = max_negative_pairs_per_role
        if neg_cap is None:
            neg_cap = max(1, n_pos)
        # Subsample candidate groups first to avoid O(n_roots²) blowup in
        # large boxes.  Hard negatives (closest centroids) are still preferred
        # within the subsample.
        candidate_groups = [
            (int(root_id), indices)
            for root_id, indices in group_items
            if len(indices) >= min_fragment_size
        ]
        if len(candidate_groups) > max_roots_for_negatives:
            chosen = rng.choice(
                len(candidate_groups), size=max_roots_for_negatives, replace=False
            )
            candidate_groups = [candidate_groups[i] for i in chosen]

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
        if neg_cap > 0:
            pairs = pairs[:neg_cap]

        for _, root_a, root_b, idx_a, idx_b in pairs:
            left_idx = idx_a[: max(min_fragment_size, min(len(idx_a), len(idx_a)))]
            right_idx = idx_b[: max(min_fragment_size, min(len(idx_b), len(idx_b)))]
            left_seq = featurize_path_points(_ordered_points(points[left_idx]), mode=path_feature_mode)
            right_seq = featurize_path_points(_ordered_points(points[right_idx]), mode=path_feature_mode)
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
    path_feature_mode: str = DEFAULT_PATH_FEATURE_MODE,
    min_fragment_size: int = 2,
    max_negative_pairs_per_role: int | None = None,
) -> list[MergeExample]:
    synapses = fetch_synapses(box.bbox_nm, mip=box.mip)
    return build_merge_examples(
        synapses,
        path_feature_mode=path_feature_mode,
        min_fragment_size=min_fragment_size,
        max_negative_pairs_per_role=max_negative_pairs_per_role,
    )


def examples_to_arrays(
    examples: list[MergeExample],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_dim = (
        int(examples[0].left_sequence.shape[1])
        if examples and examples[0].left_sequence.ndim == 2
        else 3
    )
    left_batch = pad_path_sequences([example.left_sequence for example in examples], feature_dim=feature_dim)
    right_batch = pad_path_sequences([example.right_sequence for example in examples], feature_dim=feature_dim)
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
    feature_names = (
        np.array(path_feature_names(DEFAULT_PATH_FEATURE_MODE), dtype=object)
        if not examples
        else np.array(
            path_feature_names(DEFAULT_PATH_FEATURE_MODE)
            if examples[0].left_sequence.shape[1] == len(path_feature_names(DEFAULT_PATH_FEATURE_MODE))
            else [f"f{i}" for i in range(examples[0].left_sequence.shape[1])],
            dtype=object,
        )
    )
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
        feature_names=feature_names,
    )
