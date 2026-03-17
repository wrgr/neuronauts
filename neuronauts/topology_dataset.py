"""Dataset construction for multi-branch synapse-cluster atomicity learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .fetch import RealBoxSpec, SynapseTable, fetch_synapses, fetch_volume, load_cached_membrane
from .fields import compute_membrane_field
from .grammar import PathEncoder, build_path_batch

BRANCH_FEATURE_NAME = "branch_embedding"


@dataclass(frozen=True)
class ClusterExample:
    role: str
    label: int
    synapse_indices: tuple[int, ...]
    root_ids: tuple[int, ...]
    branch_embeddings: tuple[np.ndarray, ...]


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


def _branch_point_splits(points: np.ndarray, max_branches: int) -> list[np.ndarray]:
    ordered = _ordered_points(points)
    if len(ordered) < 2:
        return []

    branch_count = min(max_branches, max(1, len(ordered) // 2))
    parts = np.array_split(ordered, branch_count, axis=0)
    return [part.astype(np.float32, copy=False) for part in parts if len(part) >= 2]


def _encode_branch(points: np.ndarray, encoder: PathEncoder) -> np.ndarray:
    ordered = _ordered_points(points)
    diffs = np.diff(ordered, axis=0)
    edge_len = np.linalg.norm(diffs, axis=1).astype(np.float32, copy=False)
    centroid = ordered.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(ordered[1:] - centroid, axis=1).astype(np.float32, copy=False)
    curvature = _curvature_from_points(ordered)
    batch = build_path_batch(edge_len=edge_len, radius=radius, curvature=curvature)
    return encoder.encode(batch)


def _cluster_branch_embeddings(
    points: np.ndarray,
    *,
    encoder: PathEncoder,
    max_branches: int,
) -> tuple[np.ndarray, ...]:
    branches = []
    for branch_points in _branch_point_splits(points, max_branches=max_branches):
        branches.append(_encode_branch(branch_points, encoder))

    if not branches and len(points) >= 2:
        branches.append(_encode_branch(points, encoder))
    return tuple(np.asarray(branch, dtype=np.float32) for branch in branches)


def _atomic_examples_for_role(
    role: str,
    points: np.ndarray,
    root_ids: np.ndarray,
    *,
    encoder: PathEncoder,
    min_cluster_size: int,
    max_branches: int,
) -> list[ClusterExample]:
    examples = []
    for root_id, indices in _root_groups(root_ids).items():
        if len(indices) < min_cluster_size:
            continue
        cluster_points = points[indices]
        examples.append(
            ClusterExample(
                role=role,
                label=1,
                synapse_indices=tuple(int(i) for i in indices),
                root_ids=(int(root_id),),
                branch_embeddings=_cluster_branch_embeddings(
                    cluster_points,
                    encoder=encoder,
                    max_branches=max_branches,
                ),
            )
        )
    return examples


def _non_atomic_examples_for_role(
    role: str,
    points: np.ndarray,
    root_ids: np.ndarray,
    *,
    encoder: PathEncoder,
    min_cluster_size: int,
    max_negative_pairs_per_role: int,
    max_branches: int,
    rng: np.random.Generator,
) -> list[ClusterExample]:
    groups = [(root_id, indices) for root_id, indices in _root_groups(root_ids).items() if len(indices) >= min_cluster_size]
    if len(groups) < 2:
        return []

    centroids = {int(root_id): points[indices].mean(axis=0) for root_id, indices in groups}
    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            root_a, idx_a = groups[i]
            root_b, idx_b = groups[j]
            dist = float(np.linalg.norm(centroids[int(root_a)] - centroids[int(root_b)]))
            pairs.append((dist, int(root_a), int(root_b), idx_a, idx_b))
    pairs.sort(key=lambda item: item[0])
    if max_negative_pairs_per_role > 0:
        pairs = pairs[:max_negative_pairs_per_role]

    examples = []
    for _, root_a, root_b, idx_a, idx_b in pairs:
        merged = list(idx_a) + list(idx_b)
        rng.shuffle(merged)
        cluster_points = points[merged]
        examples.append(
            ClusterExample(
                role=role,
                label=0,
                synapse_indices=tuple(int(i) for i in merged),
                root_ids=(int(root_a), int(root_b)),
                branch_embeddings=_cluster_branch_embeddings(
                    cluster_points,
                    encoder=encoder,
                    max_branches=max_branches,
                ),
            )
        )
    return examples


def build_cluster_examples(
    synapses: SynapseTable,
    membrane_field: np.ndarray,
    *,
    min_cluster_size: int = 2,
    max_negative_pairs_per_role: int = 32,
    max_branches: int = 32,
    seed: int = 42,
) -> list[ClusterExample]:
    del membrane_field
    rng = np.random.default_rng(seed)
    encoder = PathEncoder(output_dim=32)
    examples: list[ClusterExample] = []
    role_specs = [
        ("pre", synapses.pre_pt, synapses.pre_root_id),
        ("post", synapses.post_pt, synapses.post_root_id),
    ]
    for role, points, root_ids in role_specs:
        examples.extend(
            _atomic_examples_for_role(
                role,
                points,
                root_ids,
                encoder=encoder,
                min_cluster_size=min_cluster_size,
                max_branches=max_branches,
            )
        )
        examples.extend(
            _non_atomic_examples_for_role(
                role,
                points,
                root_ids,
                encoder=encoder,
                min_cluster_size=min_cluster_size,
                max_negative_pairs_per_role=max_negative_pairs_per_role,
                max_branches=max_branches,
                rng=rng,
            )
        )
    return examples


def build_cluster_examples_for_box(
    box: RealBoxSpec,
    *,
    membrane_source: str = "auto",
    membrane_cache_dir: str = "cache/membranes",
    min_cluster_size: int = 2,
    max_negative_pairs_per_role: int = 32,
    max_branches: int = 32,
    seed: int = 42,
) -> list[ClusterExample]:
    synapses = fetch_synapses(box.bbox_nm, mip=box.mip)
    chunk = fetch_volume(box.bbox_nm, mip=box.mip)
    membrane = None
    if membrane_source in {"auto", "cache"}:
        membrane = load_cached_membrane(box, membrane_cache_dir)
        if membrane is None and membrane_source == "cache":
            raise FileNotFoundError(f"missing cached membrane for {box.center_nm} in {membrane_cache_dir}")
    if membrane is None:
        membrane = compute_membrane_field(chunk.data)
    return build_cluster_examples(
        synapses,
        membrane,
        min_cluster_size=min_cluster_size,
        max_negative_pairs_per_role=max_negative_pairs_per_role,
        max_branches=max_branches,
        seed=seed,
    )


def examples_to_multi_branch_arrays(
    examples: list[ClusterExample],
    *,
    max_branches: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not examples:
        return (
            np.zeros((0, max_branches, 32), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, max_branches), dtype=bool),
        )

    embed_dim = max((embedding.shape[0] for example in examples for embedding in example.branch_embeddings), default=32)
    x = np.zeros((len(examples), max_branches, embed_dim), dtype=np.float32)
    y = np.zeros((len(examples),), dtype=np.int64)
    mask = np.ones((len(examples), max_branches), dtype=bool)

    for i, example in enumerate(examples):
        y[i] = int(example.label)
        for branch_idx, embedding in enumerate(example.branch_embeddings[:max_branches]):
            x[i, branch_idx, : embedding.shape[0]] = embedding.astype(np.float32, copy=False)
            mask[i, branch_idx] = False

    return x, y, mask


def save_multi_branch_npz(
    path: str | Path,
    examples: list[ClusterExample],
    *,
    max_branches: int = 32,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x, y, mask = examples_to_multi_branch_arrays(examples, max_branches=max_branches)
    roles = np.array([example.role for example in examples], dtype=object)
    synapse_indices = np.array([np.array(example.synapse_indices, dtype=np.int64) for example in examples], dtype=object)
    root_ids = np.array([np.array(example.root_ids, dtype=np.int64) for example in examples], dtype=object)
    np.savez(
        path,
        x=x,
        y=y,
        mask=mask,
        roles=roles,
        synapse_indices=synapse_indices,
        root_ids=root_ids,
        feature_names=np.array([BRANCH_FEATURE_NAME], dtype=object),
    )
