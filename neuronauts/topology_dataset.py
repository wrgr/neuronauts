"""Dataset construction for synapse-cluster atomicity learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from .fetch import RealBoxSpec, SynapseTable, fetch_synapses, fetch_volume, load_cached_membrane
from .fields import compute_membrane_field


@dataclass(frozen=True)
class ClusterExample:
    role: str
    label: int
    synapse_indices: tuple[int, ...]
    root_ids: tuple[int, ...]
    features: np.ndarray


FEATURE_NAMES = [
    "role_is_post",
    "cluster_size",
    "extent_x",
    "extent_y",
    "extent_z",
    "centroid_std_x",
    "centroid_std_y",
    "centroid_std_z",
    "pairwise_mean",
    "pairwise_max",
    "pairwise_std",
    "membrane_mean",
    "membrane_std",
]


def _sample_field(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    idx = np.clip(np.rint(points).astype(int), 0, np.array(field.shape) - 1)
    return field[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32)


def _cluster_features(role: str, points: np.ndarray, membrane_field: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    extent = pts.max(axis=0) - pts.min(axis=0)
    std = centered.std(axis=0)
    pairwise = pdist(pts) if len(pts) >= 2 else np.array([0.0], dtype=np.float32)
    membrane_vals = _sample_field(membrane_field, pts)
    return np.array(
        [
            1.0 if role == "post" else 0.0,
            float(len(pts)),
            float(extent[0]),
            float(extent[1]),
            float(extent[2]),
            float(std[0]),
            float(std[1]),
            float(std[2]),
            float(pairwise.mean()),
            float(pairwise.max()),
            float(pairwise.std()),
            float(membrane_vals.mean()),
            float(membrane_vals.std()),
        ],
        dtype=np.float32,
    )


def _root_groups(root_ids: np.ndarray) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for idx, root_id in enumerate(root_ids.tolist()):
        groups.setdefault(int(root_id), []).append(idx)
    return groups


def _atomic_examples_for_role(
    role: str,
    points: np.ndarray,
    root_ids: np.ndarray,
    membrane_field: np.ndarray,
    *,
    min_cluster_size: int,
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
                features=_cluster_features(role, cluster_points, membrane_field),
            )
        )
    return examples


def _non_atomic_examples_for_role(
    role: str,
    points: np.ndarray,
    root_ids: np.ndarray,
    membrane_field: np.ndarray,
    *,
    min_cluster_size: int,
    max_negatives: int,
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
    if max_negatives > 0:
        pairs = pairs[:max_negatives]

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
                features=_cluster_features(role, cluster_points, membrane_field),
            )
        )
    return examples


def build_cluster_examples(
    synapses: SynapseTable,
    membrane_field: np.ndarray,
    *,
    min_cluster_size: int = 2,
    max_negative_pairs_per_role: int = 32,
    seed: int = 42,
) -> list[ClusterExample]:
    rng = np.random.default_rng(seed)
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
                membrane_field,
                min_cluster_size=min_cluster_size,
            )
        )
        examples.extend(
            _non_atomic_examples_for_role(
                role,
                points,
                root_ids,
                membrane_field,
                min_cluster_size=min_cluster_size,
                max_negatives=max_negative_pairs_per_role,
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
        seed=seed,
    )


def examples_to_arrays(examples: list[ClusterExample]) -> tuple[np.ndarray, np.ndarray]:
    if not examples:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    x = np.stack([example.features for example in examples], axis=0).astype(np.float32)
    y = np.array([example.label for example in examples], dtype=np.int64)
    return x, y


def save_examples_npz(path: str | Path, examples: list[ClusterExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x, y = examples_to_arrays(examples)
    roles = np.array([example.role for example in examples], dtype=object)
    synapse_indices = np.array([np.array(example.synapse_indices, dtype=np.int64) for example in examples], dtype=object)
    root_ids = np.array([np.array(example.root_ids, dtype=np.int64) for example in examples], dtype=object)
    np.savez(
        path,
        x=x,
        y=y,
        roles=roles,
        synapse_indices=synapse_indices,
        root_ids=root_ids,
        feature_names=np.array(FEATURE_NAMES, dtype=object),
    )
