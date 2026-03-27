"""Skeleton-backed graph construction with base-materialization leakage guards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fetch import (
    MICRONS_DATASTACK,
    CAVE_SERVER,
    MIP_VOXEL_SIZES,
    RealBoxSpec,
    SynapseTable,
    fetch_root_skeletons,
)
from .merge import ConnectivityGraph, MergedNeuron


@dataclass(frozen=True)
class SkeletonGraphConfig:
    """Configuration for base-materialization skeleton graph construction."""

    base_version: int
    target_version: int
    skeleton_version: int
    graph_source: str = "skeleton"


def validate_skeleton_graph_config(
    *,
    base_version: int,
    target_version: int,
    skeleton_version: int,
    graph_source: str,
) -> SkeletonGraphConfig:
    """Validate that skeleton-derived graph inputs do not leak target labels."""
    cfg = SkeletonGraphConfig(
        base_version=int(base_version),
        target_version=int(target_version),
        skeleton_version=int(skeleton_version),
        graph_source=str(graph_source),
    )
    if cfg.graph_source == "skeleton" and cfg.skeleton_version != cfg.base_version:
        raise ValueError(
            "graph_source='skeleton' requires skeleton_version == base_version "
            f"to avoid leakage (got base_version={cfg.base_version}, "
            f"skeleton_version={cfg.skeleton_version}, target_version={cfg.target_version})."
        )
    return cfg


def _globalize_points(points_local: np.ndarray, box: RealBoxSpec) -> np.ndarray:
    vox = np.array(MIP_VOXEL_SIZES[box.mip], dtype=np.float32)
    origin_nm = np.array(box.bbox_nm[0], dtype=np.float32)
    return points_local.astype(np.float32) * vox[None, :] + origin_nm[None, :]


def _connected_components(points: np.ndarray, radius_nm: float) -> list[list[int]]:
    if len(points) == 0:
        return []
    parent = list(range(len(points)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for i in range(len(points)):
        dist = np.linalg.norm(points[i + 1 :] - points[i], axis=1)
        for off in np.flatnonzero(dist <= radius_nm).tolist():
            union(i, i + 1 + off)

    groups: dict[int, list[int]] = {}
    for i in range(len(points)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _ordered_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= 2:
        return pts
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    order = np.argsort(centered @ axis)
    return pts[order]


def _fragment_from_skeleton(vertices_nm: np.ndarray, anchor_points_nm: np.ndarray, *, max_vertices: int = 32) -> np.ndarray:
    if len(vertices_nm) < 2:
        return _ordered_points(anchor_points_nm)
    centroid = anchor_points_nm.mean(axis=0)
    dist = np.linalg.norm(vertices_nm - centroid, axis=1)
    take = max(4, min(int(max_vertices), len(vertices_nm)))
    chosen = vertices_nm[np.argsort(dist)[:take]]
    return _ordered_points(chosen)


def _build_role_neurons(
    *,
    role_name: str,
    points_nm: np.ndarray,
    root_ids: np.ndarray,
    skeletons_by_root: dict[int, object],
    start_neuron_id: int,
    fragment_radius_nm: float,
) -> tuple[dict[int, MergedNeuron], dict[int, int], dict[int, np.ndarray], int]:
    neurons: dict[int, MergedNeuron] = {}
    synapse_to_neuron: dict[int, int] = {}
    centroids: dict[int, np.ndarray] = {}
    next_id = int(start_neuron_id)

    for root_id in sorted({int(r) for r in root_ids.tolist() if int(r) > 0}):
        syn_idx = np.flatnonzero(root_ids == root_id)
        if len(syn_idx) == 0:
            continue
        clusters = _connected_components(points_nm[syn_idx], radius_nm=fragment_radius_nm)
        skel = skeletons_by_root.get(root_id)
        vertices_nm = getattr(skel, "vertices", np.zeros((0, 3), dtype=np.float32))

        for cluster_local in clusters:
            cluster_idx = syn_idx[np.asarray(cluster_local, dtype=np.int64)]
            anchor_points = points_nm[cluster_idx]
            path_points = _fragment_from_skeleton(vertices_nm, anchor_points)
            if len(path_points) < 2:
                path_points = _ordered_points(anchor_points)
            neuron_id = next_id
            next_id += 1
            neurons[neuron_id] = MergedNeuron(
                neuron_id=neuron_id,
                agent_ids=[],
                path_points=path_points.astype(np.float32),
                synapse_indices=cluster_idx.astype(np.int64).tolist(),
                role=role_name,
            )
            centroids[neuron_id] = anchor_points.mean(axis=0).astype(np.float32)
            for syn in cluster_idx.tolist():
                synapse_to_neuron[int(syn)] = neuron_id

    return neurons, synapse_to_neuron, centroids, next_id


def _nearest_alternative(centroids: dict[int, np.ndarray], true_neuron_id: int, point_nm: np.ndarray) -> int | None:
    alternatives = [
        (neuron_id, float(np.linalg.norm(center - point_nm)))
        for neuron_id, center in centroids.items()
        if neuron_id != true_neuron_id
    ]
    if not alternatives:
        return None
    alternatives.sort(key=lambda item: item[1])
    return int(alternatives[0][0])


def build_skeleton_connectivity_graph(
    box: RealBoxSpec,
    synapses: SynapseTable,
    *,
    version: int,
    datastack: str = MICRONS_DATASTACK,
    cave_server: str = CAVE_SERVER,
    token: str | None = None,
    skeleton_service_version: int = 4,
    skeleton_cache_dir: str | None = None,
    fragment_radius_nm: float = 2_500.0,
    add_decoy_edges: bool = True,
) -> ConnectivityGraph:
    """Build a GAT-ready candidate graph from base-version skeleton fragments.

    The graph is intentionally permissive: in addition to the direct pre/post
    fragment edge for each synapse, we also add one nearby decoy alternative on
    each side when available so the downstream GAT sees negative candidates.
    """
    pre_points_nm = _globalize_points(synapses.pre_pt, box)
    post_points_nm = _globalize_points(synapses.post_pt, box)

    pre_skeletons = fetch_root_skeletons(
        synapses.pre_root_id,
        version=version,
        datastack=datastack,
        cave_server=cave_server,
        token=token,
        skeleton_service_version=skeleton_service_version,
        cache_dir=skeleton_cache_dir,
    )
    post_skeletons = fetch_root_skeletons(
        synapses.post_root_id,
        version=version,
        datastack=datastack,
        cave_server=cave_server,
        token=token,
        skeleton_service_version=skeleton_service_version,
        cache_dir=skeleton_cache_dir,
    )

    pre_neurons, pre_owner, pre_centroids, next_id = _build_role_neurons(
        role_name="pre",
        points_nm=pre_points_nm,
        root_ids=synapses.pre_root_id,
        skeletons_by_root=pre_skeletons,
        start_neuron_id=0,
        fragment_radius_nm=fragment_radius_nm,
    )
    post_neurons, post_owner, post_centroids, _ = _build_role_neurons(
        role_name="post",
        points_nm=post_points_nm,
        root_ids=synapses.post_root_id,
        skeletons_by_root=post_skeletons,
        start_neuron_id=next_id,
        fragment_radius_nm=fragment_radius_nm,
    )

    neurons = {**pre_neurons, **post_neurons}
    edge_set: set[tuple[int, int, int]] = set()
    unresolved: list[int] = []

    for syn_idx in range(len(synapses.synapse_id)):
        pre_nid = pre_owner.get(syn_idx)
        post_nid = post_owner.get(syn_idx)
        if pre_nid is None or post_nid is None:
            unresolved.append(int(syn_idx))
            continue
        edge_set.add((int(pre_nid), int(post_nid), int(syn_idx)))
        if add_decoy_edges:
            alt_pre = _nearest_alternative(pre_centroids, int(pre_nid), pre_points_nm[syn_idx])
            if alt_pre is not None:
                edge_set.add((int(alt_pre), int(post_nid), int(syn_idx)))
            alt_post = _nearest_alternative(post_centroids, int(post_nid), post_points_nm[syn_idx])
            if alt_post is not None:
                edge_set.add((int(pre_nid), int(alt_post), int(syn_idx)))

    return ConnectivityGraph(
        neurons=neurons,
        edges=sorted(edge_set),
        unresolved_synapse_indices=sorted(set(unresolved)),
    )
