"""Label-blind candidate panels from real L2-coordinate geometry."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
import numpy as np

from ._scipy_compat import cKDTree
from .real_dense_soma import Fragment, endpoint_paths


@dataclass(frozen=True)
class EndpointRecord:
    root_id: int
    path_nm: np.ndarray
    outward_tangent: np.ndarray


@dataclass(frozen=True)
class EndpointPair:
    left_root: int
    right_root: int
    distance_nm: float
    facing: float
    tangent_opposition: float
    left_path_nm: np.ndarray
    right_path_nm: np.ndarray


def endpoint_records(fragments: list[Fragment], *, max_paths: int = 8,
                     max_path_points: int = 64) -> list[EndpointRecord]:
    records = []
    for fragment in fragments:
        for path in endpoint_paths(fragment, max_paths=max_paths,
                                   max_points=max_path_points):
            tangent = np.asarray(path[0] - path[1], dtype=np.float32)
            norm = float(np.linalg.norm(tangent))
            if norm > 0:
                records.append(EndpointRecord(
                    int(fragment.root_id), np.asarray(path, dtype=np.float32),
                    tangent / norm))
    return records


def candidate_endpoint_pairs(records: list[EndpointRecord], *,
                             max_distance_nm: float) -> list[EndpointPair]:
    if not records:
        return []
    xyz = np.stack([record.path_nm[0] for record in records])
    raw = cKDTree(xyz).query_pairs(float(max_distance_nm), output_type="set")
    best = {}
    for i, j in sorted(raw):
        left, right = records[i], records[j]
        if left.root_id == right.root_id:
            continue
        gap = right.path_nm[0] - left.path_nm[0]
        distance = float(np.linalg.norm(gap))
        if distance <= 0:
            continue
        direction = gap / distance
        facing = min(float(np.dot(left.outward_tangent, direction)),
                     float(np.dot(right.outward_tangent, -direction)))
        opposition = float(np.dot(left.outward_tangent,
                                  -right.outward_tangent))
        key = tuple(sorted((left.root_id, right.root_id)))
        pair = EndpointPair(left.root_id, right.root_id, distance, facing,
                            opposition, left.path_nm, right.path_nm)
        current = best.get(key)
        if (current is None or pair.facing > current.facing or
                (pair.facing == current.facing and
                 pair.distance_nm < current.distance_nm)):
            best[key] = pair
    return sorted(best.values(), key=lambda p: (-p.facing, p.distance_nm,
                                                min(p.left_root, p.right_root),
                                                max(p.left_root, p.right_root)))


def filter_candidate_pairs(pairs: list[EndpointPair], *,
                           max_distance_nm: float,
                           cone_degrees: float) -> list[EndpointPair]:
    minimum = cos(radians(float(cone_degrees)))
    return [pair for pair in pairs
            if pair.distance_nm <= float(max_distance_nm)
            and pair.facing >= minimum]


def panel_sizes(root_ids: list[int], pairs: list[EndpointPair]) -> np.ndarray:
    panels = {int(root): set() for root in root_ids}
    for pair in pairs:
        if pair.left_root in panels and pair.right_root in panels:
            panels[pair.left_root].add(pair.right_root)
            panels[pair.right_root].add(pair.left_root)
    return np.asarray([len(panels[int(root)]) for root in root_ids],
                      dtype=np.int64)
