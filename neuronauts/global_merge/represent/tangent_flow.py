"""
Directional Tangent-Flow & Ray-Casting Collinearity Engine.
Enables long-range bridging across anisotropic missing sections without requiring
individual global identity embeddings on fine fragments.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import KDTree

from neuronauts.global_merge.schemas import EndpointTangent, SegmentFragment, AssemblyEdge, EdgeType


def extract_endpoints_from_skeleton(
    fragment_id: str,
    vertices_nm: np.ndarray,
    radii_nm: np.ndarray,
    edges: np.ndarray,
    tangent_lookback_hops: int = 3
) -> List[EndpointTangent]:
    """
    Extract leaf vertices (degree == 1) from a skeleton tree and compute their
    outward-pointing 3D unit tangent vectors and local caliber.
    """
    n_verts = len(vertices_nm)
    if n_verts == 0:
        return []
    if n_verts == 1:
        # Isolated point: no defined tangent
        return [EndpointTangent(
            fragment_id=fragment_id,
            vertex_idx=0,
            coord_nm=vertices_nm[0],
            tangent=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            radius_nm=float(radii_nm[0]) if len(radii_nm) > 0 else 50.0,
            curvature=0.0
        )]

    # Build adjacency list
    adj: Dict[int, List[int]] = {i: [] for i in range(n_verts)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    endpoints: List[EndpointTangent] = []

    for v_idx, neighbors in adj.items():
        if len(neighbors) == 1:  # Leaf node
            # Walk inward up to  to estimate robust tangent
            curr = v_idx
            prev = None
            path = [curr]
            
            for _ in range(tangent_lookback_hops):
                nbrs = [n for n in adj[curr] if n != prev]
                if not nbrs:
                    break
                next_node = nbrs[0]
                prev = curr
                curr = next_node
                path.append(curr)
                if len(adj[curr]) > 2:  # Reached a branch point
                    break

            # Tangent points outward from inner path to leaf
            if len(path) >= 2:
                inward_pt = vertices_nm[path[-1]]
                leaf_pt = vertices_nm[v_idx]
                tangent_vec = leaf_pt - inward_pt
                dist = np.linalg.norm(tangent_vec)
                if dist > 1e-3:
                    unit_tangent = tangent_vec / dist
                else:
                    unit_tangent = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                unit_tangent = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            endpoints.append(EndpointTangent(
                fragment_id=fragment_id,
                vertex_idx=v_idx,
                coord_nm=vertices_nm[v_idx].astype(np.float32),
                tangent=unit_tangent.astype(np.float32),
                radius_nm=float(radii_nm[v_idx]) if v_idx < len(radii_nm) else 50.0,
                curvature=0.0
            ))

    return endpoints


def compute_collinearity(
    ep1: EndpointTangent,
    ep2: EndpointTangent,
    sigma_dist_nm: float = 15000.0,
    sigma_radius_nm: float = 150.0
) -> float:
    """
    Compute bidirectional ray-casting collinearity score in [0.0, 1.0].
    Measures whether endpoint 1 points at endpoint 2 AND endpoint 2 points back at endpoint 1.
    """
    disp = ep2.coord_nm - ep1.coord_nm
    dist = float(np.linalg.norm(disp))
    if dist < 1e-4:
        return 1.0  # Coincident endpoints

    unit_disp = disp / dist
    
    # Cosine alignment of ep1 tangent with ray towards ep2
    cos_1 = float(np.dot(ep1.tangent, unit_disp))
    # Cosine alignment of ep2 tangent with ray towards ep1
    cos_2 = float(np.dot(ep2.tangent, -unit_disp))

    # Both must point towards each other (positive cosine)
    if cos_1 <= 0.0 or cos_2 <= 0.0:
        return 0.0

    angular_score = cos_1 * cos_2
    dist_decay = math.exp(-dist / sigma_dist_nm)
    radius_match = math.exp(-abs(ep1.radius_nm - ep2.radius_nm) / sigma_radius_nm)

    score = angular_score * dist_decay * radius_match
    return float(np.clip(score, 0.0, 1.0))


def find_tangent_flow_bridges(
    fragments: List[SegmentFragment],
    max_distance_nm: float = 25000.0,
    min_collinearity: float = 0.25
) -> List[AssemblyEdge]:
    """
    Fast spatial indexing of endpoints and skeleton vertices to construct long-range attraction bridge edges.
    Supports both tip-to-tip collinearity and tip-to-skeleton proximity.
    """
    all_endpoints: List[EndpointTangent] = []
    for frag in fragments:
        all_endpoints.extend(frag.endpoints)

    if len(all_endpoints) < 2:
        return []

    # 1. Index all endpoints
    coords = np.array([ep.coord_nm for ep in all_endpoints])
    kdtree = KDTree(coords)

    pairs = kdtree.query_pairs(r=max_distance_nm)
    bridges: List[AssemblyEdge] = []
    seen_frag_pairs = set()

    for idx1, idx2 in pairs:
        ep1 = all_endpoints[idx1]
        ep2 = all_endpoints[idx2]

        if ep1.fragment_id == ep2.fragment_id:
            continue

        collin = compute_collinearity(ep1, ep2, sigma_dist_nm=15000.0)
        dist = float(np.linalg.norm(ep2.coord_nm - ep1.coord_nm))
        
        # If very close in space (cut seam <= 3000 nm) or collinear across larger distance
        if dist <= 3000.0 or collin >= min_collinearity:
            pair_key = tuple(sorted([ep1.fragment_id, ep2.fragment_id]))
            seen_frag_pairs.add(pair_key)
            effective_score = max(collin, float(np.exp(-dist / 3000.0)))
            bridges.append(AssemblyEdge(
                src_id=ep1.fragment_id,
                dst_id=ep2.fragment_id,
                edge_type=EdgeType.TANGENT_FLOW,
                distance_nm=dist,
                collinearity_score=effective_score,
                weight=effective_score,
                metadata={"ep1": ep1.to_dict(), "ep2": ep2.to_dict()}
            ))

    # 2. Index fragment skeleton point clouds for tip-to-skeleton proximity
    for frag1 in fragments:
        if len(frag1.endpoints) == 0:
            continue
        for ep in frag1.endpoints:
            for frag2 in fragments:
                if frag1.fragment_id == frag2.fragment_id:
                    continue
                pair_key = tuple(sorted([frag1.fragment_id, frag2.fragment_id]))
                if pair_key in seen_frag_pairs:
                    continue
                
                # Check distance from endpoint to frag2 vertices
                if len(frag2.vertices_nm) == 0:
                    continue
                dists = np.linalg.norm(frag2.vertices_nm - ep.coord_nm, axis=1)
                min_idx = int(np.argmin(dists))
                min_d = float(dists[min_idx])
                if min_d <= 3000.0:  # tight cut seam tolerance (<= 3um)
                    seen_frag_pairs.add(pair_key)
                    prox_score = float(np.exp(-min_d / 3000.0))
                    bridges.append(AssemblyEdge(
                        src_id=frag1.fragment_id,
                        dst_id=frag2.fragment_id,
                        edge_type=EdgeType.TANGENT_FLOW,
                        distance_nm=min_d,
                        collinearity_score=prox_score,
                        weight=prox_score,
                        metadata={"type": "tip_to_skeleton", "distance_nm": min_d}
                    ))

    return bridges
