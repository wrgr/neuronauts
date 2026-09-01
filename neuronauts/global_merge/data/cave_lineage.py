"""
Data & Lineage Loader with Hard-Negative Frankenmerge Mining (Autoproof / CAVE Lineage).
Extracts human split operations from v117 -> v1412 proofreading history to provide
clean negative supervision for false merges.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from neuronauts.global_merge.schemas import SegmentFragment, AssemblyEdge, EdgeType
from neuronauts.global_merge.represent.tangent_flow import extract_endpoints_from_skeleton


def parse_skeleton_to_fragment(
    fragment_id: str,
    segment_id: int,
    vertices_nm: np.ndarray,
    radii_nm: np.ndarray,
    edges: np.ndarray,
    synapse_coords_nm: Optional[np.ndarray] = None,
    soma_coord_nm: Optional[np.ndarray] = None,
) -> SegmentFragment:
    """Construct typed SegmentFragment and precompute endpoint tangents."""
    vertices_nm = np.asarray(vertices_nm, dtype=np.float32)
    radii_nm = np.asarray(radii_nm, dtype=np.float32) if len(radii_nm) > 0 else np.full(len(vertices_nm), 50.0, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64) if len(edges) > 0 else np.empty((0, 2), dtype=np.int64)

    endpoints = extract_endpoints_from_skeleton(
        fragment_id=fragment_id,
        vertices_nm=vertices_nm,
        radii_nm=radii_nm,
        edges=edges
    )

    is_soma = False
    soma_conf = 0.0
    if soma_coord_nm is not None and len(vertices_nm) > 0:
        dists = np.linalg.norm(vertices_nm - soma_coord_nm, axis=1)
        if np.min(dists) < 5000.0:  # within 5 microns
            is_soma = True
            soma_conf = 1.0

    return SegmentFragment(
        fragment_id=fragment_id,
        segment_id=segment_id,
        vertices_nm=vertices_nm,
        radii_nm=radii_nm,
        edges=edges,
        endpoints=endpoints,
        synapse_coords_nm=synapse_coords_nm,
        is_soma=is_soma,
        soma_confidence=soma_conf
    )


def pre_split_frankenmerges(
    fragment: SegmentFragment,
    max_radius_ratio: float = 3.5,
    min_split_len_nm: float = 2000.0
) -> List[SegmentFragment]:
    """
    Inspects a single segment for internal caliber discontinuities (e.g. an axon abruptly
    fused to a thick dendrite trunk) and severs the edge at the bottleneck.
    """
    if len(fragment.edges) == 0 or len(fragment.vertices_nm) < 3:
        return [fragment]

    # Find edges with extreme radius ratios
    cut_edges = []
    for e_idx, (u, v) in enumerate(fragment.edges):
        r_u = float(fragment.radii_nm[u])
        r_v = float(fragment.radii_nm[v])
        ratio = max(r_u, r_v) / max(min(r_u, r_v), 1.0)
        
        # Abrupt caliber step
        if ratio > max_radius_ratio:
            dist = float(np.linalg.norm(fragment.vertices_nm[u] - fragment.vertices_nm[v]))
            if dist < 1000.0:  # step occurs over short span
                cut_edges.append(e_idx)

    if not cut_edges:
        return [fragment]

    # Build new graph without cut edges
    keep_edges_mask = np.ones(len(fragment.edges), dtype=bool)
    keep_edges_mask[cut_edges] = False
    new_edges = fragment.edges[keep_edges_mask]

    # Find connected components in the cut graph
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(fragment.vertices_nm)))
    G.add_edges_from(new_edges)
    
    sub_graphs = list(nx.connected_components(G))
    if len(sub_graphs) <= 1:
        return [fragment]

    result_fragments = []
    for sub_idx, node_set in enumerate(sub_graphs):
        nodes = sorted(list(node_set))
        if len(nodes) < 2:
            continue
            
        old_to_new = {old: new for new, old in enumerate(nodes)}
        sub_v = fragment.vertices_nm[nodes]
        sub_r = fragment.radii_nm[nodes]
        
        # Filter edges belonging to this component
        sub_e_list = []
        for u, v in new_edges:
            if u in old_to_new and v in old_to_new:
                sub_e_list.append([old_to_new[u], old_to_new[v]])
                
        sub_e = np.array(sub_e_list, dtype=np.int64) if sub_e_list else np.empty((0, 2), dtype=np.int64)
        sub_id = f"{fragment.fragment_id}_sub{sub_idx}"
        
        sub_frag = parse_skeleton_to_fragment(
            fragment_id=sub_id,
            segment_id=fragment.segment_id,
            vertices_nm=sub_v,
            radii_nm=sub_r,
            edges=sub_e,
        )
        result_fragments.append(sub_frag)

    return result_fragments if result_fragments else [fragment]


def mine_hard_negative_splits(
    fragments: List[SegmentFragment],
    ground_truth_neuron_map: Dict[str, str]
) -> List[AssemblyEdge]:
    """
    Identifies pairs of fragments that share the same v117 segment_id but belong
    to different ground-truth v1412 neurons (former frankenmerges that were split).
    Returns explicit hard-negative edges for edge-classifier training.
    """
    from collections import defaultdict
    seg_to_frags = defaultdict(list)
    for frag in fragments:
        seg_to_frags[frag.segment_id].append(frag)

    hard_negatives: List[AssemblyEdge] = []

    for seg_id, frag_list in seg_to_frags.items():
        if len(frag_list) < 2:
            continue
            
        for i in range(len(frag_list)):
            for j in range(i + 1, len(frag_list)):
                f1 = frag_list[i]
                f2 = frag_list[j]
                
                gt1 = ground_truth_neuron_map.get(f1.fragment_id)
                gt2 = ground_truth_neuron_map.get(f2.fragment_id)
                
                # If ground truth says they are different neurons
                if gt1 is not None and gt2 is not None and gt1 != gt2:
                    dist = float(np.linalg.norm(f1.centroid_nm - f2.centroid_nm))
                    hard_negatives.append(AssemblyEdge(
                        src_id=f1.fragment_id,
                        dst_id=f2.fragment_id,
                        edge_type=EdgeType.EDIT_SPLIT_HARD_NEG,
                        distance_nm=dist,
                        weight=-1.0,
                        is_hard_negative=True,
                        metadata={"seg_id": seg_id, "gt1": gt1, "gt2": gt2}
                    ))

    return hard_negatives
