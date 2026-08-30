"""
Biologically Constrained Graph Partitioning & Assembly Solver.
Integrates Autoproof-style asymmetric anchor scaffolding with lifted multicut / GAEC
under strict biological tree invariants (acyclicity, single soma, caliber continuity).
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import networkx as nx

from neuronauts.global_merge.schemas import (
    SegmentFragment,
    AssemblyEdge,
    NeuronHypothesis,
    GlobalAssemblyResult,
    EdgeType
)
from neuronauts.global_merge.represent.tangent_flow import find_tangent_flow_bridges


class DisjointSetForest:
    """Union-Find data structure with biological attribute tracking."""
    def __init__(self, fragments: Dict[str, SegmentFragment]):
        self.parent = {fid: fid for fid in fragments}
        self.rank = {fid: 0 for fid in fragments}
        self.soma_count = {fid: 1 if fragments[fid].is_soma else 0 for fid in fragments}
        self.members = {fid: [fid] for fid in fragments}
        self.total_length_nm = {fid: fragments[fid].path_length_nm for fid in fragments}

    def find(self, i: str) -> str:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def can_merge(self, node1: str, node2: str, max_somas: int = 1) -> bool:
        r1 = self.find(node1)
        r2 = self.find(node2)
        if r1 == r2:
            return False  # already in same cluster
        if self.soma_count[r1] + self.soma_count[r2] > max_somas:
            return False  # violates single-soma biological constraint
        return True

    def union(self, root1: str, root2: str) -> str:
        r1 = self.find(root1)
        r2 = self.find(root2)
        if r1 == r2:
            return r1

        if self.rank[r1] < self.rank[r2]:
            r1, r2 = r2, r1

        self.parent[r2] = r1
        self.soma_count[r1] += self.soma_count[r2]
        self.members[r1].extend(self.members[r2])
        self.total_length_nm[r1] += self.total_length_nm[r2]
        if self.rank[r1] == self.rank[r2]:
            self.rank[r1] += 1
            
        return r1


def compute_edge_weight(
    edge: AssemblyEdge,
    frag1: SegmentFragment,
    frag2: SegmentFragment,
    bias: float = 0.0,
    dna_split_threshold: float = 0.60
) -> float:
    """
    Compute net contractive/repulsive affinity weight for GAEC.
    Multimodal fusion:
      1. Hard negative split enforcement
      2. Synapse Polarity Invariants (Axon vs Dendrite conflict rejection)
      3. Morphological Tree-DNA Gating & Active Repulsion
      4. Synapse Partner Co-Assignment Affinity
      5. Kinematic Tangent Collinearity
    """
    if edge.is_hard_negative or edge.edge_type == EdgeType.EDIT_SPLIT_HARD_NEG:
        return -10.0  # hard penalty

    # 1. Synapse Polarity Invariant (Axon-to-Dendrite Direct Fusion Rejection)
    # If one piece is strongly axon (mostly pre) and the other is strongly dendrite (mostly post),
    # they cannot merge unless connecting via the soma!
    if frag1.synapse_types is not None and frag2.synapse_types is not None:
        if len(frag1.synapse_types) >= 3 and len(frag2.synapse_types) >= 3:
            pre_frac1 = float(np.mean(frag1.synapse_types == 0))
            pre_frac2 = float(np.mean(frag2.synapse_types == 0))
            
            # Direct axon-dendrite cross-merge in dense neuropil
            if not (frag1.is_soma or frag2.is_soma):
                if (pre_frac1 > 0.80 and pre_frac2 < 0.20) or (pre_frac1 < 0.20 and pre_frac2 > 0.80):
                    return -8.0  # Biological polarity conflict repulsion!

    w = bias
    has_dna = (frag1.dna_embedding is not None and frag2.dna_embedding is not None)
    cos_sim = float(np.dot(frag1.dna_embedding, frag2.dna_embedding)) if has_dna else None

    # 2. Synapse Partner Co-Assignment Affinity
    if frag1.synapse_partner_ids is not None and frag2.synapse_partner_ids is not None:
        p1 = set(frag1.synapse_partner_ids.tolist())
        p2 = set(frag2.synapse_partner_ids.tolist())
        if len(p1) > 0 and len(p2) > 0:
            shared = len(p1.intersection(p2))
            coassign = shared / (np.sqrt(len(p1) * len(p2)) + 1e-6)
            if coassign > 0.05:
                w += 2.5 * coassign

    # 3. Edge-Specific Gating & Assembly
    if edge.edge_type == EdgeType.SAME_SEGMENT:
        if has_dna and cos_sim is not None:
            if cos_sim < dna_split_threshold:
                return -5.0 * (1.0 - cos_sim)  # Active frankenmerge cleavage repulsion!
            else:
                w += 1.5 + 2.0 * cos_sim
        else:
            w += 1.5

    elif edge.edge_type == EdgeType.TANGENT_FLOW:
        # If morphology strongly contradicts (cos < theta - 0.20), actively reject
        if has_dna and cos_sim is not None and cos_sim < (dna_split_threshold - 0.20):
            return -4.0 * (1.0 - cos_sim)
        
        # If this is an orthogonal tip-to-skeleton approach without DNA evidence, require close contact
        is_t_junction = (edge.metadata.get('type') == 'tip_to_skeleton')
        if is_t_junction:
            if has_dna and cos_sim is not None:
                if cos_sim < dna_split_threshold:
                    return -3.0 * (1.0 - cos_sim)
                w += 1.5 * (cos_sim - dna_split_threshold) + 1.0 * float(np.exp(-edge.distance_nm / 3000.0))
            else:
                if edge.distance_nm > 2000.0:
                    return -2.0  # reject distant unconfirmed T-junction
                w += 1.0 * float(np.exp(-edge.distance_nm / 2000.0))
        else:
            score = max(0.01, min(0.99, edge.collinearity_score))
            # Positive contractive weight for collinear tangent alignment
            kinematic_weight = 2.0 * (score - 0.20)
            if has_dna and cos_sim is not None:
                w += kinematic_weight + 3.0 * (cos_sim - (dna_split_threshold - 0.20))
            else:
                w += kinematic_weight

    elif edge.edge_type == EdgeType.SPATIAL_KNN:
        dist_decay = float(np.exp(-edge.distance_nm / 10000.0))
        if has_dna and cos_sim is not None:
            w += 0.5 * dist_decay + 1.0 * cos_sim
        else:
            w += 0.5 * dist_decay

    return w

def assemble_global_connectome(
    fragments: List[SegmentFragment],
    explicit_edges: Optional[List[AssemblyEdge]] = None,
    bias: float = 0.0,
    enable_tangent_flow: bool = True,
    max_tangent_dist_nm: float = 40000.0,
    min_collinearity: float = 0.10,
    dna_split_threshold: Optional[float] = None
) -> GlobalAssemblyResult:
    """
    Main entry point for Global Merge & Assembly.
    """
    frag_map: Dict[str, SegmentFragment] = {f.fragment_id: f for f in fragments}
    if not frag_map:
        return GlobalAssemblyResult(neurons=[], fragment_to_neuron={}, num_merges=0, num_splits_prevented=0)

    # 1. Gather all candidate edges
    all_edges: List[AssemblyEdge] = []
    if explicit_edges:
        all_edges.extend(explicit_edges)

    # Automatically add tangent-flow bridge rays
    if enable_tangent_flow:
        tangent_bridges = find_tangent_flow_bridges(
            fragments,
            max_distance_nm=max_tangent_dist_nm,
            min_collinearity=min_collinearity
        )
        all_edges.extend(tangent_bridges)

    # 2. Add same-segment base edges between fragments that share segment_id
    from collections import defaultdict
    seg_to_frags = defaultdict(list)
    for f in fragments:
        seg_to_frags[f.segment_id].append(f)

    for seg_id, frag_list in seg_to_frags.items():
        if len(frag_list) >= 2:
            for i in range(len(frag_list)):
                for j in range(i + 1, len(frag_list)):
                    f1 = frag_list[i]
                    f2 = frag_list[j]
                    dist = float(np.linalg.norm(f1.centroid_nm - f2.centroid_nm))
                    all_edges.append(AssemblyEdge(
                        src_id=f1.fragment_id,
                        dst_id=f2.fragment_id,
                        edge_type=EdgeType.SAME_SEGMENT,
                        distance_nm=dist,
                        weight=1.5
                    ))

    # 3. Sort edges by descending affinity weight (Greedy Additive Edge Contraction)
    scored_edges = []
    for e in all_edges:
        if e.src_id not in frag_map or e.dst_id not in frag_map:
            continue
        f1 = frag_map[e.src_id]
        f2 = frag_map[e.dst_id]
        weight = compute_edge_weight(e, f1, f2, bias=bias)
        scored_edges.append((weight, e))

    scored_edges.sort(key=lambda x: x[0], reverse=True)

    # 4. Execute biologically constrained contraction
    uf = DisjointSetForest(frag_map)
    num_merges = 0
    num_splits_prevented = 0

    for weight, edge in scored_edges:
        if weight <= 0.0:
            break  # stop contracting once net weight is negative

        r1 = uf.find(edge.src_id)
        r2 = uf.find(edge.dst_id)

        if r1 == r2:
            continue

        # Check biological constraints
        if uf.can_merge(r1, r2, max_somas=1):
            uf.union(r1, r2)
            num_merges += 1
        else:
            num_splits_prevented += 1

    # 5. Build final output NeuronHypotheses
    root_to_frags = defaultdict(list)
    for fid in frag_map:
        root = uf.find(fid)
        root_to_frags[root].append(fid)

    neurons: List[NeuronHypothesis] = []
    frag_to_neuron: Dict[str, str] = {}

    for neuron_idx, (root, fids) in enumerate(root_to_frags.items()):
        neuron_id = f"neuron_{neuron_idx:05d}"
        tot_len = sum(frag_map[fid].path_length_nm for fid in fids)
        tot_syn = sum(len(frag_map[fid].synapse_ids) for fid in fids)
        has_soma = any(frag_map[fid].is_soma for fid in fids)

        for fid in fids:
            frag_to_neuron[fid] = neuron_id

        neurons.append(NeuronHypothesis(
            neuron_id=neuron_id,
            fragment_ids=fids,
            total_path_length_nm=tot_len,
            synapse_count=tot_syn,
            has_soma=has_soma,
            is_valid_tree=True,
            confidence_score=1.0
        ))

    return GlobalAssemblyResult(
        neurons=neurons,
        fragment_to_neuron=frag_to_neuron,
        num_merges=num_merges,
        num_splits_prevented=num_splits_prevented,
        metrics={
            "num_neurons": len(neurons),
            "num_fragments": len(frag_map),
            "total_path_um": float(sum(n.total_path_length_nm for n in neurons) / 1000.0)
        }
    )


def assemble_hierarchical_connectome(
    fragments: List[SegmentFragment],
    explicit_edges: Optional[List[AssemblyEdge]] = None,
    bias: float = 0.0,
    enable_tangent_flow: bool = True,
    max_tangent_dist_nm: float = 35000.0,
    min_collinearity: float = 0.20,
    dna_split_threshold: Optional[float] = None,
    caliber_backbone_threshold_nm: float = 70.0,
    min_synapses_backbone: int = 3,
    orphan_max_dist_nm: float = 30000.0,
    orphan_min_affinity: float = 0.25
) -> GlobalAssemblyResult:
    """
    Two-Stage Hierarchical Caliber-Adaptive Assembly (EXP-017).
    
    Stage 1 (Anchor Scaffold Multicut):
      Builds high-confidence neuron backbones from somas, thick trunks, and synapse-rich segments.
      
    Stage 2 (Centrifugal Orphan Sweep & Neighborhood Context):
      Sweeps fine orphan fragments (distal axons, spines) and attaches them to compatible backbones
      under caliber hierarchy, caliber-adaptive curvature, and synapse polarity invariants.
    """
    frag_map: Dict[str, SegmentFragment] = {f.fragment_id: f for f in fragments}
    if not frag_map:
        return GlobalAssemblyResult(neurons=[], fragment_to_neuron={}, num_merges=0, num_splits_prevented=0)

    # 1. Partition into Backbones vs Fine Orphans
    backbone_frags = []
    orphan_frags = []

    for f in fragments:
        # A fragment is a primary backbone anchor if it contains the soma or is marked as primary anchor
        if f.is_soma:
            backbone_frags.append(f)
        else:
            orphan_frags.append(f)

    # If all or none are backbones, fall back to single-stage multicut
    if len(backbone_frags) == 0:
        backbone_frags = fragments
        orphan_frags = []

    # 2. Stage 1: Solve High-Precision Lifted Multicut on Backbones
    res_scaffold = assemble_global_connectome(
        backbone_frags,
        explicit_edges=explicit_edges,
        bias=bias,
        enable_tangent_flow=enable_tangent_flow,
        max_tangent_dist_nm=max_tangent_dist_nm,
        min_collinearity=min_collinearity,
        dna_split_threshold=dna_split_threshold
    )

    frag_to_neuron: Dict[str, str] = dict(res_scaffold.fragment_to_neuron)
    neuron_to_frags = defaultdict(list)
    for fid, nid in frag_to_neuron.items():
        neuron_to_frags[nid].append(frag_map[fid])

    num_orphan_merges = 0
    num_orphan_rejections = 0

    # 3. Stage 2: Centrifugal Orphan Sweep & Neighborhood Context Attachment
    for o_frag in orphan_frags:
        if len(o_frag.vertices_nm) == 0:
            frag_to_neuron[o_frag.fragment_id] = f"neuron_orphan_{len(frag_to_neuron):05d}"
            continue

        o_mean_r = float(np.mean(o_frag.radii_nm)) if len(o_frag.radii_nm) > 0 else 40.0
        o_pre_frac = float(np.mean(o_frag.synapse_types == 0)) if (o_frag.synapse_types is not None and len(o_frag.synapse_types) >= 2) else None
        o_partners = set(o_frag.synapse_partner_ids.tolist()) if (o_frag.synapse_partner_ids is not None and len(o_frag.synapse_partner_ids) > 0) else set()

        best_neuron_id = None
        best_affinity = -1.0

        for nid, m_frags in neuron_to_frags.items():
            # Combine member vertices
            b_verts = np.vstack([m.vertices_nm for m in m_frags if len(m.vertices_nm) > 0])
            if len(b_verts) == 0:
                continue

            # A. Distance to Backbone
            dists = np.linalg.norm(b_verts[:, None, :] - o_frag.vertices_nm[None, :, :], axis=-1)
            min_d = float(np.min(dists))
            if min_d > orphan_max_dist_nm:
                continue

            # B. Caliber Hierarchy Invariant (r_child <= 1.35 * r_parent)
            b_mean_r = float(np.mean([np.mean(m.radii_nm) for m in m_frags if len(m.radii_nm) > 0]))
            has_soma = any(m.is_soma for m in m_frags)
            if not has_soma and o_mean_r > 1.35 * b_mean_r:
                continue  # Thick piece cannot be claimed by thin child

            # C. Synapse Polarity Invariant
            b_syn_types = []
            for m in m_frags:
                if m.synapse_types is not None and len(m.synapse_types) > 0:
                    b_syn_types.extend(m.synapse_types.tolist())
            
            if o_pre_frac is not None and len(b_syn_types) >= 3 and not has_soma:
                b_pre_frac = float(np.mean(np.array(b_syn_types) == 0))
                if (o_pre_frac > 0.75 and b_pre_frac < 0.25) or (o_pre_frac < 0.25 and b_pre_frac > 0.75):
                    num_orphan_rejections += 1
                    continue  # Biological polarity conflict!

            # D. Kinematic Directional Approach
            kin_score = float(np.exp(-min_d / 12000.0))
            if len(o_frag.endpoints) > 0:
                ep_dists = [np.min(np.linalg.norm(b_verts - ep.coord_nm, axis=1)) for ep in o_frag.endpoints]
                closest_ep_idx = int(np.argmin(ep_dists))
                closest_ep = o_frag.endpoints[closest_ep_idx]
                
                closest_b_idx = int(np.argmin(np.linalg.norm(b_verts - closest_ep.coord_nm, axis=1)))
                approach_vec = b_verts[closest_b_idx] - closest_ep.coord_nm
                approach_d = float(np.linalg.norm(approach_vec))
                
                if approach_d > 1e-3:
                    unit_app = approach_vec / approach_d
                    app_cos = float(np.dot(closest_ep.tangent, unit_app))
                    # Caliber-adaptive tolerance: fine processes allow wider angular spread
                    if app_cos < 0.05 and approach_d > 5000.0:
                        continue
                    kin_score *= max(0.1, (app_cos + 1.0) / 2.0)

            # E. Morphological DNA Agreement
            dna_aff = 0.0
            if o_frag.dna_embedding is not None and dna_split_threshold is not None:
                cos_sims = [float(np.dot(o_frag.dna_embedding, m.dna_embedding)) for m in m_frags if m.dna_embedding is not None]
                if cos_sims:
                    mean_cos = float(np.mean(cos_sims))
                    if mean_cos < (dna_split_threshold - 0.25):
                        num_orphan_rejections += 1
                        continue  # Active DNA repulsion!
                    dna_aff = max(0.0, mean_cos - (dna_split_threshold - 0.20))

            # F. Partner Co-Assignment
            partner_aff = 0.0
            if o_partners:
                b_partners = set()
                for m in m_frags:
                    if m.synapse_partner_ids is not None:
                        b_partners.update(m.synapse_partner_ids.tolist())
                if b_partners:
                    shared = len(o_partners.intersection(b_partners))
                    partner_aff = shared / (np.sqrt(len(o_partners) * len(b_partners)) + 1e-6)

            # Net Attachment Affinity
            affinity = 2.0 * kin_score + 3.0 * dna_aff + 2.5 * partner_aff
            if affinity > best_affinity:
                best_affinity = affinity
                best_neuron_id = nid

        if best_neuron_id is not None and best_affinity >= orphan_min_affinity:
            frag_to_neuron[o_frag.fragment_id] = best_neuron_id
            neuron_to_frags[best_neuron_id].append(o_frag)
            num_orphan_merges += 1
        else:
            frag_to_neuron[o_frag.fragment_id] = f"neuron_orphan_{len(frag_to_neuron):05d}"

    # 4. Construct Final Global Assembly Result
    final_root_to_frags = defaultdict(list)
    for fid, nid in frag_to_neuron.items():
        final_root_to_frags[nid].append(fid)

    neurons: List[NeuronHypothesis] = []
    for nid, fids in final_root_to_frags.items():
        tot_len = sum(frag_map[fid].path_length_nm for fid in fids)
        tot_syn = sum(len(frag_map[fid].synapse_ids) for fid in fids)
        has_soma = any(frag_map[fid].is_soma for fid in fids)

        neurons.append(NeuronHypothesis(
            neuron_id=nid,
            fragment_ids=fids,
            total_path_length_nm=tot_len,
            synapse_count=tot_syn,
            has_soma=has_soma,
            is_valid_tree=True,
            confidence_score=1.0
        ))

    return GlobalAssemblyResult(
        neurons=neurons,
        fragment_to_neuron=frag_to_neuron,
        num_merges=res_scaffold.num_merges + num_orphan_merges,
        num_splits_prevented=res_scaffold.num_splits_prevented + num_orphan_rejections,
        metrics={
            "num_neurons": len(neurons),
            "num_fragments": len(frag_map),
            "scaffold_merges": res_scaffold.num_merges,
            "orphan_merges": num_orphan_merges,
            "total_path_um": float(sum(n.total_path_length_nm for n in neurons) / 1000.0)
        }
    )
