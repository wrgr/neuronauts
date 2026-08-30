import numpy as np
from neuronauts.global_merge.schemas import SegmentFragment, AssemblyEdge, EdgeType, EndpointTangent
from neuronauts.global_merge.solver.constrained_multicut import compute_edge_weight, assemble_global_connectome

def test_axon_dendrite_polarity_rejection():
    # Fragment 1: Pure Axon (all pre-synaptic)
    f1 = SegmentFragment(
        fragment_id="axon_1", segment_id=10,
        vertices_nm=np.array([[0.0, 0.0, 0.0], [2000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([40.0, 40.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("axon_1", 1, np.array([2000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 40.0)
        ],
        synapse_types=np.array([0, 0, 0, 0, 0], dtype=np.int64),  # all pre
        synapse_partner_ids=np.array([101, 102, 103, 104, 105], dtype=np.int64)
    )

    # Fragment 2: Pure Dendrite (all post-synaptic)
    f2 = SegmentFragment(
        fragment_id="dendrite_1", segment_id=20,
        vertices_nm=np.array([[2200.0, 0.0, 0.0], [5000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([120.0, 120.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("dendrite_1", 0, np.array([2200.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 120.0)
        ],
        synapse_types=np.array([1, 1, 1, 1, 1], dtype=np.int64),  # all post
        synapse_partner_ids=np.array([201, 202, 203, 204, 205], dtype=np.int64)
    )

    edge = AssemblyEdge(
        src_id="axon_1", dst_id="dendrite_1",
        edge_type=EdgeType.TANGENT_FLOW,
        distance_nm=200.0, collinearity_score=0.95
    )

    w = compute_edge_weight(edge, f1, f2)
    assert w == -8.0, f"Expected biological polarity rejection (-8.0), got {w}"

    res = assemble_global_connectome([f1, f2], enable_tangent_flow=True)
    assert res.fragment_to_neuron["axon_1"] != res.fragment_to_neuron["dendrite_1"], "Axon and dendrite should NOT merge!"

def test_synapse_coassignment_affinity_boost():
    # Fragment 1 & 2: Two axon pieces sharing identical circuit partners
    f1 = SegmentFragment(
        fragment_id="axon_a", segment_id=30,
        vertices_nm=np.array([[0.0, 0.0, 0.0], [2000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([45.0, 45.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("axon_a", 1, np.array([2000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 45.0)
        ],
        synapse_types=np.array([0, 0, 0, 0], dtype=np.int64),
        synapse_partner_ids=np.array([500, 501, 502, 503], dtype=np.int64)
    )

    f2 = SegmentFragment(
        fragment_id="axon_b", segment_id=40,
        vertices_nm=np.array([[2500.0, 0.0, 0.0], [5000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([45.0, 45.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("axon_b", 0, np.array([2500.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 45.0)
        ],
        synapse_types=np.array([0, 0, 0, 0], dtype=np.int64),
        synapse_partner_ids=np.array([500, 501, 502, 503], dtype=np.int64)  # 100% shared partners
    )

    edge = AssemblyEdge(
        src_id="axon_a", dst_id="axon_b",
        edge_type=EdgeType.TANGENT_FLOW,
        distance_nm=500.0, collinearity_score=0.90
    )

    w = compute_edge_weight(edge, f1, f2)
    assert w > 2.0, f"Expected co-assignment boost, got {w}"

    res = assemble_global_connectome([f1, f2], enable_tangent_flow=True)
    assert res.fragment_to_neuron["axon_a"] == res.fragment_to_neuron["axon_b"], "Axon pieces should merge!"
