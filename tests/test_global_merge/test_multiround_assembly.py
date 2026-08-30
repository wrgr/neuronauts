import numpy as np
import pytest

from neuronauts.global_merge.schemas import EndpointTangent, SegmentFragment
from neuronauts.global_merge.solver.constrained_multicut import assemble_multiround_hierarchical_connectome


def test_multiround_hierarchical_assembly_convergence():
    # Soma trunk
    trunk = SegmentFragment(
        fragment_id="soma_trunk", segment_id=1,
        vertices_nm=np.array([[0.0, 0.0, 0.0], [10000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([300.0, 300.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("soma_trunk", 0, np.array([0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 300.0),
            EndpointTangent("soma_trunk", 1, np.array([10000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 300.0)
        ],
        is_soma=True,
        dna_embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        synapse_types=np.array([1, 1, 1], dtype=np.int64) # dendrite
    )

    # Round 2 branch
    r2_branch = SegmentFragment(
        fragment_id="r2_branch", segment_id=2,
        vertices_nm=np.array([[11000.0, 0.0, 0.0], [15000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([80.0, 80.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("r2_branch", 0, np.array([11000.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 80.0),
            EndpointTangent("r2_branch", 1, np.array([15000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 80.0)
        ],
        dna_embedding=np.array([0.95, 0.05, 0.0], dtype=np.float32),
        synapse_types=np.array([1, 1], dtype=np.int64)
    )

    # Round 3 micro-spine enclosed within the arbor envelope
    r3_spine = SegmentFragment(
        fragment_id="r3_spine", segment_id=3,
        vertices_nm=np.array([[12000.0, 500.0, 0.0], [12500.0, 500.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([30.0, 30.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[],
        dna_embedding=np.array([0.85, 0.15, 0.0], dtype=np.float32),
        synapse_types=np.array([1], dtype=np.int64) # dendrite spine
    )

    # Unrelated axon (must NOT merge into dendrite arbor)
    unrelated_axon = SegmentFragment(
        fragment_id="unrelated_axon", segment_id=4,
        vertices_nm=np.array([[12000.0, 800.0, 0.0], [12500.0, 800.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([30.0, 30.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[],
        dna_embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        synapse_types=np.array([0, 0], dtype=np.int64) # axon polarity
    )

    res = assemble_multiround_hierarchical_connectome(
        [trunk, r2_branch, r3_spine, unrelated_axon],
        dna_split_threshold=0.65
    )

    # Trunk, r2_branch, and r3_spine should all be merged
    assert res.fragment_to_neuron["soma_trunk"] == res.fragment_to_neuron["r2_branch"]
    assert res.fragment_to_neuron["soma_trunk"] == res.fragment_to_neuron["r3_spine"]
    # Axon must be separate
    assert res.fragment_to_neuron["soma_trunk"] != res.fragment_to_neuron["unrelated_axon"]
