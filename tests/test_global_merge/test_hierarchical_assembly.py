import numpy as np
import pytest

from neuronauts.global_merge.schemas import EndpointTangent, SegmentFragment
from neuronauts.global_merge.solver.constrained_multicut import assemble_hierarchical_connectome
from neuronauts.global_merge.eval.benchmark import compute_pairwise_partition_metrics


def test_hierarchical_assembly_basic_scaffold_and_orphan():
    # True neuron: 1 thick trunk (backbone) and 1 fine distal branch (orphan)
    trunk = SegmentFragment(
        fragment_id="trunk", segment_id=1,
        vertices_nm=np.array([[0.0, 0.0, 0.0], [10000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([200.0, 200.0], dtype=np.float32), # thick trunk
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("trunk", 0, np.array([0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 200.0),
            EndpointTangent("trunk", 1, np.array([10000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 200.0)
        ],
        dna_embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        synapse_types=np.array([1, 1, 1, 1], dtype=np.int64) # dendrite
    )

    orphan = SegmentFragment(
        fragment_id="orphan", segment_id=2,
        vertices_nm=np.array([[12000.0, 0.0, 0.0], [16000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([40.0, 40.0], dtype=np.float32), # fine orphan
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("orphan", 0, np.array([12000.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 40.0),
            EndpointTangent("orphan", 1, np.array([16000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 40.0)
        ],
        dna_embedding=np.array([0.9, 0.1, 0.0], dtype=np.float32), # strong DNA agreement
        synapse_types=np.array([1], dtype=np.int64) # dendrite
    )

    # Unrelated distant axon piece
    unrelated_axon = SegmentFragment(
        fragment_id="axon", segment_id=3,
        vertices_nm=np.array([[12000.0, 30000.0, 0.0], [16000.0, 30000.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([40.0, 40.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("axon", 0, np.array([12000.0, 30000.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 40.0),
            EndpointTangent("axon", 1, np.array([16000.0, 30000.0, 0.0]), np.array([1.0, 0.0, 0.0]), 40.0)
        ],
        dna_embedding=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        synapse_types=np.array([0, 0, 0], dtype=np.int64)
    )

    res = assemble_hierarchical_connectome(
        [trunk, orphan, unrelated_axon],
        enable_tangent_flow=True,
        dna_split_threshold=0.60
    )

    # The orphan should be attached to the trunk
    assert res.fragment_to_neuron["trunk"] == res.fragment_to_neuron["orphan"]
    # The unrelated axon must remain in a separate cluster
    assert res.fragment_to_neuron["trunk"] != res.fragment_to_neuron["axon"]
