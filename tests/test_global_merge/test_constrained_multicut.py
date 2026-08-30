"""
Tests for Biologically Constrained Lifted Multicut & Assembly Solver.
"""

import pytest
import numpy as np
from neuronauts.global_merge.schemas import (
    SegmentFragment,
    AssemblyEdge,
    EdgeType,
    EndpointTangent
)
from neuronauts.global_merge.solver.constrained_multicut import (
    DisjointSetForest,
    assemble_global_connectome
)


def test_disjoint_set_single_soma_constraint():
    # Create two soma fragments and one non-soma fragment
    f1 = SegmentFragment("soma1", 1, np.zeros((2,3)), np.ones(2), np.array([[0,1]]), is_soma=True)
    f2 = SegmentFragment("soma2", 2, np.zeros((2,3)), np.ones(2), np.array([[0,1]]), is_soma=True)
    f3 = SegmentFragment("axon1", 1, np.zeros((2,3)), np.ones(2), np.array([[0,1]]), is_soma=False)

    frag_map = {"soma1": f1, "soma2": f2, "axon1": f3}
    uf = DisjointSetForest(frag_map)

    # Soma1 + Axon1 is allowed
    assert uf.can_merge("soma1", "axon1", max_somas=1) is True
    uf.union("soma1", "axon1")

    # Soma1 (merged with axon1) + Soma2 MUST be rejected
    assert uf.can_merge("soma1", "soma2", max_somas=1) is False
    assert uf.can_merge("axon1", "soma2", max_somas=1) is False


def test_assemble_global_connectome_simple_chain():
    # Three collinear fragments of a single neuron
    f1 = SegmentFragment(
        fragment_id="f1", segment_id=10,
        vertices_nm=np.array([[0.0, 0.0, 0.0], [5000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([50.0, 50.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("f1", 0, np.array([0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 50.0),
            EndpointTangent("f1", 1, np.array([5000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 50.0)
        ]
    )

    f2 = SegmentFragment(
        fragment_id="f2", segment_id=20,
        vertices_nm=np.array([[6000.0, 0.0, 0.0], [11000.0, 0.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([50.0, 50.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("f2", 0, np.array([6000.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 50.0),
            EndpointTangent("f2", 1, np.array([11000.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 50.0)
        ]
    )

    # An unrelated orthogonal fragment
    f3 = SegmentFragment(
        fragment_id="f3", segment_id=30,
        vertices_nm=np.array([[6000.0, 5000.0, 0.0], [6000.0, 10000.0, 0.0]], dtype=np.float32),
        radii_nm=np.array([50.0, 50.0], dtype=np.float32),
        edges=np.array([[0, 1]], dtype=np.int64),
        endpoints=[
            EndpointTangent("f3", 0, np.array([6000.0, 5000.0, 0.0]), np.array([0.0, -1.0, 0.0]), 50.0),
            EndpointTangent("f3", 1, np.array([6000.0, 10000.0, 0.0]), np.array([0.0, 1.0, 0.0]), 50.0)
        ]
    )

    res = assemble_global_connectome([f1, f2, f3], enable_tangent_flow=True, min_collinearity=0.3)
    
    # f1 and f2 should be merged into one neuron; f3 should be its own neuron
    assert res.fragment_to_neuron["f1"] == res.fragment_to_neuron["f2"]
    assert res.fragment_to_neuron["f1"] != res.fragment_to_neuron["f3"]
    assert len(res.neurons) == 2
