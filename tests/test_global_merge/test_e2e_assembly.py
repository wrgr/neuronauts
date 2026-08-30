"""
End-to-End Integration & Benchmark Evaluation Tests.
"""

import pytest
import numpy as np
from neuronauts.global_merge.schemas import SegmentFragment, EndpointTangent
from neuronauts.global_merge.solver.constrained_multicut import assemble_global_connectome
from neuronauts.global_merge.eval.benchmark import (
    compute_pairwise_partition_metrics,
    evaluate_frankenmerge_split_rate
)
from neuronauts.global_merge.data.cave_lineage import pre_split_frankenmerges


def test_e2e_synthetic_connectome_assembly():
    # Build 3 ground truth neurons, each split into 3 collinear fragments (9 fragments total)
    fragments = []
    gt_map = {}

    for neuron_idx in range(3):
        n_id = f"true_neuron_{neuron_idx}"
        y_pos = neuron_idx * 10000.0  # separate by 10 microns along Y
        
        for piece_idx in range(3):
            f_id = f"n{neuron_idx}_p{piece_idx}"
            gt_map[f_id] = n_id
            
            x_start = piece_idx * 6000.0
            x_end = x_start + 4500.0  # 1500 nm gap between consecutive pieces
            
            v = np.array([[x_start, y_pos, 0.0], [x_end, y_pos, 0.0]], dtype=np.float32)
            r = np.array([60.0, 60.0], dtype=np.float32)
            e = np.array([[0, 1]], dtype=np.int64)
            
            eps = [
                EndpointTangent(f_id, 0, np.array([x_start, y_pos, 0.0]), np.array([-1.0, 0.0, 0.0]), 60.0),
                EndpointTangent(f_id, 1, np.array([x_end, y_pos, 0.0]), np.array([1.0, 0.0, 0.0]), 60.0),
            ]
            
            frag = SegmentFragment(
                fragment_id=f_id,
                segment_id=100 + neuron_idx * 10 + piece_idx,
                vertices_nm=v,
                radii_nm=r,
                edges=e,
                endpoints=eps
            )
            fragments.append(frag)

    # Run assembly
    res = assemble_global_connectome(fragments, enable_tangent_flow=True, min_collinearity=0.3)
    
    # Compute metrics
    metrics = compute_pairwise_partition_metrics(res.fragment_to_neuron, gt_map)
    
    # Check that all pieces of each neuron were correctly assembled
    assert metrics["ari"] == 1.0
    assert metrics["merge_P"] == 1.0
    assert metrics["merge_R"] == 1.0
    assert len(res.neurons) == 3


def test_frankenmerge_pre_split_and_fk_rate():
    # Construct a frankenmerge: an axon (r=40) fused to a thick dendrite (r=500)
    v = np.array([
        [0.0, 0.0, 0.0],     # axon
        [1000.0, 0.0, 0.0],  # axon
        [1200.0, 0.0, 0.0],  # dendrite (step)
        [2500.0, 0.0, 0.0],  # dendrite
        [4000.0, 0.0, 0.0],  # dendrite
    ], dtype=np.float32)
    r = np.array([40.0, 40.0, 500.0, 500.0, 500.0], dtype=np.float32)
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)

    franken_frag = SegmentFragment(
        fragment_id="franken_01",
        segment_id=999,
        vertices_nm=v,
        radii_nm=r,
        edges=e
    )

    split_frags = pre_split_frankenmerges(franken_frag, max_radius_ratio=3.5)
    # Must be split into 2 sub-fragments
    assert len(split_frags) == 2
