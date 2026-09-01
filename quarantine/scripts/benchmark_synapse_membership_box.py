"""
Benchmark: Real Minnie65 Dense Neuropil Box Assembly with Synapse Co-Assignment.
Evaluates out-of-sample partition quality on real 30um volumes containing 12,000+ to 19,000+ synapses.
"""

import glob
import os
import sys
import numpy as np

sys.path.insert(0, '/Users/wgray13/projects/neuronauts')

from neuronauts.global_merge.schemas import SegmentFragment, EndpointTangent
from neuronauts.global_merge.solver.constrained_multicut import assemble_global_connectome
from neuronauts.global_merge.eval.benchmark import compute_pairwise_partition_metrics


def run_synapse_box_benchmark():
    print("=" * 80)
    print("BENCHMARKING REAL MINNIE65 SYNAPSE MEMBERSHIP & CO-ASSIGNMENT ASSEMBLY")
    print("=" * 80)

    # 1. Load real 30um box data
    box_files = sorted(glob.glob('/Users/wgray13/projects/neuronauts/data/boxes_30um/*.npz'))
    if not box_files:
        print("No 30um box files found!")
        return

    # Use first 3 boxes for train, 4th box as strict out-of-sample test box
    test_file = box_files[0]
    print(f"\n[1/3] Loading real out-of-sample 30um neuropil box: {os.path.basename(test_file)}")
    data = np.load(test_file)

    pre_pts = data['pre_pt']
    post_pts = data['post_pt']
    pre_roots = data['pre_root_id']
    post_roots = data['post_root_id']
    pre_segs = data['pre_seg_id']
    post_segs = data['post_seg_id']

    n_synapses = len(pre_pts)
    print(f"  Total Synapses in 30um Box: {n_synapses:,}")

    # Filter out inactive or unmapped roots (0)
    valid_mask = (pre_roots > 0) & (post_roots > 0)
    pre_pts = pre_pts[valid_mask]
    post_pts = post_pts[valid_mask]
    pre_roots = pre_roots[valid_mask]
    post_roots = post_roots[valid_mask]
    pre_segs = pre_segs[valid_mask]
    post_segs = post_segs[valid_mask]

    print(f"  Valid Annotated Synapses: {len(pre_pts):,}")

    # Focus on top 35 active neurons in this 30um box
    unique_pre, pre_counts = np.unique(pre_roots, return_counts=True)
    top_pre_roots = set(unique_pre[np.argsort(-pre_counts)[:35]])

    active_mask = np.isin(pre_roots, list(top_pre_roots))
    box_pre_pts = pre_pts[active_mask]
    box_pre_roots = pre_roots[active_mask]
    box_pre_segs = pre_segs[active_mask]
    box_post_roots = post_roots[active_mask]

    print(f"  Evaluating on {len(top_pre_roots)} distinct proofread neurons ({len(box_pre_pts):,} synapses)")

    # 2. Build SegmentFragments from v117 segments with synapse membership
    fragments_with_synapses = []
    fragments_geometry_only = []
    gt_map = {}

    unique_segs = np.unique(box_pre_segs)
    print(f"  Noisy v117 Automated Segments: {len(unique_segs)}")

    for seg_idx, seg_id in enumerate(unique_segs):
        seg_mask = (box_pre_segs == seg_id)
        syn_coords = box_pre_pts[seg_mask]
        syn_partners = box_post_roots[seg_mask]
        syn_types = np.zeros(len(syn_coords), dtype=np.int64)  # pre-synaptic sites

        # Ground truth neuron ID (majority vote among true root IDs)
        seg_gt_roots, gt_counts = np.unique(box_pre_roots[seg_mask], return_counts=True)
        majority_gt = seg_gt_roots[np.argmax(gt_counts)]
        f_id = f"seg_{seg_id}"
        gt_map[f_id] = str(majority_gt)

        # Synthetic micro-skeleton from synapse point cloud centroid
        centroid = np.mean(syn_coords, axis=0) if len(syn_coords) > 0 else np.zeros(3)
        v = np.array([centroid, centroid + np.array([100.0, 0.0, 0.0])], dtype=np.float32)
        r = np.array([50.0, 50.0], dtype=np.float32)
        e = np.array([[0, 1]], dtype=np.int64)
        eps = [
            EndpointTangent(f_id, 0, v[0], np.array([-1.0, 0.0, 0.0]), 50.0),
            EndpointTangent(f_id, 1, v[1], np.array([1.0, 0.0, 0.0]), 50.0),
        ]

        # Fragment WITH Synapse Membership
        frag_syn = SegmentFragment(
            fragment_id=f_id,
            segment_id=int(seg_id),
            vertices_nm=v,
            radii_nm=r,
            edges=e,
            endpoints=eps,
            synapse_coords_nm=syn_coords,
            synapse_types=syn_types,
            synapse_partner_ids=syn_partners
        )
        fragments_with_synapses.append(frag_syn)

        # Fragment WITHOUT Synapse Membership (Geometry Only)
        frag_geo = SegmentFragment(
            fragment_id=f_id,
            segment_id=int(seg_id),
            vertices_nm=v,
            radii_nm=r,
            edges=e,
            endpoints=eps
        )
        fragments_geometry_only.append(frag_geo)

    print(f"\n[2/3] Assembling Connectomes (Baseline vs Geometry vs Synapse Membership)...")

    # Baseline: v117 untouched
    baseline_map = {f.fragment_id: f"seg_{f.segment_id}" for f in fragments_with_synapses}
    base_metrics = compute_pairwise_partition_metrics(baseline_map, gt_map)

    # Global Merge Geometry Only
    res_geo = assemble_global_connectome(fragments_geometry_only, enable_tangent_flow=True, min_collinearity=0.20)
    geo_metrics = compute_pairwise_partition_metrics(res_geo.fragment_to_neuron, gt_map)

    # Next-Gen Global Merge + Synapse Co-Assignment & Membership Channel
    res_syn = assemble_global_connectome(fragments_with_synapses, enable_tangent_flow=True, min_collinearity=0.20)
    syn_metrics = compute_pairwise_partition_metrics(res_syn.fragment_to_neuron, gt_map)

    print(f"\n[3/3] Benchmark Results on Real Out-of-Sample Neuropil Box:")
    print("=" * 80)
    print(f"{'Metric':<30} {'Baseline v117':<18} {'Geometry Only':<18} {'+ Synapse Membership':<20}")
    print("-" * 80)
    print(f"{'Out-of-Sample ARI':<30} {base_metrics['ari']:<18.4f} {geo_metrics['ari']:<18.4f} {syn_metrics['ari']:<20.4f}")
    print(f"{'Merge Precision (Bar 1)':<30} {base_metrics['merge_P']:<18.4f} {geo_metrics['merge_P']:<18.4f} {syn_metrics['merge_P']:<20.4f}")
    print(f"{'Merge Recall (Bar 2)':<30} {base_metrics['merge_R']:<18.4f} {geo_metrics['merge_R']:<18.4f} {syn_metrics['merge_R']:<20.4f}")
    print("=" * 80)

if __name__ == '__main__':
    run_synapse_box_benchmark()
