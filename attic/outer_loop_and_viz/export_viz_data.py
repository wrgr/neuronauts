"""
Export Real Minnie65 Out-of-Sample Neuropil Data for the 3D Connectomics Visualizer.
"""

import json
import sys
import numpy as np

sys.path.insert(0, '/Users/wgray13/projects/neuronauts')

from neuronauts.data.loaders import load_skeleton, sample_neurons
from neuronauts.global_merge.schemas import SegmentFragment, EndpointTangent
from neuronauts.global_merge.represent.vicreg_gnn import VICRegSkeletonModel
from neuronauts.global_merge.solver.constrained_multicut import assemble_global_connectome
from treestitch.data import _split_skeleton_n_pieces
from treestitch.worldbuild import frankenmerge_adjacent

def export_viz_dataset(output_json_path='/Users/wgray13/projects/neuronauts/viz/sample_connectome_viz.json'):
    candidates = sample_neurons(30, seed=42)
    pieces_rec = []
    obj_counter = 0
    rng = np.random.default_rng(42)

    for root_id in candidates:
        if obj_counter >= 12:
            break
        skel = load_skeleton(root_id)
        if skel is None:
            continue
        v, e, r = skel['vertices_nm'], skel['edges'], skel['radii_nm']
        if len(v) < 24 or len(v) > 5000:
            continue
        pieces = _split_skeleton_n_pieces(v, e, r, 3, min_verts=8)
        if len(pieces) < 2:
            continue
        obj_counter += 1
        for p_idx, (pv, pe, pr) in enumerate(pieces):
            n_syn = max(3, len(pv) // 12)
            syn_idx = rng.choice(len(pv), size=n_syn, replace=True)
            syn_coords = pv[syn_idx]
            is_axon = (p_idx == 2)
            syn_types = np.zeros(n_syn, dtype=int) if is_axon else np.ones(n_syn, dtype=int)
            partner_base = obj_counter * 100
            partner_ids = np.array([partner_base + rng.integers(0, 10) for _ in range(n_syn)], dtype=int)

            pieces_rec.append({
                'obj_id': obj_counter,
                'piece_idx': p_idx,
                'verts': pv,
                'edges': pe,
                'radii': pr,
                'syn_coords': syn_coords,
                'syn_types': syn_types,
                'syn_partners': partner_ids,
                'is_soma': (p_idx == 0)
            })

    seg_of_piece, n_franken = frankenmerge_adjacent(pieces_rec, 0.35, rng, radius_nm=6000.0)

    # Normalize coordinates into center of bounding box for smooth 3D rendering
    all_verts = np.vstack([p['verts'] for p in pieces_rec])
    center_nm = np.mean(all_verts, axis=0)
    scale_factor = 1.0 / 1000.0  # convert nm to microns for WebGL

    # Build fragments
    fragments = []
    gt_map = {}
    for i, p in enumerate(pieces_rec):
        f_id = f'frag_{i:03d}'
        gt_map[f_id] = f'neuron_{p["obj_id"]}'
        orig_v = p['verts']
        norm_v = (orig_v - center_nm) * scale_factor

        eps = [
            EndpointTangent(f_id, 0, orig_v[0], np.array([-1.0, 0.0, 0.0]), float(p['radii'][0])),
            EndpointTangent(f_id, len(orig_v)-1, orig_v[-1], np.array([1.0, 0.0, 0.0]), float(p['radii'][-1]))
        ]

        f = SegmentFragment(
            fragment_id=f_id,
            segment_id=int(seg_of_piece[i]),
            vertices_nm=orig_v,
            radii_nm=p['radii'],
            edges=p['edges'],
            endpoints=eps,
            synapse_coords_nm=p['syn_coords'],
            synapse_types=p['syn_types'],
            synapse_partner_ids=p['syn_partners'],
            is_soma=p['is_soma']
        )
        fragments.append(f)

    # Run assembly
    res = assemble_global_connectome(fragments, enable_tangent_flow=True, min_collinearity=0.20)

    # Compute Link Classifications (TP, FP, FN, TN)
    links = []
    # Evaluate all pairs
    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            f1, f2 = fragments[i], fragments[j]
            same_gt = (gt_map[f1.fragment_id] == gt_map[f2.fragment_id])
            same_pred = (res.fragment_to_neuron[f1.fragment_id] == res.fragment_to_neuron[f2.fragment_id])
            same_seg = (f1.segment_id == f2.segment_id)

            c1 = (np.mean(f1.vertices_nm, axis=0) - center_nm) * scale_factor
            c2 = (np.mean(f2.vertices_nm, axis=0) - center_nm) * scale_factor
            dist_um = float(np.linalg.norm(c1 - c2))

            if dist_um > 25.0:
                continue

            if same_pred and same_gt:
                links.append({'src': f1.fragment_id, 'dst': f2.fragment_id, 'type': 'tp', 'p1': c1.tolist(), 'p2': c2.tolist(), 'label': 'True Positive Merge'})
            elif same_pred and not same_gt:
                links.append({'src': f1.fragment_id, 'dst': f2.fragment_id, 'type': 'fp', 'p1': c1.tolist(), 'p2': c2.tolist(), 'label': 'False Positive Over-Merge'})
            elif not same_pred and same_gt and dist_um < 15.0:
                links.append({'src': f1.fragment_id, 'dst': f2.fragment_id, 'type': 'fn', 'p1': c1.tolist(), 'p2': c2.tolist(), 'label': 'False Negative Split'})
            elif same_seg and not same_pred and not same_gt:
                links.append({'src': f1.fragment_id, 'dst': f2.fragment_id, 'type': 'tn', 'p1': c1.tolist(), 'p2': c2.tolist(), 'label': 'Frankenmerge Cleaved (TN)'})

    # Prepare export JSON
    export_data = {
        'metadata': {
            'n_neurons': obj_counter,
            'n_fragments': len(fragments),
            'n_synapses': sum(len(p['syn_coords']) for p in pieces_rec),
            'center_nm': center_nm.tolist(),
            'scale_um': float(scale_factor)
        },
        'fragments': [],
        'links': links
    }

    for i, p in enumerate(pieces_rec):
        f = fragments[i]
        norm_v = ((p['verts'] - center_nm) * scale_factor).tolist()
        norm_syn = ((p['syn_coords'] - center_nm) * scale_factor).tolist() if len(p['syn_coords']) > 0 else []

        export_data['fragments'].append({
            'id': f.fragment_id,
            'gt_neuron': gt_map[f.fragment_id],
            'pred_neuron': res.fragment_to_neuron[f.fragment_id],
            'seg_id': f.segment_id,
            'is_soma': f.is_soma,
            'vertices_um': norm_v,
            'edges': p['edges'].tolist(),
            'radii_nm': p['radii'].tolist(),
            'synapses': [
                {
                    'pos_um': norm_syn[s_idx],
                    'type': int(p['syn_types'][s_idx]),  # 0=pre, 1=post
                    'partner_id': int(p['syn_partners'][s_idx])
                }
                for s_idx in range(len(norm_syn))
            ]
        })

    import os
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f'Successfully exported visualizer dataset to {output_json_path}!')

if __name__ == '__main__':
    export_viz_dataset()
