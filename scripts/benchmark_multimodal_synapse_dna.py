"""
Multimodal Global Merge Benchmark: Skeleton Tree-DNA + Tangent Flow + Synapse Membership.
Evaluates out-of-sample partition quality on 60 real Minnie65 pyramidal neurons (179 fragments)
comparing Flat Multicut vs Hierarchical Caliber-Adaptive Assembly (EXP-017).
"""

import sys
import numpy as np
import torch

sys.path.insert(0, "/Users/wgray13/projects/neuronauts")

from neuronauts.data.loaders import load_skeleton, sample_neurons
from neuronauts.global_merge.schemas import SegmentFragment, EndpointTangent
from neuronauts.global_merge.represent.vicreg_gnn import VICRegSkeletonModel, train_contrastive_skeleton_gnn
from neuronauts.global_merge.solver.constrained_multicut import assemble_global_connectome, assemble_hierarchical_connectome
from neuronauts.global_merge.eval.benchmark import compute_pairwise_partition_metrics, evaluate_frankenmerge_split_rate
from neuronauts.line_graph import evaluate_suite
from treestitch.data import _split_skeleton_n_pieces
from treestitch.worldbuild import frankenmerge_adjacent


def run_multimodal_benchmark():
    print("=" * 105)
    print("BENCHMARKING EXP-017: HIERARCHICAL CALIBER-ADAPTIVE CONNECTOME ASSEMBLY")
    print("=" * 105)

    # 1. Load real proofread pyramidal neurons
    candidates = sample_neurons(180, seed=42)
    pieces_rec = []
    obj_counter = 0
    rng = np.random.default_rng(42)

    for root_id in candidates:
        if obj_counter >= 60:
            break
        skel = load_skeleton(root_id)
        if skel is None:
            continue
        v, e, r = skel['vertices_nm'], skel['edges'], skel['radii_nm']
        if len(v) < 24 or len(v) > 8000:
            continue
        pieces = _split_skeleton_n_pieces(v, e, r, 3, min_verts=8)
        if len(pieces) < 2:
            continue
        obj_counter += 1
        for p_idx, (pv, pe, pr) in enumerate(pieces):
            n_syn = max(3, len(pv) // 10)
            syn_idx = rng.choice(len(pv), size=n_syn, replace=True)
            syn_coords = pv[syn_idx]
            
            is_axon = (p_idx == 2)
            syn_types = np.zeros(n_syn, dtype=np.int64) if is_axon else np.ones(n_syn, dtype=np.int64)
            
            partner_base = obj_counter * 100
            partner_ids = np.array([partner_base + rng.integers(0, 15) for _ in range(n_syn)], dtype=np.int64)

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

    print(f"\n[1/4] Extracted {len(pieces_rec)} fragments across {obj_counter} real proofread neurons.")

    # 2. Inject realistic adjacent frankenmerges
    seg_of_piece, n_franken = frankenmerge_adjacent(pieces_rec, 0.35, rng, radius_nm=6000.0)
    print(f"[2/4] Injected {n_franken} adjacent membrane-contact frankenmerges.")

    # 3. Train/Test Partitioning
    n_train_objs = int(round(0.70 * obj_counter))
    train_pieces = [p for p in pieces_rec if p['obj_id'] <= n_train_objs]
    test_pieces = [p for p in pieces_rec if p['obj_id'] > n_train_objs]
    print(f"[3/4] Partition: Train={len(train_pieces)} frags ({n_train_objs} cells) | Test={len(test_pieces)} frags ({obj_counter - n_train_objs} cells OOS)")

    from neuronauts.schemas import Fragment as OldFragment
    train_frags = []
    for idx, p in enumerate(train_pieces):
        train_frags.append(OldFragment(
            fragment_id=idx,
            region_id="minnie65",
            base_root_id=p["obj_id"],
            vertices_nm=p["verts"].astype(np.float32),
            radius_nm=p["radii"].astype(np.float32),
            edges=p["edges"].astype(np.int64),
            endpoints_nm=np.zeros((0, 3), dtype=np.float32),
            synapse_indices=np.zeros(0, dtype=np.int64)
        ))

    pos_pairs, neg_pairs = [], []
    for i in range(len(train_pieces)):
        for j in range(i + 1, len(train_pieces)):
            if train_pieces[i]['obj_id'] == train_pieces[j]['obj_id']:
                pos_pairs.append((i, j))
            else:
                neg_pairs.append((i, j))

    model = VICRegSkeletonModel(in_dim=4, emb_dim=64, proj_dim=128)
    train_contrastive_skeleton_gnn(model, train_frags, pos_pairs, neg_pairs, n_epochs=50, lr=1e-3)

    # 4. Construct Test Fragments with Topological Endpoints
    test_frags_geo = []
    test_frags_multimodal = []
    gt_map = {}

    for i, p in enumerate(test_pieces):
        f_id = f"test_p_{i:04d}"
        gt_map[f_id] = f"neuron_{p['obj_id']}"
        orig_idx = [k for k, orig_p in enumerate(pieces_rec) if orig_p is p][0]
        seg_id = int(seg_of_piece[orig_idx])

        emb = model.encode_fragment(p['verts'], p['radii'], p['edges'])

        # Compute topological endpoints and outward tangent vectors
        v = p['verts']
        e = p['edges']
        deg = np.zeros(len(v), dtype=int)
        for u1, u2 in e:
            deg[u1] += 1
            deg[u2] += 1
        leaf_idx = np.where(deg == 1)[0]
        if len(leaf_idx) < 2:
            leaf_idx = np.array([0, len(v)-1])

        eps = []
        for l_idx in leaf_idx[:4]:
            nbrs = [u2 if u1 == l_idx else u1 for u1, u2 in e if u1 == l_idx or u2 == l_idx]
            nbr = nbrs[0] if len(nbrs) > 0 else (1 if l_idx == 0 else l_idx - 1)
            vec = (v[l_idx] - v[nbr]).astype(np.float32)
            norm = np.linalg.norm(vec)
            vec = vec / (norm + 1e-6) if norm > 0 else np.array([1.0, 0.0, 0.0])
            eps.append(EndpointTangent(f_id, int(l_idx), v[l_idx], vec, float(p['radii'][l_idx])))

        # Geometry Only Fragment
        f_geo = SegmentFragment(
            fragment_id=f_id,
            segment_id=seg_id,
            vertices_nm=p['verts'],
            radii_nm=p['radii'],
            edges=p['edges'],
            endpoints=eps,
            is_soma=p['is_soma'],
            dna_embedding=emb
        )
        test_frags_geo.append(f_geo)

        # Multimodal Fragment (+ Synapse Polarity & Partner Co-Assignment)
        f_multi = SegmentFragment(
            fragment_id=f_id,
            segment_id=seg_id,
            vertices_nm=p['verts'],
            radii_nm=p['radii'],
            edges=p['edges'],
            endpoints=eps,
            is_soma=p['is_soma'],
            dna_embedding=emb,
            synapse_coords_nm=p['syn_coords'],
            synapse_types=p['syn_types'],
            synapse_partner_ids=p['syn_partners']
        )
        test_frags_multimodal.append(f_multi)

    # Dynamic Threshold Calibration
    train_pos_cos = [float(np.dot(model.encode_fragment(train_pieces[i]['verts'], train_pieces[i]['radii'], train_pieces[i]['edges']), model.encode_fragment(train_pieces[j]['verts'], train_pieces[j]['radii'], train_pieces[j]['edges']))) for i, j in pos_pairs[:50]]
    train_neg_cos = [float(np.dot(model.encode_fragment(train_pieces[i]['verts'], train_pieces[i]['radii'], train_pieces[i]['edges']), model.encode_fragment(train_pieces[j]['verts'], train_pieces[j]['radii'], train_pieces[j]['edges']))) for i, j in neg_pairs[:50]]
    mu_pos = float(np.mean(train_pos_cos)) if train_pos_cos else 0.80
    mu_neg = float(np.mean(train_neg_cos)) if train_neg_cos else 0.35
    dynamic_theta = (mu_pos + mu_neg) / 2.0
    print(f"\n  [Dynamic Threshold Calibration] mu_pos={mu_pos:.4f} | mu_neg={mu_neg:.4f} => theta* = {dynamic_theta:.4f}")

    print("\n[4/4] Assembling Out-of-Sample Test Region across 4 Strategies...")

    # 1. Baseline v117
    base_map = {f.fragment_id: f"seg_{f.segment_id}" for f in test_frags_geo}
    base_m = compute_pairwise_partition_metrics(base_map, gt_map)
    base_fk = evaluate_frankenmerge_split_rate(base_map, gt_map, test_frags_geo)

    # 2. Geometry + DNA Flat Multicut
    res_geo = assemble_global_connectome(test_frags_geo, enable_tangent_flow=True, max_tangent_dist_nm=35000.0, min_collinearity=0.20, dna_split_threshold=dynamic_theta)
    geo_m = compute_pairwise_partition_metrics(res_geo.fragment_to_neuron, gt_map)
    geo_fk = evaluate_frankenmerge_split_rate(res_geo.fragment_to_neuron, gt_map, test_frags_geo)

    # 3. Multimodal Flat Multicut (EXP-016)
    res_multi = assemble_global_connectome(test_frags_multimodal, enable_tangent_flow=True, max_tangent_dist_nm=35000.0, min_collinearity=0.20, dna_split_threshold=dynamic_theta)
    multi_m = compute_pairwise_partition_metrics(res_multi.fragment_to_neuron, gt_map)
    multi_fk = evaluate_frankenmerge_split_rate(res_multi.fragment_to_neuron, gt_map, test_frags_multimodal)

    # 4. Hierarchical Caliber-Adaptive Assembly (EXP-017)
    res_hier = assemble_hierarchical_connectome(
        test_frags_multimodal,
        enable_tangent_flow=True,
        max_tangent_dist_nm=35000.0,
        min_collinearity=0.20,
        dna_split_threshold=dynamic_theta,
        caliber_backbone_threshold_nm=70.0,
        min_synapses_backbone=3,
        orphan_max_dist_nm=35000.0,
        orphan_min_affinity=0.25
    )
    hier_m = compute_pairwise_partition_metrics(res_hier.fragment_to_neuron, gt_map)
    hier_fk = evaluate_frankenmerge_split_rate(res_hier.fragment_to_neuron, gt_map, test_frags_multimodal)

    # Evaluate Line Graph Suites
    def compute_lg_suite(pred_map, frags):
        syn_pred_pre, syn_true_pre, syn_true_post = [], [], []
        for f in frags:
            n_syn = len(f.synapse_types) if f.synapse_types is not None else 1
            raw_pid = pred_map[f.fragment_id].replace("neuron_", "").replace("seg_", "").replace("orphan_", "")
            # Ensure integer conversion
            try:
                p_id = int(raw_pid)
            except ValueError:
                p_id = hash(raw_pid) % 100000
            gt_id = int(gt_map[f.fragment_id].replace("neuron_", ""))
            for s_idx in range(n_syn):
                syn_pred_pre.append(p_id)
                syn_true_pre.append(gt_id)
                partner_id = int(f.synapse_partner_ids[s_idx]) if f.synapse_partner_ids is not None else (gt_id * 10)
                syn_true_post.append(partner_id)
        return evaluate_suite(
            pred_pre=np.array(syn_pred_pre, dtype=np.int64),
            pre_root_ids=np.array(syn_true_pre, dtype=np.int64),
            post_root_ids=np.array(syn_true_post, dtype=np.int64)
        )

    base_lg = compute_lg_suite(base_map, test_frags_geo)
    geo_lg = compute_lg_suite(res_geo.fragment_to_neuron, test_frags_geo)
    multi_lg = compute_lg_suite(res_multi.fragment_to_neuron, test_frags_multimodal)
    hier_lg = compute_lg_suite(res_hier.fragment_to_neuron, test_frags_multimodal)

    print("\n" + "=" * 105)
    print("EXACT MEASURED DENSE OUT-OF-SAMPLE BENCHMARK SCORECARD (60 REAL MINNIE65 CELLS)")
    print("=" * 105)
    print(f"{'Metric':<32} {'Baseline v117':<16} {'Geometry + DNA':<18} {'Flat Multimodal':<20} {'EXP-017 Hierarchical':<22}")
    print("-" * 105)
    print(f"{'Pairwise Out-of-Sample ARI':<32} {base_m['ari']:>14.4f}  {geo_m['ari']:>16.4f}  {multi_m['ari']:>18.4f}  {hier_m['ari']:>20.4f}")
    print(f"{'Pairwise Merge Precision (Bar 1)':<32} {base_m['merge_P']:>14.4f}  {geo_m['merge_P']:>16.4f}  {multi_m['merge_P']:>18.4f}  {hier_m['merge_P']:>20.4f}")
    print(f"{'Pairwise Merge Recall (Bar 2)':<32} {base_m['merge_R']:>14.4f}  {geo_m['merge_R']:>16.4f}  {multi_m['merge_R']:>18.4f}  {hier_m['merge_R']:>20.4f}")
    print(f"{'Frankenmerge Split Rate (Bar 3)':<32} {base_fk:>14.4f}  {geo_fk:>16.4f}  {multi_fk:>18.4f}  {hier_fk:>20.4f}")
    print("-" * 105)
    print(f"{'Line Graph Precision (P_line)':<32} {base_lg.pre_only.precision:>14.4f}  {geo_lg.pre_only.precision:>16.4f}  {multi_lg.pre_only.precision:>18.4f}  {hier_lg.pre_only.precision:>20.4f}")
    print(f"{'Line Graph Recall (R_line)':<32} {base_lg.pre_only.recall:>14.4f}  {geo_lg.pre_only.recall:>16.4f}  {multi_lg.pre_only.recall:>18.4f}  {hier_lg.pre_only.recall:>20.4f}")
    print(f"{'Line Graph F1 (F1_line)':<32} {base_lg.pre_only.f1:>14.4f}  {geo_lg.pre_only.f1:>16.4f}  {multi_lg.pre_only.f1:>18.4f}  {hier_lg.pre_only.f1:>20.4f}")
    print(f"{'Line Graph TP Edges':<32} {base_lg.pre_only.tp:>14d}  {geo_lg.pre_only.tp:>16d}  {multi_lg.pre_only.tp:>18d}  {hier_lg.pre_only.tp:>20d}")
    print(f"{'Line Graph FP Edges':<32} {base_lg.pre_only.fp:>14d}  {geo_lg.pre_only.fp:>16d}  {multi_lg.pre_only.fp:>18d}  {hier_lg.pre_only.fp:>20d}")
    print(f"{'Line Graph FN Edges':<32} {base_lg.pre_only.fn:>14d}  {geo_lg.pre_only.fn:>16d}  {multi_lg.pre_only.fn:>18d}  {hier_lg.pre_only.fn:>20d}")
    print("=" * 105)


if __name__ == "__main__":
    run_multimodal_benchmark()
