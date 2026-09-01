#!/usr/bin/env python3
"""
Dense Multi-Region Global Merge Benchmark:
Evaluates Next-Gen Global Merge & Assembly at scale across 60+ real Minnie65 proofread neurons
with leak-free train/test spatial separation.
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuronauts.data.loaders import load_skeleton, sample_neurons
from treestitch.data import _split_skeleton_n_pieces
from treestitch.worldbuild import frankenmerge_adjacent
from neuronauts.global_merge.schemas import SegmentFragment, EndpointTangent, EdgeType, AssemblyEdge
from neuronauts.global_merge.represent.tangent_flow import extract_endpoints_from_skeleton
from neuronauts.global_merge.represent.vicreg_gnn import VICRegSkeletonModel, train_contrastive_skeleton_gnn
from neuronauts.global_merge.data.cave_lineage import pre_split_frankenmerges
from neuronauts.global_merge.solver.constrained_multicut import assemble_global_connectome
from neuronauts.global_merge.eval.benchmark import compute_pairwise_partition_metrics, evaluate_frankenmerge_split_rate
from neuronauts.schemas import Fragment as OldFragment


def run_dense_multi_region_benchmark(
    n_total_neurons: int = 60,
    n_pieces_per_neuron: int = 3,
    franken_frac: float = 0.35,
    franken_radius_nm: float = 6000.0,
    train_frac: float = 0.70,
    train_epochs: int = 40,
    seed: int = 42
):
    print("=" * 80)
    print(f"RUNNING DENSE MULTI-REGION GLOBAL MERGE BENCHMARK (60+ REAL MINNIE65 NEURONS)")
    print(f"Total Neurons: {n_total_neurons} | Pieces/Cell: {n_pieces_per_neuron} | Franken Frac: {franken_frac:.2f} | Radius: {franken_radius_nm:.0f} nm")
    print("=" * 80)

    # 1. Fetch real proofread skeletons from Minnie65
    candidates = sample_neurons(n_total_neurons * 3, seed=seed)
    pieces_rec = []
    obj_counter = 0
    rng = np.random.default_rng(seed)

    print("\n[1/5] Loading real Minnie65 pyramidal neurons...")
    for root_id in candidates:
        if obj_counter >= n_total_neurons:
            break
        skel = load_skeleton(root_id)
        if skel is None:
            continue

        verts, edges_raw, radii = skel["vertices_nm"], skel["edges"], skel["radii_nm"]
        if len(verts) < 24 or len(verts) > 8000:
            continue

        pieces = _split_skeleton_n_pieces(verts, edges_raw, radii, n_pieces_per_neuron, min_verts=8)
        if len(pieces) < 2:
            continue

        obj_counter += 1
        for p_idx, (pv, pe, pr) in enumerate(pieces):
            anchor = rng.integers(0, len(pv), 5)
            obs_pts = (pv[anchor] + rng.normal(0, 500.0, (5, 3)).astype(np.float32))
            pieces_rec.append({
                "obj_id": obj_counter,
                "piece_idx": p_idx,
                "verts": pv,
                "edges": pe if len(pe) else np.zeros((0, 2), dtype=np.int64),
                "radii": pr,
                "obs_pts": obs_pts,
            })
        if obj_counter % 10 == 0 or obj_counter == n_total_neurons:
            print(f"  Loaded {obj_counter}/{n_total_neurons} neurons ({len(pieces_rec)} fragments)...")

    print(f"\nSuccessfully extracted {len(pieces_rec)} fragments across {obj_counter} real proofread neurons.")

    # 2. Inject realistic adjacent-neuron contact frankenmerges
    print("\n[2/5] Synthesizing realistic membrane contact frankenmerges...")
    seg_of_piece, n_franken = frankenmerge_adjacent(
        pieces_rec, franken_frac, rng, radius_nm=franken_radius_nm
    )
    print(f"Injected {n_franken} adjacent-neuron frankenmerges across dense neuropil.")

    # 3. Split into Train & Out-of-Sample Test partitions
    n_train_neurons = int(round(train_frac * obj_counter))
    train_indices = [i for i, p in enumerate(pieces_rec) if p["obj_id"] <= n_train_neurons]
    test_indices = [i for i, p in enumerate(pieces_rec) if p["obj_id"] > n_train_neurons]

    print(f"\n[3/5] Dataset Partitioning:")
    print(f"  Train: {n_train_neurons} neurons ({len(train_indices)} fragments)")
    print(f"  Test:  {obj_counter - n_train_neurons} neurons ({len(test_indices)} fragments, out-of-sample)")

    # Prepare training fragments
    old_frags = []
    for idx, p in enumerate(pieces_rec):
        old_frags.append(OldFragment(
            fragment_id=idx,
            region_id="minnie65",
            base_root_id=p["obj_id"],
            vertices_nm=p["verts"].astype(np.float32),
            radius_nm=p["radii"].astype(np.float32),
            edges=p["edges"].astype(np.int64),
            endpoints_nm=np.zeros((0, 3), dtype=np.float32),
            synapse_indices=np.zeros(0, dtype=np.int64)
        ))

    train_frags = [old_frags[i] for i in train_indices]
    train_pieces = [pieces_rec[i] for i in train_indices]

    train_obj_to_indices = {}
    for local_idx, p in enumerate(train_pieces):
        train_obj_to_indices.setdefault(p["obj_id"], []).append(local_idx)

    pos_pairs = []
    for o_id, idxs in train_obj_to_indices.items():
        if len(idxs) >= 2:
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    pos_pairs.append((idxs[i], idxs[j]))

    neg_pairs = []
    for i in range(len(train_pieces)):
        for j in range(i + 1, len(train_pieces)):
            if train_pieces[i]["obj_id"] != train_pieces[j]["obj_id"]:
                neg_pairs.append((i, j))

    # 4. Train Contrastive Skeleton Model on Train Region
    print(f"\n[4/5] Training Contrastive Skeleton Model ({train_epochs} epochs on Train)...")
    model = VICRegSkeletonModel(in_dim=4, emb_dim=32, proj_dim=64)
    train_contrastive_skeleton_gnn(
        model, train_frags, pos_pairs, neg_pairs,
        n_epochs=train_epochs, lr=1e-3, margin_neg=0.30, std_coeff=10.0, log_every=10
    )

    # Encode all test fragments
    print("\n[5/5] Assembling and Evaluating Out-of-Sample Test Region...")
    test_frags = []
    test_gt_map = {}
    for local_idx, orig_idx in enumerate(test_indices):
        p = pieces_rec[orig_idx]
        f_id = f"test_piece_{local_idx:04d}"
        test_gt_map[f_id] = str(p["obj_id"])
        seg_id = int(seg_of_piece[orig_idx])

        v = p["verts"].astype(np.float32)
        r = p["radii"].astype(np.float32)
        e = p["edges"].astype(np.int64)

        eps = extract_endpoints_from_skeleton(f_id, v, r, e)
        is_soma = (p["piece_idx"] == 0)
        dna_emb = model.encode_fragment(v, r, e)

        frag = SegmentFragment(
            fragment_id=f_id,
            segment_id=seg_id,
            vertices_nm=v,
            radii_nm=r,
            edges=e,
            endpoints=eps,
            is_soma=is_soma,
            soma_confidence=1.0 if is_soma else 0.0,
            dna_embedding=dna_emb
        )
        test_frags.append(frag)

    # Naive baseline on test
    pred_uf = {f.fragment_id: str(f.segment_id) for f in test_frags}
    m_uf = compute_pairwise_partition_metrics(pred_uf, test_gt_map)
    fk_uf = evaluate_frankenmerge_split_rate(pred_uf, test_gt_map, test_frags)

    # Clean pre-split & Assemble Next-Gen
    clean_test_frags = []
    for f in test_frags:
        split_frags = pre_split_frankenmerges(f, max_radius_ratio=2.5)
        for sf in split_frags:
            sf.dna_embedding = f.dna_embedding
        clean_test_frags.extend(split_frags)

    # Dynamic empirical calibration from training distribution
    train_pos_cos = [float(np.dot(model.encode_fragment(train_pieces[i]['verts'], train_pieces[i]['radii'], train_pieces[i]['edges']), model.encode_fragment(train_pieces[j]['verts'], train_pieces[j]['radii'], train_pieces[j]['edges']))) for i, j in pos_pairs[:50]]
    train_neg_cos = [float(np.dot(model.encode_fragment(train_pieces[i]['verts'], train_pieces[i]['radii'], train_pieces[i]['edges']), model.encode_fragment(train_pieces[j]['verts'], train_pieces[j]['radii'], train_pieces[j]['edges']))) for i, j in neg_pairs[:50]]
    mu_pos = float(np.mean(train_pos_cos)) if train_pos_cos else 0.80
    mu_neg = float(np.mean(train_neg_cos)) if train_neg_cos else 0.25
    dynamic_theta = (mu_pos + mu_neg) / 2.0
    print(f"\n  [Dynamic Threshold Calibration] mu_pos={mu_pos:.4f} | mu_neg={mu_neg:.4f} => Dynamic theta* = {dynamic_theta:.4f}")

    res_gm = assemble_global_connectome(
        clean_test_frags,
        enable_tangent_flow=True,
        max_tangent_dist_nm=25000.0,
        min_collinearity=0.20,
        dna_split_threshold=dynamic_theta
    )

    pred_gm = {}
    for f in test_frags:
        assigned_neurons = [res_gm.fragment_to_neuron[sf.fragment_id] for sf in clean_test_frags if sf.fragment_id.startswith(f.fragment_id) and sf.fragment_id in res_gm.fragment_to_neuron]
        if assigned_neurons:
            pred_gm[f.fragment_id] = assigned_neurons[0]
        else:
            pred_gm[f.fragment_id] = f.fragment_id

    m_gm = compute_pairwise_partition_metrics(pred_gm, test_gt_map)
    fk_gm = evaluate_frankenmerge_split_rate(pred_gm, test_gt_map, test_frags)

    print()
    print("=" * 80)
    print("DENSE MULTI-REGION BENCHMARK RESULTS (OUT-OF-SAMPLE TEST EVALUATION)")
    print("=" * 80)
    print(f"{'Metric':<32} {'Naive v117 (Baseline)':>22} {'Next-Gen Global Merge':>22}")
    print("-" * 80)
    print(f"{'Out-of-Sample ARI (Bar 1 & 2)':<32} {m_uf['ari']:>22.4f} {m_gm['ari']:>22.4f}")
    print(f"{'Merge Precision (Bar 1 > 0.95)':<32} {m_uf['merge_P']:>22.4f} {m_gm['merge_P']:>22.4f}")
    print(f"{'Merge Recall (Bar 2 > 0.70)':<32} {m_uf['merge_R']:>22.4f} {m_gm['merge_R']:>22.4f}")
    print(f"{'Frankenmerge Split Rate (Bar 3)':<32} {fk_uf:>22.4f} {fk_gm:>22.4f}")
    print("=" * 80)
    print(f"✓ Bar 1: {'PASS' if m_gm['merge_P'] >= 0.95 else 'FAIL'} (Merge_P = {m_gm['merge_P']:.4f})")
    print(f"✓ Bar 2: {'PASS' if m_gm['ari'] >= 0.70 else 'FAIL'} (ARI = {m_gm['ari']:.4f})")
    print(f"✓ Bar 3: {'PASS' if fk_gm >= 0.50 else 'FAIL'} (fk_split = {fk_gm:.4f} vs Baseline {fk_uf:.4f})")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-neurons', type=int, default=60)
    parser.add_argument('--n-pieces', type=int, default=3)
    parser.add_argument('--franken-frac', type=float, default=0.35)
    parser.add_argument('--radius-nm', type=float, default=6000.0)
    parser.add_argument('--train-frac', type=float, default=0.70)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_dense_multi_region_benchmark(
        n_total_neurons=args.n_neurons,
        n_pieces_per_neuron=args.n_pieces,
        franken_frac=args.franken_frac,
        franken_radius_nm=args.radius_nm,
        train_frac=args.train_frac,
        train_epochs=args.epochs,
        seed=args.seed
    )
