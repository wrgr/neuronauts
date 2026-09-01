#!/usr/bin/env python3
"""
Bar 3 Frankenmerge Benchmark:
Compares Naive Union-Find, Edge-CC, and Next-Gen Global Merge (Pre-Splitting + Tangent Flow + Lifted Multicut)
on real Minnie65 adjacent-neuron frankenmerges.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuronauts.data.loaders import load_skeleton, sample_neurons
from treestitch.data import _split_skeleton_n_pieces
from treestitch.worldbuild import frankenmerge_adjacent
from neuronauts.global_merge.schemas import SegmentFragment, EndpointTangent, EdgeType, AssemblyEdge
from neuronauts.global_merge.represent.tangent_flow import extract_endpoints_from_skeleton
from neuronauts.global_merge.data.cave_lineage import pre_split_frankenmerges
from neuronauts.global_merge.solver.constrained_multicut import assemble_global_connectome
from neuronauts.global_merge.eval.benchmark import compute_pairwise_partition_metrics, evaluate_frankenmerge_split_rate


def run_benchmark(n_objects: int = 15, n_pieces: int = 3, franken_frac: float = 0.30, franken_radius_nm: float = 6000.0, seed: int = 42):
    print("=" * 80)
    print(f"BAR 3 BENCHMARK: REAL MINNIE65 FRANKENMERGE RESOLUTION")
    print(f"Neurons: {n_objects} | Pieces/Neuron: {n_pieces} | Frankenmerge Frac: {franken_frac:.2f} | Radius: {franken_radius_nm:.0f} nm")
    print("=" * 80)

    # 1. Fetch real neurons
    candidates = sample_neurons(n_objects * 4, seed=seed)
    pieces_rec = []
    obj_counter = 0
    rng = np.random.default_rng(seed)

    for root_id in candidates:
        if obj_counter >= n_objects:
            break
        skel = load_skeleton(root_id)
        if skel is None:
            continue

        verts, edges_raw, radii = skel["vertices_nm"], skel["edges"], skel["radii_nm"]
        if len(verts) < 24 or len(verts) > 8000:
            continue

        pieces = _split_skeleton_n_pieces(verts, edges_raw, radii, n_pieces, min_verts=8)
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
        print(f"  [{obj_counter:3d}] root={root_id}  V={len(verts)}  pieces={len(pieces)}")

    print(f"\nExtracted {len(pieces_rec)} pieces across {obj_counter} real proofread neurons.")

    # 2. Inject realistic adjacent-neuron frankenmerges
    seg_of_piece, n_franken = frankenmerge_adjacent(
        pieces_rec, franken_frac, rng, radius_nm=franken_radius_nm
    )
    print(f"Injected {n_franken} adjacent-neuron frankenmerges.")

    # 3. Build input fragments
    fragments: list[SegmentFragment] = []
    gt_map: dict[str, str] = {}

    for idx, p in enumerate(pieces_rec):
        f_id = f"piece_{idx:03d}"
        gt_map[f_id] = str(p["obj_id"])
        seg_id = int(seg_of_piece[idx])

        v = p["verts"].astype(np.float32)
        r = p["radii"].astype(np.float32)
        e = p["edges"].astype(np.int64)

        eps = extract_endpoints_from_skeleton(f_id, v, r, e)
        is_soma = (p["piece_idx"] == 0)

        frag = SegmentFragment(
            fragment_id=f_id,
            segment_id=seg_id,
            vertices_nm=v,
            radii_nm=r,
            edges=e,
            endpoints=eps,
            is_soma=is_soma,
            soma_confidence=1.0 if is_soma else 0.0
        )
        fragments.append(frag)

    # 4. Method A: Naive Union-Find Baseline (trusts v117 segment IDs blindly)
    pred_uf = {f.fragment_id: str(f.segment_id) for f in fragments}
    m_uf = compute_pairwise_partition_metrics(pred_uf, gt_map)
    fk_uf = evaluate_frankenmerge_split_rate(pred_uf, gt_map, fragments)

    # 5. Method B: Next-Gen Global Merge (Pre-Splitting + Tangent-Flow + Constrained Multicut)
    clean_fragments = []
    for f in fragments:
        split_frags = pre_split_frankenmerges(f, max_radius_ratio=2.5)
        clean_fragments.extend(split_frags)

    res_gm = assemble_global_connectome(
        clean_fragments,
        enable_tangent_flow=True,
        max_tangent_dist_nm=18000.0,
        min_collinearity=0.25
    )

    pred_gm = {}
    for f in fragments:
        assigned_neurons = [res_gm.fragment_to_neuron[sf.fragment_id] for sf in clean_fragments if sf.fragment_id.startswith(f.fragment_id) and sf.fragment_id in res_gm.fragment_to_neuron]
        if assigned_neurons:
            pred_gm[f.fragment_id] = assigned_neurons[0]
        else:
            pred_gm[f.fragment_id] = f.fragment_id

    m_gm = compute_pairwise_partition_metrics(pred_gm, gt_map)
    fk_gm = evaluate_frankenmerge_split_rate(pred_gm, gt_map, fragments)

    print()
    print("=" * 80)
    print("BAR 3 FRANKENMERGE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Method':<25} {'ARI':>8} {'Merge_P':>10} {'Merge_R':>10} {'fk_split (Bar 3)':>18}")
    print("-" * 80)
    print(f"{'Naive v117 (UF)':<25} {m_uf['ari']:>8.4f} {m_uf['merge_P']:>10.4f} {m_uf['merge_R']:>10.4f} {fk_uf:>18.4f}")
    print(f"{'Next-Gen Global Merge':<25} {m_gm['ari']:>8.4f} {m_gm['merge_P']:>10.4f} {m_gm['merge_R']:>10.4f} {fk_gm:>18.4f}")
    print("=" * 80)
    print(f"• ΔARI:              {m_gm['ari'] - m_uf['ari']:+.4f}")
    print(f"• Δfk_split (Bar 3): {fk_gm - fk_uf:+.4f} (Bar 3 Target: > 0.50)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-objects", type=int, default=15)
    parser.add_argument("--n-pieces", type=int, default=3)
    parser.add_argument("--franken-frac", type=float, default=0.30)
    parser.add_argument("--radius-nm", type=float, default=6000.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(
        n_objects=args.n_objects,
        n_pieces=args.n_pieces,
        franken_frac=args.franken_frac,
        franken_radius_nm=args.radius_nm,
        seed=args.seed
    )
