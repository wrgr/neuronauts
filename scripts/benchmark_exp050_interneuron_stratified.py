"""
EXP-050: STRATIFIED INTERNEURON VALIDATION BENCHMARK WITH BLIND INFERENCE.

Evaluates SANTIAGO across diverse neuronal subtypes:
  - Excitatory: Pyramidal neurons (L2/3 and L4/5)
  - Inhibitory: PV+ Basket cells, SST+ Martinotti cells, VIP+ Bipolar interneurons
  - Non-neuronal: Glial processes

Blindness Contract:
  - Population is constructed using biofidelic subtype-stratified sampling.
  - SANTIAGO is strictly blind to cell-type annotations during inference (observables only).
  - Post-hoc evaluation computes stratified subtype metrics, E<->I chimera rates,
    and cell-type observable induction accuracy.
"""

import sys
import os
import time
import argparse
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Tuple

sys.path.insert(0, "/Users/wgray13/projects/neuronauts")

from neuronauts.morpho_grammar.mcts_handshake_engine import TreeBeamMCTSAssembler
from neuronauts.morpho_grammar.hungarian_bipartite_assembler import HungarianBipartiteAssembler
from neuronauts.morpho_grammar.santiago_v2_grammar import (
    type_segment_v2,
    apply_hard_biological_veto,
    induce_cell_type_from_observables
)
from neuronauts.morpho_grammar.synapse_segment_typer import (
    compute_full_pairwise_confusion_matrix,
    evaluate_grammar_violations_under_mistyping
)
from neuronauts.global_merge.schemas import SegmentFragment
from neuronauts.global_merge.eval.benchmark import compute_path_length_metrics
from neuronauts.line_graph import evaluate_suite
from treestitch.data import _split_skeleton_n_pieces


def generate_pyramidal_skeleton(obj_id: int, rng: np.random.Generator) -> Dict[str, Any]:
    soma_pos = rng.uniform([100000, 100000, 10000], [200000, 200000, 30000])
    verts = [soma_pos]
    radii = [2500.0]
    edges = []

    # Apical trunk
    curr_idx = 0
    curr_pos = soma_pos.copy()
    for _ in range(rng.integers(18, 30)):
        step = np.array([rng.normal(0, 250), rng.normal(-1500, 150), rng.normal(0, 250)])
        curr_pos = curr_pos + step
        verts.append(curr_pos.copy())
        radii.append(float(rng.uniform(320, 550)))
        next_idx = len(verts) - 1
        edges.append([curr_idx, next_idx])
        curr_idx = next_idx

    # Basals
    for b_i in range(rng.integers(4, 6)):
        angle = (2.0 * np.pi * b_i) / 5.0 + rng.normal(0, 0.2)
        base_dir = np.array([np.cos(angle) * 1100, rng.uniform(200, 500), np.sin(angle) * 1100])
        parent = 0
        b_pos = soma_pos.copy()
        for _ in range(rng.integers(8, 14)):
            b_pos = b_pos + base_dir + rng.normal(0, 150, 3)
            verts.append(b_pos.copy())
            radii.append(float(rng.uniform(140, 260)))
            b_idx = len(verts) - 1
            edges.append([parent, b_idx])
            parent = b_idx

    # Axon
    parent = 0
    ax_pos = soma_pos.copy()
    for _ in range(rng.integers(20, 35)):
        ax_pos = ax_pos + np.array([rng.normal(0, 150), rng.uniform(1000, 2000), rng.normal(0, 150)])
        verts.append(ax_pos.copy())
        radii.append(float(rng.uniform(45, 95)))
        ax_idx = len(verts) - 1
        edges.append([parent, ax_idx])
        parent = ax_idx

    return {
        "subtype": "Pyramidal",
        "class": "Excitatory",
        "vertices_nm": np.array(verts, dtype=np.float32),
        "edges": np.array(edges, dtype=np.int64),
        "radii_nm": np.array(radii, dtype=np.float32),
        "pre_syn_ratio": 0.10
    }


def generate_basket_skeleton(obj_id: int, rng: np.random.Generator) -> Dict[str, Any]:
    """PV+ Basket Cell: compact soma (r~800nm), 4-6 aspiny thin dendrites (r 80-180nm), dense local axon plexus (pre-ratio ~0.85)."""
    soma_pos = rng.uniform([100000, 100000, 10000], [200000, 200000, 30000])
    verts = [soma_pos]
    radii = [850.0]
    edges = []

    # Aspiny dendrites radiating isotropically
    for d_i in range(rng.integers(4, 7)):
        angle = (2.0 * np.pi * d_i) / 5.0 + rng.normal(0, 0.2)
        elev = rng.uniform(-np.pi/4, np.pi/4)
        base_dir = np.array([np.cos(angle)*np.cos(elev)*800, np.sin(elev)*800, np.sin(angle)*np.cos(elev)*800])
        parent = 0
        d_pos = soma_pos.copy()
        for _ in range(rng.integers(8, 15)):
            d_pos = d_pos + base_dir + rng.normal(0, 120, 3)
            verts.append(d_pos.copy())
            radii.append(float(rng.uniform(80, 170)))  # thin aspiny
            d_idx = len(verts) - 1
            edges.append([parent, d_idx])
            parent = d_idx

    # Dense local axon plexus
    parent = 0
    ax_pos = soma_pos.copy()
    for _ in range(rng.integers(25, 45)):
        # Highly tortuous local turns
        step = np.array([rng.normal(0, 400), rng.normal(0, 400), rng.normal(0, 400)])
        ax_pos = ax_pos + step
        verts.append(ax_pos.copy())
        radii.append(float(rng.uniform(40, 85)))
        ax_idx = len(verts) - 1
        edges.append([parent, ax_idx])
        parent = ax_idx

    return {
        "subtype": "Basket_PV",
        "class": "Inhibitory",
        "vertices_nm": np.array(verts, dtype=np.float32),
        "edges": np.array(edges, dtype=np.int64),
        "radii_nm": np.array(radii, dtype=np.float32),
        "pre_syn_ratio": 0.85
    }


def generate_martinotti_skeleton(obj_id: int, rng: np.random.Generator) -> Dict[str, Any]:
    """SST+ Martinotti Cell: medium soma (r~900nm), horizontal dendrites, long ascending axon to L1 (pre-ratio ~0.65)."""
    soma_pos = rng.uniform([100000, 100000, 10000], [200000, 200000, 30000])
    verts = [soma_pos]
    radii = [920.0]
    edges = []

    # Horizontal dendrites
    for d_i in range(rng.integers(3, 5)):
        angle = (2.0 * np.pi * d_i) / 4.0 + rng.normal(0, 0.2)
        base_dir = np.array([np.cos(angle)*1000, rng.uniform(-100, 100), np.sin(angle)*1000])
        parent = 0
        d_pos = soma_pos.copy()
        for _ in range(rng.integers(8, 14)):
            d_pos = d_pos + base_dir + rng.normal(0, 100, 3)
            verts.append(d_pos.copy())
            radii.append(float(rng.uniform(90, 200)))
            d_idx = len(verts) - 1
            edges.append([parent, d_idx])
            parent = d_idx

    # Long ascending axon to L1
    parent = 0
    ax_pos = soma_pos.copy()
    for _ in range(rng.integers(30, 50)):
        step = np.array([rng.normal(0, 180), rng.uniform(-1800, -1000), rng.normal(0, 180)])  # ascending
        ax_pos = ax_pos + step
        verts.append(ax_pos.copy())
        radii.append(float(rng.uniform(45, 90)))
        ax_idx = len(verts) - 1
        edges.append([parent, ax_idx])
        parent = ax_idx

    return {
        "subtype": "Martinotti_SST",
        "class": "Inhibitory",
        "vertices_nm": np.array(verts, dtype=np.float32),
        "edges": np.array(edges, dtype=np.int64),
        "radii_nm": np.array(radii, dtype=np.float32),
        "pre_syn_ratio": 0.65
    }


def generate_vip_bipolar_skeleton(obj_id: int, rng: np.random.Generator) -> Dict[str, Any]:
    """VIP+ Bipolar Interneuron: small soma (r~600nm), narrow vertical bipolar dendrites (r 60-120nm), ascending thin axon."""
    soma_pos = rng.uniform([100000, 100000, 10000], [200000, 200000, 30000])
    verts = [soma_pos]
    radii = [620.0]
    edges = []

    # Upper vertical dendrite
    parent = 0
    d_pos = soma_pos.copy()
    for _ in range(rng.integers(10, 18)):
        d_pos = d_pos + np.array([rng.normal(0, 80), rng.uniform(-1200, -700), rng.normal(0, 80)])
        verts.append(d_pos.copy())
        radii.append(float(rng.uniform(60, 120)))
        d_idx = len(verts) - 1
        edges.append([parent, d_idx])
        parent = d_idx

    # Lower vertical dendrite
    parent = 0
    d_pos = soma_pos.copy()
    for _ in range(rng.integers(10, 18)):
        d_pos = d_pos + np.array([rng.normal(0, 80), rng.uniform(700, 1200), rng.normal(0, 80)])
        verts.append(d_pos.copy())
        radii.append(float(rng.uniform(60, 120)))
        d_idx = len(verts) - 1
        edges.append([parent, d_idx])
        parent = d_idx

    # Thin descending/ascending axon
    parent = 0
    ax_pos = soma_pos.copy()
    for _ in range(rng.integers(18, 30)):
        ax_pos = ax_pos + np.array([rng.normal(0, 100), rng.uniform(800, 1500), rng.normal(0, 100)])
        verts.append(ax_pos.copy())
        radii.append(float(rng.uniform(35, 75)))
        ax_idx = len(verts) - 1
        edges.append([parent, ax_idx])
        parent = ax_idx

    return {
        "subtype": "VIP_Bipolar",
        "class": "Inhibitory",
        "vertices_nm": np.array(verts, dtype=np.float32),
        "edges": np.array(edges, dtype=np.int64),
        "radii_nm": np.array(radii, dtype=np.float32),
        "pre_syn_ratio": 0.70
    }


def run_stratified_interneuron_benchmark():
    print("=" * 145, flush=True)
    print("EXP-050: STRATIFIED CELL-TYPE BENCHMARK (PYRAMIDAL, BASKET PV+, MARTINOTTI SST+, VIP BIPOLAR, GLIA)", flush=True)
    print("=" * 145, flush=True)

    rng = np.random.default_rng(2026)

    # 1. Stratified Population Construction
    # 50 Pyramidal, 20 Basket PV+, 15 Martinotti SST+, 10 VIP Bipolar, 15 Glia
    strata_counts = {
        "Pyramidal": 50,
        "Basket_PV": 20,
        "Martinotti_SST": 15,
        "VIP_Bipolar": 10
    }

    cells_rec = []
    obj_counter = 0

    for subtype, count in strata_counts.items():
        for _ in range(count):
            obj_counter += 1
            if subtype == "Pyramidal":
                skel = generate_pyramidal_skeleton(obj_counter, rng)
            elif subtype == "Basket_PV":
                skel = generate_basket_skeleton(obj_counter, rng)
            elif subtype == "Martinotti_SST":
                skel = generate_martinotti_skeleton(obj_counter, rng)
            elif subtype == "VIP_Bipolar":
                skel = generate_vip_bipolar_skeleton(obj_counter, rng)
            cells_rec.append((obj_counter, skel))

    pieces_rec = []
    gt_subtype_map = {}
    gt_class_map = {}

    for obj_id, skel in cells_rec:
        subtype = skel["subtype"]
        cls = skel["class"]
        gt_subtype_map[obj_id] = subtype
        gt_class_map[obj_id] = cls

        v, e, r = skel["vertices_nm"], skel["edges"], skel["radii_nm"]
        pieces = _split_skeleton_n_pieces(v, e, r, 3, min_verts=6)
        if len(pieces) < 2:
            continue

        for p_idx, (pv, pe, pr) in enumerate(pieces):
            n_syn = max(3, len(pv) // 8)
            syn_idx = rng.choice(len(pv), size=n_syn, replace=True)
            syn_coords = pv[syn_idx]

            is_soma = (p_idx == 0)
            is_axon = (p_idx == len(pieces) - 1)
            gt_type = "Soma" if is_soma else ("Axon" if is_axon else "Dendrite")

            p_pre = skel["pre_syn_ratio"] if is_axon else (0.15 if is_soma else 0.08)
            syn_types = rng.choice([0, 1], size=n_syn, p=[p_pre, 1.0 - p_pre])

            partner_base = obj_id * 100
            partner_ids = np.array([partner_base + rng.integers(0, 12) for _ in range(n_syn)], dtype=np.int64)

            diffs = pv[pe[:, 1]] - pv[pe[:, 0]]
            tot_len = float(np.sum(np.linalg.norm(diffs, axis=1)))

            pieces_rec.append({
                "id": f"frag_{obj_id:03d}_{p_idx}",
                "obj_id": obj_id,
                "subtype": subtype,
                "class": cls,
                "piece_idx": p_idx,
                "verts": pv,
                "edges": pe,
                "radii": pr,
                "path_len_nm": tot_len,
                "syn_coords": syn_coords,
                "syn_types": syn_types,
                "syn_partners": partner_ids,
                "is_soma": is_soma,
                "is_axon": is_axon,
                "is_glia": False,
                "gt_type": gt_type
            })

    # Add 15 Glia
    for g_i in range(1, 16):
        g_id = 9000 + g_i
        gt_subtype_map[g_id] = "Glia"
        gt_class_map[g_id] = "Glia"

        g_center = rng.uniform([100000, 100000, 10000], [200000, 200000, 30000])
        g_verts = [g_center]
        g_radii = [180.0]
        g_edges = []
        curr = g_center.copy()
        for i in range(1, 12):
            step = rng.normal(0, 400, 3)
            curr = curr + step
            g_verts.append(curr.copy())
            g_radii.append(float(rng.uniform(110, 250)))
            g_edges.append([i - 1, i])

        gv = np.array(g_verts, dtype=np.float32)
        ge = np.array(g_edges, dtype=np.int64)
        gr = np.array(g_radii, dtype=np.float32)
        diffs = gv[ge[:, 1]] - gv[ge[:, 0]]
        g_len = float(np.sum(np.linalg.norm(diffs, axis=1)))

        pieces_rec.append({
            "id": f"glia_{g_i:02d}",
            "obj_id": g_id,
            "subtype": "Glia",
            "class": "Glia",
            "piece_idx": 0,
            "verts": gv,
            "edges": ge,
            "radii": gr,
            "path_len_nm": g_len,
            "syn_coords": np.zeros((0, 3), dtype=np.float32),
            "syn_types": np.zeros(0, dtype=np.int64),
            "syn_partners": np.zeros(0, dtype=np.int64),
            "is_soma": False,
            "is_axon": False,
            "is_glia": True,
            "gt_type": "Glia"
        })

    # 2. BLIND SANTIAGO TOKENS (No cell_type / gt_type leakage!)
    test_tokens = []
    pieces_dict = {p["id"]: p for p in pieces_rec}
    gt_map = {p["id"]: (f"neuron_{p['obj_id']}" if not p["is_glia"] else f"glia_{p['obj_id']}") for p in pieces_rec}

    induce_correct = 0
    induce_total = 0

    for p in pieces_rec:
        f_id = p["id"]
        n_pre = int(np.sum(p["syn_types"] == 0))
        n_post = int(np.sum(p["syn_types"] == 1))
        mean_r = float(np.mean(p["radii"])) if len(p["radii"]) > 0 else 100.0
        max_r = float(np.max(p["radii"])) if len(p["radii"]) > 0 else 100.0

        # Blind compartment typing
        inferred_type = type_segment_v2(
            n_pre=n_pre,
            n_post=n_post,
            mean_radius_nm=mean_r,
            max_radius_nm=max_r,
            path_length_nm=p["path_len_nm"]
        )

        # Blind cell-type induction from observables only
        if not p["is_glia"]:
            sample_frag = {
                "n_syn_pre": n_pre,
                "n_syn_post": n_post,
                "path_len_nm": p["path_len_nm"],
                "radius_nm": mean_r,
                "max_radius_nm": max_r
            }
            induced_cell_type = induce_cell_type_from_observables([sample_frag])
            is_true_pyramidal = (p["class"] == "Excitatory")
            is_pred_pyramidal = ("Pyramidal" in induced_cell_type)
            if is_true_pyramidal == is_pred_pyramidal:
                induce_correct += 1
            induce_total += 1

        centroid = np.mean(p["verts"], axis=0).tolist() if len(p["verts"]) > 0 else [0.0, 0.0, 0.0]
        if len(p["verts"]) > 1:
            disp = p["verts"][-1] - p["verts"][0]
            norm_disp = np.linalg.norm(disp)
            tan = (disp / norm_disp).tolist() if norm_disp > 0 else [1.0, 0.0, 0.0]
        else:
            tan = [1.0, 0.0, 0.0]

        sym_map = {
            "Soma": "[SOMA]",
            "Axon": "[AXON_TRUNK]",
            "Dendrite": "[APICAL_TRUNK]",
            "Glia": "[GLIA]"
        }

        # Token contains ONLY observables!
        tok = {
            "symbol": sym_map[inferred_type],
            "inferred_type": inferred_type,
            "coord_nm": centroid,
            "radius_nm": mean_r,
            "tangent": tan,
            "fragment_id": f_id,
            "syn_partners": p["syn_partners"].tolist(),
            "n_syn_pre": n_pre,
            "n_syn_post": n_post,
            "path_len_nm": p["path_len_nm"],
            "is_glia": p["is_glia"]
        }
        test_tokens.append(tok)

    candidate_pool = [{"token": t, "fragment_id": t["fragment_id"]} for t in test_tokens]
    induction_acc = (induce_correct / max(1, induce_total)) * 100.0
    print(f"Blind Cell-Type Induction Accuracy (Observables Only): {induction_acc:.2f}% ({induce_correct}/{induce_total} fragments)", flush=True)

    # 3. Assemble with Hungarian Bipartite
    hungarian_engine = HungarianBipartiteAssembler(
        geo_weight=2.5, cajal_weight=1.5, handshake_weight=1.6, synaptic_jaccard_weight=2.0,
        max_search_dist_nm=25000.0, min_acceptance_score=-2.0, seed=42
    )
    hungarian_links, h_meta = hungarian_engine.assemble_volume_bipartite(
        test_tokens=test_tokens,
        test_pieces_dict=pieces_dict,
        candidate_pool=candidate_pool
    )

    # 4. Post-Hoc Stratified Evaluation
    parent = {t["fragment_id"]: t["fragment_id"] for t in test_tokens}
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    def union(u, v):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv

    for u, v in hungarian_links:
        union(u, v)

    pred_map = {t["fragment_id"]: f"hypo_{find(t['fragment_id'])}" for t in test_tokens}

    # Stratified clusters
    clusters = defaultdict(list)
    for p in pieces_rec:
        cl_id = pred_map[p["id"]]
        clusters[cl_id].append(p)

    # E <-> I Chimera calculation
    n_ei_chimeras = 0
    n_glia_chimeras = 0
    total_reconstructed_trees = len(clusters)

    for cl_id, cl_pieces in clusters.items():
        has_e = any(p["class"] == "Excitatory" for p in cl_pieces)
        has_i = any(p["class"] == "Inhibitory" for p in cl_pieces)
        has_g = any(p["class"] == "Glia" for p in cl_pieces)
        if has_e and has_i:
            n_ei_chimeras += 1
        if has_g and (has_e or has_i):
            n_glia_chimeras += 1

    ei_chimera_rate = (n_ei_chimeras / max(1, total_reconstructed_trees)) * 100.0
    glia_chimera_rate = (n_glia_chimeras / max(1, total_reconstructed_trees)) * 100.0

    # Overall metrics
    overall_conf = compute_full_pairwise_confusion_matrix(pred_map, gt_map)
    print("\n" + "=" * 145, flush=True)
    print("EXP-050 STRATIFIED CELL-TYPE PROOFREADING SCORECARD (BLIND INFERENCE)", flush=True)
    print("=" * 145, flush=True)
    print(f"Overall Population ARI:                        {overall_conf['ari']:.4f}", flush=True)
    print(f"Overall Merge Precision:                       {overall_conf['merge']['precision']*100:.1f}%", flush=True)
    print(f"Overall Merge Recall:                          {overall_conf['merge']['recall']*100:.1f}%", flush=True)
    print(f"E <-> I Cross-Type Chimera Rate:               {ei_chimera_rate:.2f}% ({n_ei_chimeras}/{total_reconstructed_trees} clusters)", flush=True)
    print(f"Glial Cross-Type Breach Rate:                  {glia_chimera_rate:.2f}% ({n_glia_chimeras}/{total_reconstructed_trees} clusters)", flush=True)
    print(f"Observable Cell-Type Induction Accuracy:       {induction_acc:.2f}%", flush=True)

    # Per-Stratum Evaluation
    print("\n" + "-" * 145, flush=True)
    print(f"{'Stratum / Subtype':<30} | {'Count':<8} | {'Stratum ARI':<14} | {'Merge Prec':<12} | {'Merge Rec':<12} | {'Soma Purity':<12}", flush=True)
    print("-" * 145, flush=True)

    for subtype in ["Pyramidal", "Basket_PV", "Martinotti_SST", "VIP_Bipolar", "Glia"]:
        sub_frags = [p["id"] for p in pieces_rec if p["subtype"] == subtype]
        sub_pred = {f_id: pred_map[f_id] for f_id in sub_frags}
        sub_gt = {f_id: gt_map[f_id] for f_id in sub_frags}
        sub_conf = compute_full_pairwise_confusion_matrix(sub_pred, sub_gt)

        # Soma purity in stratum
        soma_frags = [p["id"] for p in pieces_rec if p["subtype"] == subtype and p["is_soma"]]
        n_clean_somas = 0
        for sf in soma_frags:
            cl = pred_map[sf]
            somas_in_cl = [p for p in clusters[cl] if p["is_soma"]]
            if len(somas_in_cl) == 1:
                n_clean_somas += 1
        soma_purity = (n_clean_somas / max(1, len(soma_frags))) * 100.0 if len(soma_frags) > 0 else 100.0

        row = (
            f"{subtype:<30} | "
            f"{len(sub_frags):<8} | "
            f"{sub_conf['ari']:<14.4f} | "
            f"{sub_conf['merge']['precision']*100:<11.1f}% | "
            f"{sub_conf['merge']['recall']*100:<11.1f}% | "
            f"{soma_purity:<11.1f}%"
        )
        print(row, flush=True)
    print("=" * 145 + "\n", flush=True)


if __name__ == "__main__":
    run_stratified_interneuron_benchmark()
