"""
Synapse-Driven Segment Typing and Grammar Violation Evaluation.
Implements:
  1. Segment-level typing using presynaptic bouton count, postsynaptic density count, and caliber.
  2. Full Pairwise Merge & Split Contingency Matrix (TP, FP, FN, TN).
  3. Empirical Biological Grammar Violation tracking under real-world mistyping.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import numpy as np

from neuronauts.global_merge.eval.benchmark import adjusted_rand_index, _comb2


def type_segment_from_synapses(
    n_pre: int,
    n_post: int,
    mean_radius_nm: float,
    max_radius_nm: float
) -> str:
    """
    Types an EM segment using observable biological properties:
      - Soma: Large soma caliber (max radius > 1200 nm).
      - Axon: Dominated by presynaptic release sites (n_pre > n_post) or thin caliber (r < 100 nm).
      - Dendrite: Dominated by postsynaptic spines (n_post > n_pre) or thick caliber (r >= 100 nm).
    """
    if max_radius_nm > 1200.0 or mean_radius_nm > 800.0:
        return "Soma"
    
    total_syn = n_pre + n_post
    if total_syn > 0:
        if n_pre > n_post:
            return "Axon"
        elif n_post > n_pre:
            return "Dendrite"
    
    # Fallback to physical continuous caliber
    return "Axon" if mean_radius_nm < 105.0 else "Dendrite"


def compute_full_pairwise_confusion_matrix(
    pred_map: Dict[str, str],
    gt_map: Dict[str, str]
) -> Dict[str, Any]:
    """
    Computes complete Merge and Split Confusion Matrices:
      - Merge Class (Pairs in same cluster): Merge_TP, Merge_FP, Merge_FN, Merge_TN
      - Split Class (Pairs in different clusters): Split_TP, Split_FP, Split_FN, Split_TN
    """
    common_keys = sorted(list(set(pred_map.keys()).intersection(set(gt_map.keys()))))
    n = len(common_keys)
    total_pairs = _comb2(n)
    
    if n < 2:
        return {
            "ari": 1.0,
            "total_pairs": 0,
            "merge": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0},
            "split": {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0}
        }

    labels_pred = [pred_map[k] for k in common_keys]
    labels_gt = [gt_map[k] for k in common_keys]

    ari = float(adjusted_rand_index(labels_gt, labels_pred))

    contingency = defaultdict(lambda: defaultdict(int))
    a_dict = defaultdict(int)
    b_dict = defaultdict(int)

    for t, p in zip(labels_gt, labels_pred):
        contingency[t][p] += 1
        a_dict[t] += 1
        b_dict[p] += 1

    sum_comb_nij = sum(_comb2(count) for row in contingency.values() for count in row.values())
    sum_comb_gt = sum(_comb2(count) for count in a_dict.values())
    sum_comb_pred = sum(_comb2(count) for count in b_dict.values())

    # Merge Confusion Matrix (Positive = Same Cluster)
    merge_tp = sum_comb_nij
    merge_fp = sum_comb_pred - merge_tp
    merge_fn = sum_comb_gt - merge_tp
    merge_tn = total_pairs - (merge_tp + merge_fp + merge_fn)

    merge_p = float(merge_tp / (merge_tp + merge_fp)) if (merge_tp + merge_fp) > 0 else 1.0
    merge_r = float(merge_tp / (merge_tp + merge_fn)) if (merge_tp + merge_fn) > 0 else 1.0
    merge_f1 = float(2 * merge_p * merge_r / (merge_p + merge_r)) if (merge_p + merge_r) > 0 else 0.0
    merge_acc = float((merge_tp + merge_tn) / total_pairs) if total_pairs > 0 else 1.0

    # Split Confusion Matrix (Positive = Different Clusters)
    split_tp = merge_tn
    split_fp = merge_fn
    split_fn = merge_fp
    split_tn = merge_tp

    split_p = float(split_tp / (split_tp + split_fp)) if (split_tp + split_fp) > 0 else 1.0
    split_r = float(split_tp / (split_tp + split_fn)) if (split_tp + split_fn) > 0 else 1.0
    split_f1 = float(2 * split_p * split_r / (split_p + split_r)) if (split_p + split_r) > 0 else 0.0
    split_acc = float((split_tp + split_tn) / total_pairs) if total_pairs > 0 else 1.0

    return {
        "ari": ari,
        "total_pairs": total_pairs,
        "merge": {
            "tp": merge_tp, "fp": merge_fp, "fn": merge_fn, "tn": merge_tn,
            "precision": merge_p, "recall": merge_r, "f1": merge_f1, "accuracy": merge_acc
        },
        "split": {
            "tp": split_tp, "fp": split_fp, "fn": split_fn, "tn": split_tn,
            "precision": split_p, "recall": split_r, "f1": split_f1, "accuracy": split_acc
        }
    }


def evaluate_grammar_violations_under_mistyping(
    pred_map: Dict[str, str],
    gt_map: Dict[str, str],
    fragment_metadata: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates whether segment mistyping led to empirical Ground Truth Biological Grammar Violations:
      1. Multi-Soma Chimera: A reconstructed neuron containing >1 true distinct soma.
      2. Axon-Dendrite Chimera: Merging an axon fragment directly with a dendrite from another cell.
      3. Cross-Neuron Chimera: Merging fragments from multiple distinct ground-truth neurons.
    """
    clusters = defaultdict(list)
    meta_by_id = {f["id"]: f for f in fragment_metadata}

    for f_id, cluster_id in pred_map.items():
        if f_id in meta_by_id:
            clusters[cluster_id].append(meta_by_id[f_id])

    total_clusters = len(clusters)
    multi_soma_violations = 0
    axon_dendrite_violations = 0
    cross_neuron_violations = 0
    pure_clusters = 0

    for cluster_id, frags in clusters.items():
        gt_neuron_ids = set(gt_map[f["id"]] for f in frags if f["id"] in gt_map)
        n_somas = sum(1 for f in frags if f.get("gt_type") == "Soma" or f.get("is_soma", False))
        has_axon = any(f.get("gt_type") == "Axon" or f.get("is_axon", False) for f in frags)
        has_dendrite = any(f.get("gt_type") in ("Dendrite", "Apical", "Basal") for f in frags)

        is_violating = False

        if n_somas > 1:
            multi_soma_violations += 1
            is_violating = True

        if len(gt_neuron_ids) > 1:
            cross_neuron_violations += 1
            is_violating = True
            if has_axon and has_dendrite:
                axon_dendrite_violations += 1

        if not is_violating:
            pure_clusters += 1

    return {
        "total_clusters": total_clusters,
        "pure_clusters": pure_clusters,
        "pure_rate": float(pure_clusters / total_clusters) if total_clusters > 0 else 1.0,
        "multi_soma_violations": multi_soma_violations,
        "axon_dendrite_violations": axon_dendrite_violations,
        "cross_neuron_violations": cross_neuron_violations,
        "violation_rate": float((total_clusters - pure_clusters) / total_clusters) if total_clusters > 0 else 0.0
    }
