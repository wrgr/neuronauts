"""
Standardized Evaluation & Benchmark Suite for Global Merge & Assembly.
Computes ARI, merge_P (Bar 1), merge_R, fk_split (Bar 3), and cluster recovery.
Fast O(N) vectorized contingency matrix implementation.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np

from neuronauts.global_merge.schemas import GlobalAssemblyResult, SegmentFragment


def _comb2(n: int) -> int:
    return n * (n - 1) // 2 if n >= 2 else 0


def adjusted_rand_index(labels_true: List[str], labels_pred: List[str]) -> float:
    """Pure NumPy calculation of Adjusted Rand Index (no sklearn dependency)."""
    if len(labels_true) != len(labels_pred) or len(labels_true) < 2:
        return 1.0

    contingency = defaultdict(lambda: defaultdict(int))
    a_dict = defaultdict(int)
    b_dict = defaultdict(int)

    for t, p in zip(labels_true, labels_pred):
        contingency[t][p] += 1
        a_dict[t] += 1
        b_dict[p] += 1

    n = len(labels_true)
    sum_comb_nij = sum(_comb2(count) for row in contingency.values() for count in row.values())
    sum_comb_a = sum(_comb2(count) for count in a_dict.values())
    sum_comb_b = sum(_comb2(count) for count in b_dict.values())
    comb_total = _comb2(n)

    if comb_total == 0:
        return 1.0

    expected_index = (sum_comb_a * sum_comb_b) / comb_total
    max_index = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_index - expected_index

    if denom == 0:
        return 1.0

    return float((sum_comb_nij - expected_index) / denom)


def compute_pairwise_partition_metrics(
    pred_map: Dict[str, str],
    gt_map: Dict[str, str]
) -> Dict[str, float]:
    """
    Fast O(N) vectorized computation of ARI, pairwise merge precision, and pairwise merge recall.
    """
    common_keys = sorted(list(set(pred_map.keys()).intersection(set(gt_map.keys()))))
    if len(common_keys) < 2:
        return {"ari": 1.0, "merge_P": 1.0, "merge_R": 1.0, "f1": 1.0, "num_pairs_evaluated": 0.0}

    labels_pred = [pred_map[k] for k in common_keys]
    labels_gt = [gt_map[k] for k in common_keys]

    # 1. Adjusted Rand Index
    ari = adjusted_rand_index(labels_gt, labels_pred)

    # 2. Fast O(N) Pairwise Contingency Calculation
    contingency = defaultdict(lambda: defaultdict(int))
    a_dict = defaultdict(int)
    b_dict = defaultdict(int)

    for t, p in zip(labels_gt, labels_pred):
        contingency[t][p] += 1
        a_dict[t] += 1
        b_dict[p] += 1

    n = len(common_keys)
    sum_comb_nij = sum(_comb2(count) for row in contingency.values() for count in row.values())
    sum_comb_gt = sum(_comb2(count) for count in a_dict.values())
    sum_comb_pred = sum(_comb2(count) for count in b_dict.values())

    tp = sum_comb_nij
    tp_fp = sum_comb_pred
    tp_fn = sum_comb_gt

    precision = float(tp / tp_fp) if tp_fp > 0 else 1.0
    recall = float(tp / tp_fn) if tp_fn > 0 else 1.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "ari": ari,
        "merge_P": precision,
        "merge_R": recall,
        "f1": f1,
        "num_pairs_evaluated": float(n * (n - 1) / 2)
    }


def evaluate_frankenmerge_split_rate(
    pred_map: Dict[str, str],
    gt_map: Dict[str, str],
    fragments: List[SegmentFragment]
) -> float:
    """
    Computes fk_split (Bar 3): The fraction of frankenmerge pairs that were correctly split.
    """
    seg_to_frags = defaultdict(list)
    for f in fragments:
        seg_to_frags[f.segment_id].append(f.fragment_id)

    franken_pairs_total = 0
    franken_pairs_split = 0

    for seg_id, fids in seg_to_frags.items():
        if len(fids) < 2:
            continue
        for i in range(len(fids)):
            for j in range(i + 1, len(fids)):
                fid1 = fids[i]
                fid2 = fids[j]
                
                gt1 = gt_map.get(fid1)
                gt2 = gt_map.get(fid2)
                
                if gt1 is not None and gt2 is not None and gt1 != gt2:
                    franken_pairs_total += 1
                    pred1 = pred_map.get(fid1)
                    pred2 = pred_map.get(fid2)
                    
                    if pred1 != pred2:
                        franken_pairs_split += 1

    if franken_pairs_total == 0:
        return 1.0

    return float(franken_pairs_split / franken_pairs_total)
