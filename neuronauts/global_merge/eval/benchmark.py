"""
Global-merge benchmark metrics — kept here for backward compatibility.

The implementations now live in :mod:`neuronauts.metrics`: ARI and pairwise
merge P/R/F1 in :mod:`neuronauts.metrics.partition`, frankenmerge split rate
in :mod:`neuronauts.metrics.frankenmerge`, and the cable-weighted P/R/ERL in
the same partition module (as ``wpair_*`` / ``erl``, computed from one sparse
contingency table instead of the O(N^2) pair loop this module used to run).
This module adapts the dict-of-strings call shape used across
``scripts/benchmark_*.py`` to that array-based API. New code should call
:mod:`neuronauts.metrics` directly.
"""

from __future__ import annotations

from typing import Dict, List

from neuronauts.global_merge.schemas import SegmentFragment
from neuronauts.metrics._core import labels_from_maps
from neuronauts.metrics.frankenmerge import frankenmerge_metrics
from neuronauts.metrics.partition import adjusted_rand_index as _ari
from neuronauts.metrics.partition import partition_metrics


def _comb2(n: int) -> int:
    return n * (n - 1) // 2 if n >= 2 else 0


def adjusted_rand_index(labels_true: List[str], labels_pred: List[str]) -> float:
    """Adjusted Rand Index. Delegates to :func:`neuronauts.metrics.adjusted_rand_index`."""
    return _ari(labels_true, labels_pred)


def compute_pairwise_partition_metrics(
    pred_map: Dict[str, str],
    gt_map: Dict[str, str],
) -> Dict[str, float]:
    """ARI, pairwise merge precision and pairwise merge recall.

    Historical convention preserved: precision/recall default to 1.0 (not
    NaN) when their denominator is zero.
    """
    keys, pred, true = labels_from_maps(pred_map, gt_map)
    n = len(keys)
    if n < 2:
        return {"ari": 1.0, "merge_P": 1.0, "merge_R": 1.0, "f1": 1.0,
                "num_pairs_evaluated": 0.0}

    m = partition_metrics(pred, true, ignore=None, undefined=1.0)
    return {
        "ari": m["ari"],
        "merge_P": m["pair_precision"],
        "merge_R": m["pair_recall"],
        "f1": m["pair_f1"],
        "num_pairs_evaluated": float(m["n_pairs"]),
    }


def evaluate_frankenmerge_split_rate(
    pred_map: Dict[str, str],
    gt_map: Dict[str, str],
    fragments: List[SegmentFragment],
) -> float:
    """Fraction of frankenmerge (cross-truth, same-segment) fragment pairs
    the prediction correctly split ("Bar 3"). 1.0 when there are none.

    Historical quirk preserved: a fragment absent from ``pred_map`` (never
    assigned a cluster) is *not* dropped the way a missing ground-truth entry
    is. It is given a single shared placeholder cluster, so two such
    fragments count as merged with each other but split from every real
    prediction — matching the original ``dict.get()``-based comparison.
    """
    seg_of = {f.fragment_id: f.segment_id for f in fragments}
    keys = [k for k in gt_map if k in seg_of]
    if not keys:
        return 1.0
    true = [gt_map[k] for k in keys]
    parent = [seg_of[k] for k in keys]
    pred = [pred_map.get(k, "__unassigned__") for k in keys]

    fk = frankenmerge_metrics(pred, true, parent)
    rate = fk["fk_pair_split_rate"]
    return 1.0 if rate != rate else float(rate)  # NaN (no frankenmerges) -> 1.0


def compute_path_length_metrics(
    pred_map: Dict[str, str],
    gt_map: Dict[str, str],
    fragments: List[SegmentFragment],
) -> Dict[str, float]:
    """Path-length-weighted precision/recall and expected run length (ERL).

      - Path-Weighted Precision: Sum(L_i * L_j for TP) / Sum(L_i * L_j for TP+FP)
      - Path-Weighted Recall:    Sum(L_i * L_j for TP) / Sum(L_i * L_j for TP+FN)
      - Expected Run Length (ERL, um): Sum(L_c^2) / L_total
      - Total Ground Truth Path Length (um)

    Historical convention preserved: path_P/path_R default to 1.0 when their
    denominator is zero; erl_um defaults to 0.0 without any labelled length.
    """
    frag_len = {}
    for f in fragments:
        l_um = float(f.path_length_nm / 1000.0) if f.path_length_nm > 0 \
            else float(len(f.vertices_nm) * 40.0 / 1000.0)
        frag_len[f.fragment_id] = max(0.1, l_um)

    keys, pred, true = labels_from_maps(pred_map, gt_map)
    if len(keys) < 2:
        return {"path_P": 1.0, "path_R": 1.0, "erl_um": 0.0, "total_gt_path_um": 0.0}

    weights = [frag_len.get(k, 1.0) for k in keys]
    m = partition_metrics(pred, true, ignore=None, weights=weights, undefined=1.0)
    erl = m["erl"]
    return {
        "path_P": m["wpair_precision"],
        "path_R": m["wpair_recall"],
        "erl_um": 0.0 if erl != erl else float(erl),
        "total_gt_path_um": m["weight_total"],
    }
