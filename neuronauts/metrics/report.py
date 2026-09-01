"""Key registry, formatting and serialisation for metric dicts.

Every key a suite call can emit is described in :data:`KEY_DOCS` (exact
names) or :data:`PATTERN_DOCS` (families such as the line-graph variants).
The registry is the single place to look up what a number means, and a test
fails if a metric function starts emitting a key nobody documented.
"""

from __future__ import annotations

import json
import math
import re
from typing import Iterable, Optional

import numpy as np

KEY_DOCS: dict[str, str] = {
    # -- partition (from one contingency table) --------------------------
    "n_items": "Items scored after dropping unknown truth.",
    "n_clusters_pred": "Distinct predicted clusters among scored items.",
    "n_clusters_true": "Distinct true labels among scored items.",
    "n_pairs": "Unordered item pairs scored, C(n_items, 2).",
    "pair_tp": "Pairs together in both prediction and truth.",
    "pair_fp": "Pairs together in prediction only (false merges).",
    "pair_fn": "Pairs together in truth only (false splits).",
    "pair_tn": "Pairs apart in both.",
    "pair_precision": "pair_tp / (pair_tp + pair_fp); the merge precision 'Bar 1'.",
    "pair_recall": "pair_tp / (pair_tp + pair_fn); merge recall.",
    "pair_f1": "Harmonic mean of pair_precision and pair_recall.",
    "rand_disagreement": "pair_fp + pair_fn: pairs a proofreader must fix.",
    "ari": "Adjusted Rand index; 1 = identical partitions, ~0 = chance.",
    "homogeneity": "1 - H(true | pred): each predicted cluster holds one true label.",
    "completeness": "1 - H(pred | true): each true label sits in one predicted cluster.",
    "v_measure": "Harmonic mean of homogeneity and completeness.",
    "vi": "Variation of information in bits; vi_split + vi_merge.",
    "vi_split": "H(pred | true) in bits: over-segmentation term.",
    "vi_merge": "H(true | pred) in bits: false-merge term.",
    "purity_mass": "Item-weighted purity: sum over predicted clusters of majority-label count, / n_items.",
    "purity_mean": "Mean over predicted clusters of majority-label fraction.",
    "frac_pure_clusters": "Fraction of predicted clusters holding a single true label.",
    # -- weighted pairs -------------------------------------------------------
    "wpair_tp": "Sum of w_i*w_j over pair_tp pairs.",
    "wpair_fp": "Sum of w_i*w_j over false-merge pairs.",
    "wpair_fn": "Sum of w_i*w_j over false-split pairs.",
    "wpair_precision": "Weight-weighted pair precision (cable-weighted merge precision).",
    "wpair_recall": "Weight-weighted pair recall.",
    "wpair_f1": "Harmonic mean of the weighted pair precision and recall.",
    "erl": "Expected run length: sum(piece weight^2) / total weight over (true, pred) cells; weight units.",
    "weight_total": "Sum of item weights among scored items.",
    # -- candidate edges ------------------------------------------------------
    "merge_precision": "Over candidate edges: tp_merges / (tp_merges + fp_merges).",
    "merge_recall": "Over candidate edges: tp_merges / (tp_merges + fn_merges).",
    "merge_f1": "Harmonic mean of merge_precision and merge_recall.",
    "over_merge_rate": "fp_merges / n_edges_eval.",
    "under_merge_rate": "fn_merges / n_edges_eval.",
    "n_edges_eval": "Candidate edges with known truth at both ends.",
    "n_merges_pred": "Edges the prediction merged.",
    "n_splits_pred": "Edges the prediction split.",
    "n_true_merges": "Edges whose endpoints share a true label.",
    "tp_merges": "Merged edges that are truly same-label.",
    "fp_merges": "Merged edges that are truly different (false merges).",
    "fn_merges": "Split edges that are truly same-label (missed merges).",
    "tn_splits": "Split edges that are truly different.",
    "frankenmerge_rate": "Among same-fragment edges: fraction crossing a true boundary.",
    "n_frankenmerge_edges": "Same-fragment edges that cross a true boundary.",
    "frankenmerge_split_recall": "Fraction of frankenmerge edges the prediction split ('Bar 3', edge form).",
    "abstain_rate": "Fraction of items the prediction abstained on.",
    # -- frankenmerge (item/parent form) -------------------------------------
    "fk_n_parents": "Input objects carrying >= 2 true labels.",
    "fk_n_cross_pairs": "Within-parent item pairs with different truth.",
    "fk_n_cross_pairs_split": "Those pairs the prediction put in different clusters.",
    "fk_pair_split_rate": "fk_n_cross_pairs_split / fk_n_cross_pairs (global-merge fk_split).",
    "fk_n_separated": "Frankenmerge parents with no predicted cluster spanning two of their labels.",
    "fk_separation": "fk_n_separated / fk_n_parents (treestitch fk_separation).",
    "fk_parents": "Ids of the frankenmerge parents.",
    # -- completeness --------------------------------------------------------
    "cmpl_precision": "Of fragments predicted complete, fraction truly complete.",
    "cmpl_recall": "Of truly complete fragments, fraction predicted complete.",
    "cmpl_f1": "Harmonic mean of cmpl_precision and cmpl_recall.",
    "cmpl_accuracy": "Fraction of fragments whose completeness was predicted correctly.",
    "cmpl_n_complete_gt": "Fragments that need no edit under ground truth.",
    "cmpl_n_fragments": "Fragments scored for completeness.",
    "cmpl_tp": "Predicted complete and truly complete.",
    "cmpl_fp": "Predicted complete but needs an edit.",
    "cmpl_fn": "Predicted needs-edit but truly complete.",
    "cmpl_tn": "Predicted needs-edit and truly needs an edit.",
    # -- naive baseline (prediction = input fragment id) ---------------------
    "naive_ari": "ARI of the do-nothing partition (each input object its own cluster).",
    "naive_pair_precision": "Pair precision of the do-nothing partition.",
    "naive_pair_recall": "Pair recall of the do-nothing partition.",
    "naive_pair_f1": "Pair F1 of the do-nothing partition.",
    "naive_vi_split": "vi_split of the do-nothing partition.",
    "naive_vi_merge": "vi_merge of the do-nothing partition.",
    "naive_wpair_precision": "Weighted pair precision of the do-nothing partition.",
    "naive_wpair_recall": "Weighted pair recall of the do-nothing partition.",
    "naive_wpair_f1": "Weighted pair F1 of the do-nothing partition.",
    "naive_erl": "ERL of the do-nothing partition.",
    "naive_cmpl_precision": "Completeness precision if every fragment is called complete.",
    "naive_cmpl_recall": "Completeness recall if every fragment is called complete (1.0).",
    "naive_cmpl_f1": "Completeness F1 if every fragment is called complete.",
    # -- connectome ----------------------------------------------------------
    "synapse_attr_acc": "Fraction of labelled synapses whose cluster maps (majority vote) to their true pre neuron.",
    "conn_edge_precision": "Directed neuron->neuron edges (>= min_syn) that exist in truth.",
    "conn_edge_recall": "True directed edges recovered.",
    "conn_edge_f1": "Harmonic mean of the directed edge precision and recall.",
    "n_true_edges": "Distinct true directed connections.",
    "n_pred_edges": "Distinct predicted directed connections.",
    "conn_edge_precision_undir": "Undirected (reciprocal-summed) edge precision.",
    "conn_edge_recall_undir": "Undirected edge recall.",
    "conn_edge_f1_undir": "Undirected edge F1.",
    "n_true_edges_undir": "Distinct true undirected connections.",
    "n_pred_edges_undir": "Distinct predicted undirected connections.",
    "n_synapses_labelled": "Synapses with known pre and post neuron.",
    "n_synapses_both_sides": "Synapses observed on both the pre and post side (dual-side protocol).",
    "n_synapses_pre_only": "Synapses observed only on the pre side.",
    "n_synapses_post_only": "Synapses observed only on the post side.",
}

PATTERN_DOCS: list[tuple[str, str]] = [
    (r"^lg_(pre_only|or_metric|post_only|and_metric)_(tp|fp|fn)$",
     "Line-graph pair confusion for the named variant."),
    (r"^lg_(pre_only|or_metric|post_only|and_metric)_(precision|recall|f1)$",
     "Line-graph P/R/F1 for the named variant (pre_only, or_metric, post_only, and_metric)."),
    (r"^lg_(pre_only|or_metric|post_only|and_metric)_(n_true_edges|n_estimated_edges|n_synapses)$",
     "Line-graph pair-set sizes for the named variant."),
]

_SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("PARTITION", ("n_", "pair_", "rand_", "ari", "homogeneity", "completeness",
                   "v_measure", "vi", "purity_", "frac_pure")),
    ("WEIGHTED", ("wpair_", "erl", "weight_")),
    ("EDGES", ("merge_", "over_merge", "under_merge", "tp_merges", "fp_merges",
               "fn_merges", "tn_splits", "frankenmerge", "abstain")),
    ("FRANKENMERGE", ("fk_",)),
    ("COMPLETENESS", ("cmpl_",)),
    ("NAIVE BASELINE", ("naive_",)),
    ("CONNECTOME", ("conn_", "synapse_attr")),
    ("LINE GRAPH", ("lg_",)),
]


def describe_key(key: str) -> Optional[str]:
    """Documentation for a metric key, or ``None`` if it is not registered."""
    if key in KEY_DOCS:
        return KEY_DOCS[key]
    for pat, doc in PATTERN_DOCS:
        if re.match(pat, key):
            return doc
    return None


def undocumented_keys(metrics: Iterable[str]) -> list[str]:
    return sorted(k for k in metrics if describe_key(k) is None)


def _fmt(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return "n/a" if math.isnan(v) else f"{v:.4f}"
    if isinstance(v, (list, tuple, np.ndarray)):
        return f"[{len(v)} values]"
    return str(v)


def format_metrics(metrics: dict, *, title: str = "", width: int = 34) -> str:
    """Render a flat metric dict grouped into sections, one key per line."""
    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("=" * max(len(title), 20))
    seen: set = set()
    for name, prefixes in _SECTIONS:
        keys = [k for k in metrics if k not in seen and k.startswith(prefixes)]
        if not keys:
            continue
        lines.append(f"[{name}]")
        for k in keys:
            lines.append(f"  {k:<{width}} {_fmt(metrics[k])}")
            seen.add(k)
    rest = [k for k in metrics if k not in seen]
    if rest:
        lines.append("[OTHER]")
        for k in rest:
            lines.append(f"  {k:<{width}} {_fmt(metrics[k])}")
    return "\n".join(lines)


def to_jsonable(obj):
    """Convert numpy scalars/arrays and NaN so ``json.dumps`` is standard JSON."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return obj


def metrics_to_json(metrics: dict, **kwargs) -> str:
    return json.dumps(to_jsonable(metrics), **kwargs)


__all__ = [
    "KEY_DOCS",
    "PATTERN_DOCS",
    "describe_key",
    "format_metrics",
    "metrics_to_json",
    "to_jsonable",
    "undocumented_keys",
]
