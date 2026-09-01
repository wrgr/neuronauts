"""The one home for evaluation metrics.

Every metric the project reports is defined once here and reached from the
older entry points (``treestitch.partition``, ``treestitch.connectivity``,
``neuronauts.global_merge.eval.benchmark``, ``neuronauts.line_graph``, ...)
through thin delegating wrappers, so scripts keep working while the maths
lives in one place. See ``docs/metrics.md`` for the registry and the
migration map.

Conventions shared by every function:

* Inputs are aligned per-item arrays (``pred``, ``true``, optional
  ``weights``), any dtype ``np.unique`` accepts. Dict-shaped inputs go through
  :func:`labels_from_maps`.
* ``ignore`` marks unknown *truth* (dropped); ``pred_ignore`` marks an
  *abstained* prediction (kept as a singleton).
* Undefined ratios are NaN unless a wrapper explicitly asks otherwise.
* Everything is O(N log N) via one sparse contingency table; nothing
  materialises N x N or all pairs.

Quick start::

    from neuronauts.metrics import evaluate_partition_suite, format_metrics
    m = evaluate_partition_suite(pred, true, weights=cable_um, fragment_id=atom_id)
    print(format_metrics(m, title="PCFG vs learned, val region"))
"""

from ._core import (
    NAN,
    Contingency,
    align_labels,
    contingency,
    joint_labels,
    labels_from_maps,
    pair_confusion,
    prf1,
    safe_div,
    weighted_pair_confusion,
)
from .calibration import (
    brier_score,
    calibration_metrics,
    expected_calibration_error,
    reliability_bins,
)
from .completeness import (
    completeness_metrics,
    fragment_completeness,
    pred_fragment_completeness,
)
from .connectome import (
    connectome_metrics,
    dual_side_connectome_metrics,
    edge_set_prf1,
    match_clusters_majority,
    undirected_edge_set,
)
from .edges import edge_merge_metrics
from .frankenmerge import frankenmerge_metrics
from .line_graph import (
    LineGraphMetrics,
    LineGraphSuite,
    build_true_line_graph,
    build_true_pairs_and,
    build_true_pairs_post,
    build_true_pairs_pre,
    compute_line_graph_f1,
    compute_sampled_line_graph_f1,
    evaluate_from_root_ids,
    evaluate_suite,
    line_graph_from_counts,
    sample_synapse_pairs,
)
from .partition import (
    adjusted_rand_from_confusion,
    adjusted_rand_index,
    cluster_purity,
    expected_run_length,
    homogeneity_completeness_v,
    partition_metrics,
    rand_disagreement,
    variation_of_information,
)
from .ranking import (
    average_precision,
    best_f1_threshold,
    edit_metrics_vs_baseline,
    roc_auc,
    threshold_metrics,
)
from .report import (
    KEY_DOCS,
    PATTERN_DOCS,
    describe_key,
    format_metrics,
    metrics_to_json,
    to_jsonable,
    undocumented_keys,
)
from .suite import evaluate_partition_suite

__all__ = [
    # core
    "NAN", "Contingency", "align_labels", "contingency", "joint_labels",
    "labels_from_maps", "pair_confusion", "prf1", "safe_div", "weighted_pair_confusion",
    # partition
    "adjusted_rand_from_confusion", "adjusted_rand_index", "cluster_purity",
    "expected_run_length", "homogeneity_completeness_v", "partition_metrics",
    "rand_disagreement", "variation_of_information",
    # edges / frankenmerge / completeness / connectome
    "edge_merge_metrics", "frankenmerge_metrics",
    "completeness_metrics", "fragment_completeness", "pred_fragment_completeness",
    "connectome_metrics", "dual_side_connectome_metrics", "edge_set_prf1",
    "match_clusters_majority", "undirected_edge_set",
    # line graph
    "LineGraphMetrics", "LineGraphSuite", "build_true_line_graph", "build_true_pairs_and",
    "build_true_pairs_post", "build_true_pairs_pre", "compute_line_graph_f1",
    "compute_sampled_line_graph_f1", "evaluate_from_root_ids", "evaluate_suite",
    "line_graph_from_counts", "sample_synapse_pairs",
    # ranking / calibration
    "average_precision", "best_f1_threshold", "edit_metrics_vs_baseline", "roc_auc",
    "threshold_metrics", "brier_score", "calibration_metrics",
    "expected_calibration_error", "reliability_bins",
    # suite / report
    "evaluate_partition_suite", "KEY_DOCS", "PATTERN_DOCS", "describe_key",
    "format_metrics", "metrics_to_json", "to_jsonable", "undocumented_keys",
]
