"""Line-graph F1 — kept here for backward compatibility.

The implementation now lives in :mod:`neuronauts.metrics.line_graph`, which
has no dependency on the legacy :class:`~neuronauts.merge.ConnectivityGraph`.
This module re-exports everything from there and adds back the three
``ConnectivityGraph``-based entry points (``evaluate``,
``build_estimated_line_graph``, ``evaluate_sampled``) that legacy callers
still use. New code should import from :mod:`neuronauts.metrics` directly.
"""

from __future__ import annotations

from typing import Set, Tuple

from .merge import ConnectivityGraph
from .metrics.line_graph import (  # noqa: F401 (re-exported)
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
from .helpers import pairwise_edges


def build_estimated_line_graph(
    graph: ConnectivityGraph,
    n_synapses: int,
) -> Set[Tuple[int, int]]:
    """Build estimated pairs from a ConnectivityGraph (legacy API)."""
    del n_synapses
    edges: Set[Tuple[int, int]] = set()
    for neuron in graph.neurons.values():
        edges |= pairwise_edges(neuron.synapse_indices)
    return edges


def evaluate(
    graph: ConnectivityGraph,
    pre_root_ids,
    post_root_ids,
) -> LineGraphMetrics:
    """Legacy: OR-metric from a ConnectivityGraph."""
    n = len(pre_root_ids)
    true_edges = build_true_line_graph(pre_root_ids, post_root_ids)
    est_edges = build_estimated_line_graph(graph, n)
    return compute_line_graph_f1(true_edges, est_edges, n)


def evaluate_sampled(
    graph: ConnectivityGraph,
    pre_root_ids,
    post_root_ids,
    *,
    max_pairs: int = 10000,
    seed: int = 42,
) -> LineGraphMetrics:
    """Sampled OR-metric from a ConnectivityGraph."""
    n = len(pre_root_ids)
    true_edges = build_true_line_graph(pre_root_ids, post_root_ids)
    est_edges = build_estimated_line_graph(graph, n)
    return compute_sampled_line_graph_f1(
        true_edges, est_edges, n,
        max_pairs=max_pairs, seed=seed,
    )


__all__ = [
    "LineGraphMetrics",
    "LineGraphSuite",
    "build_estimated_line_graph",
    "build_true_line_graph",
    "build_true_pairs_and",
    "build_true_pairs_post",
    "build_true_pairs_pre",
    "compute_line_graph_f1",
    "compute_sampled_line_graph_f1",
    "evaluate",
    "evaluate_from_root_ids",
    "evaluate_sampled",
    "evaluate_suite",
    "line_graph_from_counts",
    "sample_synapse_pairs",
]
