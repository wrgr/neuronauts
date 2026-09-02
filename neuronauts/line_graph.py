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


def _clutter_keep(pre_root_ids, post_root_ids, min_root_synapses: int):
    """Synapses whose pre- AND post-root each occur >= ``min_root_synapses``
    times across both columns -- the same rule as
    :meth:`neuronauts.fetch.SynapseTable.filter_clutter`, applied at
    evaluation time so a clutter root never contributes a line-graph pair.
    ``min_root_synapses <= 1`` keeps everything."""
    import numpy as np

    pre = np.asarray(pre_root_ids)
    post = np.asarray(post_root_ids)
    if min_root_synapses <= 1 or len(pre) == 0:
        return np.ones(len(pre), bool)
    roots = np.concatenate([pre, post])
    unique, counts = np.unique(roots, return_counts=True)
    keep_roots = unique[counts >= min_root_synapses]
    return np.isin(pre, keep_roots) & np.isin(post, keep_roots)


def _restrict(true_edges, est_edges, keep):
    """Drop every pair touching a dropped synapse and renumber the survivors
    compactly, so the sampled variant's index space stays consistent."""
    import numpy as np

    if keep.all():
        return true_edges, est_edges, int(len(keep))
    new = np.cumsum(keep) - 1

    def remap(edges):
        return {(int(new[a]), int(new[b])) for a, b in edges
                if keep[a] and keep[b]}

    return remap(true_edges), remap(est_edges), int(keep.sum())


def evaluate(
    graph: ConnectivityGraph,
    pre_root_ids,
    post_root_ids,
    *,
    min_root_synapses: int = 0,
) -> LineGraphMetrics:
    """Legacy: OR-metric from a ConnectivityGraph.

    ``min_root_synapses`` excludes clutter roots from the metric. The kwarg
    was added to every ``scripts/train.py`` call site by the "Filter connectome
    clutter" change (79e2550dc) but never to this function, so the training
    CLI's evaluate path raised ``TypeError`` for months and six tests failed
    at HEAD; the QA pass of 2026-09-02 traced it. Default 0 is the old
    behaviour exactly.
    """
    n = len(pre_root_ids)
    true_edges = build_true_line_graph(pre_root_ids, post_root_ids)
    est_edges = build_estimated_line_graph(graph, n)
    keep = _clutter_keep(pre_root_ids, post_root_ids, min_root_synapses)
    true_edges, est_edges, n = _restrict(true_edges, est_edges, keep)
    return compute_line_graph_f1(true_edges, est_edges, n)


def evaluate_sampled(
    graph: ConnectivityGraph,
    pre_root_ids,
    post_root_ids,
    *,
    max_pairs: int = 10000,
    seed: int = 42,
    min_root_synapses: int = 0,
) -> LineGraphMetrics:
    """Sampled OR-metric from a ConnectivityGraph. See :func:`evaluate` for
    ``min_root_synapses``."""
    n = len(pre_root_ids)
    true_edges = build_true_line_graph(pre_root_ids, post_root_ids)
    est_edges = build_estimated_line_graph(graph, n)
    keep = _clutter_keep(pre_root_ids, post_root_ids, min_root_synapses)
    true_edges, est_edges, n = _restrict(true_edges, est_edges, keep)
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
