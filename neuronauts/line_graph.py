"""Line graph F1 metric for connectome evaluation.

Four variants, ordered from most permissive to most demanding:

  pre_only   truth = same pre-neuron,                  est = same pre-cluster
  or_metric  truth = same pre-neuron OR post-neuron,   est = same pre-cluster
  post_only  truth = same post-neuron,                 est = same post-cluster
  and_metric truth = same (pre-neuron, post-neuron),   est = same (pre-cluster, post-cluster)

``post_only`` and ``and_metric`` require a post-side partition (``pred_post``).
A never-merge partition gets recall=0 under ``and_metric`` (no two singletons
share a (pre-cluster, post-cluster) pair) and recall=0 under ``pre_only``/``post_only``
(no two singletons share a cluster label).  ``or_metric`` matches the original
line-graph metric and is insensitive to over-fragmentation when estimated from
the pre-side alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np

from .helpers import pairwise_edges
from .merge import ConnectivityGraph


@dataclass
class LineGraphMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    n_true_edges: int
    n_estimated_edges: int
    n_synapses: int

    def __str__(self) -> str:
        return (
            f"LineGraph F1={self.f1:.3f}  "
            f"P={self.precision:.3f}  R={self.recall:.3f}  "
            f"TP={self.tp} FP={self.fp} FN={self.fn}  "
            f"(true={self.n_true_edges}, est={self.n_estimated_edges})"
        )


@dataclass
class LineGraphSuite:
    """All four line-graph F1 variants from a single evaluation.

    ``pre_only``:   truth = same pre-neuron, est = same pre-cluster.
                    Penalises axonal over-fragmentation and false merges.
    ``or_metric``:  truth = same pre OR post, est = same pre-cluster.
                    Original line-graph metric (backward-compatible).
    ``post_only``:  truth = same post-neuron, est = same post-cluster.
                    ``None`` without a post-side partition.
    ``and_metric``: truth = same (pre, post) circuit edge,
                    est = same (pre-cluster, post-cluster) pair.
                    Most demanding: penalises over-fragmentation on either side,
                    and a never-merge partition on both sides gives recall = 0.
                    ``None`` without a post-side partition.
    """
    pre_only:  LineGraphMetrics
    or_metric: LineGraphMetrics
    post_only: Optional[LineGraphMetrics]
    and_metric: Optional[LineGraphMetrics]


# ── low-level pair builders ──────────────────────────────────────────────────

def _pairs_from_labels(labels: np.ndarray) -> Set[Tuple[int, int]]:
    """Canonical (i<j) pairs for all observations sharing the same integer label."""
    groups: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(idx)
    edges: Set[Tuple[int, int]] = set()
    for g in groups.values():
        if len(g) >= 2:
            edges |= pairwise_edges(g)
    return edges


def _pairs_from_joint(a: np.ndarray, b: np.ndarray) -> Set[Tuple[int, int]]:
    """Canonical (i<j) pairs where a[i]==a[j] AND b[i]==b[j]."""
    groups: dict[tuple, list[int]] = {}
    for idx in range(len(a)):
        key = (int(a[idx]), int(b[idx]))
        groups.setdefault(key, []).append(idx)
    edges: Set[Tuple[int, int]] = set()
    for g in groups.values():
        if len(g) >= 2:
            edges |= pairwise_edges(g)
    return edges


# ── public true-pair builders (also used directly in tests) ──────────────────

def build_true_line_graph(
    pre_root_ids: np.ndarray,
    post_root_ids: np.ndarray,
) -> Set[Tuple[int, int]]:
    """OR variant: pairs sharing same pre-neuron OR same post-neuron."""
    return _pairs_from_labels(pre_root_ids) | _pairs_from_labels(post_root_ids)


def build_true_pairs_pre(pre_root_ids: np.ndarray) -> Set[Tuple[int, int]]:
    """Pairs sharing the same pre-neuron only."""
    return _pairs_from_labels(pre_root_ids)


def build_true_pairs_post(post_root_ids: np.ndarray) -> Set[Tuple[int, int]]:
    """Pairs sharing the same post-neuron only."""
    return _pairs_from_labels(post_root_ids)


def build_true_pairs_and(
    pre_root_ids: np.ndarray,
    post_root_ids: np.ndarray,
) -> Set[Tuple[int, int]]:
    """AND variant: pairs sharing the same directed circuit edge (pre, post)."""
    return _pairs_from_joint(pre_root_ids, post_root_ids)


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


# ── F1 computation ────────────────────────────────────────────────────────────

def compute_line_graph_f1(
    true_edges: Set[Tuple[int, int]],
    estimated_edges: Set[Tuple[int, int]],
    n_synapses: int,
) -> LineGraphMetrics:
    tp = len(true_edges & estimated_edges)
    fp = len(estimated_edges - true_edges)
    fn = len(true_edges - estimated_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return LineGraphMetrics(
        tp=tp, fp=fp, fn=fn,
        precision=precision, recall=recall, f1=f1,
        n_true_edges=len(true_edges),
        n_estimated_edges=len(estimated_edges),
        n_synapses=n_synapses,
    )


# ── high-level evaluation API ─────────────────────────────────────────────────

def evaluate_suite(
    pred_pre: np.ndarray,
    pre_root_ids: np.ndarray,
    post_root_ids: np.ndarray,
    pred_post: Optional[np.ndarray] = None,
) -> LineGraphSuite:
    """Compute all four line-graph F1 variants in one pass.

    Parameters
    ----------
    pred_pre:
        [N] predicted cluster label per synapse observation (pre-side partition).
    pre_root_ids:
        [N] ground-truth pre-neuron id per observation.
    post_root_ids:
        [N] ground-truth post-neuron id per observation.
    pred_post:
        [N] predicted post-side cluster label, aligned to the same N observations
        as ``pred_pre`` by synapse_id.  Required for ``post_only`` and
        ``and_metric``; pass ``None`` for single-side evaluation.

    Returns
    -------
    LineGraphSuite
        All four metrics.  ``post_only`` and ``and_metric`` are ``None`` when
        ``pred_post`` is not provided.
    """
    n = len(pred_pre)
    true_pre  = _pairs_from_labels(pre_root_ids)
    true_post = _pairs_from_labels(post_root_ids)
    true_or   = true_pre | true_post
    true_and  = _pairs_from_joint(pre_root_ids, post_root_ids)
    est_pre   = _pairs_from_labels(pred_pre)

    pre_only  = compute_line_graph_f1(true_pre, est_pre, n)
    or_metric = compute_line_graph_f1(true_or,  est_pre, n)

    if pred_post is not None:
        est_post   = _pairs_from_labels(pred_post)
        est_and    = _pairs_from_joint(pred_pre, pred_post)
        post_only  = compute_line_graph_f1(true_post, est_post, n)
        and_metric = compute_line_graph_f1(true_and,  est_and,  n)
    else:
        post_only  = None
        and_metric = None

    return LineGraphSuite(
        pre_only=pre_only,
        or_metric=or_metric,
        post_only=post_only,
        and_metric=and_metric,
    )


def evaluate(
    graph: ConnectivityGraph,
    pre_root_ids: np.ndarray,
    post_root_ids: np.ndarray,
) -> LineGraphMetrics:
    """Legacy: OR-metric from a ConnectivityGraph."""
    n = len(pre_root_ids)
    true_edges = build_true_line_graph(pre_root_ids, post_root_ids)
    est_edges = build_estimated_line_graph(graph, n)
    return compute_line_graph_f1(true_edges, est_edges, n)


def evaluate_from_root_ids(
    estimated_pre_root_ids: np.ndarray,
    estimated_post_root_ids: np.ndarray,
    true_pre_root_ids: np.ndarray,
    true_post_root_ids: np.ndarray,
) -> LineGraphMetrics:
    """OR-metric when both sides are expressed as root-id arrays."""
    true_edges = build_true_line_graph(true_pre_root_ids, true_post_root_ids)
    est_edges  = build_true_line_graph(estimated_pre_root_ids, estimated_post_root_ids)
    return compute_line_graph_f1(true_edges, est_edges, len(true_pre_root_ids))


# ── sampled approximation (for large synapse sets) ───────────────────────────

def sample_synapse_pairs(
    n_synapses: int,
    *,
    max_pairs: int,
    seed: int = 42,
) -> Set[Tuple[int, int]]:
    """Sample up to ``max_pairs`` canonical synapse pairs without replacement."""
    if n_synapses < 2 or max_pairs <= 0:
        return set()

    total_pairs = n_synapses * (n_synapses - 1) // 2
    if max_pairs >= total_pairs:
        return {
            (i, j)
            for i in range(n_synapses)
            for j in range(i + 1, n_synapses)
        }

    rng = np.random.default_rng(seed)
    pairs: set[Tuple[int, int]] = set()
    while len(pairs) < max_pairs:
        ij = rng.integers(0, n_synapses, size=2)
        i, j = int(ij[0]), int(ij[1])
        if i == j:
            continue
        pairs.add((min(i, j), max(i, j)))
    return pairs


def compute_sampled_line_graph_f1(
    true_edges: Set[Tuple[int, int]],
    estimated_edges: Set[Tuple[int, int]],
    n_synapses: int,
    *,
    max_pairs: int = 10000,
    seed: int = 42,
) -> LineGraphMetrics:
    """Approximate line-graph F1 on a sampled subset of synapse pairs."""
    sampled_pairs = sample_synapse_pairs(n_synapses, max_pairs=max_pairs, seed=seed)
    sampled_true = true_edges & sampled_pairs
    sampled_est  = estimated_edges & sampled_pairs
    return compute_line_graph_f1(sampled_true, sampled_est, n_synapses)


def evaluate_sampled(
    graph: ConnectivityGraph,
    pre_root_ids: np.ndarray,
    post_root_ids: np.ndarray,
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
