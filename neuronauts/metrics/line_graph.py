"""Synapse line-graph F1: the project's terminal metric.

Two synapses are *linked* in the line graph when they belong to the same
neuron. Line-graph F1 compares the linked pairs implied by a predicted
partition with those implied by ground truth. Four variants, from most
permissive to most demanding:

  pre_only    truth = same pre-neuron,                 est = same pre-cluster
  or_metric   truth = same pre OR same post neuron,    est = same pre-cluster
  post_only   truth = same post-neuron,                est = same post-cluster
  and_metric  truth = same (pre, post) circuit edge,   est = same (pre, post) cluster pair

``post_only`` and ``and_metric`` need a post-side partition. ``or_metric`` is
the original metric and is insensitive to over-fragmentation when only the
pre side is estimated.

Two implementations agree exactly and are tested against each other:

* the **set** form (:func:`compute_line_graph_f1` over explicit pair sets),
  kept because tests and legacy callers build pair sets directly; and
* the **counting** form used by :func:`evaluate_suite`, which gets the same
  TP/FP/FN from contingency tables in O(N log N) instead of O(pairs), so the
  terminal metric is affordable on a full harness region.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Set, Tuple

import numpy as np

from ._core import contingency, joint_labels, pair_confusion, pairs_in, prf1


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
    """All four line-graph F1 variants from a single evaluation."""
    pre_only: LineGraphMetrics
    or_metric: LineGraphMetrics
    post_only: Optional[LineGraphMetrics]
    and_metric: Optional[LineGraphMetrics]

    def to_dict(self, prefix: str = "lg_") -> dict:
        """Flatten to ``{prefix}{variant}_{field}`` for reports and JSON."""
        out: dict = {}
        for name in ("pre_only", "or_metric", "post_only", "and_metric"):
            m = getattr(self, name)
            if m is None:
                continue
            for k, v in asdict(m).items():
                out[f"{prefix}{name}_{k}"] = v
        return out


# ── pair-set builders ────────────────────────────────────────────────────────

def _pairwise(indices) -> Set[Tuple[int, int]]:
    items = sorted(set(indices))
    return {(items[i], items[j])
            for i in range(len(items)) for j in range(i + 1, len(items))}


def _pairs_from_labels(labels: np.ndarray) -> Set[Tuple[int, int]]:
    """Canonical ``(i < j)`` pairs of positions sharing the same label."""
    groups: dict = {}
    for idx, lab in enumerate(np.asarray(labels).tolist()):
        groups.setdefault(lab, []).append(idx)
    edges: Set[Tuple[int, int]] = set()
    for g in groups.values():
        if len(g) >= 2:
            edges |= _pairwise(g)
    return edges


def _pairs_from_joint(a: np.ndarray, b: np.ndarray) -> Set[Tuple[int, int]]:
    """Canonical pairs where ``a[i]==a[j]`` AND ``b[i]==b[j]``."""
    return _pairs_from_labels(joint_labels(a, b))


def build_true_line_graph(pre_root_ids, post_root_ids) -> Set[Tuple[int, int]]:
    """OR variant: pairs sharing the same pre-neuron OR the same post-neuron."""
    return _pairs_from_labels(pre_root_ids) | _pairs_from_labels(post_root_ids)


def build_true_pairs_pre(pre_root_ids) -> Set[Tuple[int, int]]:
    return _pairs_from_labels(pre_root_ids)


def build_true_pairs_post(post_root_ids) -> Set[Tuple[int, int]]:
    return _pairs_from_labels(post_root_ids)


def build_true_pairs_and(pre_root_ids, post_root_ids) -> Set[Tuple[int, int]]:
    """AND variant: pairs sharing the same directed circuit edge (pre, post)."""
    return _pairs_from_joint(pre_root_ids, post_root_ids)


# ── F1 from counts / sets ────────────────────────────────────────────────────

def line_graph_from_counts(tp: int, fp: int, fn: int, n_synapses: int) -> LineGraphMetrics:
    """Historical convention for this metric: undefined ratios are 0.0."""
    p, r, f = prf1(tp, fp, fn, undefined=0.0)
    return LineGraphMetrics(
        tp=int(tp), fp=int(fp), fn=int(fn), precision=p, recall=r, f1=f,
        n_true_edges=int(tp + fn), n_estimated_edges=int(tp + fp),
        n_synapses=int(n_synapses),
    )


def compute_line_graph_f1(
    true_edges: Set[Tuple[int, int]],
    estimated_edges: Set[Tuple[int, int]],
    n_synapses: int,
) -> LineGraphMetrics:
    tp = len(true_edges & estimated_edges)
    fp = len(estimated_edges - true_edges)
    fn = len(true_edges - estimated_edges)
    return line_graph_from_counts(tp, fp, fn, n_synapses)


# ── counting form ────────────────────────────────────────────────────────────

def _tp_same(pred_labels, true_labels) -> Tuple[int, int, int]:
    """``(tp, n_pred_pairs, n_true_pairs)`` for "same label" pair sets."""
    ct = contingency(true_labels, pred_labels)
    tp, fp, fn, _ = pair_confusion(ct)
    return tp, tp + fp, tp + fn


def evaluate_suite(
    pred_pre,
    pre_root_ids,
    post_root_ids,
    pred_post=None,
) -> LineGraphSuite:
    """All four line-graph variants without materialising pair sets.

    Parameters
    ----------
    pred_pre:
        ``[N]`` predicted cluster per synapse observation (pre side).
    pre_root_ids, post_root_ids:
        ``[N]`` true pre- and post-neuron per observation.
    pred_post:
        ``[N]`` predicted post-side cluster aligned to the same observations;
        required for ``post_only`` and ``and_metric``.
    """
    pred_pre = np.asarray(pred_pre)
    pre = np.asarray(pre_root_ids)
    post = np.asarray(post_root_ids)
    n = len(pred_pre)
    if pre.shape != pred_pre.shape or post.shape != pred_pre.shape:
        raise ValueError("pred_pre, pre_root_ids and post_root_ids must align")

    n_est = pairs_in(np.unique(pred_pre, return_counts=True)[1]) if n else 0
    tp_pre, _, n_true_pre = _tp_same(pred_pre, pre)
    pre_only = line_graph_from_counts(tp_pre, n_est - tp_pre, n_true_pre - tp_pre, n)

    # |pre ∪ post| = |pre| + |post| - |pre ∩ post|, and the same identity for the
    # intersection with the estimated set, because pre ∩ post is the joint set.
    tp_post_vs_pre, _, n_true_post = _tp_same(pred_pre, post)
    tp_joint_vs_pre, _, n_true_joint = _tp_same(pred_pre, joint_labels(pre, post))
    n_true_or = n_true_pre + n_true_post - n_true_joint
    tp_or = tp_pre + tp_post_vs_pre - tp_joint_vs_pre
    or_metric = line_graph_from_counts(tp_or, n_est - tp_or, n_true_or - tp_or, n)

    post_only = and_metric = None
    if pred_post is not None:
        pred_post = np.asarray(pred_post)
        if pred_post.shape != pred_pre.shape:
            raise ValueError("pred_post must align with pred_pre")
        tp_p, n_est_p, n_true_p = _tp_same(pred_post, post)
        post_only = line_graph_from_counts(tp_p, n_est_p - tp_p, n_true_p - tp_p, n)
        tp_a, n_est_a, n_true_a = _tp_same(
            joint_labels(pred_pre, pred_post), joint_labels(pre, post))
        and_metric = line_graph_from_counts(tp_a, n_est_a - tp_a, n_true_a - tp_a, n)

    return LineGraphSuite(pre_only=pre_only, or_metric=or_metric,
                          post_only=post_only, and_metric=and_metric)


def evaluate_from_root_ids(
    estimated_pre_root_ids,
    estimated_post_root_ids,
    true_pre_root_ids,
    true_post_root_ids,
) -> LineGraphMetrics:
    """OR-metric when both sides are expressed as root-id arrays."""
    true_edges = build_true_line_graph(true_pre_root_ids, true_post_root_ids)
    est_edges = build_true_line_graph(estimated_pre_root_ids, estimated_post_root_ids)
    return compute_line_graph_f1(true_edges, est_edges, len(np.asarray(true_pre_root_ids)))


# ── sampled approximation ────────────────────────────────────────────────────

def sample_synapse_pairs(n_synapses: int, *, max_pairs: int, seed: int = 42
                         ) -> Set[Tuple[int, int]]:
    """Sample up to ``max_pairs`` canonical synapse pairs without replacement."""
    if n_synapses < 2 or max_pairs <= 0:
        return set()
    total_pairs = n_synapses * (n_synapses - 1) // 2
    if max_pairs >= total_pairs:
        return {(i, j) for i in range(n_synapses) for j in range(i + 1, n_synapses)}
    rng = np.random.default_rng(seed)
    pairs: set = set()
    while len(pairs) < max_pairs:
        i, j = (int(x) for x in rng.integers(0, n_synapses, size=2))
        if i != j:
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
    sampled = sample_synapse_pairs(n_synapses, max_pairs=max_pairs, seed=seed)
    return compute_line_graph_f1(true_edges & sampled, estimated_edges & sampled, n_synapses)


__all__ = [
    "LineGraphMetrics",
    "LineGraphSuite",
    "build_true_line_graph",
    "build_true_pairs_and",
    "build_true_pairs_post",
    "build_true_pairs_pre",
    "compute_line_graph_f1",
    "compute_sampled_line_graph_f1",
    "evaluate_from_root_ids",
    "evaluate_suite",
    "line_graph_from_counts",
    "sample_synapse_pairs",
]
