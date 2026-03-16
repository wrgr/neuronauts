"""Line graph F1 metric for connectome evaluation."""

from dataclasses import dataclass
from typing import Set, Tuple

import numpy as np

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
            f"(true edges={self.n_true_edges}, est edges={self.n_estimated_edges})"
        )


def build_true_line_graph(
    pre_root_ids: np.ndarray,
    post_root_ids: np.ndarray,
) -> Set[Tuple[int, int]]:
    edges = set()
    n = len(pre_root_ids)
    pre_groups: dict[int, list[int]] = {}
    post_groups: dict[int, list[int]] = {}

    for idx in range(n):
        pre_groups.setdefault(int(pre_root_ids[idx]), []).append(idx)
        post_groups.setdefault(int(post_root_ids[idx]), []).append(idx)

    for group in pre_groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                edges.add((min(a, b), max(a, b)))

    for group in post_groups.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                edges.add((min(a, b), max(a, b)))

    return edges


def build_estimated_line_graph(
    graph: ConnectivityGraph,
    n_synapses: int,
) -> Set[Tuple[int, int]]:
    del n_synapses
    edges = set()
    for neuron in graph.neurons.values():
        syns = sorted(set(neuron.synapse_indices))
        for i in range(len(syns)):
            for j in range(i + 1, len(syns)):
                a, b = syns[i], syns[j]
                edges.add((min(a, b), max(a, b)))
    return edges


def compute_line_graph_f1(
    true_edges: Set[Tuple[int, int]],
    estimated_edges: Set[Tuple[int, int]],
    n_synapses: int,
) -> LineGraphMetrics:
    tp = len(true_edges & estimated_edges)
    fp = len(estimated_edges - true_edges)
    fn = len(true_edges - estimated_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return LineGraphMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        n_true_edges=len(true_edges),
        n_estimated_edges=len(estimated_edges),
        n_synapses=n_synapses,
    )


def evaluate(
    graph: ConnectivityGraph,
    pre_root_ids: np.ndarray,
    post_root_ids: np.ndarray,
) -> LineGraphMetrics:
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
    true_edges = build_true_line_graph(true_pre_root_ids, true_post_root_ids)
    est_edges = build_true_line_graph(estimated_pre_root_ids, estimated_post_root_ids)
    return compute_line_graph_f1(true_edges, est_edges, len(true_pre_root_ids))
