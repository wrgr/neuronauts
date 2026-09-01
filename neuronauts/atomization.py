"""Label-blind geometry utilities for atomizing mixed real roots."""

from __future__ import annotations
import numpy as np


def euclidean_mst(points_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact Euclidean MST for a modest observation cloud."""
    points = np.asarray(points_nm, dtype=np.float64)
    if len(points) < 2:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.float64)
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import squareform, pdist

    distances = squareform(pdist(points))
    tree = minimum_spanning_tree(distances).tocoo()
    edges = np.column_stack([tree.row, tree.col]).astype(np.int64)
    return edges, np.asarray(tree.data, dtype=np.float64)


def cut_components(n_vertices: int, edges: np.ndarray,
                   lengths: np.ndarray, cutoff_nm: float) -> np.ndarray:
    """Connected-component ids after removing MST edges above a cutoff."""
    parent = list(range(n_vertices))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for (left, right), length in zip(edges, lengths):
        if float(length) <= float(cutoff_nm):
            a, b = find(int(left)), find(int(right))
            if a != b:
                parent[a] = b
    roots = [find(index) for index in range(n_vertices)]
    _, inverse = np.unique(roots, return_inverse=True)
    return inverse.astype(np.int64)


def pair_counts(true_labels: np.ndarray,
                predicted_labels: np.ndarray) -> dict[str, int]:
    """Pairwise sufficient statistics for one partition."""
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    _, true_inverse = np.unique(true_labels, return_inverse=True)
    _, pred_inverse = np.unique(predicted_labels, return_inverse=True)
    n_pred = int(pred_inverse.max()) + 1 if len(pred_inverse) else 0
    joint = np.bincount(true_inverse * max(n_pred, 1) + pred_inverse)
    true_count = np.bincount(true_inverse)
    pred_count = np.bincount(pred_inverse)
    choose2 = lambda values: int(np.sum(values * (values - 1) // 2))
    total = len(true_labels) * (len(true_labels) - 1) // 2
    true_pairs = choose2(true_count)
    predicted_pairs = choose2(pred_count)
    true_positive = choose2(joint)
    false_positive = predicted_pairs - true_positive
    false_negative = true_pairs - true_positive
    cross_pairs = total - true_pairs
    true_negative = cross_pairs - false_positive
    return {
        "tp": true_positive, "fp": false_positive,
        "fn": false_negative, "tn": true_negative,
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    split_recall = tn / (tn + fp) if tn + fp else 1.0
    return {"pair_precision": precision, "pair_recall": recall,
            "pair_f1": f1, "cross_lineage_split_recall": split_recall}
