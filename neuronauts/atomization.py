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
    """Pairwise sufficient statistics for one partition.

    Delegates to :mod:`neuronauts.metrics` (one sparse contingency table
    instead of a hand-rolled joint-key bincount).
    """
    from neuronauts.metrics._core import contingency, pair_confusion

    tp, fp, fn, tn = pair_confusion(
        contingency(np.asarray(true_labels), np.asarray(predicted_labels)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Precision/recall/F1 plus cross-lineage split recall (specificity).

    Historical convention preserved: every ratio defaults to 1.0 (not NaN)
    when its denominator is zero.
    """
    from neuronauts.metrics._core import prf1, safe_div

    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision, recall, f1 = prf1(tp, fp, fn, undefined=1.0)
    split_recall = safe_div(tn, tn + fp, undefined=1.0)
    return {"pair_precision": precision, "pair_recall": recall,
            "pair_f1": f1, "cross_lineage_split_recall": split_recall}
