"""Correlation clustering and evaluation metrics.

Correlation clustering
----------------------
Given P(same neuron) per edge, find the partition of nodes into neuron
cliques that maximises total co-assignment log-likelihood.

This is NP-hard in general. We use the greedy pivot algorithm (O(E)),
which achieves a 3-approximation and works well in practice.

K materializations
------------------
Run greedy K times with different random node orderings. Return the K
highest-scoring unique partitions. A well-calibrated model should have
the true partition in the top-K — coverage@K measures this.

Metrics
-------
All metrics operate on synapse IDs, not segment IDs. This makes them
stable across segmentation versions (v117, v1412, v1433, ...).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Greedy pivot correlation clustering
# ---------------------------------------------------------------------------

def greedy_cluster(
    n_nodes: int,
    edge_src: np.ndarray,      # [E]
    edge_dst: np.ndarray,      # [E]
    edge_probs: np.ndarray,    # [E] P(same neuron)
    *,
    threshold: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:               # [N] int64 cluster labels
    """Greedy pivot correlation clustering.

    Nodes are processed in random order. Each unassigned node either joins
    the neighbouring cluster with the highest mean edge probability (if any
    neighbour is already assigned with p >= threshold), or starts a new one.
    """
    rng = rng or np.random.default_rng()

    # Deduplicated adjacency: node → {neighbour: max_prob}.
    # The edge list may contain both directions (u→v and v→u) for same-segment
    # edges; using a dict keyed by neighbour collapses duplicates and keeps the
    # highest probability so no neighbour is double-counted in votes.
    adj: list[dict[int, float]] = [{} for _ in range(n_nodes)]
    for u, v, p in zip(edge_src.tolist(), edge_dst.tolist(), edge_probs.tolist()):
        adj[u][v] = max(adj[u].get(v, 0.0), p)
        adj[v][u] = max(adj[v].get(u, 0.0), p)

    labels = np.full(n_nodes, -1, dtype=np.int64)
    next_label = 0

    for node in rng.permutation(n_nodes):
        if labels[node] >= 0:
            continue

        # Gather votes from already-assigned neighbours above threshold
        votes: dict[int, list[float]] = {}
        for nbr, prob in adj[node].items():
            if labels[nbr] >= 0 and prob >= threshold:
                votes.setdefault(int(labels[nbr]), []).append(prob)

        if votes:
            best = max(votes, key=lambda c: float(np.mean(votes[c])))
            labels[node] = best
        else:
            labels[node] = next_label
            next_label += 1

    return labels


def _partition_log_score(
    labels: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_probs: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Log-likelihood of a partition under the edge probability model."""
    same = labels[edge_src] == labels[edge_dst]
    p = np.clip(edge_probs, eps, 1 - eps)
    return float(np.where(same, np.log(p), np.log(1 - p)).sum())


def materializations(
    n_nodes: int,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_probs: np.ndarray,
    K: int = 5,
    *,
    threshold: float = 0.5,
    seed: int = 0,
) -> list[tuple[np.ndarray, float]]:
    """Return K candidate neuron partitions, sorted by log-likelihood (best first).

    Each entry is (labels [N], log_score). Diverse in the sense that no two
    returned partitions are identical.

    The coverage@K metric — does the true partition appear in the top-K? —
    measures model calibration and is the primary quality signal.
    """
    rng = np.random.default_rng(seed)
    seen: set[bytes] = set()
    results: list[tuple[np.ndarray, float]] = []

    for _ in range(K * 6):
        labels = greedy_cluster(
            n_nodes, edge_src, edge_dst, edge_probs,
            threshold=threshold, rng=rng,
        )
        key = labels.tobytes()
        if key in seen:
            continue
        seen.add(key)
        score = _partition_log_score(labels, edge_src, edge_dst, edge_probs)
        results.append((labels, score))
        if len(results) >= K:
            break

    results.sort(key=lambda x: -x[1])
    return results


# ---------------------------------------------------------------------------
# Metrics — stable across segmentation versions
# ---------------------------------------------------------------------------

def pairwise_precision_recall(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    *,
    ignore_label: int = 0,
) -> dict[str, float]:
    """Pairwise co-assignment precision, recall, and F1.

    For every pair of synapses (i, j):
      TP: predicted same-neuron AND truly same-neuron
      FP: predicted same-neuron AND truly different-neuron
      FN: predicted different-neuron AND truly same-neuron

    These metrics are defined purely on synapse pairs — independent of
    segmentation version or number of neurons.
    """
    mask = true_labels != ignore_label
    p = pred_labels[mask]
    t = true_labels[mask]
    N = len(p)
    if N < 2:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0.0, "fp": 0.0, "fn": 0.0}

    pred_same = (p[:, None] == p[None, :]) & ~np.eye(N, dtype=bool)
    true_same = (t[:, None] == t[None, :]) & ~np.eye(N, dtype=bool)

    tp = float((pred_same & true_same).sum()) / 2
    fp = float((pred_same & ~true_same).sum()) / 2
    fn = float((~pred_same & true_same).sum()) / 2

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def coverage_at_k(
    mats: list[tuple[np.ndarray, float]],
    true_labels: np.ndarray,
    *,
    recall_threshold: float = 0.9,
    ignore_label: int = 0,
) -> bool:
    """Does any of the K materializations recover >= recall_threshold of true pairs?

    This is the primary quality metric for the probabilistic output:
    the true partition should appear (or nearly appear) in the top-K.
    """
    for labels, _ in mats:
        r = pairwise_precision_recall(labels, true_labels, ignore_label=ignore_label)
        if r["recall"] >= recall_threshold:
            return True
    return False
