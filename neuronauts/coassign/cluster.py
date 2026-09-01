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


def _canonical_key(labels: np.ndarray) -> bytes:
    """Stable bytes key for a partition regardless of arbitrary cluster ID assignment.

    Two label arrays that represent the same partition (same clusters, different
    integer IDs) produce the same key. IDs are remapped in first-occurrence order
    so the result is deterministic.
    """
    mapping: dict[int, int] = {}
    canonical = np.empty_like(labels)
    next_id = 0
    for i, lbl in enumerate(labels.tolist()):
        if lbl not in mapping:
            mapping[lbl] = next_id
            next_id += 1
        canonical[i] = mapping[lbl]
    return canonical.tobytes()


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
        key = _canonical_key(labels)
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
    segmentation version or number of neurons. Delegates to
    :func:`neuronauts.metrics.partition_metrics` (one sparse contingency
    table, O(N log N)) rather than the dense N x N pair matrix this used to
    build. Historical convention preserved: precision/recall/f1 default to
    0.0 (not NaN) when their denominator is zero.
    """
    from neuronauts.metrics.partition import partition_metrics

    m = partition_metrics(pred_labels, true_labels, ignore=ignore_label, undefined=0.0)
    return {"precision": m["pair_precision"], "recall": m["pair_recall"],
            "f1": m["pair_f1"], "tp": m["pair_tp"], "fp": m["pair_fp"],
            "fn": m["pair_fn"]}


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


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_threshold(
    n_nodes: int,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_probs: np.ndarray,
    true_labels: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    seed: int = 0,
    ignore_label: int = 0,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Pick the greedy-cluster threshold that maximises pairwise F1.

    The model produces well-calibrated edge probabilities, but the partition
    quality depends strongly on the cut point fed to ``greedy_cluster``. A
    single fixed 0.5 is rarely optimal — denser graphs need a higher bar to
    avoid over-merging, sparser ones a lower bar to avoid over-splitting.

    This sweeps candidate thresholds, clusters at each, and scores the result
    against ``true_labels``. Returns ``(best_threshold, best_f1, curve)`` where
    ``curve`` is the list of ``(threshold, f1)`` pairs (useful for plotting).

    Note: with a single region this is in-sample calibration of one scalar
    parameter. For an unbiased estimate, calibrate on held-out graphs and apply
    the chosen threshold to unseen data — a single threshold cannot meaningfully
    overfit thousands of pairs, but the principle matters at scale.
    """
    if thresholds is None:
        thresholds = np.linspace(0.30, 0.90, 25)
    rng = np.random.default_rng(seed)

    curve: list[tuple[float, float]] = []
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        labels = greedy_cluster(
            n_nodes, edge_src, edge_dst, edge_probs,
            threshold=float(t), rng=np.random.default_rng(rng.integers(1 << 30)),
        )
        f1 = pairwise_precision_recall(labels, true_labels, ignore_label=ignore_label)["f1"]
        curve.append((float(t), float(f1)))
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1, curve
