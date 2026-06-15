"""Connectivity accuracy metrics for neuron partition evaluation.

Measures how well a predicted pre-side partition preserves the true synapse-level
connectome (directed neuron→neuron connection graph).

A false merge of neurons A and B into cluster AB inflates connectivity: all of A's
targets and all of B's targets become attributed to a single "super-neuron", creating
spurious or duplicated directed edges in the reconstructed circuit.

Usage
-----
    from treestitch.connectivity import connectome_accuracy

    # region.post_root_id must be real (non-zero) — populated by build_region_world
    metrics = connectome_accuracy(pred_labels, region)
    print(f"conn_edge_F1={metrics['conn_edge_f1']:.3f}  "
          f"syn_attr_acc={metrics['synapse_attr_acc']:.3f}")

Requires
--------
    region.pre_root_id  [N] int64 — true pre-neuron label per synapse
    region.post_root_id [N] int64 — true post-neuron label per synapse (must be real)
    pred_labels         [N] int64 — predicted cluster per synapse (from partition)
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def connectome_from_partition(
    pred_labels: np.ndarray,
    region,
    *,
    ignore_label: int = 0,
) -> Counter:
    """Build the predicted connection table from a pre-side partition.

    Each synapse i contributes a directed edge
        (pred_labels[i], region.post_root_id[i])
    representing "predicted pre-neuron → true post-neuron".

    Parameters
    ----------
    pred_labels:
        [N] int64 — predicted cluster per synapse (output of partition_observations_cc).
    region:
        Region with real post_root_id (non-zero).
    ignore_label:
        Skip observations where pred_labels == ignore_label (abstained) or
        post_root_id == ignore_label (unknown post neuron).

    Returns
    -------
    Counter mapping (pred_pre_cluster, true_post_root_id) → synapse count.
    """
    post = region.post_root_id
    counts: Counter = Counter()
    for i in range(len(pred_labels)):
        pre_c = int(pred_labels[i])
        post_r = int(post[i])
        if pre_c == ignore_label or post_r == ignore_label:
            continue
        counts[(pre_c, post_r)] += 1
    return counts


def _true_connectome(region, *, ignore_label: int = 0) -> Counter:
    """True directed connection table: (pre_root_id, post_root_id) → synapse count."""
    counts: Counter = Counter()
    pre = region.pre_root_id
    post = region.post_root_id
    for i in range(len(pre)):
        p = int(pre[i])
        q = int(post[i])
        if p == ignore_label or q == ignore_label:
            continue
        counts[(p, q)] += 1
    return counts


def _match_clusters_to_neurons(pred_labels: np.ndarray, true_labels: np.ndarray,
                                ignore: int = 0) -> dict[int, int]:
    """Map each predicted cluster to the majority-vote true neuron."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for p, t in zip(pred_labels, true_labels):
        p, t = int(p), int(t)
        if p == ignore or t == ignore:
            continue
        votes[p][t] += 1
    return {c: cnt.most_common(1)[0][0] for c, cnt in votes.items()}


def connectome_accuracy(
    pred_labels: np.ndarray,
    region,
    *,
    min_syn: int = 1,
    ignore_label: int = 0,
) -> dict:
    """Measure how well the predicted partition preserves synapse-level connectivity.

    Builds the true connectome {(pre_root_id, post_root_id): count} and the
    predicted connectome {(pred_cluster, post_root_id): count}, matches predicted
    clusters to true neurons by majority vote, then computes directed-edge F1 and
    per-synapse attribution accuracy.

    Parameters
    ----------
    pred_labels:
        [N] int64 — predicted cluster per synapse.
    region:
        Region with real pre_root_id and post_root_id (both non-zero where meaningful).
    min_syn:
        Minimum synapse count for a directed connection to count as an edge.
        Default 1 (every synapse counts).
    ignore_label:
        Skip observations with this label on either side (0 = unlabelled).

    Returns
    -------
    dict with keys:
        synapse_attr_acc   — fraction of labelled synapses correctly attributed to
                             their (pre_neuron, post_neuron) pair after cluster→neuron
                             majority-vote matching
        conn_edge_precision — fraction of predicted directed edges (≥min_syn) that
                              are true directed edges
        conn_edge_recall    — fraction of true directed edges recovered by prediction
        conn_edge_f1        — harmonic mean of precision and recall
        n_true_edges        — unique directed (A→B) connections in ground truth
        n_pred_edges        — unique directed connections in prediction
        n_synapses_labelled — synapses with known pre AND post neuron
    """
    pre = region.pre_root_id
    post = region.post_root_id

    # Match predicted clusters to true neurons by majority vote
    cluster_to_neuron = _match_clusters_to_neurons(pred_labels, pre, ignore=ignore_label)

    # Synapse attribution accuracy
    n_labelled = 0
    n_correct = 0
    for i in range(len(pred_labels)):
        pc = int(pred_labels[i])
        tr = int(pre[i])
        po = int(post[i])
        if pc == ignore_label or tr == ignore_label or po == ignore_label:
            continue
        n_labelled += 1
        mapped = cluster_to_neuron.get(pc, -1)
        if mapped == tr:
            n_correct += 1
    synapse_attr_acc = n_correct / n_labelled if n_labelled > 0 else float("nan")

    # True and predicted connection sets (directed edges with ≥ min_syn synapses)
    true_counts = _true_connectome(region, ignore_label=ignore_label)
    pred_counts = connectome_from_partition(pred_labels, region, ignore_label=ignore_label)

    # Remap predicted connections: (pred_cluster, post) → (mapped_neuron, post)
    remapped_pred: Counter = Counter()
    for (pred_c, post_r), cnt in pred_counts.items():
        mapped = cluster_to_neuron.get(pred_c, -1)
        if mapped != -1:
            remapped_pred[(mapped, post_r)] += cnt

    true_edges = {e for e, c in true_counts.items() if c >= min_syn}
    pred_edges = {e for e, c in remapped_pred.items() if c >= min_syn}

    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if precision != precision or recall != recall:  # nan check
        f1 = float("nan")
    else:
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

    return {
        "synapse_attr_acc": synapse_attr_acc,
        "conn_edge_precision": precision,
        "conn_edge_recall": recall,
        "conn_edge_f1": f1,
        "n_true_edges": len(true_edges),
        "n_pred_edges": len(pred_edges),
        "n_synapses_labelled": n_labelled,
    }
