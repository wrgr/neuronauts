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


def _undirected_edge_set(counts: Counter, *, min_syn: int = 1) -> set:
    """Collapse a directed (a, b) -> count Counter into canonical undirected edges.

    Counts of (a, b) and (b, a) are summed before applying the min_syn threshold, so
    a reciprocal connection is one undirected edge. Self-loops (a == b, autapses) are
    kept. Returns a set of canonical (min(a, b), max(a, b)) tuples with summed count
    >= min_syn.
    """
    merged: Counter = Counter()
    for (a, b), c in counts.items():
        key = (a, b) if a <= b else (b, a)
        merged[key] += c
    return {e for e, c in merged.items() if c >= min_syn}


def _prf1(true_edges: set, pred_edges: set) -> tuple[float, float, float]:
    """Precision, recall, F1 between two edge sets (nan-safe)."""
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
    return precision, recall, f1


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
        conn_edge_precision_undir / conn_edge_recall_undir / conn_edge_f1_undir
                            — same as above but on the UNDIRECTED neuron-neuron graph
                              (A↔B; reciprocal connections summed into one edge)
        n_true_edges_undir / n_pred_edges_undir — unique undirected connections
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

    # Directed edge F1: (A -> B) ordered pairs with >= min_syn synapses.
    true_edges = {e for e, c in true_counts.items() if c >= min_syn}
    pred_edges = {e for e, c in remapped_pred.items() if c >= min_syn}
    precision, recall, f1 = _prf1(true_edges, pred_edges)

    # Undirected edge F1: (A <-> B) unordered pairs; reciprocal counts summed.
    true_edges_u = _undirected_edge_set(true_counts, min_syn=min_syn)
    pred_edges_u = _undirected_edge_set(remapped_pred, min_syn=min_syn)
    precision_u, recall_u, f1_u = _prf1(true_edges_u, pred_edges_u)

    return {
        "synapse_attr_acc": synapse_attr_acc,
        # directed
        "conn_edge_precision": precision,
        "conn_edge_recall": recall,
        "conn_edge_f1": f1,
        "n_true_edges": len(true_edges),
        "n_pred_edges": len(pred_edges),
        # undirected
        "conn_edge_precision_undir": precision_u,
        "conn_edge_recall_undir": recall_u,
        "conn_edge_f1_undir": f1_u,
        "n_true_edges_undir": len(true_edges_u),
        "n_pred_edges_undir": len(pred_edges_u),
        "n_synapses_labelled": n_labelled,
    }


def dual_side_connectome_accuracy(
    pred_pre: np.ndarray,
    region_pre,
    pred_post: np.ndarray,
    region_post,
    *,
    min_syn: int = 1,
    ignore_label: int = 0,
) -> dict:
    """Reconstruct the connectome from BOTH partitions and score it.

    Unlike :func:`connectome_accuracy` — which predicts the pre neuron and reads the
    post neuron from ground truth — this builds the connectome with NO ground-truth
    root ids: the pre neuron comes from the pre-side partition and the post neuron from
    the post-side partition. A physical synapse is observed on both sides (its pre point
    and its post point both fall in the bbox) and the two observations are joined by the
    shared CAVE synapse id (``region.synapse_id``).

    Steps
    -----
    1. ``syn_id -> pred_pre_cluster``  from ``region_pre.synapse_id`` + ``pred_pre``;
       ``syn_id -> pred_post_cluster`` from ``region_post.synapse_id`` + ``pred_post``.
    2. ``pred_pre_cluster  -> true pre neuron``  (majority vote on ``region_pre.pre_root_id``);
       ``pred_post_cluster -> true post neuron`` (majority vote on ``region_post.post_root_id``).
    3. For each synapse seen on BOTH sides, predicted edge ``(mapped_pre, mapped_post)``;
       true edge from ground truth ``(pre_root_id, post_root_id)`` of the same synapse.
    4. Directed and undirected edge F1 over the resulting connection sets.

    Parameters
    ----------
    pred_pre / pred_post:
        [N_pre] / [N_post] int64 — predicted clusters from each side's partition.
    region_pre / region_post:
        Regions from ``build_region_world(side="pre")`` / ``side="post")`` over the same
        bbox. Both must carry real ``synapse_id`` (the CAVE join key).
    min_syn:
        Minimum synapse count for a directed connection to count as an edge.
    ignore_label:
        Skip observations / neurons with this label on either side (0 = unlabelled).

    Returns
    -------
    dict with the directed + undirected F1 keys (same names as
    :func:`connectome_accuracy`) plus coverage diagnostics:
        n_synapses_both_sides — synapses joined across pre and post partitions
        n_synapses_pre_only / n_synapses_post_only — synapses seen on only one side
    """
    pre_ids = np.asarray(region_pre.synapse_id, dtype=np.int64)
    post_ids = np.asarray(region_post.synapse_id, dtype=np.int64)

    # Map predicted clusters -> true neurons on each side (majority vote).
    pre_cluster_to_neuron = _match_clusters_to_neurons(
        pred_pre, region_pre.pre_root_id, ignore=ignore_label)
    post_cluster_to_neuron = _match_clusters_to_neurons(
        pred_post, region_post.post_root_id, ignore=ignore_label)

    # syn_id -> predicted cluster on each side (ignore abstained/unlabelled).
    pre_by_syn = {int(s): int(c) for s, c in zip(pre_ids, pred_pre)
                  if int(c) != ignore_label and int(s) >= 0}
    post_by_syn = {int(s): int(c) for s, c in zip(post_ids, pred_post)
                   if int(c) != ignore_label and int(s) >= 0}

    # syn_id -> ground-truth (pre_root, post_root) from the pre-side region.
    true_pre = np.asarray(region_pre.pre_root_id, dtype=np.int64)
    true_post = np.asarray(region_pre.post_root_id, dtype=np.int64)
    gt_by_syn = {int(s): (int(p), int(q))
                 for s, p, q in zip(pre_ids, true_pre, true_post)}

    both = set(pre_by_syn) & set(post_by_syn)
    pre_only = set(pre_by_syn) - set(post_by_syn)
    post_only = set(post_by_syn) - set(pre_by_syn)

    pred_counts: Counter = Counter()
    true_counts: Counter = Counter()
    for sid in both:
        mapped_pre = pre_cluster_to_neuron.get(pre_by_syn[sid], -1)
        mapped_post = post_cluster_to_neuron.get(post_by_syn[sid], -1)
        if mapped_pre == -1 or mapped_post == -1:
            continue
        pred_counts[(mapped_pre, mapped_post)] += 1
        gt = gt_by_syn.get(sid)
        if gt is not None and gt[0] != ignore_label and gt[1] != ignore_label:
            true_counts[gt] += 1

    # Directed + undirected edge F1.
    true_edges = {e for e, c in true_counts.items() if c >= min_syn}
    pred_edges = {e for e, c in pred_counts.items() if c >= min_syn}
    precision, recall, f1 = _prf1(true_edges, pred_edges)

    true_edges_u = _undirected_edge_set(true_counts, min_syn=min_syn)
    pred_edges_u = _undirected_edge_set(pred_counts, min_syn=min_syn)
    precision_u, recall_u, f1_u = _prf1(true_edges_u, pred_edges_u)

    return {
        # directed
        "conn_edge_precision": precision,
        "conn_edge_recall": recall,
        "conn_edge_f1": f1,
        "n_true_edges": len(true_edges),
        "n_pred_edges": len(pred_edges),
        # undirected
        "conn_edge_precision_undir": precision_u,
        "conn_edge_recall_undir": recall_u,
        "conn_edge_f1_undir": f1_u,
        "n_true_edges_undir": len(true_edges_u),
        "n_pred_edges_undir": len(pred_edges_u),
        # coverage
        "n_synapses_both_sides": len(both),
        "n_synapses_pre_only": len(pre_only),
        "n_synapses_post_only": len(post_only),
    }
