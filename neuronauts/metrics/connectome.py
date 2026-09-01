"""Connectome-level metrics: does the partition preserve who talks to whom.

A false merge of neurons A and B does not just cost pair-level precision; it
attributes all of A's targets to B and vice versa, inventing directed edges in
the reconstructed circuit. These metrics build the neuron-to-neuron connection
table from the partition and score it against ground truth.

Two protocols:

* :func:`connectome_metrics` predicts the *pre* side and reads the post
  neuron from ground truth (the treestitch default). Predicted clusters are
  matched to true neurons by majority vote before edges are compared.
* :func:`dual_side_connectome_metrics` uses **no** ground-truth root on either
  side: pre from the pre-side partition, post from the post-side partition,
  joined on the CAVE synapse id.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from ._core import NAN, prf1


def undirected_edge_set(counts: Counter, *, min_syn: int = 1) -> set:
    """Collapse ``{(a, b): count}`` into canonical undirected edges.

    Counts of ``(a, b)`` and ``(b, a)`` are summed before thresholding, so a
    reciprocal connection is one edge. Autapses (``a == b``) are kept.
    """
    merged: Counter = Counter()
    for (a, b), c in counts.items():
        merged[(a, b) if a <= b else (b, a)] += c
    return {e for e, c in merged.items() if c >= min_syn}


def edge_set_prf1(true_edges: set, pred_edges: set) -> tuple[float, float, float]:
    """Precision / recall / F1 between two edge sets (NaN when undefined)."""
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    return prf1(tp, fp, fn)


def match_clusters_majority(pred, true, *, ignore=0) -> dict[int, int]:
    """Map each predicted cluster to the true label holding the majority."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for p, t in zip(np.asarray(pred).tolist(), np.asarray(true).tolist()):
        if p == ignore or t == ignore:
            continue
        votes[p][t] += 1
    return {c: cnt.most_common(1)[0][0] for c, cnt in votes.items()}


def _directed_counts(a, b, *, ignore) -> Counter:
    counts: Counter = Counter()
    for x, y in zip(np.asarray(a).tolist(), np.asarray(b).tolist()):
        if x == ignore or y == ignore:
            continue
        counts[(x, y)] += 1
    return counts


def _score_edge_sets(true_counts: Counter, pred_counts: Counter, *, min_syn: int) -> dict:
    true_edges = {e for e, c in true_counts.items() if c >= min_syn}
    pred_edges = {e for e, c in pred_counts.items() if c >= min_syn}
    p, r, f = edge_set_prf1(true_edges, pred_edges)
    true_u = undirected_edge_set(true_counts, min_syn=min_syn)
    pred_u = undirected_edge_set(pred_counts, min_syn=min_syn)
    pu, ru, fu = edge_set_prf1(true_u, pred_u)
    return {
        "conn_edge_precision": p,
        "conn_edge_recall": r,
        "conn_edge_f1": f,
        "n_true_edges": len(true_edges),
        "n_pred_edges": len(pred_edges),
        "conn_edge_precision_undir": pu,
        "conn_edge_recall_undir": ru,
        "conn_edge_f1_undir": fu,
        "n_true_edges_undir": len(true_u),
        "n_pred_edges_undir": len(pred_u),
    }


def connectome_metrics(
    pred_pre,
    true_pre,
    true_post,
    *,
    min_syn: int = 1,
    ignore=0,
) -> dict:
    """Score the connectome implied by a pre-side partition.

    Parameters
    ----------
    pred_pre:
        ``[N]`` predicted cluster per synapse.
    true_pre, true_post:
        ``[N]`` true pre- and post-neuron per synapse. Items where either is
        ``ignore`` are skipped.
    min_syn:
        Minimum synapse count for a directed connection to count as an edge.

    Returns
    -------
    ``synapse_attr_acc`` (fraction of labelled synapses whose cluster maps to
    their true pre neuron), the directed and undirected ``conn_edge_*`` block,
    and ``n_synapses_labelled``.
    """
    pred_pre = np.asarray(pred_pre)
    true_pre = np.asarray(true_pre)
    true_post = np.asarray(true_post)
    cluster_to_neuron = match_clusters_majority(pred_pre, true_pre, ignore=ignore)

    n_labelled = n_correct = 0
    remapped: Counter = Counter()
    for pc, tr, po in zip(pred_pre.tolist(), true_pre.tolist(), true_post.tolist()):
        if pc == ignore or tr == ignore or po == ignore:
            continue
        n_labelled += 1
        mapped = cluster_to_neuron.get(pc, -1)
        if mapped == tr:
            n_correct += 1
        if mapped != -1:
            remapped[(mapped, po)] += 1

    out = {"synapse_attr_acc": n_correct / n_labelled if n_labelled else NAN}
    out.update(_score_edge_sets(
        _directed_counts(true_pre, true_post, ignore=ignore), remapped, min_syn=min_syn))
    out["n_synapses_labelled"] = n_labelled
    return out


def dual_side_connectome_metrics(
    pred_pre,
    syn_id_pre,
    true_pre,
    true_post,
    pred_post,
    syn_id_post,
    true_post_on_post_side,
    *,
    min_syn: int = 1,
    ignore=0,
) -> dict:
    """Score the connectome built from a pre-side AND a post-side partition.

    ``syn_id_*`` are the CAVE synapse ids that join the two observation sets;
    ``true_pre`` / ``true_post`` are ground truth aligned to the pre side and
    ``true_post_on_post_side`` is the post-neuron truth aligned to the post
    side (used only for majority-vote matching of post clusters).

    Returns the ``conn_edge_*`` block plus coverage counts
    ``n_synapses_both_sides, n_synapses_pre_only, n_synapses_post_only``.
    """
    pre_ids = np.asarray(syn_id_pre, dtype=np.int64)
    post_ids = np.asarray(syn_id_post, dtype=np.int64)
    pred_pre = np.asarray(pred_pre)
    pred_post = np.asarray(pred_post)

    pre_c2n = match_clusters_majority(pred_pre, true_pre, ignore=ignore)
    post_c2n = match_clusters_majority(pred_post, true_post_on_post_side, ignore=ignore)

    pre_by_syn = {int(s): c for s, c in zip(pre_ids, pred_pre.tolist())
                  if c != ignore and int(s) >= 0}
    post_by_syn = {int(s): c for s, c in zip(post_ids, pred_post.tolist())
                   if c != ignore and int(s) >= 0}
    gt_by_syn = {int(s): (p, q) for s, p, q in zip(
        pre_ids, np.asarray(true_pre).tolist(), np.asarray(true_post).tolist())}

    both = set(pre_by_syn) & set(post_by_syn)
    pred_counts: Counter = Counter()
    true_counts: Counter = Counter()
    for sid in both:
        mp = pre_c2n.get(pre_by_syn[sid], -1)
        mq = post_c2n.get(post_by_syn[sid], -1)
        if mp == -1 or mq == -1:
            continue
        pred_counts[(mp, mq)] += 1
        gt = gt_by_syn.get(sid)
        if gt is not None and gt[0] != ignore and gt[1] != ignore:
            true_counts[gt] += 1

    out = _score_edge_sets(true_counts, pred_counts, min_syn=min_syn)
    out.update({
        "n_synapses_both_sides": len(both),
        "n_synapses_pre_only": len(set(pre_by_syn) - set(post_by_syn)),
        "n_synapses_post_only": len(set(post_by_syn) - set(pre_by_syn)),
    })
    return out


__all__ = [
    "connectome_metrics",
    "dual_side_connectome_metrics",
    "edge_set_prf1",
    "match_clusters_majority",
    "undirected_edge_set",
]
