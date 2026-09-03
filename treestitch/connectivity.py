"""Connectivity accuracy — kept here for backward compatibility.

The implementation now lives in :mod:`neuronauts.metrics.connectome`, which
works on plain arrays instead of ``Region`` objects. This module adapts
``Region``-shaped calls to that array API. New code should call
:mod:`neuronauts.metrics` directly (:func:`connectome_metrics`,
:func:`dual_side_connectome_metrics`).

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

from collections import Counter

import numpy as np

from neuronauts.metrics.connectome import (
    connectome_metrics,
    dual_side_connectome_metrics,
    edge_set_prf1,
    match_clusters_majority,
    undirected_edge_set,
)

# Re-exported under their historical private names: a couple of callers in
# this repo (attic/prior_results/spatial_variance.py) import these directly.
_prf1 = edge_set_prf1
_match_clusters_to_neurons = match_clusters_majority
_undirected_edge_set = undirected_edge_set


def connectome_from_partition(
    pred_labels: np.ndarray,
    region,
    *,
    ignore_label: int = 0,
) -> Counter:
    """Build the predicted connection table from a pre-side partition.

    Each synapse i contributes a directed edge
        (pred_labels[i], region.post_root_id[i])
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


def connectome_accuracy(
    pred_labels: np.ndarray,
    region,
    *,
    min_syn: int = 1,
    ignore_label: int = 0,
) -> dict:
    """Measure how well the predicted partition preserves synapse-level connectivity.

    See :func:`neuronauts.metrics.connectome.connectome_metrics` for the
    algorithm; this is that function applied to ``region.pre_root_id`` /
    ``region.post_root_id``.
    """
    return connectome_metrics(
        pred_labels, region.pre_root_id, region.post_root_id,
        min_syn=min_syn, ignore=ignore_label,
    )


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

    See :func:`neuronauts.metrics.connectome.dual_side_connectome_metrics`;
    this unpacks the two ``Region`` objects into the array arguments it needs.
    """
    return dual_side_connectome_metrics(
        pred_pre=pred_pre,
        syn_id_pre=region_pre.synapse_id,
        true_pre=region_pre.pre_root_id,
        true_post=region_pre.post_root_id,
        pred_post=pred_post,
        syn_id_post=region_post.synapse_id,
        true_post_on_post_side=region_post.post_root_id,
        min_syn=min_syn, ignore=ignore_label,
    )


__all__ = [
    "connectome_accuracy",
    "connectome_from_partition",
    "dual_side_connectome_accuracy",
]
