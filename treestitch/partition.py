"""PartitionGNN: cluster observations into parent-tree groups.

The PartitionGNN reads the typed-edge ObservationGraph and learns per-node
embeddings such that observations from the same parent tree are close and
observations from different trees are far apart.

After training, partition_observations performs conservative union-find
clustering: merge observation pairs whose cosine similarity exceeds a
threshold.  evaluate_partition measures cluster quality (ARI).

Usage
-----
    from treestitch.partition import PartitionGNN, train_partition, partition_observations, evaluate_partition

    gnn, history = train_partition(graph, n_epochs=40, lr=1e-3)
    pred_labels = partition_observations(gnn, graph, threshold=0.87)
    metrics = evaluate_partition(pred_labels, graph.labels)
    print(f"ARI = {metrics['ari']:.3f}")
"""

from __future__ import annotations

from typing import Any

import numpy as np

from treestitch.schemas import ObservationGraph


def PartitionGNN(
    input_dim: int,
    d_model: int = 64,
    n_layers: int = 3,
    output_dim: int = 32,
    dropout: float = 0.1,
    n_edge_types: int | None = None,
) -> Any:
    """Construct a PartitionGNN module.

    The architecture is a typed-edge message-passing GNN: one message
    projection per edge type, scatter-add aggregation, residual + LayerNorm.
    Output is L2-normalised.

    Parameters
    ----------
    input_dim:
        Node feature dimension (= 3 + embed_dim from ObservationGraph.node_dim).
    d_model:
        Hidden dimension.
    n_layers:
        Number of message-passing layers.
    output_dim:
        Embedding dimension of the output.
    dropout:
        Dropout rate in the update MLP.
    n_edge_types:
        Number of distinct edge types.  ``None`` means it is inferred at
        training time from ``ObservationGraph.edge_type.max() + 1``.
        For manual construction pass the expected value (2 without endpoint
        edges, 3 with).
    """
    from neuronauts.assemble.partition_gnn import HalfSynapseGNN

    n_types = n_edge_types if n_edge_types is not None else 2
    return HalfSynapseGNN(
        input_dim=input_dim,
        d_model=d_model,
        n_layers=n_layers,
        n_edge_types=n_types,
        output_dim=output_dim,
        dropout=dropout,
    )


def train_partition(
    graph: ObservationGraph,
    *,
    n_epochs: int = 50,
    lr: float = 1e-3,
    margin: float = 0.5,
    max_pairs: int = 1000,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 10,
    hard_neg_frac: float = 0.5,
) -> tuple[Any, dict]:
    """Train a PartitionGNN on an ObservationGraph.

    Uses cosine contrastive loss: pull same-object pairs to cos_sim → 1,
    push different-object pairs to cos_sim < (1 − margin).

    50% of negatives are drawn from the hard-negative pool: type-1/2 edges
    that connect observations from different objects (spatially close but
    actually separate — the hardest cases for the model).

    Returns
    -------
    (gnn, history)
        gnn — trained PartitionGNN (eval mode).
        history — {"loss": [...], "pos_sim": [...], "neg_sim": [...]}.
    """
    import torch

    from neuronauts.assemble.partition_gnn import train_partition_gnn

    # Convert ObservationGraph → HalfSynapseGraph-compatible namespace
    # train_partition_gnn accesses graph.node_feat, graph.edge_src, etc.
    # ObservationGraph has the same fields, so we can pass it directly.
    return train_partition_gnn(
        graph,  # type: ignore[arg-type]
        n_epochs=n_epochs,
        lr=lr,
        margin=margin,
        max_pairs=max_pairs,
        device=device,
        seed=seed,
        log_every=log_every,
        hard_neg_frac=hard_neg_frac,
    )


def partition_observations(
    gnn: Any,
    graph: ObservationGraph,
    *,
    threshold: float = 0.8,
    same_fragment_threshold: float | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Partition observations into object clusters via GNN embeddings.

    Conservative union-find: only merge pairs whose cosine similarity
    exceeds the threshold.  Under-merging is acceptable; over-merging is
    harder to recover from.

    Parameters
    ----------
    threshold:
        Cosine similarity threshold for spatial (type 1) and
        endpoint-adjacent (type 2) edges.
    same_fragment_threshold:
        Threshold for same-fragment (type 0) edges.  Defaults to
        ``threshold``.  Use a lower value to aggressively merge same-fragment
        observations, or a higher value to resist frankenmerge errors.

    Returns
    -------
    ndarray [N] int64 — consecutive cluster IDs starting from 0.
    """
    from neuronauts.assemble.partition_gnn import partition_half_synapses

    return partition_half_synapses(
        gnn,
        graph,  # type: ignore[arg-type]
        threshold=threshold,
        same_seg_threshold=same_fragment_threshold,
        device=device,
    )


def evaluate_partition(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    *,
    ignore_label: int = 0,
) -> dict:
    """Evaluate cluster quality.

    Returns a dict with:
        ari, homogeneity, completeness, v_measure,
        n_clusters_pred, n_clusters_true, n_nodes.

    ``ignore_label`` (default 0) marks unlabelled / held-out observations.
    """
    from neuronauts.assemble.partition_gnn import evaluate_partition_ari

    return evaluate_partition_ari(pred_labels, true_labels, ignore_label=ignore_label)


# ---------------------------------------------------------------------------
# Edge-classification + correlation clustering: learn f(fragment -> object)
# ---------------------------------------------------------------------------

def train_edge_partition(
    graph: ObservationGraph,
    *,
    n_epochs: int = 60,
    lr: float = 1e-3,
    d_model: int = 64,
    output_dim: int = 32,
    n_layers: int = 3,
    dropout: float = 0.1,
    weight_decay: float = 0.0,
    max_edges_per_epoch: int = 4000,
    hard_neg_frac: float = 0.5,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 10,
) -> tuple[Any, dict]:
    """Train an edge classifier that learns the fragment→object function.

    This is the direct supervised formulation: for every edge (a pair of
    observations joined by same-fragment / spatial / endpoint evidence), learn
    the probability that the two share a single parent object.  In the neuro
    domain this is exactly learning f(v117 seg → v1412 neuron) at the edge
    level.

    Each training epoch uses a balanced mini-batch (``max_edges_per_epoch`` edges,
    split 50/50 positives/negatives) with ``hard_neg_frac`` of negatives drawn
    from spatially close cross-neuron edges.  This prevents the class-imbalance
    collapse where the model predicts "same" for all edges.

    Returns
    -------
    (model, history)
        model — trained EdgePartitionGNN (eval mode).
        history — {"loss", "p_pos", "p_neg", "edge_acc"} per epoch.
    """
    from neuronauts.assemble.edge_partition import train_edge_partition_gnn

    return train_edge_partition_gnn(
        graph,  # type: ignore[arg-type]
        n_epochs=n_epochs,
        lr=lr,
        d_model=d_model,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout=dropout,
        weight_decay=weight_decay,
        max_edges_per_epoch=max_edges_per_epoch,
        hard_neg_frac=hard_neg_frac,
        device=device,
        seed=seed,
        log_every=log_every,
    )


def partition_observations_cc(
    model: Any,
    graph: ObservationGraph,
    *,
    bias: float = 0.0,
    device: str = "cpu",
) -> np.ndarray:
    """Partition observations with edge classifier + correlation clustering.

    Lifts per-edge same-object log-odds to a global partition via greedy
    additive edge contraction (GAEC).  Unlike threshold union-find, a single
    spuriously-similar cross-object edge cannot force an irreversible merge —
    the contraction respects the *net* evidence between clusters.

    ``bias < 0`` clusters conservatively (lower over-merge rate); ``bias > 0``
    merges more aggressively.
    """
    from neuronauts.assemble.edge_partition import partition_by_correlation

    return partition_by_correlation(model, graph, bias=bias, device=device)  # type: ignore[arg-type]


def merge_metrics(
    graph: ObservationGraph,
    pred_labels: np.ndarray,
    *,
    ignore_label: int = 0,
) -> dict:
    """Edge-level over/under-merge metrics for a predicted partition.

    Returns merge_precision / merge_recall / merge_f1 plus over_merge_rate and
    under_merge_rate.  Over-merge (false merge of two objects) is the costly,
    hard-to-undo error; under-merge can be fixed by a later stitching pass.
    """
    from neuronauts.assemble.edge_partition import edge_merge_metrics

    return edge_merge_metrics(graph, pred_labels, ignore_label=ignore_label)  # type: ignore[arg-type]


__all__ = [
    "PartitionGNN",
    "train_partition",
    "partition_observations",
    "evaluate_partition",
    "train_edge_partition",
    "partition_observations_cc",
    "merge_metrics",
]
