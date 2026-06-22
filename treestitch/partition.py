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
    franken_hard_frac: float = 0.1,
    max_train_nodes: int = 0,
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
    from spatially close cross-neuron edges.  ``franken_hard_frac`` of negatives
    are drawn from frankenmerge cut edges (type-0 same-fragment edges that cross
    a neuron boundary), explicitly oversampling the rarest but most important
    training signal.

    ``max_train_nodes``: when > 0 and graph has more nodes, each epoch trains on a
    random spatial subgraph of this many nodes instead of the full graph.  Prevents
    OOM on large training regions (e.g. 178k nodes / 3M edges).

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
        franken_hard_frac=franken_hard_frac,
        max_train_nodes=max_train_nodes,
        device=device,
        seed=seed,
        log_every=log_every,
    )


def partition_observations_cc(
    model: Any,
    graph: ObservationGraph,
    *,
    bias: float = 0.0,
    abstain_threshold: float = 0.0,
    device: str = "cpu",
) -> np.ndarray:
    """Partition observations with edge classifier + correlation clustering.

    Lifts per-edge same-object log-odds to a global partition via greedy
    additive edge contraction (GAEC).  Unlike threshold union-find, a single
    spuriously-similar cross-object edge cannot force an irreversible merge —
    the contraction respects the *net* evidence between clusters.

    ``bias < 0`` clusters conservatively (lower over-merge rate); ``bias > 0``
    merges more aggressively.

    ``abstain_threshold > 0`` enables uncertainty-based abstention: observations
    where max_same_cluster_prob − max_diff_cluster_prob < threshold are left
    unassigned (label -k) rather than forced into a potentially-wrong cluster.
    Frankenmerge boundary synapses are the primary target — their type-0 edges
    claim one cluster while their spatial k-NN edges point toward another.
    """
    from neuronauts.assemble.edge_partition import partition_by_correlation

    return partition_by_correlation(
        model, graph,  # type: ignore[arg-type]
        bias=bias, abstain_threshold=abstain_threshold, device=device,
    )


def partition_observations_soft(
    model: Any,
    graph: ObservationGraph,
    *,
    bias: float = 0.0,
    abstain_threshold: float = 0.0,
    device: str = "cpu",
) -> dict:
    """Probabilistic connectome readout: hard clusters + per-observation confidence.

    Extends ``partition_observations_cc`` with a soft membership distribution
    over predicted clusters for each observation.  Uncertain slivers and
    frankenmerge fragments get fractional membership rather than forced hard
    assignments.

    Probabilistic connectome construction
    --------------------------------------
    Connection probability between neuron A and neuron B via synapse (pre, post):

        P(A→B via synapse) = P(pre_obs in A) × P(post_obs in B)

    Summing over all synapse pairs gives a weighted adjacency matrix where
    high-confidence synapses contribute near-1.0 weight and uncertain slivers
    contribute partial weights proportional to their assignment confidence.

    Returns
    -------
    dict with keys:
        pred             [N] int64   — hard cluster IDs
        cluster_conf     [N] float32 — max_same_p − max_diff_p (confidence margin)
        membership_probs [N, K] float32 — row-normalised soft assignment
        cluster_ids      [K] int64   — cluster ID per column
        entropy          [N] float32 — Shannon entropy (high = uncertain)
        abstain_mask     [N] bool    — True for abstained observations
    """
    from neuronauts.assemble.edge_partition import soft_partition

    return soft_partition(
        model, graph,  # type: ignore[arg-type]
        bias=bias, abstain_threshold=abstain_threshold, device=device,
    )


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


def assemble_partition_shapes(
    fragment_list: list,
    pred_labels: np.ndarray,
    seg_ids: np.ndarray,
    *,
    stitch_radius_nm: float = 5_000.0,
    min_fragments: int = 1,
) -> dict:
    """Build a dict of {cluster_id → merged neuron Fragment} from a partition.

    Takes the output of ``partition_observations_cc`` and the Fragment objects
    from world-building to produce whole-neuron skeleton geometries.  Each
    merged Fragment is a tree (or forest) produced by Kruskal stitching of the
    constituent fragment skeletons — no cycles can be introduced.

    Parameters
    ----------
    fragment_list:
        List[Fragment] from ``build_region_world`` or ``build_lineage_world``.
    pred_labels:
        [N] int64 per-synapse cluster IDs (negative = abstained, ignored).
    seg_ids:
        [N] int64 per-synapse v117 root IDs — ``graph.fragment_id`` from ObservationGraph.
    stitch_radius_nm:
        Max endpoint gap in nm for bridging adjacent fragment skeletons.
    min_fragments:
        Skip clusters with fewer than this many distinct fragments.

    Returns
    -------
    dict[int, Fragment]
        One merged-skeleton Fragment per predicted neuron cluster.
    """
    from treestitch.assemble import assemble_partition_shapes as _aps

    return _aps(
        fragment_list, pred_labels, seg_ids,
        stitch_radius_nm=stitch_radius_nm,
        min_fragments=min_fragments,
    )


def neuron_shape_metrics(neuron: Any) -> dict:
    """Morphological sanity metrics for an assembled neuron skeleton Fragment.

    Returns
    -------
    dict with keys:
        cable_length_um        — total edge length in micrometres
        n_branch_points        — degree-≥3 vertices
        n_endpoints            — degree-≤1 vertices (leaves)
        n_connected_components — 1 = fully connected; >1 = stitch gap or merge error
        is_tree                — True iff no cycles (n_edges == n_verts − n_components)
        bbox_volume_um3        — axis-aligned bounding-box volume in μm³
    """
    from treestitch.assemble import neuron_shape_metrics as _nsm

    return _nsm(neuron)


def train_edge_partition_multi_region(
    graphs: list,
    *,
    n_epochs: int = 150,
    lr: float = 1e-3,
    d_model: int = 64,
    output_dim: int = 32,
    n_layers: int = 3,
    dropout: float = 0.1,
    weight_decay: float = 0.0,
    max_edges_per_epoch: int = 4000,
    hard_neg_frac: float = 0.5,
    franken_hard_frac: float = 0.3,
    max_train_nodes: int = 0,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 10,
) -> tuple:
    """Train an EdgePartitionGNN on multiple spatial regions simultaneously.

    Concatenates the per-region ObservationGraphs into a single mega-graph
    (edges remain intra-region — no spurious cross-region connections) and
    calls the standard single-graph training loop.  This forces the model to
    learn transferable features (e.g. the abstract synaptic signature of a
    frankenmerge) rather than memorising which root IDs are frankenmerges in
    one specific spatial region.

    Parameters
    ----------
    graphs:
        List[ObservationGraph] — one per training bbox.  Must all have the
        same ``node_dim`` (identical FragmentEncoder output dimension).

    Returns
    -------
    (model, history)  — same as ``train_edge_partition``.
    """
    from treestitch.graph import concat_observation_graphs
    from treestitch.schemas import ObservationGraph

    typed = [g if isinstance(g, ObservationGraph) else ObservationGraph.from_half_synapse_graph(g)
             for g in graphs]
    mega = concat_observation_graphs(typed)
    n_regions = len(graphs)
    n_nodes = [g.n_nodes for g in typed]
    n_edges = [g.n_edges for g in typed]
    print(f"Multi-region training: {n_regions} regions, "
          f"{mega.n_nodes} nodes ({n_nodes}), {mega.n_edges} edges ({n_edges})")

    return train_edge_partition(
        mega,
        n_epochs=n_epochs,
        lr=lr,
        d_model=d_model,
        output_dim=output_dim,
        n_layers=n_layers,
        dropout=dropout,
        weight_decay=weight_decay,
        max_edges_per_epoch=max_edges_per_epoch,
        hard_neg_frac=hard_neg_frac,
        franken_hard_frac=franken_hard_frac,
        max_train_nodes=max_train_nodes,
        device=device,
        seed=seed,
        log_every=log_every,
    )


__all__ = [
    "PartitionGNN",
    "train_partition",
    "partition_observations",
    "evaluate_partition",
    "train_edge_partition",
    "train_edge_partition_multi_region",
    "partition_observations_cc",
    "partition_observations_soft",
    "partition_observations_tiled",
    "merge_metrics",
    "assemble_partition_shapes",
    "neuron_shape_metrics",
]


def _extract_subgraph(g, node_indices: np.ndarray):
    """Return a new ObservationGraph containing only ``node_indices`` nodes.

    Edges whose src or dst is absent are dropped; remaining indices are
    remapped to 0…len(node_indices)-1.
    """
    from treestitch.schemas import ObservationGraph
    node_indices = np.asarray(node_indices, dtype=np.int64)
    n_orig = g.n_nodes
    remap = np.full(n_orig, -1, dtype=np.int64)
    remap[node_indices] = np.arange(len(node_indices), dtype=np.int64)
    # Vectorized edge mask: both endpoints must be in the keep set
    src_remap = remap[g.edge_src]
    dst_remap = remap[g.edge_dst]
    edge_mask = (src_remap >= 0) & (dst_remap >= 0)
    return ObservationGraph(
        node_feat=g.node_feat[node_indices],
        node_pos=g.node_pos[node_indices],
        edge_src=src_remap[edge_mask],
        edge_dst=dst_remap[edge_mask],
        edge_type=g.edge_type[edge_mask],
        edge_feat=g.edge_feat[edge_mask],
        labels=g.labels[node_indices],
        fragment_id=g.fragment_id[node_indices],
        side=g.side,
    )


def partition_observations_tiled(
    model,
    graph,
    *,
    tile_size: int = 600,
    bias: float = 0.0,
    device: str = "cpu",
) -> np.ndarray:
    """Partition a large ObservationGraph by spatial tiling.

    For graphs with ≤ ``tile_size`` nodes, falls back to
    ``partition_observations_cc`` directly.  For larger graphs:

    1. Sort nodes along the longest spatial axis.
    2. Partition contiguous tiles of ``tile_size`` nodes independently.
    3. Assign globally unique cluster IDs per tile.
    4. Reconcile: nodes sharing a same-fragment edge (edge_type==0) across
       tile boundaries must end up in the same cluster.  A union-find pass
       over same-fragment edges achieves this without any model calls.

    The reconciliation preserves the fragment-level co-membership guarantee
    that same-fragment synapses belong to the same neuron, which is the only
    cross-tile constraint the GNN would have seen during training.
    """
    n = graph.n_nodes
    if n <= tile_size:
        return partition_observations_cc(model, graph, bias=bias, device=device)

    # Sort by the dimension with greatest spread for compact tiles.
    spread = graph.node_pos.max(axis=0) - graph.node_pos.min(axis=0)
    sort_dim = int(np.argmax(spread))
    sorted_idx = np.argsort(graph.node_pos[:, sort_dim])

    pred = np.full(n, -1, dtype=np.int64)
    next_cluster = 0
    for start in range(0, n, tile_size):
        tile_nodes = sorted_idx[start:start + tile_size]
        tile_g = _extract_subgraph(graph, tile_nodes)
        tile_pred = partition_observations_cc(model, tile_g, bias=bias, device=device)
        pred[tile_nodes] = tile_pred + next_cluster
        next_cluster += int(tile_pred.max()) + 1

    # Union-find reconciliation over same-fragment edges (edge_type == 0).
    parent = np.arange(next_cluster, dtype=np.int64)

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    frag_id = graph.fragment_id
    # For each fragment, union all cluster IDs assigned to its nodes.
    frag_rep: dict[int, int] = {}
    for node in range(n):
        fid = int(frag_id[node])
        c = int(pred[node])
        if fid not in frag_rep:
            frag_rep[fid] = c
        else:
            rc, rr = _find(c), _find(frag_rep[fid])
            if rc != rr:
                parent[rc] = rr

    # Remap to contiguous IDs.
    canonical: dict[int, int] = {}
    result = np.empty(n, dtype=np.int64)
    next_id = 0
    for node in range(n):
        root = _find(int(pred[node]))
        if root not in canonical:
            canonical[root] = next_id
            next_id += 1
        result[node] = canonical[root]
    return result


def fragment_completeness(
    root_label_map: dict[int, set[int]],
) -> dict[int, bool]:
    """Classify each v117 fragment as complete (no edit needed) or not.

    A fragment is *complete* when it maps 1-to-1 with a single v1718 neuron:
    the v117 root and the v1718 root are already identical — no merging with
    other fragments is needed, and it isn't a frankenmerge that needs splitting.

    Parameters
    ----------
    root_label_map:
        ``{v117_root: set_of_v1718_roots}`` as returned by
        ``build_region_world``.  Entries with ``len > 1`` are frankenmerges.

    Returns
    -------
    dict[int, bool]
        ``True``  → fragment already correct, no edit needed.
        ``False`` → fragment needs merging with others, or is a frankenmerge.
    """
    # Invert: v1718 → all v117 roots that map to it
    v1718_to_v117s: dict[int, list[int]] = {}
    for v117, v1718s in root_label_map.items():
        for v1718 in v1718s:
            v1718_to_v117s.setdefault(v1718, []).append(v117)

    result: dict[int, bool] = {}
    for v117, v1718s in root_label_map.items():
        if len(v1718s) != 1:
            # frankenmerge: straddles two v1718 neurons → needs splitting
            result[v117] = False
            continue
        (v1718,) = v1718s
        # complete only if this is the sole contributor to the v1718 neuron
        result[v117] = len(v1718_to_v117s[v1718]) == 1

    return result


def completeness_metrics(
    root_label_map: dict[int, set[int]],
    pred_completeness: dict[int, bool],
) -> dict:
    """Precision / recall / F1 for completeness prediction.

    Parameters
    ----------
    root_label_map:
        Ground-truth ``{v117_root: set_of_v1718_roots}`` from world-building.
    pred_completeness:
        ``{v117_root: predicted_complete_bool}`` from a model or heuristic.

    Returns
    -------
    dict with keys: precision, recall, f1, accuracy, n_complete_gt, n_fragments.
    """
    gt = fragment_completeness(root_label_map)
    common = [f for f in gt if f in pred_completeness]
    if not common:
        return {"precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "accuracy": float("nan"),
                "n_complete_gt": sum(gt.values()), "n_fragments": len(gt)}

    y_true = np.array([gt[f] for f in common], dtype=bool)
    y_pred = np.array([pred_completeness[f] for f in common], dtype=bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())
    acc = float((y_true == y_pred).mean())
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = (2 * prec * rec / (prec + rec)
            if prec + rec > 0 else float("nan"))

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "n_complete_gt": int(y_true.sum()),
        "n_fragments": len(common),
    }
