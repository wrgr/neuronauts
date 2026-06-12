"""Edge-classification + correlation clustering for partition learning.

This is the direct formulation of the core task: **learn f(v117 → v1412)**.

The v117 segmentation is a noisy over/under-segmentation of the true v1412
neurons.  Instead of learning a per-node embedding and merging by a cosine
threshold (``partition_gnn.py``), this module learns an explicit *edge*
function: for every pair of observations joined by evidence (same-segment,
spatial, or endpoint-adjacent), predict the probability that the two share a
single v1412 identity.  Training supervises that probability directly against
the ground-truth v1412 co-membership of the edge's endpoints.

Inference then lifts the per-edge predictions to a global relabelling with
**correlation clustering** (greedy additive edge contraction, GAEC).  Unlike
threshold union-find, correlation clustering can *cut* a high-confidence edge
when the rest of the graph disagrees — the net weight between two clusters can
go negative even if a few constituent edges are positive — so a single
spuriously-similar cross-neuron edge no longer triggers an irreversible merge.

Pipeline
--------
    model, history = train_edge_partition_gnn(graph)          # learn f(117→1412)
    pred = partition_by_correlation(model, graph)             # global relabel
    ari  = evaluate_partition_ari(pred, graph.labels)         # quality
    merr = edge_merge_metrics(graph, pred)                    # over/under-merge

The over-merge rate in ``edge_merge_metrics`` is the operationally costly error
(a false merge of two neurons is hard to undo downstream); the ``bias`` knob on
``partition_by_correlation`` trades it against under-merge.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from typing import Any

import numpy as np

from .half_synapse_graph import HalfSynapseGraph
from .partition_gnn import HalfSynapseGNN


# ---------------------------------------------------------------------------
# Model: embedding backbone + edge-classification head
# ---------------------------------------------------------------------------

class EdgePartitionGNN:
    """Typed-edge GNN with an edge-classification head.

    The backbone is the same typed message-passing GNN as ``HalfSynapseGNN``
    (one message projection per edge type, residual + LayerNorm, L2-normalised
    output).  On top sits an MLP head that scores each candidate edge from the
    embeddings of its two endpoints plus the edge feature vector:

        logit(u, v) = MLP([emb_u, emb_v, |emb_u - emb_v|, emb_u * emb_v,
                           edge_feat]) -> R

    ``sigmoid(logit)`` is the predicted probability that ``u`` and ``v`` belong
    to the same v1412 object.

    Created via ``EdgePartitionGNN(input_dim, ...)`` — factory ``__new__`` so
    the torch module is built lazily when torch is available.
    """

    def __new__(
        cls,
        input_dim: int,
        d_model: int = 64,
        n_layers: int = 3,
        n_edge_types: int = 2,
        output_dim: int = 32,
        dropout: float = 0.1,
        edge_feat_dim: int = 3,
    ):
        import torch
        import torch.nn as nn

        backbone = HalfSynapseGNN(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_edge_types=n_edge_types,
            output_dim=output_dim,
            dropout=dropout,
        )

        class _EdgeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                # head input: [emb_u, emb_v, |emb_u-emb_v|, emb_u*emb_v,
                #              edge_feat, |pos_u-pos_v| (3), ||pos_u-pos_v|| (1)]
                # The endpoint separation is the franken-merge discriminator: a
                # same-segment edge spanning a large distance is almost certainly
                # a v117 merge error, not a true within-object link.
                head_in = output_dim * 4 + edge_feat_dim + 4
                self.edge_head = nn.Sequential(
                    nn.Linear(head_in, d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Linear(d_model // 2, 1),
                )
                self._output_dim = output_dim
                self._edge_feat_dim = edge_feat_dim

            def embed(self, node_feat, edge_src, edge_dst, edge_type):
                return self.backbone(node_feat, edge_src, edge_dst, edge_type)

            def edge_logits(self, emb, node_pos, score_src, score_dst, score_feat):
                u = emb[score_src]
                v = emb[score_dst]
                parts = [u, v, (u - v).abs(), u * v]
                if score_feat is not None and self._edge_feat_dim > 0:
                    parts.append(score_feat)
                # Spatial separation of the two endpoints (the first 3 node
                # feature columns are the normalised position).
                pu = node_pos[score_src]
                pv = node_pos[score_dst]
                dpos = (pu - pv).abs()
                dist = dpos.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
                parts.append(dpos)
                parts.append(dist)
                feats = torch.cat(parts, dim=-1)
                return self.edge_head(feats).squeeze(-1)

            def forward(self, node_feat, edge_src, edge_dst, edge_type, edge_feat):
                emb = self.embed(node_feat, edge_src, edge_dst, edge_type)
                node_pos = node_feat[:, :3]
                logits = self.edge_logits(emb, node_pos, edge_src, edge_dst, edge_feat)
                return emb, logits

        return _EdgeModel()


# ---------------------------------------------------------------------------
# Training: supervise the edge function against v1412 co-membership
# ---------------------------------------------------------------------------

def _edge_targets(graph: HalfSynapseGraph) -> tuple[np.ndarray, np.ndarray]:
    """Return (valid_mask, target) for every edge.

    target[e] = 1.0 if the endpoints share a v1412 label, else 0.0.
    valid_mask[e] = True only when both endpoints carry a known label
    (label != 0); unlabelled edges are excluded from the loss.
    """
    labels = graph.labels
    src = graph.edge_src
    dst = graph.edge_dst
    if len(src) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)
    lab_s = labels[src]
    lab_d = labels[dst]
    valid = (lab_s != 0) & (lab_d != 0)
    target = (lab_s == lab_d).astype(np.float32)
    return valid, target


def train_edge_partition_gnn(
    graph: HalfSynapseGraph,
    *,
    n_epochs: int = 60,
    lr: float = 1e-3,
    d_model: int = 64,
    output_dim: int = 32,
    n_layers: int = 3,
    dropout: float = 0.1,
    weight_decay: float = 0.0,
    pos_weight: float | None = 1.0,
    max_edges_per_epoch: int = 4000,
    hard_neg_frac: float = 0.5,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 10,
) -> tuple[Any, dict]:
    """Train an EdgePartitionGNN to predict v1412 co-membership per edge.

    This is the supervised core of learning f(v117 → v1412): each graph edge
    connects two v117-seg observations, and the model learns whether they
    belong to the same v1412 neuron.  Loss is binary cross-entropy.

    ``pos_weight`` scales the positive (same-object) class in the BCE loss:
      - ``1.0`` (default) — no reweighting.
      - ``None`` — auto-balance to ``n_neg / n_pos``.

    ``max_edges_per_epoch``: max total edges per mini-batch, split evenly between
    positives and negatives.  Prevents class-imbalance collapse (most graph edges
    are within-neuron; without balancing the classifier predicts "same" for all edges).

    ``hard_neg_frac``: fraction of sampled negatives taken from the hard-negative
    pool (spatial/endpoint-adjacent edges crossing neuron boundaries — the confusable
    negatives the model must actively learn to push apart).  The rest are random.

    Parameters
    ----------
    graph:
        HalfSynapseGraph / ObservationGraph with ``labels`` (v1412 supervision)
        and typed edges.
    n_epochs, lr, weight_decay:
        Standard Adam training controls.
    d_model, output_dim, n_layers, dropout:
        Backbone architecture.

    Returns
    -------
    (model, history)
        model — trained EdgePartitionGNN (eval mode).
        history — {"loss", "p_pos", "p_neg", "edge_acc"} per epoch, where
        ``p_pos``/``p_neg`` are the mean predicted same-object probability on
        true-positive / true-negative edges (separation is the learning signal).
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)

    n_edge_types = (
        max(2, int(graph.edge_type.max()) + 1) if len(graph.edge_type) > 0 else 2
    )
    edge_feat_dim = graph.edge_feat.shape[1] if graph.edge_feat.ndim == 2 else 0

    model = EdgePartitionGNN(
        input_dim=graph.node_dim,
        d_model=d_model,
        n_layers=n_layers,
        n_edge_types=n_edge_types,
        output_dim=output_dim,
        dropout=dropout,
        edge_feat_dim=edge_feat_dim,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    valid_np, target_np = _edge_targets(graph)
    history: dict[str, list[float]] = {"loss": [], "p_pos": [], "p_neg": [], "edge_acc": []}

    if valid_np.sum() == 0:
        model.eval()
        return model, history

    valid_idx = np.where(valid_np)[0]
    tgt_all = target_np[valid_idx]

    pos_idx_valid = valid_idx[tgt_all > 0.5]   # indices into full edge array
    neg_idx_valid = valid_idx[tgt_all < 0.5]

    # Hard-negative pool: cross-neuron spatial (type 1) and endpoint-adj (type 2) edges
    hard_neg_idx: list[int] = []
    for i in range(len(graph.edge_src)):
        if graph.edge_type[i] not in (1, 2):
            continue
        u, v = int(graph.edge_src[i]), int(graph.edge_dst[i])
        lu, lv = int(graph.labels[u]), int(graph.labels[v])
        if lu != 0 and lv != 0 and lu != lv:
            hard_neg_idx.append(i)
    hard_neg_arr = np.array(hard_neg_idx, dtype=np.int64) if hard_neg_idx else None

    rng = np.random.default_rng(seed)
    half = max(1, max_edges_per_epoch // 2)

    n_all_pos = len(pos_idx_valid)
    n_all_neg = len(neg_idx_valid)
    n_all_valid = n_all_pos + n_all_neg
    pw = (max(n_all_neg, 1.0) / max(n_all_pos, 1.0)) if pos_weight is None else float(pos_weight)
    pos_weight_t = torch.tensor([pw], device=device)

    node_feat_t = torch.from_numpy(graph.node_feat).to(device)
    edge_src_t = torch.from_numpy(graph.edge_src).long().to(device)
    edge_dst_t = torch.from_numpy(graph.edge_dst).long().to(device)
    edge_type_t = torch.from_numpy(graph.edge_type).long().to(device)
    edge_feat_dim = graph.edge_feat.shape[1] if graph.edge_feat.ndim == 2 else 0
    edge_feat_t = (
        torch.from_numpy(graph.edge_feat).float().to(device)
        if edge_feat_dim > 0
        else None
    )
    target_t_full = torch.from_numpy(target_np.astype(np.float32)).to(device)

    # For full-eval metrics (reporting), build full valid mask
    valid_t_full = torch.from_numpy(valid_np).to(device)

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()

        # Sample balanced mini-batch
        n_pos_batch = min(half, n_all_pos)
        n_neg_batch = min(half, n_all_neg)

        pos_sel = rng.choice(n_all_pos, n_pos_batch, replace=n_all_pos < n_pos_batch)
        batch_pos = pos_idx_valid[pos_sel]

        if hard_neg_arr is not None and n_neg_batch > 0:
            n_hard = min(int(n_neg_batch * hard_neg_frac), len(hard_neg_arr))
            n_rand = n_neg_batch - n_hard
            hard_sel = rng.choice(len(hard_neg_arr), n_hard, replace=len(hard_neg_arr) < n_hard)
            rand_sel = rng.choice(n_all_neg, n_rand, replace=n_all_neg < n_rand)
            batch_neg = np.concatenate([hard_neg_arr[hard_sel], neg_idx_valid[rand_sel]])
        else:
            rand_sel = rng.choice(n_all_neg, n_neg_batch, replace=n_all_neg < n_neg_batch)
            batch_neg = neg_idx_valid[rand_sel]

        batch_idx = np.concatenate([batch_pos, batch_neg])
        batch_t = torch.from_numpy(batch_idx).long().to(device)

        _, logits = model(node_feat_t, edge_src_t, edge_dst_t, edge_type_t, edge_feat_t)
        logits_batch = logits[batch_t]
        tgt_batch = target_t_full[batch_t]

        loss = F.binary_cross_entropy_with_logits(
            logits_batch, tgt_batch, pos_weight=pos_weight_t
        )
        loss.backward()
        opt.step()

        with torch.no_grad():
            # Report on the full valid set
            logits_v = logits[valid_t_full].detach()
            probs_v = torch.sigmoid(logits_v)
            tgt_v = target_t_full[valid_t_full]
            pos_full = tgt_v > 0.5
            neg_full = ~pos_full
            p_pos = float(probs_v[pos_full].mean().item()) if pos_full.any() else 0.0
            p_neg = float(probs_v[neg_full].mean().item()) if neg_full.any() else 0.0
            pred_link = probs_v > 0.5
            edge_acc = float((pred_link == (tgt_v > 0.5)).float().mean().item())

        history["loss"].append(float(loss.item()))
        history["p_pos"].append(p_pos)
        history["p_neg"].append(p_neg)
        history["edge_acc"].append(edge_acc)

        if log_every > 0 and (epoch % log_every == 0 or epoch == 1):
            print(
                f"  epoch {epoch:4d}: loss={loss.item():.4f}  "
                f"p_pos={p_pos:.3f}  p_neg={p_neg:.3f}  edge_acc={edge_acc:.3f}"
            )

    model.eval()
    return model, history


# ---------------------------------------------------------------------------
# Correlation clustering (greedy additive edge contraction)
# ---------------------------------------------------------------------------

def correlation_cluster(
    n_nodes: int,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_weight: np.ndarray,
) -> np.ndarray:
    """Greedy additive edge contraction (GAEC) for correlation clustering.

    Solves (heuristically) the correlation-clustering / multicut objective:
    partition the nodes to maximise the sum of within-cluster edge weights,
    where positive weights attract and negative weights repel.

    Repeatedly contracts the cluster pair with the largest positive aggregated
    weight; parallel edges are summed on contraction, so two clusters joined by
    several positive edges but one strong negative may have net-negative weight
    and stay split.  Stops when no positive aggregated weight remains.

    Bidirectional input edges (u, v) and (v, u) are summed into a single
    undirected weight, so the directed graph emitted by
    ``build_half_synapse_graph`` is handled directly.

    Parameters
    ----------
    n_nodes:
        Number of nodes.
    edge_src, edge_dst:
        Endpoint index arrays ``[E]``.
    edge_weight:
        Signed edge weight ``[E]`` — typically the edge-classifier log-odds
        (``logit``), positive ⇒ same object.

    Returns
    -------
    ndarray [n_nodes] int64 — consecutive cluster IDs from 0.
    """
    parent = list(range(n_nodes))
    rank = [0] * n_nodes

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Aggregate bidirectional / parallel edges into undirected cluster weights.
    adj: dict[int, dict[int, float]] = defaultdict(dict)
    for u, v, w in zip(
        edge_src.tolist(), edge_dst.tolist(), edge_weight.tolist()
    ):
        if u == v:
            continue
        adj[u][v] = adj[u].get(v, 0.0) + w
        adj[v][u] = adj[v].get(u, 0.0) + w

    heap: list[tuple[float, int, int]] = []
    for a in adj:
        for b, w in adj[a].items():
            if a < b:
                heap.append((-w, a, b))
    heapq.heapify(heap)

    while heap:
        neg_w, a, b = heapq.heappop(heap)
        w = -neg_w
        if w <= 0.0:
            break
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        cur = adj[ra].get(rb)
        if cur is None or abs(cur - w) > 1e-9:
            # Stale entry — a fresher one with the current weight is in the heap.
            continue

        # Contract rb into ra (union by rank, but keep ra as the surviving id).
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
            cur = adj[ra].get(rb)
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

        adj[ra].pop(rb, None)
        adj[rb].pop(ra, None)
        for x, wx in list(adj[rb].items()):
            adj[x].pop(rb, None)
            new_w = adj[ra].get(x, 0.0) + wx
            adj[ra][x] = new_w
            adj[x][ra] = new_w
            lo, hi = (ra, x) if ra < x else (x, ra)
            heapq.heappush(heap, (-new_w, lo, hi))
        adj[rb].clear()

    roots = [find(i) for i in range(n_nodes)]
    remap: dict[int, int] = {}
    out = np.empty(n_nodes, dtype=np.int64)
    for i, r in enumerate(roots):
        if r not in remap:
            remap[r] = len(remap)
        out[i] = remap[r]
    return out


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def partition_by_correlation(
    model: Any,
    graph: HalfSynapseGraph,
    *,
    bias: float = 0.0,
    device: str = "cpu",
) -> np.ndarray:
    """Partition observations via edge-classifier + correlation clustering.

    Runs the trained EdgePartitionGNN to get per-edge log-odds, shifts them by
    ``bias``, and contracts the graph with GAEC.

    Parameters
    ----------
    model:
        Trained EdgePartitionGNN.
    graph:
        HalfSynapseGraph / ObservationGraph.
    bias:
        Added to every edge log-odds before clustering.  ``bias < 0`` makes the
        clustering conservative (fewer, purer merges → lower over-merge rate);
        ``bias > 0`` merges more aggressively.  Default 0 = decision boundary at
        predicted probability 0.5.
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    ndarray [N] int64 — cluster IDs.
    """
    import torch

    if graph.n_edges == 0:
        return np.arange(graph.n_nodes, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        node_feat_t = torch.from_numpy(graph.node_feat).to(device)
        edge_src_t = torch.from_numpy(graph.edge_src).long().to(device)
        edge_dst_t = torch.from_numpy(graph.edge_dst).long().to(device)
        edge_type_t = torch.from_numpy(graph.edge_type).long().to(device)
        edge_feat_t = (
            torch.from_numpy(graph.edge_feat).float().to(device)
            if graph.edge_feat.ndim == 2 and graph.edge_feat.shape[1] > 0
            else None
        )
        _, logits = model(node_feat_t, edge_src_t, edge_dst_t, edge_type_t, edge_feat_t)

    weights = logits.cpu().numpy().astype(np.float64) + float(bias)
    return correlation_cluster(graph.n_nodes, graph.edge_src, graph.edge_dst, weights)


# ---------------------------------------------------------------------------
# Edge-level merge metrics (the over/under-merge asymmetry)
# ---------------------------------------------------------------------------

def edge_merge_metrics(
    graph: HalfSynapseGraph,
    pred_labels: np.ndarray,
    *,
    ignore_label: int = 0,
) -> dict:
    """Edge-level over/under-merge metrics for a predicted partition.

    Computed over edges with both endpoints labelled.  A *merge* is the event
    "endpoints land in the same predicted cluster".

        over_merge  = edges merged that are truly different neurons (false merge)
        under_merge = edges split that are truly the same neuron (missed merge)

    Over-merge is the operationally costly error: a false merge of two neurons
    is hard to recover downstream, whereas an under-merge can be fixed by a
    later stitching pass.

    Returns
    -------
    dict with keys:
        merge_precision  — TP / (TP + FP); 1.0 = no false merges
        merge_recall     — TP / (TP + FN); 1.0 = no missed merges
        merge_f1
        over_merge_rate  — FP / (TP + FP + FN + TN) over labelled edges
        under_merge_rate — FN / (TP + FP + FN + TN)
        n_edges_eval
    """
    pred = np.asarray(pred_labels, dtype=np.int64)
    labels = np.asarray(graph.labels, dtype=np.int64)
    src = graph.edge_src
    dst = graph.edge_dst

    if len(src) == 0:
        return {
            "merge_precision": 1.0, "merge_recall": 1.0, "merge_f1": 1.0,
            "over_merge_rate": 0.0, "under_merge_rate": 0.0, "n_edges_eval": 0,
        }

    lab_s = labels[src]
    lab_d = labels[dst]
    valid = (lab_s != ignore_label) & (lab_d != ignore_label)
    if valid.sum() == 0:
        return {
            "merge_precision": 1.0, "merge_recall": 1.0, "merge_f1": 1.0,
            "over_merge_rate": 0.0, "under_merge_rate": 0.0, "n_edges_eval": 0,
        }

    same_true = (lab_s == lab_d)[valid]
    same_pred = (pred[src] == pred[dst])[valid]

    tp = int(np.sum(same_true & same_pred))
    fp = int(np.sum(~same_true & same_pred))
    fn = int(np.sum(same_true & ~same_pred))
    n = int(valid.sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "merge_precision": precision,
        "merge_recall": recall,
        "merge_f1": f1,
        "over_merge_rate": fp / n,
        "under_merge_rate": fn / n,
        "n_edges_eval": n,
    }


__all__ = [
    "EdgePartitionGNN",
    "train_edge_partition_gnn",
    "correlation_cluster",
    "partition_by_correlation",
    "edge_merge_metrics",
]
