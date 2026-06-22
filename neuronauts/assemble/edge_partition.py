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
    franken_hard_frac: float = 0.1,
    max_train_nodes: int = 0,
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

    ``franken_hard_frac``: fraction of sampled negatives taken from the frankenmerge
    cut pool (type-0 same-fragment edges that cross a neuron boundary).  These are
    the rarest negatives in the graph (typically < 1% of type-0 edges) but the most
    important for frankenmerge detection.  Explicit oversampling ensures the classifier
    sees enough "same fragment, different neuron → cut" examples despite their rarity.

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

    rng = np.random.default_rng(seed)
    half = max(1, max_edges_per_epoch // 2)

    # Pre-build edge type arrays as masks for fast vectorized pool building
    etype = graph.edge_type
    labels = graph.labels
    esrc, edst = graph.edge_src, graph.edge_dst

    def _build_pools(valid_mask, tgt, etype_arr):
        """Compute pos/neg/hard_neg/franken pools given per-edge valid+target arrays.

        ``etype_arr`` must be the same length as ``valid_mask`` and ``tgt``.
        """
        vi = np.where(valid_mask)[0]
        ta = tgt[vi]
        pos_i = vi[ta > 0.5]
        neg_i = vi[ta < 0.5]
        # Hard-negative: type 1/2 cross-neuron edges
        mask_hard = ((etype_arr == 1) | (etype_arr == 2)) & valid_mask & (tgt < 0.5)
        hn = np.where(mask_hard)[0] if mask_hard.any() else None
        # Franken cut: type-0 same-frag cross-neuron edges
        mask_fk = (etype_arr == 0) & valid_mask & (tgt < 0.5)
        fk = np.where(mask_fk)[0] if mask_fk.any() else None
        return pos_i, neg_i, hn, fk

    def _sample_batch(pos_i, neg_i, hn, fk, pw_val):
        """Sample balanced mini-batch; return (batch_idx, pos_weight_tensor)."""
        n_pos = min(half, len(pos_i))
        n_neg = min(half, len(neg_i))
        bp = pos_i[rng.choice(len(pos_i), n_pos, replace=len(pos_i) < n_pos)]
        parts = []
        rem = n_neg
        if hn is not None and rem > 0:
            nh = min(int(n_neg * hard_neg_frac), len(hn))
            parts.append(hn[rng.choice(len(hn), nh, replace=len(hn) < nh)])
            rem -= nh
        if fk is not None and rem > 0:
            nf = min(int(n_neg * franken_hard_frac), len(fk))
            parts.append(fk[rng.choice(len(fk), nf, replace=len(fk) < nf)])
            rem -= nf
        if rem > 0:
            parts.append(neg_i[rng.choice(len(neg_i), rem, replace=len(neg_i) < rem)])
        bn = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        return np.concatenate([bp, bn])

    use_subgraph = max_train_nodes > 0 and graph.n_nodes > max_train_nodes

    if not use_subgraph:
        # Full graph: load once to device (original path)
        pos_iv, neg_iv, hard_neg_arr, franken_cut_arr = _build_pools(valid_np, target_np, etype)
        n_all_pos, n_all_neg = len(pos_iv), len(neg_iv)
        pw = (max(n_all_neg, 1.0) / max(n_all_pos, 1.0)) if pos_weight is None else float(pos_weight)
        pos_weight_t = torch.tensor([pw], device=device)

        node_feat_t = torch.from_numpy(graph.node_feat).to(device)
        edge_src_t = torch.from_numpy(esrc).long().to(device)
        edge_dst_t = torch.from_numpy(edst).long().to(device)
        edge_type_t = torch.from_numpy(etype).long().to(device)
        efd = graph.edge_feat.shape[1] if graph.edge_feat.ndim == 2 else 0
        edge_feat_t = (
            torch.from_numpy(graph.edge_feat).float().to(device) if efd > 0 else None
        )
        target_t_full = torch.from_numpy(target_np.astype(np.float32)).to(device)
        valid_t_full = torch.from_numpy(valid_np).to(device)

        for epoch in range(1, n_epochs + 1):
            model.train()
            opt.zero_grad()

            batch_idx = _sample_batch(pos_iv, neg_iv, hard_neg_arr, franken_cut_arr, pw)
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
                logits_v = logits[valid_t_full].detach()
                probs_v = torch.sigmoid(logits_v)
                tgt_v = target_t_full[valid_t_full]
                pos_full = tgt_v > 0.5
                p_pos = float(probs_v[pos_full].mean().item()) if pos_full.any() else 0.0
                p_neg = float(probs_v[~pos_full].mean().item()) if (~pos_full).any() else 0.0
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
    else:
        # Subgraph-sampling path: sample max_train_nodes nodes per epoch,
        # extract their local subgraph, push only that to device.
        # Avoids loading the full 3M-edge graph onto GPU.
        # Node features have positions in the first 3 columns (normalised).
        # Sort by spatial dimension with greatest spread for coherent tiles.
        node_pos = graph.node_pos if hasattr(graph, 'node_pos') else graph.node_feat[:, :3]
        spread = node_pos.max(axis=0) - node_pos.min(axis=0)
        sort_dim = int(np.argmax(spread))
        sort_order = np.argsort(node_pos[:, sort_dim])  # spatial sort for locality

        # Pre-build a fast remap buffer and edge endpoint arrays
        n_total = graph.n_nodes
        remap_buf = np.full(n_total, -1, dtype=np.int64)
        efd = graph.edge_feat.shape[1] if graph.edge_feat.ndim == 2 else 0
        # Global pos_weight (approximate over full graph)
        gv_np, gt_np = valid_np, target_np
        n_all_pos_g = int((gv_np & (gt_np > 0.5)).sum())
        n_all_neg_g = int((gv_np & (gt_np < 0.5)).sum())
        pw = (max(n_all_neg_g, 1.0) / max(n_all_pos_g, 1.0)) if pos_weight is None else float(pos_weight)
        pos_weight_t = torch.tensor([pw], device=device)

        # Tile stride: shift the window each epoch to cover the full graph
        n_tiles = max(1, -(-n_total // max_train_nodes))  # ceil div
        tile_stride = n_total // n_tiles

        for epoch in range(1, n_epochs + 1):
            model.train()
            opt.zero_grad()

            # Pick a random starting offset within the sorted order, take a contiguous block
            tile_start = int(rng.integers(0, n_total))
            raw_idx = (tile_start + np.arange(max_train_nodes)) % n_total
            node_idx = sort_order[raw_idx]  # actual node indices in graph

            # Extract subgraph (vectorized) — reuse remap_buf for both passes.
            node_idx_sorted = np.sort(node_idx)
            remap_buf[node_idx_sorted] = np.arange(len(node_idx_sorted), dtype=np.int64)
            src_r = remap_buf[esrc]  # local src index (-1 if not in tile)
            dst_r = remap_buf[edst]  # local dst index (-1 if not in tile)
            emask = (src_r >= 0) & (dst_r >= 0)
            remap_buf[node_idx_sorted] = -1  # reset

            # Sub-edge indices into full graph edge array; local node indices
            sub_edge_full_idx = np.where(emask)[0]
            if len(sub_edge_full_idx) == 0:
                history["loss"].append(0.0)
                history["p_pos"].append(0.0)
                history["p_neg"].append(0.0)
                history["edge_acc"].append(0.0)
                continue

            sub_valid = valid_np[sub_edge_full_idx]
            sub_target = target_np[sub_edge_full_idx]

            sub_etype = etype[sub_edge_full_idx]
            pos_iv, neg_iv, hard_arr, fk_arr = _build_pools(sub_valid, sub_target, sub_etype)
            if len(pos_iv) == 0 and len(neg_iv) == 0:
                history["loss"].append(0.0)
                history["p_pos"].append(0.0)
                history["p_neg"].append(0.0)
                history["edge_acc"].append(0.0)
                continue

            batch_local = _sample_batch(pos_iv, neg_iv, hard_arr, fk_arr, pw)

            # Tensors for the subgraph only — local edge indices from src_r/dst_r
            sub_node_feat_t = torch.from_numpy(graph.node_feat[node_idx_sorted]).to(device)
            sub_src_t = torch.from_numpy(src_r[sub_edge_full_idx]).long().to(device)
            sub_dst_t = torch.from_numpy(dst_r[sub_edge_full_idx]).long().to(device)
            sub_etype_t = torch.from_numpy(sub_etype).long().to(device)
            sub_efeat_t = (
                torch.from_numpy(graph.edge_feat[sub_edge_full_idx]).float().to(device)
                if efd > 0 else None
            )
            sub_target_t = torch.from_numpy(sub_target.astype(np.float32)).to(device)

            _, logits = model(sub_node_feat_t, sub_src_t, sub_dst_t, sub_etype_t, sub_efeat_t)
            logits_batch = logits[batch_local]
            tgt_batch = sub_target_t[batch_local]

            loss = F.binary_cross_entropy_with_logits(
                logits_batch, tgt_batch, pos_weight=pos_weight_t
            )
            loss.backward()
            opt.step()

            with torch.no_grad():
                sub_valid_t = torch.from_numpy(sub_valid).to(device)
                probs_v = torch.sigmoid(logits[sub_valid_t].detach())
                tgt_v = sub_target_t[sub_valid_t]
                pos_m = tgt_v > 0.5
                p_pos = float(probs_v[pos_m].mean().item()) if pos_m.any() else 0.0
                p_neg = float(probs_v[~pos_m].mean().item()) if (~pos_m).any() else 0.0
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
                    f"  (subgraph {len(node_idx_sorted)}n/{len(sub_edge_full_idx)}e)"
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

def _abstain_uncertain(
    pred: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    probs: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Reassign low-confidence observations to unique abstain IDs (-1, -2, ...).

    For each observation, compute a confidence margin:
        confidence = max_same_cluster_prob - max_diff_cluster_prob

    where max_same_cluster_prob is the highest co-membership probability among
    edges connecting this observation to any other observation in the same
    predicted cluster, and max_diff_cluster_prob is the highest among edges to
    any other cluster.

    Low confidence signals conflicting evidence: the observation belongs to one
    cluster by fragment identity (type-0 edges predict "merge") but evidence
    from spatial neighbors suggests it belongs somewhere else.  Frankenmerge
    boundary synapses are the canonical case — they have type-0 edges pulling
    them into the wrong fragment's cluster AND spatial k-NN edges pointing toward
    their true cluster.  Rather than forcing a wrong assignment, return -k
    (unique negative per abstained node) so downstream users can treat it as
    unassigned.

    threshold=0.0 disables abstention (default, backward-compatible).
    """
    N = len(pred)
    max_same_p = np.zeros(N, dtype=np.float32)
    max_diff_p = np.zeros(N, dtype=np.float32)

    same_mask = pred[edge_src] == pred[edge_dst]
    diff_mask = ~same_mask
    p32 = probs.astype(np.float32)

    np.maximum.at(max_same_p, edge_src[same_mask], p32[same_mask])
    np.maximum.at(max_same_p, edge_dst[same_mask], p32[same_mask])
    np.maximum.at(max_diff_p, edge_src[diff_mask], p32[diff_mask])
    np.maximum.at(max_diff_p, edge_dst[diff_mask], p32[diff_mask])

    confidence = max_same_p - max_diff_p
    uncertain_idx = np.where(confidence < threshold)[0]
    result = pred.copy()
    if len(uncertain_idx) > 0:
        result[uncertain_idx] = -np.arange(1, len(uncertain_idx) + 1, dtype=np.int64)
    return result


def partition_by_correlation(
    model: Any,
    graph: HalfSynapseGraph,
    *,
    bias: float = 0.0,
    abstain_threshold: float = 0.0,
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
    abstain_threshold:
        Confidence margin below which an observation is left unassigned (label -k)
        rather than kept in its GAEC-assigned cluster.  Confidence is defined as
        ``max_same_cluster_prob − max_diff_cluster_prob`` over the observation's edges.
        Low confidence signals conflicting evidence — the canonical case is a
        frankenmerge boundary synapse whose type-0 edges pull it into the wrong
        fragment's cluster while its spatial k-NN edges point toward its true cluster.
        Default 0.0 = no abstention (backward-compatible).  Values in [0.1, 0.5]
        are reasonable starting points; higher values abstain more aggressively.
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    ndarray [N] int64 — cluster IDs.  Abstained observations have unique negative
    IDs (-1, -2, ...) and are treated as singletons by downstream metrics.
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

    logits_np = logits.cpu().numpy().astype(np.float64)
    weights = logits_np + float(bias)
    pred = correlation_cluster(graph.n_nodes, graph.edge_src, graph.edge_dst, weights)

    if abstain_threshold > 0.0:
        probs = (1.0 / (1.0 + np.exp(-logits_np))).astype(np.float32)
        pred = _abstain_uncertain(pred, graph.edge_src, graph.edge_dst, probs,
                                   abstain_threshold)
    return pred


def soft_partition(
    model: Any,
    graph: HalfSynapseGraph,
    *,
    bias: float = 0.0,
    abstain_threshold: float = 0.0,
    device: str = "cpu",
) -> dict:
    """Probabilistic connectome readout: hard clusters + per-observation confidence.

    Runs the full partition pipeline (GAEC + optional abstention) and additionally
    computes, for each observation, a soft membership distribution over predicted
    clusters.  This is the foundation for a probabilistic connectome where
    uncertain slivers and frankenmerge fragments contribute fractional edge
    weights rather than hard assignments.

    Probabilistic connectome construction
    --------------------------------------
    The connection probability between neuron A and neuron B via a particular
    synapse (pre_obs, post_obs) is:

        P(A→B via this synapse) = P(pre_obs in A) × P(post_obs in B)

    Summing over all synapse pairs gives the weighted adjacency matrix where
    high-confidence synapses contribute near-1.0 weight and uncertain slivers
    contribute partial weights proportional to their assignment confidence.

    Expert contribution
    -------------------
    Observations ranked by entropy of ``membership_probs[i]`` are the highest-
    value targets for human review.  A single expert decision (obs i → cluster k)
    propagates via the edge predictions to update neighboring uncertainties.

    Parameters
    ----------
    model, graph, bias, abstain_threshold, device:
        Same as ``partition_by_correlation``.

    Returns
    -------
    dict with keys:
        pred          [N] int64  — hard cluster IDs (same as partition_by_correlation)
        cluster_conf  [N] float32 — confidence in hard assignment
                                    = max_same_cluster_prob − max_diff_cluster_prob
        membership_probs [N, K] float32 — row-normalised probability over K clusters
                                           (K = number of distinct non-abstain clusters)
        cluster_ids   [K] int64  — cluster ID corresponding to each column
        entropy       [N] float32 — Shannon entropy of membership_probs[i];
                                    high = uncertain, low = confident
        abstain_mask  [N] bool   — True for observations that were abstained
    """
    import torch

    if graph.n_edges == 0:
        pred = np.arange(graph.n_nodes, dtype=np.int64)
        N = graph.n_nodes
        K = N
        return {
            "pred": pred,
            "cluster_conf": np.ones(N, dtype=np.float32),
            "membership_probs": np.eye(N, dtype=np.float32),
            "cluster_ids": pred.copy(),
            "entropy": np.zeros(N, dtype=np.float32),
            "abstain_mask": np.zeros(N, dtype=bool),
        }

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

    logits_np = logits.cpu().numpy().astype(np.float64)
    probs = (1.0 / (1.0 + np.exp(-logits_np))).astype(np.float32)
    weights = logits_np + float(bias)
    pred = correlation_cluster(graph.n_nodes, graph.edge_src, graph.edge_dst, weights)

    if abstain_threshold > 0.0:
        pred = _abstain_uncertain(pred, graph.edge_src, graph.edge_dst, probs,
                                   abstain_threshold)

    abstain_mask = pred < 0
    N = graph.n_nodes

    # Per-observation confidence: max_same_cluster_p - max_diff_cluster_p
    max_same_p = np.zeros(N, dtype=np.float32)
    max_diff_p = np.zeros(N, dtype=np.float32)
    same_mask = pred[graph.edge_src] == pred[graph.edge_dst]
    diff_mask = ~same_mask
    np.maximum.at(max_same_p, graph.edge_src[same_mask], probs[same_mask])
    np.maximum.at(max_same_p, graph.edge_dst[same_mask], probs[same_mask])
    np.maximum.at(max_diff_p, graph.edge_src[diff_mask], probs[diff_mask])
    np.maximum.at(max_diff_p, graph.edge_dst[diff_mask], probs[diff_mask])
    cluster_conf = max_same_p - max_diff_p

    # Soft membership: for each observation i, P(i in cluster k) ∝ mean edge
    # prediction to neighbours already in cluster k.  For confident observations
    # this collapses to a delta; for uncertain ones it spreads across candidates.
    non_abstain_ids = sorted(set(int(x) for x in pred if x >= 0))
    cluster_id_to_col = {cid: col for col, cid in enumerate(non_abstain_ids)}
    K = len(non_abstain_ids)
    membership = np.zeros((N, K), dtype=np.float32)

    for e_idx in range(len(graph.edge_src)):
        s = int(graph.edge_src[e_idx])
        d = int(graph.edge_dst[e_idx])
        p_e = float(probs[e_idx])
        # Edge (s, d): update soft evidence of s toward d's cluster and vice versa
        d_cluster = int(pred[d])
        s_cluster = int(pred[s])
        if d_cluster >= 0 and d_cluster in cluster_id_to_col:
            membership[s, cluster_id_to_col[d_cluster]] += p_e
        if s_cluster >= 0 and s_cluster in cluster_id_to_col:
            membership[d, cluster_id_to_col[s_cluster]] += p_e

    # Row-normalise; fall back to uniform for isolated nodes
    row_sums = membership.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    membership /= row_sums

    # Shannon entropy H = -sum(p log p), clipped at log(K) for normalisation
    eps = 1e-9
    log_p = np.log(membership + eps)
    entropy = -(membership * log_p).sum(axis=1).astype(np.float32)

    return {
        "pred": pred,
        "cluster_conf": cluster_conf,
        "membership_probs": membership,
        "cluster_ids": np.array(non_abstain_ids, dtype=np.int64),
        "entropy": entropy,
        "abstain_mask": abstain_mask,
    }


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

    # Fraction of same-fragment (type-0) edges that cross a neuron boundary.
    # These are the ground-truth frankenmerge cuts the edge classifier must learn.
    type0 = graph.edge_type == 0
    valid_t0 = type0 & valid
    franken_cut_mask = valid_t0 & (lab_s != lab_d)
    n_franken_edges = int(franken_cut_mask.sum())
    frankenmerge_rate = n_franken_edges / max(int(valid_t0.sum()), 1)

    # Among those frankenmerge same-fragment edges, what fraction does the
    # predicted partition correctly split? (The "Bar 3" viability metric.)
    # 1.0 = every frankenmerge pair lands in different predicted clusters.
    frankenmerge_split_recall = (
        float((franken_cut_mask & (pred[src] != pred[dst])).sum())
        / max(n_franken_edges, 1)
    )

    # Abstain rate: fraction of observations with negative predicted label (unassigned)
    abstain_rate = float((pred < 0).sum()) / max(len(pred), 1)

    return {
        "merge_precision": precision,
        "merge_recall": recall,
        "merge_f1": f1,
        "over_merge_rate": fp / n,
        "under_merge_rate": fn / n,
        "n_edges_eval": n,
        "n_merges_pred": tp + fp,          # edges the model chose to merge
        "n_splits_pred": n - (tp + fp),    # edges the model chose to split
        "n_true_merges": tp + fn,          # ground-truth same-neuron edges
        "tp_merges": tp,                   # correct merges
        "fp_merges": fp,                   # false merges (over-merge errors)
        "fn_merges": fn,                   # missed merges (under-merge errors)
        "tn_splits": n - (tp + fp + fn),   # correct splits
        "n_edges_total": n,
        "frankenmerge_rate": frankenmerge_rate,
        "frankenmerge_split_recall": frankenmerge_split_recall,
        "abstain_rate": abstain_rate,
    }


__all__ = [
    "EdgePartitionGNN",
    "train_edge_partition_gnn",
    "correlation_cluster",
    "partition_by_correlation",
    "_abstain_uncertain",
    "soft_partition",
    "edge_merge_metrics",
]
