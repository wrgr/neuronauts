"""Typed-edge GNN for half-synapse partitioning (Phase 2.1).

The GNN learns to produce per-node embeddings such that nodes from the same
neuron are close in embedding space and nodes from different neurons are far
apart.  Same-segment edges carry a strong but noisy same-neuron signal; spatial
edges carry weak proximity evidence.  The model learns to weigh these by
optimising a contrastive loss against the ground-truth label-version partition.

Partition at inference:
    1.  Forward pass → L2-normalised per-node embeddings.
    2.  For every same-segment edge compute cosine similarity.
    3.  Union-find: merge node pairs whose cosine similarity > threshold.
    4.  Remaining isolated nodes become singletons.

This is deliberately conservative (under-merge is acceptable).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .half_synapse_graph import HalfSynapseGraph


# ---------------------------------------------------------------------------
# Numpy-only partition metrics (no sklearn dependency)
# ---------------------------------------------------------------------------

def _contingency(labels_a: np.ndarray, labels_b: np.ndarray):
    """Build contingency matrix and return (matrix, row_sums, col_sums)."""
    ua, a_inv = np.unique(labels_a, return_inverse=True)
    ub, b_inv = np.unique(labels_b, return_inverse=True)
    nc, nk = len(ua), len(ub)
    ct = np.zeros((nc, nk), dtype=np.int64)
    np.add.at(ct, (a_inv, b_inv), 1)
    return ct, ct.sum(axis=1), ct.sum(axis=0)


def _adjusted_rand_score_np(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand Index, implemented with numpy only."""
    n = len(labels_true)
    if n < 2:
        return 1.0
    ct, row_sum, col_sum = _contingency(labels_true, labels_pred)
    sum_comb_c = int(np.sum(ct * (ct - 1))) // 2
    sum_comb_a = int(np.sum(row_sum * (row_sum - 1))) // 2
    sum_comb_b = int(np.sum(col_sum * (col_sum - 1))) // 2
    n_comb = n * (n - 1) // 2
    expected = sum_comb_a * sum_comb_b / n_comb if n_comb > 0 else 0.0
    max_val = (sum_comb_a + sum_comb_b) / 2.0
    denominator = max_val - expected
    numerator = sum_comb_c - expected
    if abs(denominator) < 1e-10:
        return 1.0 if abs(numerator) < 1e-10 else 0.0
    return float(numerator / denominator)


def _homogeneity_completeness_np(labels_true: np.ndarray, labels_pred: np.ndarray):
    """Return (homogeneity, completeness, v_measure)."""
    n = len(labels_true)
    if n == 0:
        return 1.0, 1.0, 1.0

    ct, row_sum, col_sum = _contingency(labels_true, labels_pred)

    def _entropy(counts: np.ndarray) -> float:
        p = counts[counts > 0].astype(float) / n
        return float(-np.sum(p * np.log(p)))

    h_c = _entropy(row_sum)
    h_k = _entropy(col_sum)

    # H(C|K) and H(K|C) via contingency
    ct_f = ct.astype(float)
    col_safe = np.where(col_sum > 0, col_sum, 1).astype(float)
    row_safe = np.where(row_sum > 0, row_sum, 1).astype(float)

    # H(C|K) = -sum_{k} sum_{c} (n_ck/n) * log(n_ck / n_k)
    ratio_c_given_k = ct_f / col_safe[None, :]
    mask = ct > 0
    h_c_given_k = float(
        -np.sum(ct_f[mask] / n * np.log(ratio_c_given_k[mask]))
    )

    # H(K|C) = -sum_{c} sum_{k} (n_ck/n) * log(n_ck / n_c)
    ratio_k_given_c = ct_f / row_safe[:, None]
    h_k_given_c = float(
        -np.sum(ct_f[mask] / n * np.log(ratio_k_given_c[mask]))
    )

    homogeneity = 1.0 - h_c_given_k / h_c if h_c > 1e-10 else 1.0
    completeness = 1.0 - h_k_given_c / h_k if h_k > 1e-10 else 1.0
    denom = homogeneity + completeness
    v_measure = 2 * homogeneity * completeness / denom if denom > 1e-10 else 0.0
    return homogeneity, completeness, v_measure


# ---------------------------------------------------------------------------
# Union-Find for conservative merge
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def labels(self) -> np.ndarray:
        roots = [self.find(i) for i in range(len(self.parent))]
        unique_roots = sorted(set(roots))
        root_map = {r: i for i, r in enumerate(unique_roots)}
        return np.array([root_map[r] for r in roots], dtype=np.int64)


# ---------------------------------------------------------------------------
# HalfSynapseGNN
# ---------------------------------------------------------------------------

class HalfSynapseGNN:
    """Typed-edge GNN for half-synapse partition learning.

    Created via ``HalfSynapseGNN(input_dim, ...)`` — uses factory ``__new__``
    pattern so the torch module is instantiated lazily when torch is available.

    Architecture per layer:
    - Per-edge-type source projection: Linear(d_model) → messages
    - Scatter-add per type → [N, d_model] per type
    - Concat [x, agg_0, agg_1] → Linear(d_model*(n_edge_types+1), d_model)
    - + residual, LayerNorm
    Final projection + L2-normalise output.
    """

    def __new__(
        cls,
        input_dim: int,
        d_model: int = 64,
        n_layers: int = 3,
        n_edge_types: int = 2,
        output_dim: int = 32,
        dropout: float = 0.1,
    ):
        import torch
        import torch.nn as nn

        class _TypedMPLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.msg = nn.ModuleList([
                    nn.Linear(d_model, d_model) for _ in range(n_edge_types)
                ])
                self.update = nn.Sequential(
                    nn.Linear(d_model * (n_edge_types + 1), d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                )
                self.norm = nn.LayerNorm(d_model)

            def forward(self, x, edge_src, edge_dst, edge_type):
                N = x.size(0)
                aggs = []
                for t in range(n_edge_types):
                    mask = edge_type == t
                    agg = torch.zeros(N, d_model, device=x.device, dtype=x.dtype)
                    if mask.any():
                        src_t = edge_src[mask]
                        dst_t = edge_dst[mask]
                        msg = self.msg[t](x[src_t])
                        agg.scatter_add_(0, dst_t.unsqueeze(1).expand(-1, d_model), msg)
                    aggs.append(agg)
                combined = torch.cat([x] + aggs, dim=-1)
                updated = self.update(combined)
                return self.norm(x + updated)

        class _GNNModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, d_model)
                self.layers = nn.ModuleList([_TypedMPLayer() for _ in range(n_layers)])
                self.output_proj = nn.Linear(d_model, output_dim)
                self._input_dim = input_dim
                self._output_dim = output_dim

            def forward(self, node_feat, edge_src, edge_dst, edge_type):
                x = torch.relu(self.input_proj(node_feat))
                for layer in self.layers:
                    x = layer(x, edge_src, edge_dst, edge_type)
                out = self.output_proj(x)
                return torch.nn.functional.normalize(out, p=2, dim=-1)

        return _GNNModule()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _sample_pairs(
    label_groups: dict[int, list[int]],
    pos_labels: list[int],
    all_labels: list[int],
    max_pairs: int,
    rng: np.random.Generator,
) -> tuple[list, list]:
    pos_pairs: list[tuple[int, int]] = []
    neg_pairs: list[tuple[int, int]] = []

    if not pos_labels:
        return pos_pairs, neg_pairs

    rng.shuffle(pos_labels)
    for lbl in pos_labels:
        idxs = label_groups[lbl]
        if len(idxs) < 2:
            continue
        pair_a = int(rng.integers(len(idxs)))
        pair_b = int(rng.integers(len(idxs)))
        while pair_b == pair_a and len(idxs) > 1:
            pair_b = int(rng.integers(len(idxs)))
        pos_pairs.append((idxs[pair_a], idxs[pair_b]))
        if len(pos_pairs) >= max_pairs:
            break

    for _ in range(len(pos_pairs)):
        la = int(rng.choice(all_labels))
        lb = int(rng.choice(all_labels))
        while lb == la:
            lb = int(rng.choice(all_labels))
        ia = int(rng.choice(label_groups[la]))
        ib = int(rng.choice(label_groups[lb]))
        neg_pairs.append((ia, ib))

    return pos_pairs, neg_pairs


def train_partition_gnn(
    graph: HalfSynapseGraph,
    *,
    n_epochs: int = 50,
    lr: float = 1e-3,
    margin: float = 0.5,
    max_pairs: int = 1000,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 10,
) -> tuple[Any, dict]:
    """Train HalfSynapseGNN with contrastive loss on the label partition.

    Parameters
    ----------
    graph:
        HalfSynapseGraph with labels (label-version root IDs) and seg DNA.
    n_epochs:
        Training epochs.
    lr:
        Adam learning rate.
    margin:
        Hinge margin for negative pairs in cosine loss.
    max_pairs:
        Positive pairs sampled per epoch (equal negative pairs).
    device:
        ``"cpu"`` or ``"cuda"``.
    seed:
        RNG seed for pair sampling.
    log_every:
        Print loss every N epochs (0 = silent).

    Returns
    -------
    (gnn, history)
        gnn — trained HalfSynapseGNN module (eval mode).
        history — dict with ``loss``, ``pos_sim``, ``neg_sim`` lists.
    """
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    input_dim = graph.node_dim
    gnn = HalfSynapseGNN(input_dim).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=lr)

    node_feat_t = torch.from_numpy(graph.node_feat).to(device)
    edge_src_t = torch.from_numpy(graph.edge_src).long().to(device)
    edge_dst_t = torch.from_numpy(graph.edge_dst).long().to(device)
    edge_type_t = torch.from_numpy(graph.edge_type).long().to(device)

    # Build label groups for pair sampling (exclude label 0 = unlabelled)
    label_groups: dict[int, list[int]] = {}
    for i, lbl in enumerate(graph.labels):
        l = int(lbl)
        if l != 0:
            label_groups.setdefault(l, []).append(i)

    pos_labels = [lbl for lbl, idxs in label_groups.items() if len(idxs) >= 2]
    all_labels = list(label_groups.keys())

    history: dict[str, list[float]] = {"loss": [], "pos_sim": [], "neg_sim": []}

    for epoch in range(1, n_epochs + 1):
        gnn.train()
        opt.zero_grad()

        emb = gnn(node_feat_t, edge_src_t, edge_dst_t, edge_type_t)

        pos_pairs, neg_pairs = _sample_pairs(
            label_groups, pos_labels, all_labels, max_pairs, rng
        )

        if not pos_pairs:
            history["loss"].append(0.0)
            history["pos_sim"].append(0.0)
            history["neg_sim"].append(0.0)
            continue

        loss = torch.tensor(0.0, device=device)
        pos_sims: list[float] = []
        neg_sims: list[float] = []

        src_p = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=device)
        dst_p = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=device)
        sim_p = (emb[src_p] * emb[dst_p]).sum(dim=-1)
        loss = loss + (1.0 - sim_p).mean()
        pos_sims = sim_p.detach().cpu().tolist()

        if neg_pairs:
            src_n = torch.tensor([p[0] for p in neg_pairs], dtype=torch.long, device=device)
            dst_n = torch.tensor([p[1] for p in neg_pairs], dtype=torch.long, device=device)
            sim_n = (emb[src_n] * emb[dst_n]).sum(dim=-1)
            loss = loss + F.relu(sim_n - (1.0 - margin)).mean()
            neg_sims = sim_n.detach().cpu().tolist()

        loss.backward()
        opt.step()

        lv = float(loss.item())
        ps = float(np.mean(pos_sims))
        ns = float(np.mean(neg_sims)) if neg_sims else 0.0
        history["loss"].append(lv)
        history["pos_sim"].append(ps)
        history["neg_sim"].append(ns)

        if log_every > 0 and (epoch % log_every == 0 or epoch == 1):
            print(f"  epoch {epoch:4d}: loss={lv:.4f}  pos_sim={ps:.3f}  neg_sim={ns:.3f}")

    gnn.eval()
    return gnn, history


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------

def partition_half_synapses(
    gnn: Any,
    graph: HalfSynapseGraph,
    *,
    threshold: float = 0.8,
    same_seg_threshold: float | None = None,
    device: str = "cpu",
) -> np.ndarray:
    """Partition half-synapses into neuron clusters via GNN embeddings.

    Computes cosine similarity for all edges, then merges node pairs above the
    threshold via union-find.  The GNN's learned embeddings are the signal —
    after training, same-neuron pairs (even across segments) should have high
    similarity while different-neuron pairs should have low similarity.

    Two optional thresholds allow a more conservative strategy:
    - ``threshold`` applies to spatial edges (edge type 1).
    - ``same_seg_threshold`` applies to same-segment edges (edge type 0);
      defaults to ``threshold``.  Set higher to avoid merging frankenmerges.

    Parameters
    ----------
    gnn:
        Trained HalfSynapseGNN.
    graph:
        HalfSynapseGraph (same graph the GNN was trained on, or a new one
        with the same node feature dimension).
    threshold:
        Cosine similarity threshold for merging a spatial-edge pair.
    same_seg_threshold:
        Cosine similarity threshold for same-segment edges.  Defaults to
        ``threshold``.  Use a lower value to aggressively merge same-segment
        pairs, or a higher value to avoid frankenmerge errors.
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    ndarray [N] int64
        Cluster IDs (0-indexed consecutive integers).
    """
    import torch

    if same_seg_threshold is None:
        same_seg_threshold = threshold

    gnn.eval()
    with torch.no_grad():
        node_feat_t = torch.from_numpy(graph.node_feat).to(device)
        edge_src_t = torch.from_numpy(graph.edge_src).long().to(device)
        edge_dst_t = torch.from_numpy(graph.edge_dst).long().to(device)
        edge_type_t = torch.from_numpy(graph.edge_type).long().to(device)
        emb = gnn(node_feat_t, edge_src_t, edge_dst_t, edge_type_t)

    emb_np = emb.cpu().numpy()
    cos_sim = (emb_np[graph.edge_src] * emb_np[graph.edge_dst]).sum(axis=1)

    uf = _UnionFind(graph.n_nodes)

    for idx in range(len(graph.edge_src)):
        t = int(graph.edge_type[idx])
        thresh = same_seg_threshold if t == 0 else threshold
        if cos_sim[idx] >= thresh:
            uf.union(int(graph.edge_src[idx]), int(graph.edge_dst[idx]))

    return uf.labels()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_partition_ari(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    *,
    ignore_label: int = 0,
) -> dict:
    """Compute partition quality metrics.

    Parameters
    ----------
    pred_labels:
        Predicted cluster IDs ``[N]`` from ``partition_half_synapses``.
    true_labels:
        Ground-truth neuron IDs ``[N]`` (e.g. ``HalfSynapseGraph.labels``).
    ignore_label:
        True-label value to exclude (default 0 = unlabelled nodes).

    Returns
    -------
    dict with keys:
        ari, homogeneity, completeness, v_measure,
        n_clusters_pred, n_clusters_true, n_nodes
    """
    pred = np.asarray(pred_labels, dtype=np.int64)
    true = np.asarray(true_labels, dtype=np.int64)

    keep = true != ignore_label
    pred = pred[keep]
    true = true[keep]

    n = int(keep.sum())
    if n == 0:
        return {
            "ari": 0.0, "homogeneity": 0.0, "completeness": 0.0,
            "v_measure": 0.0, "n_clusters_pred": 0, "n_clusters_true": 0,
            "n_nodes": 0,
        }

    ari = _adjusted_rand_score_np(true, pred)
    h, c, v = _homogeneity_completeness_np(true, pred)

    return {
        "ari": ari,
        "homogeneity": h,
        "completeness": c,
        "v_measure": v,
        "n_clusters_pred": int(len(np.unique(pred))),
        "n_clusters_true": int(len(np.unique(true))),
        "n_nodes": n,
    }
