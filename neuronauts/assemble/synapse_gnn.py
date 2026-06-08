"""Train CellGNN on a GlobalSynapseGraph and produce neuron cluster assignments.

Phase 2 pipeline:
    1. DNA encoder (Phase 1) → Fragment.dna per seg root.
    2. build_global_synapse_graph → GlobalSynapseGraph (k-NN + DNA node feats).
    3. train_global_gnn → refine per-synapse embeddings via message passing.
    4. assemble_neurons → partition_from_embeddings → integer cluster labels.

The GNN uses the existing CellGNN architecture with:
    node_input_dim = dna_dim  (e.g. 32)
    edge_input_dim = 1        (log-normalised distance)
    embedding_dim  = 32       (output embedding per synapse)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .global_synapse_graph import GlobalSynapseGraph


def train_global_gnn(
    graph: GlobalSynapseGraph,
    *,
    n_epochs: int = 50,
    lr: float = 1e-3,
    d_model: int = 64,
    embedding_dim: int = 32,
    n_layers: int = 3,
    n_heads: int = 4,
    dropout: float = 0.1,
    margin: float = 0.5,
    max_pairs: int = 1000,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 10,
) -> tuple[Any, dict]:
    """Train a CellGNN on the global synapse graph.

    Parameters
    ----------
    graph:
        GlobalSynapseGraph with node_feat (DNA) and pre_root_id set.
    n_epochs:
        Training epochs.
    lr:
        Adam learning rate.
    d_model:
        Internal GNN width.
    embedding_dim:
        Output embedding dimension (used for clustering at inference).
    n_layers:
        GNN message-passing layers.
    n_heads:
        Attention heads (unused in current CellGNN forward pass; kept for
        compat with CellGNN __new__ signature).
    dropout:
        Dropout rate.
    margin:
        Cosine similarity separation target for negative pairs.
    max_pairs:
        Pairs sampled per epoch for contrastive loss.
    device:
        "cpu" or "cuda".
    seed:
        Random seed for pair sampling.
    log_every:
        Print loss every N epochs (0 = silent).

    Returns
    -------
    (gnn, history)
        gnn — trained CellGNN module (eval mode).
        history — dict with "loss", "pos_sim", "neg_sim" lists (one per epoch).
    """
    import torch
    import torch.nn.functional as F

    from ..cell_graph import CellGNN

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    dna_dim = graph.dna_dim
    gnn = CellGNN(
        node_input_dim=dna_dim,
        edge_input_dim=1,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        embedding_dim=embedding_dim,
    ).to(device)

    opt = torch.optim.Adam(gnn.parameters(), lr=lr)

    node_feat_t = torch.from_numpy(graph.node_feat).to(device)
    edge_src_t = torch.from_numpy(graph.edge_src).long().to(device)
    edge_dst_t = torch.from_numpy(graph.edge_dst).long().to(device)
    edge_feat_t = torch.from_numpy(graph.edge_feat).to(device)

    root_ids = graph.pre_root_id
    root_groups: dict[int, list[int]] = {}
    for i, rid in enumerate(root_ids):
        rid_int = int(rid)
        if rid_int > 0:
            root_groups.setdefault(rid_int, []).append(i)

    pos_root_ids = [rid for rid, idxs in root_groups.items() if len(idxs) >= 2]
    all_root_ids = list(root_groups.keys())

    history: dict[str, list[float]] = {"loss": [], "pos_sim": [], "neg_sim": []}

    for epoch in range(1, n_epochs + 1):
        gnn.train()
        opt.zero_grad()

        emb = gnn(node_feat_t, edge_src_t, edge_dst_t, edge_feat_t)
        emb_norm = F.normalize(emb, p=2, dim=-1)

        pos_pairs, neg_pairs = _sample_pairs(
            root_groups, pos_root_ids, all_root_ids, max_pairs, rng
        )
        if not pos_pairs and not neg_pairs:
            history["loss"].append(0.0)
            history["pos_sim"].append(0.0)
            history["neg_sim"].append(0.0)
            continue

        loss = torch.tensor(0.0, device=device)
        pos_sims: list[float] = []
        neg_sims: list[float] = []

        if pos_pairs:
            src_p = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long, device=device)
            dst_p = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long, device=device)
            sim_p = (emb_norm[src_p] * emb_norm[dst_p]).sum(dim=-1)
            loss = loss + (1.0 - sim_p).mean()
            pos_sims = sim_p.detach().cpu().tolist()

        if neg_pairs:
            src_n = torch.tensor([p[0] for p in neg_pairs], dtype=torch.long, device=device)
            dst_n = torch.tensor([p[1] for p in neg_pairs], dtype=torch.long, device=device)
            sim_n = (emb_norm[src_n] * emb_norm[dst_n]).sum(dim=-1)
            loss = loss + F.relu(sim_n - (1.0 - margin)).mean()
            neg_sims = sim_n.detach().cpu().tolist()

        loss.backward()
        opt.step()

        loss_val = float(loss.item())
        ps = float(np.mean(pos_sims)) if pos_sims else 0.0
        ns = float(np.mean(neg_sims)) if neg_sims else 0.0
        history["loss"].append(loss_val)
        history["pos_sim"].append(ps)
        history["neg_sim"].append(ns)

        if log_every > 0 and epoch % log_every == 0:
            print(f"  epoch {epoch:3d}: loss={loss_val:.4f}  pos_sim={ps:.3f}  neg_sim={ns:.3f}")

    gnn.eval()
    return gnn, history


def run_global_gnn(
    graph: GlobalSynapseGraph,
    gnn: Any,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Run a trained CellGNN and return per-synapse embeddings.

    Parameters
    ----------
    graph:
        GlobalSynapseGraph (same one used for training, or a new one).
    gnn:
        Trained CellGNN module.
    device:
        Inference device.

    Returns
    -------
    ndarray [N, embedding_dim] float32 — L2-normalised.
    """
    import torch
    import torch.nn.functional as F

    gnn.eval()
    gnn = gnn.to(device)

    with torch.no_grad():
        node_feat_t = torch.from_numpy(graph.node_feat).to(device)
        edge_src_t = torch.from_numpy(graph.edge_src).long().to(device)
        edge_dst_t = torch.from_numpy(graph.edge_dst).long().to(device)
        edge_feat_t = torch.from_numpy(graph.edge_feat).to(device)

        emb = gnn(node_feat_t, edge_src_t, edge_dst_t, edge_feat_t)
        emb = F.normalize(emb, p=2, dim=-1)

    return emb.cpu().numpy().astype(np.float32)


def assemble_neurons(
    graph: GlobalSynapseGraph,
    gnn: Any,
    *,
    threshold: float = 0.5,
    method: str = "complete",
    device: str = "cpu",
) -> np.ndarray:
    """Run GNN inference and cluster synapse embeddings into neuron assignments.

    Parameters
    ----------
    graph:
        GlobalSynapseGraph.
    gnn:
        Trained CellGNN.
    threshold:
        Cosine similarity threshold for same-neuron assignment.
    method:
        Clustering method: "complete", "agglomerative", or "greedy".
    device:
        Inference device.

    Returns
    -------
    ndarray [N] int64 — integer neuron label per synapse (0-based contiguous).
    """
    from ..cell_graph import partition_from_embeddings

    emb = run_global_gnn(graph, gnn, device=device)
    return partition_from_embeddings(emb, threshold=threshold, method=method)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_pairs(
    root_groups: dict[int, list[int]],
    pos_root_ids: list[int],
    all_root_ids: list[int],
    max_pairs: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Sample balanced positive and negative synapse pairs for contrastive loss."""
    n_each = max_pairs // 2

    pos_pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(pos_pairs) < n_each and attempts < n_each * 4 and pos_root_ids:
        attempts += 1
        rid = pos_root_ids[int(rng.integers(len(pos_root_ids)))]
        idxs = root_groups[rid]
        ia, ib = rng.choice(len(idxs), size=2, replace=False)
        pos_pairs.append((idxs[int(ia)], idxs[int(ib)]))

    neg_pairs: list[tuple[int, int]] = []
    attempts = 0
    while len(neg_pairs) < n_each and attempts < n_each * 4 and len(all_root_ids) >= 2:
        attempts += 1
        ra, rb = rng.choice(len(all_root_ids), size=2, replace=False)
        rid_a = all_root_ids[int(ra)]
        rid_b = all_root_ids[int(rb)]
        if rid_a == rid_b:
            continue
        ia = int(rng.integers(len(root_groups[rid_a])))
        ib = int(rng.integers(len(root_groups[rid_b])))
        neg_pairs.append((root_groups[rid_a][ia], root_groups[rid_b][ib]))

    return pos_pairs, neg_pairs
