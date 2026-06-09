"""Training loop for SynapseCoassigner.

Loss: calibrated binary cross-entropy on edge labels.
  y = 1  if edge connects two synapses from the same neuron
  y = 0  otherwise

Hard negative mining: spatial edges connecting different-neuron synapses
are systematically included. These are the cases the model most needs to
learn — spatially close synapses from different, interdigitated neurons.

Edges involving unknown-label synapses (label = 0) are masked out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .graph import SynapseGraph


def train(
    model: Any,                       # SynapseCoassigner
    graphs: list[SynapseGraph],
    *,
    n_epochs: int = 60,
    lr: float = 1e-3,
    max_edges_per_graph: int = 2000,
    hard_neg_frac: float = 0.5,       # fraction of negatives from spatial cross-neuron edges
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 10,
) -> dict[str, list[float]]:
    """Train the co-assigner with balanced BCE loss.

    Parameters
    ----------
    model:
        SynapseCoassigner instance.
    graphs:
        Training graphs with ground-truth labels.
    n_epochs:
        Training epochs.
    lr:
        Adam learning rate.
    max_edges_per_graph:
        Positive edges sampled per graph per epoch (equal negatives sampled).
    hard_neg_frac:
        Fraction of negative edges sampled from spatial edges that cross
        neuron boundaries. The rest are random negatives.
    device:
        ``"cpu"`` or ``"cuda"``.
    seed:
        RNG seed.
    log_every:
        Print stats every N epochs (0 = silent).

    Returns
    -------
    history dict with keys ``loss``, ``precision``, ``recall`` (per epoch,
    averaged over graphs).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"loss": [], "precision": [], "recall": []}

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = epoch_prec = epoch_rec = 0.0
        n_graphs_used = 0

        for graph in graphs:
            node_feat = torch.from_numpy(
                np.concatenate([graph.node_pos, graph.node_dna], axis=1)
            ).float().to(device)
            edge_src_t  = torch.from_numpy(graph.edge_src).long().to(device)
            edge_dst_t  = torch.from_numpy(graph.edge_dst).long().to(device)
            same_seg_t  = torch.from_numpy(graph.same_seg).float().to(device)

            opt.zero_grad()
            logits = model(node_feat, edge_src_t, edge_dst_t, same_seg_t)

            # Ground-truth edge labels (float for BCE)
            y_all = (
                graph.labels[graph.edge_src] == graph.labels[graph.edge_dst]
            ).astype(np.float32)

            # Mask: only edges where both endpoints have a known label
            known = (
                (graph.labels[graph.edge_src] != 0) &
                (graph.labels[graph.edge_dst] != 0)
            )

            pos_idx  = np.where(known & (y_all == 1))[0]
            # Hard negatives: same_seg=0 edges that cross neuron boundaries
            hard_idx = np.where(known & (y_all == 0) & (graph.same_seg == 0))[0]
            rand_idx = np.where(known & (y_all == 0))[0]

            n_pos = min(len(pos_idx), max_edges_per_graph // 2)
            if n_pos == 0 or len(rand_idx) == 0:
                continue

            n_neg   = n_pos
            n_hard  = int(n_neg * hard_neg_frac) if len(hard_idx) > 0 else 0
            n_rand  = n_neg - n_hard

            sel_pos = rng.choice(pos_idx, n_pos, replace=len(pos_idx) < n_pos)
            sel_neg_parts = []
            if n_hard > 0:
                sel_neg_parts.append(
                    rng.choice(hard_idx, n_hard, replace=len(hard_idx) < n_hard)
                )
            if n_rand > 0:
                sel_neg_parts.append(
                    rng.choice(rand_idx, n_rand, replace=len(rand_idx) < n_rand)
                )

            sel = np.concatenate([sel_pos] + sel_neg_parts)
            sel_t = torch.from_numpy(sel).long().to(device)
            y_t   = torch.from_numpy(y_all[sel]).to(device)

            loss = F.binary_cross_entropy_with_logits(logits[sel_t], y_t)
            loss.backward()
            opt.step()

            # Quick per-batch precision / recall
            with torch.no_grad():
                pred = (torch.sigmoid(logits[sel_t]) > 0.5).float().cpu().numpy()
                y_np = y_all[sel]
                tp = float(((pred == 1) & (y_np == 1)).sum())
                fp = float(((pred == 1) & (y_np == 0)).sum())
                fn = float(((pred == 0) & (y_np == 1)).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            epoch_loss += float(loss.item())
            epoch_prec += prec
            epoch_rec  += rec
            n_graphs_used += 1

        if n_graphs_used == 0:
            history["loss"].append(0.0)
            history["precision"].append(0.0)
            history["recall"].append(0.0)
            continue

        history["loss"].append(epoch_loss / n_graphs_used)
        history["precision"].append(epoch_prec / n_graphs_used)
        history["recall"].append(epoch_rec / n_graphs_used)

        if log_every > 0 and (epoch % log_every == 0 or epoch == 1):
            print(
                f"  epoch {epoch:4d}: "
                f"loss={history['loss'][-1]:.4f}  "
                f"prec={history['precision'][-1]:.3f}  "
                f"rec={history['recall'][-1]:.3f}"
            )

    model.eval()
    return history
