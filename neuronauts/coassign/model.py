"""Synapse co-assignment model.

Requires PyTorch.

Architecture
------------
SynapseCoassigner = GNN encoder + edge-scoring MLP.

Encoder
  Input: [pos (3) || dna (D)] for each synapse node.
  LayerNorm handles position/DNA normalisation — no hardcoded scales.
  L message-passing layers. Each layer:
    message  = Linear([h_src || same_seg])   ← same_seg is learned evidence, not hardcoded weight
    agg      = scatter_add(messages, dst)
    h_new    = LayerNorm(h + Linear([h || agg]))

Scorer
  Input per edge: [h_u || h_v || |h_u − h_v| || same_seg]
  MLP → scalar logit → sigmoid → P(same neuron)

The model learns from data:
  - how much to trust same-seg vs spatial edges
  - how to normalise position and DNA scales
  - which differences in embedding space signal a neuron boundary
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MPLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.msg = nn.Linear(d_model + 1, d_model)   # +1 for same_seg
        self.update = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,          # [N, d]
        edge_src: torch.Tensor,   # [E]
        edge_dst: torch.Tensor,   # [E]
        same_seg: torch.Tensor,   # [E]
    ) -> torch.Tensor:
        msgs = self.msg(torch.cat([h[edge_src], same_seg.unsqueeze(1)], dim=1))
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, edge_dst.unsqueeze(1).expand(-1, h.size(1)), msgs)
        return self.norm(h + self.update(torch.cat([h, agg], dim=1)))


class SynapseCoassigner(nn.Module):
    """GNN encoder + edge scorer for synapse co-assignment.

    Parameters
    ----------
    node_dim:
        Input feature dimension: 3 (position) + dna_dim.
    d_model:
        Hidden dimension throughout the network.
    n_layers:
        Number of message-passing layers.
    dropout:
        Dropout rate in update MLPs.
    """

    def __init__(
        self,
        node_dim: int,
        d_model: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(node_dim)
        self.input_proj = nn.Linear(node_dim, d_model)
        self.layers = nn.ModuleList([_MPLayer(d_model, dropout) for _ in range(n_layers)])
        # Scorer sees: h_u, h_v, |h_u - h_v|, same_seg scalar
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 3 + 1, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def encode(
        self,
        node_feat: torch.Tensor,   # [N, node_dim]
        edge_src: torch.Tensor,    # [E]
        edge_dst: torch.Tensor,    # [E]
        same_seg: torch.Tensor,    # [E]
    ) -> torch.Tensor:             # [N, d_model]
        h = F.relu(self.input_proj(self.input_norm(node_feat)))
        for layer in self.layers:
            h = layer(h, edge_src, edge_dst, same_seg)
        return h

    def score_edges(
        self,
        h: torch.Tensor,           # [N, d_model]
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        same_seg: torch.Tensor,
    ) -> torch.Tensor:             # [E] logits
        hu, hv = h[edge_src], h[edge_dst]
        edge_feat = torch.cat([hu, hv, (hu - hv).abs(), same_seg.unsqueeze(1)], dim=1)
        return self.scorer(edge_feat).squeeze(1)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        same_seg: torch.Tensor,
    ) -> torch.Tensor:             # [E] logits
        h = self.encode(node_feat, edge_src, edge_dst, same_seg)
        return self.score_edges(h, edge_src, edge_dst, same_seg)

    @torch.no_grad()
    def edge_probs(
        self,
        node_feat: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        same_seg: torch.Tensor,
    ) -> torch.Tensor:             # [E] probabilities in [0, 1]
        self.eval()
        return torch.sigmoid(self(node_feat, edge_src, edge_dst, same_seg))
