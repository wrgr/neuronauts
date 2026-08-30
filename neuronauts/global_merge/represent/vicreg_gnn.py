"""
SkeletonGNN with VICReg (Variance-Invariance-Covariance Regularization).
Solves the within-type fine-fragment collapse problem on small/quarter skeletons.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SkeletonGNNBackbone(nn.Module):
    """Message-passing network over 3D skeleton tree topology."""
    def __init__(self, in_dim: int = 4, hidden_dim: int = 64, out_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "node_up": nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ),
                "self_up": nn.Linear(hidden_dim, hidden_dim)
            }))
            
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: shape (N, 4) -> [dx, dy, dz, radius] (centered at fragment centroid)
            edges: shape (E, 2) -> directed edge indices
        Returns:
            fragment_embedding: shape (out_dim,)
        """
        if nodes.size(0) == 0:
            return torch.zeros(self.readout[-1].out_features, device=nodes.device)
            
        h = self.in_proj(nodes)  # (N, hidden_dim)
        
        if edges.size(0) > 0 and nodes.size(0) > 1:
            src, dst = edges[:, 0], edges[:, 1]
            for layer in self.layers:
                # Aggregate neighbor messages
                msg = h[src]
                agg = torch.zeros_like(h)
                agg.index_add_(0, dst, msg)
                
                # Degree normalization
                deg = torch.zeros(nodes.size(0), 1, device=nodes.device)
                deg.index_add_(0, dst, torch.ones_like(src, dtype=torch.float).unsqueeze(-1))
                deg = torch.clamp(deg, min=1.0)
                agg = agg / deg
                
                # Update
                combined = torch.cat([h, agg], dim=-1)
                h = h + layer["node_up"](combined)
                
        # Global readout: Concat Mean + Max pooling
        h_mean = torch.mean(h, dim=0, keepdim=True)
        h_max, _ = torch.max(h, dim=0, keepdim=True)
        pooled = torch.cat([h_mean, h_max], dim=-1)
        out = self.readout(pooled).squeeze(0)
        return out


class VICRegProjector(nn.Module):
    """Expandable projection head for non-contrastive self-supervised learning."""
    def __init__(self, in_dim: int = 64, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VICRegLoss(nn.Module):
    """
    VICReg Loss: Invariance + Variance Regularization + Covariance Decorrelation.
    Prevents representation collapse on within-type fine fragments.
    """
    def __init__(self, sim_coeff: float = 25.0, std_coeff: float = 25.0, cov_coeff: float = 1.0, gamma: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.eps = eps

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            z_a, z_b: shape (BatchSize, ProjDim) embeddings of two augmented views / positive fragments
        """
        N, D = z_a.shape
        if N <= 1:
            # Fallback for single sample
            inv_loss = F.mse_loss(z_a, z_b)
            return inv_loss, {"loss": inv_loss.item(), "invariance": inv_loss.item(), "variance": 0.0, "covariance": 0.0}

        # 1. Invariance term (mean squared distance)
        sim_loss = F.mse_loss(z_a, z_b)

        # 2. Variance term (hinge on standard deviation)
        std_a = torch.sqrt(z_a.var(dim=0) + self.eps)
        std_b = torch.sqrt(z_b.var(dim=0) + self.eps)
        std_loss = torch.mean(F.relu(self.gamma - std_a)) + torch.mean(F.relu(self.gamma - std_b))

        # 3. Covariance term (penalize off-diagonal correlations)
        z_a_centered = z_a - z_a.mean(dim=0)
        z_b_centered = z_b - z_b.mean(dim=0)
        
        cov_a = (z_a_centered.T @ z_a_centered) / (N - 1)
        cov_b = (z_b_centered.T @ z_b_centered) / (N - 1)
        
        # Zero out diagonal
        cov_a_offdiag = cov_a.pow(2).sum() - cov_a.diagonal().pow(2).sum()
        cov_b_offdiag = cov_b.pow(2).sum() - cov_b.diagonal().pow(2).sum()
        cov_loss = (cov_a_offdiag + cov_b_offdiag) / (D * D)

        # Weighted total
        total_loss = (
            self.sim_coeff * sim_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        
        metrics = {
            "loss": float(total_loss.item()),
            "invariance": float(sim_loss.item()),
            "variance": float(std_loss.item()),
            "covariance": float(cov_loss.item()),
        }
        return total_loss, metrics


class VICRegSkeletonModel(nn.Module):
    """Full model combining backbone and projector."""
    def __init__(self, in_dim: int = 4, emb_dim: int = 64, proj_dim: int = 128):
        super().__init__()
        self.backbone = SkeletonGNNBackbone(in_dim=in_dim, hidden_dim=emb_dim, out_dim=emb_dim)
        self.projector = VICRegProjector(in_dim=emb_dim, proj_dim=proj_dim)

    def encode_fragment(self, vertices_nm: np.ndarray, radii_nm: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Encode single fragment to normalized DNA embedding vector."""
        if len(vertices_nm) == 0:
            return np.zeros(self.backbone.readout[-1].out_features, dtype=np.float32)
            
        centroid = np.mean(vertices_nm, axis=0)
        centered_v = (vertices_nm - centroid) / 1000.0  # normalize to microns
        norm_r = radii_nm / 1000.0
        
        node_feats = np.hstack([centered_v, norm_r[:, None]]).astype(np.float32)
        nodes_t = torch.from_numpy(node_feats)
        
        if len(edges) > 0:
            # Make bidirectional
            edges_bi = np.vstack([edges, edges[:, [1, 0]]])
            edges_t = torch.from_numpy(edges_bi.astype(np.int64))
        else:
            edges_t = torch.empty((0, 2), dtype=torch.int64)
            
        self.eval()
        with torch.no_grad():
            emb = self.backbone(nodes_t, edges_t)
            emb = F.normalize(emb, p=2, dim=-1)
            return emb.cpu().numpy()
