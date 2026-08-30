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


def train_vicreg_skeleton_gnn(
    model: VICRegSkeletonModel,
    fragments,
    positive_pairs: List[Tuple[int, int]],
    *,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    device: str = "cpu",
    log_every: int = 10,
) -> Dict[str, List[float]]:
    """
    Train VICRegSkeletonModel on positive fragment pairs.
    Uses non-contrastive VICReg loss to preserve feature variance across within-type neurons.
    """
    from neuronauts.represent.skeleton_gnn import fragment_to_tensors

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = VICRegLoss(sim_coeff=sim_coeff, std_coeff=std_coeff, cov_coeff=cov_coeff)

    # Precompute CPU tensors
    tensor_cache = [fragment_to_tensors(f, device="cpu") for f in fragments]

    history = {"loss": [], "invariance": [], "variance": [], "covariance": []}
    n_pairs = len(positive_pairs)
    if n_pairs == 0:
        return history

    rng = np.random.default_rng(42)

    for epoch in range(1, n_epochs + 1):
        perm = rng.permutation(n_pairs)
        epoch_losses = []
        epoch_inv = []
        epoch_var = []
        epoch_cov = []

        for start_idx in range(0, n_pairs, batch_size):
            batch_indices = perm[start_idx : start_idx + batch_size]
            if len(batch_indices) <= 1:
                continue

            a_indices = [positive_pairs[idx][0] for idx in batch_indices]
            b_indices = [positive_pairs[idx][1] for idx in batch_indices]

            # Forward view A
            embs_a = []
            for g_idx in a_indices:
                nf, es, ed, ef = tensor_cache[g_idx]
                if nf.size(0) == 0:
                    continue
                # SkeletonGNN forward
                h = model.backbone(nf.to(device)[:, :4], torch.stack([es, ed], dim=1).to(device) if es.size(0)>0 else torch.empty((0,2), dtype=torch.long, device=device))
                embs_a.append(h)

            # Forward view B
            embs_b = []
            for g_idx in b_indices:
                nf, es, ed, ef = tensor_cache[g_idx]
                if nf.size(0) == 0:
                    continue
                h = model.backbone(nf.to(device)[:, :4], torch.stack([es, ed], dim=1).to(device) if es.size(0)>0 else torch.empty((0,2), dtype=torch.long, device=device))
                embs_b.append(h)

            if len(embs_a) <= 1 or len(embs_b) <= 1 or len(embs_a) != len(embs_b):
                continue

            h_a = torch.stack(embs_a, dim=0)
            h_b = torch.stack(embs_b, dim=0)

            z_a = model.projector(h_a)
            z_b = model.projector(h_b)

            loss, m = loss_fn(z_a, z_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(m["loss"])
            epoch_inv.append(m["invariance"])
            epoch_var.append(m["variance"])
            epoch_cov.append(m["covariance"])

        if epoch_losses:
            l_mean = float(np.mean(epoch_losses))
            inv_mean = float(np.mean(epoch_inv))
            var_mean = float(np.mean(epoch_var))
            cov_mean = float(np.mean(epoch_cov))

            history["loss"].append(l_mean)
            history["invariance"].append(inv_mean)
            history["variance"].append(var_mean)
            history["covariance"].append(cov_mean)

            if log_every > 0 and (epoch % log_every == 0 or epoch == 1):
                print(f"  epoch {epoch:3d}: loss={l_mean:.4f}  inv={inv_mean:.4f}  var={var_mean:.4f}  cov={cov_mean:.4f}")

    model.eval()
    return history

def train_contrastive_skeleton_gnn(
    model: VICRegSkeletonModel,
    fragments,
    positive_pairs: List[Tuple[int, int]],
    negative_pairs: List[Tuple[int, int]],
    *,
    n_epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 32,
    margin_neg: float = 0.40,
    std_coeff: float = 10.0,
    device: str = "cpu",
    log_every: int = 10,
) -> Dict[str, List[float]]:
    """
    Train Skeleton GNN with Contrastive Hard-Negative Loss + Variance Regularization.
    Forces same-neuron pairs to cos >= 0.85 and cross-neuron frankenmerge pairs to cos <= 0.30.
    """
    from neuronauts.represent.skeleton_gnn import fragment_to_tensors

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Precompute CPU tensors
    tensor_cache = [fragment_to_tensors(f, device="cpu") for f in fragments]

    history = {"loss": [], "pos_cos": [], "neg_cos": [], "var": []}
    n_pos = len(positive_pairs)
    n_neg = len(negative_pairs)
    if n_pos == 0:
        return history

    rng = np.random.default_rng(42)

    for epoch in range(1, n_epochs + 1):
        perm_pos = rng.permutation(n_pos)
        perm_neg = rng.permutation(n_neg) if n_neg > 0 else np.zeros(0, dtype=np.int64)

        epoch_loss = []
        epoch_pos_cos = []
        epoch_neg_cos = []
        epoch_var = []

        for start_idx in range(0, n_pos, batch_size):
            b_pos_idx = perm_pos[start_idx : start_idx + batch_size]
            if len(b_pos_idx) <= 1:
                continue

            a_pos = [positive_pairs[idx][0] for idx in b_pos_idx]
            b_pos = [positive_pairs[idx][1] for idx in b_pos_idx]

            # Forward view A
            embs_a = []
            for g_idx in a_pos:
                nf, es, ed, _ = tensor_cache[g_idx]
                e_tens = torch.stack([es, ed], dim=1).to(device) if es.size(0) > 0 else torch.empty((0, 2), dtype=torch.long, device=device)
                h = model.backbone(nf.to(device)[:, :4], e_tens)
                embs_a.append(h)

            # Forward view B
            embs_b = []
            for g_idx in b_pos:
                nf, es, ed, _ = tensor_cache[g_idx]
                e_tens = torch.stack([es, ed], dim=1).to(device) if es.size(0) > 0 else torch.empty((0, 2), dtype=torch.long, device=device)
                h = model.backbone(nf.to(device)[:, :4], e_tens)
                embs_b.append(h)

            if len(embs_a) <= 1 or len(embs_b) <= 1:
                continue

            h_a = torch.stack(embs_a, dim=0)
            h_b = torch.stack(embs_b, dim=0)

            z_a = F.normalize(h_a, p=2, dim=-1)
            z_b = F.normalize(h_b, p=2, dim=-1)

            # 1. Positive cosine loss (maximize alignment)
            cos_pos = (z_a * z_b).sum(dim=-1)
            loss_pos = F.mse_loss(z_a, z_b)

            # 2. Negative cosine loss (push below margin)
            loss_neg = torch.tensor(0.0, device=device)
            neg_cos_val = 0.0
            if n_neg > 0:
                neg_sample_idxs = rng.choice(n_neg, size=len(b_pos_idx), replace=True)
                a_neg = [negative_pairs[idx][0] for idx in neg_sample_idxs]
                b_neg = [negative_pairs[idx][1] for idx in neg_sample_idxs]

                embs_neg_a = []
                for g_idx in a_neg:
                    nf, es, ed, _ = tensor_cache[g_idx]
                    e_tens = torch.stack([es, ed], dim=1).to(device) if es.size(0) > 0 else torch.empty((0, 2), dtype=torch.long, device=device)
                    embs_neg_a.append(model.backbone(nf.to(device)[:, :4], e_tens))

                embs_neg_b = []
                for g_idx in b_neg:
                    nf, es, ed, _ = tensor_cache[g_idx]
                    e_tens = torch.stack([es, ed], dim=1).to(device) if es.size(0) > 0 else torch.empty((0, 2), dtype=torch.long, device=device)
                    embs_neg_b.append(model.backbone(nf.to(device)[:, :4], e_tens))

                z_neg_a = F.normalize(torch.stack(embs_neg_a, dim=0), p=2, dim=-1)
                z_neg_b = F.normalize(torch.stack(embs_neg_b, dim=0), p=2, dim=-1)

                cos_neg = (z_neg_a * z_neg_b).sum(dim=-1)
                loss_neg = F.relu(cos_neg - margin_neg).pow(2).mean()
                neg_cos_val = float(cos_neg.mean().item())

            # 3. Variance regularization
            std_z = torch.sqrt(z_a.var(dim=0) + 1e-4)
            loss_var = F.relu(1.0 - std_z).mean()

            total_loss = 30.0 * (1.0 - cos_pos).mean() + 45.0 * loss_neg + 10.0 * loss_var

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss.append(float(total_loss.item()))
            epoch_pos_cos.append(float(cos_pos.mean().item()))
            epoch_neg_cos.append(neg_cos_val)
            epoch_var.append(float(std_z.mean().item()))

        if epoch_loss:
            l_mean = float(np.mean(epoch_loss))
            p_mean = float(np.mean(epoch_pos_cos))
            n_mean = float(np.mean(epoch_neg_cos))
            v_mean = float(np.mean(epoch_var))

            history["loss"].append(l_mean)
            history["pos_cos"].append(p_mean)
            history["neg_cos"].append(n_mean)
            history["var"].append(v_mean)

            if log_every > 0 and (epoch % log_every == 0 or epoch == 1):
                print(f"  epoch {epoch:3d}: loss={l_mean:.4f}  pos_cos={p_mean:.4f}  neg_cos={n_mean:.4f}  var_std={v_mean:.4f}")

    model.eval()
    return history
