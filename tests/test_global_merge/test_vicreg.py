"""
Tests for SkeletonGNN with VICReg Loss.
"""

import pytest
import torch
import numpy as np
from neuronauts.global_merge.represent.vicreg_gnn import (
    SkeletonGNNBackbone,
    VICRegProjector,
    VICRegLoss,
    VICRegSkeletonModel
)


def test_skeleton_gnn_forward():
    model = SkeletonGNNBackbone(in_dim=4, hidden_dim=32, out_dim=32, num_layers=2)
    # 5 nodes with (dx, dy, dz, r)
    nodes = torch.randn(5, 4)
    # Simple line graph edges: 0-1-2-3-4
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4],
                          [1, 0], [2, 1], [3, 2], [4, 3]], dtype=torch.long)
    
    emb = model(nodes, edges)
    assert emb.shape == (32,)
    assert not torch.isnan(emb).any()


def test_vicreg_loss_prevents_collapse():
    loss_fn = VICRegLoss(sim_coeff=10.0, std_coeff=10.0, cov_coeff=1.0, gamma=1.0)
    
    # 1. Batch of non-collapsed representations
    torch.manual_seed(42)
    z_a = torch.randn(16, 64)
    z_b = z_a + 0.1 * torch.randn(16, 64)
    
    loss, metrics = loss_fn(z_a, z_b)
    assert loss.item() > 0.0
    assert metrics["variance"] >= 0.0
    assert metrics["covariance"] >= 0.0

    # 2. Batch of completely collapsed representations (all identical)
    z_collapsed = torch.ones(16, 64)
    loss_col, metrics_col = loss_fn(z_collapsed, z_collapsed)
    
    # Invariance is 0, but variance penalty must be maximal (gamma = 1.0)
    assert metrics_col["invariance"] < 1e-5
    assert metrics_col["variance"] >= 1.0
    assert loss_col.item() > 0.0  # Loss must heavily penalize collapse!


def test_vicreg_skeleton_model_encode():
    model = VICRegSkeletonModel(in_dim=4, emb_dim=32, proj_dim=64)
    
    verts = np.array([
        [0.0, 0.0, 0.0],
        [100.0, 0.0, 0.0],
        [200.0, 0.0, 0.0],
    ], dtype=np.float32)
    radii = np.array([50.0, 50.0, 50.0], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    
    emb = model.encode_fragment(verts, radii, edges)
    assert emb.shape == (32,)
    # Should be unit normalized
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)
