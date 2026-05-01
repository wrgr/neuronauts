"""Unit tests for :mod:`neuronauts.path_edge_encoder`."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuronauts.path_edge_encoder import (  # noqa: E402
    PATH_STEP_FEAT_DIM,
    PathEdgeEncoder,
    pad_path_sequences,
)


def test_pad_path_sequences_empty_paths():
    """Empty path list yields no real-path rows and full padding."""
    paths = [np.zeros((0, PATH_STEP_FEAT_DIM), dtype=np.float32)] * 3
    seq, mask, has = pad_path_sequences(paths, max_len=10)
    assert seq.shape == (3, 10, PATH_STEP_FEAT_DIM)
    assert mask.shape == (3, 10)
    assert mask.all()        # all padding
    assert not has.any()     # no real paths


def test_pad_path_sequences_mixed():
    """Mixed-length paths get padded and masked correctly."""
    paths = [
        np.random.randn(5, PATH_STEP_FEAT_DIM).astype(np.float32),
        np.zeros((0, PATH_STEP_FEAT_DIM), dtype=np.float32),
        np.random.randn(8, PATH_STEP_FEAT_DIM).astype(np.float32),
    ]
    seq, mask, has = pad_path_sequences(paths, max_len=6)

    assert seq.shape == (3, 6, PATH_STEP_FEAT_DIM)
    # First path has 5 real steps + 1 pad
    assert mask[0, :5].sum() == 0
    assert mask[0, 5]
    # Second path is all padding
    assert mask[1].all()
    # Third path is truncated to 6 steps, no padding
    assert mask[2].sum() == 0

    assert has.tolist() == [True, False, True]


def test_pad_path_sequences_truncation():
    """Long paths are truncated to max_len."""
    paths = [np.random.randn(100, PATH_STEP_FEAT_DIM).astype(np.float32)]
    seq, mask, has = pad_path_sequences(paths, max_len=16)
    assert seq.shape == (1, 16, PATH_STEP_FEAT_DIM)
    assert not mask.any()
    assert has.all()


def test_path_edge_encoder_basic_forward():
    """Encoder produces output_dim vector per edge with valid gradients."""
    encoder = PathEdgeEncoder(d_model=16, n_heads=2, n_layers=1, output_dim=8)

    # 4 edges: 2 with paths, 2 empty
    paths = [
        np.random.randn(5, PATH_STEP_FEAT_DIM).astype(np.float32),
        np.zeros((0, PATH_STEP_FEAT_DIM), dtype=np.float32),
        np.random.randn(3, PATH_STEP_FEAT_DIM).astype(np.float32),
        np.zeros((0, PATH_STEP_FEAT_DIM), dtype=np.float32),
    ]
    seq_np, mask_np, has_np = pad_path_sequences(paths, max_len=8)

    seq = torch.from_numpy(seq_np)
    mask = torch.from_numpy(mask_np)
    has = torch.from_numpy(has_np)

    out = encoder(seq, mask, has)
    assert out.shape == (4, 8)

    # Empty-path rows must equal the no_path_embedding parameter
    np_emb = encoder.no_path_embedding.detach()
    for i in (1, 3):
        assert torch.allclose(out[i].detach(), np_emb)

    # Real-path rows must NOT equal the no_path embedding
    for i in (0, 2):
        assert not torch.allclose(out[i].detach(), np_emb)

    # Backprop check
    loss = out.sum()
    loss.backward()
    assert encoder.input_proj.weight.grad is not None


def test_path_edge_encoder_no_path_embedding_is_learnable():
    """When all rows are empty, gradient flows back through no_path_embedding."""
    encoder = PathEdgeEncoder(d_model=8, n_heads=2, n_layers=1, output_dim=4)
    paths = [np.zeros((0, PATH_STEP_FEAT_DIM), dtype=np.float32)] * 5
    seq_np, mask_np, has_np = pad_path_sequences(paths, max_len=4)

    out = encoder(
        torch.from_numpy(seq_np),
        torch.from_numpy(mask_np),
        torch.from_numpy(has_np),
    )
    assert out.shape == (5, 4)
    out.sum().backward()
    assert encoder.no_path_embedding.grad is not None
    assert encoder.no_path_embedding.grad.abs().sum().item() > 0


def test_path_edge_encoder_deterministic_in_eval_mode():
    """In eval mode (dropout off), the same input produces the same output."""
    encoder = PathEdgeEncoder(d_model=16, n_heads=2, n_layers=1, output_dim=8)
    encoder.eval()

    paths = [np.random.randn(4, PATH_STEP_FEAT_DIM).astype(np.float32)]
    seq_np, mask_np, has_np = pad_path_sequences(paths, max_len=8)

    seq = torch.from_numpy(seq_np)
    mask = torch.from_numpy(mask_np)
    has = torch.from_numpy(has_np)

    with torch.no_grad():
        a = encoder(seq, mask, has)
        b = encoder(seq, mask, has)
    assert torch.allclose(a, b)
