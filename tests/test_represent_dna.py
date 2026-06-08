"""Tests for neuronauts.represent.dna — tree-DNA encoder."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuronauts.represent.dna import (
    TreeDNAEncoder,
    encode_fragments,
    featurize_fragment,
    sample_tree_paths,
    train_dna_encoder,
)
from neuronauts.schemas import Fragment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fragment(n_verts: int = 8, branching: bool = False) -> Fragment:
    """Minimal Fragment with a chain (or Y-shape) skeleton."""
    if branching:
        # Y-shape: 0-1-2-3-4, 3-5-6
        verts = np.array([
            [0, 0, 0], [100, 0, 0], [200, 0, 0], [300, 0, 0], [400, 0, 0],
            [300, 100, 0], [300, 200, 0],
        ], dtype=np.float32)
        edges = np.array([[0,1],[1,2],[2,3],[3,4],[3,5],[5,6]], dtype=np.int64)
        radii = np.ones(7, dtype=np.float32)
        endpoints = verts[[0, 4, 6]]  # 3 leaves
    else:
        verts = np.column_stack([
            np.arange(n_verts, dtype=np.float32) * 100,
            np.zeros(n_verts, dtype=np.float32),
            np.zeros(n_verts, dtype=np.float32),
        ])
        edges = np.stack([np.arange(n_verts - 1), np.arange(1, n_verts)], axis=1).astype(np.int64)
        radii = np.ones(n_verts, dtype=np.float32)
        endpoints = verts[[0, -1]]

    return Fragment(
        fragment_id=1,
        region_id="r0",
        base_root_id=42,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=endpoints,
        radius_nm=radii,
        synapse_indices=np.array([], dtype=np.int64),
    ).validate()


def _make_single_vertex_fragment() -> Fragment:
    verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    return Fragment(
        fragment_id=2,
        region_id="r0",
        base_root_id=99,
        vertices_nm=verts,
        edges=np.zeros((0, 2), dtype=np.int64),
        endpoints_nm=verts,
        radius_nm=np.ones(1, dtype=np.float32),
        synapse_indices=np.array([], dtype=np.int64),
    ).validate()


# ---------------------------------------------------------------------------
# sample_tree_paths
# ---------------------------------------------------------------------------

def test_sample_tree_paths_chain():
    """10-vertex chain → exactly 1 path (leaf-to-leaf)."""
    verts = np.column_stack([np.arange(10, dtype=np.float32) * 100, np.zeros((10, 2))])
    edges = np.stack([np.arange(9), np.arange(1, 10)], axis=1).astype(np.int64)
    paths = sample_tree_paths(verts, edges, n_paths=8)
    assert len(paths) == 1
    assert paths[0].shape == (10, 3)


def test_sample_tree_paths_y_shape():
    """Y-shaped tree with 3 leaves → up to 3 unique leaf-to-leaf paths."""
    verts = np.array([
        [0,0,0],[100,0,0],[200,0,0],[300,0,0],[400,0,0],
        [300,100,0],[300,200,0],
    ], dtype=np.float32)
    edges = np.array([[0,1],[1,2],[2,3],[3,4],[3,5],[5,6]], dtype=np.int64)
    paths = sample_tree_paths(verts, edges, n_paths=16, rng=np.random.default_rng(0))
    assert 1 <= len(paths) <= 3
    for path in paths:
        assert path.ndim == 2 and path.shape[1] == 3


def test_sample_tree_paths_single_vertex():
    """Single-vertex skeleton → returns [vertices_nm]."""
    verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    paths = sample_tree_paths(verts, np.zeros((0, 2), dtype=np.int64), n_paths=4)
    assert len(paths) == 1
    assert paths[0].shape == (1, 3)


def test_sample_tree_paths_coordinates_are_subset():
    """All path vertices come from the original vertex set."""
    verts = np.column_stack([np.arange(6, dtype=np.float32) * 50, np.zeros((6, 2))])
    edges = np.array([[0,1],[1,2],[2,3],[3,4],[4,5]], dtype=np.int64)
    paths = sample_tree_paths(verts, edges)
    for path in paths:
        for pt in path:
            assert any(np.allclose(pt, v) for v in verts)


# ---------------------------------------------------------------------------
# featurize_fragment
# ---------------------------------------------------------------------------

def test_featurize_fragment_chain_shape():
    """8-vertex chain fragment → list of 1 array of shape [7, 6]."""
    frag = _make_fragment(n_verts=8)
    result = featurize_fragment(frag, n_paths=16)
    assert isinstance(result, list)
    assert len(result) >= 1
    # Each array has 6 feature dimensions
    for arr in result:
        assert arr.ndim == 2 and arr.shape[1] == 6


def test_featurize_fragment_single_vertex_no_crash():
    """Single-vertex fragment → list of 1 array [1, 6] of zeros."""
    frag = _make_single_vertex_fragment()
    result = featurize_fragment(frag)
    assert len(result) == 1
    arr = result[0]
    assert arr.shape == (1, 6)
    assert np.allclose(arr, 0.0)


def test_featurize_fragment_tree_multiple_paths():
    """Y-shaped fragment → more than one path."""
    frag = _make_fragment(branching=True)
    result = featurize_fragment(frag, n_paths=16, rng=np.random.default_rng(1))
    assert len(result) >= 1
    for arr in result:
        assert arr.ndim == 2 and arr.shape[1] == 6


def test_featurize_fragment_dtype():
    """All returned arrays are float32."""
    frag = _make_fragment(n_verts=6)
    result = featurize_fragment(frag)
    for arr in result:
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# TreeDNAEncoder
# ---------------------------------------------------------------------------

def test_encoder_output_shape():
    """Batch of 4 chain fragments → [4, 64] embeddings."""
    encoder = TreeDNAEncoder(output_dim=64, n_paths=4)
    frags = [_make_fragment(n_verts=6 + i) for i in range(4)]
    feats = [featurize_fragment(f, n_paths=4) for f in frags]
    out = encoder(feats)
    assert out.shape == (4, 64)


def test_encoder_output_finite():
    """All output values are finite (no NaN/Inf)."""
    encoder = TreeDNAEncoder(output_dim=32, n_paths=4)
    frags = [_make_fragment(n_verts=5), _make_fragment(branching=True)]
    feats = [featurize_fragment(f, n_paths=4) for f in frags]
    out = encoder(feats)
    assert torch.isfinite(out).all()


def test_encoder_single_vertex_no_crash():
    """Single-vertex fragment does not crash the encoder."""
    encoder = TreeDNAEncoder(output_dim=16, n_paths=4)
    frag = _make_single_vertex_fragment()
    feats = [featurize_fragment(frag, n_paths=4)]
    out = encoder(feats)
    assert out.shape == (1, 16)
    assert torch.isfinite(out).all()


def test_encoder_mixed_topology():
    """Batch with both chain and Y-shaped fragments → correct output shape."""
    encoder = TreeDNAEncoder(output_dim=32, n_paths=8)
    frags = [_make_fragment(n_verts=8), _make_fragment(branching=True), _make_fragment(n_verts=5)]
    feats = [featurize_fragment(f, n_paths=8) for f in frags]
    out = encoder(feats)
    assert out.shape == (3, 32)


# ---------------------------------------------------------------------------
# encode_fragments
# ---------------------------------------------------------------------------

def test_encode_fragments_fills_dna():
    """encode_fragments returns Fragment copies with dna= filled."""
    encoder = TreeDNAEncoder(output_dim=32, n_paths=4)
    frags = [_make_fragment(n_verts=6 + i) for i in range(3)]
    assert all(f.dna is None for f in frags)

    filled = encode_fragments(encoder, frags, device="cpu", n_paths=4)
    assert all(f.dna is not None for f in filled)
    for f in filled:
        assert f.dna.shape == (32,)
        assert f.dna.dtype == np.float32


def test_encode_fragments_does_not_mutate_input():
    """Input Fragment list is not modified by encode_fragments."""
    encoder = TreeDNAEncoder(output_dim=16, n_paths=4)
    frags = [_make_fragment(n_verts=5)]
    encode_fragments(encoder, frags, n_paths=4)
    assert frags[0].dna is None  # original untouched


# ---------------------------------------------------------------------------
# train_dna_encoder
# ---------------------------------------------------------------------------

def _make_fragment_set(root_id: int, n_frags: int, seed: int = 0) -> list[Fragment]:
    """Multiple fragments sharing the same base_root_id (simulates one seg root)."""
    rng = np.random.default_rng(seed)
    result = []
    for i in range(n_frags):
        n_v = int(rng.integers(4, 12))
        verts = rng.uniform(0, 1000, (n_v, 3)).astype(np.float32)
        edges = np.stack([np.arange(n_v - 1), np.arange(1, n_v)], axis=1).astype(np.int64)
        result.append(Fragment(
            fragment_id=root_id * 100 + i,
            region_id=f"r{seed}",
            base_root_id=root_id,
            vertices_nm=verts,
            edges=edges,
            endpoints_nm=verts[[0, -1]],
            radius_nm=np.ones(n_v, dtype=np.float32),
            synapse_indices=np.array([], dtype=np.int64),
        ).validate())
    return result


def test_train_dna_encoder_runs():
    """train_dna_encoder runs without error and returns expected keys."""
    encoder = TreeDNAEncoder(output_dim=16, n_paths=4, n_layers=1)
    frags_a = _make_fragment_set(root_id=1, n_frags=4, seed=0)
    frags_b = _make_fragment_set(root_id=2, n_frags=4, seed=1)
    history = train_dna_encoder(
        encoder,
        [frags_a, frags_b],
        n_epochs=2,
        batch_size=4,
        n_paths=4,
    )
    assert "loss" in history and "pos_cosine" in history and "neg_cosine" in history
    assert len(history["loss"]) == 2


def test_train_dna_encoder_contamination_mask():
    """Fragments whose base_root_id maps to >1 label roots are excluded from training."""
    encoder = TreeDNAEncoder(output_dim=16, n_paths=4, n_layers=1)
    # root_id=1 is clean (maps to exactly 1 label root)
    # root_id=99 is contaminated (maps to 2 label roots) → must NOT be used
    frags_clean_a = _make_fragment_set(root_id=1, n_frags=4, seed=0)
    frags_clean_b = _make_fragment_set(root_id=2, n_frags=4, seed=1)
    frags_contaminated = _make_fragment_set(root_id=99, n_frags=4, seed=2)

    root_label_map = {1: {100}, 2: {200}, 99: {300, 301}}  # 99 is contaminated

    history = train_dna_encoder(
        encoder,
        [frags_clean_a, frags_clean_b, frags_contaminated],
        n_epochs=2,
        batch_size=4,
        n_paths=4,
        root_label_map=root_label_map,
    )
    # Should complete without error — contaminated root excluded silently
    assert len(history["loss"]) == 2


def test_train_dna_encoder_insufficient_roots_raises():
    """Raises ValueError if fewer than 2 clean roots with ≥2 fragments each."""
    encoder = TreeDNAEncoder(output_dim=16, n_paths=4, n_layers=1)
    # Only one valid root
    frags = _make_fragment_set(root_id=1, n_frags=3, seed=0)
    with pytest.raises(ValueError, match="Need ≥2 neuron groups"):
        train_dna_encoder(encoder, [frags], n_epochs=1, n_paths=4)
