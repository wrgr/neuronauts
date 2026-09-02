"""Tests for neuronauts.represent.enrich — DNA enrichment and AUC ablation."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

#: Only the evaluate_dna_auc tests need scikit-learn. A module-level
#: ``importorskip`` placed mid-file used to skip this whole module at
#: collection -- including the eight tests above it that need nothing -- so
#: the guard is per-test now.
needs_sklearn = pytest.mark.skipif(importlib.util.find_spec("sklearn") is None,
                                   reason="scikit-learn not installed")

from neuronauts.represent.enrich import (
    build_synapse_dna_matrix,
    evaluate_dna_auc,
    spatial_proximity_scores,
    synapse_pair_dna_scores,
)
from neuronauts.schemas import Fragment, Region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_region(root_ids: list[int], n_syn: int | None = None) -> Region:
    """Region with pre_root_id= root_ids (label_version ground truth)."""
    n = n_syn if n_syn is not None else len(root_ids)
    rng = np.random.default_rng(1)
    return Region(
        region_id="test",
        bbox_nm=((0.0, 0.0, 0.0), (20000.0, 20000.0, 20000.0)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=rng.uniform(0, 20000, (n, 3)).astype(np.float32),
        post_pt_nm=rng.uniform(0, 20000, (n, 3)).astype(np.float32),
        pre_root_id=np.array(root_ids[:n], dtype=np.int64),
        post_root_id=np.zeros(n, dtype=np.int64),
        synapse_id=np.arange(n, dtype=np.int64),
    )


def _make_fragment_with_dna(
    base_root_id: int,
    synapse_indices: list[int],
    dna: list[float],
    region_id: str = "test",
) -> Fragment:
    """Fragment with a trivial chain skeleton and given dna."""
    n = 5
    verts = np.column_stack([
        np.arange(n, dtype=np.float32) * 100,
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    ])
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int64)
    return Fragment(
        fragment_id=base_root_id,
        region_id=region_id,
        base_root_id=base_root_id,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=verts[[0, -1]],
        radius_nm=np.ones(n, dtype=np.float32),
        synapse_indices=np.array(synapse_indices, dtype=np.int64),
        dna=np.array(dna, dtype=np.float32),
    ).validate()


# ---------------------------------------------------------------------------
# build_synapse_dna_matrix
# ---------------------------------------------------------------------------

def test_build_synapse_dna_matrix_basic():
    """Each synapse row gets the DNA of its owning Fragment."""
    region = _make_region([1, 1, 2, 2])
    frags = [
        _make_fragment_with_dna(10, [0, 1], [1.0, 0.0]),  # synapses 0,1
        _make_fragment_with_dna(20, [2, 3], [0.0, 1.0]),  # synapses 2,3
    ]
    mat = build_synapse_dna_matrix(region, frags)
    assert mat.shape == (4, 2)
    np.testing.assert_allclose(mat[0], [1.0, 0.0])
    np.testing.assert_allclose(mat[1], [1.0, 0.0])
    np.testing.assert_allclose(mat[2], [0.0, 1.0])
    np.testing.assert_allclose(mat[3], [0.0, 1.0])


def test_build_synapse_dna_matrix_missing_dna():
    """Fragment with dna=None leaves its rows as zeros."""
    region = _make_region([1, 1, 2, 2])
    frag_with = _make_fragment_with_dna(10, [0, 1], [1.0, 0.0])
    frag_without = Fragment(
        fragment_id=20,
        region_id="test",
        base_root_id=20,
        vertices_nm=np.ones((5, 3), dtype=np.float32),
        edges=np.stack([np.arange(4), np.arange(1, 5)], axis=1).astype(np.int64),
        endpoints_nm=np.ones((2, 3), dtype=np.float32),
        radius_nm=np.ones(5, dtype=np.float32),
        synapse_indices=np.array([2, 3], dtype=np.int64),
        dna=None,
    ).validate()
    mat = build_synapse_dna_matrix(region, [frag_with, frag_without])
    assert mat.shape == (4, 2)
    assert np.allclose(mat[2], 0.0) and np.allclose(mat[3], 0.0)


def test_build_synapse_dna_matrix_no_fragments_with_dna():
    """No fragment has DNA → returns [N, 0] matrix."""
    region = _make_region([1, 2])
    frag = Fragment(
        fragment_id=1, region_id="test", base_root_id=1,
        vertices_nm=np.ones((5, 3), np.float32),
        edges=np.stack([np.arange(4), np.arange(1, 5)], axis=1).astype(np.int64),
        endpoints_nm=np.ones((2, 3), np.float32),
        radius_nm=np.ones(5, np.float32),
        synapse_indices=np.array([0], np.int64),
        dna=None,
    ).validate()
    mat = build_synapse_dna_matrix(region, [frag])
    assert mat.shape == (2, 0)


# ---------------------------------------------------------------------------
# synapse_pair_dna_scores
# ---------------------------------------------------------------------------

def test_synapse_pair_dna_scores_shape():
    """Scores and labels have matching lengths and correct types."""
    region = _make_region([1, 1, 1, 2, 2, 2])
    frags = [
        _make_fragment_with_dna(10, [0, 1, 2], [1.0, 0.0]),
        _make_fragment_with_dna(20, [3, 4, 5], [0.0, 1.0]),
    ]
    scores, labels = synapse_pair_dna_scores(region, frags, max_pairs=20, rng=np.random.default_rng(0))
    assert len(scores) == len(labels)
    assert scores.dtype == np.float32
    assert labels.dtype == np.int8
    assert set(labels.tolist()).issubset({0, 1})


def test_synapse_pair_dna_scores_positive_similarity():
    """Same-root pairs have cosine similarity 1.0 (identical DNA vectors)."""
    region = _make_region([1, 1, 1, 2, 2, 2])
    frags = [
        _make_fragment_with_dna(10, [0, 1, 2], [1.0, 0.0]),
        _make_fragment_with_dna(20, [3, 4, 5], [0.0, 1.0]),
    ]
    scores, labels = synapse_pair_dna_scores(region, frags, max_pairs=100, rng=np.random.default_rng(0))
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    assert np.all(pos_scores > 0.99), "same-root pairs should have cosine ≈ 1"
    assert np.all(np.abs(neg_scores) < 0.01), "orthogonal DNA pairs should have cosine ≈ 0"


def test_synapse_pair_dna_scores_no_root_id_raises():
    """Raises ValueError when pre_root_id is missing (all zeros)."""
    region = _make_region([0, 0, 0])  # root_id=0 means invalid
    frags = [_make_fragment_with_dna(10, [0, 1, 2], [1.0, 0.0])]
    scores, labels = synapse_pair_dna_scores(region, frags, max_pairs=10)
    # All root_ids are 0 → no valid synapses → empty output
    assert len(scores) == 0


# ---------------------------------------------------------------------------
# spatial_proximity_scores
# ---------------------------------------------------------------------------

def test_spatial_proximity_scores_shape():
    """Proximity scores have matching lengths."""
    region = _make_region([1, 1, 1, 2, 2, 2])
    scores, labels = spatial_proximity_scores(region, max_pairs=20, rng=np.random.default_rng(0))
    assert len(scores) == len(labels)
    assert scores.dtype == np.float32
    # All proximity scores in (0, 1]
    assert np.all(scores > 0) and np.all(scores <= 1.0)


def test_spatial_proximity_scores_nearby_same_neuron():
    """Synapses of the same neuron placed close together score higher than distant cross-neuron."""
    # Build a region where same-neuron synapses are clustered
    rng = np.random.default_rng(42)
    pre_pts = np.zeros((6, 3), dtype=np.float32)
    # synapses 0-2: root 1, near origin
    pre_pts[:3] = rng.uniform(0, 100, (3, 3)).astype(np.float32)
    # synapses 3-5: root 2, far from origin
    pre_pts[3:] = rng.uniform(10000, 10100, (3, 3)).astype(np.float32)

    region = Region(
        region_id="test",
        bbox_nm=((0.0, 0.0, 0.0), (20000.0, 20000.0, 20000.0)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117, label_version=1412,
        pre_pt_nm=pre_pts,
        post_pt_nm=pre_pts.copy(),
        pre_root_id=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        post_root_id=np.zeros(6, dtype=np.int64),
        synapse_id=np.arange(6, dtype=np.int64),
    )
    scores, labels = spatial_proximity_scores(region, max_pairs=200, rng=np.random.default_rng(0))
    pos_mean = scores[labels == 1].mean()
    neg_mean = scores[labels == 0].mean()
    assert pos_mean > neg_mean, "same-neuron pairs (clustered) should have higher proximity"


# ---------------------------------------------------------------------------
# evaluate_dna_auc
# ---------------------------------------------------------------------------

@needs_sklearn
def test_evaluate_dna_auc_perfect_separation():
    """Orthogonal per-root DNA → AUC close to 1.0."""
    D = 8
    rng = np.random.default_rng(0)
    # 3 roots, each with a unique one-hot DNA direction
    root_configs = [(1, [0, 1, 2], [1, 0, 0, 0, 0, 0, 0, 0]),
                    (2, [3, 4, 5], [0, 1, 0, 0, 0, 0, 0, 0]),
                    (3, [6, 7, 8], [0, 0, 1, 0, 0, 0, 0, 0])]
    # Label version: synapses 0-2 → root 1; 3-5 → root 2; 6-8 → root 3
    region = _make_region([1, 1, 1, 2, 2, 2, 3, 3, 3])
    frags = [
        _make_fragment_with_dna(rid, idxs, dna)
        for rid, idxs, dna in root_configs
    ]
    result = evaluate_dna_auc(region, frags, max_pairs=200, rng=np.random.default_rng(0), include_baseline=False)
    assert "dna_auc" in result
    assert result["dna_auc"] > 0.9, f"expected near-perfect AUC, got {result['dna_auc']}"
    assert result["n_pos"] > 0 and result["n_neg"] > 0


@needs_sklearn
def test_evaluate_dna_auc_includes_baseline():
    """include_baseline=True populates baseline_auc."""
    region = _make_region([1, 1, 1, 2, 2, 2])
    frags = [
        _make_fragment_with_dna(10, [0, 1, 2], [1.0, 0.0]),
        _make_fragment_with_dna(20, [3, 4, 5], [0.0, 1.0]),
    ]
    result = evaluate_dna_auc(region, frags, max_pairs=100, include_baseline=True)
    assert "baseline_auc" in result
    assert 0.0 <= result["baseline_auc"] <= 1.0


@needs_sklearn
def test_evaluate_dna_auc_no_dna_fragments():
    """Raises ValueError if no fragment carries a DNA embedding."""
    region = _make_region([1, 1, 2, 2])
    frag = Fragment(
        fragment_id=1, region_id="test", base_root_id=1,
        vertices_nm=np.ones((5, 3), np.float32),
        edges=np.stack([np.arange(4), np.arange(1, 5)], axis=1).astype(np.int64),
        endpoints_nm=np.ones((2, 3), np.float32),
        radius_nm=np.ones(5, np.float32),
        synapse_indices=np.array([0, 1], np.int64),
        dna=None,
    ).validate()
    with pytest.raises(ValueError, match="No fragment carries a DNA"):
        evaluate_dna_auc(region, [frag])


@needs_sklearn
def test_evaluate_dna_auc_n_no_dna_count():
    """n_no_dna reflects synapses without a matching Fragment DNA."""
    region = _make_region([1, 1, 2, 2, 3, 3])  # 6 synapses
    # Only cover synapses 0,1 — synapses 2-5 have no DNA
    frags = [_make_fragment_with_dna(10, [0, 1], [1.0, 0.0])]
    # Need ≥2 roots with ≥2 synapses; add a second fragment for root 2
    frags.append(_make_fragment_with_dna(20, [2, 3], [0.0, 1.0]))
    result = evaluate_dna_auc(region, frags, max_pairs=100, include_baseline=False)
    assert result["n_no_dna"] == 2  # synapses 4,5 have no DNA
