"""Tests for PathGrammarReranker: intrinsic features + cross-attention affinity."""
from __future__ import annotations

import numpy as np
import pytest

from neuronauts.schemas import Fragment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chain(n: int, frag_id: int = 1, step_nm: float = 500.0) -> Fragment:
    verts = np.column_stack([
        np.arange(n, dtype=np.float32) * step_nm,
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    ])
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)]).astype(np.int64)
    radii = np.ones(n, dtype=np.float32) * 200.0
    return Fragment(
        fragment_id=frag_id,
        region_id="test",
        base_root_id=frag_id,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=verts[[0, -1]],
        radius_nm=radii,
        synapse_indices=np.array([], dtype=np.int64),
        dna=None,
    ).validate()


def _tree(frag_id: int = 2) -> Fragment:
    verts = np.array([
        [0, 0, 0], [500, 0, 0], [1000, 300, 0], [1000, -300, 0], [1500, 0, 0],
    ], dtype=np.float32)
    edges = np.array([[0,1],[1,2],[1,3],[2,4]], dtype=np.int64)
    radii = np.ones(5, dtype=np.float32) * 250.0
    return Fragment(
        fragment_id=frag_id, region_id="test", base_root_id=frag_id,
        vertices_nm=verts, edges=edges, endpoints_nm=verts[[0,3,4]],
        radius_nm=radii, synapse_indices=np.array([], dtype=np.int64), dna=None,
    ).validate()


# ---------------------------------------------------------------------------
# path_to_intrinsic
# ---------------------------------------------------------------------------

class TestPathToIntrinsic:
    def test_shape(self):
        from neuronauts.represent.path_grammar import path_to_intrinsic
        verts = np.random.randn(10, 3).astype(np.float32) * 1000
        radii = np.ones(10, dtype=np.float32) * 200
        feat = path_to_intrinsic(verts, radii)
        assert feat.shape == (10, 4)
        assert feat.dtype == np.float32

    def test_single_vertex(self):
        from neuronauts.represent.path_grammar import path_to_intrinsic
        feat = path_to_intrinsic(
            np.array([[0., 0., 0.]], dtype=np.float32),
            np.array([100.], dtype=np.float32),
        )
        assert feat.shape == (1, 4)
        assert np.isfinite(feat).all()

    def test_step_length_positive(self):
        from neuronauts.represent.path_grammar import path_to_intrinsic
        verts = np.array([[0,0,0],[500,0,0],[1000,0,0]], dtype=np.float32)
        radii = np.ones(3, dtype=np.float32) * 200
        feat = path_to_intrinsic(verts, radii)
        # log(step+1) for first two vertices should be > 0
        assert feat[0, 0] > 0
        assert feat[1, 0] > 0

    def test_straight_path_zero_turn(self):
        """Perfectly straight path should have zero turning angle."""
        from neuronauts.represent.path_grammar import path_to_intrinsic
        verts = np.column_stack([np.arange(5) * 500.0,
                                  np.zeros(5), np.zeros(5)]).astype(np.float32)
        radii = np.ones(5, dtype=np.float32) * 200
        feat = path_to_intrinsic(verts, radii)
        # Interior vertices (1..3) should have turn ~ 0
        np.testing.assert_allclose(feat[1:-1, 1], 0.0, atol=1e-5)

    def test_translation_invariant(self):
        """Shifting all vertices must not change intrinsic features."""
        from neuronauts.represent.path_grammar import path_to_intrinsic
        verts = np.random.randn(8, 3).astype(np.float32) * 500
        radii = np.random.rand(8).astype(np.float32) * 300 + 100
        offset = np.array([1e6, -2e5, 3e5], dtype=np.float32)
        f1 = path_to_intrinsic(verts, radii)
        f2 = path_to_intrinsic(verts + offset, radii)
        np.testing.assert_allclose(f1, f2, atol=1e-4)

    def test_rotation_invariant(self):
        """Rotating all vertices must not change intrinsic features."""
        from neuronauts.represent.path_grammar import path_to_intrinsic
        rng = np.random.default_rng(7)
        verts = rng.standard_normal((8, 3)).astype(np.float32) * 500
        radii = (rng.uniform(100, 400, 8)).astype(np.float32)
        # Rotation matrix (90° around z-axis)
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        f1 = path_to_intrinsic(verts, radii)
        f2 = path_to_intrinsic(verts @ R.T, radii)
        np.testing.assert_allclose(f1, f2, atol=1e-4)

    def test_all_finite(self):
        from neuronauts.represent.path_grammar import path_to_intrinsic
        verts = np.random.randn(20, 3).astype(np.float32) * 2000
        radii = np.random.rand(20).astype(np.float32) * 500
        feat = path_to_intrinsic(verts, radii)
        assert np.isfinite(feat).all()


# ---------------------------------------------------------------------------
# fragment_to_intrinsic_paths
# ---------------------------------------------------------------------------

class TestFragmentToIntrinsicPaths:
    def test_returns_list_of_arrays(self):
        from neuronauts.represent.path_grammar import fragment_to_intrinsic_paths
        frag = _chain(12)
        paths = fragment_to_intrinsic_paths(frag, n_paths=4)
        assert isinstance(paths, list)
        # A chain has only 1 unique leaf-to-leaf path; count is ≤ n_paths
        assert 1 <= len(paths) <= 4
        for p in paths:
            assert p.ndim == 2
            assert p.shape[1] == 4

    def test_tree_has_multiple_paths(self):
        from neuronauts.represent.path_grammar import fragment_to_intrinsic_paths
        frag = _tree()  # Y-shape: 3 leaves → multiple unique paths
        paths = fragment_to_intrinsic_paths(frag, n_paths=4)
        assert len(paths) >= 2

    def test_single_vertex_fragment(self):
        from neuronauts.represent.path_grammar import fragment_to_intrinsic_paths
        frag = Fragment(
            fragment_id=1, region_id="t", base_root_id=1,
            vertices_nm=np.array([[0.,0.,0.]], dtype=np.float32),
            edges=np.zeros((0,2), dtype=np.int64),
            endpoints_nm=np.array([[0.,0.,0.]], dtype=np.float32),
            radius_nm=np.array([200.], dtype=np.float32),
            synapse_indices=np.array([], dtype=np.int64), dna=None,
        ).validate()
        paths = fragment_to_intrinsic_paths(frag, n_paths=2)
        assert len(paths) >= 1  # at minimum the single-vertex path


# ---------------------------------------------------------------------------
# PathGrammarReranker forward pass
# ---------------------------------------------------------------------------

class TestPathGrammarRerankerForward:
    def _get_paths(self, frag, n=4, device="cpu"):
        import torch
        from neuronauts.represent.path_grammar import fragment_to_intrinsic_paths
        raw = fragment_to_intrinsic_paths(frag, n_paths=n)
        return [torch.from_numpy(p).to(device) for p in raw]

    def test_output_scalar(self):
        from neuronauts.represent.path_grammar import PathGrammarReranker
        r = PathGrammarReranker(d_model=32, n_heads=2, n_path_layers=1, n_cross_layers=1)
        fa, fb = _chain(10, 1), _chain(10, 2)
        pa, pb = self._get_paths(fa), self._get_paths(fb)
        out = r(pa, pb)
        assert out.shape == ()

    def test_output_in_0_1(self):
        from neuronauts.represent.path_grammar import PathGrammarReranker
        r = PathGrammarReranker(d_model=32, n_heads=2, n_path_layers=1, n_cross_layers=1)
        fa, fb = _chain(10, 1), _tree(2)
        pa, pb = self._get_paths(fa), self._get_paths(fb)
        out = float(r(pa, pb).item())
        assert 0.0 <= out <= 1.0

    def test_output_finite(self):
        import torch
        from neuronauts.represent.path_grammar import PathGrammarReranker
        r = PathGrammarReranker(d_model=32, n_heads=2, n_path_layers=1, n_cross_layers=1)
        fa, fb = _chain(15, 1), _chain(15, 2)
        pa, pb = self._get_paths(fa), self._get_paths(fb)
        out = r(pa, pb)
        assert torch.isfinite(out)

    def test_asymmetric_path_counts(self):
        """A with 3 paths, B with 5 paths — cross-attention handles variable K."""
        from neuronauts.represent.path_grammar import PathGrammarReranker
        r = PathGrammarReranker(d_model=32, n_heads=2, n_path_layers=1, n_cross_layers=1)
        fa, fb = _chain(10, 1), _chain(12, 2)
        pa = self._get_paths(fa, n=3)
        pb = self._get_paths(fb, n=5)
        out = r(pa, pb)
        assert out.shape == ()


# ---------------------------------------------------------------------------
# score_fragment_pairs
# ---------------------------------------------------------------------------

class TestScoreFragmentPairs:
    def test_returns_array(self):
        from neuronauts.represent.path_grammar import PathGrammarReranker, score_fragment_pairs
        r = PathGrammarReranker(d_model=32, n_heads=2, n_path_layers=1, n_cross_layers=1)
        pairs = [(_chain(10, i), _chain(10, i+1)) for i in range(4)]
        scores = score_fragment_pairs(r, pairs, n_paths=4)
        assert scores.shape == (4,)
        assert scores.dtype == np.float32

    def test_values_in_range(self):
        from neuronauts.represent.path_grammar import PathGrammarReranker, score_fragment_pairs
        r = PathGrammarReranker(d_model=32, n_heads=2, n_path_layers=1, n_cross_layers=1)
        pairs = [(_chain(8, i), _chain(8, i+1)) for i in range(6)]
        scores = score_fragment_pairs(r, pairs, n_paths=4)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()


# ---------------------------------------------------------------------------
# train_path_grammar_reranker
# ---------------------------------------------------------------------------

class TestTrainPathGrammarReranker:
    def _world(self, n_neurons=4, frags_per=2):
        import dataclasses
        frags_by_group = []
        rlm = {}
        fid = 100
        for label in range(1, n_neurons + 1):
            group = []
            for k in range(frags_per):
                f = dataclasses.replace(_chain(10 + k, fid), base_root_id=fid)
                rlm[fid] = {label}
                group.append(f)
                fid += 1
            frags_by_group.append(group)
        return frags_by_group, rlm

    def test_history_keys(self):
        from neuronauts.represent.path_grammar import (
            PathGrammarReranker, train_path_grammar_reranker,
        )
        r = PathGrammarReranker(d_model=16, n_heads=2, n_path_layers=1, n_cross_layers=1)
        fl, rlm = self._world()
        h = train_path_grammar_reranker(r, fl, n_epochs=2, root_label_map=rlm,
                                         n_pairs_per_epoch=8, log_every=0)
        assert {"loss", "pos_score", "neg_score"} <= set(h.keys())

    def test_history_length(self):
        from neuronauts.represent.path_grammar import (
            PathGrammarReranker, train_path_grammar_reranker,
        )
        r = PathGrammarReranker(d_model=16, n_heads=2, n_path_layers=1, n_cross_layers=1)
        fl, rlm = self._world()
        h = train_path_grammar_reranker(r, fl, n_epochs=4, root_label_map=rlm,
                                         n_pairs_per_epoch=8, log_every=0)
        assert len(h["loss"]) == 4

    def test_training_mechanics(self):
        """Training loop runs, produces finite history, and updates weights."""
        import torch
        from neuronauts.represent.path_grammar import (
            PathGrammarReranker, train_path_grammar_reranker,
        )
        torch.manual_seed(0)
        fl, rlm = self._world(n_neurons=5, frags_per=2)
        r = PathGrammarReranker(d_model=32, n_heads=2, n_path_layers=1, n_cross_layers=1)

        # Capture initial weight snapshot
        before = {k: v.clone() for k, v in r.named_parameters()}
        h = train_path_grammar_reranker(r, fl, n_epochs=3, root_label_map=rlm,
                                         n_pairs_per_epoch=16, log_every=0)

        # Loss values are finite
        assert all(np.isfinite(h["loss"]))
        assert all(np.isfinite(h["pos_score"]))
        assert all(np.isfinite(h["neg_score"]))
        # Weights changed (optimizer actually stepped)
        after = dict(r.named_parameters())
        changed = sum(
            1 for k in before if not torch.equal(before[k], after[k].data)
        )
        assert changed > 0, "no weights changed during training"

    def test_few_groups_raises(self):
        from neuronauts.represent.path_grammar import (
            PathGrammarReranker, train_path_grammar_reranker,
        )
        r = PathGrammarReranker(d_model=16, n_heads=2, n_path_layers=1, n_cross_layers=1)
        with pytest.raises(ValueError, match="≥2"):
            train_path_grammar_reranker(r, [[_chain(8, 1)]], n_epochs=1)
