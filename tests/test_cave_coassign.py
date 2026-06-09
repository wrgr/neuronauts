"""Unit tests for neuronauts.data.cave that do not require network access.

The network-dependent paths (fetch_v117_region) are exercised by the live
harness in scripts/v117_coassign.py. Here we test the pure-Python helpers and
the local DNA-encoding path, which run SkeletonGNN on in-memory skeletons.
"""

import numpy as np
import pytest

from neuronauts.data.cave import _leaf_vertices, encode_seg_dna
from neuronauts.fetch import SkeletonData

torch = pytest.importorskip("torch")


def _chain_skeleton(root_id: int, n: int = 6, version: int = 117) -> SkeletonData:
    """A straight chain skeleton with n vertices (two leaves at the ends)."""
    verts = np.stack(
        [np.arange(n), np.zeros(n), np.zeros(n)], axis=1
    ).astype(np.float32) * 1000.0
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int64)
    radius = np.full(n, 150.0, dtype=np.float32)
    return SkeletonData(
        root_id=root_id,
        materialization_version=version,
        vertices=verts,
        edges=edges,
        radius=radius,
    )


class TestLeafVertices:
    def test_chain_has_two_leaves(self):
        sk = _chain_skeleton(1, n=5)
        leaves = _leaf_vertices(sk.vertices, sk.edges)
        # A chain's two endpoints are the only degree-1 vertices
        assert leaves.shape == (2, 3)

    def test_empty_skeleton(self):
        leaves = _leaf_vertices(np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int64))
        assert leaves.shape == (0, 3)

    def test_no_edges_returns_single_vertex(self):
        verts = np.ones((3, 3), dtype=np.float32)
        leaves = _leaf_vertices(verts, np.zeros((0, 2), np.int64))
        assert leaves.shape == (1, 3)


class TestEncodeSegDNA:
    def test_returns_embedding_per_segment(self):
        skeletons = {1: _chain_skeleton(1), 2: _chain_skeleton(2)}
        seg_ids = np.array([1, 1, 2, 2], dtype=np.int64)
        seg_dna = encode_seg_dna(skeletons, seg_ids, dna_dim=32)
        assert set(seg_dna.keys()) == {1, 2}
        for v in seg_dna.values():
            assert v.shape == (32,)
            assert v.dtype == np.float32
            assert np.isfinite(v).all()

    def test_empty_skeleton_gets_zero_embedding(self):
        empty = SkeletonData(
            root_id=9,
            materialization_version=117,
            vertices=np.zeros((0, 3), np.float32),
            edges=np.zeros((0, 2), np.int64),
            radius=None,
        )
        skeletons = {9: empty}
        seg_dna = encode_seg_dna(skeletons, np.array([9, 9], dtype=np.int64), dna_dim=16)
        assert seg_dna[9].shape == (16,)
        assert np.allclose(seg_dna[9], 0.0)

    def test_missing_radius_is_handled(self):
        sk = _chain_skeleton(3)
        sk_no_radius = SkeletonData(
            root_id=3,
            materialization_version=117,
            vertices=sk.vertices,
            edges=sk.edges,
            radius=None,
        )
        seg_dna = encode_seg_dna({3: sk_no_radius}, np.array([3], dtype=np.int64), dna_dim=16)
        assert seg_dna[3].shape == (16,)
        assert np.isfinite(seg_dna[3]).all()
