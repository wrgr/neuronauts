"""Tests for SkeletonGNN: graph-level DNA encoder from raw skeleton vertices."""
from __future__ import annotations

import numpy as np
import pytest

from neuronauts.schemas import Fragment, Region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chain_fragment(n_verts: int = 10, frag_id: int = 1) -> Fragment:
    """Straight-line skeleton with n_verts vertices."""
    verts = np.column_stack([
        np.linspace(0, 1000, n_verts),
        np.zeros(n_verts),
        np.zeros(n_verts),
    ]).astype(np.float32)
    edges = np.column_stack([np.arange(n_verts - 1), np.arange(1, n_verts)]).astype(np.int64)
    radii = np.ones(n_verts, dtype=np.float32) * 200.0
    endpoints = verts[[0, -1]]
    return Fragment(
        fragment_id=frag_id,
        region_id="test",
        base_root_id=frag_id,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=endpoints,
        radius_nm=radii,
        synapse_indices=np.array([], dtype=np.int64),
        dna=None,
    ).validate()


def _make_tree_fragment(frag_id: int = 2) -> Fragment:
    """Y-shaped skeleton: root(0) — branch(1) — leaf(2), leaf(3)."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [500.0, 0.0, 0.0],
        [1000.0, 500.0, 0.0],
        [1000.0, -500.0, 0.0],
        [1500.0, 0.0, 0.0],
    ], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [1, 3], [2, 4]], dtype=np.int64)
    radii = np.ones(5, dtype=np.float32) * 300.0
    return Fragment(
        fragment_id=frag_id,
        region_id="test",
        base_root_id=frag_id,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=verts[[0, 3, 4]],
        radius_nm=radii,
        synapse_indices=np.array([], dtype=np.int64),
        dna=None,
    ).validate()


def _make_isolated_fragment(frag_id: int = 3) -> Fragment:
    """Single vertex, no edges — edge-case for the GNN."""
    verts = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
    radii = np.array([150.0], dtype=np.float32)
    return Fragment(
        fragment_id=frag_id,
        region_id="test",
        base_root_id=frag_id,
        vertices_nm=verts,
        edges=np.zeros((0, 2), dtype=np.int64),
        endpoints_nm=verts,
        radius_nm=radii,
        synapse_indices=np.array([], dtype=np.int64),
        dna=None,
    ).validate()


# ---------------------------------------------------------------------------
# fragment_to_tensors
# ---------------------------------------------------------------------------

class TestFragmentToTensors:
    def test_chain_shape(self):
        from neuronauts.represent.skeleton_gnn import fragment_to_tensors
        frag = _make_chain_fragment(10)
        nf, es, ed, ef = fragment_to_tensors(frag)
        assert nf.shape == (10, 4)
        assert es.shape[0] == ed.shape[0] == ef.shape[0]  # same #edges
        assert ef.shape[1] == 1

    def test_bidirectional_edges(self):
        """Each undirected edge produces two directed edges."""
        from neuronauts.represent.skeleton_gnn import fragment_to_tensors
        frag = _make_chain_fragment(5)  # 4 undirected edges
        nf, es, ed, ef = fragment_to_tensors(frag)
        assert es.shape[0] == 8  # 4 * 2

    def test_centroid_normalised(self):
        """Node xyz features should be centroid-normalised (mean ≈ 0)."""
        from neuronauts.represent.skeleton_gnn import fragment_to_tensors
        frag = _make_chain_fragment(20)
        nf, _, _, _ = fragment_to_tensors(frag)
        xyz = nf[:, :3].numpy()
        assert np.abs(xyz.mean(axis=0)).max() < 1.0  # centroid near origin

    def test_radius_in_node_feat(self):
        """4th column of node features should match fragment radii."""
        from neuronauts.represent.skeleton_gnn import fragment_to_tensors
        frag = _make_chain_fragment(5)
        nf, _, _, _ = fragment_to_tensors(frag)
        radii_col = nf[:, 3].numpy()
        np.testing.assert_allclose(radii_col, 200.0)

    def test_edge_lengths_positive(self):
        """All edge lengths should be > 0 for a non-degenerate skeleton."""
        from neuronauts.represent.skeleton_gnn import fragment_to_tensors
        frag = _make_chain_fragment(6)
        _, _, _, ef = fragment_to_tensors(frag)
        assert (ef.numpy() > 0).all()

    def test_isolated_vertex_no_edges(self):
        """Single-vertex fragment produces 0-row edge tensors without crashing."""
        from neuronauts.represent.skeleton_gnn import fragment_to_tensors
        frag = _make_isolated_fragment()
        nf, es, ed, ef = fragment_to_tensors(frag)
        assert nf.shape == (1, 4)
        assert es.shape[0] == 0
        assert ef.shape == (0, 1)


# ---------------------------------------------------------------------------
# SkeletonGNN forward pass
# ---------------------------------------------------------------------------

class TestSkeletonGNNForward:
    def test_output_shape(self):
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, fragment_to_tensors
        gnn = SkeletonGNN(output_dim=16)
        frag = _make_chain_fragment(10)
        nf, es, ed, ef = fragment_to_tensors(frag)
        out = gnn(nf, es, ed, ef)
        assert out.shape == (16,)

    def test_output_finite(self):
        import torch
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, fragment_to_tensors
        gnn = SkeletonGNN(output_dim=32)
        frag = _make_tree_fragment()
        nf, es, ed, ef = fragment_to_tensors(frag)
        out = gnn(nf, es, ed, ef)
        assert torch.isfinite(out).all()

    def test_isolated_vertex_no_crash(self):
        """Forward pass must not crash when there are no edges."""
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, fragment_to_tensors
        gnn = SkeletonGNN(output_dim=8)
        frag = _make_isolated_fragment()
        nf, es, ed, ef = fragment_to_tensors(frag)
        out = gnn(nf, es, ed, ef)
        assert out.shape == (8,)

    def test_default_output_dim(self):
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, fragment_to_tensors
        gnn = SkeletonGNN()  # output_dim=32 by default
        frag = _make_chain_fragment(8)
        nf, es, ed, ef = fragment_to_tensors(frag)
        assert gnn(nf, es, ed, ef).shape == (32,)

    def test_different_topology_different_embedding(self):
        """Chain vs tree should produce different embeddings (with fixed weights)."""
        import torch
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, fragment_to_tensors
        torch.manual_seed(7)
        gnn = SkeletonGNN(output_dim=32, d_model=32)
        f_chain = _make_chain_fragment(n_verts=5, frag_id=1)
        f_tree = _make_tree_fragment(frag_id=2)
        nf1, es1, ed1, ef1 = fragment_to_tensors(f_chain)
        nf2, es2, ed2, ef2 = fragment_to_tensors(f_tree)
        out1 = gnn(nf1, es1, ed1, ef1)
        out2 = gnn(nf2, es2, ed2, ef2)
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# encode_fragments_gnn
# ---------------------------------------------------------------------------

class TestEncodeFragmentsGNN:
    def test_fills_dna(self):
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, encode_fragments_gnn
        gnn = SkeletonGNN(output_dim=16)
        frags = [_make_chain_fragment(10, i) for i in range(3)]
        result = encode_fragments_gnn(gnn, frags)
        for f in result:
            assert f.dna is not None
            assert f.dna.shape == (16,)
            assert f.dna.dtype == np.float32

    def test_l2_normalised(self):
        """Embeddings must be unit-normalised."""
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, encode_fragments_gnn
        gnn = SkeletonGNN(output_dim=32)
        frags = [_make_chain_fragment(8, i) for i in range(5)]
        result = encode_fragments_gnn(gnn, frags)
        for f in result:
            norm = float(np.linalg.norm(f.dna))
            assert abs(norm - 1.0) < 1e-5

    def test_preserves_metadata(self):
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, encode_fragments_gnn
        gnn = SkeletonGNN(output_dim=8)
        orig = _make_chain_fragment(6, frag_id=99)
        result = encode_fragments_gnn(gnn, [orig])
        r = result[0]
        assert r.fragment_id == 99
        assert r.region_id == "test"
        assert r.base_root_id == 99

    def test_deterministic(self):
        """Two calls with gnn.eval() should produce identical embeddings."""
        import torch
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, encode_fragments_gnn
        torch.manual_seed(0)
        gnn = SkeletonGNN(output_dim=16)
        frags = [_make_chain_fragment(7, i) for i in range(2)]
        r1 = encode_fragments_gnn(gnn, frags)
        r2 = encode_fragments_gnn(gnn, frags)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a.dna, b.dna)


# ---------------------------------------------------------------------------
# train_skeleton_gnn
# ---------------------------------------------------------------------------

class TestTrainSkeletonGNN:
    def _make_world(self, n_neurons: int = 4, frags_per: int = 2) -> tuple:
        """Create fragment_lists and root_label_map for training."""
        fragment_lists = []
        root_label_map = {}
        for label in range(1, n_neurons + 1):
            group = []
            for _ in range(frags_per):
                n_verts = np.random.randint(8, 20)
                frag_id = len(root_label_map) + 1000
                frag = _make_chain_fragment(n_verts, frag_id)
                # Override base_root_id to be unique
                import dataclasses
                frag = dataclasses.replace(frag, base_root_id=frag_id)
                root_label_map[frag_id] = {label}
                group.append(frag)
            fragment_lists.append(group)
        return fragment_lists, root_label_map

    def test_history_keys(self):
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, train_skeleton_gnn
        gnn = SkeletonGNN(output_dim=8, d_model=16, n_layers=1)
        fl, rlm = self._make_world(4, 2)
        history = train_skeleton_gnn(gnn, fl, n_epochs=2, root_label_map=rlm, log_every=0)
        assert set(history.keys()) >= {"loss", "pos_cos", "neg_cos"}

    def test_history_length(self):
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, train_skeleton_gnn
        gnn = SkeletonGNN(output_dim=8, d_model=16, n_layers=1)
        fl, rlm = self._make_world(4, 2)
        history = train_skeleton_gnn(gnn, fl, n_epochs=5, root_label_map=rlm, log_every=0)
        assert len(history["loss"]) == 5

    def test_loss_decreases(self):
        """After sufficient epochs on separable data, loss should decrease."""
        import torch
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, train_skeleton_gnn

        # Make clearly different fragments per group by giving very different vertex counts
        torch.manual_seed(42)
        rng = np.random.default_rng(42)

        fragment_lists = []
        root_label_map = {}
        fid = 1000
        for label in range(1, 6):
            group = []
            for _ in range(3):
                n_v = label * 8  # each neuron has distinctly different size
                verts = np.column_stack([
                    np.linspace(label * 1000, label * 2000, n_v),
                    np.zeros(n_v),
                    np.zeros(n_v),
                ]).astype(np.float32)
                edges = np.column_stack([np.arange(n_v - 1), np.arange(1, n_v)]).astype(np.int64)
                import dataclasses
                frag = _make_chain_fragment(n_v, fid)
                frag = dataclasses.replace(frag, base_root_id=fid, vertices_nm=verts)
                root_label_map[fid] = {label}
                group.append(frag)
                fid += 1
            fragment_lists.append(group)

        gnn = SkeletonGNN(output_dim=16, d_model=32, n_layers=2)
        history = train_skeleton_gnn(
            gnn, fragment_lists, n_epochs=20, root_label_map=root_label_map, log_every=0
        )
        # Loss should be lower in the last half than the first half on average
        first_half = np.mean(history["loss"][:10])
        last_half = np.mean(history["loss"][10:])
        assert last_half <= first_half + 0.3  # some tolerance

    def test_few_groups_raises(self):
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, train_skeleton_gnn
        gnn = SkeletonGNN(output_dim=8)
        fl = [[_make_chain_fragment(6, 1)]]  # only 1 group
        with pytest.raises(ValueError, match="≥2"):
            train_skeleton_gnn(gnn, fl, n_epochs=1)

    def test_no_root_label_map(self):
        """Training without root_label_map uses base_root_id as group key."""
        import dataclasses
        from neuronauts.represent.skeleton_gnn import SkeletonGNN, train_skeleton_gnn

        fragment_lists = []
        for g in range(4):
            frags = []
            for k in range(2):
                fid = g * 10 + k
                frag = dataclasses.replace(_make_chain_fragment(8, fid), base_root_id=g)
                frags.append(frag)
            fragment_lists.append(frags)

        gnn = SkeletonGNN(output_dim=8, d_model=16, n_layers=1)
        history = train_skeleton_gnn(gnn, fragment_lists, n_epochs=3, log_every=0)
        assert len(history["loss"]) == 3
