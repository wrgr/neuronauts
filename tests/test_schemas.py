"""Smoke tests for the inter-stage contracts in neuronauts/schemas.py.

These guard the artifact boundaries every pipeline stage depends on: shape /
dtype validation and pickle-free disk round-trips.
"""

import numpy as np
import pytest

from neuronauts.schemas import (
    ConnectomeGraph,
    Fragment,
    NeuronHypothesis,
    Region,
    load_fragments,
    save_fragments,
)


class _FakeSynapseTable:
    """Duck-typed stand-in for fetch.SynapseTable (box-relative voxel coords)."""

    def __init__(self, n=5, with_seg=True):
        rng = np.random.default_rng(0)
        self.pre_pt = rng.integers(0, 100, size=(n, 3)).astype(np.float32)
        self.post_pt = rng.integers(0, 100, size=(n, 3)).astype(np.float32)
        self.pre_root_id = np.arange(1, n + 1, dtype=np.int64)
        self.post_root_id = np.arange(101, 101 + n, dtype=np.int64)
        self.synapse_id = np.arange(1000, 1000 + n, dtype=np.int64)
        self.pre_seg_id = self.pre_root_id.copy() if with_seg else None
        self.post_seg_id = self.post_root_id.copy() if with_seg else None


def _make_fragment(fid=0, region="r0", v=6, d=8, with_dna=True):
    verts = np.linspace(0, 1, v * 3).reshape(v, 3).astype(np.float32)
    edges = np.stack([np.arange(v - 1), np.arange(1, v)], axis=1).astype(np.int64)
    return Fragment(
        fragment_id=fid,
        region_id=region,
        base_root_id=42 + fid,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=verts[[0, -1]],
        radius_nm=np.ones(v, dtype=np.float32),
        synapse_indices=np.array([0, 2], dtype=np.int64),
        dna=np.arange(d, dtype=np.float32) if with_dna else None,
    )


def test_region_from_synapse_table_globalizes_coords():
    syn = _FakeSynapseTable(n=4)
    bbox = ((1000.0, 2000.0, 3000.0), (5000.0, 6000.0, 7000.0))
    vox = (8.0, 8.0, 40.0)
    region = Region.from_synapse_table(
        syn, region_id="tile_0", bbox_nm=bbox, voxel_size_nm=vox,
        seg_version=117, label_version=1412,
    )
    assert region.n_synapses == 4
    # global = voxel * voxel_size + origin
    expected = syn.pre_pt[0] * np.array(vox) + np.array(bbox[0])
    np.testing.assert_allclose(region.pre_pt_nm[0], expected, rtol=1e-5)
    assert region.pre_pt_nm.dtype == np.float32


def test_region_npz_roundtrip(tmp_path):
    syn = _FakeSynapseTable(n=7, with_seg=True)
    bbox = ((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0))
    region = Region.from_synapse_table(
        syn, region_id="tile_x", bbox_nm=bbox, voxel_size_nm=(4, 4, 40),
        seg_version=117, label_version=1412,
    )
    path = tmp_path / "region.npz"
    region.save_npz(str(path))
    loaded = Region.load_npz(str(path))
    assert loaded.region_id == "tile_x"
    assert loaded.seg_version == 117 and loaded.label_version == 1412
    assert loaded.voxel_size_nm == (4.0, 4.0, 40.0)
    np.testing.assert_array_equal(loaded.synapse_id, region.synapse_id)
    np.testing.assert_allclose(loaded.pre_pt_nm, region.pre_pt_nm)
    np.testing.assert_array_equal(loaded.pre_seg_id, region.pre_seg_id)


def test_region_validate_rejects_length_mismatch():
    region = Region(
        region_id="bad", bbox_nm=((0, 0, 0), (1, 1, 1)), voxel_size_nm=(1, 1, 1),
        seg_version=1, label_version=2,
        pre_pt_nm=np.zeros((3, 3), np.float32), post_pt_nm=np.zeros((3, 3), np.float32),
        pre_root_id=np.zeros(3, np.int64), post_root_id=np.zeros(2, np.int64),  # mismatch
        synapse_id=np.zeros(3, np.int64),
    )
    with pytest.raises(ValueError):
        region.validate()


def test_fragments_roundtrip_with_dna(tmp_path):
    frags = [_make_fragment(fid=i, region=f"r{i}", v=5 + i, d=8) for i in range(3)]
    path = tmp_path / "frags.npz"
    save_fragments(str(path), frags)
    loaded = load_fragments(str(path))
    assert len(loaded) == 3
    for a, b in zip(frags, loaded):
        assert a.fragment_id == b.fragment_id
        assert a.region_id == b.region_id
        assert a.base_root_id == b.base_root_id
        np.testing.assert_allclose(a.vertices_nm, b.vertices_nm)
        np.testing.assert_array_equal(a.edges, b.edges)
        np.testing.assert_allclose(a.endpoints_nm, b.endpoints_nm)
        np.testing.assert_array_equal(a.synapse_indices, b.synapse_indices)
        np.testing.assert_allclose(a.dna, b.dna)


def test_fragments_roundtrip_without_dna(tmp_path):
    frags = [_make_fragment(fid=i, v=4, with_dna=False) for i in range(2)]
    path = tmp_path / "frags_nodna.npz"
    save_fragments(str(path), frags)
    loaded = load_fragments(str(path))
    assert len(loaded) == 2
    assert all(f.dna is None for f in loaded)


def test_fragment_validate_rejects_bad_edge_index():
    frag = _make_fragment()
    frag.edges = np.array([[0, 999]], dtype=np.int64)  # out of range
    with pytest.raises(ValueError):
        frag.validate()


def test_neuron_hypothesis_records_cross_region_span():
    hyp = NeuronHypothesis(
        neuron_id=1,
        fragment_ids=[10, 11, 12],
        synapse_indices=np.array([0, 1, 2, 3], dtype=np.int64),
        pooled_dna=np.ones(8, dtype=np.float32),
        spans_regions=["tile_0", "tile_1"],
    ).validate()
    assert len(hyp.spans_regions) == 2  # genuine cross-box assembly
    with pytest.raises(ValueError):
        NeuronHypothesis(neuron_id=2, fragment_ids=[], synapse_indices=np.array([], np.int64)).validate()


def test_connectome_graph_validate():
    g = ConnectomeGraph(
        neuron_ids=np.array([100, 200, 300], dtype=np.int64),
        node_features=np.ones((3, 4), dtype=np.float32),
        src=np.array([0, 1], dtype=np.int64),
        dst=np.array([1, 2], dtype=np.int64),
        edge_synapse_count=np.array([5, 9], dtype=np.int64),
    ).validate()
    assert g.n_nodes == 3 and g.n_edges == 2
    g.dst = np.array([1, 99], dtype=np.int64)  # out-of-range node
    with pytest.raises(ValueError):
        g.validate()
