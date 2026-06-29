"""Tests for neuronauts.data.fragments — skeleton-to-Fragment extraction."""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.data.fragments import extract_fragments_for_region, skeleton_to_fragment
from neuronauts.schemas import Fragment, Region


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_region(n_syn: int = 4, root_ids: list[int] | None = None) -> Region:
    """Minimal Region with synapses assigned to the given root IDs."""
    rng = np.random.default_rng(0)
    if root_ids is None:
        root_ids = [1] * n_syn
    n = len(root_ids)
    pre_root = np.array(root_ids, dtype=np.int64)
    post_root = np.zeros(n, dtype=np.int64)
    return Region(
        region_id="test",
        bbox_nm=((0.0, 0.0, 0.0), (10000.0, 10000.0, 10000.0)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=rng.uniform(0, 10000, (n, 3)).astype(np.float32),
        post_pt_nm=rng.uniform(0, 10000, (n, 3)).astype(np.float32),
        pre_root_id=pre_root,
        post_root_id=post_root,
        synapse_id=np.arange(n, dtype=np.int64),
    )


def _chain_skeleton(n: int = 10):
    """Linear chain of n vertices along the x-axis."""
    verts = np.column_stack([
        np.arange(n, dtype=np.float32) * 100,
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    ])
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.int64)
    radii = np.ones(n, dtype=np.float32) * 200.0
    return verts, edges, radii


def _y_skeleton():
    """Y-shaped skeleton: branch at vertex 3, 3 leaves at 0, 7, 9."""
    # 0-1-2-3-4-5-6-7  (branch at 3)
    #         \\-8-9
    verts = np.array([
        [0, 0, 0], [100, 0, 0], [200, 0, 0], [300, 0, 0],
        [400, 0, 0], [500, 0, 0], [600, 0, 0], [700, 0, 0],
        [400, 100, 0], [400, 200, 0],
    ], dtype=np.float32)
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
        [3, 8], [8, 9],
    ], dtype=np.int64)
    radii = np.ones(len(verts), dtype=np.float32) * 150.0
    return verts, edges, radii


def _fake_archive_npz(tmp_path, skeletons: dict[int, tuple]) -> str:
    """Write a fake skeleton archive .npz and return its path."""
    all_verts = []
    all_edges = []
    all_radii = []
    root_ids = []
    v_offsets = [0]
    n_verts_list = []
    n_edges_list = []
    v_offset = 0

    for rid, (verts, edges, radii) in skeletons.items():
        root_ids.append(rid)
        v_offsets.append(v_offset + len(verts))
        n_verts_list.append(len(verts))
        n_edges_list.append(len(edges))
        all_verts.append(verts)
        all_edges.append(edges + v_offset)
        all_radii.append(radii)
        v_offset += len(verts)

    path = str(tmp_path / "archive.npz")
    np.savez_compressed(
        path,
        root_ids=np.array(root_ids, dtype=np.int64),
        v_offsets=np.array(v_offsets, dtype=np.int64),
        n_verts=np.array(n_verts_list, dtype=np.int64),
        n_edges=np.array(n_edges_list, dtype=np.int64),
        vertices=np.concatenate(all_verts, axis=0),
        edges=np.concatenate(all_edges, axis=0),
        radii=np.concatenate(all_radii, axis=0),
        voxel_size_nm=np.array([8.0, 8.0, 40.0], dtype=np.float32),
    )
    return path


# ---------------------------------------------------------------------------
# skeleton_to_fragment tests
# ---------------------------------------------------------------------------

def test_skeleton_to_fragment_chain():
    """Linear 10-vertex chain → 1 Fragment with exactly 2 leaf endpoints."""
    verts, edges, radii = _chain_skeleton(10)
    region = _make_region(n_syn=0, root_ids=[])
    frag = skeleton_to_fragment(verts, edges, radii, 42, region, fragment_id=42)

    assert frag is not None
    assert frag.fragment_id == 42
    assert frag.base_root_id == 42
    assert frag.n_vertices == 10
    assert len(frag.endpoints_nm) == 2  # two leaves on a chain
    assert frag.dna is None


def test_skeleton_to_fragment_tree_leaf_count():
    """Y-shaped skeleton → Fragment with 3 leaf endpoints."""
    verts, edges, radii = _y_skeleton()
    region = _make_region(n_syn=0, root_ids=[])
    frag = skeleton_to_fragment(verts, edges, radii, 7, region, fragment_id=7)

    assert frag is not None
    assert len(frag.endpoints_nm) == 3  # 3 leaves in the Y


def test_skeleton_to_fragment_below_min_vertices():
    """Skeleton with 2 vertices and min_vertices=3 → None."""
    verts, edges, radii = _chain_skeleton(2)
    region = _make_region(n_syn=0, root_ids=[])
    result = skeleton_to_fragment(verts, edges, radii, 1, region, fragment_id=1, min_vertices=3)
    assert result is None


def test_skeleton_to_fragment_synapse_assignment():
    """Synapses with pre_root_id matching base_root_id appear in synapse_indices."""
    verts, edges, radii = _chain_skeleton(10)
    # 4 synapses: first 3 belong to root_id=1, last belongs to root_id=99
    region = _make_region(n_syn=4, root_ids=[1, 1, 1, 99])
    frag = skeleton_to_fragment(verts, edges, radii, base_root_id=1, region=region, fragment_id=1)

    assert frag is not None
    # Synapses 0,1,2 have pre_root_id=1; synapse 3 has pre_root_id=99
    assert set(frag.synapse_indices.tolist()) == {0, 1, 2}


def test_skeleton_to_fragment_no_synapses():
    """No synapses matching root_id → empty synapse_indices."""
    verts, edges, radii = _chain_skeleton(5)
    region = _make_region(n_syn=3, root_ids=[99, 99, 99])
    frag = skeleton_to_fragment(verts, edges, radii, base_root_id=1, region=region, fragment_id=1)
    assert frag is not None
    assert len(frag.synapse_indices) == 0


def test_skeleton_to_fragment_validates_output():
    """Fragment.validate() does not raise for a well-formed extraction."""
    verts, edges, radii = _chain_skeleton(8)
    region = _make_region()
    frag = skeleton_to_fragment(verts, edges, radii, 1, region, fragment_id=1)
    assert frag is not None
    frag.validate()  # should not raise


# ---------------------------------------------------------------------------
# extract_fragments_for_region tests
# ---------------------------------------------------------------------------

def test_extract_fragments_for_region_count(tmp_path):
    """Two root IDs in archive → two Fragments returned."""
    verts_a, edges_a, radii_a = _chain_skeleton(8)
    verts_b, edges_b, radii_b = _chain_skeleton(6)
    archive_path = _fake_archive_npz(tmp_path, {10: (verts_a, edges_a, radii_a), 20: (verts_b, edges_b, radii_b)})
    region = _make_region(n_syn=0, root_ids=[])
    frags = extract_fragments_for_region(region, archive_path)

    assert len(frags) == 2
    ids = {f.fragment_id for f in frags}
    assert ids == {10, 20}


def test_extract_fragments_for_region_min_vertices_filter(tmp_path):
    """Root with 2 vertices is skipped when min_vertices=3."""
    verts_short, edges_short, radii_short = _chain_skeleton(2)
    verts_ok, edges_ok, radii_ok = _chain_skeleton(8)
    archive_path = _fake_archive_npz(tmp_path, {
        1: (verts_short, edges_short, radii_short),
        2: (verts_ok, edges_ok, radii_ok),
    })
    region = _make_region(n_syn=0, root_ids=[])
    frags = extract_fragments_for_region(region, archive_path, min_vertices=3)
    assert len(frags) == 1
    assert frags[0].fragment_id == 2


def test_extract_fragments_region_id_set(tmp_path):
    """All returned Fragments carry the region's region_id."""
    verts, edges, radii = _chain_skeleton(5)
    archive_path = _fake_archive_npz(tmp_path, {42: (verts, edges, radii)})
    region = _make_region(n_syn=0, root_ids=[])
    frags = extract_fragments_for_region(region, archive_path)
    assert all(f.region_id == "test" for f in frags)


def test_extract_fragments_dna_none(tmp_path):
    """Extraction does not fill dna — that is the represent/ stage's job."""
    verts, edges, radii = _chain_skeleton(7)
    archive_path = _fake_archive_npz(tmp_path, {5: (verts, edges, radii)})
    region = _make_region()
    frags = extract_fragments_for_region(region, archive_path)
    assert all(f.dna is None for f in frags)
