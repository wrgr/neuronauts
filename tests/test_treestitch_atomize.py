"""Tests for treestitch.atomize — level −1 atomization."""

from __future__ import annotations

import numpy as np
import pytest

from neuronauts.schemas import Fragment, Region
from treestitch.atomize import (
    atomize_world,
    flag_odd_fragments,
    frankenmerge_separation,
    odd_edge_mask,
    oddness_scores,
    split_fragment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frag_from(verts, edges, fid=1) -> Fragment:
    verts = np.asarray(verts, dtype=np.float32)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return Fragment(
        fragment_id=fid,
        region_id="test",
        base_root_id=fid,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=verts[:1],
        radius_nm=np.full(len(verts), 100.0, dtype=np.float32),
        synapse_indices=np.zeros(0, dtype=np.int64),
        dna=None,
    ).validate()


def y_fragment(fid=1) -> Fragment:
    """A Y: chain 0-1-2, branches 2-3 and 2-4 (vertex 2 has degree 3)."""
    verts = [(0, 0, 0), (1000, 0, 0), (2000, 0, 0),
             (3000, 1000, 0), (3000, -1000, 0)]
    edges = [(0, 1), (1, 2), (2, 3), (2, 4)]
    return frag_from(verts, edges, fid)


def bridged_fragment(fid=1) -> Fragment:
    """Two 3-vertex chains joined by one 50 µm bridge edge (the franken glue)."""
    verts = [(0, 0, 0), (1000, 0, 0), (2000, 0, 0),
             (52000, 0, 0), (53000, 0, 0), (54000, 0, 0)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    return frag_from(verts, edges, fid)


# ---------------------------------------------------------------------------
# odd_edge_mask / oddness
# ---------------------------------------------------------------------------

def test_odd_edge_mask_flags_bridge():
    f = bridged_fragment()
    mask = odd_edge_mask(f.vertices_nm, f.edges)
    assert mask.tolist() == [False, False, True, False, False]


def test_odd_edge_mask_respects_min_length():
    # uniform 1 µm edges, one 3 µm edge: ratio 3 > factor would flag it,
    # but 3 µm < 10 µm floor keeps it normal
    verts = [(0, 0, 0), (1000, 0, 0), (2000, 0, 0), (5000, 0, 0)]
    f = frag_from(verts, [(0, 1), (1, 2), (2, 3)])
    assert not odd_edge_mask(f.vertices_nm, f.edges, long_edge_factor=2.0).any()


def test_oddness_scores_and_flags():
    normal = y_fragment(fid=1)
    odd = bridged_fragment(fid=2)
    assert not oddness_scores(normal)["is_odd"]
    s = oddness_scores(odd)
    assert s["is_odd"] and s["n_odd_edges"] == 1
    assert s["max_edge_nm"] == pytest.approx(50_000.0)
    assert flag_odd_fragments([normal, odd]) == {2}


def test_oddness_flags_disconnected_skeleton():
    # two chains, no bridge edge at all (the synthetic-frankenmerge shape):
    # no odd edge, but 2 components → odd
    verts = [(0, 0, 0), (1000, 0, 0), (52000, 0, 0), (53000, 0, 0)]
    f = frag_from(verts, [(0, 1), (2, 3)], fid=3)
    s = oddness_scores(f)
    assert s["n_odd_edges"] == 0
    assert s["n_components"] == 2
    assert s["is_odd"]


# ---------------------------------------------------------------------------
# split_fragment
# ---------------------------------------------------------------------------

def test_split_at_branches_yields_three_atoms():
    atoms = split_fragment(y_fragment(), at_branches=True)
    assert len(atoms) == 3
    sizes = sorted(len(a["edges"]) for a in atoms)
    assert sizes == [1, 1, 2]  # two single-edge arms + the 0-1-2 chain


def test_split_no_branches_keeps_whole():
    atoms = split_fragment(y_fragment(), at_branches=False)
    assert len(atoms) == 1
    assert len(atoms[0]["edges"]) == 4


def test_cut_odd_edges_separates_bridge():
    # branch-only split does NOT separate the bridge (degree-2 path) …
    assert len(split_fragment(bridged_fragment(), at_branches=True,
                              cut_odd_edges=False)) == 1
    # … cutting odd edges does
    atoms = split_fragment(bridged_fragment(), at_branches=False,
                           cut_odd_edges=True)
    assert len(atoms) == 2
    assert sorted(len(a["vertices_nm"]) for a in atoms) == [3, 3]


def test_split_isolated_vertices_become_singletons():
    verts = [(0, 0, 0), (1000, 0, 0), (99000, 0, 0)]
    f = frag_from(verts, [(0, 1)])
    atoms = split_fragment(f, at_branches=True)
    assert len(atoms) == 2
    assert sorted(len(a["vertices_nm"]) for a in atoms) == [1, 2]


# ---------------------------------------------------------------------------
# atomize_world
# ---------------------------------------------------------------------------

def make_world(frags, obs_pos, obs_seg, obs_label):
    obs_pos = np.asarray(obs_pos, dtype=np.float32)
    n = len(obs_pos)
    region = Region(
        region_id="test",
        bbox_nm=((-1e5, -1e5, -1e5), (1e6, 1e6, 1e6)),
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1718,
        pre_pt_nm=obs_pos,
        post_pt_nm=obs_pos.copy(),
        pre_root_id=np.asarray(obs_label, dtype=np.int64),
        post_root_id=np.zeros(n, dtype=np.int64),
        synapse_id=np.arange(n, dtype=np.int64),
        pre_seg_id=np.asarray(obs_seg, dtype=np.int64),
        post_seg_id=np.zeros(n, dtype=np.int64),
    ).validate()
    return frags, region


def test_atomize_world_shatter_separates_franken_halves():
    f = bridged_fragment(fid=7)
    # 2 obs on the left chain (label 100), 2 on the right chain (label 200)
    obs_pos = [(500, 0, 0), (1500, 0, 0), (52500, 0, 0), (53500, 0, 0)]
    frags, region = make_world([f], obs_pos, [7, 7, 7, 7], [100, 100, 200, 200])

    aw = atomize_world(frags, region, mode="shatter")
    assert len(aw.fragments) == 2
    seg = aw.region.pre_seg_id
    # halves get different atom ids, and labels align with the split
    assert seg[0] == seg[1] and seg[2] == seg[3] and seg[0] != seg[2]
    assert all(aw.atom_parent[int(a)] == 7 for a in np.unique(seg))
    assert aw.root_label_map[int(seg[0])] == {100}
    assert aw.root_label_map[int(seg[2])] == {200}
    assert (aw.parent_ids_per_obs == 7).all()
    assert aw.odd_parents == {7}


def test_atomize_world_odd_mode_touches_only_odd_fragments():
    odd = bridged_fragment(fid=7)
    normal = y_fragment(fid=8)
    obs_pos = [(500, 0, 0), (53500, 0, 0), (1000, 0, 0), (3000, 1000, 0)]
    frags, region = make_world([odd, normal], obs_pos,
                               [7, 7, 8, 8], [100, 200, 300, 300])
    aw = atomize_world(frags, region, mode="odd")
    # odd fragment split into 2 atoms; normal (branchy) fragment left whole
    parents = [aw.atom_parent[int(f.base_root_id)] for f in aw.fragments]
    assert parents.count(7) == 2 and parents.count(8) == 1


def test_atomize_world_branch_mode_leaves_bridge_glued():
    f = bridged_fragment(fid=7)
    obs_pos = [(500, 0, 0), (53500, 0, 0)]
    frags, region = make_world([f], obs_pos, [7, 7], [100, 200])
    aw = atomize_world(frags, region, mode="branch")
    # no branch vertices → single atom, halves stay glued (the failure the
    # user predicted for branch-only atomization)
    assert len(aw.fragments) == 1
    seg = aw.region.pre_seg_id
    assert seg[0] == seg[1]


def test_atomize_world_drops_obs_free_atoms():
    f = y_fragment(fid=7)  # 3 atoms after branch split
    obs_pos = [(500, 0, 0)]  # only the 0-1-2 chain gets an observation
    frags, region = make_world([f], obs_pos, [7], [100])
    aw = atomize_world(frags, region, mode="branch")
    assert len(aw.fragments) == 1  # arm atoms dropped
    assert len(aw.fragments[0].synapse_indices) == 1


# ---------------------------------------------------------------------------
# frankenmerge_separation
# ---------------------------------------------------------------------------

def test_frankenmerge_separation_metric():
    # parent 7 is a frankenmerge (labels 100/200); parent 8 is clean
    parents = np.array([7, 7, 7, 7, 8, 8])
    true = np.array([100, 100, 200, 200, 300, 300])

    separated = np.array([0, 0, 1, 1, 2, 2])
    m = frankenmerge_separation(separated, true, parents)
    assert m["n_frankenmerges"] == 1
    assert m["fk_separation"] == pytest.approx(1.0)

    glued = np.array([0, 0, 0, 1, 2, 2])  # cluster 0 spans both labels
    m2 = frankenmerge_separation(glued, true, parents)
    assert m2["fk_separation"] == pytest.approx(0.0)


def test_frankenmerge_separation_ignores_abstained():
    parents = np.array([7, 7, 7])
    true = np.array([100, 200, 200])
    pred = np.array([0, -1, 1])  # abstained obs doesn't glue
    m = frankenmerge_separation(pred, true, parents)
    assert m["fk_separation"] == pytest.approx(1.0)
