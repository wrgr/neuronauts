"""Tests for treestitch.stitch_viz — Neuroglancer views of stitch products."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from neuronauts.schemas import Fragment
from treestitch.stitch import StitchCandidate, StitchResult, SuperFragment
from treestitch.stitch_viz import (
    export_stitch_viz,
    frankenmerge_state,
    odd_fragments_state,
    stitch_edges_state,
    stitch_overview_state,
)


def make_line_fragment(fid, start, direction, n=5, step=1_000.0):
    start = np.asarray(start, dtype=np.float32)
    d = np.asarray(direction, dtype=np.float32)
    d = d / (np.linalg.norm(d) + 1e-9)
    verts = np.stack([start + d * step * i for i in range(n)]).astype(np.float32)
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    return Fragment(
        fragment_id=fid, region_id="test", base_root_id=fid,
        vertices_nm=verts, edges=edges, endpoints_nm=verts[[0, -1]],
        radius_nm=np.full(n, 100.0, dtype=np.float32),
        synapse_indices=np.zeros(0, dtype=np.int64), dna=None,
    ).validate()


def make_super(tile_id, cluster_id, frag, *, obs_keys=(), majority_label=0):
    keys = np.asarray(list(obs_keys), dtype=np.int64)
    return SuperFragment(
        tile_id=tile_id, cluster_id=cluster_id,
        atom_ids=frozenset([frag.base_root_id]), skeleton=frag, dna=None,
        n_obs=max(len(keys), 1), obs_keys=keys,
        majority_label=majority_label,
    )


def _mini_world():
    fa = make_line_fragment(1, (0, 0, 0), (1, 0, 0))
    fb = make_line_fragment(2, (6_000, 0, 0), (1, 0, 0))
    a = make_super("A", 0, fa, obs_keys=[0, 1], majority_label=7)
    b = make_super("B", 0, fb, obs_keys=[2, 3], majority_label=7)
    c = make_super("B", 1, fb, obs_keys=[4], majority_label=8)
    cand_ok = StitchCandidate(score=0.9, i=0, j=1, ep_i=1, ep_j=0,
                              gap_nm=2000.0, dna_cos=0.5)
    cand_bad = StitchCandidate(score=0.7, i=0, j=2, ep_i=0, ep_j=1,
                               gap_nm=3000.0, dna_cos=0.1)
    res = StitchResult(
        super_cluster=np.array([0, 0, 1]),
        accepted=[cand_ok, cand_bad],
        forced_pairs=[(0, 1)],
    )
    obs_pos = np.array([[0, 0, 0], [1000, 0, 0], [6000, 0, 0],
                        [7000, 0, 0], [8000, 0, 0]], dtype=np.float64)
    glab = np.array([0, 0, 0, 0, 1])
    return [a, b, c], res, [fa, fb], obs_pos, glab


def _layer(state, name):
    return next(l for l in state["layers"] if l["name"] == name)


def test_overview_state_layers_and_colors():
    supers, res, frags, pos, glab = _mini_world()
    st = stitch_overview_state(supers, res.super_cluster, pos, glab)
    obs = _layer(st, "observations_by_global_cluster")["annotations"]
    assert len(obs) == 5
    # same global cluster → same colour; different → different
    assert obs[0]["props"] == obs[3]["props"]
    assert obs[0]["props"] != obs[4]["props"]
    skel = _layer(st, "super_skeletons")["annotations"]
    assert len(skel) > 0
    # EM + seg base layers present
    assert {l["name"] for l in st["layers"]} >= {"EM", "seg"}


def test_stitch_edges_state_correctness_colors():
    supers, res, *_ = _mini_world()
    st = stitch_edges_state(supers, res)
    ann = _layer(st, "stitch_decisions")["annotations"]
    by_id = {a["id"]: a for a in ann}
    assert by_id["acc0"]["props"][0].startswith("rgba(0,220,0")   # 7↔7 correct
    assert by_id["acc1"]["props"][0].startswith("rgba(255,40,40")  # 7↔8 wrong
    assert by_id["forced0"]["props"][0].startswith("rgba(60,120,255")


def test_odd_fragments_state_separates_colors():
    _, _, frags, *_ = _mini_world()
    st = odd_fragments_state(frags, odd_parents={1})
    ann = _layer(st, "odd_vs_normal_fragments")["annotations"]
    colors = {a["props"][0] for a in ann}
    assert any(c.startswith("rgba(255,150,0") for c in colors)  # odd
    assert any(c.startswith("rgba(0,200,200") for c in colors)  # normal


def test_frankenmerge_state_only_multilabel_parents():
    pos = np.array([[0, 0, 0], [1000, 0, 0], [2000, 0, 0], [3000, 0, 0]],
                   dtype=np.float64)
    true = np.array([7, 8, 9, 9])
    par = np.array([1, 1, 2, 2])   # parent 1 is a frankenmerge; 2 is clean
    pred = np.array([0, 1, 2, 2])
    st = frankenmerge_state(pos, true, par, pred)
    ann = _layer(st, "frankenmerge_parents")["annotations"]
    assert len(ann) == 2                      # only parent 1's observations
    assert all(a["id"].startswith("fk1_") for a in ann)
    assert "true=7" in ann[0]["description"]


def test_export_stitch_viz_writes_files(tmp_path):
    supers, res, frags, pos, glab = _mini_world()
    urls = export_stitch_viz(
        str(tmp_path), supers=supers, result=res, fragments=frags,
        odd_parents={1}, obs_pos_nm=pos, obs_global_labels=glab,
        true_labels=np.array([7, 7, 7, 7, 8]),
        parent_ids=np.array([1, 1, 2, 2, 2]),
    )
    assert set(urls) == {"overview", "stitch_edges", "odd_fragments",
                         "frankenmerges"}
    for name in urls:
        assert urls[name].startswith("https://")
        p = tmp_path / f"{name}.json"
        assert p.exists()
        json.loads(p.read_text())             # valid JSON
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "urls.txt").exists()
    html = (tmp_path / "index.html").read_text()
    assert "overview" in html and "stitch_edges" in html
