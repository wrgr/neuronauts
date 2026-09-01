"""Neuroglancer state builder: nm coordinates, layers, URL round-trip, atom view."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neuronauts.harness.geometry import AtomGeometryStore
from neuronauts.report import ngl


def test_state_is_in_nanometres_and_round_trips():
    st = ngl.NglState(seg_timestamp=ngl.V117_TIMESTAMP)
    st.add_points("pts", [[1000, 2000, 3000]], descriptions=["one"])
    st.add_lines("ln", [[0, 0, 0]], [[10, 10, 10]])
    st.add_box("box", [0, 0, 0], [500, 500, 500])
    st.select_segments([864691135361314119])
    st.look_at([100, 200, 300], 30_000)
    d = st.to_dict()
    assert d["dimensions"] == {"x": [1e-9, "m"], "y": [1e-9, "m"], "z": [1e-9, "m"]}
    assert d["position"] == [100.0, 200.0, 300.0]
    names = [l["name"] for l in d["layers"]]
    assert names == ["em", "segmentation@1623399000", "pts", "ln", "box"]
    seg = d["layers"][1]
    assert seg["segments"] == ["864691135361314119"] and seg["timestamp"] == 1623399000
    pts = d["layers"][2]
    assert pts["source"]["transform"]["outputDimensions"] == ngl.DIMENSIONS_NM
    assert pts["annotations"][0] == {"type": "point", "id": "pts-0",
                                     "point": [1000.0, 2000.0, 3000.0], "description": "one"}
    assert d["layers"][4]["annotations"][0]["type"] == "axis_aligned_bounding_box"
    url = st.to_url("demo")
    assert url.startswith(ngl.VIEWERS["demo"])
    assert ngl.url_to_state(url) == json.loads(json.dumps(d))


def test_neuroglancer_package_accepts_state():
    pytest.importorskip("neuroglancer")
    st = ngl.NglState(seg_timestamp=ngl.V117_TIMESTAMP)
    st.add_points("pts", np.arange(30).reshape(10, 3) * 100.0,
                  colors=["#ff0000"] * 10)
    st.add_box("box", [0, 0, 0], [500, 500, 500])
    st.select_segments([42])
    back = ngl.validate(st.to_dict())
    layers = {l["name"]: l for l in back["layers"]}
    assert len(layers["pts"]["annotations"]) == 10
    assert layers["pts"]["annotationProperties"][0]["id"] == "color"
    assert layers["segmentation@1623399000"]["segments"] == ["42"]
    assert back["dimensions"]["x"] == [1e-9, "m"]


def test_annotation_cap_is_deterministic_and_labelled():
    st = ngl.NglState(show_em=False, show_seg=False, max_annotations=5)
    st.add_points("many", np.random.default_rng(1).random((50, 3)))
    layer = st.to_dict()["layers"][0]
    assert layer["name"] == "many (5 of 50)" and len(layer["annotations"]) == 5
    again = ngl.NglState(show_em=False, show_seg=False, max_annotations=5)
    again.add_points("many", np.random.default_rng(1).random((50, 3)))
    assert again.to_dict()["layers"][0]["annotations"] == layer["annotations"]


def test_large_state_has_no_url():
    st = ngl.NglState(show_em=False, show_seg=False, max_annotations=100_000)
    st.add_points("many", np.zeros((30_000, 3)))
    assert st.to_url() is None


def test_region_and_experiment_views():
    r = ngl.region_view([663, 591, 860], 100).to_dict()
    box = r["layers"][2]["annotations"][0]
    assert box["pointA"] == [613_000.0, 541_000.0, 810_000.0]
    assert box["pointB"] == [713_000.0, 641_000.0, 910_000.0]
    assert r["position"] == [663_000.0, 591_000.0, 860_000.0]

    class Rec:
        id = "EXP-1"
        provenance = {"bbox_nm": [[0, 0, 0], [30000, 30000, 30000]],
                      "anchor_soma_nm": [1, 2, 3], "anchor_target_root": 7}
    e = ngl.experiment_view(Rec()).to_dict()
    assert e["position"] == [15000.0, 15000.0, 15000.0]
    assert e["layers"][1]["segments"] == ["7"]
    assert [l["name"] for l in e["layers"][2:]] == ["experiment bbox", "anchor soma"]

    class Bare:
        provenance = {"git_commit": "x"}
    assert ngl.experiment_view(Bare()) is None


def test_atom_view_from_synthetic_store(tmp_path):
    """A 5-node path plus an isolated node; one node lacks coordinates."""
    store = AtomGeometryStore(tmp_path / "geom")
    l2 = np.array([100, 101, 102, 103, 104, 105], np.uint64)
    edges = np.array([[100, 101], [101, 102], [102, 103], [103, 104]], np.uint64)
    store.write_shard("k10_00000", [
        {"atom": 5, "l2_ids": l2, "edges": edges},
        {"atom": 6, "l2_ids": np.array([200], np.uint64), "edges": np.zeros((0, 2), np.uint64)},
    ])
    ids = np.array([105, 103, 100, 101, 102, 200], np.uint64)        # 104 has no coords
    pos = np.array([[9, 9, 9], [3, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0], [7, 7, 7]], np.float32) * 1000
    np.savez_compressed(tmp_path / "geom" / "l2_attributes.npz", l2_id=ids, pos_nm=pos)
    np.savez_compressed(tmp_path / "k10.npz",
                        ep_atom=np.array([5, 5, 6], np.uint64),
                        ep_pos_nm=np.array([[0, 0, 0], [3000, 0, 0], [7000, 7000, 7000]], np.float32),
                        ep_seg_len_nm=np.array([3000, 3000, 0], np.float32),
                        ep_caliber_nm=np.array([20, 30, 0], np.float32))
    np.savez_compressed(tmp_path / "pop.npz",
                        syn_atom_pre=np.array([5, 6], np.uint64),
                        syn_atom_post=np.array([6, 5], np.uint64),
                        syn_ctr_nm=np.array([[1, 1, 1], [2, 2, 2]], np.float32))

    positions = ngl.L2Positions(tmp_path / "geom" / "l2_attributes.npz")
    st, summary = ngl.atom_view(5, geom_dir=tmp_path / "geom", positions=positions,
                                topology_npz=tmp_path / "k10.npz",
                                population_npz=tmp_path / "pop.npz")
    assert summary == {"atom": 5, "n_l2": 6, "n_edges": 4, "n_l2_with_coords": 5,
                       "n_edges_drawn": 3, "shard": "k10_00000.npz",
                       "n_endpoints": 2, "n_pre": 1, "n_post": 1}
    d = st.to_dict()
    layers = {l["name"]: l for l in d["layers"]}
    assert layers["segmentation@1623399000"]["segments"] == ["5"]
    skel = layers["L2 skeleton"]["annotations"]
    assert len(skel) == 3
    assert {tuple(a["pointA"]) for a in skel} == {(0, 0, 0), (1000, 0, 0), (2000, 0, 0)}
    assert len(layers["endpoints"]["annotations"]) == 2
    assert layers["endpoints"]["annotations"][0]["description"] == "leaf 3000 nm, caliber 20 nm"
    assert layers["synapses (atom is pre)"]["annotations"][0]["point"] == [1.0, 1.0, 1.0]
    # view centred on the coordinate-bearing nodes only
    assert d["position"] == [4500.0, 4500.0, 4500.0]

    with pytest.raises(KeyError):
        ngl.atom_view(99, geom_dir=tmp_path / "geom", positions=positions)


def test_large_states_are_written_compactly(tmp_path):
    small = ngl.NglState(show_em=False, show_seg=False)
    small.add_points("few", np.zeros((3, 3)))
    assert "\n" in small.to_json()

    big = ngl.NglState(show_em=False, show_seg=False, max_annotations=2000)
    big.add_points("many", np.zeros((2000, 3)))
    text = big.to_json()
    assert "\n" not in text
    path = big.save(tmp_path / "big.json")
    assert json.loads(path.read_text()) == big.to_dict()
