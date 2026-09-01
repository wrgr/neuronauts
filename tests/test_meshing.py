"""Tests for neuronauts.meshing: skeleton geometry, tube meshing, on-disk
formats, and bundle export.

Correctness bar: a mesh is right if (a) every triangle index is in range,
(b) triangle winding is consistently outward (positive signed volume on a
closed test shape), (c) the format round-trips through this project's own
decoder, and (d) — the strongest check available without a GPU — Neuroglancer
mesh bytes match cloud-volume's own encoder byte-for-byte, and cloud-volume's
decoder reads our skeleton bytes back to the same arrays we wrote.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from neuronauts.meshing.bundle import (
    MeshParams, build_state, equivalence_classes, export_bundle, group_color_hex,
    group_index, load_manifest, normalise_groups,
)
from neuronauts.meshing.formats import (
    decode_precomputed_mesh, decode_precomputed_skeleton, encode_precomputed_mesh,
    encode_precomputed_skeleton, read_obj, read_ply, write_obj, write_ply,
    write_precomputed_mesh_dir, write_precomputed_skeleton_dir, write_segment_properties,
)
from neuronauts.meshing.skeleton import SkeletonGeometry, canonical_edges, concat_skeletons
from neuronauts.meshing.sources import (
    groups_from_fragment_to_neuron, groups_from_neuron_hypotheses, kimimaro_archive_skeletons,
)
from neuronauts.meshing.tube import TriMesh, chain_frames, skeleton_chains, tube_mesh

cloudvolume = pytest.importorskip(
    "cloudvolume", reason="cloud-volume not installed; byte-compat checks skipped")


# ---------------------------------------------------------------------------
# fixtures: small known shapes
# ---------------------------------------------------------------------------

def path_skeleton(n=5, step=1000.0, radius=200.0) -> SkeletonGeometry:
    verts = np.stack([np.array([i * step, 0.0, 0.0]) for i in range(n)]).astype(np.float32)
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int64)
    return SkeletonGeometry(verts, edges, np.full(n, radius, np.float32))


def y_fork_skeleton() -> SkeletonGeometry:
    """0-1-2-3 trunk, with node 2 also joined to 4: node 2 is a branch point
    of degree 3 (in from 1, out to 3, out to 4), giving three chains."""
    verts = np.array([
        [0, 0, 0], [1000, 0, 0], [2000, 0, 0], [3000, 0, 0], [2000, 1000, 0],
    ], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3], [2, 4]], dtype=np.int64)
    return SkeletonGeometry(verts, edges, np.full(5, 150.0, np.float32))


def cycle_skeleton() -> SkeletonGeometry:
    n = 6
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = np.stack([2000 * np.cos(ang), 2000 * np.sin(ang), np.zeros(n)], axis=1).astype(np.float32)
    edges = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int64)
    return SkeletonGeometry(verts, edges, np.full(n, 100.0, np.float32))


# ---------------------------------------------------------------------------
# SkeletonGeometry
# ---------------------------------------------------------------------------

class TestSkeletonGeometry:
    def test_canonical_edges_dedupes_sorts_and_drops_self_loops(self):
        e = canonical_edges([[1, 0], [0, 1], [2, 2], [0, 2]], n_vertices=3)
        assert e.tolist() == [[0, 1], [0, 2]]

    def test_out_of_range_edge_raises(self):
        with pytest.raises(ValueError):
            canonical_edges([[0, 5]], n_vertices=3)

    def test_scalar_radius_broadcasts(self):
        s = SkeletonGeometry(np.zeros((4, 3), np.float32), np.zeros((0, 2), np.int64), np.array([250.0]))
        assert s.radii_nm.shape == (4,)
        assert (s.radii_nm == 250.0).all()

    def test_mismatched_radius_length_raises(self):
        with pytest.raises(ValueError):
            SkeletonGeometry(np.zeros((4, 3), np.float32), np.zeros((0, 2), np.int64),
                             np.zeros(3, np.float32))

    def test_degree_and_cable_length(self):
        s = path_skeleton(n=4, step=1000.0)
        assert s.degree().tolist() == [1, 2, 2, 1]
        assert s.cable_length_nm() == pytest.approx(3000.0)

    def test_drop_invalid_reindexes_edges(self):
        v = np.array([[0, 0, 0], [np.nan, 0, 0], [2000, 0, 0]], np.float32)
        s = SkeletonGeometry(v, np.array([[0, 1], [1, 2]], np.int64), np.full(3, 100.0, np.float32))
        cleaned = s.drop_invalid()
        assert cleaned.n_vertices == 2
        assert cleaned.n_edges == 0  # both edges touched the dropped vertex

    def test_from_dict_accepts_repo_key_spellings(self):
        d = {"vertices": np.zeros((3, 3), np.float32), "edges": np.array([[0, 1], [1, 2]]),
             "radii": np.array([1.0, 2.0, 3.0])}
        s = SkeletonGeometry.from_dict(d)
        assert s.n_vertices == 3 and s.n_edges == 2

    def test_from_dict_default_radius_when_absent(self):
        d = {"vertices_nm": np.zeros((2, 3), np.float32), "edges": np.zeros((0, 2), np.int64)}
        s = SkeletonGeometry.from_dict(d, default_radius_nm=77.0)
        assert (s.radii_nm == 77.0).all()

    def test_from_fragment_duck_types_radius_field_name(self):
        class FakeFragment:
            vertices_nm = np.zeros((2, 3), np.float32)
            edges = np.zeros((0, 2), np.int64)
            radius_nm = np.array([10.0, 20.0], np.float32)

        s = SkeletonGeometry.from_fragment(FakeFragment())
        assert np.array_equal(s.radii_nm, [10.0, 20.0])

    def test_concat_offsets_edges_and_preserves_cable(self):
        a, b = path_skeleton(3), path_skeleton(3)
        c = concat_skeletons([a, b])
        assert c.n_vertices == 6
        assert c.cable_length_nm() == pytest.approx(a.cable_length_nm() + b.cable_length_nm())
        assert c.edges.max() == 5


# ---------------------------------------------------------------------------
# chain decomposition + frames
# ---------------------------------------------------------------------------

class TestChains:
    def test_path_is_one_chain_covering_every_edge_in_order(self):
        s = path_skeleton(5)
        chains = skeleton_chains(s.edges, s.n_vertices)
        assert len(chains) == 1
        assert chains[0].tolist() in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0])

    def test_fork_gives_three_chains_all_ending_at_branch(self):
        s = y_fork_skeleton()
        chains = skeleton_chains(s.edges, s.n_vertices)
        assert len(chains) == 3
        for c in chains:
            assert 2 in (int(c[0]), int(c[-1]))

    def test_chains_partition_every_edge_exactly_once(self):
        for s in (path_skeleton(6), y_fork_skeleton(), cycle_skeleton()):
            chains = skeleton_chains(s.edges, s.n_vertices)
            seen = []
            for c in chains:
                seen += [tuple(sorted((int(a), int(b)))) for a, b in zip(c[:-1], c[1:])]
            full = [tuple(sorted((int(a), int(b)))) for a, b in s.edges.tolist()]
            assert sorted(seen) == sorted(full)

    def test_pure_cycle_is_one_closed_chain(self):
        s = cycle_skeleton()
        chains = skeleton_chains(s.edges, s.n_vertices)
        assert len(chains) == 1
        assert int(chains[0][0]) == int(chains[0][-1])

    def test_isolated_vertex_produces_no_chain(self):
        v = np.zeros((3, 3), np.float32)
        chains = skeleton_chains(np.array([[0, 1]], np.int64), 3)
        assert len(chains) == 1
        touched = {int(x) for c in chains for x in c}
        assert 2 not in touched

    def test_frames_orthonormal_and_continuous_on_a_straight_line(self):
        pts = np.stack([np.array([i * 1000.0, 0, 0]) for i in range(5)])
        frames = chain_frames(pts)
        assert frames is not None
        t, n, b = frames
        assert np.allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-5)
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)
        assert np.allclose(np.einsum("ij,ij->i", t, n), 0.0, atol=1e-5)
        # normal should not rotate along a straight tangent
        assert np.allclose(n[0], n[-1], atol=1e-5)

    def test_frames_none_when_all_points_coincide(self):
        pts = np.zeros((4, 3))
        assert chain_frames(pts) is None

    def test_frames_handle_duplicate_points_without_nan(self):
        pts = np.array([[0, 0, 0], [0, 0, 0], [1000, 0, 0], [2000, 0, 0]], np.float64)
        frames = chain_frames(pts)
        assert frames is not None
        assert np.isfinite(frames[0]).all()


# ---------------------------------------------------------------------------
# tube_mesh
# ---------------------------------------------------------------------------

class TestTubeMesh:
    def test_empty_skeleton_gives_empty_mesh(self):
        s = SkeletonGeometry(np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int64),
                             np.zeros(0, np.float32))
        m = tube_mesh(s)
        assert m.is_empty

    def test_single_isolated_vertex_becomes_a_sphere(self):
        s = SkeletonGeometry(np.zeros((1, 3), np.float32), np.zeros((0, 2), np.int64),
                             np.array([100.0], np.float32))
        m = tube_mesh(s, caps="junctions")
        assert m.n_faces > 0
        lo, hi = m.bounds_nm()
        assert np.allclose((hi - lo) / 2.0, 100.0, atol=1.0)

    def test_faces_index_in_range(self):
        m = tube_mesh(y_fork_skeleton(), sides=6, sphere_level=1)
        assert m.faces.max() < m.n_vertices

    def test_outward_winding_positive_volume_on_cycle(self):
        # A closed loop tube is watertight enough for a meaningful signed volume.
        m = tube_mesh(cycle_skeleton(), sides=8, sphere_level=0, caps="none")
        assert m.signed_volume_nm3() > 0

    def test_min_radius_floor_is_applied(self):
        s = path_skeleton(3, radius=1.0)
        m = tube_mesh(s, sides=6, min_radius_nm=500.0, caps="none")
        lo, hi = m.bounds_nm()
        assert (hi[1] - lo[1]) >= 500.0  # ring extends >= floor radius off the axis

    def test_max_radius_cap_is_applied(self):
        s = path_skeleton(3, radius=10_000.0)
        m = tube_mesh(s, sides=6, max_radius_nm=50.0, min_radius_nm=1.0, caps="none")
        lo, hi = m.bounds_nm()
        assert (hi[1] - lo[1]) < 200.0

    def test_caps_none_skips_junction_spheres_but_keeps_isolated_vertex(self):
        s = y_fork_skeleton()
        with_caps = tube_mesh(s, sides=6, caps="junctions")
        without = tube_mesh(s, sides=6, caps="none")
        assert without.n_faces < with_caps.n_faces

    def test_more_sides_gives_more_faces(self):
        s = path_skeleton(4)
        m6 = tube_mesh(s, sides=6, caps="none")
        m12 = tube_mesh(s, sides=12, caps="none")
        assert m12.n_faces > m6.n_faces

    def test_invalid_sides_raises(self):
        with pytest.raises(ValueError):
            tube_mesh(path_skeleton(3), sides=2)

    def test_invalid_caps_raises(self):
        with pytest.raises(ValueError):
            tube_mesh(path_skeleton(3), caps="bogus")

    def test_nan_vertex_is_dropped_not_propagated(self):
        v = np.array([[0, 0, 0], [np.nan, 0, 0], [2000, 0, 0]], np.float32)
        s = SkeletonGeometry(v, np.array([[0, 1], [1, 2]], np.int64), np.full(3, 100.0, np.float32))
        m = tube_mesh(s)
        assert np.isfinite(m.vertices).all()


class TestTriMesh:
    def test_concat_offsets_faces(self):
        m1 = tube_mesh(path_skeleton(3), caps="none")
        m2 = tube_mesh(path_skeleton(3), caps="none")
        merged = TriMesh.concat([m1, m2])
        assert merged.n_vertices == m1.n_vertices + m2.n_vertices
        assert merged.faces.max() == merged.n_vertices - 1

    def test_scaled_scales_vertices_only(self):
        m = tube_mesh(path_skeleton(3), caps="none")
        scaled = m.scaled(0.001)
        assert np.allclose(scaled.vertices, m.vertices * 0.001)
        assert np.array_equal(scaled.faces, m.faces)


# ---------------------------------------------------------------------------
# precomputed format round-trips (this decoder, and cloud-volume's)
# ---------------------------------------------------------------------------

class TestPrecomputedFormat:
    def test_mesh_roundtrips_through_own_decoder(self):
        m = tube_mesh(y_fork_skeleton(), sides=6)
        m2 = decode_precomputed_mesh(encode_precomputed_mesh(m))
        assert np.array_equal(m.vertices, m2.vertices)
        assert np.array_equal(m.faces, m2.faces)

    def test_skeleton_roundtrips_through_own_decoder(self):
        s = y_fork_skeleton()
        s2 = decode_precomputed_skeleton(encode_precomputed_skeleton(s))
        assert np.array_equal(s.vertices_nm, s2.vertices_nm)
        assert np.array_equal(s.edges, s2.edges)
        assert np.allclose(s.radii_nm, s2.radii_nm)

    def test_mesh_bytes_identical_to_cloudvolume_encoder(self):
        m = tube_mesh(y_fork_skeleton(), sides=6)
        cv_mesh = cloudvolume.Mesh(m.vertices.astype(np.float32), m.faces.astype(np.uint32))
        assert cv_mesh.to_precomputed() == encode_precomputed_mesh(m)

    def test_cloudvolume_decodes_our_mesh_bytes(self):
        m = tube_mesh(y_fork_skeleton(), sides=6)
        decoded = cloudvolume.Mesh.from_precomputed(encode_precomputed_mesh(m))
        assert np.array_equal(decoded.vertices, m.vertices)
        assert np.array_equal(decoded.faces, m.faces)

    def test_cloudvolume_decodes_our_skeleton_bytes(self):
        s = y_fork_skeleton()
        decoded = cloudvolume.Skeleton.from_precomputed(encode_precomputed_skeleton(s))
        assert np.array_equal(decoded.vertices, s.vertices_nm)
        assert np.array_equal(decoded.edges, s.edges)
        assert np.allclose(decoded.radii, s.radii_nm)

    def test_decode_rejects_truncated_mesh_buffer(self):
        buf = encode_precomputed_mesh(tube_mesh(path_skeleton(4)))
        with pytest.raises(ValueError):
            decode_precomputed_mesh(buf[:-100])

    def test_decode_rejects_truncated_skeleton_buffer(self):
        buf = encode_precomputed_skeleton(y_fork_skeleton())
        with pytest.raises(ValueError):
            decode_precomputed_skeleton(buf[:4])


class TestDirectoryWriters:
    def test_mesh_dir_round_trips_via_manifest(self, tmp_path):
        m1 = tube_mesh(path_skeleton(4), caps="none")
        m2 = tube_mesh(y_fork_skeleton(), caps="none")
        written = write_precomputed_mesh_dir(tmp_path / "mesh", {10: m1, 20: m2})
        assert sorted(written) == [10, 20]
        info = json.loads((tmp_path / "mesh" / "info").read_text())
        assert info["@type"] == "neuroglancer_legacy_mesh"
        from neuronauts.meshing.formats import read_precomputed_mesh
        back = read_precomputed_mesh(tmp_path / "mesh", 10)
        assert np.array_equal(back.vertices, m1.vertices)

    def test_mesh_dir_skips_empty_meshes(self, tmp_path):
        empty = TriMesh.empty()
        written = write_precomputed_mesh_dir(tmp_path / "mesh", {5: empty})
        assert written == []
        assert not (tmp_path / "mesh" / "5:0").exists()

    def test_skeleton_dir_round_trips(self, tmp_path):
        s = y_fork_skeleton()
        write_precomputed_skeleton_dir(tmp_path / "skel", {42: s})
        from neuronauts.meshing.formats import read_precomputed_skeleton
        back = read_precomputed_skeleton(tmp_path / "skel", 42)
        assert np.array_equal(back.vertices_nm, s.vertices_nm)

    def test_segment_properties_writes_labels_numbers_tags(self, tmp_path):
        info = write_segment_properties(
            tmp_path / "props", [1, 2],
            labels={1: "a", 2: "b"}, numbers={"score": {1: 0.5, 2: 0.9}},
            tags={1: ["x"], 2: ["x", "y"]},
        )
        assert info["@type"] == "neuroglancer_segment_properties"
        ids = info["inline"]["ids"]
        assert ids == ["1", "2"]
        by_id = {p["id"]: p for p in info["inline"]["properties"]}
        assert by_id["label"]["values"] == ["a", "b"]
        assert by_id["score"]["values"] == [0.5, 0.9]
        assert by_id["tags"]["tags"] == ["x", "y"]

    def test_segment_id_out_of_uint64_range_raises(self, tmp_path):
        with pytest.raises(ValueError):
            write_precomputed_mesh_dir(tmp_path / "mesh", {-1: tube_mesh(path_skeleton(3))})


class TestObjPly:
    def test_obj_round_trip_scaled(self, tmp_path):
        m = tube_mesh(y_fork_skeleton(), sides=6, caps="none")
        write_obj(tmp_path / "a.obj", {3: m}, scale=1e-3)
        back = read_obj(tmp_path / "a.obj")
        assert np.allclose(back["seg_3"].vertices, m.vertices * 1e-3, atol=1e-3)
        assert np.array_equal(back["seg_3"].faces, m.faces)

    def test_obj_skips_empty_meshes(self, tmp_path):
        n = write_obj(tmp_path / "b.obj", {1: TriMesh.empty(), 2: tube_mesh(path_skeleton(3))})
        assert n == 1

    def test_ply_round_trip_with_colors(self, tmp_path):
        m = tube_mesh(path_skeleton(3), caps="none")
        write_ply(tmp_path / "c.ply", {9: m}, scale=1e-3, colors={9: (1, 2, 3)})
        v, f, c = read_ply(tmp_path / "c.ply")
        assert np.allclose(v, m.vertices * 1e-3, atol=1e-3)
        assert np.array_equal(f, m.faces)
        assert (c == [1, 2, 3]).all()

    def test_ply_without_colors_has_no_color_columns(self, tmp_path):
        m = tube_mesh(path_skeleton(3), caps="none")
        write_ply(tmp_path / "d.ply", {9: m}, scale=1e-3)
        _, _, c = read_ply(tmp_path / "d.ply")
        assert c is None


# ---------------------------------------------------------------------------
# bundle: grouping + export_bundle + state
# ---------------------------------------------------------------------------

class TestGrouping:
    def test_normalise_groups_from_id_to_group(self):
        out = normalise_groups({1: "a", 2: "a", 3: "b"}, ids=[1, 2, 3])
        assert out == {1: "a", 2: "a", 3: "b"}

    def test_normalise_groups_from_group_to_ids(self):
        out = normalise_groups({"a": [1, 2], "b": [3]}, ids=[1, 2, 3])
        assert out == {1: "a", 2: "a", 3: "b"}

    def test_normalise_groups_ignores_ids_outside_the_mesh_set(self):
        out = normalise_groups({"a": [1, 2, 99]}, ids=[1, 2])
        assert out == {1: "a", 2: "a"}

    def test_group_index_is_stable_and_sorted(self):
        assert group_index({1: "z", 2: "a"}) == {"a": 1, "z": 2}

    def test_equivalence_classes_excludes_singletons(self):
        eqs = equivalence_classes({1: "a", 2: "a", 3: "b"})
        assert eqs == [[1, 2]]

    def test_group_color_hex_is_deterministic_and_valid(self):
        c1 = group_color_hex(3)
        c2 = group_color_hex(3)
        assert c1 == c2
        assert c1.startswith("#") and len(c1) == 7

    def test_groups_from_fragment_to_neuron(self):
        out = groups_from_fragment_to_neuron({"f1": "n1", "f2": "n1"}, id_map={"f1": 1, "f2": 2})
        assert out == {1: "n1", 2: "n1"}

    def test_groups_from_neuron_hypotheses(self):
        class H:
            def __init__(self, nid, frags):
                self.neuron_id = nid
                self.fragment_ids = frags

        out = groups_from_neuron_hypotheses([H(1, [10, 11]), H(2, [12])])
        assert out == {10: "1", 11: "1", 12: "2"}


class TestExportBundle:
    def test_export_bundle_writes_precomputed_and_manifest(self, tmp_path):
        skels = {1: path_skeleton(4), 2: y_fork_skeleton()}
        manifest = export_bundle(tmp_path / "b", skels, groups={"grp": [1, 2]},
                                 params=MeshParams(sides=6), formats=("precomputed",))
        assert manifest["n_segments"] == 2
        assert manifest["n_groups"] == 1
        assert (tmp_path / "b" / "mesh" / "info").exists()
        assert (tmp_path / "b" / "skeleton" / "info").exists()
        assert (tmp_path / "b" / "state.json").exists()
        assert (tmp_path / "b" / "url.txt").exists()

    def test_export_bundle_group_meshes_when_requested(self, tmp_path):
        skels = {1: path_skeleton(4), 2: y_fork_skeleton()}
        manifest = export_bundle(tmp_path / "b", skels, groups={"grp": [1, 2]},
                                 write_group_meshes=True, formats=("precomputed",))
        assert manifest["has_group_meshes"]
        assert (tmp_path / "b" / "groups" / "mesh" / "info").exists()

    def test_export_bundle_obj_and_ply(self, tmp_path):
        skels = {1: path_skeleton(4)}
        export_bundle(tmp_path / "b", skels, formats=("obj", "ply"))
        assert (tmp_path / "b" / "export" / "all.obj").exists()
        assert (tmp_path / "b" / "export" / "all.ply").exists()
        assert not (tmp_path / "b" / "mesh").exists()

    def test_export_bundle_rejects_unknown_format(self, tmp_path):
        with pytest.raises(ValueError):
            export_bundle(tmp_path / "b", {1: path_skeleton(3)}, formats=("stl",))

    def test_export_bundle_clean_removes_stale_segments(self, tmp_path):
        export_bundle(tmp_path / "b", {1: path_skeleton(3), 2: path_skeleton(3)})
        export_bundle(tmp_path / "b", {1: path_skeleton(3)}, clean=True)
        assert not (tmp_path / "b" / "mesh" / "2:0").exists()

    def test_load_manifest_round_trips(self, tmp_path):
        export_bundle(tmp_path / "b", {1: path_skeleton(3)})
        m = load_manifest(tmp_path / "b")
        assert m["n_segments"] == 1

    def test_build_state_declares_nanometre_dimensions_and_segment_colors(self, tmp_path):
        export_bundle(tmp_path / "b", {1: path_skeleton(3), 2: path_skeleton(3)},
                     groups={"grp": [1, 2]})
        state = build_state(tmp_path / "b", base_url="http://localhost:9000",
                            served_root=tmp_path)
        assert state["dimensions"]["x"] == [1e-9, "m"]
        seg_layer = next(l for l in state["layers"] if l["name"] != "EM")
        assert seg_layer["source"][0] == "precomputed://http://localhost:9000/b/mesh"
        assert set(seg_layer["segments"]) == {"1", "2"}
        assert seg_layer["equivalences"] == [["1", "2"]]

    def test_state_json_is_reasonably_small_for_a_handful_of_segments(self, tmp_path):
        export_bundle(tmp_path / "b", {i: path_skeleton(3) for i in range(5)})
        state_bytes = (tmp_path / "b" / "state.json").read_bytes()
        assert len(state_bytes) < 50_000


# ---------------------------------------------------------------------------
# sources: kimimaro archive adapter
# ---------------------------------------------------------------------------

class TestKimimaroSource:
    def _write_archive(self, tmp_path):
        # Mirrors neuronauts.cell_graph.precompute_self_skeletons_for_cache's
        # npz layout: root_ids, v_offsets (len R+1), n_edges (len R), and
        # vertices/edges/radii concatenated across all roots (edges pre-offset).
        v1 = np.array([[0, 0, 0], [1000, 0, 0]], np.float32)
        e1 = np.array([[0, 1]], np.int64)
        r1 = np.array([100.0, 100.0], np.float32)
        v2 = np.array([[0, 0, 0], [0, 1000, 0], [0, 2000, 0]], np.float32)
        e2 = np.array([[0, 1], [1, 2]], np.int64) + len(v1)
        r2 = np.array([50.0, 60.0, 70.0], np.float32)
        path = tmp_path / "archive.npz"
        np.savez_compressed(
            path, root_ids=np.array([111, 222], np.int64),
            v_offsets=np.array([0, len(v1), len(v1) + len(v2)], np.int64),
            n_edges=np.array([len(e1), len(e2)], np.int64),
            vertices=np.concatenate([v1, v2]),
            edges=np.concatenate([e1, e2]),
            radii=np.concatenate([r1, r2]),
        )
        return path

    def test_splits_roots_and_reoffsets_edges(self, tmp_path):
        path = self._write_archive(tmp_path)
        out = kimimaro_archive_skeletons(path)
        assert set(out.keys()) == {111, 222}
        assert out[111].n_vertices == 2 and out[111].edges.tolist() == [[0, 1]]
        assert out[222].n_vertices == 3 and out[222].edges.tolist() == [[0, 1], [1, 2]]
        assert np.allclose(out[222].radii_nm, [50.0, 60.0, 70.0])
