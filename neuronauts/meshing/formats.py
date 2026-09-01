"""On-disk formats: Neuroglancer precomputed (mesh + skeleton + segment
properties), Wavefront OBJ, and binary PLY.

Precomputed is what Neuroglancer reads directly. The legacy single-resolution
mesh layout is one JSON manifest ``<id>:0`` per segment naming its fragment
files, each fragment being ``uint32 n_vertices | float32 xyz * n | uint32 abc
per triangle``. Skeletons are ``uint32 n_vertices | uint32 n_edges | float32
xyz * n | uint32 uv * e | vertex attributes``. Both are in **nanometres with an
identity transform**: the viewer's precomputed datasource builds a model space
of ``["m","m","m"]`` at ``1e-9`` for standalone mesh and skeleton sources, and
cloud-volume writes compatible bytes: each side's decoder reads the other's
encoder output correctly (checked in ``tests/test_meshing.py`` when
cloud-volume is installed). Mesh bytes are identical; skeleton bytes are not
byte-for-byte because cloud-volume's ``Skeleton`` defaults to also writing a
``vertex_types`` attribute we omit -- our ``info`` only declares ``radius``,
so our own writer/reader pair (and Neuroglancer, which sizes each attribute
from ``info``) stay consistent.

OBJ and PLY are for everything else (Blender, MeshLab, three.js, ParaView);
they default to micrometres because nm-scale coordinates (~1e6) are awkward in
those tools.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from neuronauts.meshing.skeleton import SkeletonGeometry
from neuronauts.meshing.tube import TriMesh

MESH_INFO = {"@type": "neuroglancer_legacy_mesh"}
IDENTITY_TRANSFORM = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
RADIUS_ATTRIBUTE = {"id": "radius", "data_type": "float32", "num_components": 1}
SEGMENT_PROPERTIES_DIR = "segment_properties"


def skeleton_info(with_radius: bool = True) -> dict:
    return {
        "@type": "neuroglancer_skeletons",
        "transform": list(IDENTITY_TRANSFORM),
        "vertex_attributes": [dict(RADIUS_ATTRIBUTE)] if with_radius else [],
    }


def _check_segment_id(sid) -> int:
    s = int(sid)
    if s < 0 or s >= 2 ** 64:
        raise ValueError(f"segment id {sid} is not a uint64")
    return s


# ---------------------------------------------------------------------------
# binary encoders / decoders
# ---------------------------------------------------------------------------

def encode_precomputed_mesh(mesh: TriMesh) -> bytes:
    v = np.ascontiguousarray(mesh.vertices, dtype="<f4")
    f = np.ascontiguousarray(mesh.faces, dtype="<u4")
    return struct.pack("<I", len(v)) + v.tobytes("C") + f.tobytes("C")


def decode_precomputed_mesh(buf: bytes) -> TriMesh:
    if len(buf) < 4:
        raise ValueError("mesh buffer too short")
    n = struct.unpack("<I", buf[:4])[0]
    end_v = 4 + 12 * n
    if len(buf) < end_v or (len(buf) - end_v) % 12:
        raise ValueError("mesh buffer length inconsistent with its vertex count")
    v = np.frombuffer(buf, dtype="<f4", count=3 * n, offset=4).reshape(n, 3)
    f = np.frombuffer(buf, dtype="<u4", offset=end_v).reshape(-1, 3)
    return TriMesh(v.copy(), f.copy())


def encode_precomputed_skeleton(skel: SkeletonGeometry, *, with_radius: bool = True) -> bytes:
    v = np.ascontiguousarray(skel.vertices_nm, dtype="<f4")
    e = np.ascontiguousarray(skel.edges, dtype="<u4")
    out = struct.pack("<II", len(v), len(e)) + v.tobytes("C") + e.tobytes("C")
    if with_radius:
        out += np.ascontiguousarray(skel.radii_nm, dtype="<f4").tobytes("C")
    return out


def decode_precomputed_skeleton(buf: bytes, *, with_radius: bool = True) -> SkeletonGeometry:
    if len(buf) < 8:
        raise ValueError("skeleton buffer too short")
    nv, ne = struct.unpack("<II", buf[:8])
    off = 8
    v = np.frombuffer(buf, dtype="<f4", count=3 * nv, offset=off).reshape(nv, 3)
    off += 12 * nv
    e = np.frombuffer(buf, dtype="<u4", count=2 * ne, offset=off).reshape(ne, 2)
    off += 8 * ne
    if with_radius and len(buf) >= off + 4 * nv:
        r = np.frombuffer(buf, dtype="<f4", count=nv, offset=off).copy()
    else:
        r = np.zeros(nv, np.float32)
    return SkeletonGeometry(v.copy(), e.astype(np.int64), r)


# ---------------------------------------------------------------------------
# precomputed directories
# ---------------------------------------------------------------------------

def mesh_manifest_name(sid: int) -> str:
    return f"{sid}:0"


def mesh_fragment_name(sid: int) -> str:
    return f"{sid}.mesh"


def write_precomputed_mesh_dir(path: str | Path, meshes: Mapping[int, TriMesh], *,
                               segment_properties: bool = False) -> list[int]:
    """Write ``info`` + one manifest and one fragment per non-empty mesh.

    Returns the ids written. Existing files for other ids are left alone, so a
    directory can be extended incrementally.
    """
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    info = dict(MESH_INFO)
    if segment_properties:
        info["segment_properties"] = SEGMENT_PROPERTIES_DIR
    (d / "info").write_text(json.dumps(info))
    written = []
    for sid, mesh in meshes.items():
        sid = _check_segment_id(sid)
        if mesh.is_empty:
            continue
        (d / mesh_fragment_name(sid)).write_bytes(encode_precomputed_mesh(mesh))
        (d / mesh_manifest_name(sid)).write_text(
            json.dumps({"fragments": [mesh_fragment_name(sid)]}))
        written.append(sid)
    return written


def read_precomputed_mesh(path: str | Path, sid: int) -> TriMesh:
    d = Path(path)
    manifest = json.loads((d / mesh_manifest_name(int(sid))).read_text())
    parts = [decode_precomputed_mesh((d / name).read_bytes()) for name in manifest["fragments"]]
    return TriMesh.concat(parts)


def write_precomputed_skeleton_dir(path: str | Path, skels: Mapping[int, SkeletonGeometry], *,
                                   segment_properties: bool = False) -> list[int]:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    info = skeleton_info(with_radius=True)
    if segment_properties:
        info["segment_properties"] = SEGMENT_PROPERTIES_DIR
    (d / "info").write_text(json.dumps(info))
    written = []
    for sid, skel in skels.items():
        sid = _check_segment_id(sid)
        if skel.n_vertices == 0:
            continue
        (d / str(sid)).write_bytes(encode_precomputed_skeleton(skel, with_radius=True))
        written.append(sid)
    return written


def read_precomputed_skeleton(path: str | Path, sid: int) -> SkeletonGeometry:
    return decode_precomputed_skeleton((Path(path) / str(int(sid))).read_bytes())


_NUMBER_DTYPES = {"uint8", "int8", "uint16", "int16", "uint32", "int32", "float32"}


def write_segment_properties(
    path: str | Path,
    ids: Sequence[int],
    *,
    labels: Mapping[int, str] | None = None,
    descriptions: Mapping[int, str] | None = None,
    numbers: Mapping[str, Mapping[int, float]] | None = None,
    number_dtype: str = "float32",
    tags: Mapping[int, Sequence[str]] | None = None,
) -> dict:
    """Write ``<path>/info`` in the ``neuroglancer_segment_properties`` format.

    Labels show up next to ids in the viewer's segment list and are searchable;
    numbers become sortable columns; tags become filter chips.
    """
    if number_dtype not in _NUMBER_DTYPES:
        raise ValueError(f"number_dtype must be one of {sorted(_NUMBER_DTYPES)}")
    ids = [_check_segment_id(s) for s in ids]
    props: list[dict] = []
    if labels:
        props.append({"id": "label", "type": "label",
                      "values": [str(labels.get(s, "")) for s in ids]})
    if descriptions:
        props.append({"id": "description", "type": "description",
                      "values": [str(descriptions.get(s, "")) for s in ids]})
    for name, col in (numbers or {}).items():
        vals = []
        for s in ids:
            x = float(col.get(s, 0.0))
            vals.append(x if np.isfinite(x) else 0.0)
        if number_dtype != "float32":
            vals = [int(round(x)) for x in vals]
        props.append({"id": name, "type": "number", "data_type": number_dtype, "values": vals})
    if tags:
        vocab = sorted({t for s in ids for t in tags.get(s, ())})
        index = {t: i for i, t in enumerate(vocab)}
        props.append({"id": "tags", "type": "tags", "tags": vocab,
                      "values": [sorted(index[t] for t in tags.get(s, ())) for s in ids]})
    info = {"@type": "neuroglancer_segment_properties",
            "inline": {"ids": [str(s) for s in ids], "properties": props}}
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    (d / "info").write_text(json.dumps(info))
    return info


# ---------------------------------------------------------------------------
# OBJ / PLY
# ---------------------------------------------------------------------------

def write_obj(path: str | Path, meshes: Mapping[int, TriMesh], *, scale: float = 1e-3,
              names: Mapping[int, str] | None = None) -> int:
    """One OBJ with an ``o`` group per segment. ``scale`` converts nm -> output
    units (default micrometres). Returns the number of groups written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with p.open("w") as fh:
        fh.write(f"# neuronauts mesh export; coordinates = nm * {scale:g}\n")
        offset = 1
        for sid, mesh in meshes.items():
            if mesh.is_empty:
                continue
            name = (names or {}).get(sid, f"seg_{sid}")
            fh.write(f"o {name}\n")
            v = mesh.vertices.astype(np.float64) * scale
            fh.write("".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in v))
            f = mesh.faces.astype(np.int64) + offset
            fh.write("".join(f"f {a} {b} {c}\n" for a, b, c in f))
            offset += mesh.n_vertices
            n_written += 1
    return n_written


def read_obj(path: str | Path) -> dict[str, TriMesh]:
    """Minimal OBJ reader (v / f / o only) for round-trip checks."""
    verts: list[list[float]] = []
    groups: dict[str, list[list[int]]] = {}
    current = "default"
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if parts[0] == "o":
            current = " ".join(parts[1:])
            groups.setdefault(current, [])
        elif parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            groups.setdefault(current, []).append(
                [int(x.split("/")[0]) - 1 for x in parts[1:4]])
    all_v = np.asarray(verts, np.float32).reshape(-1, 3)
    out = {}
    for name, faces in groups.items():
        f = np.asarray(faces, np.int64).reshape(-1, 3)
        if len(f) == 0:
            continue
        used = np.unique(f)
        remap = np.full(len(all_v), -1, np.int64)
        remap[used] = np.arange(len(used))
        out[name] = TriMesh(all_v[used], remap[f])
    return out


def write_ply(path: str | Path, meshes: Mapping[int, TriMesh], *, scale: float = 1e-3,
              colors: Mapping[int, Sequence[int]] | None = None) -> tuple[int, int]:
    """Binary little-endian PLY, all segments merged, optional per-vertex RGB
    by segment. Returns ``(n_vertices, n_faces)``."""
    parts = [(sid, m) for sid, m in meshes.items() if not m.is_empty]
    n_v = sum(m.n_vertices for _, m in parts)
    n_f = sum(m.n_faces for _, m in parts)
    with_color = colors is not None
    header = ["ply", "format binary_little_endian 1.0",
              f"comment neuronauts mesh export; coordinates = nm * {scale:g}",
              f"element vertex {n_v}",
              "property float x", "property float y", "property float z"]
    if with_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += [f"element face {n_f}", "property list uchar uint vertex_indices", "end_header"]

    vtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if with_color:
        vtype += [("r", "u1"), ("g", "u1"), ("b", "u1")]
    vbuf = np.empty(n_v, dtype=vtype)
    ftype = [("n", "u1"), ("a", "<u4"), ("b", "<u4"), ("c", "<u4")]
    fbuf = np.empty(n_f, dtype=ftype)
    vo = fo = 0
    for sid, m in parts:
        v = m.vertices.astype(np.float64) * scale
        sl = slice(vo, vo + m.n_vertices)
        vbuf["x"][sl], vbuf["y"][sl], vbuf["z"][sl] = v[:, 0], v[:, 1], v[:, 2]
        if with_color:
            r, g, b = (colors or {}).get(sid, (200, 200, 200))
            vbuf["r"][sl], vbuf["g"][sl], vbuf["b"][sl] = r, g, b
        fs = slice(fo, fo + m.n_faces)
        fbuf["n"][fs] = 3
        fbuf["a"][fs] = m.faces[:, 0] + vo
        fbuf["b"][fs] = m.faces[:, 1] + vo
        fbuf["c"][fs] = m.faces[:, 2] + vo
        vo += m.n_vertices
        fo += m.n_faces
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        fh.write(("\n".join(header) + "\n").encode("ascii"))
        fh.write(vbuf.tobytes())
        fh.write(fbuf.tobytes())
    return n_v, n_f


def read_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reader for the PLY layout ``write_ply`` produces (round-trip checks)."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    n_v = n_f = 0
    with_color = False
    for line in header:
        if line.startswith("element vertex"):
            n_v = int(line.split()[-1])
        elif line.startswith("element face"):
            n_f = int(line.split()[-1])
        elif line == "property uchar red":
            with_color = True
    vtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if with_color:
        vtype += [("r", "u1"), ("g", "u1"), ("b", "u1")]
    vbuf = np.frombuffer(data, dtype=vtype, count=n_v, offset=end)
    off = end + vbuf.nbytes
    fbuf = np.frombuffer(data, dtype=[("n", "u1"), ("a", "<u4"), ("b", "<u4"), ("c", "<u4")],
                         count=n_f, offset=off)
    verts = np.stack([vbuf["x"], vbuf["y"], vbuf["z"]], axis=1).astype(np.float32)
    faces = np.stack([fbuf["a"], fbuf["b"], fbuf["c"]], axis=1).astype(np.uint32)
    colors = np.stack([vbuf["r"], vbuf["g"], vbuf["b"]], axis=1) if with_color else None
    return verts, faces, colors


__all__ = [
    "MESH_INFO", "RADIUS_ATTRIBUTE", "SEGMENT_PROPERTIES_DIR",
    "decode_precomputed_mesh", "decode_precomputed_skeleton",
    "encode_precomputed_mesh", "encode_precomputed_skeleton",
    "mesh_fragment_name", "mesh_manifest_name",
    "read_obj", "read_ply", "read_precomputed_mesh", "read_precomputed_skeleton",
    "skeleton_info", "write_obj", "write_ply", "write_precomputed_mesh_dir",
    "write_precomputed_skeleton_dir", "write_segment_properties",
]
