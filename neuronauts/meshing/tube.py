"""Skeleton -> triangle mesh, with no dependency beyond numpy.

The mesh is a swept tube: one ring of ``sides`` vertices per skeleton vertex,
consecutive rings joined by quads along every unbranched run, and a sphere at
every junction and tip so the runs meet without cracks. Rings are oriented by
a rotation-minimising frame (one normal parallel-transported along the run),
which avoids the twist a fixed "up" vector produces on a bending neurite. The
per-vertex radius is honoured, so the tube tapers where the caliber does.

Why not marching cubes over the segmentation? That needs voxels, which the
harness deliberately does not keep. The L2 graph plus its distance-transform
caliber *is* the geometry we have; this turns it into a surface at roughly
30 triangles per skeleton vertex (6 sides, level-1 spheres).

Triangle winding is counter-clockwise seen from outside, so normals point
outward for every consumer that cares (Blender, MeshLab, three.js).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np

from neuronauts.meshing.skeleton import SkeletonGeometry

CAP_MODES = ("junctions", "all", "none")


# ---------------------------------------------------------------------------
# triangle mesh container
# ---------------------------------------------------------------------------

@dataclass
class TriMesh:
    """Indexed triangle mesh, vertices in nm (float32), faces uint32."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=np.float32).reshape(-1, 3)
        f = np.asarray(self.faces, dtype=np.uint32).reshape(-1, 3)
        if len(f) and int(f.max()) >= len(v):
            raise ValueError(f"face index {int(f.max())} out of range for {len(v)} vertices")
        self.vertices, self.faces = v, f

    @classmethod
    def empty(cls) -> "TriMesh":
        return cls(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def is_empty(self) -> bool:
        return self.n_faces == 0

    def bounds_nm(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_vertices == 0:
            z = np.zeros(3, np.float64)
            return z, z
        v = self.vertices.astype(np.float64)
        return v.min(axis=0), v.max(axis=0)

    def center_nm(self) -> np.ndarray:
        lo, hi = self.bounds_nm()
        return (lo + hi) / 2.0

    def scaled(self, factor: float) -> "TriMesh":
        return TriMesh(self.vertices * np.float32(factor), self.faces)

    def face_normals(self) -> np.ndarray:
        """Unnormalised face normals (length = 2 x triangle area), float64."""
        v = self.vertices.astype(np.float64)
        a, b, c = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        return np.cross(b - a, c - a)

    def surface_area_nm2(self) -> float:
        return float(0.5 * np.linalg.norm(self.face_normals(), axis=1).sum())

    def signed_volume_nm3(self) -> float:
        """Positive for a closed, outward-wound surface."""
        v = self.vertices.astype(np.float64)
        a, b, c = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    @staticmethod
    def concat(meshes: Sequence["TriMesh"]) -> "TriMesh":
        meshes = [m for m in meshes if m.n_vertices > 0]
        if not meshes:
            return TriMesh.empty()
        offs = np.cumsum([0] + [m.n_vertices for m in meshes[:-1]]).astype(np.uint32)
        faces = [m.faces + o for m, o in zip(meshes, offs) if m.n_faces]
        return TriMesh(
            np.concatenate([m.vertices for m in meshes]),
            np.concatenate(faces) if faces else np.zeros((0, 3), np.uint32),
        )


# ---------------------------------------------------------------------------
# chains: partition the edge set into maximal unbranched runs
# ---------------------------------------------------------------------------

def skeleton_chains(edges: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    """Split an undirected graph into maximal runs whose interior nodes have
    degree 2. Every edge lands in exactly one chain exactly once; a component
    that is a pure cycle becomes one chain that starts and ends on the same
    node. Isolated nodes appear in no chain.
    """
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n_edges = len(edges)
    if n_edges == 0:
        return []
    both_node = np.concatenate([edges[:, 0], edges[:, 1]])
    both_eid = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    order = np.argsort(both_node, kind="stable")
    inc_eid = both_eid[order]
    deg = np.bincount(both_node, minlength=n_vertices)
    indptr = np.zeros(n_vertices + 1, np.int64)
    np.cumsum(deg, out=indptr[1:])
    used = np.zeros(n_edges, bool)
    e_u, e_v = edges[:, 0], edges[:, 1]

    def walk(u: int, e: int) -> np.ndarray:
        path = [u]
        while True:
            used[e] = True
            v = int(e_v[e]) if int(e_u[e]) == u else int(e_u[e])
            path.append(v)
            if deg[v] != 2:
                break
            e0, e1 = int(inc_eid[indptr[v]]), int(inc_eid[indptr[v] + 1])
            e_next = e1 if e0 == e else e0
            if used[e_next]:
                break
            u, e = v, e_next
        return np.asarray(path, dtype=np.int64)

    chains: list[np.ndarray] = []
    for u in np.flatnonzero(deg != 2).tolist():
        for e in inc_eid[indptr[u]:indptr[u + 1]].tolist():
            if not used[e]:
                chains.append(walk(u, e))
    # whatever is left lies on pure cycles
    for e in range(n_edges):
        if not used[e]:
            chains.append(walk(int(e_u[e]), e))
    return chains


# ---------------------------------------------------------------------------
# frames along a chain
# ---------------------------------------------------------------------------

def _perpendicular(t: np.ndarray) -> np.ndarray:
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(t)))] = 1.0
    n = np.cross(t, axis)
    return n / np.linalg.norm(n)


def chain_frames(points: np.ndarray, eps: float = 1e-6
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Tangent, normal, binormal at every point of a polyline (``[k+1, 3]`` each).

    Tangents are the mean of the incoming and outgoing directions; a repeated
    point borrows its neighbour's direction; a hairpin keeps the incoming one.
    Normals are parallel-transported so the frame does not spin around the
    tangent. Returns ``None`` when every point coincides.
    """
    p = np.asarray(points, dtype=np.float64)
    d = np.diff(p, axis=0)
    length = np.linalg.norm(d, axis=1)
    valid = length > eps
    if not valid.any():
        return None
    dirs = np.zeros_like(d)
    dirs[valid] = d[valid] / length[valid, None]
    if not valid.all():
        vi = np.flatnonzero(valid)
        near = vi[np.clip(np.searchsorted(vi, np.arange(len(d))), 0, len(vi) - 1)]
        dirs[~valid] = dirs[near[~valid]]

    k1 = len(p)
    tangent = np.empty((k1, 3))
    tangent[0], tangent[-1] = dirs[0], dirs[-1]
    if k1 > 2:
        mid = dirs[:-1] + dirs[1:]
        norm = np.linalg.norm(mid, axis=1)
        hairpin = norm < eps
        mid[hairpin] = dirs[:-1][hairpin]
        norm[hairpin] = 1.0
        tangent[1:-1] = mid / norm[:, None]

    normal = np.empty_like(tangent)
    n = _perpendicular(tangent[0])
    normal[0] = n
    for i in range(1, k1):
        t = tangent[i]
        n = n - np.dot(n, t) * t
        nn = np.linalg.norm(n)
        n = _perpendicular(t) if nn < 1e-6 else n / nn
        normal[i] = n
    binormal = np.cross(tangent, normal)
    return tangent, normal, binormal


# ---------------------------------------------------------------------------
# unit sphere (subdivided octahedron), cached per level
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def unit_sphere(level: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Outward-wound unit sphere: level 0 = 8 faces, each level x4."""
    if level < 0:
        raise ValueError("sphere level must be >= 0")
    verts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    verts = [np.asarray(v, dtype=np.float64) for v in verts]
    for _ in range(level):
        cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (a, b) if a < b else (b, a)
            idx = cache.get(key)
            if idx is None:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                idx = len(verts) - 1
                cache[key] = idx
            return idx

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        faces = new_faces
    v = np.stack(verts).astype(np.float64)
    f = np.asarray(faces, dtype=np.int64)
    v.setflags(write=False)
    f.setflags(write=False)
    return v, f


# ---------------------------------------------------------------------------
# the mesher
# ---------------------------------------------------------------------------

def tube_mesh(
    skel: SkeletonGeometry,
    *,
    sides: int = 6,
    sphere_level: int = 1,
    min_radius_nm: float = 30.0,
    max_radius_nm: float | None = None,
    radius_scale: float = 1.0,
    caps: str = "junctions",
) -> TriMesh:
    """Sweep a tube along every unbranched run of ``skel`` and cap the joints.

    Parameters
    ----------
    sides
        Vertices per ring (>= 3). Six is enough for neurites viewed at a
        distance; use 12+ for close-ups.
    sphere_level
        Subdivision level of the joint spheres (0: 8 faces, 1: 32, 2: 128).
    min_radius_nm, max_radius_nm, radius_scale
        Radii are ``clip(r * radius_scale, min, max)``. The floor keeps thin
        axons and zero-caliber L2 nodes visible.
    caps
        ``"junctions"`` (default): spheres where degree != 2, i.e. tips and
        branch points -- the only places two runs meet. ``"all"``: a sphere at
        every vertex, for very coarse ``sides`` on sharply bending paths.
        ``"none"``: no spheres except on isolated vertices, which would
        otherwise vanish.
    """
    if sides < 3:
        raise ValueError("sides must be >= 3")
    if caps not in CAP_MODES:
        raise ValueError(f"caps must be one of {CAP_MODES}")
    skel = skel.drop_invalid()
    n_vert = skel.n_vertices
    if n_vert == 0:
        return TriMesh.empty()

    pos = skel.vertices_nm.astype(np.float64)
    hi = np.inf if not max_radius_nm else float(max_radius_nm)
    rad = np.clip(skel.radii_nm.astype(np.float64) * float(radius_scale),
                  float(min_radius_nm), hi)
    deg = skel.degree()

    phi = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    j = np.arange(sides)
    jn = (j + 1) % sides

    vert_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    n_out = 0

    for chain in skeleton_chains(skel.edges, n_vert):
        frames = chain_frames(pos[chain])
        if frames is None:            # every point coincides; the caps cover it
            continue
        _, normal, binormal = frames
        r = rad[chain]
        rings = (pos[chain][:, None, :]
                 + r[:, None, None] * (cos_phi[None, :, None] * normal[:, None, :]
                                       + sin_phi[None, :, None] * binormal[:, None, :]))
        k = len(chain) - 1
        i = np.arange(k)[:, None]
        a = n_out + i * sides + j[None, :]
        b = n_out + i * sides + jn[None, :]
        c = n_out + (i + 1) * sides + jn[None, :]
        d = n_out + (i + 1) * sides + j[None, :]
        faces = np.concatenate([
            np.stack([a, b, c], axis=-1).reshape(-1, 3),
            np.stack([a, c, d], axis=-1).reshape(-1, 3),
        ])
        vert_blocks.append(rings.reshape(-1, 3))
        face_blocks.append(faces)
        n_out += (k + 1) * sides

    if caps == "all":
        cap_nodes = np.arange(n_vert)
    elif caps == "junctions":
        cap_nodes = np.flatnonzero(deg != 2)
    else:
        cap_nodes = np.flatnonzero(deg == 0)
    if len(cap_nodes):
        unit_v, unit_f = unit_sphere(sphere_level)
        centres = pos[cap_nodes]
        r = rad[cap_nodes]
        sphere_v = centres[:, None, :] + r[:, None, None] * unit_v[None, :, :]
        base = n_out + np.arange(len(cap_nodes)) * len(unit_v)
        sphere_f = unit_f[None, :, :] + base[:, None, None]
        vert_blocks.append(sphere_v.reshape(-1, 3))
        face_blocks.append(sphere_f.reshape(-1, 3))
        n_out += len(cap_nodes) * len(unit_v)

    if not vert_blocks:
        return TriMesh.empty()
    return TriMesh(
        np.concatenate(vert_blocks).astype(np.float32),
        np.concatenate(face_blocks).astype(np.uint32),
    )


def mesh_stats(mesh: TriMesh) -> dict[str, float]:
    lo, hi = mesh.bounds_nm()
    return {
        "n_vertices": int(mesh.n_vertices),
        "n_faces": int(mesh.n_faces),
        "bytes_precomputed": int(4 + 12 * mesh.n_vertices + 12 * mesh.n_faces),
        "bounds_lo_nm": lo.tolist(),
        "bounds_hi_nm": hi.tolist(),
    }


__all__ = ["CAP_MODES", "TriMesh", "chain_frames", "mesh_stats", "skeleton_chains",
           "tube_mesh", "unit_sphere"]
