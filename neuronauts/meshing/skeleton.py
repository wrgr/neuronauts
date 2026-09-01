"""Skeleton geometry: the one input type every mesher and writer consumes.

Every result in this project that has a shape -- a v117 atom's L2 graph, a
kimimaro or CAVE skeleton, a ``Fragment``, an assembled neuron -- reduces to
vertices in global nanometres, undirected edges between them, and a radius per
vertex. This container holds exactly that, so adapters (``sources.py``) only
have to produce it and the tube mesher (``tube.py``) only has to read it.

Invariants enforced on construction:

* ``vertices_nm`` is float32 ``[V, 3]``; ``radii_nm`` is float32 ``[V]``.
* ``edges`` is int64 ``[E, 2]`` with ``u < v``, unique, no self loops, all
  indices in range. An out-of-range edge raises rather than being clipped: a
  clipped edge would connect the wrong vertices and the mesh would still look
  plausible, which is the worst kind of bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_RADIUS_NM = 200.0


def canonical_edges(edges: Any, n_vertices: int) -> np.ndarray:
    """Undirected edge list as int64 ``[E, 2]``, ``u < v``, unique, no loops."""
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(e) == 0:
        return np.zeros((0, 2), np.int64)
    if e.min() < 0 or e.max() >= n_vertices:
        raise ValueError(
            f"edge index out of range: got {e.min()}..{e.max()} for {n_vertices} vertices")
    e = e[e[:, 0] != e[:, 1]]
    if len(e) == 0:
        return np.zeros((0, 2), np.int64)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


@dataclass
class SkeletonGeometry:
    """Vertices in global nm, undirected edges, radius per vertex.

    ``node_ids`` is optional provenance (for example the L2 chunk id behind
    each vertex) and rides along through filtering and concatenation.
    """

    vertices_nm: np.ndarray
    edges: np.ndarray
    radii_nm: np.ndarray
    node_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ basics
    def validate(self) -> "SkeletonGeometry":
        v = np.asarray(self.vertices_nm, dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices_nm must be [V, 3], got {v.shape}")
        r = np.asarray(self.radii_nm, dtype=np.float32).reshape(-1)
        if r.size == 1 and len(v) != 1:
            r = np.full(len(v), float(r[0]), np.float32)
        if len(r) != len(v):
            raise ValueError(f"radii_nm length {len(r)} != n_vertices {len(v)}")
        self.vertices_nm = v
        self.radii_nm = r
        self.edges = canonical_edges(self.edges, len(v))
        if self.node_ids is not None:
            ids = np.asarray(self.node_ids).reshape(-1)
            if len(ids) != len(v):
                raise ValueError(f"node_ids length {len(ids)} != n_vertices {len(v)}")
            self.node_ids = ids
        return self

    @property
    def n_vertices(self) -> int:
        return len(self.vertices_nm)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def degree(self) -> np.ndarray:
        return np.bincount(self.edges.reshape(-1), minlength=self.n_vertices).astype(np.int32)

    def cable_length_nm(self) -> float:
        if self.n_edges == 0:
            return 0.0
        p = self.vertices_nm.astype(np.float64)
        return float(np.linalg.norm(p[self.edges[:, 0]] - p[self.edges[:, 1]], axis=1).sum())

    def bounds_nm(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_vertices == 0:
            z = np.zeros(3, np.float64)
            return z, z
        v = self.vertices_nm.astype(np.float64)
        return v.min(axis=0), v.max(axis=0)

    # --------------------------------------------------------------- filtering
    def drop_invalid(self) -> "SkeletonGeometry":
        """Remove vertices with a non-finite coordinate or radius, reindexing edges.

        Returns ``self`` when nothing needs dropping.
        """
        keep = np.isfinite(self.vertices_nm).all(axis=1) & np.isfinite(self.radii_nm)
        if keep.all():
            return self
        new_index = np.full(self.n_vertices, -1, np.int64)
        new_index[keep] = np.arange(int(keep.sum()))
        e = new_index[self.edges]
        e = e[(e >= 0).all(axis=1)]
        return SkeletonGeometry(
            vertices_nm=self.vertices_nm[keep],
            edges=e,
            radii_nm=self.radii_nm[keep],
            node_ids=None if self.node_ids is None else self.node_ids[keep],
        )

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *,
                  default_radius_nm: float = DEFAULT_RADIUS_NM) -> "SkeletonGeometry":
        """Build from a dict or ``NpzFile`` using the key spellings found in this repo.

        Accepts ``vertices_nm``/``vertices``, ``edges``, ``radii_nm``/``radius_nm``/
        ``radii``/``radius`` (constant ``default_radius_nm`` when absent) and the
        optional ``node_ids``/``l2_ids``.
        """
        keys = set(d.keys()) if hasattr(d, "keys") else set(d.files)  # type: ignore[attr-defined]

        def pick(*names):
            for n in names:
                if n in keys:
                    return d[n]
            return None

        verts = pick("vertices_nm", "vertices", "verts")
        if verts is None:
            raise KeyError("no vertices key (vertices_nm / vertices / verts)")
        edges = pick("edges")
        if edges is None:
            edges = np.zeros((0, 2), np.int64)
        radii = pick("radii_nm", "radius_nm", "radii", "radius")
        if radii is None:
            radii = np.full(len(verts), float(default_radius_nm), np.float32)
        return cls(vertices_nm=verts, edges=edges, radii_nm=radii,
                   node_ids=pick("node_ids", "l2_ids"))

    @classmethod
    def from_fragment(cls, frag: Any, *,
                      default_radius_nm: float = DEFAULT_RADIUS_NM) -> "SkeletonGeometry":
        """Duck-typed adapter for ``neuronauts.schemas.Fragment`` and
        ``neuronauts.global_merge.schemas.SegmentFragment`` (``radius_nm`` vs
        ``radii_nm``)."""
        radii = getattr(frag, "radius_nm", None)
        if radii is None:
            radii = getattr(frag, "radii_nm", None)
        verts = frag.vertices_nm
        if radii is None:
            radii = np.full(len(verts), float(default_radius_nm), np.float32)
        return cls(vertices_nm=verts, edges=frag.edges, radii_nm=radii)

    def to_dict(self) -> dict[str, np.ndarray]:
        out = {"vertices_nm": self.vertices_nm, "edges": self.edges, "radii_nm": self.radii_nm}
        if self.node_ids is not None:
            out["node_ids"] = self.node_ids
        return out


def concat_skeletons(parts: Sequence[SkeletonGeometry]) -> SkeletonGeometry:
    """Disjoint union of several skeletons (edges re-offset). Used to give an
    assembled neuron one skeleton out of its fragments' skeletons."""
    parts = [p for p in parts if p.n_vertices > 0]
    if not parts:
        return SkeletonGeometry(np.zeros((0, 3), np.float32), np.zeros((0, 2), np.int64),
                                np.zeros(0, np.float32))
    offs = np.cumsum([0] + [p.n_vertices for p in parts[:-1]])
    edges = [p.edges + o for p, o in zip(parts, offs) if p.n_edges]
    has_ids = all(p.node_ids is not None for p in parts)
    return SkeletonGeometry(
        vertices_nm=np.concatenate([p.vertices_nm for p in parts]),
        edges=np.concatenate(edges) if edges else np.zeros((0, 2), np.int64),
        radii_nm=np.concatenate([p.radii_nm for p in parts]),
        node_ids=np.concatenate([p.node_ids for p in parts]) if has_ids else None,
    )


__all__ = ["DEFAULT_RADIUS_NM", "SkeletonGeometry", "canonical_edges", "concat_skeletons"]
