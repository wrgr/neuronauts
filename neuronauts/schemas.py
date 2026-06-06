"""Typed inter-stage contracts for the neuronauts pipeline.

This module defines the **artifacts that flow between pipeline stages**. It is
the backbone of the modular, team-friendly architecture described in
``docs/roadmap_global_assembly.md``: each stage reads and writes one of these
typed artifacts on disk, so a stage owner depends only on the *schema* of the
upstream artifact — never on the upstream stage's code.

Stage → artifact map
--------------------
================  ===================================================
Stage             Produces
================  ===================================================
``data/``         :class:`Region`     (synapses + skeletons for a tile)
``represent/``    :class:`Fragment`   (skeleton arc + learned tree-DNA)
``assemble/``     :class:`NeuronHypothesis` (fragments grouped into a neuron)
``connectome/``   :class:`ConnectomeGraph`  (neuron × neuron graph)
================  ===================================================

Design rules for everything in this file:

* **Dependency-light.** Only ``numpy`` + the standard library are imported, so
  the contract can be produced/consumed without pulling in CAVE, torch, or the
  legacy simulation stack.
* **Plain arrays.** Fields are numpy arrays / primitives with documented dtype
  and shape, not opaque objects, so artifacts serialize cleanly.
* **Explicit validation.** Each type has a ``validate()`` that raises on shape /
  dtype mismatch. There is no hidden coercion.
* **Pickle-free I/O.** On-disk round-trips use plain ``.npz`` (CSR-style ragged
  packing), never ``allow_pickle``.

Coordinates: synapse positions in :class:`fetch.SynapseTable` are
*box-relative voxel* coordinates. Everything in this module is in **global
nanometers** so that fragments from different tiles share one coordinate frame
and can be stitched across box seams (the core move of global assembly). Use
:meth:`Region.from_synapse_table` to convert, which mirrors
``skeleton_graph._globalize_points``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 1

Vec3 = Tuple[float, float, float]
BBox = Tuple[Vec3, Vec3]  # (min_xyz, max_xyz) in global nm


def _as_f32(arr, name: str, *, cols: int | None = None) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if cols is None:
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    else:
        if a.ndim != 2 or a.shape[1] != cols:
            raise ValueError(f"{name} must have shape [N, {cols}], got {a.shape}")
    return a


def _as_i64(arr, name: str, *, cols: int | None = None) -> np.ndarray:
    a = np.asarray(arr, dtype=np.int64)
    if cols is None:
        if a.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    elif a.ndim != 2 or a.shape[1] != cols:
        raise ValueError(f"{name} must have shape [N, {cols}], got {a.shape}")
    return a


# --------------------------------------------------------------------------- #
# Stage: data/  ->  Region
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    """A spatial tile: its synapses (in global nm) plus the materialization
    versions needed for leakage-safe skeletonization and supervision.

    ``seg_version`` is the *base* materialization the segmentation / skeletons
    are derived from; ``label_version`` is the *target* materialization whose
    root IDs are used as supervision and evaluation ground truth. Keeping them
    distinct is what lets the model learn the base→target transfer function
    without leaking labels (see ``skeleton_graph.validate_skeleton_graph_config``).
    """

    region_id: str
    bbox_nm: BBox
    voxel_size_nm: Vec3
    seg_version: int
    label_version: int

    pre_pt_nm: np.ndarray  # [N, 3] float32, global nm
    post_pt_nm: np.ndarray  # [N, 3] float32, global nm
    pre_root_id: np.ndarray  # [N] int64 @ label_version
    post_root_id: np.ndarray  # [N] int64 @ label_version
    synapse_id: np.ndarray  # [N] int64

    pre_seg_id: np.ndarray | None = None  # [N] int64 @ seg_version (scaffold)
    post_seg_id: np.ndarray | None = None

    @property
    def n_synapses(self) -> int:
        return len(self.synapse_id)

    def validate(self) -> "Region":
        n = self.n_synapses
        self.pre_pt_nm = _as_f32(self.pre_pt_nm, "pre_pt_nm", cols=3)
        self.post_pt_nm = _as_f32(self.post_pt_nm, "post_pt_nm", cols=3)
        for name in ("pre_root_id", "post_root_id", "synapse_id"):
            setattr(self, name, _as_i64(getattr(self, name), name))
        for name in ("pre_pt_nm", "post_pt_nm", "pre_root_id", "post_root_id"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length {len(getattr(self, name))} != n_synapses {n}")
        for name in ("pre_seg_id", "post_seg_id"):
            val = getattr(self, name)
            if val is not None:
                val = _as_i64(val, name)
                if len(val) != n:
                    raise ValueError(f"{name} length {len(val)} != n_synapses {n}")
                setattr(self, name, val)
        if self.seg_version == self.label_version:
            # Not fatal, but a strong smell: skeletons would carry target labels.
            pass
        return self

    @classmethod
    def from_synapse_table(
        cls,
        synapses,
        *,
        region_id: str,
        bbox_nm: BBox,
        voxel_size_nm: Sequence[float],
        seg_version: int,
        label_version: int,
    ) -> "Region":
        """Build a Region from a ``fetch.SynapseTable`` (duck-typed).

        Converts the table's box-relative voxel coordinates to global nm via
        ``pt * voxel_size_nm + bbox_min`` — the same transform as
        ``skeleton_graph._globalize_points``.
        """
        origin = np.asarray(bbox_nm[0], dtype=np.float32)
        vox = np.asarray(voxel_size_nm, dtype=np.float32)
        pre = np.asarray(synapses.pre_pt, dtype=np.float32) * vox[None, :] + origin[None, :]
        post = np.asarray(synapses.post_pt, dtype=np.float32) * vox[None, :] + origin[None, :]
        return cls(
            region_id=region_id,
            bbox_nm=(tuple(map(float, bbox_nm[0])), tuple(map(float, bbox_nm[1]))),
            voxel_size_nm=tuple(map(float, vox.tolist())),
            seg_version=int(seg_version),
            label_version=int(label_version),
            pre_pt_nm=pre,
            post_pt_nm=post,
            pre_root_id=np.asarray(synapses.pre_root_id, dtype=np.int64),
            post_root_id=np.asarray(synapses.post_root_id, dtype=np.int64),
            synapse_id=np.asarray(synapses.synapse_id, dtype=np.int64),
            pre_seg_id=None if synapses.pre_seg_id is None else np.asarray(synapses.pre_seg_id, dtype=np.int64),
            post_seg_id=None if synapses.post_seg_id is None else np.asarray(synapses.post_seg_id, dtype=np.int64),
        ).validate()

    def save_npz(self, path: str) -> None:
        self.validate()
        meta = dict(
            schema_version=SCHEMA_VERSION,
            region_id=self.region_id,
            bbox_nm=self.bbox_nm,
            voxel_size_nm=self.voxel_size_nm,
            seg_version=self.seg_version,
            label_version=self.label_version,
            has_seg=self.pre_seg_id is not None,
        )
        arrays = dict(
            pre_pt_nm=self.pre_pt_nm,
            post_pt_nm=self.post_pt_nm,
            pre_root_id=self.pre_root_id,
            post_root_id=self.post_root_id,
            synapse_id=self.synapse_id,
            meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
        )
        if self.pre_seg_id is not None:
            arrays["pre_seg_id"] = self.pre_seg_id
            arrays["post_seg_id"] = self.post_seg_id
        np.savez(path, **arrays)

    @classmethod
    def load_npz(cls, path: str) -> "Region":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            bbox = meta["bbox_nm"]
            return cls(
                region_id=meta["region_id"],
                bbox_nm=(tuple(bbox[0]), tuple(bbox[1])),
                voxel_size_nm=tuple(meta["voxel_size_nm"]),
                seg_version=int(meta["seg_version"]),
                label_version=int(meta["label_version"]),
                pre_pt_nm=z["pre_pt_nm"],
                post_pt_nm=z["post_pt_nm"],
                pre_root_id=z["pre_root_id"],
                post_root_id=z["post_root_id"],
                synapse_id=z["synapse_id"],
                pre_seg_id=z["pre_seg_id"] if meta["has_seg"] else None,
                post_seg_id=z["post_seg_id"] if meta["has_seg"] else None,
            ).validate()


# --------------------------------------------------------------------------- #
# Stage: represent/  ->  Fragment (carries tree-DNA)
# --------------------------------------------------------------------------- #
@dataclass
class Fragment:
    """A skeleton arc with a global coordinate frame and a learned morphology
    embedding ("tree-DNA").

    ``fragment_id`` is unique across *all* regions, so fragments from adjacent
    tiles can live in one global pool and be stitched at seams. ``endpoints_nm``
    are the tip vertices used as the seam-stitch handles. ``dna`` is filled by
    the ``represent/`` stage; it is ``None`` for a freshly extracted fragment.
    """

    fragment_id: int
    region_id: str
    base_root_id: int  # noisy seg root @ Region.seg_version
    vertices_nm: np.ndarray  # [V, 3] float32, global nm
    edges: np.ndarray  # [E, 2] int64, indices into vertices
    endpoints_nm: np.ndarray  # [T, 3] float32, tip coordinates
    radius_nm: np.ndarray  # [V] float32, caliber profile
    synapse_indices: np.ndarray  # [S] int64, rows of the owning Region
    dna: np.ndarray | None = None  # [D] float32, learned embedding

    @property
    def n_vertices(self) -> int:
        return len(self.vertices_nm)

    def validate(self) -> "Fragment":
        self.vertices_nm = _as_f32(self.vertices_nm, "vertices_nm", cols=3)
        self.endpoints_nm = _as_f32(self.endpoints_nm, "endpoints_nm", cols=3)
        self.radius_nm = _as_f32(self.radius_nm, "radius_nm")
        self.edges = _as_i64(self.edges, "edges", cols=2)
        self.synapse_indices = _as_i64(self.synapse_indices, "synapse_indices")
        if len(self.radius_nm) != self.n_vertices:
            raise ValueError(f"radius_nm length {len(self.radius_nm)} != n_vertices {self.n_vertices}")
        if len(self.edges) and int(self.edges.max()) >= self.n_vertices:
            raise ValueError("edges index out of range of vertices_nm")
        if self.dna is not None:
            self.dna = _as_f32(self.dna, "dna")
        return self


# --------------------------------------------------------------------------- #
# Stage: assemble/  ->  NeuronHypothesis
# --------------------------------------------------------------------------- #
@dataclass
class NeuronHypothesis:
    """A set of fragments asserted to be one neuron.

    ``spans_regions`` records which tiles the constituent fragments came from;
    a hypothesis with more than one entry is direct evidence of cross-box
    assembly — the thing the box-local pipeline cannot produce.
    """

    neuron_id: int
    fragment_ids: list[int]
    synapse_indices: np.ndarray  # [S] int64 (global synapse rows)
    pooled_dna: np.ndarray | None = None  # [D] float32
    spans_regions: list[str] = field(default_factory=list)

    def validate(self) -> "NeuronHypothesis":
        self.synapse_indices = _as_i64(self.synapse_indices, "synapse_indices")
        if not self.fragment_ids:
            raise ValueError("NeuronHypothesis must contain at least one fragment")
        if self.pooled_dna is not None:
            self.pooled_dna = _as_f32(self.pooled_dna, "pooled_dna")
        return self


# --------------------------------------------------------------------------- #
# Stage: connectome/  ->  ConnectomeGraph
# --------------------------------------------------------------------------- #
@dataclass
class ConnectomeGraph:
    """A directed neuron × neuron graph. ``src``/``dst`` index into
    ``neuron_ids``; ``node_features`` are typically pooled tree-DNA + stats."""

    neuron_ids: np.ndarray  # [M] int64
    node_features: np.ndarray  # [M, F] float32
    src: np.ndarray  # [E] int64, index into neuron_ids
    dst: np.ndarray  # [E] int64
    edge_synapse_count: np.ndarray  # [E] int64

    @property
    def n_nodes(self) -> int:
        return len(self.neuron_ids)

    @property
    def n_edges(self) -> int:
        return len(self.src)

    def validate(self) -> "ConnectomeGraph":
        self.neuron_ids = _as_i64(self.neuron_ids, "neuron_ids")
        self.src = _as_i64(self.src, "src")
        self.dst = _as_i64(self.dst, "dst")
        self.edge_synapse_count = _as_i64(self.edge_synapse_count, "edge_synapse_count")
        self.node_features = _as_f32(self.node_features, "node_features", cols=self.node_features.shape[1] if self.node_features.ndim == 2 else None)
        if self.node_features.ndim != 2 or len(self.node_features) != self.n_nodes:
            raise ValueError("node_features must be [n_nodes, F]")
        if not (len(self.src) == len(self.dst) == len(self.edge_synapse_count)):
            raise ValueError("src, dst, edge_synapse_count must be equal length")
        if self.n_edges and max(int(self.src.max()), int(self.dst.max())) >= self.n_nodes:
            raise ValueError("edge endpoint indexes a non-existent node")
        return self


# --------------------------------------------------------------------------- #
# Fragment collection I/O (pickle-free, CSR-style ragged packing)
# --------------------------------------------------------------------------- #
def save_fragments(path: str, fragments: Sequence[Fragment]) -> None:
    """Serialize a list of fragments to one ``.npz`` without pickling.

    Ragged per-fragment arrays are concatenated with offset indices (CSR style).
    DNA is stored only if *every* fragment carries an embedding of equal width.
    """
    frags = [f.validate() for f in fragments]
    fragment_id = np.array([f.fragment_id for f in frags], dtype=np.int64)
    base_root_id = np.array([f.base_root_id for f in frags], dtype=np.int64)
    region_ids = json.dumps([f.region_id for f in frags])

    def _pack(get):
        chunks = [np.asarray(get(f)) for f in frags]
        offs = np.zeros(len(chunks) + 1, dtype=np.int64)
        offs[1:] = np.cumsum([len(c) for c in chunks])
        cat = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)
        return cat, offs

    vert, vert_off = _pack(lambda f: f.vertices_nm)
    edges, edges_off = _pack(lambda f: f.edges)
    ends, ends_off = _pack(lambda f: f.endpoints_nm)
    rad, rad_off = _pack(lambda f: f.radius_nm)
    syn, syn_off = _pack(lambda f: f.synapse_indices)

    arrays = dict(
        schema_version=np.array([SCHEMA_VERSION], dtype=np.int64),
        fragment_id=fragment_id,
        base_root_id=base_root_id,
        region_ids=np.frombuffer(region_ids.encode("utf-8"), dtype=np.uint8),
        vertices_nm=vert.astype(np.float32), vertices_off=vert_off,
        edges=edges.astype(np.int64).reshape(-1, 2) if len(edges) else np.zeros((0, 2), np.int64), edges_off=edges_off,
        endpoints_nm=ends.astype(np.float32), endpoints_off=ends_off,
        radius_nm=rad.astype(np.float32), radius_off=rad_off,
        synapse_indices=syn.astype(np.int64), synapse_off=syn_off,
    )
    has_dna = bool(frags) and all(f.dna is not None for f in frags)
    if has_dna:
        widths = {len(f.dna) for f in frags}
        if len(widths) == 1:
            arrays["dna"] = np.stack([f.dna for f in frags]).astype(np.float32)
    np.savez(path, **arrays)


def load_fragments(path: str) -> list[Fragment]:
    with np.load(path, allow_pickle=False) as z:
        region_ids = json.loads(bytes(z["region_ids"]).decode("utf-8"))
        n = len(z["fragment_id"])
        dna = z["dna"] if "dna" in z.files else None

        def _slice(cat, off, i):
            return cat[int(off[i]):int(off[i + 1])]

        out: list[Fragment] = []
        for i in range(n):
            out.append(
                Fragment(
                    fragment_id=int(z["fragment_id"][i]),
                    region_id=region_ids[i],
                    base_root_id=int(z["base_root_id"][i]),
                    vertices_nm=_slice(z["vertices_nm"], z["vertices_off"], i),
                    edges=_slice(z["edges"], z["edges_off"], i),
                    endpoints_nm=_slice(z["endpoints_nm"], z["endpoints_off"], i),
                    radius_nm=_slice(z["radius_nm"], z["radius_off"], i),
                    synapse_indices=_slice(z["synapse_indices"], z["synapse_off"], i),
                    dna=None if dna is None else dna[i],
                ).validate()
            )
    return out


__all__ = [
    "SCHEMA_VERSION",
    "Vec3",
    "BBox",
    "Region",
    "Fragment",
    "NeuronHypothesis",
    "ConnectomeGraph",
    "save_fragments",
    "load_fragments",
]
