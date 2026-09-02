"""The object point cloud: every L2 node of every atom, not just its tips.

``topology/k*.npz`` stores an *endpoint* table -- the degree-1 nodes of each
atom's contracted L2 skeleton -- and every proximity experiment to date
(EXP-060, EXP-060B, EXP-061) measured distance between those tips. That is a
skeleton-space distance, and it is not the same question as "how close do these
two objects come to each other". A fragment's nearest approach to its
continuation partner need not happen at a tip.

This module loads the other point set, which the raw fetch already contains but
nothing consumed: for each atom, the positions of *all* its L2 nodes, from
``geom/shards/*.npz`` (atom -> L2 ids) joined to ``geom/l2_attributes.npz``
(L2 id -> representative coordinate and distance transform). Built by
``scripts/build_object_geometry.py``.

The two point sets are nested -- an atom's endpoints are a subset of its nodes,
verified exactly at build time -- so for any pair of atoms

    min-distance over nodes  <=  min-distance over endpoints

always, with equality when the closest approach happens to be tip-to-tip. That
inequality is what makes the two measurements comparable rather than merely
different, and the builder refuses to write a file that violates it.

    from neuronauts.harness.objgeom import load_object_geometry
    g = load_object_geometry("data/substrate/geom/objgeom_k10.npz")
    g.points(atom_id)          # [N,3] float32 nm, non-finite rows dropped
    g.min_gap(atom_a, atom_b)  # nm, inf if either atom has no positioned node
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.spatial import cKDTree

from neuronauts.harness.labels import lookup_index


@dataclass
class ObjectGeometry:
    """Per-atom L2 node positions, stored CSR-style over one flat array.

    ``node_ptr`` has one more entry than ``atom_id``; atom ``k``'s nodes are
    rows ``node_ptr[k]:node_ptr[k+1]``. ``resolved`` marks the rows whose
    position is finite -- a handful of L2 nodes carry no representative
    coordinate, and dropping them silently would quietly shrink an atom.
    """

    atom_id: np.ndarray        # [A] uint64
    node_ptr: np.ndarray       # [A+1] int64
    l2_id: np.ndarray          # [N] uint64
    pos_nm: np.ndarray         # [N,3] float32
    max_dt_nm: np.ndarray      # [N] float32, local radius (distance transform)
    resolved: np.ndarray       # [N] bool, position is finite
    meta: dict = field(default_factory=dict)

    _index: Optional[dict] = field(default=None, repr=False, compare=False)
    _trees: dict = field(default_factory=dict, repr=False, compare=False)

    # -- lookup ------------------------------------------------------------
    def row_of(self, atoms: np.ndarray) -> np.ndarray:
        """Row index of each atom id, or -1 when the atom has no geometry."""
        return lookup_index(self.atom_id, np.asarray(atoms, np.uint64))

    def _row(self, atom: int) -> int:
        if self._index is None:
            self._index = {int(a): k for k, a in enumerate(self.atom_id.tolist())}
        return self._index.get(int(atom), -1)

    def points(self, atom: int) -> np.ndarray:
        """Positioned L2 node coordinates of one atom, ``[N,3]`` nm."""
        k = self._row(atom)
        if k < 0:
            return np.empty((0, 3), np.float32)
        s, e = int(self.node_ptr[k]), int(self.node_ptr[k + 1])
        return self.pos_nm[s:e][self.resolved[s:e]]

    def radii(self, atom: int) -> np.ndarray:
        """Local radius at each positioned node, ``[N]`` nm."""
        k = self._row(atom)
        if k < 0:
            return np.empty(0, np.float32)
        s, e = int(self.node_ptr[k]), int(self.node_ptr[k + 1])
        return self.max_dt_nm[s:e][self.resolved[s:e]]

    def n_nodes(self, atom: int) -> int:
        return len(self.points(atom))

    # -- distance ----------------------------------------------------------
    def tree(self, atom: int) -> Optional[cKDTree]:
        """Cached KD-tree over one atom's nodes, or None when it has none."""
        k = int(atom)
        if k not in self._trees:
            P = self.points(k)
            self._trees[k] = cKDTree(P) if len(P) else None
        return self._trees[k]

    def min_gap(self, a: int, b: int, *, surface: bool = False) -> float:
        """Closest approach between two atoms, in nm.

        ``surface=True`` subtracts the local radius at each of the two closest
        nodes, approximating a surface-to-surface gap rather than a
        centre-to-centre one. L2 radii here are small (median 116 nm), so this
        is a correction of a few hundred nm, not a change of regime; it can go
        negative where two objects interpenetrate at L2 resolution, and is
        returned as-is rather than clipped.
        """
        A, tb = self.points(a), self.tree(b)
        if not len(A) or tb is None:
            return float("inf")
        d, j = tb.query(A, k=1)
        i = int(np.argmin(d))
        gap = float(d[i])
        if surface:
            gap -= float(self.radii(a)[i]) + float(self.radii(b)[int(j[i])])
        return gap

    def cloud(self, atoms: Optional[Iterable[int]] = None):
        """``(positions, atom_of_point)`` over all positioned nodes.

        With ``atoms=None`` this is every node in the substrate -- the point
        set a global range query runs against.
        """
        m = self.resolved
        owner = np.repeat(self.atom_id, np.diff(self.node_ptr))
        if atoms is None:
            return self.pos_nm[m], owner[m]
        want = np.isin(owner, np.asarray(list(atoms), np.uint64)) & m
        return self.pos_nm[want], owner[want]

    # -- io ----------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, atom_id=self.atom_id, node_ptr=self.node_ptr,
            l2_id=self.l2_id, pos_nm=self.pos_nm, max_dt_nm=self.max_dt_nm,
            resolved=self.resolved,
            meta=np.frombuffer(json.dumps(self.meta).encode(), np.uint8))


def load_object_geometry(path: str | Path) -> ObjectGeometry:
    with np.load(Path(path), allow_pickle=False) as z:
        return ObjectGeometry(
            atom_id=z["atom_id"], node_ptr=z["node_ptr"], l2_id=z["l2_id"],
            pos_nm=z["pos_nm"], max_dt_nm=z["max_dt_nm"],
            resolved=z["resolved"],
            meta=json.loads(bytes(z["meta"]).decode()) if "meta" in z else {})


def endpoint_points(topology_npz: str | Path) -> dict[int, np.ndarray]:
    """Per-atom endpoint positions from a topology file, for comparison.

    The same shape as :meth:`ObjectGeometry.points`, so a measurement can be run
    twice over the two point sets with one code path and no other difference.
    """
    with np.load(Path(topology_npz), allow_pickle=False) as z:
        ep_atom, ep_pos = z["ep_atom"], z["ep_pos_nm"]
    o = np.argsort(ep_atom, kind="stable")
    ea, ep = ep_atom[o], ep_pos[o]
    ua, starts = np.unique(ea, return_index=True)
    ends = np.r_[starts[1:], len(ea)]
    out = {}
    for a, s, e in zip(ua.tolist(), starts, ends):
        P = ep[s:e]
        out[int(a)] = P[np.isfinite(P).all(axis=1)]
    return out


def min_gap_between(A: np.ndarray, B: np.ndarray) -> float:
    """Closest approach between two point sets, inf if either is empty."""
    if not len(A) or not len(B):
        return float("inf")
    return float(cKDTree(B).query(A, k=1)[0].min())
