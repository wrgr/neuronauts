"""Adapters: this project's on-disk result formats -> :class:`SkeletonGeometry`.

Every format below already exists elsewhere in the repo; nothing here fetches
data, it only reads what another stage already wrote.

====================================  =========================================
Source                                Loader
====================================  =========================================
v117 atom, real L2 adjacency          :class:`HarnessAtomGeometry`
kimimaro skeleton archive (box-hash)  :func:`kimimaro_archive_skeletons`
``fetch.load_skeleton`` / CAVE dict   ``SkeletonGeometry.from_dict`` (skeleton.py)
``neuronauts.schemas.Fragment`` /
``global_merge.schemas.SegmentFragment``   ``SkeletonGeometry.from_fragment`` (skeleton.py)
====================================  =========================================

Grouping (which segments belong to one assembled result) is likewise read,
never invented: :func:`groups_from_fragment_to_neuron` and
:func:`groups_from_neuron_hypotheses` turn assembly outputs already produced by
``treestitch``/``neuronauts.assemble``/``neuronauts.global_merge`` into the
``{segment_id: group}`` mapping :func:`neuronauts.meshing.bundle.export_bundle`
expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from neuronauts.meshing.skeleton import SkeletonGeometry

# ---------------------------------------------------------------------------
# v117 atoms: real L2 adjacency + pooled attributes (neuronauts.harness.*)
# ---------------------------------------------------------------------------


class HarnessAtomGeometry:
    """Per-atom L2 skeleton geometry straight from the harness fetch.

    Wraps ``neuronauts.harness.geometry.AtomGeometryStore`` (per-atom L2 node
    ids + real ``lvl2_graph`` adjacency) and ``neuronauts.harness.topology.
    L2Attributes`` (pooled ``pos_nm`` / ``mean_dt_nm`` caliber). Position and
    caliber are looked up per atom lazily so opening this does not require the
    ~350 MB attribute cache to be re-read per atom.

    ``mean_dt_nm`` (the L2 chunk's distance-transform mean) is used as the
    tube radius: it is the real caliber this project already validated as
    "free" geometry (see ``docs/grammar_harness_handoff.md``), not a stand-in.
    An atom missing coordinates for some or all of its L2 nodes (the attribute
    fetch trails the topology fetch — see the same handoff doc) drops those
    nodes rather than plotting them at the origin; ``coverage`` on the result
    reports what fraction survived.
    """

    def __init__(self, geom_dir: str | Path, *, radius_col: str = "mean_dt_nm"):
        from neuronauts.harness.geometry import AtomGeometryStore
        from neuronauts.harness.topology import L2Attributes

        self.geom_dir = Path(geom_dir)
        self.store = AtomGeometryStore(self.geom_dir)
        attrs_path = self.geom_dir / "l2_attributes.npz"
        self.attrs = L2Attributes(attrs_path, cols=["pos_nm", radius_col]) \
            if attrs_path.exists() else None
        self.radius_col = radius_col
        self._all: dict[int, dict] | None = None

    def atom_ids(self) -> list[int]:
        return sorted(self.store.done_atoms())

    def _records(self) -> dict[int, dict]:
        if self._all is None:
            self._all = self.store.load_all()
        return self._all

    def skeleton(self, atom_id: int, *, default_radius_nm: float = 200.0
                ) -> tuple[SkeletonGeometry, float]:
        """Return ``(geometry, coverage)``; ``coverage`` is the fraction of
        the atom's L2 nodes that had a coordinate in the attribute cache."""
        rec = self._records().get(int(atom_id))
        if rec is None:
            raise KeyError(f"atom {atom_id} not found in {self.geom_dir}")
        l2_ids = rec["l2_ids"]
        if self.attrs is None:
            raise RuntimeError(
                f"no l2_attributes.npz under {self.geom_dir}; fetch attributes first "
                "(scripts/fetch_atom_geometry.py)")
        pos = self.attrs.take(l2_ids, "pos_nm")
        rad = self.attrs.take(l2_ids, self.radius_col)
        ok = np.isfinite(pos).all(axis=1)
        coverage = float(ok.mean()) if len(ok) else 0.0
        rad = np.where(np.isfinite(rad), rad, default_radius_nm)

        keep_idx = np.flatnonzero(ok)
        remap = np.full(len(l2_ids), -1, np.int64)
        remap[keep_idx] = np.arange(len(keep_idx))
        edges = rec["edges"]
        # l2_ids is not guaranteed sorted, so index explicitly rather than
        # np.searchsorted (which requires a sorted array).
        index = {int(v): k for k, v in enumerate(l2_ids.tolist())}
        e = np.asarray([(index[int(u)], index[int(v)]) for u, v in edges.tolist()],
                       dtype=np.int64).reshape(-1, 2) if len(edges) else np.zeros((0, 2), np.int64)
        e = remap[e]
        e = e[(e >= 0).all(axis=1)] if len(e) else e

        geom = SkeletonGeometry(
            vertices_nm=pos[keep_idx].astype(np.float32),
            edges=e,
            radii_nm=rad[keep_idx].astype(np.float32),
            node_ids=l2_ids[keep_idx],
        )
        return geom, coverage


def load_harness_atoms(
    geom_dir: str | Path,
    atom_ids: Iterable[int],
    *,
    radius_col: str = "mean_dt_nm",
    min_coverage: float = 0.0,
) -> tuple[dict[int, SkeletonGeometry], dict[int, float]]:
    """Batch wrapper: ``{atom_id: geometry}`` for atoms meeting ``min_coverage``.

    Returns a second dict of coverage per requested atom (including ones that
    were dropped) so a caller can report what was excluded and why.
    """
    src = HarnessAtomGeometry(geom_dir, radius_col=radius_col)
    out: dict[int, SkeletonGeometry] = {}
    coverage: dict[int, float] = {}
    for aid in atom_ids:
        geom, cov = src.skeleton(aid)
        coverage[int(aid)] = cov
        if cov >= min_coverage and geom.n_vertices:
            out[int(aid)] = geom
    return out, coverage


def top_atoms_by_synapse_count(population_npz: str | Path, n: int,
                               min_synapses: int = 1) -> list[int]:
    """Convenience: the ``n`` best-connected atoms in a harness population, a
    reasonable default subset to mesh (meshing all ~280k atoms is not)."""
    from neuronauts.harness.population import load_population

    pop = load_population(population_npz)
    keep = pop.n_synapses >= min_synapses
    order = np.argsort(-pop.n_synapses[keep])
    return [int(a) for a in pop.atom_id[keep][order][:n]]


# ---------------------------------------------------------------------------
# kimimaro skeleton archives (neuronauts.cell_graph.precompute_self_skeletons_for_cache)
# ---------------------------------------------------------------------------

def kimimaro_archive_skeletons(path: str | Path) -> dict[int, SkeletonGeometry]:
    """Every root skeleton in one box's kimimaro archive, keyed by root id.

    Archive layout (written by ``cell_graph.precompute_self_skeletons_for_cache``):
    ``root_ids`` [R], ``v_offsets`` [R+1] into the concatenated ``vertices``/
    ``edges`` (``n_verts``/``n_edges`` give each root's counts; edges are
    stored pre-offset into the shared vertex array), ``radii`` [V].
    """
    with np.load(Path(path), allow_pickle=False) as z:
        root_ids = z["root_ids"]
        v_off = z["v_offsets"]
        n_edges = z["n_edges"]
        verts, edges, radii = z["vertices"], z["edges"], z["radii"]

    e_off = np.zeros(len(n_edges) + 1, np.int64)
    np.cumsum(n_edges, out=e_off[1:])
    out: dict[int, SkeletonGeometry] = {}
    for i, rid in enumerate(root_ids.tolist()):
        vs, ve = int(v_off[i]), int(v_off[i + 1])
        es, ee = int(e_off[i]), int(e_off[i + 1])
        out[int(rid)] = SkeletonGeometry(
            vertices_nm=verts[vs:ve],
            edges=edges[es:ee] - vs,
            radii_nm=radii[vs:ve],
        )
    return out


# ---------------------------------------------------------------------------
# grouping: assembly outputs -> {segment_id: group}
# ---------------------------------------------------------------------------

def groups_from_fragment_to_neuron(fragment_to_neuron: Mapping[Any, Any],
                                   id_map: Mapping[Any, int] | None = None
                                   ) -> dict[int, str]:
    """``GlobalAssemblyResult.fragment_to_neuron`` (fragment id -> neuron id),
    both often strings — remapped to the integer segment ids meshed here via
    ``id_map`` (identity if the fragment ids are already the mesh's ids)."""
    out = {}
    for frag_id, neuron_id in fragment_to_neuron.items():
        sid = id_map[frag_id] if id_map is not None else int(frag_id)
        out[sid] = str(neuron_id)
    return out


def groups_from_neuron_hypotheses(hypotheses: Sequence[Any]) -> dict[int, str]:
    """``NeuronHypothesis`` list (``neuronauts.schemas`` / ``global_merge.schemas``)
    -> ``{fragment_id: neuron_id}``, reading whichever of ``fragment_ids`` /
    ``neuron_id`` the dataclass carries."""
    out: dict[int, str] = {}
    for h in hypotheses:
        gid = str(getattr(h, "neuron_id"))
        for fid in getattr(h, "fragment_ids"):
            out[int(fid) if isinstance(fid, (int, np.integer)) else fid] = gid
    return out


__all__ = [
    "HarnessAtomGeometry", "groups_from_fragment_to_neuron",
    "groups_from_neuron_hypotheses", "kimimaro_archive_skeletons",
    "load_harness_atoms", "top_atoms_by_synapse_count",
]
