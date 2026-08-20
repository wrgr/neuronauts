"""Level −1 atomization: over-segment fragments so cuts happen once, at the bottom.

Implements experiment 2 of ``docs/tree_assembly_algorithm.md``, extended with
the "odd component" idea: a frankenmerge's two halves are glued by an
*abnormally long* L2-MST bridge edge (the MST must jump between the two
neurons' L2 clouds), and that bridge usually sits mid-path — NOT at a branch
point.  So three split strategies are provided:

- **branch**    — split the skeleton at branch vertices (degree ≥ 3).
- **shatter**   — split at branch vertices AND cut "odd" edges (longer than
  ``max(long_edge_min_nm, long_edge_factor × median edge length)``).
- **odd**       — leave normal fragments whole; only fragments flagged *odd*
  (they contain an odd edge) get their odd edges cut (no branch splitting).

Every strategy converts the frankenmerge *cut* problem into a merge-abstention
problem: the halves become separate atoms and simply are not re-merged unless
the (spatially transferable) merge evidence says so.  ``oddness_scores`` /
``flag_odd_fragments`` also support the *skip* treatment: keep odd fragments
intact but exclude them from exact-identity evidence channels (same-atom
links), so a possibly-frankenmerged id never forces a merge.

``atomize_world`` rewrites a ``(fragments, region, root_label_map)`` world in
place-compatible form: observations keep their positions/labels/keys, but
``pre_seg_id``/``post_seg_id`` (whichever carries the fragment id) is rewritten
to fresh atom ids.  The returned parent map preserves the original v117 root of
every atom for evaluation (``frankenmerge_separation``).

numpy-only; no torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuronauts.schemas import Fragment


# ---------------------------------------------------------------------------
# Skeleton splitting
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, x: int) -> int:
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[ra] = rb


def odd_edge_mask(
    vertices_nm: np.ndarray,
    edges: np.ndarray,
    *,
    long_edge_factor: float = 4.0,
    long_edge_min_nm: float = 10_000.0,
) -> np.ndarray:
    """Boolean mask of "odd" (abnormally long) skeleton edges.

    An edge is odd when it is longer than BOTH ``long_edge_min_nm`` and
    ``long_edge_factor × median edge length``.  For a real L2-MST fragment the
    frankenmerge glue is exactly such an edge: the MST has to bridge the gap
    between the two constituent neurons' L2 clouds.
    """
    if len(edges) == 0:
        return np.zeros(0, dtype=bool)
    v = np.asarray(vertices_nm, dtype=np.float64)
    e = np.asarray(edges, dtype=np.int64)
    lens = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    med = float(np.median(lens))
    thresh = max(long_edge_min_nm, long_edge_factor * med)
    return lens > thresh


def split_fragment(
    frag: Fragment,
    *,
    at_branches: bool = True,
    cut_odd_edges: bool = False,
    long_edge_factor: float = 4.0,
    long_edge_min_nm: float = 10_000.0,
) -> list[dict]:
    """Split one fragment's skeleton into atoms.

    Atoms are connected groups of *edges*: two edges belong to the same atom
    when they share a vertex that is not a split point.  Split points are
    branch vertices (``at_branches``); odd edges are removed entirely
    (``cut_odd_edges``), so the components on either side separate.  Vertices
    at split points are duplicated into every adjacent atom (harmless
    geometric duplication).  Isolated vertices become singleton atoms.

    Returns a list of dicts ``{"vertices_nm", "edges", "radius_nm",
    "vert_index"}`` where ``vert_index`` maps atom-local vertices back to the
    parent fragment's vertex indices (used for observation reassignment).
    """
    verts = np.asarray(frag.vertices_nm, dtype=np.float32)
    edges = np.asarray(frag.edges, dtype=np.int64).reshape(-1, 2)
    radii = np.asarray(frag.radius_nm, dtype=np.float32)
    V, E = len(verts), len(edges)

    keep = np.ones(E, dtype=bool)
    if cut_odd_edges and E:
        keep &= ~odd_edge_mask(verts, edges,
                               long_edge_factor=long_edge_factor,
                               long_edge_min_nm=long_edge_min_nm)
    kept_edges = edges[keep]

    degree = np.zeros(V, dtype=np.int64)
    for u, v in kept_edges:
        degree[u] += 1
        degree[v] += 1
    split_vertex = (degree >= 3) if at_branches else np.zeros(V, dtype=bool)

    # Union kept edges that share a non-split vertex.
    uf = _UF(len(kept_edges))
    incident: dict[int, int] = {}
    for ei, (u, v) in enumerate(kept_edges):
        for vtx in (int(u), int(v)):
            if split_vertex[vtx]:
                continue
            if vtx in incident:
                uf.union(incident[vtx], ei)
            else:
                incident[vtx] = ei

    atoms: list[dict] = []
    if len(kept_edges):
        comp_of = {}
        for ei in range(len(kept_edges)):
            comp_of.setdefault(uf.find(ei), []).append(ei)
        for members in comp_of.values():
            e_sub = kept_edges[members]
            vert_index = np.unique(e_sub)
            remap = {int(g): l for l, g in enumerate(vert_index)}
            atoms.append({
                "vertices_nm": verts[vert_index],
                "edges": np.array([[remap[int(u)], remap[int(v)]]
                                   for u, v in e_sub], dtype=np.int64),
                "radius_nm": radii[vert_index],
                "vert_index": vert_index.astype(np.int64),
            })

    # Vertices touched by no kept edge → singleton atoms.
    covered = np.zeros(V, dtype=bool)
    for a in atoms:
        covered[a["vert_index"]] = True
    for vi in np.where(~covered)[0]:
        atoms.append({
            "vertices_nm": verts[[vi]],
            "edges": np.zeros((0, 2), dtype=np.int64),
            "radius_nm": radii[[vi]],
            "vert_index": np.array([vi], dtype=np.int64),
        })
    return atoms


# ---------------------------------------------------------------------------
# Oddness diagnostics
# ---------------------------------------------------------------------------

def oddness_scores(
    frag: Fragment,
    *,
    long_edge_factor: float = 4.0,
    long_edge_min_nm: float = 10_000.0,
) -> dict:
    """Label-free per-fragment oddness diagnostics.

    Returns ``max_edge_nm``, ``median_edge_nm``, ``max_over_median``,
    ``n_odd_edges``, ``is_odd``.  A fragment is odd when it contains at least
    one odd edge (see ``odd_edge_mask``) — the frankenmerge / bad-skeleton
    signature.
    """
    edges = np.asarray(frag.edges, dtype=np.int64).reshape(-1, 2)
    if len(edges) == 0:
        return {"max_edge_nm": 0.0, "median_edge_nm": 0.0,
                "max_over_median": 0.0, "n_odd_edges": 0, "is_odd": False}
    v = np.asarray(frag.vertices_nm, dtype=np.float64)
    lens = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    med = float(np.median(lens))
    odd = odd_edge_mask(frag.vertices_nm, edges,
                        long_edge_factor=long_edge_factor,
                        long_edge_min_nm=long_edge_min_nm)
    return {
        "max_edge_nm": float(lens.max()),
        "median_edge_nm": med,
        "max_over_median": float(lens.max() / med) if med > 0 else float("inf"),
        "n_odd_edges": int(odd.sum()),
        "is_odd": bool(odd.any()),
    }


def flag_odd_fragments(
    fragments: list[Fragment],
    *,
    long_edge_factor: float = 4.0,
    long_edge_min_nm: float = 10_000.0,
) -> set[int]:
    """Fragment ids (``base_root_id``) flagged odd by ``oddness_scores``."""
    return {
        int(f.base_root_id) for f in fragments
        if oddness_scores(f, long_edge_factor=long_edge_factor,
                          long_edge_min_nm=long_edge_min_nm)["is_odd"]
    }


# ---------------------------------------------------------------------------
# World rewrite
# ---------------------------------------------------------------------------

@dataclass
class AtomizedWorld:
    fragments: list[Fragment]          # atoms (fresh ids)
    region: object                     # Region with rewritten seg ids
    root_label_map: dict               # {atom_id: {labels}}
    atom_parent: dict                  # {atom_id: original fragment id}
    parent_ids_per_obs: np.ndarray     # [N] original fragment id per observation
    odd_parents: set                   # original fragment ids flagged odd


def atomize_world(
    fragments: list[Fragment],
    region,
    *,
    mode: str = "shatter",
    long_edge_factor: float = 4.0,
    long_edge_min_nm: float = 10_000.0,
    side: str = "pre",
) -> AtomizedWorld:
    """Rewrite a partition world so each fragment becomes its atoms.

    ``mode``:
      - ``"branch"``  — split at branch vertices only.
      - ``"shatter"`` — split at branch vertices AND cut odd edges.
      - ``"odd"``     — only odd fragments are touched: their odd edges are
        cut (no branch splitting anywhere).

    Observations are reassigned to the atom owning their nearest parent
    vertex.  Atoms that receive no observations are dropped (they cannot
    appear in the observation graph).  Original per-observation fragment ids
    are preserved in ``parent_ids_per_obs`` for evaluation.
    """
    from neuronauts.schemas import Region

    if mode not in ("branch", "shatter", "odd"):
        raise ValueError(f"unknown atomize mode: {mode!r}")

    seg_attr = f"{side}_seg_id"
    seg_ids = np.asarray(getattr(region, seg_attr), dtype=np.int64)
    pos = np.asarray(region.pre_pt_nm if side == "pre" else region.post_pt_nm,
                     dtype=np.float64)
    labels = np.asarray(region.pre_root_id if side == "pre"
                        else region.post_root_id, dtype=np.int64)

    odd_parents = flag_odd_fragments(
        fragments, long_edge_factor=long_edge_factor,
        long_edge_min_nm=long_edge_min_nm)

    new_seg = seg_ids.copy()
    atom_fragments: list[Fragment] = []
    root_label_map: dict[int, set[int]] = {}
    atom_parent: dict[int, int] = {}
    next_id = 1

    for frag in fragments:
        parent = int(frag.base_root_id)
        obs_idx = np.where(seg_ids == parent)[0]

        if mode == "odd" and parent not in odd_parents:
            atoms = None          # leave whole
        elif mode == "branch":
            atoms = split_fragment(frag, at_branches=True, cut_odd_edges=False)
        else:                     # "shatter", or "odd" on an odd fragment
            at_branches = (mode == "shatter")
            atoms = split_fragment(frag, at_branches=at_branches,
                                   cut_odd_edges=True,
                                   long_edge_factor=long_edge_factor,
                                   long_edge_min_nm=long_edge_min_nm)

        if atoms is None or len(atoms) <= 1:
            # unsplit: keep geometry, but re-key into the fresh atom id space
            aid = next_id
            next_id += 1
            atom_fragments.append(_rekey_fragment(frag, aid, obs_idx))
            new_seg[obs_idx] = aid
            atom_parent[aid] = parent
            if len(obs_idx):
                root_label_map[aid] = {int(x) for x in np.unique(labels[obs_idx])
                                       if x != 0}
            continue

        # nearest parent vertex per observation → owning atom
        verts = np.asarray(frag.vertices_nm, dtype=np.float64)
        vert_atom = np.full(len(verts), -1, dtype=np.int64)
        for ai, a in enumerate(atoms):
            for gv in a["vert_index"]:
                if vert_atom[gv] < 0:
                    vert_atom[gv] = ai
        if len(obs_idx):
            try:
                from scipy.spatial import cKDTree
                _, nearest = cKDTree(verts).query(pos[obs_idx])
            except ImportError:
                d = np.linalg.norm(pos[obs_idx][:, None, :] - verts[None], axis=2)
                nearest = np.argmin(d, axis=1)
            obs_atom = vert_atom[np.asarray(nearest, dtype=np.int64)]
        else:
            obs_atom = np.zeros(0, dtype=np.int64)

        for ai, a in enumerate(atoms):
            a_obs = obs_idx[obs_atom == ai]
            if len(a_obs) == 0:
                continue          # no observations → invisible to the graph
            aid = next_id
            next_id += 1
            atom_fragments.append(Fragment(
                fragment_id=aid,
                region_id=frag.region_id,
                base_root_id=aid,
                vertices_nm=a["vertices_nm"],
                edges=a["edges"],
                endpoints_nm=_leaf_vertices(a["vertices_nm"], a["edges"]),
                radius_nm=a["radius_nm"],
                synapse_indices=a_obs.astype(np.int64),
                dna=None,
            ).validate())
            new_seg[a_obs] = aid
            atom_parent[aid] = parent
            root_label_map[aid] = {int(x) for x in np.unique(labels[a_obs])
                                   if x != 0}

    new_region = Region(
        region_id=region.region_id + f"_atom_{mode}",
        bbox_nm=region.bbox_nm,
        voxel_size_nm=region.voxel_size_nm,
        seg_version=region.seg_version,
        label_version=region.label_version,
        pre_pt_nm=region.pre_pt_nm,
        post_pt_nm=region.post_pt_nm,
        pre_root_id=region.pre_root_id,
        post_root_id=region.post_root_id,
        synapse_id=region.synapse_id,
        pre_seg_id=new_seg if side == "pre" else region.pre_seg_id,
        post_seg_id=new_seg if side == "post" else region.post_seg_id,
    ).validate()

    return AtomizedWorld(
        fragments=atom_fragments,
        region=new_region,
        root_label_map=root_label_map,
        atom_parent=atom_parent,
        parent_ids_per_obs=seg_ids,
        odd_parents=odd_parents,
    )


def _rekey_fragment(frag: Fragment, new_id: int, obs_idx: np.ndarray) -> Fragment:
    return Fragment(
        fragment_id=new_id,
        region_id=frag.region_id,
        base_root_id=new_id,
        vertices_nm=frag.vertices_nm,
        edges=frag.edges,
        endpoints_nm=frag.endpoints_nm,
        radius_nm=frag.radius_nm,
        synapse_indices=np.asarray(obs_idx, dtype=np.int64),
        dna=None,
    ).validate()


def _leaf_vertices(verts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(verts) == 0 or len(edges) == 0:
        return np.asarray(verts, dtype=np.float32).copy()
    degree = np.zeros(len(verts), dtype=np.int64)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    leaves = np.where(degree <= 1)[0]
    out = np.asarray(verts, dtype=np.float32)
    return out[leaves] if len(leaves) else out[[0]]


# ---------------------------------------------------------------------------
# Frankenmerge separation metric
# ---------------------------------------------------------------------------

def frankenmerge_separation(
    pred: np.ndarray,
    true: np.ndarray,
    parent_ids: np.ndarray,
    *,
    ignore_true: int = 0,
) -> dict:
    """Fraction of frankenmerge parents whose halves end in disjoint clusters.

    A parent fragment is a frankenmerge when its observations carry ≥2 distinct
    true labels.  It counts as *separated* when, for every pair of its true
    labels, the sets of predicted clusters covering them are disjoint
    (abstained observations, pred < 0, are ignored).  This is the assembly-side
    analogue of the Phase-2.x ``fk_split`` diagnostic, computable with or
    without atomization via the preserved parent ids.
    """
    pred = np.asarray(pred, dtype=np.int64)
    true = np.asarray(true, dtype=np.int64)
    parent_ids = np.asarray(parent_ids, dtype=np.int64)

    n_sep = 0
    franken: list[int] = []
    for p in np.unique(parent_ids):
        m = (parent_ids == p) & (true != ignore_true)
        labs = np.unique(true[m])
        if len(labs) < 2:
            continue
        franken.append(int(p))
        clusters = []
        for lab in labs:
            c = set(pred[m & (true == lab)].tolist())
            c.discard(-1)
            c = {x for x in c if x >= 0}
            clusters.append(c)
        separated = all(
            clusters[a].isdisjoint(clusters[b])
            for a in range(len(clusters)) for b in range(a + 1, len(clusters))
        )
        if separated:
            n_sep += 1

    n_fk = len(franken)
    return {
        "fk_separation": (n_sep / n_fk if n_fk else float("nan")),
        "n_frankenmerges": n_fk,
        "n_separated": n_sep,
        "frankenmerge_parents": franken,
    }


__all__ = [
    "AtomizedWorld",
    "atomize_world",
    "split_fragment",
    "odd_edge_mask",
    "oddness_scores",
    "flag_odd_fragments",
    "frankenmerge_separation",
]
