"""EXP-089 -- does the where-to-edit prior survive v117-only features?

EXP-082 fitted a where-to-edit prior on the human proofreading log and reached
held-out-by-cell area under the curve 0.779 with a 3.9x lift, "dominated by
caliber". Every feature it used was read off the **final, proofread**
reconstruction: skeleton radius, compartment (axon / dendrite / soma) and path
distance from the soma all come from the finished cell. EXP-082 says so plainly
in its own limits -- "the Q1 model is measured, but its deployment on v117
fragments is assumed, not shown" -- and its usable-signal table names exactly
one thing the prior needs before it can be spent: "recompute radius from v117
fragments instead of the proofread skeleton."

That substitution is not cosmetic. Caliber measured on a proofread skeleton and
caliber measured on a v117 fragment are different quantities: the proofread
skeleton is a single soma-rooted tree through tissue a human has already decided
belongs together, and its radius is estimated with that whole tree in hand. A
grower has fragments. If the prior is a property of *tissue*, as EXP-082 argues,
it should survive; if it is partly a property of the finished reconstruction,
the 0.779 is not a number a grower can have. This experiment measures which.

What is substituted, and what is not
------------------------------------
The **evaluation lattice, the label and the split are held fixed** at EXP-082's,
so the two areas under the curve are comparable: one row per vertex of the final
proofread skeleton, positive when a merge endpoint landed within 2 um of it,
five folds grouped by cell so every score is on a cell the model never saw. Only
the *features* change. Holding the lattice fixed is itself a limit and is stated
below rather than hidden: a real grower would score v117 fragment nodes, not
proofread skeleton vertices, and that is a different experiment.

Feature by feature, proofread -> v117:

===================  ==========================================================
``radius_nm``        ``radius_v117_nm`` -- EXP-088's caliber measurement on the
                     containing v117 fragment. **Imported, never reimplemented**
                     (see the dependency note below).
``path_soma_um``     ``path_in_piece_um`` -- geodesic distance inside the
                     vertex's own v117 fragment, from that fragment's entry node
                     (its point of closest approach to the seed soma). A grower
                     can walk a fragment it holds; it cannot walk cable it has
                     not yet joined.
``euclid_soma_um``   unchanged. A soma-seeded grower knows its seed's position,
                     so straight-line distance to it needs no substitution. It
                     is listed in the drop table as surviving *by construction*,
                     not as a measured survival.
``degree``           ``degree_v117`` -- degree inside the v117 fragment. Differs
                     from the proofread degree at every fragment boundary, which
                     is exactly where the humans acted.
``x/y/z_um``         unchanged; position is position.
``is_axon``          **dropped.** This repository has no compartment predictor
                     that runs on a v117 fragment. Substituting the true
                     pcg_skel compartment label would smuggle the proofread
                     reconstruction back in, so it is dropped and the cost of
                     dropping it is measured directly: a seven-column proofread
                     arm (EXP-082 minus ``is_axon``) is fitted alongside, so the
                     cost of losing compartment and the cost of substituting
                     caliber are separated rather than pooled.
===================  ==========================================================

The control that makes the comparison interpretable
---------------------------------------------------
EXP-082's own eight-column feature set is refitted here, under this harness, on
the same cells. If it does not land within +/- 0.03 of 0.779 then the
reimplementation is the difference and no statement about the substitution is
warranted -- the run reports that and fails, rather than attributing the gap to
v117. This control is not optional and is not omitted.

Dependency on EXP-088's caliber module
--------------------------------------
The v117 caliber measurement belongs to EXP-088
(``neuronauts/experiments/exp088_conservation_joins.py`` plus a caliber module
under ``neuronauts/harness/`` or ``neuronauts/metrics/``). **It was not present
when this module was written**, so this module imports it and does not implement
it. The assumed interface is deliberately one function::

    fragment_caliber_nm(points_nm: np.ndarray) -> np.ndarray
        points_nm : [N, 3] float, the node positions of ONE v117 fragment, nm
        returns   : [N]    float, a radius estimate at each of those N points, nm

resolved from ``neuronauts.harness.caliber`` or ``neuronauts.metrics.caliber``.
Importing this module always succeeds; the missing dependency surfaces at
``run`` time as a ``ModuleNotFoundError`` that names both candidate paths and
the expected signature. A per-fragment scalar is refused rather than broadcast:
EXP-082's caliber signal is a *within-arbor* gradient (26x top to bottom inside
axon alone), so one number per fragment would not be the same feature and
silently accepting it would fake the result.

How the v117 fragmentation is obtained
--------------------------------------
Each cached skeleton carries ``lvl2_ids``; level-2 node ids are agglomeration
independent, so mapping them through the chunkedgraph at the v117 timestamp
gives, for every skeleton vertex, the v117 object that covered that tissue
before the humans edited. Vertices whose v117 roots agree and that are adjacent
on the skeleton form one v117 *piece*: the connected run of this cell's cable
that v117 already held together. ``scripts/probe_v117_geometry_route.py`` does
the same ``roots_at(..., V117_TIMESTAMP)`` call on level-2 ids and got usable
answers, which is why that route is used; the mapping is cached per cell under
``data/external/v117_fragment_map/`` so a second run needs no network.

Limits this run does not paper over
-----------------------------------
- The lattice is the proofread skeleton (see above). Only the features are v117.
- A piece is a fragment *restricted to this cell's cable*. A v117 object that
  also spans a neighbouring cell, or that leaves the cached bounds, is measured
  smaller here than it really is. That biases ``piece_cable_um`` and
  ``piece_reach_um``, which is why they sit in the reported ``v117_plus`` arm
  and not in the arm the bar is set on.
- Caliber is evaluated at the proofread skeleton's vertex positions. Those
  positions sample the fragment's medial axis, but the sampling was chosen by a
  skeletonization of the finished cell.
- The 103 cells are gold-proofread and atypical, exactly as EXP-082 warned.

    python -m neuronauts.experiments.exp089_edit_prior_v117
"""

from __future__ import annotations

import glob
import importlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from neuronauts.data import lineage as L
from neuronauts.experiments._runner import Context, Outcome, Spec, main
from neuronauts.metrics.ranking import roc_auc

# --- inputs ----------------------------------------------------------------
#: pcg_skel v4 skeletons of the gold cells (scripts/fetch_seed_skeletons.py).
SKELETON_DIR = "data/external/cell_skeletons"
#: Tabular change logs (scripts/fetch_edit_history.py) AFTER enrichment with
#: per-operation coordinates (attic/one_off_analyses/fetch_edit_locations.py,
#: which stamps ``located: true``). Without the enrichment there are no
#: ``edit_points_nm`` and there is no label.
EDIT_DIR = "data/external/edit_history"
#: Built on demand by this module and cached, not a declared input.
FRAGMENT_CACHE = "data/external/v117_fragment_map"
#: Optional diagnostic rung: pooled level-2 attributes including the raw
#: distance transform (scripts/fetch_cell_l2_positions.py). Skipped if absent.
L2_ATTR_CACHE = "data/external/soma_viz/connective_l2_attrs.npz"

# --- protocol, held identical to EXP-082 -----------------------------------
MERGE_MATCH_NM = 2000.0        # EXP-082's d_skel < 2000 label rule
N_FOLDS = 5                    # GroupKFold(5), groups = cell root id
HEADLINE_ITERS = 150           # EXP-082 model.py: max_iter=150
ABLATION_ITERS = 80            # EXP-082 model.py used 80 for its ablations
MAX_BINS = 64
SEED = 0
COMPARTMENT_SOMA = 1           # pcg_skel v4 codes
COMPARTMENT_AXON = 2

# --- the bar, declared before the run --------------------------------------
EXP082_AUC = 0.779             # the proofread-feature number being reproduced
CONTROL_TOL = 0.03
V117_AUC_BAR = 0.70
#: Above this fraction of vertices unmapped to a v117 root, the run still
#: reports but says loudly that the fragmentation is incomplete.
UNMAPPED_WARN_FRAC = 0.05

# --- EXP-088's caliber module ----------------------------------------------
CALIBER_CANDIDATES = (
    ("neuronauts.harness.caliber", "fragment_caliber_nm"),
    ("neuronauts.metrics.caliber", "fragment_caliber_nm"),
)


def load_caliber_estimator() -> tuple[Callable, str]:
    """EXP-088's v117 caliber measurement, imported rather than reimplemented.

    Raises a ``ModuleNotFoundError`` that names both candidate paths and the
    expected signature. Deliberately called from ``run`` and not at import time,
    so ``import neuronauts.experiments.exp089_edit_prior_v117`` succeeds while
    the dependency is still being written and the failure, when it comes, is
    explained rather than silent.
    """
    tried: list[str] = []
    for mod_name, fn_name in CALIBER_CANDIDATES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as exc:
            tried.append(f"{mod_name}: {type(exc).__name__}: {exc}")
            continue
        fn = getattr(mod, fn_name, None)
        if fn is None:
            tried.append(f"{mod_name}: imported but has no {fn_name}(); "
                         f"exports {sorted(n for n in vars(mod) if not n.startswith('_'))[:12]}")
            continue
        return fn, f"{mod_name}.{fn_name}"
    raise ModuleNotFoundError(
        "EXP-089 needs EXP-088's v117 caliber measurement and does not "
        "implement its own. Expected one function\n"
        "    fragment_caliber_nm(points_nm: [N,3] float ndarray, nm) -> [N] "
        "float ndarray of radius estimates, nm\n"
        "at one of " + " or ".join(m for m, _ in CALIBER_CANDIDATES) + ".\n"
        "Tried:\n  " + "\n  ".join(tried) + "\n"
        "If EXP-088 named it differently, add the (module, function) pair to "
        "CALIBER_CANDIDATES rather than writing a second caliber estimator: "
        "two independent caliber definitions would make this experiment's "
        "comparison against EXP-082 uninterpretable.")


def _caliber_of_piece(fn: Callable, points_nm: np.ndarray) -> np.ndarray:
    """Per-point radius for one v117 piece, with the contract enforced."""
    out = np.asarray(fn(points_nm), dtype=np.float64)
    if out.ndim == 0 or out.shape != (len(points_nm),):
        raise ValueError(
            f"caliber estimator returned shape {out.shape} for a piece of "
            f"{len(points_nm)} points; EXP-089 needs one radius PER POINT. "
            "EXP-082's caliber signal is a within-arbor gradient (26x top to "
            "bottom inside axon alone), so a single number per fragment is a "
            "different feature and is refused rather than broadcast.")
    return out


# ---------------------------------------------------------------------------
# per-cell substrate
# ---------------------------------------------------------------------------

def _vertex_level2(z, n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """``(vertex_index, level2_id)`` pairs, or raise saying why it cannot.

    pcg_skel stores ``lvl2_ids`` either one per skeleton vertex or one per
    level-2 node with ``mesh_to_skel_map`` giving the skeleton vertex. Both are
    handled; anything else fails loudly with the shapes, because guessing here
    would silently mis-assign the v117 fragmentation.
    """
    files = set(z.files)
    if "lvl2_ids" not in files:
        raise KeyError(
            "skeleton cache has no 'lvl2_ids', so no vertex can be mapped to a "
            "v117 object. scripts/fetch_seed_skeletons.py keeps it; re-fetch, "
            "and add 'mesh_to_skel_map' to its KEEP tuple while doing so.")
    lv = np.asarray(z["lvl2_ids"]).ravel().astype(np.uint64)
    if len(lv) == n_vertices:
        return np.arange(n_vertices, dtype=np.int64), lv
    if "mesh_to_skel_map" in files:
        m2s = np.asarray(z["mesh_to_skel_map"]).ravel().astype(np.int64)
        if len(m2s) == len(lv):
            ok = (m2s >= 0) & (m2s < n_vertices)
            return m2s[ok], lv[ok]
    raise ValueError(
        f"lvl2_ids has length {len(lv)} but the skeleton has {n_vertices} "
        f"vertices, and no usable mesh_to_skel_map is cached "
        f"(files: {sorted(files)}). Re-fetch the skeletons with "
        f"'mesh_to_skel_map' included.")


def _v117_root_per_vertex(root: int, z, n_vertices: int, cache_dir: Path,
                          token: Optional[str]) -> np.ndarray:
    """v117 object id for each skeleton vertex; 0 where unmapped.

    Cached per cell. The mapping is level-2 id -> root at ``V117_TIMESTAMP``;
    a vertex backed by several level-2 nodes takes the majority v117 root.
    """
    vidx, l2 = _vertex_level2(z, n_vertices)
    uniq = np.unique(l2)
    if not len(uniq):
        return np.zeros(n_vertices, np.uint64)
    cache = cache_dir / f"{root}_v117.npz"
    have_id = np.zeros(0, np.uint64)
    have_root = np.zeros(0, np.uint64)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as c:
            have_id, have_root = c["l2_id"], c["v117_root"]
    missing = np.setdiff1d(uniq, have_id)
    if len(missing):
        got = L.roots_at(missing, L.V117_TIMESTAMP, token=token) if token else \
            L.roots_at(missing, L.V117_TIMESTAMP)
        if got is None:
            raise RuntimeError(
                f"cell {root}: roots_at() returned None for {len(missing):,} "
                "level-2 ids at the v117 timestamp. That is a failed fetch, not "
                "an empty answer -- it is reported rather than filled with "
                "zeros, because a zero here would silently become a wrong "
                "fragmentation. Check the CAVE token and rerun; the per-cell "
                "cache makes the rerun cheap.")
        have_id = np.concatenate([have_id, missing])
        have_root = np.concatenate([have_root, got.astype(np.uint64)])
        order = np.argsort(have_id)
        have_id, have_root = have_id[order], have_root[order]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, l2_id=have_id, v117_root=have_root)

    j = np.searchsorted(have_id, l2)
    j = np.clip(j, 0, len(have_id) - 1)
    frag_of_pair = np.where(have_id[j] == l2, have_root[j], np.uint64(0))

    # majority v117 root per vertex (usually a single level-2 node per vertex)
    per_vertex = np.zeros(n_vertices, np.uint64)
    keep = frag_of_pair > 0
    if keep.any():
        vk, fk = vidx[keep], frag_of_pair[keep]
        order = np.lexsort((fk, vk))
        vk, fk = vk[order], fk[order]
        # run-length encode (vertex, frag) and take the longest run per vertex
        new = np.r_[True, (vk[1:] != vk[:-1]) | (fk[1:] != fk[:-1])]
        starts = np.flatnonzero(new)
        lengths = np.diff(np.r_[starts, len(vk)])
        rv, rf = vk[starts], fk[starts]
        best_len = np.zeros(n_vertices, np.int64)
        for v, f, ln in zip(rv.tolist(), rf.tolist(), lengths.tolist()):
            if ln > best_len[v]:
                best_len[v], per_vertex[v] = ln, f
    return per_vertex


def _pieces(edges: np.ndarray, frag: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected runs of cable that share a v117 object id.

    ``frag == 0`` (unmapped) never joins anything, so those vertices come back
    as singleton components and are dropped by the caller rather than being
    quietly attached to a neighbour.
    """
    n = len(frag)
    same = (frag[edges[:, 0]] == frag[edges[:, 1]]) & (frag[edges[:, 0]] > 0)
    e = edges[same]
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    n_comp, comp = connected_components(A, directed=False)
    return comp.astype(np.int64), int(n_comp)


def _weighted_graph(V: np.ndarray, edges: np.ndarray) -> tuple[coo_matrix, np.ndarray]:
    n = len(V)
    w = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
    A = coo_matrix((np.r_[w, w],
                    (np.r_[edges[:, 0], edges[:, 1]],
                     np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    return A, w


def _cell_features(root: int, skel_path: Path, edit_path: Path, cache_dir: Path,
                   caliber: Callable, token: Optional[str]) -> Optional[dict]:
    """One cell's vertex table: EXP-082's proofread features and the v117 ones."""
    with np.load(skel_path, allow_pickle=False) as z:
        V = z["vertices"].astype(np.float64)
        E = z["edges"].astype(np.int64)
        radius = z["radius"].astype(np.float64)
        comp = z["compartment"].astype(np.int64)
        frag = _v117_root_per_vertex(root, z, len(V), cache_dir, token)
    n = len(V)

    soma_rows = np.flatnonzero(comp == COMPARTMENT_SOMA)
    if not len(soma_rows):
        return None
    soma = int(soma_rows[0])

    # --- proofread arm, exactly EXP-082's build_join.py ---------------------
    A, w = _weighted_graph(V, E)
    path_soma = dijkstra(A, indices=soma, directed=False)
    euclid_soma = np.linalg.norm(V - V[soma], axis=1)
    degree = np.bincount(E.ravel(), minlength=n).astype(np.float64)
    cable = np.zeros(n)
    np.add.at(cable, E[:, 0], w / 2)
    np.add.at(cable, E[:, 1], w / 2)

    # --- v117 arm -----------------------------------------------------------
    piece, _ = _pieces(E, frag)
    mapped = frag > 0
    radius_v117 = np.full(n, np.nan)
    path_in_piece = np.full(n, np.nan)
    piece_cable = np.zeros(n)
    piece_reach = np.zeros(n)

    same = (frag[E[:, 0]] == frag[E[:, 1]]) & (frag[E[:, 0]] > 0)
    Ein = E[same]
    Ain, win = _weighted_graph(V, Ein)
    degree_v117 = np.bincount(Ein.ravel(), minlength=n).astype(np.float64)
    cable_in = np.zeros(n)
    np.add.at(cable_in, Ein[:, 0], win / 2)
    np.add.at(cable_in, Ein[:, 1], win / 2)

    rows_of: dict[int, np.ndarray] = {}
    mrows = np.flatnonzero(mapped)
    if len(mrows):
        mrows = mrows[np.argsort(piece[mrows], kind="stable")]
        mpiece = piece[mrows]
        starts = np.flatnonzero(np.r_[True, mpiece[1:] != mpiece[:-1]])
        ends = np.r_[starts[1:], len(mrows)]
        rows_of = {int(mpiece[s]): mrows[s:e] for s, e in zip(starts, ends)}

    entries: list[int] = []
    for p, rows in rows_of.items():
        radius_v117[rows] = _caliber_of_piece(caliber, V[rows])
        entries.append(int(rows[np.argmin(euclid_soma[rows])]))
        piece_cable[rows] = cable_in[rows].sum()
    if entries:
        # sources sit in disjoint components, so min_only gives each vertex the
        # distance from its OWN piece's entry node.
        d = dijkstra(Ain, indices=np.asarray(entries), directed=False,
                     min_only=True)
        path_in_piece[mapped] = d[mapped]
        for rows in rows_of.values():
            finite = d[rows][np.isfinite(d[rows])]
            piece_reach[rows] = float(finite.max()) if len(finite) else 0.0

    # --- label, exactly EXP-082's rule --------------------------------------
    rec = json.loads(edit_path.read_text())
    if not rec.get("located"):
        raise ValueError(
            f"{edit_path} has no per-operation coordinates ('located' is not "
            "true). Run attic/one_off_analyses/fetch_edit_locations.py over "
            "data/external/edit_history first; without it there is no label.")
    pts = [p for o in rec["ops"] if o.get("is_merge")
           for p in (o.get("edit_points_nm") or [])]
    label = np.zeros(n, bool)
    n_merge_pts = len(pts)
    n_matched = 0
    if pts:
        dd, ii = cKDTree(V).query(np.asarray(pts, float))
        hit = dd < MERGE_MATCH_NM
        n_matched = int(hit.sum())
        label[ii[hit]] = True

    return dict(
        root=root, n=n, soma=soma, label=label, mapped=mapped,
        radius=radius, is_axon=(comp == COMPARTMENT_AXON).astype(np.float64),
        path_soma=path_soma, euclid_soma=euclid_soma, degree=degree,
        pos=V, cable=cable,
        radius_v117=radius_v117, path_in_piece=path_in_piece,
        degree_v117=degree_v117, piece_cable=piece_cable,
        piece_reach=piece_reach, frag=frag, piece=piece,
        n_pieces=int(len(np.unique(piece[mapped]))) if mapped.any() else 0,
        n_merge_points=n_merge_pts, n_merge_points_matched=n_matched,
    )


# ---------------------------------------------------------------------------
# columns and feature sets
# ---------------------------------------------------------------------------

COLS = [
    "radius_nm",          # 0  proofread
    "is_axon",            # 1  proofread
    "path_soma_um",       # 2  proofread
    "euclid_soma_um",     # 3  shared -- v117-available without substitution
    "degree",             # 4  proofread
    "x_um",               # 5  shared
    "y_depth_um",         # 6  shared
    "z_um",               # 7  shared
    "radius_v117_nm",     # 8  v117
    "path_in_piece_um",   # 9  v117
    "degree_v117",        # 10 v117
    "piece_cable_um",     # 11 v117, context only
    "piece_reach_um",     # 12 v117, context only
    "radius_l2dt_raw_nm", # 13 optional diagnostic rung
]
IDX = {c: i for i, c in enumerate(COLS)}

#: EXP-082's eight columns, in EXP-082's order. The control.
SET_PROOFREAD = ["radius_nm", "is_axon", "path_soma_um", "euclid_soma_um",
                 "degree", "x_um", "y_depth_um", "z_um"]
#: The same minus compartment, so "cost of dropping is_axon" and "cost of
#: substituting caliber" are separable rather than pooled.
SET_PROOFREAD_NO_AXON = [c for c in SET_PROOFREAD if c != "is_axon"]
#: The arm the bar is set on: one v117 substitute per surviving EXP-082 column.
SET_V117 = ["radius_v117_nm", "path_in_piece_um", "euclid_soma_um",
            "degree_v117", "x_um", "y_depth_um", "z_um"]
#: Reported, not gated: fragment context a grower also has, but which EXP-082
#: had no counterpart for, and which the piece-truncation limit biases.
SET_V117_PLUS = SET_V117 + ["piece_cable_um", "piece_reach_um"]
#: Reported, not gated, and skipped when the level-2 attribute cache is absent:
#: the raw distance transform at the vertex's own level-2 node, with no
#: estimator on top. Separates "any v117 caliber" from "EXP-088's estimator".
SET_V117_RAW_DT = ["radius_l2dt_raw_nm"] + SET_V117[1:]

#: proofread column -> its v117 counterpart, or None when it is dropped.
SUBSTITUTION = [
    ("radius_nm", "radius_v117_nm"),
    ("path_soma_um", "path_in_piece_um"),
    ("euclid_soma_um", "euclid_soma_um"),
    ("degree", "degree_v117"),
    ("x_um", "x_um"),
    ("y_depth_um", "y_depth_um"),
    ("z_um", "z_um"),
    ("is_axon", None),
]


def _cv(X, y, groups, cols, n_iter) -> np.ndarray:
    """Out-of-fold scores, five folds grouped by cell -- EXP-082's protocol."""
    j = [IDX[c] for c in cols]
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, y, groups=groups):
        m = HistGradientBoostingClassifier(max_iter=n_iter, random_state=SEED,
                                           max_bins=MAX_BINS)
        m.fit(X[tr][:, j], y[tr])
        out[te] = m.predict_proba(X[te][:, j])[:, 1]
    return out


def _fmt(v) -> str:
    return "unmeasured" if v is None else f"{v:.3f}"


def _operating_point(y: np.ndarray, score: np.ndarray) -> dict:
    """EXP-082's operating point: flag as many vertices as there are positives."""
    k = int(y.sum())
    order = np.argsort(-score)
    base = float(y.mean())
    prec = float(y[order[:k]].mean()) if k else float("nan")
    row = {"auc": round(float(roc_auc(y, score)), 6),
           "base_rate": round(base, 6), "k": k,
           "precision_at_k": round(prec, 6),
           "lift_at_k": round(prec / base, 4) if base > 0 else float("nan")}
    for f in (0.02, 0.10, 0.30):
        kk = int(f * len(y))
        oo = order[:kk]
        row[f"recall_top{int(f*100)}pct"] = round(float(y[oo].sum() / max(y.sum(), 1)), 6)
        row[f"precision_top{int(f*100)}pct"] = round(float(y[oo].mean()), 6)
    return row


# ---------------------------------------------------------------------------
SPEC = Spec(
    id="EXP-089",
    title="Where-to-edit prior on v117-only features",
    question="Does EXP-082's where-to-edit prior survive when every feature is "
             "computed from what a grower actually has -- v117 fragments -- "
             "instead of from the final proofread reconstruction?",
    criterion=(
        f"Same cells, same lattice (one row per final-skeleton vertex), same "
        f"label (a merge endpoint within {MERGE_MATCH_NM/1000:.0f} um) and the "
        f"same held-out-by-cell protocol ({N_FOLDS}-fold GroupKFold on cell "
        f"root id) as EXP-082, so the two numbers are comparable; only the "
        f"features change. PASS when (a) the v117-only arm -- caliber from "
        f"EXP-088's v117 measurement, path distance inside the vertex's own "
        f"v117 fragment, degree inside that fragment, position, and compartment "
        f"DROPPED rather than substituted with its true label -- reaches "
        f"held-out area under the curve >= {V117_AUC_BAR:.2f} against the "
        f"{EXP082_AUC} proofread-feature ceiling, AND (b) the proofread-feature "
        f"control refitted under this harness reproduces {EXP082_AUC} within "
        f"+/- {CONTROL_TOL}. If (b) fails, the reimplementation is the "
        f"difference and no claim about the substitution is made. The "
        f"feature-by-feature drop, proofread versus v117, is reported for every "
        f"column and is the result that matters more than the headline; the "
        f"v117_plus and raw-distance-transform arms are reported, not gated."),
    requires=[], requires_ran=[],
    inputs=[SKELETON_DIR, EDIT_DIR],
    params={"merge_match_nm": MERGE_MATCH_NM, "n_folds": N_FOLDS,
            "headline_iters": HEADLINE_ITERS, "ablation_iters": ABLATION_ITERS,
            "max_bins": MAX_BINS, "seed": SEED,
            "v117_auc_bar": V117_AUC_BAR, "exp082_auc": EXP082_AUC,
            "control_tol": CONTROL_TOL,
            "v117_timestamp": L.V117_TIMESTAMP,
            "feature_sets": {"proofread": SET_PROOFREAD,
                             "proofread_no_axon": SET_PROOFREAD_NO_AXON,
                             "v117": SET_V117,
                             "v117_plus": SET_V117_PLUS,
                             "v117_raw_dt": SET_V117_RAW_DT}},
    flags={"synthetic_fallback": False,
           "network": True,
           "labels_used_only_for_evaluation": False,
           "labels_used_for_training": "human proofreading merge endpoints, "
                                       "train folds only, grouped by cell"},
    budget_minutes=150,
)


def run(ctx: Context) -> Outcome:
    root = ctx.root
    caliber, caliber_src = load_caliber_estimator()
    print(f"  caliber estimator: {caliber_src} (EXP-088's, imported)", flush=True)
    token = os.environ.get("CAVE_TOKEN") or L.DEFAULT_TOKEN
    cache_dir = root / FRAGMENT_CACHE

    skels = sorted(glob.glob(str(root / SKELETON_DIR / "*_skv4.npz")))
    cells: list[dict] = []
    skipped: dict[str, str] = {}
    t0 = time.time()
    for sf in skels:
        r = int(os.path.basename(sf).split("_")[0])
        ef = root / EDIT_DIR / f"{r}.json"
        if not ef.exists():
            skipped[str(r)] = "no edit history"
            continue
        try:
            rec = _cell_features(r, Path(sf), ef, cache_dir, caliber, token)
        except (KeyError, ValueError, RuntimeError) as exc:
            skipped[str(r)] = f"{type(exc).__name__}: {exc}"
            print(f"  cell {r}: SKIPPED -- {type(exc).__name__}: "
                  f"{str(exc)[:160]}", flush=True)
            continue
        if rec is None:
            skipped[str(r)] = "no soma vertex in the skeleton"
            continue
        cells.append(rec)
        if len(cells) % 10 == 0:
            print(f"  {len(cells)} cells built ({time.time()-t0:.0f}s)", flush=True)
    if not cells:
        raise RuntimeError(
            "no cell produced a vertex table. Skipped reasons: "
            + json.dumps(skipped)[:2000])

    cat = lambda k: np.concatenate([c[k] for c in cells])          # noqa: E731
    y = cat("label")
    groups = np.concatenate([np.full(c["n"], c["root"], np.int64) for c in cells])
    mapped = cat("mapped")
    pos = np.concatenate([c["pos"] for c in cells])

    X = np.full((len(y), len(COLS)), np.nan)
    X[:, IDX["radius_nm"]] = cat("radius")
    X[:, IDX["is_axon"]] = cat("is_axon")
    X[:, IDX["path_soma_um"]] = cat("path_soma") / 1000.0
    X[:, IDX["euclid_soma_um"]] = cat("euclid_soma") / 1000.0
    X[:, IDX["degree"]] = cat("degree")
    X[:, IDX["x_um"]] = pos[:, 0] / 1000.0
    X[:, IDX["y_depth_um"]] = pos[:, 1] / 1000.0
    X[:, IDX["z_um"]] = pos[:, 2] / 1000.0
    X[:, IDX["radius_v117_nm"]] = cat("radius_v117")
    X[:, IDX["path_in_piece_um"]] = cat("path_in_piece") / 1000.0
    X[:, IDX["degree_v117"]] = cat("degree_v117")
    X[:, IDX["piece_cable_um"]] = cat("piece_cable") / 1000.0
    X[:, IDX["piece_reach_um"]] = cat("piece_reach") / 1000.0

    # --- optional raw distance-transform rung --------------------------------
    raw_dt_note = "not attempted"
    l2_cache = root / L2_ATTR_CACHE
    have_raw_dt = False
    if l2_cache.exists():
        try:
            with np.load(l2_cache, allow_pickle=False) as z:
                a_id, a_dt = z["l2_id"], z["max_dt_nm"]
            order = np.argsort(a_id)
            a_id, a_dt = a_id[order], a_dt[order]
            col = np.full(len(y), np.nan)
            off = 0
            for c in cells:
                with np.load(root / SKELETON_DIR / f"{c['root']}_skv4.npz",
                             allow_pickle=False) as z:
                    vidx, l2 = _vertex_level2(z, c["n"])
                j = np.clip(np.searchsorted(a_id, l2), 0, len(a_id) - 1)
                ok = a_id[j] == l2
                col[off + vidx[ok]] = a_dt[j[ok]]
                off += c["n"]
            X[:, IDX["radius_l2dt_raw_nm"]] = col
            have_raw_dt = bool(np.isfinite(col).mean() > 0.5)
            raw_dt_note = (f"level-2 distance transform present for "
                           f"{float(np.isfinite(col).mean()):.1%} of vertices")
        except (KeyError, ValueError) as exc:
            raw_dt_note = f"skipped: {type(exc).__name__}: {exc}"
    else:
        raw_dt_note = f"skipped: {L2_ATTR_CACHE} not present"
    print(f"  raw distance-transform rung: {raw_dt_note}", flush=True)

    # --- the two populations --------------------------------------------------
    # Everything comparing arms runs on the v117-mapped rows; the control also
    # runs on the full EXP-082 population, because THAT is the number 0.779 was
    # measured on and the +/- 0.03 check has to be against the same population.
    full = np.ones(len(y), bool)
    n_unmapped = int((~mapped).sum())
    frac_unmapped = float(n_unmapped / len(y))
    n_cells_full = len(cells)
    cells_mapped = sorted({int(g) for g in np.unique(groups[mapped])})

    print(f"  vertices {len(y):,}  positive {int(y.sum()):,} "
          f"(base {y.mean():.4f})  cells {n_cells_full}", flush=True)
    print(f"  v117-mapped vertices {int(mapped.sum()):,} "
          f"({1-frac_unmapped:.1%}), pieces "
          f"{sum(c['n_pieces'] for c in cells):,}", flush=True)

    def score(cols, sel, n_iter=HEADLINE_ITERS):
        s = _cv(X[sel], y[sel], groups[sel], cols, n_iter)
        return _operating_point(y[sel], s)

    results: dict[str, dict] = {}
    results["proofread_control_full"] = score(SET_PROOFREAD, full)
    print(f"  control (EXP-082 features, all cells): AUC "
          f"{results['proofread_control_full']['auc']:.3f} "
          f"vs {EXP082_AUC} +/- {CONTROL_TOL}", flush=True)

    for name, cols in (("proofread_control_mapped", SET_PROOFREAD),
                       ("proofread_no_axon_mapped", SET_PROOFREAD_NO_AXON),
                       ("v117", SET_V117),
                       ("v117_plus", SET_V117_PLUS)):
        results[name] = score(cols, mapped)
        print(f"  {name:<26} AUC {results[name]['auc']:.3f}  "
              f"lift@k {results[name]['lift_at_k']:.2f}x", flush=True)
    if have_raw_dt:
        results["v117_raw_dt"] = score(SET_V117_RAW_DT, mapped)
        print(f"  {'v117_raw_dt':<26} AUC {results['v117_raw_dt']['auc']:.3f}",
              flush=True)

    # --- the feature-by-feature drop -----------------------------------------
    single: dict[str, float] = {}
    for c in dict.fromkeys([c for c in SET_PROOFREAD] + SET_V117_PLUS
                           + (["radius_l2dt_raw_nm"] if have_raw_dt else [])):
        single[c] = round(float(roc_auc(y[mapped],
                                        _cv(X[mapped], y[mapped], groups[mapped],
                                            [c], ABLATION_ITERS))), 6)
        print(f"    single {c:<20} AUC {single[c]:.3f}", flush=True)

    drop_table = []
    for pf, vf in SUBSTITUTION:
        row = {"proofread_feature": pf, "v117_feature": vf,
               "proofread_single_auc": single.get(pf),
               "v117_single_auc": single.get(vf) if vf else None,
               "identical": bool(vf == pf)}
        if vf and single.get(pf) is not None and single.get(vf) is not None:
            row["delta"] = round(single[vf] - single[pf], 6)
        row["status"] = ("unchanged -- v117-available without substitution"
                         if vf == pf else
                         "dropped -- no compartment predictor runs on a v117 "
                         "fragment; cost measured by proofread_no_axon_mapped"
                         if vf is None else "substituted")
        drop_table.append(row)

    ablate = {}
    for tag, cols, sel in (("proofread", SET_PROOFREAD, mapped),
                           ("v117", SET_V117, mapped)):
        for c in cols:
            keep = [k for k in cols if k != c]
            a = float(roc_auc(y[sel], _cv(X[sel], y[sel], groups[sel], keep,
                                          ABLATION_ITERS)))
            ablate[f"{tag}/drop_{c}"] = round(a, 6)
            print(f"    ablate {tag}/{c:<20} AUC {a:.3f}", flush=True)

    # --- verdict --------------------------------------------------------------
    ctrl = results["proofread_control_full"]["auc"]
    v117_auc = results["v117"]["auc"]
    control_ok = bool(abs(ctrl - EXP082_AUC) <= CONTROL_TOL)
    bar_ok = bool(v117_auc >= V117_AUC_BAR)
    passed = bool(control_ok and bar_ok)

    if not control_ok:
        note = (f"CONTROL FAILED, so nothing is claimed about the substitution: "
                f"the EXP-082 feature set refitted here scores {ctrl:.3f}, "
                f"outside {EXP082_AUC} +/- {CONTROL_TOL}. The reimplementation "
                f"is the difference until that is explained; the v117 arm's "
                f"{v117_auc:.3f} is reported but is not evidence about v117.")
    else:
        note = (f"control reproduces EXP-082 at {ctrl:.3f} (target {EXP082_AUC} "
                f"+/- {CONTROL_TOL}), so the comparison is interpretable. "
                f"v117-only features reach {v117_auc:.3f} "
                f"(bar {V117_AUC_BAR:.2f}), against "
                f"{results['proofread_no_axon_mapped']['auc']:.3f} for the same "
                f"seven columns read off the proofread reconstruction: the "
                f"substitution costs "
                f"{results['proofread_no_axon_mapped']['auc'] - v117_auc:+.3f}. "
                f"Caliber alone: {_fmt(single.get('radius_nm'))} proofread "
                f"versus {_fmt(single.get('radius_v117_nm'))} v117 "
                f"(EXP-082 measured 0.750 for radius alone). "
                f"Compartment is dropped, "
                f"not substituted; its cost on the proofread side is "
                f"{results['proofread_control_mapped']['auc'] - results['proofread_no_axon_mapped']['auc']:+.3f}.")
    if frac_unmapped > UNMAPPED_WARN_FRAC:
        note += (f" WARNING: {frac_unmapped:.1%} of vertices carry no v117 root "
                 f"and were excluded from both arms of the comparison; the "
                 f"fragmentation is incomplete and the v117 number is measured "
                 f"on the mapped remainder.")

    return Outcome(
        passed=passed,
        observed={
            "v117_auc": v117_auc,
            "proofread_control_auc_full": ctrl,
            "proofread_control_auc_mapped": results["proofread_control_mapped"]["auc"],
            "proofread_no_axon_auc_mapped": results["proofread_no_axon_mapped"]["auc"],
            "control_reproduces_exp082": control_ok,
            "v117_lift_at_k": results["v117"]["lift_at_k"],
            "proofread_lift_at_k": results["proofread_control_mapped"]["lift_at_k"],
            "substitution_cost_auc": round(
                results["proofread_no_axon_mapped"]["auc"] - v117_auc, 6),
            "feature_drop": drop_table,
            "single_feature_auc": single,
            "ablation_auc": ablate,
            "n_cells": n_cells_full,
            "n_cells_v117_mapped": len(cells_mapped),
            "n_vertices": int(len(y)),
            "n_vertices_v117_mapped": int(mapped.sum()),
            "n_positive": int(y.sum()),
            "base_rate": round(float(y.mean()), 6),
            "split": (f"GroupKFold({N_FOLDS}) on cell root id -- every score is "
                      f"out-of-fold on a cell the model never saw; identical to "
                      f"EXP-082"),
            "caliber_source": caliber_src,
        },
        population={
            "n_cells_with_skeleton_and_edits": n_cells_full,
            "n_cells_skipped": len(skipped),
            "skipped_reasons": skipped,
            "n_vertices": int(len(y)),
            "n_vertices_unmapped_to_v117": n_unmapped,
            "frac_vertices_unmapped": round(frac_unmapped, 6),
            "n_v117_pieces": int(sum(c["n_pieces"] for c in cells)),
            "pieces_per_cell_median": float(np.median(
                [c["n_pieces"] for c in cells])),
            "n_merge_points": int(sum(c["n_merge_points"] for c in cells)),
            "n_merge_points_matched": int(sum(c["n_merge_points_matched"]
                                              for c in cells)),
            "merge_match_nm": MERGE_MATCH_NM,
            "raw_dt_rung": raw_dt_note,
        },
        tables={"by_feature_set": results, "feature_columns": COLS,
                "substitution": [{"proofread": a, "v117": b}
                                 for a, b in SUBSTITUTION]},
        note=note,
    )


if __name__ == "__main__":
    raise SystemExit(main(SPEC, run))
