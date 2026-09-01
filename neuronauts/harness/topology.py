"""Contract an atom's L2 adjacency into a topology view.

The raw fetch gives each atom a set of L2 nodes and the real adjacency between
them (``lvl2_graph``). That graph is mostly path-like -- mean degree ~2.3 -- so
almost all of its nodes carry no topological information: they are beads on a
wire. What the grammar actually reasons about is the *skeleton of the skeleton*:

  endpoint  (degree 1)  -- where an atom stops. These are the merge sites: a
                           false split shows up as two endpoints facing each
                           other, so candidate edges are generated here.
  branch    (degree>=3)  -- where an atom forks.
  segment                -- the unbranched run between two such nodes, carrying
                           its cable length and caliber. A short leaf segment
                           off a dendrite is a spine; a long one is a neurite.
  component              -- an atom can be disconnected inside the region
                           bounds, which the assembler must not assume away.

So this module reduces ~890 L2 nodes per atom to a few dozen junctions and
segments, keeps the geometry on the segments, and hands back a table small
enough to hold every atom in memory at once.

Coordinates and caliber come from the pooled L2 attribute cache and are
optional: topology alone is well defined without them, and the attribute fetch
finishes after the topology fetch. Anything length-derived is NaN until they
land.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# per-atom contraction
# ---------------------------------------------------------------------------

def build_adjacency(l2_ids: np.ndarray, edges: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Local CSR adjacency for one atom.

    Returns ``(indptr, indices, deg)`` over local node indices ``0..N-1``
    aligned to ``l2_ids`` **as given** (the caller's order is preserved so ids
    can be recovered by indexing). Self loops and duplicate edges are dropped:
    the chunk graph occasionally repeats a pair, and a self loop would inflate
    degree and invent a branch point.
    """
    n = len(l2_ids)
    if n == 0:
        return np.zeros(1, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32)

    order = np.argsort(l2_ids, kind="stable")
    srt = l2_ids[order]

    e = np.asarray(edges, np.uint64).reshape(-1, 2)
    if len(e):
        pos = np.searchsorted(srt, e.reshape(-1))
        pos = np.clip(pos, 0, n - 1)
        ok = srt[pos] == e.reshape(-1)
        loc = order[pos].reshape(-1, 2)
        ok = ok.reshape(-1, 2).all(axis=1)
        loc = loc[ok]
        loc = loc[loc[:, 0] != loc[:, 1]]                      # drop self loops
        loc = np.sort(loc, axis=1)
        loc = np.unique(loc, axis=0) if len(loc) else loc      # drop duplicates
    else:
        loc = np.zeros((0, 2), np.int64)

    both = np.concatenate([loc, loc[:, ::-1]]) if len(loc) else loc
    deg = np.bincount(both[:, 0], minlength=n).astype(np.int32) if len(both) \
        else np.zeros(n, np.int32)
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(deg, out=indptr[1:])
    if len(both):
        o = np.argsort(both[:, 0], kind="stable")
        indices = both[o, 1].astype(np.int32)
    else:
        indices = np.zeros(0, np.int32)
    return indptr, indices, deg


def connected_components(indptr: np.ndarray, indices: np.ndarray, n: int
                         ) -> np.ndarray:
    """Component label per node, by iterative BFS (no scipy dependency)."""
    comp = np.full(n, -1, np.int32)
    stack = np.empty(n, np.int32)
    c = 0
    for s in range(n):
        if comp[s] >= 0:
            continue
        comp[s] = c
        stack[0] = s
        top = 1
        while top:
            top -= 1
            u = stack[top]
            for v in indices[indptr[u]:indptr[u + 1]]:
                if comp[v] < 0:
                    comp[v] = c
                    stack[top] = v
                    top += 1
        c += 1
    return comp


@dataclass
class AtomTopo:
    """Contracted topology of one atom, in local node indices."""

    n_nodes: int
    deg: np.ndarray                 # [N] int32
    comp: np.ndarray                # [N] int32
    junctions: np.ndarray           # [J] local idx of nodes with deg != 2
    seg_ends: np.ndarray            # [S, 2] local idx of the two segment ends
    seg_nodes: list[np.ndarray]     # [S] interior path nodes, in order
    seg_len_nm: np.ndarray          # [S] float32, NaN without coords
    seg_is_leaf: np.ndarray         # [S] bool, one end is an endpoint
    cycles: int                     # components that are pure loops


def contract(indptr: np.ndarray, indices: np.ndarray, deg: np.ndarray,
             pos: Optional[np.ndarray] = None) -> AtomTopo:
    """Collapse degree-2 runs into segments between junctions.

    A component with no junction at all is a pure cycle (or an isolated node);
    it is counted but produces no segment, because it has no distinguished
    endpoint to hang a segment on.
    """
    n = len(deg)
    comp = connected_components(indptr, indices, n)
    junc = np.flatnonzero(deg != 2).astype(np.int32)

    seg_ends: list[tuple[int, int]] = []
    seg_nodes: list[np.ndarray] = []
    visited_start: set[tuple[int, int]] = set()

    for j in junc.tolist():
        for first in indices[indptr[j]:indptr[j + 1]]:
            first = int(first)
            if (j, first) in visited_start:
                continue
            interior: list[int] = []
            prev, cur = j, first
            while deg[cur] == 2:
                interior.append(cur)
                nb = indices[indptr[cur]:indptr[cur + 1]]
                nxt = int(nb[0]) if int(nb[0]) != prev else int(nb[1])
                prev, cur = cur, nxt
                if cur == j and not interior:
                    break
            # mark the mirror walk so each segment is emitted once
            back = interior[-1] if interior else j
            visited_start.add((j, first))
            visited_start.add((int(cur), int(back)))
            seg_ends.append((j, int(cur)))
            seg_nodes.append(np.asarray(interior, np.int32))

    n_comp = int(comp.max()) + 1 if n else 0
    junc_comps = np.unique(comp[junc]) if len(junc) else np.zeros(0, np.int32)
    cycles = n_comp - len(junc_comps)

    ends = (np.asarray(seg_ends, np.int32).reshape(-1, 2) if seg_ends
            else np.zeros((0, 2), np.int32))
    is_leaf = ((deg[ends[:, 0]] == 1) | (deg[ends[:, 1]] == 1)) if len(ends) \
        else np.zeros(0, bool)

    topo = AtomTopo(n_nodes=n, deg=deg, comp=comp, junctions=junc,
                    seg_ends=ends, seg_nodes=seg_nodes,
                    seg_len_nm=np.full(len(ends), np.nan, np.float32),
                    seg_is_leaf=is_leaf, cycles=int(cycles))
    if pos is not None and len(ends):
        flat, ptr = segment_paths(topo)
        topo.seg_len_nm = segment_lengths(flat, ptr, pos)
    return topo


def endpoint_tangents(topo: AtomTopo, pos: np.ndarray, *, span: int = 5
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector pointing *out* of the atom at each endpoint.

    Taken over up to ``span`` nodes back along the endpoint's own segment, which
    is what a stitch candidate needs: two endpoints that face each other are a
    plausible split, two that face away are not. Returns
    ``(endpoint_local_idx, tangent)``; the tangent is NaN when coordinates are
    missing or the run is degenerate.
    """
    ends_idx: list[int] = []
    tangents: list[np.ndarray] = []
    for i, (a, b) in enumerate(topo.seg_ends.tolist()):
        for tip, other in ((a, b), (b, a)):
            if topo.deg[tip] != 1:
                continue
            interior = topo.seg_nodes[i]
            path = np.concatenate([[tip], interior[::-1] if interior.size and
                                   interior[0] != tip else interior, [other]])
            path = path.astype(np.int64)[:span + 1]
            p = pos[path]
            good = np.isfinite(p).all(axis=1)
            t = np.full(3, np.nan, np.float32)
            if good.sum() >= 2:
                p = p[good]
                v = p[0] - p[-1]
                nrm = float(np.linalg.norm(v))
                if nrm > 0:
                    t = (v / nrm).astype(np.float32)
            ends_idx.append(int(tip))
            tangents.append(t)
    return (np.asarray(ends_idx, np.int32),
            np.asarray(tangents, np.float32).reshape(-1, 3))


# ---------------------------------------------------------------------------
# attribute lookup
# ---------------------------------------------------------------------------

class L2Attributes:
    """Sorted lookup from the pooled attribute cache."""

    def __init__(self, path: str | Path, cols: Optional[list[str]] = None):
        with np.load(Path(path), allow_pickle=False) as z:
            self.l2_id = z["l2_id"]
            want = [k for k in z.files
                    if k != "l2_id" and (cols is None or k in cols)]
            self.cols = {k: z[k] for k in want}
        self._order = np.argsort(self.l2_id, kind="stable")
        self._srt = self.l2_id[self._order]

    def take(self, l2_ids: np.ndarray, col: str) -> np.ndarray:
        """Rows for ``l2_ids``; missing ids come back NaN."""
        v = self.cols[col]
        out = np.full((len(l2_ids),) + v.shape[1:], np.nan, np.float32)
        if not len(l2_ids) or not len(self._srt):
            return out
        pos = np.clip(np.searchsorted(self._srt, l2_ids), 0, len(self._srt) - 1)
        ok = self._srt[pos] == l2_ids
        out[ok] = v[self._order[pos[ok]]]
        return out


# ---------------------------------------------------------------------------
# vectorized geometry over all segments of an atom
# ---------------------------------------------------------------------------

def segment_paths(topo: AtomTopo) -> tuple[np.ndarray, np.ndarray]:
    """Every segment's node run, concatenated, with an offset per segment.

    ``flat[ptr[i]:ptr[i+1]]`` is segment ``i`` as ``[end_a, interior..., end_b]``.
    Building one flat array lets length and tangent be computed for all 12.6M
    segments of a tier with array ops instead of a Python loop per segment.
    """
    n_seg = len(topo.seg_ends)
    if n_seg == 0:
        return np.zeros(0, np.int64), np.zeros(1, np.int64)
    lens = np.fromiter((len(s) + 2 for s in topo.seg_nodes), np.int64, n_seg)
    ptr = np.zeros(n_seg + 1, np.int64)
    np.cumsum(lens, out=ptr[1:])
    flat = np.empty(int(ptr[-1]), np.int64)
    flat[ptr[:-1]] = topo.seg_ends[:, 0]
    flat[ptr[1:] - 1] = topo.seg_ends[:, 1]

    # Scatter every interior run in one shot: a per-segment loop here would be
    # 12.6M Python iterations for a single tier.
    n_int = lens - 2
    total = int(n_int.sum())
    if total:
        interior = np.concatenate([s for s in topo.seg_nodes if s.size])
        starts = ptr[:-1] + 1
        src = np.zeros(len(n_int) + 1, np.int64)
        np.cumsum(n_int, out=src[1:])
        offset = np.repeat(starts - src[:-1], n_int)
        flat[offset + np.arange(total)] = interior
    return flat, ptr


def segment_lengths(flat: np.ndarray, ptr: np.ndarray, pos: np.ndarray
                    ) -> np.ndarray:
    """Cable length of each segment in nm; NaN if a node lacks a coordinate."""
    if len(ptr) <= 1:
        return np.zeros(0, np.float32)
    p = pos[flat]
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    step = np.concatenate([step, [0.0]])
    step[ptr[1:-1] - 1] = 0.0          # the hop between two segments is not cable
    return np.add.reduceat(step, ptr[:-1]).astype(np.float32)


def segment_tip_tangents(topo: AtomTopo, flat: np.ndarray, ptr: np.ndarray,
                         pos: np.ndarray, *, span: int = 5
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit vector at every degree-1 tip, over all segments at once.

    Two endpoints that face each other are a plausible false split; two that
    face away are not. Taken over up to ``span`` nodes back along the tip's own
    segment.
    """
    if len(ptr) <= 1:
        return np.zeros(0, np.int32), np.zeros((0, 3), np.float32)
    a_at, b_at = ptr[:-1], ptr[1:] - 1
    a_in = np.minimum(a_at + span, b_at)
    b_in = np.maximum(b_at - span, a_at)

    tips, inner = [], []
    for at, inn in ((a_at, a_in), (b_at, b_in)):
        node = flat[at]
        keep = topo.deg[node] == 1
        tips.append(node[keep])
        inner.append(flat[inn[keep]])
    tip = np.concatenate(tips)
    inr = np.concatenate(inner)

    v = pos[tip] - pos[inr]
    n = np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = (v / n[:, None]).astype(np.float32)
    t[~np.isfinite(t).all(axis=1)] = np.nan
    return tip.astype(np.int32), t
