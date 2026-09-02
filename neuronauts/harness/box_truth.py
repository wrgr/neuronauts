"""What can actually be assembled inside a box, as opposed to what is true.

A proofread cell is one object in the volume. Inside a 100 um cube it usually is
not: its axon leaves through a face, runs 90-220 um outside, and comes back, so
within the box the cell arrives as several pieces with **no connecting path
between them at all**. Measured on the gold cell 864691136011850926: the full
skeleton is one connected component, the same skeleton clipped to the cube is
eight, and six of the seven floating pieces are axon.

Scoring a box-local proposer against every same-cell pair therefore puts links
in the denominator that no method operating inside the box could ever find --
the third denominator trap in this project, after the same-owner clique
(``results/EXP-060/CORRECTION.md``) and the chained-reachability collapse
(EXP-072). On the 40 held-out cells of EXP-071, 72 of 491 nearest-sibling paths
(14.7%) leave the cube.

This module defines the honest target: group a cell's fragments by whether a
path between them stays inside the box, and take a component of that grouping
as the thing to recall. The largest component is the usual choice, and
:func:`box_components` returns them all so a caller can say what it dropped.

The construction is deliberately generous to the method under test: connectivity
is decided on the *proofread* graph, which is ground truth and not available at
inference. It answers "is this link reachable in principle inside this box",
not "did the proposer have the information to find it".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class BoxTruth:
    """A cell's fragments, grouped by box-local connectivity."""

    components: list[list[int]]     # fragment ids, largest first
    dropped: list[int]              # fragments outside the largest component
    n_fragments: int

    @property
    def largest(self) -> list[int]:
        return self.components[0] if self.components else []

    @property
    def frac_in_largest(self) -> float:
        return len(self.largest) / max(self.n_fragments, 1)


def box_components(edges: np.ndarray, node_pos: np.ndarray,
                   node_fragment: np.ndarray, lo, hi) -> BoxTruth:
    """Group fragments by whether a path between them stays inside the box.

    ``edges`` are index pairs into ``node_pos`` / ``node_fragment`` (the cell's
    level-2 graph). ``node_fragment`` carries the labelled fragment a node
    belongs to, or 0. A node is in the box when its coordinate is; an edge is
    when both its ends are.

    Two fragments end up in the same group when some in-box path joins them.
    Note a fragment can itself straddle the box edge and so touch several node
    components -- those components are then merged, because the fragment
    connects them. Doing this the naive way (one group per node component) would
    split a fragment from itself.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = len(node_pos)
    inside = np.all((node_pos >= lo) & (node_pos <= hi), axis=1)
    inside &= np.isfinite(node_pos).all(axis=1)

    e = np.asarray(edges, np.int64).reshape(-1, 2)
    keep = inside[e[:, 0]] & inside[e[:, 1]]
    e = e[keep]

    if n == 0:
        return BoxTruth([], [], 0)
    g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    _, node_comp = connected_components(g, directed=False)

    frags = np.unique(node_fragment[(node_fragment > 0) & inside])
    if not len(frags):
        return BoxTruth([], [], 0)
    fidx = {int(f): i for i, f in enumerate(frags.tolist())}

    # union-find over fragments, joined through shared node components
    parent = list(range(len(frags)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen: dict[int, int] = {}
    for k in np.flatnonzero(inside & (node_fragment > 0)).tolist():
        f = fidx[int(node_fragment[k])]
        c = int(node_comp[k])
        if c in seen:
            union(seen[c], f)
        else:
            seen[c] = f

    groups: dict[int, list[int]] = {}
    for f_id, i in fidx.items():
        groups.setdefault(find(i), []).append(f_id)
    comps = sorted((sorted(v) for v in groups.values()), key=len, reverse=True)
    dropped = [f for c in comps[1:] for f in c]
    return BoxTruth(comps, sorted(dropped), len(frags))


def restrict_links(links, keep_fragments) -> set:
    """Spanning links with both ends in ``keep_fragments``."""
    k = set(int(x) for x in keep_fragments)
    return {tuple(sorted((int(a), int(b)))) for a, b in links
            if int(a) in k and int(b) in k}


def spanning_target(bt: BoxTruth, *, mode: str = "all_components") -> list[list[int]]:
    """The fragment groups a box-local proposer should be scored against.

    ``all_components``  every group of two or more fragments that a path inside
                        the box connects. Recommended: on the 40 held-out cells
                        of EXP-071 it drops the 123 links (27%) that no in-box
                        method could reach, while keeping all 499 fragments in
                        play -- there are **no singleton components**, so every
                        fragment has some in-box partner and nothing is lost by
                        scoring it.
    ``largest``         only the biggest group, one target per cell. Stricter,
                        and it additionally discards 218 fragments (43.7%) that
                        *are* assemblable inside the box, just into a separate
                        piece. Use when the question is "did we rebuild the
                        cell", not "did we find the joins that exist here".
    """
    if mode == "largest":
        return [bt.largest] if len(bt.largest) > 1 else []
    if mode == "all_components":
        return [c for c in bt.components if len(c) > 1]
    raise ValueError(f"unknown mode {mode!r}")
