"""Common helpers for the neuronauts package.

This module consolidates frequently used patterns — union-find, safe
normalization, and pairwise-edge construction — so that every call-site
shares one well-tested implementation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np


# ── Union-Find ───────────────────────────────────────────────────────

class UnionFind:
    """Disjoint-set / union-find with path compression and union-by-rank.

    Supports both integer-indexed (list-backed) and arbitrary-key
    (dict-backed) modes depending on how it is constructed.

    Examples
    --------
    >>> uf = UnionFind(5)           # items 0..4
    >>> uf.union(0, 1)
    True
    >>> uf.union(2, 3)
    True
    >>> uf.find(0) == uf.find(1)
    True
    >>> sorted(map(sorted, uf.groups()))
    [[0, 1], [2, 3], [4]]

    >>> uf = UnionFind.from_keys([10, 20, 30])
    >>> uf.union(10, 30)
    True
    >>> uf.find(10) == uf.find(30)
    True
    """

    # ── constructors ─────────────────────────────────────────────────

    def __init__(self, n: int) -> None:
        """Create a union-find for items ``0 .. n-1``."""
        self._parent: list[int] | dict[int, int] = list(range(n))
        self._rank: list[int] | dict[int, int] = [0] * n

    @classmethod
    def from_keys(cls, keys: Iterable[int]) -> "UnionFind":
        """Create a union-find over an arbitrary set of integer keys."""
        uf = cls.__new__(cls)
        uf._parent = {k: k for k in keys}
        uf._rank = {k: 0 for k in uf._parent}
        return uf

    # ── core operations ──────────────────────────────────────────────

    def find(self, x: int) -> int:
        """Return the representative of the set containing *x* (with path compression)."""
        p = self._parent
        while p[x] != x:
            p[x] = p[p[x]]  # path halving
            x = p[x]
        return x

    def union(self, x: int, y: int) -> bool:
        """Merge the sets of *x* and *y*.  Returns ``True`` if they were different."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        r = self._rank
        if r[px] < r[py]:
            px, py = py, px
        self._parent[py] = px
        if r[px] == r[py]:
            r[px] = r[px] + 1
        return True

    # ── query helpers ────────────────────────────────────────────────

    def connected(self, x: int, y: int) -> bool:
        """Return ``True`` if *x* and *y* are in the same set."""
        return self.find(x) == self.find(y)

    def groups(self) -> List[List[int]]:
        """Return all disjoint groups as lists of members."""
        g: Dict[int, List[int]] = defaultdict(list)
        keys = range(len(self._parent)) if isinstance(self._parent, list) else self._parent
        for k in keys:
            g[self.find(k)].append(k)
        return list(g.values())

    def group_dict(self) -> Dict[int, List[int]]:
        """Return ``{representative: [members]}`` mapping."""
        g: Dict[int, List[int]] = defaultdict(list)
        keys = range(len(self._parent)) if isinstance(self._parent, list) else self._parent
        for k in keys:
            g[self.find(k)].append(k)
        return dict(g)

    def __len__(self) -> int:
        return len(self._parent)

    def __contains__(self, x: int) -> bool:
        if isinstance(self._parent, list):
            return 0 <= x < len(self._parent)
        return x in self._parent


# ── Safe normalization ───────────────────────────────────────────────

def safe_normalize(
    v: np.ndarray,
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize *v* along *axis*, returning zero-vectors where magnitude < *eps*.

    This is a drop-in replacement for the common pattern::

        v / (np.linalg.norm(v, axis=..., keepdims=True) + 1e-8)

    but avoids the tiny-but-nonzero residual when the input is actually zero.
    """
    mag = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.where(mag < eps, 0.0, v / np.where(mag < eps, 1.0, mag))


# ── Pairwise edge construction ───────────────────────────────────────

def pairwise_edges(indices: Iterable[int]) -> Set[Tuple[int, int]]:
    """Return canonical ``(min, max)`` pairs for all combinations of *indices*.

    Useful for building line-graph edge sets from synapse groups.

    >>> sorted(pairwise_edges([3, 1, 2]))
    [(1, 2), (1, 3), (2, 3)]
    """
    items = sorted(set(indices))
    edges: Set[Tuple[int, int]] = set()
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            edges.add((items[i], items[j]))
    return edges


__all__ = [
    "UnionFind",
    "safe_normalize",
    "pairwise_edges",
]
