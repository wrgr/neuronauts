"""Soma detection from skeleton caliber (radius) — the verified ``n_soma`` routine.

Extracted from ``experiments/pcfg/skeleton_topology_merge.py`` (``topo_features``),
where large-radius vertices are clustered spatially and each cluster counts as one
soma.  This is the multi-soma merge signal: a single clean neuron has one soma;
two comparably-sized soma clusters means two cells were merged.

Lifted into the core package so both the pcfg topology model and the new
compartment grammar share one verified implementation (the pcfg module also has a
broken ``experiments.pcfg_synapse_partitions`` import path, so importing it
directly is not an option).

Thresholds match the original: radius > 3000 nm marks a "big" (soma-caliber)
vertex, and big vertices within 8000 nm of each other are linked into one cluster.
"""
from __future__ import annotations

import numpy as np


def soma_clusters(
    vertices_nm: np.ndarray,
    radius: np.ndarray | None,
    *,
    radius_thresh_nm: float = 3000.0,
    link_nm: float = 8000.0,
) -> list[np.ndarray]:
    """Cluster large-radius vertices into soma candidates.

    Parameters
    ----------
    vertices_nm : [V, 3] float
        Skeleton vertex coordinates in nm.
    radius : [V] float or None
        Per-vertex caliber in nm.  ``None`` (or all-NaN) -> no somas.
    radius_thresh_nm, link_nm : float
        A vertex with ``radius > radius_thresh_nm`` is "big"; big vertices within
        ``link_nm`` are unioned into one cluster.

    Returns
    -------
    list of int arrays — each array holds the vertex indices of one soma cluster,
    ordered by descending cluster size (largest soma first).
    """
    if radius is None:
        return []
    radius = np.asarray(radius, dtype=np.float64)
    vertices_nm = np.asarray(vertices_nm, dtype=np.float64)
    big = np.where((radius > radius_thresh_nm) & ~np.isnan(radius))[0]
    if len(big) == 0:
        return []

    from scipy.spatial import cKDTree

    tree = cKDTree(vertices_nm[big])
    pairs = tree.query_pairs(link_nm)

    parent = {int(b): int(b) for b in big.tolist()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(int(big[a]))] = find(int(big[b]))

    groups: dict[int, list[int]] = {}
    for b in big.tolist():
        groups.setdefault(find(int(b)), []).append(int(b))
    clusters = [np.array(sorted(g), dtype=np.int64) for g in groups.values()]
    clusters.sort(key=len, reverse=True)
    return clusters


def n_soma(vertices_nm: np.ndarray, radius: np.ndarray | None, **kwargs) -> int:
    """Number of distinct soma-caliber clusters (the multi-soma merge feature)."""
    return len(soma_clusters(vertices_nm, radius, **kwargs))
