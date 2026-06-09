"""Fragment extraction: one Fragment per seg-root skeleton tree.

Each kimimaro skeleton (whole tree, including branch points) is wrapped as a
single Fragment.  No topological chain splitting is performed here — the DNA
encoder and assembler reason over the full tree structure.

``endpoints_nm`` is set to the leaf vertices of the tree (degree ≤ 1), which
serve as seam-stitch handles for Phase 2 global assembly.  The ``dna`` field is
left as ``None``; it is filled by the ``represent/`` stage.

Contamination note (for callers)
---------------------------------
A Fragment whose ``base_root_id`` maps to >1 ``label_version`` roots in the
ground-truth is a false-merge survivor.  Extraction does not filter these out
(it has no access to the label mapping), but training code should mask them:
exclude them from positive pairs so the DNA encoder is never asked to assign
a contaminated seg root a clean identity.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..cell_graph import load_self_skeleton_archive
from ..schemas import Fragment, Region


def skeleton_to_fragment(
    vertices_nm: np.ndarray,
    edges: np.ndarray,
    radii_nm: np.ndarray,
    base_root_id: int,
    region: Region,
    fragment_id: int,
    *,
    min_vertices: int = 3,
) -> "Fragment | None":
    """Wrap a seg-root skeleton tree as a single Fragment.

    Parameters
    ----------
    vertices_nm:
        Global-nm coordinates ``[V, 3]`` float32.
    edges:
        Skeleton connectivity ``[E, 2]`` int64 (local vertex indices).
    radii_nm:
        Inscribed-sphere radius at each vertex ``[V]`` float32.
    base_root_id:
        The segmentation root ID this skeleton came from (``Region.seg_version``).
    region:
        The owning region — used to find which synapses belong to this root.
    fragment_id:
        Globally unique ID for this Fragment (caller's responsibility).
    min_vertices:
        Skeletons with fewer vertices are discarded and ``None`` is returned.

    Returns
    -------
    Fragment or None
        ``None`` if the skeleton is below ``min_vertices``.
    """
    verts = np.asarray(vertices_nm, dtype=np.float32)
    if len(verts) < min_vertices:
        return None

    eds = np.asarray(edges, dtype=np.int64)
    rads = np.asarray(radii_nm, dtype=np.float32)

    # Reshape edges to [E, 2] even if empty.
    if eds.ndim == 1:
        eds = eds.reshape(-1, 2)

    # Leaf vertices (seam-stitch handles): degree ≤ 1.
    degree = np.zeros(len(verts), dtype=np.int64)
    if len(eds):
        np.add.at(degree, eds[:, 0], 1)
        np.add.at(degree, eds[:, 1], 1)
    leaf_mask = degree <= 1
    endpoints = verts[leaf_mask]
    if len(endpoints) == 0:
        # All vertices have degree ≥ 2 (cycle); use first vertex as fallback.
        endpoints = verts[[0]]

    # Synapses in this region that belong to base_root_id.
    pre_match = region.pre_root_id == base_root_id
    post_match = region.post_root_id == base_root_id
    synapse_mask = pre_match | post_match
    synapse_indices = np.where(synapse_mask)[0].astype(np.int64)

    return Fragment(
        fragment_id=int(fragment_id),
        region_id=region.region_id,
        base_root_id=int(base_root_id),
        vertices_nm=verts,
        edges=eds,
        endpoints_nm=endpoints,
        radius_nm=rads,
        synapse_indices=synapse_indices,
        dna=None,
    ).validate()


def extract_fragments_for_region(
    region: Region,
    skeleton_archive_path: str,
    *,
    min_vertices: int = 3,
) -> list[Fragment]:
    """Load a per-box skeleton archive and wrap each seg root as one Fragment.

    Fragment IDs are set to the seg-root ID (``base_root_id``), which is
    globally unique across regions.  Seg roots with fewer than ``min_vertices``
    skeleton vertices are silently skipped.

    Parameters
    ----------
    region:
        The ``Region`` artifact for this tile — provides synapse positions
        and root IDs for synapse assignment.
    skeleton_archive_path:
        Path to the ``.npz`` archive produced by
        ``precompute_self_skeletons_for_cache``.
    min_vertices:
        Minimum number of skeleton vertices for a fragment to be kept.

    Returns
    -------
    list[Fragment]
        One Fragment per valid seg root, ``dna=None``.
    """
    archive = load_self_skeleton_archive(skeleton_archive_path)
    fragments: list[Fragment] = []
    for root_id, (verts, edges, radii) in archive.items():
        frag = skeleton_to_fragment(
            verts, edges, radii,
            base_root_id=root_id,
            region=region,
            fragment_id=root_id,
            min_vertices=min_vertices,
        )
        if frag is not None:
            fragments.append(frag)
    return fragments
