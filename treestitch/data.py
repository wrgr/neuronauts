"""Data loaders for the global tree stitching problem.

MICrONS Minnie65 reference dataset
------------------------------------
The MICrONS Minnie65 mouse V1 connectome is the reference dataset for
optimising and evaluating the pipeline.

``load_minnie65_world()`` fetches real proofread neurons, splits each
skeleton into N pieces (simulating the pre-proofread fragmented state),
places observations (synapses) near skeleton vertices, and returns a
(fragments, region, root_label_map) triple ready for the pipeline.

Low-level loaders
-----------------
``sample_neurons(n, cell_type, seed)``  — sample proofread v1412 root IDs.
``load_fragment(root_id, token)``       — fetch one skeleton as a Fragment.

Authentication
--------------
The CAVE skeleton cache requires a bearer token.  The public MICrONS token
(DEFAULT_TOKEN) works for the public minnie65_public dataset.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

import numpy as np

from neuronauts.data.loaders import (
    DEFAULT_TOKEN,
    load_nucleus_table,
    load_skeleton,
    sample_neurons,
)

__all__ = [
    "DEFAULT_TOKEN",
    "load_nucleus_table",
    "load_skeleton",
    "sample_neurons",
    "load_fragment",
    "load_minnie65_world",
]


def load_fragment(root_id: int, token: str = DEFAULT_TOKEN):
    """Fetch one skeleton from the CAVE skeleton cache and wrap it as a Fragment.

    Parameters
    ----------
    root_id:
        Proofread v1412 root ID.
    token:
        CAVE bearer auth token.

    Returns
    -------
    Fragment with ``dna=None`` (call ``encode_fragments`` to fill), or ``None``
    on fetch failure.
    """
    from neuronauts.schemas import Fragment

    skel = load_skeleton(root_id, token)
    if skel is None:
        return None

    verts = skel["vertices_nm"]
    edges = skel["edges"]
    radii = skel["radii_nm"]

    deg = np.zeros(len(verts), dtype=np.int64)
    if len(edges):
        np.add.at(deg, edges[:, 0], 1)
        np.add.at(deg, edges[:, 1], 1)
    leaf_mask = deg <= 1
    endpoints = verts[leaf_mask] if leaf_mask.any() else verts[[0, -1]]

    return Fragment(
        fragment_id=root_id,
        region_id="minnie65",
        base_root_id=root_id,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=endpoints,
        radius_nm=radii,
        synapse_indices=np.array([], dtype=np.int64),
        dna=None,
    ).validate()


def _split_skeleton_n_pieces(
    verts: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray,
    n_pieces: int,
    min_verts: int = 8,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Recursively bisect a skeleton tree into up to n_pieces sub-trees."""
    if n_pieces <= 1 or len(verts) < min_verts * 2 or len(edges) == 0 or len(verts) < 4:
        return [(verts, edges, radii)]

    n = len(verts)
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))

    # Subtree sizes (BFS from root=0)
    size = np.ones(n, dtype=np.int32)
    parent = np.full(n, -1, dtype=np.int32)
    order: list[int] = []
    visited = np.zeros(n, dtype=bool)
    q = deque([0])
    visited[0] = True
    while q:
        v = q.popleft()
        order.append(v)
        for w in adj[v]:
            if not visited[w]:
                visited[w] = True
                parent[w] = v
                q.append(w)
    for v in reversed(order):
        p = parent[v]
        if p >= 0:
            size[p] += size[v]

    best_v, best_diff = -1, n + 1
    for v in order[1:]:
        diff = abs(size[v] - (n - size[v]))
        if diff < best_diff:
            best_diff, best_v = diff, int(v)
    if best_v < 0:
        return [(verts, edges, radii)]

    # Flood-fill two sides
    visited_a = np.zeros(n, dtype=bool)
    cut_p = int(parent[best_v])
    q2 = deque([best_v])
    visited_a[best_v] = True
    while q2:
        v = q2.popleft()
        for w in adj[v]:
            if not visited_a[w] and w != cut_p:
                visited_a[w] = True
                q2.append(w)
    visited_b = ~visited_a

    def _sub(mask):
        old = np.where(mask)[0]
        remap = np.full(n, -1, dtype=np.int64)
        remap[old] = np.arange(len(old), dtype=np.int64)
        sub_v = verts[old]
        sub_r = radii[old]
        keep = mask[edges[:, 0]] & mask[edges[:, 1]]
        sub_e = remap[edges[keep]].astype(np.int64)
        return sub_v, sub_e, sub_r

    vA, eA, rA = _sub(visited_a)
    vB, eB, rB = _sub(visited_b)
    if len(vA) < min_verts or len(vB) < min_verts:
        return [(verts, edges, radii)]

    half = n_pieces // 2
    return (_split_skeleton_n_pieces(vA, eA, rA, half, min_verts) +
            _split_skeleton_n_pieces(vB, eB, rB, n_pieces - half, min_verts))


def load_minnie65_world(
    n_objects: int = 20,
    n_pieces: int = 3,
    observations_per_piece: int = 12,
    *,
    cell_type: Optional[str] = None,
    max_verts: int = 8000,
    synapse_noise_nm: float = 500.0,
    min_piece_verts: int = 8,
    token: str = DEFAULT_TOKEN,
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Load the MICrONS Minnie65 world for tree-stitching evaluation.

    Fetches ``n_objects`` real proofread neurons, splits each skeleton into
    ``n_pieces`` sub-fragments (simulating the fragmented pre-proofread state),
    and places ``observations_per_piece`` observations near each piece's
    skeleton vertices.

    Parameters
    ----------
    n_objects:
        Number of parent trees (neurons) to fetch.
    n_pieces:
        Number of fragments per tree (simulated fragmentation level).
        2 = bisection (easiest), 3–4 = typical, 5+ = hard.
    observations_per_piece:
        Observations placed near each fragment's skeleton vertices.
    cell_type:
        Optional cell-type filter (e.g. ``"23P"`` for L2/3 pyramidal).
        ``None`` = all cell types (cross-type, easier test).
    max_verts:
        Maximum skeleton vertices per neuron (skip larger neurons).
    synapse_noise_nm:
        Gaussian jitter on observation positions (nm).  Smaller = more
        informative spatial signal; larger = forces reliance on fragment DNA.
    token:
        CAVE bearer auth token.
    seed:
        RNG seed.
    verbose:
        Print progress.

    Returns
    -------
    (fragments, region, root_label_map)
        fragments:       list of Fragment (dna=None — call encode_fragments)
        region:          Region with fragment_ids + object labels
        root_label_map:  dict mapping fragment.base_root_id → {object_id}
                         (for train_fragment_encoder supervision)
    """
    from neuronauts.schemas import Fragment, Region

    rng = np.random.default_rng(seed)

    candidates = sample_neurons(n_objects * 6, cell_type=cell_type, seed=seed)
    if verbose:
        print(f"Sampling {n_objects} objects"
              + (f" (cell_type={cell_type})" if cell_type else " (mixed)") + " …")
        print(f"  {len(candidates)} candidates")
        print(f"\nFetching skeletons (target {n_objects} objects, {n_pieces} pieces) …")

    all_obs_pts: list[np.ndarray] = []
    all_frag_ids: list[int] = []
    all_obj_ids: list[int] = []
    fragments: list[Fragment] = []
    root_label_map: dict[int, set[int]] = {}
    obj_counter = 0
    frag_id_counter = 1
    obs_idx = 0

    for root_id in candidates:
        if obj_counter >= n_objects:
            break

        skel = load_skeleton(root_id, token)
        if skel is None:
            continue
        verts = skel["vertices_nm"]
        edges_raw = skel["edges"]
        radii = skel["radii_nm"]

        if len(verts) < min_piece_verts * n_pieces or len(verts) > max_verts:
            continue

        pieces = _split_skeleton_n_pieces(verts, edges_raw, radii, n_pieces,
                                          min_verts=min_piece_verts)
        if len(pieces) < 2:
            continue

        obj_counter += 1
        obj_id = obj_counter

        for pv, pe, pr in pieces:
            frag_id = frag_id_counter
            frag_id_counter += 1

            deg = np.zeros(len(pv), dtype=np.int64)
            if len(pe):
                np.add.at(deg, pe[:, 0], 1)
                np.add.at(deg, pe[:, 1], 1)
            leaf_mask = deg <= 1
            endpoints = pv[leaf_mask] if leaf_mask.any() else pv[[0]]

            anchor_idxs = rng.integers(0, len(pv), observations_per_piece)
            obs_pts = (pv[anchor_idxs] +
                       rng.normal(0, synapse_noise_nm,
                                  (observations_per_piece, 3)).astype(np.float32))

            obs_indices = np.arange(obs_idx, obs_idx + observations_per_piece,
                                    dtype=np.int64)
            obs_idx += observations_per_piece

            all_obs_pts.append(obs_pts)
            all_frag_ids.extend([frag_id] * observations_per_piece)
            all_obj_ids.extend([obj_id] * observations_per_piece)

            frag = Fragment(
                fragment_id=frag_id,
                region_id="minnie65_split",
                base_root_id=frag_id,
                vertices_nm=pv,
                edges=pe if len(pe) else np.zeros((0, 2), dtype=np.int64),
                endpoints_nm=endpoints,
                radius_nm=pr,
                synapse_indices=obs_indices,
                dna=None,
            ).validate()
            fragments.append(frag)

            root_label_map.setdefault(frag_id, set()).add(obj_id)

        piece_sizes = [len(p[0]) for p in pieces]
        if verbose:
            print(f"  [{obj_counter:3d}] root={root_id}  V={len(verts)}"
                  f"  pieces={'/'.join(str(s) for s in piece_sizes)}")
        time.sleep(0.05)

    if not fragments:
        raise RuntimeError("No usable skeletons fetched — check network access and token")

    all_pts = np.concatenate(all_obs_pts).astype(np.float32)
    post_pts = all_pts + rng.normal(0, 2000, all_pts.shape).astype(np.float32)
    mins, maxs = all_pts.min(0), all_pts.max(0)
    pad = 5000.0
    bbox = (tuple(float(v) for v in mins - pad), tuple(float(v) for v in maxs + pad))

    region = Region(
        region_id="minnie65_split",
        bbox_nm=bbox,
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=1412,
        pre_pt_nm=all_pts,
        post_pt_nm=post_pts,
        pre_root_id=np.array(all_obj_ids, dtype=np.int64),
        post_root_id=np.zeros(obs_idx, dtype=np.int64),
        synapse_id=np.arange(obs_idx, dtype=np.int64),
        pre_seg_id=np.array(all_frag_ids, dtype=np.int64),
        post_seg_id=np.zeros(obs_idx, dtype=np.int64),
    ).validate()

    n_obj = len(set(all_obj_ids))
    if verbose:
        print(f"\n  → {n_obj} objects, {len(fragments)} fragments, {obs_idx} observations")

    return fragments, region, root_label_map
