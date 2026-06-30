"""Build fully real f(v117 → v1718) partition worlds from CAVE lineage.

Everything here is real — no synthetic fragments or synthetic splits:

  - observations  = real synapses (``synapses_pni_2`` at an available version)
  - fragment id   = the real **v117** root the synapse's supervoxel belongs to
                    (chunkedgraph lineage at the v117 timestamp)
  - label         = the real **v1718** (proofread) root the synapse sits on
  - fragment shape= real L2-cache skeleton when available, synapse cloud fallback

The partition task is then exactly f(v117 → v1718): group the real v117
fragments back into their proofread neurons.  Real "trunk + slivers" splits and
real frankenmerges (a v117 root whose synapses belong to two v1718 neurons)
arise from the data, not from a generator.

Two world builders are provided:

``build_lineage_world`` — neuron-seeded sampling
    Seeds from isolated neurons.  Good for studying per-neuron structure but
    produces graphs with near-zero cross-neuron edges, starving edge_cc of the
    training signal it needs to learn merge errors.

    from treestitch.realworld import build_lineage_world
    fragments, region, label_map = build_lineage_world(n_objects=12)

``build_region_world`` — spatial region sampling
    Fetches ALL synapses in a bounding box.  Neurons are spatially interleaved,
    so the k-NN synapse graph naturally contains cross-neuron edges.  Real
    frankenmerge v117 roots appear in every spatial slice and contribute
    same-fragment cut-signals.  This is the correct sampler for edge_cc training.

    from treestitch.realworld import build_region_world
    fragments, region, label_map = build_region_world(
        bbox_nm=((1.15e6, 9.3e5, 7.8e5), (1.20e6, 9.5e5, 8.2e5)))

Both return the same ``(fragments, region, root_label_map)`` tuple and are
compatible with ``build_observation_graph`` unchanged.

Fragment morphology priority:
  1. L2 cache skeleton (``lineage.l2_skeleton``) — real L2-node rep_coord_nm
     centroids → MST skeleton; real endpoints enabling endpoint-adjacency edges.
  2. Synapse point cloud fallback — used when the L2 cache request fails.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _skeleton_endpoints(vertices_nm: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return leaf vertices (degree ≤ 1) of a skeleton tree as [K, 3] float32."""
    if len(vertices_nm) == 0:
        return vertices_nm.copy()
    if len(edges) == 0:
        return vertices_nm.copy()
    degree = np.zeros(len(vertices_nm), dtype=np.int32)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    leaf_mask = degree <= 1
    leaves = vertices_nm[leaf_mask]
    if len(leaves) == 0:
        return vertices_nm[[0]]
    return leaves


def _l2_fragment(frag_id: int, region_id: str, skel: dict,
                 syn_indices: np.ndarray):
    """Make a Fragment from a real L2-cache skeleton.

    ``skel`` is the dict returned by ``lineage.l2_skeleton``:
    ``{"vertices_nm", "edges", "radii_nm", "l2_ids"}``.
    """
    from neuronauts.schemas import Fragment
    verts = skel["vertices_nm"]  # [V, 3]
    edges = skel["edges"]        # [E, 2]
    radii = skel["radii_nm"]     # [V]
    endpoints = _skeleton_endpoints(verts, edges)
    return Fragment(
        fragment_id=frag_id,
        region_id=region_id,
        base_root_id=frag_id,
        vertices_nm=verts,
        edges=edges,
        endpoints_nm=endpoints,
        radius_nm=radii,
        synapse_indices=np.asarray(syn_indices, dtype=np.int64),
        dna=None,
    ).validate()


def _cloud_fragment(frag_id: int, region_id: str, points: np.ndarray,
                    syn_indices: np.ndarray):
    """Make a Fragment whose skeleton is the point cloud of its synapses."""
    from neuronauts._scipy_compat import cKDTree
    from neuronauts.schemas import Fragment

    pts = np.asarray(points, dtype=np.float32)
    m = len(pts)
    if m >= 2:
        k = min(3, m - 1)
        tree = cKDTree(pts)
        _, nbr = tree.query(pts, k=k + 1)
        es: set[tuple[int, int]] = set()
        for i in range(m):
            for slot in range(1, k + 1):
                j = int(nbr[i, slot])
                if i != j:
                    es.add((min(i, j), max(i, j)))
        edges = (np.array(sorted(es), dtype=np.int64)
                 if es else np.zeros((0, 2), dtype=np.int64))
        # endpoints: extremes along the principal spread axis
        centered = pts - pts.mean(0)
        axis = centered[np.argmax(np.linalg.norm(centered, axis=1))]
        proj = centered @ (axis / (np.linalg.norm(axis) + 1e-8))
        endpoints = pts[[int(np.argmin(proj)), int(np.argmax(proj))]]
        # radius_nm: distance from fragment centroid — encodes local scale
        # so the GNN mean-pool varies per fragment (breaks constant-feature collapse)
        radii = np.linalg.norm(centered, axis=1).astype(np.float32)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        endpoints = pts.copy()
        radii = np.zeros(m, dtype=np.float32)

    return Fragment(
        fragment_id=frag_id,
        region_id=region_id,
        base_root_id=frag_id,
        vertices_nm=pts,
        edges=edges,
        endpoints_nm=endpoints,
        radius_nm=radii,
        synapse_indices=np.asarray(syn_indices, dtype=np.int64),
        dna=None,
    ).validate()


def build_lineage_world(
    n_objects: int = 12,
    *,
    version: int = 1718,
    side: str = "post",
    max_syn_per_obj: int = 200,
    min_syn_per_obj: int = 20,
    v117_timestamp: Optional[int] = None,
    token: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
    l2_skeletons: bool = True,
) -> tuple:
    """Assemble a real f(v117→v{version}) partition world.

    Parameters
    ----------
    l2_skeletons:
        If ``True`` (default), attempt to fetch real L2-cache skeletons for each
        v117 fragment root (``lineage.l2_skeleton``).  These give real MST
        skeletons with real leaf endpoints, enabling endpoint-adjacency edges in
        the observation graph.  Falls back to the synapse point-cloud when the
        L2 request fails.  Set ``False`` to always use the synapse cloud (faster,
        no extra network calls).

    Returns
    -------
    (fragments, region, root_label_map)
        fragments       : list of Fragment (one per distinct v117 root; dna=None)
        region          : Region (pre_* slots hold the real synapses;
                          pre_seg_id = v117 fragment root, pre_root_id = label)
        root_label_map  : {v117_root: {v{version}_root, ...}}  — multi-label
                          entries are real frankenmerges.
    """
    from neuronauts.data import lineage as L
    from neuronauts.data.loaders import DEFAULT_TOKEN, sample_neurons
    from neuronauts.schemas import Region

    tok = token or DEFAULT_TOKEN
    v117_ts = v117_timestamp if v117_timestamp is not None else L.V117_TIMESTAMP
    rng = np.random.default_rng(seed)

    candidates = sample_neurons(n_objects * 6, seed=seed)
    if verbose:
        print(f"Building real v117→v{version} world: target {n_objects} neurons, "
              f"side={side} …")

    obs_pos: list[np.ndarray] = []
    obs_frag: list[int] = []      # v117 root (fragment id)
    obs_label: list[int] = []     # v{version} root (label)
    seen_labels: set[int] = set()
    n_obj = 0

    for nuc_root in candidates:
        if n_obj >= n_objects:
            break
        target = L.root_at_version(nuc_root, version, token=tok)
        if target is None or target in seen_labels:
            continue
        syn = L.fetch_synapses(target, version=version, side=side,
                               limit=max_syn_per_obj * 3, token=tok)
        if syn is None or len(syn["positions_nm"]) < min_syn_per_obj:
            continue

        pos = syn["positions_nm"]
        sv = syn["supervoxel_ids"]
        if len(pos) > max_syn_per_obj:
            sel = rng.choice(len(pos), max_syn_per_obj, replace=False)
            pos, sv = pos[sel], sv[sel]

        v117 = L.roots_at(sv, v117_ts, token=tok)
        if v117 is None:
            continue
        keep = v117 > 0
        if keep.sum() < min_syn_per_obj:
            continue
        pos, v117 = pos[keep], v117[keep]

        seen_labels.add(target)
        n_obj += 1
        obs_pos.append(pos.astype(np.float32))
        obs_frag.extend(int(x) for x in v117)
        obs_label.extend([int(target)] * len(v117))

        if verbose:
            nfrag = len(np.unique(v117))
            print(f"  [{n_obj:3d}] v{version} {target}: {len(v117)} synapses, "
                  f"{nfrag} v117 fragments")

    if n_obj < 2:
        raise RuntimeError("Too few neurons assembled — check network/token/version")

    all_pos = np.concatenate(obs_pos, axis=0).astype(np.float32)
    frag_ids = np.asarray(obs_frag, dtype=np.int64)
    labels = np.asarray(obs_label, dtype=np.int64)
    n_obs = len(all_pos)

    # Build one fragment per distinct v117 root: L2 skeleton if available, else synapse cloud.
    fragments = []
    root_label_map: dict[int, set[int]] = {}
    n_l2_ok = 0
    n_l2_fail = 0
    for fr in np.unique(frag_ids):
        mask = frag_ids == fr
        idxs = np.where(mask)[0]
        frag = None
        if l2_skeletons:
            from neuronauts.data.lineage import l2_skeleton
            skel = l2_skeleton(int(fr), token=tok)
            if skel is not None:
                frag = _l2_fragment(int(fr), f"minnie65_v{version}", skel, idxs)
                n_l2_ok += 1
            else:
                n_l2_fail += 1
        if frag is None:
            frag = _cloud_fragment(int(fr), f"minnie65_v{version}",
                                   all_pos[idxs], idxs)
        fragments.append(frag)
        root_label_map.setdefault(int(fr), set()).update(
            int(x) for x in np.unique(labels[mask]))

    n_franken = sum(1 for v in root_label_map.values() if len(v) > 1)

    post_pts = all_pos + rng.normal(0, 2000, all_pos.shape).astype(np.float32)
    mins, maxs = all_pos.min(0), all_pos.max(0)
    pad = 5000.0
    bbox = (tuple(float(v) for v in mins - pad), tuple(float(v) for v in maxs + pad))

    region = Region(
        region_id=f"minnie65_v{version}",
        bbox_nm=bbox,
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=version,
        pre_pt_nm=all_pos,
        post_pt_nm=post_pts,
        pre_root_id=labels,
        post_root_id=np.zeros(n_obs, dtype=np.int64),
        synapse_id=np.arange(n_obs, dtype=np.int64),
        pre_seg_id=frag_ids,
        post_seg_id=np.zeros(n_obs, dtype=np.int64),
    ).validate()

    if verbose:
        skel_str = (f", L2 skeletons: {n_l2_ok}/{n_l2_ok + n_l2_fail} ok"
                    if l2_skeletons else "")
        print(f"\n  → {n_obj} neurons, {len(fragments)} v117 fragments, "
              f"{n_obs} synapses, {n_franken} real frankenmerges"
              f"{skel_str}")

    return fragments, region, root_label_map


def build_region_world(
    bbox_nm: tuple,
    *,
    version: int = 1718,
    side: str = "pre",
    max_synapses: int = 20_000,
    min_syn_per_fragment: int = 5,
    v117_timestamp: Optional[int] = None,
    token: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
    l2_skeletons: bool = True,
    tile_x_nm: float = 0,
    per_tile_limit: int = 200_000,
) -> tuple:
    """Assemble a real f(v117→v{version}) partition world from a spatial bbox.

    Unlike ``build_lineage_world`` (which seeds from isolated neurons), this
    fetches every synapse in a spatial region. Because neurons are spatially
    interleaved, the resulting k-NN synapse graph contains cross-neuron edges
    naturally — the training signal the edge classifier needs.

    Real frankenmerge v117 roots appear in every spatial slice.
    ``root_label_map`` entries with ``len > 1`` are frankenmerges; each one
    contributes same-fragment edges with ``target=0`` (cut-signals) that teach
    the model to detect and split merge errors.

    Parameters
    ----------
    bbox_nm:
        ``((x0, y0, z0), (x1, y1, z1))`` in nm.
    max_synapses:
        Subsample to this many synapses when the region has more.
    min_syn_per_fragment:
        Discard v117 roots with fewer synapses (sliver noise).
    l2_skeletons:
        Fetch real L2-cache skeletons (True) or use synapse point clouds.

    Returns
    -------
    (fragments, region, root_label_map)
        Identical contract to ``build_lineage_world`` — drop-in replacement.
    """
    from neuronauts.data import lineage as L
    from neuronauts.data.loaders import DEFAULT_TOKEN
    from neuronauts.schemas import Region

    tok = token or DEFAULT_TOKEN
    v117_ts = v117_timestamp if v117_timestamp is not None else L.V117_TIMESTAMP
    rng = np.random.default_rng(seed)

    (x0, y0, z0), (x1, y1, z1) = bbox_nm
    if verbose:
        print(f"Building region world v117→v{version}: "
              f"[{x0:.0f},{y0:.0f},{z0:.0f}]–[{x1:.0f},{y1:.0f},{z1:.0f}] nm "
              f"side={side} …")

    syn = None
    if tile_x_nm > 0:
        if verbose:
            n_tiles = max(1, int(np.ceil(
                (bbox_nm[1][0] - bbox_nm[0][0]) / tile_x_nm)))
            print(f"  tiled fetch: {n_tiles} x-tiles × {tile_x_nm/1000:.0f} µm, "
                  f"limit={per_tile_limit}/tile …")
        syn = L.fetch_region_synapses_tiled(
            bbox_nm, version=version, side=side,
            tile_x_nm=tile_x_nm, per_tile_limit=per_tile_limit, token=tok)
        if syn is None or len(syn["positions_nm"]) == 0:
            raise RuntimeError(
                "No synapses from tiled fetch — check network/token/version/bbox")
    else:
        effective_limit = max_synapses
        while effective_limit >= 1000:
            syn = L.fetch_region_synapses(bbox_nm, version=version, side=side,
                                           limit=effective_limit, token=tok)
            if syn is not None:
                break
            if verbose:
                print(f"  limit={effective_limit} failed, retrying at "
                      f"{effective_limit // 2} …")
            effective_limit //= 2
        if syn is None or len(syn["positions_nm"]) == 0:
            raise RuntimeError(
                "No synapses returned for bbox — check network/token/version/bbox")

    pos = syn["positions_nm"]
    sv_ids = syn["supervoxel_ids"]
    v_labels = syn["root_ids"].astype(np.int64)
    other_labels = syn.get("other_root_ids", np.zeros(len(pos), dtype=np.uint64)).astype(np.int64)
    # Real CAVE synapse id (shared across pre/post fetches) — join key for dual-side.
    syn_ids = syn.get("synapse_ids", np.full(len(pos), -1, dtype=np.int64)).astype(np.int64)

    if verbose:
        n_neurons = len(np.unique(v_labels[v_labels > 0]))
        print(f"  fetched {len(pos)} synapses from {n_neurons} v{version} neurons")

    if len(pos) > max_synapses:
        sel = rng.choice(len(pos), max_synapses, replace=False)
        pos, sv_ids, v_labels, other_labels, syn_ids = (
            pos[sel], sv_ids[sel], v_labels[sel], other_labels[sel], syn_ids[sel])

    return _assemble_world_arrays(
        pos, sv_ids, v_labels, other_labels, syn_ids,
        side=side, version=version, min_syn_per_fragment=min_syn_per_fragment,
        tok=tok, v117_ts=v117_ts, verbose=verbose, l2_skeletons=l2_skeletons)


def _assemble_world_arrays(
    pos, sv_ids, v_labels, other_labels, syn_ids,
    *, side, version, min_syn_per_fragment, tok, v117_ts, verbose, l2_skeletons,
):
    """Resolve v117 roots, sliver-filter, build fragments + Region from synapse arrays.

    Shared by ``build_region_world`` (one queried side) and ``build_region_world_dual``
    (both sides from a single fetch). Inputs are already subsampled; this does the
    roots_at + sliver filter + fragment construction + Region assembly for one side.
    """
    from neuronauts.data import lineage as L
    from neuronauts.schemas import Region

    v117_roots = L.roots_at(sv_ids, v117_ts, token=tok)
    if v117_roots is None:
        raise RuntimeError("roots_at failed for v117 — check network/token")

    keep = (v117_roots > 0) & (v_labels > 0)
    pos = pos[keep].astype(np.float32)
    frag_ids = v117_roots[keep].astype(np.int64)
    labels = v_labels[keep]
    other_labels = other_labels[keep]
    syn_ids = syn_ids[keep]
    n_obs_raw = len(pos)

    # Discard slivers — always apply, error if threshold is too aggressive
    frag_uniq, frag_counts = np.unique(frag_ids, return_counts=True)
    keep_frags = {int(f) for f, c in zip(frag_uniq, frag_counts)
                  if c >= min_syn_per_fragment}
    mask = np.array([int(f) in keep_frags for f in frag_ids])
    pos, frag_ids, labels, other_labels, syn_ids = (
        pos[mask], frag_ids[mask], labels[mask], other_labels[mask], syn_ids[mask])
    if len(pos) == 0:
        raise RuntimeError(
            f"No fragments with ≥{min_syn_per_fragment} synapses survived the sliver filter. "
            f"Max observed was {int(frag_counts.max()) if len(frag_counts) else 0}. "
            f"Try a lower --min-syn-per-fragment.")

    n_obs = len(pos)
    if verbose:
        n_frags = len(np.unique(frag_ids))
        n_neurons_kept = len(np.unique(labels[labels > 0]))
        print(f"  {n_obs} synapses, {n_frags} v117 fragments "
              f"(≥{min_syn_per_fragment} syn), {n_neurons_kept} v{version} neurons "
              f"(sliver filter dropped {n_obs_raw - n_obs} synapses)")

    if len(np.unique(labels[labels > 0])) < 2:
        raise RuntimeError("Fewer than 2 neurons in region — try a larger bbox")

    fragments = []
    root_label_map: dict[int, set[int]] = {}
    n_l2_ok = n_l2_fail = 0
    for fr in np.unique(frag_ids):
        mask = frag_ids == fr
        idxs = np.where(mask)[0]
        frag = None
        if l2_skeletons:
            from neuronauts.data.lineage import l2_skeleton
            skel = l2_skeleton(int(fr), token=tok)
            if skel is not None:
                frag = _l2_fragment(int(fr), f"minnie65_v{version}", skel, idxs)
                n_l2_ok += 1
            else:
                n_l2_fail += 1
        if frag is None:
            frag = _cloud_fragment(int(fr), f"minnie65_v{version}",
                                   pos[idxs], idxs)
        fragments.append(frag)
        root_label_map.setdefault(int(fr), set()).update(
            int(x) for x in np.unique(labels[mask]))

    n_franken = sum(1 for v in root_label_map.values() if len(v) > 1)

    mins, maxs = pos.min(0), pos.max(0)
    pad = 5000.0
    region_bbox = (tuple(float(v) for v in mins - pad),
                   tuple(float(v) for v in maxs + pad))

    # other_labels: v{version} root at the OTHER synapse endpoint (post when side="pre").
    # Swapped for side="post" so region always has (pre_root_id = the queried side's label,
    # post_root_id = the other side's label) regardless of which side we queried.
    if side == "pre":
        pre_root_id_arr, post_root_id_arr = labels, other_labels
        pre_seg_id_arr = frag_ids
        post_seg_id_arr = np.zeros(n_obs, dtype=np.int64)
    else:
        post_root_id_arr, pre_root_id_arr = labels, other_labels
        pre_seg_id_arr = np.zeros(n_obs, dtype=np.int64)
        post_seg_id_arr = frag_ids  # v117 dendritic segment IDs — join key for fragment partition

    region = Region(
        region_id=f"minnie65_v{version}",
        bbox_nm=region_bbox,
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=version,
        pre_pt_nm=pos,
        post_pt_nm=pos.copy(),  # placeholder; positions not needed for connectivity
        pre_root_id=pre_root_id_arr,
        post_root_id=post_root_id_arr,
        # Real CAVE synapse ids where available (join key for dual-side); fall back to a
        # local index only when the id column was absent (all -1).
        synapse_id=(syn_ids if np.any(syn_ids >= 0)
                    else np.arange(n_obs, dtype=np.int64)),
        pre_seg_id=pre_seg_id_arr,
        post_seg_id=post_seg_id_arr,
    ).validate()

    if verbose:
        skel_str = (f", L2 skeletons: {n_l2_ok}/{n_l2_ok + n_l2_fail} ok"
                    if l2_skeletons else "")
        n_neurons_final = len(np.unique(labels[labels > 0]))
        print(f"\n  → {n_neurons_final} neurons, {len(fragments)} v117 fragments, "
              f"{n_obs} synapses, {n_franken} frankenmerges{skel_str}")

    return fragments, region, root_label_map


# ---------------------------------------------------------------------------
# L2-node substrate (large-scale merge signal)
# ---------------------------------------------------------------------------
#
# The synapse substrate (build_region_world) samples observations at synapse
# positions and sliver-filters v117 fragments with < min_syn_per_fragment
# synapses.  In dense proofread regions this drops the small axon/dendrite
# fragments — exactly the *merge tail*.  The merge signal we want lives in the
# L2 arbor, not the sparse synapse cloud.
#
# build_region_world_l2 builds the same (fragments, region, root_label_map)
# contract, but observations are **L2 nodes** (one rep_coord_nm per node) clipped
# to the bbox.  It is synapse-free: it seeds from nucleus somas in the bbox,
# walks each proofread (v{version}) neuron's L2 graph, and labels every in-bbox
# L2 node with its v117 fragment root.  post_root_id is left at 0 (no
# connectivity); connectome metrics are naturally skipped for this substrate.


def _load_nucleus_somas(cache_path: Optional[str] = None) -> np.ndarray:
    """Return nucleus somas as a structured array ``(sv_id, x_nm, y_nm, z_nm)``.

    Parses the raw merged nucleus CSV directly.  Its columns are
    ``(id, volume, supervoxel_id, root_id_v1412, x_vox, y_vox, z_vox)`` with
    voxel size **(4, 4, 40) nm** (mip0) — the same nm frame as the L2 cache's
    ``rep_coord_nm``, so soma positions and L2 node positions are directly
    comparable.  (``load_nucleus_positions`` both mis-parses the columns and
    uses an (8, 8, 40) scale, which double-counts x/y; we parse correctly here.)
    Rows without a supervoxel (sv == 0) are excluded.
    """
    import gzip, io, os
    import requests
    from neuronauts.data.loaders import _NUCLEUS_URL

    if cache_path is not None and os.path.exists(cache_path):
        d = np.load(cache_path)
        return d["somas"]

    resp = requests.get(_NUCLEUS_URL, timeout=60)
    resp.raise_for_status()
    recs: list[tuple] = []
    with gzip.open(io.BytesIO(resp.content)) as fh:
        for line in fh:
            p = line.decode().strip().split(",")
            if len(p) < 7:
                continue
            try:
                sv = int(p[2]); x = int(p[4]); y = int(p[5]); z = int(p[6])
            except ValueError:
                continue
            if sv > 0:
                recs.append((sv, x * 4.0, y * 4.0, z * 40.0))
    dt = np.dtype([("sv", np.uint64), ("x_nm", np.float64),
                   ("y_nm", np.float64), ("z_nm", np.float64)])
    somas = np.array(recs, dtype=dt)
    if cache_path is not None:
        np.savez_compressed(cache_path, somas=somas)
    return somas


def _l2_nodes_with_coords(root_id: int, *, token: str, bounds_seg_vox=None):
    """Return ``(l2_ids[uint64], coords_nm[float32, (N,3)])`` for a root.

    Walks the root's L2 nodes (``root_leaves`` at ``stop_layer=2``) and fetches
    ``rep_coord_nm`` from the L2 attribute cache.  L2 nodes without a cached
    coordinate are dropped.  Returns ``(empty, empty)`` on any failure.

    ``bounds_seg_vox`` is an optional ``((x0,x1),(y0,y1),(z0,z1))`` in
    **(8,8,40) nm segmentation voxels**.  When given, the chunkedgraph restricts
    the returned L2 nodes to chunks intersecting that box — a large speedup for
    big arbors (the caller still filters by rep_coord for an exact clip).
    """
    import requests
    from neuronauts.data import lineage as L

    params = {"stop_layer": 2}
    if bounds_seg_vox is not None:
        (bx0, bx1), (by0, by1), (bz0, bz1) = bounds_seg_vox
        params["bounds"] = f"{int(bx0)}-{int(bx1)}_{int(by0)}-{int(by1)}_{int(bz0)}-{int(bz1)}"
        url = (f"{L.CG_SERVER}/segmentation/api/v1/table/{L.SEG_TABLE}"
               f"/node/{int(root_id)}/leaves")
        try:
            resp = requests.get(url, headers=L._headers(token), params=params, timeout=120)
            l2ids = (np.asarray(resp.json()["leaf_ids"], dtype=np.uint64)
                     if resp.status_code == 200 else None)
        except Exception:
            l2ids = None
    else:
        l2ids = L.root_leaves(int(root_id), stop_layer=2, token=token)
    if l2ids is None or len(l2ids) == 0:
        return np.zeros(0, np.uint64), np.zeros((0, 3), np.float32)

    url = f"{L.L2_CACHE_SERVER}/l2cache/api/v1/table/{L.L2_TABLE}/attributes"
    hdr = {**L._headers(token), "Content-Type": "application/json"}
    coords: dict[int, np.ndarray] = {}
    try:
        for start in range(0, len(l2ids), L._L2_BATCH):
            chunk = l2ids[start:start + L._L2_BATCH].tolist()
            body = {"l2_ids": chunk, "attribute_names": ["rep_coord_nm"]}
            resp = requests.post(url, headers=hdr, json=body, timeout=120)
            if resp.status_code != 200:
                continue
            for id_str, attrs in resp.json().items():
                c = attrs.get("rep_coord_nm")
                if c is not None and len(c) == 3:
                    coords[int(id_str)] = np.asarray(c, dtype=np.float32)
    except Exception:
        return np.zeros(0, np.uint64), np.zeros((0, 3), np.float32)

    if not coords:
        return np.zeros(0, np.uint64), np.zeros((0, 3), np.float32)
    ids = np.array(list(coords.keys()), dtype=np.uint64)
    pts = np.stack([coords[int(i)] for i in ids], axis=0).astype(np.float32)
    return ids, pts


def build_region_world_l2(
    bbox_nm: tuple,
    *,
    version: int = 1718,
    max_neurons: int = 0,
    min_l2_per_fragment: int = 2,
    v117_timestamp: Optional[int] = None,
    token: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
    nucleus_cache_path: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> tuple:
    """Assemble an L2-node f(v117→v{version}) partition world from a bbox.

    Observations are **L2 nodes** (not synapses), so the small axon/dendrite
    fragments that carry the merge signal survive.  Seeds from nucleus somas in
    the bbox, walks each proofread neuron's L2 graph, clips to the bbox, and
    tags every L2 node with its v117 fragment root.

    Parameters
    ----------
    bbox_nm:
        ``((x0, y0, z0), (x1, y1, z1))`` in nm.
    version:
        Proofread label materialization (default 1718).
    max_neurons:
        Cap the number of seed neurons (0 = all somas in bbox).  Useful to
        bound runtime on large/dense boxes.
    min_l2_per_fragment:
        Discard v117 fragments with fewer than this many in-bbox L2 nodes.
        Default 2 (far less aggressive than the synapse sliver filter, by design
        — the whole point is to keep the small fragments).
    nucleus_cache_path:
        Optional cache path for the nucleus position table.
    cache_path:
        Optional ``.npz`` path caching the assembled L2-node arrays
        (pos / frag / label / l2 id) for this bbox.  The L2 walk is the
        expensive step (~seconds per neuron); with a cache, reruns are instant.

    Returns
    -------
    (fragments, region, root_label_map)
        Same contract as ``build_region_world`` — drop-in for
        ``build_observation_graph`` and the partition pipeline.  ``post_root_id``
        is all-zero (no connectivity on this substrate).
    """
    import os
    from neuronauts.data import lineage as L
    from neuronauts.data.loaders import DEFAULT_TOKEN
    from neuronauts.schemas import Region

    tok = token or DEFAULT_TOKEN
    v117_ts = v117_timestamp if v117_timestamp is not None else L.V117_TIMESTAMP
    v_ts = L.version_timestamp(version)
    rng = np.random.default_rng(seed)

    (x0, y0, z0), (x1, y1, z1) = bbox_nm
    if verbose:
        print(f"Building L2 region world v117→v{version}: "
              f"[{x0:.0f},{y0:.0f},{z0:.0f}]–[{x1:.0f},{y1:.0f},{z1:.0f}] nm …")

    cached_arrays = None
    if cache_path is not None and os.path.exists(cache_path):
        d = np.load(cache_path)
        cached_arrays = (d["pos"], d["frag_ids"], d["labels"], d["l2_ids"])
        if verbose:
            print(f"  loaded cached L2 arrays: {len(d['pos'])} nodes")

    if cached_arrays is not None:
        pos, frag_ids, labels, l2_ids = cached_arrays
        pos = pos.astype(np.float32)
        frag_ids = frag_ids.astype(np.int64)
        labels = labels.astype(np.int64)
        l2_ids = l2_ids.astype(np.uint64)
        return _assemble_l2_world(
            pos, frag_ids, labels, l2_ids, version=version,
            min_l2_per_fragment=min_l2_per_fragment, verbose=verbose)

    # 1. Nucleus somas in the bbox → seed v{version} roots.
    somas = _load_nucleus_somas(cache_path=nucleus_cache_path)
    in_box = ((somas["x_nm"] >= x0) & (somas["x_nm"] < x1) &
              (somas["y_nm"] >= y0) & (somas["y_nm"] < y1) &
              (somas["z_nm"] >= z0) & (somas["z_nm"] < z1))
    soma = somas[in_box]
    sv_seed = soma["sv"].astype(np.uint64)
    sv_seed = sv_seed[sv_seed > 0]
    if len(sv_seed) == 0:
        raise RuntimeError("No nucleus somas with supervoxels in bbox")

    seed_roots = L.roots_at(sv_seed, v_ts, token=tok)
    seed_roots = np.unique(seed_roots[seed_roots > 0])
    if max_neurons and len(seed_roots) > max_neurons:
        seed_roots = rng.choice(seed_roots, max_neurons, replace=False)
    if verbose:
        print(f"  {len(soma)} somas in bbox → {len(seed_roots)} seed v{version} neurons"
              f"{f' (capped from more)' if max_neurons else ''}")

    # 2. Walk each neuron's L2 graph, clip to bbox, tag with v117 fragment.
    #    Restrict the L2 walk to chunks intersecting the bbox (8,8,40 seg voxels)
    #    so big arbors don't pull their whole (often volume-spanning) node set.
    bounds_seg_vox = ((x0 / 8, x1 / 8), (y0 / 8, y1 / 8), (z0 / 40, z1 / 40))
    all_pos: list[np.ndarray] = []
    all_frag: list[np.ndarray] = []
    all_label: list[np.ndarray] = []
    all_l2: list[np.ndarray] = []
    n_done = 0
    for rt in seed_roots:
        ids, pts = _l2_nodes_with_coords(int(rt), token=tok,
                                         bounds_seg_vox=bounds_seg_vox)
        if len(ids) == 0:
            continue
        keep = ((pts[:, 0] >= x0) & (pts[:, 0] < x1) &
                (pts[:, 1] >= y0) & (pts[:, 1] < y1) &
                (pts[:, 2] >= z0) & (pts[:, 2] < z1))
        ids, pts = ids[keep], pts[keep]
        if len(ids) == 0:
            continue
        v117 = L.roots_at(ids, v117_ts, token=tok)
        ok = v117 > 0
        if not ok.any():
            continue
        all_pos.append(pts[ok])
        all_frag.append(v117[ok].astype(np.int64))
        all_label.append(np.full(int(ok.sum()), int(rt), dtype=np.int64))
        all_l2.append(ids[ok])
        n_done += 1
        if verbose and n_done % 25 == 0:
            print(f"    {n_done}/{len(seed_roots)} neurons walked, "
                  f"{sum(len(p) for p in all_pos)} L2 nodes so far …")

    if not all_pos:
        raise RuntimeError("No in-bbox L2 nodes resolved — check bbox/network")

    pos = np.concatenate(all_pos).astype(np.float32)
    frag_ids = np.concatenate(all_frag)
    labels = np.concatenate(all_label)
    l2_ids = np.concatenate(all_l2)

    if cache_path is not None:
        np.savez_compressed(cache_path, pos=pos, frag_ids=frag_ids,
                            labels=labels, l2_ids=l2_ids)
        if verbose:
            print(f"  cached {len(pos)} L2 nodes → {cache_path}")

    return _assemble_l2_world(
        pos, frag_ids, labels, l2_ids, version=version,
        min_l2_per_fragment=min_l2_per_fragment, verbose=verbose)


def _assemble_l2_world(pos, frag_ids, labels, l2_ids, *, version,
                       min_l2_per_fragment, verbose):
    """Sliver-filter L2 nodes, build per-v117 fragments + Region + label map.

    Shared by the fresh-walk and cached paths of ``build_region_world_l2``.
    """
    from neuronauts.schemas import Region

    # Sliver filter on fragments (keep the tail: default min=2).
    fu, fc = np.unique(frag_ids, return_counts=True)
    keep_frags = {int(f) for f, c in zip(fu, fc) if c >= min_l2_per_fragment}
    mask = np.array([int(f) in keep_frags for f in frag_ids])
    pos, frag_ids, labels, l2_ids = pos[mask], frag_ids[mask], labels[mask], l2_ids[mask]
    n_obs = len(pos)
    if n_obs == 0:
        raise RuntimeError(
            f"No fragments with ≥{min_l2_per_fragment} L2 nodes survived; "
            f"max observed was {int(fc.max()) if len(fc) else 0}.")

    # Build per-v117 fragments (point-cloud / kNN skeleton over their L2 nodes).
    fragments = []
    root_label_map: dict[int, set[int]] = {}
    for fr in np.unique(frag_ids):
        idxs = np.where(frag_ids == fr)[0]
        fragments.append(
            _cloud_fragment(int(fr), f"minnie65_v{version}", pos[idxs], idxs))
        root_label_map.setdefault(int(fr), set()).update(
            int(x) for x in np.unique(labels[idxs]))

    n_franken = sum(1 for v in root_label_map.values() if len(v) > 1)

    mins, maxs = pos.min(0), pos.max(0)
    pad = 5000.0
    region_bbox = (tuple(float(v) for v in mins - pad),
                   tuple(float(v) for v in maxs + pad))
    zeros = np.zeros(n_obs, dtype=np.int64)
    region = Region(
        region_id=f"minnie65_v{version}_l2",
        bbox_nm=region_bbox,
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=version,
        pre_pt_nm=pos,
        post_pt_nm=pos.copy(),
        pre_root_id=labels,
        post_root_id=zeros.copy(),     # no connectivity on the L2 substrate
        synapse_id=l2_ids.astype(np.int64),  # L2 node id as the observation id
        pre_seg_id=frag_ids,
        post_seg_id=zeros.copy(),
    ).validate()

    if verbose:
        n_neurons = len(np.unique(labels[labels > 0]))
        frag_per_neuron: dict[int, int] = {}
        for f, ls in root_label_map.items():
            for l in ls:
                frag_per_neuron[l] = frag_per_neuron.get(l, 0) + 1
        fpn = np.array(list(frag_per_neuron.values()))
        print(f"\n  → {n_neurons} neurons, {len(fragments)} v117 fragments, "
              f"{n_obs} L2 nodes, {n_franken} frankenmerges")
        print(f"     v117 fragments/neuron: median={np.median(fpn):.0f} "
              f"mean={fpn.mean():.1f} max={fpn.max()} "
              f"(≥2 merges: {int((fpn >= 2).sum())}/{len(fpn)})")

    return fragments, region, root_label_map


def build_region_world_dual(
    bbox_nm: tuple,
    *,
    version: int = 1718,
    max_synapses: int = 20_000,
    min_syn_per_fragment: int = 5,
    v117_timestamp: Optional[int] = None,
    token: Optional[str] = None,
    seed: int = 0,
    verbose: bool = True,
    l2_skeletons_pre: bool = True,
    l2_skeletons_post: bool = False,
    tile_x_nm: float = 0,
    per_tile_limit: int = 200_000,
) -> tuple:
    """Build pre- and post-side worlds for the SAME synapses from a single fetch.

    Two independent ``build_region_world`` calls (one per side) cannot be joined:
    each spatial fetch is capped/subsampled at ``max_synapses`` independently, so the
    pre and post fetches sample different synapses and share almost no real ids. This
    builder instead does ONE fetch (``side="pre"``) that carries BOTH endpoints of each
    synapse (positions, supervoxels, roots, and the shared CAVE id), subsamples once,
    then assembles a pre world and a post world over the identical synapse set. Every
    synapse therefore appears on both sides with the same ``synapse_id`` — a guaranteed
    join for :func:`treestitch.connectivity.dual_side_connectome_accuracy`.

    Returns
    -------
    ((frags_pre, region_pre, lmap_pre), (frags_post, region_post, lmap_post))
        Each side independently sliver-filtered; the shared ``synapse_id`` space is
        what joins them.
    """
    from neuronauts.data import lineage as L
    from neuronauts.data.loaders import DEFAULT_TOKEN

    tok = token or DEFAULT_TOKEN
    v117_ts = v117_timestamp if v117_timestamp is not None else L.V117_TIMESTAMP
    rng = np.random.default_rng(seed)

    if verbose:
        (x0, y0, z0), (x1, y1, z1) = bbox_nm
        print(f"Building DUAL region world v117→v{version}: "
              f"[{x0:.0f},{y0:.0f},{z0:.0f}]–[{x1:.0f},{y1:.0f},{z1:.0f}] nm "
              f"(single fetch, both sides) …")

    syn = None
    if tile_x_nm > 0:
        syn = L.fetch_region_synapses_tiled(
            bbox_nm, version=version, side="pre",
            tile_x_nm=tile_x_nm, per_tile_limit=per_tile_limit, token=tok)
        if syn is None or len(syn["positions_nm"]) == 0:
            raise RuntimeError(
                "No synapses from tiled fetch — check network/token/version/bbox")
    else:
        effective_limit = max_synapses
        while effective_limit >= 1000:
            syn = L.fetch_region_synapses(bbox_nm, version=version, side="pre",
                                           limit=effective_limit, token=tok)
            if syn is not None:
                break
            effective_limit //= 2
        if syn is None or len(syn["positions_nm"]) == 0:
            raise RuntimeError(
                "No synapses returned for bbox — check network/token/version/bbox")

    # Pre side = queried side; post side = the OTHER endpoint of the same rows.
    pos_pre = syn["positions_nm"]
    sv_pre = syn["supervoxel_ids"]
    root_pre = syn["root_ids"].astype(np.int64)
    pos_post = syn["other_positions_nm"]
    sv_post = syn["other_supervoxel_ids"]
    root_post = syn["other_root_ids"].astype(np.int64)
    syn_ids = syn.get("synapse_ids", np.full(len(pos_pre), -1, dtype=np.int64)).astype(np.int64)

    if np.all(sv_post == 0) or np.all(syn_ids < 0):
        raise RuntimeError(
            "Dual fetch missing other-side supervoxels or synapse ids — the synapse "
            "table did not return post_pt columns / id. Cannot build a joinable dual world.")

    if verbose:
        print(f"  fetched {len(pos_pre)} synapses (both endpoints, shared ids)")

    # Subsample ONCE on the shared synapse set so both sides see the same synapses.
    if len(pos_pre) > max_synapses:
        sel = rng.choice(len(pos_pre), max_synapses, replace=False)
        pos_pre, sv_pre, root_pre = pos_pre[sel], sv_pre[sel], root_pre[sel]
        pos_post, sv_post, root_post = pos_post[sel], sv_post[sel], root_post[sel]
        syn_ids = syn_ids[sel]

    if verbose:
        print("  [pre side]")
    pre = _assemble_world_arrays(
        pos_pre, sv_pre, root_pre, root_post, syn_ids,
        side="pre", version=version, min_syn_per_fragment=min_syn_per_fragment,
        tok=tok, v117_ts=v117_ts, verbose=verbose, l2_skeletons=l2_skeletons_pre)
    if verbose:
        print("  [post side]")
    post = _assemble_world_arrays(
        pos_post, sv_post, root_post, root_pre, syn_ids,
        side="post", version=version, min_syn_per_fragment=min_syn_per_fragment,
        tok=tok, v117_ts=v117_ts, verbose=verbose, l2_skeletons=l2_skeletons_post)
    return pre, post


__all__ = ["build_lineage_world", "build_region_world", "build_region_world_dual"]
