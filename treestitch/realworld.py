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
    else:
        edges = np.zeros((0, 2), dtype=np.int64)
        endpoints = pts.copy()

    return Fragment(
        fragment_id=frag_id,
        region_id=region_id,
        base_root_id=frag_id,
        vertices_nm=pts,
        edges=edges,
        endpoints_nm=endpoints,
        radius_nm=np.full(m, 200.0, dtype=np.float32),
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

    syn = L.fetch_region_synapses(bbox_nm, version=version, side=side,
                                   limit=max_synapses * 3, token=tok)
    if syn is None or len(syn["positions_nm"]) == 0:
        raise RuntimeError(
            "No synapses returned for bbox — check network/token/version/bbox")

    pos = syn["positions_nm"]
    sv_ids = syn["supervoxel_ids"]
    v_labels = syn["root_ids"].astype(np.int64)

    if verbose:
        n_neurons = len(np.unique(v_labels[v_labels > 0]))
        print(f"  fetched {len(pos)} synapses from {n_neurons} v{version} neurons")

    if len(pos) > max_synapses:
        sel = rng.choice(len(pos), max_synapses, replace=False)
        pos, sv_ids, v_labels = pos[sel], sv_ids[sel], v_labels[sel]

    v117_roots = L.roots_at(sv_ids, v117_ts, token=tok)
    if v117_roots is None:
        raise RuntimeError("roots_at failed for v117 — check network/token")

    keep = (v117_roots > 0) & (v_labels > 0)
    pos = pos[keep].astype(np.float32)
    frag_ids = v117_roots[keep].astype(np.int64)
    labels = v_labels[keep]
    n_obs_raw = len(pos)

    # Discard slivers
    frag_uniq, frag_counts = np.unique(frag_ids, return_counts=True)
    keep_frags = {int(f) for f, c in zip(frag_uniq, frag_counts)
                  if c >= min_syn_per_fragment}
    if keep_frags:
        mask = np.array([int(f) in keep_frags for f in frag_ids])
        pos, frag_ids, labels = pos[mask], frag_ids[mask], labels[mask]

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

    post_pts = pos + rng.normal(0, 2000, pos.shape).astype(np.float32)
    mins, maxs = pos.min(0), pos.max(0)
    pad = 5000.0
    region_bbox = (tuple(float(v) for v in mins - pad),
                   tuple(float(v) for v in maxs + pad))

    region = Region(
        region_id=f"minnie65_v{version}",
        bbox_nm=region_bbox,
        voxel_size_nm=(8.0, 8.0, 40.0),
        seg_version=117,
        label_version=version,
        pre_pt_nm=pos,
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
        n_neurons_final = len(np.unique(labels[labels > 0]))
        print(f"\n  → {n_neurons_final} neurons, {len(fragments)} v117 fragments, "
              f"{n_obs} synapses, {n_franken} frankenmerges{skel_str}")

    return fragments, region, root_label_map


__all__ = ["build_lineage_world", "build_region_world"]
