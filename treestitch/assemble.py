"""Post-partition neuron shape assembly.

Takes the output of ``partition_observations_cc`` (a cluster label per synapse)
and the per-fragment L2-cache skeletons produced during world-building, then
assembles them into whole-neuron skeleton ``Fragment`` objects.

Each assembled neuron is a tree (or forest if some fragments are too far apart
to stitch): the merger uses Kruskal on the fragment-level endpoint-proximity
graph so no cycles can be introduced.

Pipeline
--------
    # After partitioning:
    pred = partition_observations_cc(model, graph, bias=-1.0)
    shapes = assemble_partition_shapes(fragments, pred, graph.fragment_id)
    for neuron_id, neuron_frag in shapes.items():
        m = neuron_shape_metrics(neuron_frag)
        print(neuron_id, m["cable_length_um"], m["is_tree"])
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from neuronauts.schemas import Fragment


# ---------------------------------------------------------------------------
# Internal: minimal union-find for Kruskal
# ---------------------------------------------------------------------------

class _UF:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, x: int) -> int:
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]
            x = self._p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self._p[ra] = rb
        return True


# ---------------------------------------------------------------------------
# Core: merge a list of Fragment skeletons into one (tree-compliant)
# ---------------------------------------------------------------------------

def merge_fragment_skeletons(
    fragments: list[Fragment],
    *,
    stitch_radius_nm: float = 5_000.0,
) -> Fragment:
    """Merge fragment-level skeletons into one neuron-level skeleton.

    Produces a tree (or forest if some fragments are beyond ``stitch_radius_nm``
    from all others), provided that each input fragment is itself a spanning tree
    (i.e., built from an L2-cache MST via ``_l2_fragment``).  Kruskal selection of
    inter-fragment bridges guarantees no new cycles are introduced — it adds at most
    (n_fragments − 1) bridges.

    Note: synapse-cloud fragments (``_cloud_fragment``) use k-NN edges and may
    already contain cycles.  The merger never introduces additional cycles, but
    pre-existing ones are preserved.  Use ``neuron_shape_metrics``'s ``is_tree``
    field to check at runtime.

    Parameters
    ----------
    fragments:
        All fragments that belong to one predicted neuron.
    stitch_radius_nm:
        Maximum endpoint gap (nm) for a candidate bridge edge.

    Returns
    -------
    Fragment
        Merged skeleton.  ``fragment_id`` and ``base_root_id`` are taken from the
        first fragment.  ``dna`` is ``None`` — re-encode with FragmentEncoder if
        needed.  ``region_id`` is set to ``"merged"``.
    """
    if not fragments:
        raise ValueError("merge_fragment_skeletons: fragment list is empty")

    if len(fragments) == 1:
        f = fragments[0]
        return Fragment(
            fragment_id=f.fragment_id,
            region_id=f.region_id,
            base_root_id=f.base_root_id,
            vertices_nm=f.vertices_nm.copy(),
            edges=f.edges.copy(),
            endpoints_nm=f.endpoints_nm.copy(),
            radius_nm=f.radius_nm.copy(),
            synapse_indices=f.synapse_indices.copy(),
            dna=None,
        )

    # Step 1: pool vertices and reindex edges into the global vertex array.
    vert_chunks: list[np.ndarray] = []
    rad_chunks: list[np.ndarray] = []
    edge_chunks: list[np.ndarray] = []
    syn_chunks: list[np.ndarray] = []
    vert_offsets: list[int] = []  # global vertex index of fragment i's vertex 0

    offset = 0
    for f in fragments:
        vert_chunks.append(np.asarray(f.vertices_nm, dtype=np.float32))
        rad_chunks.append(np.asarray(f.radius_nm, dtype=np.float32))
        syn_chunks.append(np.asarray(f.synapse_indices, dtype=np.int64))
        if len(f.edges):
            edge_chunks.append(np.asarray(f.edges, dtype=np.int64) + offset)
        vert_offsets.append(offset)
        offset += len(f.vertices_nm)

    all_verts = np.concatenate(vert_chunks, axis=0)   # [V_total, 3]
    all_radii = np.concatenate(rad_chunks, axis=0)    # [V_total]
    all_syn   = np.concatenate(syn_chunks, axis=0) if any(len(s) for s in syn_chunks) else np.empty(0, np.int64)
    base_edges = np.concatenate(edge_chunks, axis=0) if edge_chunks else np.empty((0, 2), dtype=np.int64)

    # Step 2: find candidate inter-fragment bridge edges via endpoint proximity.
    # Collect all endpoints with their fragment and global vertex index.
    ep_frag: list[int] = []
    ep_vidx: list[int] = []   # global vertex index of this endpoint
    ep_pts:  list[np.ndarray] = []

    for fi, f in enumerate(fragments):
        verts_f = vert_chunks[fi]
        # endpoints are identified by degree ≤ 1 in the fragment's own edge set
        ep_local = _endpoint_local_indices(f)
        for li in ep_local:
            ep_frag.append(fi)
            ep_vidx.append(vert_offsets[fi] + li)
            ep_pts.append(verts_f[li])

    if not ep_pts:
        # No endpoints at all — return pooled vertices with no bridge edges.
        return _make_merged_fragment(fragments, all_verts, all_radii, base_edges, all_syn, [])

    ep_arr = np.stack(ep_pts, axis=0).astype(np.float64)  # [P, 3]

    # KD-tree (or brute force) radius search.
    candidates: list[tuple[float, int, int, int, int]] = []
    # (distance, frag_i, frag_j, global_vidx_i, global_vidx_j)
    try:
        from scipy.spatial import KDTree
        kd = KDTree(ep_arr)
        pairs = kd.query_pairs(r=stitch_radius_nm, output_type="ndarray")
    except ImportError:
        pairs = _brute_endpoint_pairs(ep_arr, stitch_radius_nm)

    for pi, pj in pairs:
        fi, fj = ep_frag[pi], ep_frag[pj]
        if fi == fj:
            continue  # same fragment — already connected
        dist = float(np.linalg.norm(ep_arr[pi] - ep_arr[pj]))
        candidates.append((dist, fi, fj, ep_vidx[pi], ep_vidx[pj]))

    # Step 3: Kruskal — accept bridges that connect previously disconnected fragments.
    candidates.sort(key=lambda x: x[0])
    uf = _UF(len(fragments))
    bridge_edges: list[tuple[int, int]] = []  # (global_vidx_i, global_vidx_j)

    for dist, fi, fj, vi, vj in candidates:
        if uf.union(fi, fj):
            bridge_edges.append((vi, vj))

    return _make_merged_fragment(fragments, all_verts, all_radii, base_edges, all_syn, bridge_edges)


def _endpoint_local_indices(f: Fragment) -> list[int]:
    """Return local vertex indices with degree ≤ 1 (leaves / endpoints)."""
    if len(f.vertices_nm) == 1:
        return [0]
    if len(f.edges) == 0:
        return list(range(len(f.vertices_nm)))
    degree = np.zeros(len(f.vertices_nm), dtype=np.int32)
    for u, v in f.edges:
        degree[u] += 1
        degree[v] += 1
    return [int(i) for i in np.where(degree <= 1)[0]]


def _brute_endpoint_pairs(pts: np.ndarray, radius: float) -> np.ndarray:
    """O(P²) fallback when scipy is unavailable."""
    P = len(pts)
    pairs = []
    for i in range(P):
        for j in range(i + 1, P):
            if float(np.linalg.norm(pts[i] - pts[j])) <= radius:
                pairs.append([i, j])
    return np.array(pairs, dtype=np.int64).reshape(-1, 2) if pairs else np.empty((0, 2), dtype=np.int64)


def _make_merged_fragment(
    fragments: list[Fragment],
    all_verts: np.ndarray,
    all_radii: np.ndarray,
    base_edges: np.ndarray,
    all_syn: np.ndarray,
    bridge_edges: list[tuple[int, int]],
) -> Fragment:
    if bridge_edges:
        bridges = np.array(bridge_edges, dtype=np.int64)
        if len(base_edges):
            all_edges = np.concatenate([base_edges, bridges], axis=0)
        else:
            all_edges = bridges
    else:
        all_edges = base_edges

    # Re-extract endpoints (degree ≤ 1) of the merged skeleton.
    merged_endpoints = _endpoint_global_indices(all_verts, all_edges)

    f0 = fragments[0]
    return Fragment(
        fragment_id=f0.fragment_id,
        region_id="merged",
        base_root_id=f0.base_root_id,
        vertices_nm=all_verts,
        edges=all_edges,
        endpoints_nm=merged_endpoints,
        radius_nm=all_radii,
        synapse_indices=all_syn,
        dna=None,
    )


def _endpoint_global_indices(verts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return [T, 3] float32 of leaf vertex coordinates."""
    n = len(verts)
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(edges) == 0:
        return verts.copy()
    degree = np.zeros(n, dtype=np.int32)
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    leaf_idx = np.where(degree <= 1)[0]
    return verts[leaf_idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Partition → shape dictionary
# ---------------------------------------------------------------------------

def assemble_partition_shapes(
    fragment_list: list[Fragment],
    pred_labels: np.ndarray,
    seg_ids: np.ndarray,
    *,
    stitch_radius_nm: float = 5_000.0,
    min_fragments: int = 1,
) -> dict[int, Fragment]:
    """Build a dict of {cluster_id → merged neuron Fragment} from a partition.

    Parameters
    ----------
    fragment_list:
        All Fragment objects from the world-building step (one per v117 root).
        Order must match the ``base_root_id`` values used in world-building.
    pred_labels:
        [N] int64 — per-synapse/observation cluster ID from ``partition_observations_cc``.
        Abstained observations have negative IDs and are ignored.
    seg_ids:
        [N] int64 — per-synapse v117 root ID (``graph.fragment_id`` from ObservationGraph).
        Maps each observation to the fragment it belongs to.
    stitch_radius_nm:
        Forwarded to ``merge_fragment_skeletons``.
    min_fragments:
        Skip clusters with fewer than this many fragments.

    Returns
    -------
    dict[int, Fragment]
        Merged skeleton per predicted neuron cluster.  Only non-abstained clusters
        are included.
    """
    # Build root_id → Fragment lookup
    root_to_frag: dict[int, Fragment] = {f.base_root_id: f for f in fragment_list}

    # Group fragments by predicted cluster label (skip abstained = negative labels)
    cluster_to_roots: dict[int, set[int]] = defaultdict(set)
    for obs_idx in range(len(pred_labels)):
        label = int(pred_labels[obs_idx])
        if label < 0:
            continue
        root_id = int(seg_ids[obs_idx])
        if root_id in root_to_frag:
            cluster_to_roots[label].add(root_id)

    # Assemble each cluster
    result: dict[int, Fragment] = {}
    for cluster_id, root_ids in cluster_to_roots.items():
        frags = [root_to_frag[r] for r in root_ids if r in root_to_frag]
        if len(frags) < min_fragments:
            continue
        result[cluster_id] = merge_fragment_skeletons(frags, stitch_radius_nm=stitch_radius_nm)

    return result


# ---------------------------------------------------------------------------
# Shape metrics
# ---------------------------------------------------------------------------

def _max_path_bfs(adj: list[list[tuple[int, float]]], start: int
                   ) -> tuple[int, float, list[int]]:
    """BFS from `start` on a weighted undirected adjacency list.

    Returns (farthest_node, arc_length, path) using arc-length as distance.
    """
    from collections import deque
    dist = {start: 0.0}
    parent: dict[int, int] = {start: -1}
    queue: deque[int] = deque([start])
    farthest, max_d = start, 0.0
    while queue:
        u = queue.popleft()
        for v, w in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + w
                parent[v] = u
                queue.append(v)
                if dist[v] > max_d:
                    max_d, farthest = dist[v], v
    # Reconstruct path
    path = []
    node = farthest
    while node != -1:
        path.append(node)
        node = parent[node]
    return farthest, max_d, path[::-1]


def neuron_shape_metrics(neuron: Fragment) -> dict:
    """Morphological sanity metrics for an assembled neuron skeleton.

    Returns
    -------
    dict with keys:
        cable_length_um        float  — total edge length in micrometres
        n_branch_points        int    — vertices with degree ≥ 3
        n_endpoints            int    — vertices with degree ≤ 1
        n_connected_components int    — 1 = fully connected; >1 = stitch gap
        is_tree                bool   — True if no cycles
        bbox_volume_um3        float  — axis-aligned bounding-box volume in μm³
        max_path_length_um     float  — geodesic diameter (longest endpoint-to-endpoint path)
        tortuosity             float  — arc_length / Euclidean distance for the diameter path
        mean_caliber_um        float  — mean vertex radius in µm (nan if radius unavailable)
    """
    verts = neuron.vertices_nm  # [V, 3] float32
    edges = neuron.edges        # [E, 2] int64
    V = len(verts)
    E = len(edges)

    if V == 0:
        return {
            "cable_length_um": 0.0,
            "n_branch_points": 0,
            "n_endpoints": 0,
            "n_connected_components": 0,
            "is_tree": True,
            "bbox_volume_um3": 0.0,
            "max_path_length_um": 0.0,
            "tortuosity": float("nan"),
            "mean_caliber_um": float("nan"),
        }

    # Edge lengths in µm
    if E > 0:
        u_idx = edges[:, 0]
        v_idx = edges[:, 1]
        diffs = verts[u_idx].astype(np.float64) - verts[v_idx].astype(np.float64)
        edge_len_um = np.linalg.norm(diffs, axis=1) / 1_000.0
        cable_length_um = float(edge_len_um.sum())
    else:
        edge_len_um = np.zeros(0)
        cable_length_um = 0.0

    # Degree-based metrics + adjacency list for BFS
    degree = np.zeros(V, dtype=np.int32)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(V)]
    for k, (u, v) in enumerate(edges):
        w = float(edge_len_um[k]) if E > 0 else 0.0
        degree[int(u)] += 1
        degree[int(v)] += 1
        adj[int(u)].append((int(v), w))
        adj[int(v)].append((int(u), w))
    n_branch_points = int((degree >= 3).sum())
    n_endpoints = int((degree <= 1).sum())

    # Connected components via union-find
    uf = _UF(V)
    for u, v in edges:
        uf.union(int(u), int(v))
    n_components = len({uf.find(i) for i in range(V)})
    is_tree = (E == V - n_components)

    # Bounding box
    lo = verts.min(axis=0).astype(np.float64)
    hi = verts.max(axis=0).astype(np.float64)
    extents_um = (hi - lo) / 1_000.0
    bbox_volume_um3 = float(extents_um[0] * extents_um[1] * extents_um[2])

    # Geodesic diameter: BFS from each endpoint; take max arc-length path.
    # For isolated single vertices (no edges) treat as diameter=0.
    leaf_nodes = [i for i in range(V) if degree[i] <= 1]
    if not leaf_nodes:
        leaf_nodes = [0]    # fully branched graph — start from any node
    max_path_length_um = 0.0
    tortuosity = float("nan")
    if E > 0:
        # Double-sweep: BFS from first leaf → find far end → BFS again for true diameter
        _, _, _ = _max_path_bfs(adj, leaf_nodes[0])
        far1, arc1, path1 = _max_path_bfs(adj, leaf_nodes[0])
        far2, arc2, path2 = _max_path_bfs(adj, far1)
        if arc2 >= arc1:
            diam_arc, diam_path = arc2, path2
        else:
            diam_arc, diam_path = arc1, path1
        max_path_length_um = diam_arc
        if len(diam_path) >= 2:
            p0 = verts[diam_path[0]].astype(np.float64)
            p1 = verts[diam_path[-1]].astype(np.float64)
            eucl = float(np.linalg.norm(p1 - p0)) / 1_000.0
            tortuosity = diam_arc / eucl if eucl > 0 else float("nan")

    # Mean caliber (radius_nm available from L2-cache skeletons)
    radius = getattr(neuron, "radius_nm", None)
    if radius is not None and len(radius) > 0 and float(np.max(radius)) > 0:
        mean_caliber_um = float(np.mean(radius)) / 1_000.0
    else:
        mean_caliber_um = float("nan")

    return {
        "cable_length_um": cable_length_um,
        "n_branch_points": n_branch_points,
        "n_endpoints": n_endpoints,
        "n_connected_components": n_components,
        "is_tree": is_tree,
        "bbox_volume_um3": bbox_volume_um3,
        "max_path_length_um": max_path_length_um,
        "tortuosity": tortuosity,
        "mean_caliber_um": mean_caliber_um,
    }


__all__ = [
    "merge_fragment_skeletons",
    "assemble_partition_shapes",
    "neuron_shape_metrics",
]
