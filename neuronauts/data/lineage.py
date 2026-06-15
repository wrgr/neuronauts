"""ChunkedGraph lineage access for the MICrONS minnie65_public dataset.

This module resolves the **real** mapping between materialization versions of
the Minnie65 graphene segmentation over plain HTTP (no ``caveclient``).  v117,
v1718, etc. are timestamped snapshots of one segmentation; mapping between them
is supervoxel lineage — resolve the same supervoxels to their root id *at a
chosen timestamp*.

This gives us genuine f(v117 → v1718) supervision: a "fragment" is a v117 root,
its ground-truth label is the v1718 (proofread) root that owns it, and the real
split/merge structure ("one trunk + slivers", frankenmerges) comes for free
rather than being synthesised.

All requests use the public CAVE bearer token, the same pattern as
``neuronauts/data/loaders.py``.  The endpoints used (all confirmed 200):

  GET  {cg}/segmentation/api/v1/table/{tbl}/node/{root}/leaves?stop_layer={1,2}
  POST {cg}/segmentation/api/v1/table/{tbl}/roots_binary?timestamp={unix}
  GET  {mat}/materialize/api/v2/datastack/{ds}/version/{v}     (version metadata)

See ``docs/seg_117_to_1412.md`` for the full access map.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests

from .loaders import DEFAULT_TOKEN

DATASTACK = "minnie65_public"
CG_SERVER = "https://minnie.microns-daf.com"
SEG_TABLE = "minnie65_public"
MAT_SERVER = "https://minnie.microns-daf.com"
SYNAPSE_TABLE = "synapses_pni_2"

# synapses_pni_2 positions are stored in 4×4×40 nm voxels.
SYNAPSE_VOXEL_NM = (4.0, 4.0, 40.0)

# v117 = 2021-06-11T08:10:00Z (the early/raw segmentation state).
V117_TIMESTAMP = 1623399000

_REQUEST_SLEEP = 0.25  # be gentle with the chunkedgraph
_ROOTS_BATCH = 4000    # max supervoxels per roots_binary POST

_version_ts_cache: dict[int, int] = {117: V117_TIMESTAMP}


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Version timestamps
# ---------------------------------------------------------------------------

def version_timestamp(version: int, token: str = DEFAULT_TOKEN) -> Optional[int]:
    """Return the UNIX timestamp (seconds) for a materialization version.

    Cached.  Returns ``None`` if the version metadata cannot be fetched.
    """
    if version in _version_ts_cache:
        return _version_ts_cache[version]
    url = f"{MAT_SERVER}/materialize/api/v2/datastack/{DATASTACK}/version/{version}"
    try:
        time.sleep(_REQUEST_SLEEP)
        resp = requests.get(url, headers=_headers(token), timeout=60)
        if resp.status_code != 200:
            return None
        meta = resp.json()
        ts_raw = meta.get("time_stamp")
        if ts_raw is None:
            return None
        # e.g. "2025-09-...Z" or with millis; normalise to UTC epoch seconds.
        s = ts_raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ts = int(dt.timestamp())
        _version_ts_cache[version] = ts
        return ts
    except Exception:
        return None


def list_versions(token: str = DEFAULT_TOKEN) -> list[int]:
    """Return the list of available public materialization versions."""
    url = f"{MAT_SERVER}/materialize/api/v2/datastack/{DATASTACK}/versions"
    try:
        time.sleep(_REQUEST_SLEEP)
        resp = requests.get(url, headers=_headers(token), timeout=60)
        if resp.status_code != 200:
            return []
        return sorted(int(v) for v in resp.json())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# ChunkedGraph: leaves and timestamped root resolution
# ---------------------------------------------------------------------------

def root_leaves(
    root_id: int,
    *,
    stop_layer: int = 1,
    token: str = DEFAULT_TOKEN,
) -> Optional[np.ndarray]:
    """Return the leaf ids of a root.

    ``stop_layer=1`` → supervoxels (finest, timestamp-invariant).
    ``stop_layer=2`` → L2 nodes (a coarser, cheaper proxy for size).
    Returns a uint64 ndarray, or ``None`` on failure.
    """
    url = f"{CG_SERVER}/segmentation/api/v1/table/{SEG_TABLE}/node/{root_id}/leaves"
    try:
        time.sleep(_REQUEST_SLEEP)
        resp = requests.get(url, headers=_headers(token),
                            params={"stop_layer": stop_layer}, timeout=120)
        if resp.status_code != 200:
            return None
        data = resp.json()
        leaves = data.get("leaf_ids", data) if isinstance(data, dict) else data
        return np.asarray(leaves, dtype=np.uint64)
    except Exception:
        return None


def roots_at(
    svids: np.ndarray,
    timestamp: Optional[int],
    *,
    token: str = DEFAULT_TOKEN,
) -> Optional[np.ndarray]:
    """Map supervoxel ids → root ids at a timestamp (``None`` = current).

    Batches large inputs into ``_ROOTS_BATCH``-sized binary POSTs.  Returns a
    uint64 ndarray aligned with ``svids``, or ``None`` on any failure.
    """
    svids = np.asarray(svids, dtype=np.uint64)
    url = f"{CG_SERVER}/segmentation/api/v1/table/{SEG_TABLE}/roots_binary"
    if timestamp is not None:
        url = f"{url}?timestamp={int(timestamp)}"
    out = np.empty(len(svids), dtype=np.uint64)
    hdr = {**_headers(token), "Content-Type": "application/octet-stream"}
    try:
        for start in range(0, len(svids), _ROOTS_BATCH):
            chunk = svids[start:start + _ROOTS_BATCH]
            time.sleep(_REQUEST_SLEEP)
            resp = requests.post(url, headers=hdr,
                                 data=chunk.astype("<u8").tobytes(), timeout=120)
            if resp.status_code != 200:
                return None
            roots = np.frombuffer(resp.content, dtype="<u8")
            if len(roots) != len(chunk):
                return None
            out[start:start + _ROOTS_BATCH] = roots
        return out
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lineage helpers
# ---------------------------------------------------------------------------

def root_at_version(
    seed_root: int,
    version: int,
    *,
    token: str = DEFAULT_TOKEN,
) -> Optional[int]:
    """Resolve ``seed_root`` (any version) to its root at ``version``.

    Picks one supervoxel of ``seed_root`` and maps it to the root at the target
    version's timestamp.  Useful to carry a v1412 nucleus root forward to the
    v1718 label space.
    """
    ts = version_timestamp(version, token=token)
    if ts is None:
        return None
    svs = root_leaves(seed_root, stop_layer=1, token=token)
    if svs is None or len(svs) == 0:
        return None
    mapped = roots_at(svs[:1], ts, token=token)
    if mapped is None or len(mapped) == 0:
        return None
    r = int(mapped[0])
    return r if r != 0 else None


def fragment_breakdown(
    target_root: int,
    *,
    from_timestamp: int = V117_TIMESTAMP,
    max_sv: int = 4000,
    token: str = DEFAULT_TOKEN,
    seed: int = 0,
) -> Optional[dict[int, int]]:
    """Break a proofread neuron into its real fragments at an earlier version.

    Maps ``target_root``'s supervoxels back to their roots at
    ``from_timestamp`` (default v117).  Each distinct earlier root is a *real
    fragment*; the count is its supervoxel mass (subsampled to ``max_sv``).

    Returns ``{from_root_id: sv_count}`` sorted descending by count, or ``None``.
    A neuron that was already one v117 root → a single-entry dict; the long tail
    of small entries is the real "trunk + slivers" split structure.
    """
    svs = root_leaves(target_root, stop_layer=1, token=token)
    if svs is None or len(svs) == 0:
        return None
    if len(svs) > max_sv:
        rng = np.random.default_rng(seed)
        svs = svs[rng.choice(len(svs), max_sv, replace=False)]
    past = roots_at(svs, from_timestamp, token=token)
    if past is None:
        return None
    past = past[past > 0]
    if len(past) == 0:
        return None
    uniq, counts = np.unique(past, return_counts=True)
    order = np.argsort(-counts)
    return {int(uniq[i]): int(counts[i]) for i in order}


# ---------------------------------------------------------------------------
# Real synapses (materialization v3 query → Arrow)
# ---------------------------------------------------------------------------

def _read_arrow(content: bytes):
    """Parse an Arrow IPC payload (file or stream framing) into a Table."""
    import io

    import pyarrow as pa

    buf = io.BytesIO(content)
    try:
        return pa.ipc.open_file(buf).read_all()
    except Exception:
        buf.seek(0)
        return pa.ipc.open_stream(buf).read_all()


def fetch_synapses(
    root_id: int,
    *,
    version: int = 1718,
    side: str = "post",
    limit: int = 2000,
    token: str = DEFAULT_TOKEN,
) -> Optional[dict]:
    """Fetch real synapses on a neuron from the materialization synapse table.

    Uses the **v3** query API (the v2 endpoint has a server-side serialization
    bug — ``ipc_compress``).  ``side="post"`` returns incoming synapses (the
    synapse sits on this neuron's dendrite); ``side="pre"`` returns outgoing
    synapses (on this neuron's axon).

    Parameters
    ----------
    root_id:
        Proofread root id at ``version``.
    version:
        Materialization version (must be AVAILABLE — e.g. 1718, 1621, 1507,
        1300; **not** the expired 1412).
    side:
        ``"post"`` or ``"pre"`` — which end of the synapse must lie on
        ``root_id``.

    Returns
    -------
    dict with keys (all aligned, length N):
        positions_nm   : [N, 3] float32 — the on-neuron synapse position in nm
        supervoxel_ids : [N] uint64 — supervoxel of the on-neuron point
        root_ids       : [N] uint64 — root_id at ``version`` (== root_id)
    or ``None`` on failure.
    """
    if side not in ("pre", "post"):
        raise ValueError("side must be 'pre' or 'post'")

    url = (f"{MAT_SERVER}/materialize/api/v3/datastack/{DATASTACK}"
           f"/version/{version}/table/{SYNAPSE_TABLE}/query")
    body = {
        "filter_equal_dict": {SYNAPSE_TABLE: {f"{side}_pt_root_id": int(root_id)}},
        "limit": int(limit),
    }
    try:
        time.sleep(_REQUEST_SLEEP)
        resp = requests.post(url, headers=_headers(token), json=body, timeout=120)
        if resp.status_code != 200:
            return None
        tbl = _read_arrow(resp.content)
    except Exception:
        return None

    cols = tbl.schema.names
    px, py, pz = (f"{side}_pt_position_{a}" for a in "xyz")
    if px not in cols or f"{side}_pt_supervoxel_id" not in cols:
        return None
    d = tbl.to_pydict()
    n = tbl.num_rows
    if n == 0:
        return {
            "positions_nm": np.zeros((0, 3), dtype=np.float32),
            "supervoxel_ids": np.zeros(0, dtype=np.uint64),
            "root_ids": np.zeros(0, dtype=np.uint64),
        }
    vx, vy, vz = SYNAPSE_VOXEL_NM
    pos = np.stack([
        np.asarray(d[px], dtype=np.float64) * vx,
        np.asarray(d[py], dtype=np.float64) * vy,
        np.asarray(d[pz], dtype=np.float64) * vz,
    ], axis=1).astype(np.float32)
    return {
        "positions_nm": pos,
        "supervoxel_ids": np.asarray(d[f"{side}_pt_supervoxel_id"], dtype=np.uint64),
        "root_ids": np.asarray(d[f"{side}_pt_root_id"], dtype=np.uint64),
    }


def fetch_region_synapses(
    bbox_nm: tuple,
    *,
    version: int = 1718,
    side: str = "pre",
    limit: int = 50_000,
    token: str = DEFAULT_TOKEN,
) -> Optional[dict]:
    """Fetch all synapses whose pre/post point falls within a 3-D bounding box.

    Unlike ``fetch_synapses`` (single root ID), this returns ALL neurons in the
    region. Useful for building region-based training graphs where spatially
    interleaved neurons provide cross-neuron edges — the training signal the
    edge classifier needs to learn to cut merge errors and recognize frankenmerges.

    Parameters
    ----------
    bbox_nm:
        ``((x0, y0, z0), (x1, y1, z1))`` in nm — the spatial bounding box.
        Coordinates are converted to 4×4×40 nm synapse-table voxels internally.
    version:
        Materialization version for root ID resolution.
    side:
        ``"pre"`` or ``"post"`` — which synapse end must lie in the bbox.
    limit:
        Maximum synapses returned (server-side cap).

    Returns
    -------
    dict with keys (all length-N, aligned):
        positions_nm   : [N, 3] float32 in nm
        supervoxel_ids : [N] uint64
        root_ids       : [N] uint64 — v{version} root id (ground-truth label)
    or ``None`` on failure.
    """
    if side not in ("pre", "post"):
        raise ValueError("side must be 'pre' or 'post'")

    (x0, y0, z0), (x1, y1, z1) = bbox_nm
    vx, vy, vz = SYNAPSE_VOXEL_NM
    lo = [x0 / vx, y0 / vy, z0 / vz]
    hi = [x1 / vx, y1 / vy, z1 / vz]

    url = (f"{MAT_SERVER}/materialize/api/v3/datastack/{DATASTACK}"
           f"/version/{version}/table/{SYNAPSE_TABLE}/query")
    body = {
        "filter_spatial_dict": {
            SYNAPSE_TABLE: {f"{side}_pt_position": [lo, hi]}
        },
        "limit": int(limit),
    }
    try:
        time.sleep(_REQUEST_SLEEP)
        resp = requests.post(url, headers=_headers(token), json=body, timeout=300)
        if resp.status_code != 200:
            return None
        tbl = _read_arrow(resp.content)
    except Exception:
        return None

    cols = tbl.schema.names
    px, py, pz = (f"{side}_pt_position_{a}" for a in "xyz")
    if px not in cols or f"{side}_pt_supervoxel_id" not in cols:
        return None
    d = tbl.to_pydict()
    n = tbl.num_rows
    if n == 0:
        return {
            "positions_nm": np.zeros((0, 3), dtype=np.float32),
            "supervoxel_ids": np.zeros(0, dtype=np.uint64),
            "root_ids": np.zeros(0, dtype=np.uint64),
            "other_root_ids": np.zeros(0, dtype=np.uint64),
        }
    pos = np.stack([
        np.asarray(d[px], dtype=np.float64) * vx,
        np.asarray(d[py], dtype=np.float64) * vy,
        np.asarray(d[pz], dtype=np.float64) * vz,
    ], axis=1).astype(np.float32)
    other_side = "post" if side == "pre" else "pre"
    other_key = f"{other_side}_pt_root_id"
    other_root_ids = (np.asarray(d[other_key], dtype=np.uint64)
                      if other_key in d else np.zeros(n, dtype=np.uint64))
    return {
        "positions_nm": pos,
        "supervoxel_ids": np.asarray(d[f"{side}_pt_supervoxel_id"], dtype=np.uint64),
        "root_ids": np.asarray(d[f"{side}_pt_root_id"], dtype=np.uint64),
        "other_root_ids": other_root_ids,
    }


# ---------------------------------------------------------------------------
# L2 cache: real skeleton from L2-node representative coordinates
# ---------------------------------------------------------------------------

L2_CACHE_SERVER = "https://minnie.microns-daf.com"
L2_TABLE = "minnie65_public"
_L2_BATCH = 500  # max L2 ids per attribute fetch


def l2_skeleton(
    root_id: int,
    *,
    token: str = DEFAULT_TOKEN,
    max_l2_nodes: int = 2000,
    seed: int = 0,
) -> Optional[dict]:
    """Fetch a real skeleton for a v117 fragment root via the L2 attribute cache.

    Retrieves ``rep_coord_nm`` (representative nm coordinates) for each L2 node
    in the root, then builds a skeleton using a minimum spanning tree on
    pairwise Euclidean distances.

    Returns
    -------
    dict with keys:
        vertices_nm : [V, 3] float32 — one vertex per L2 node
        edges       : [E, 2] int64  — MST edges (symmetric, undirected)
        radii_nm    : [V] float32   — proxy radius from L2 count (constant 200)
        l2_ids      : [V] uint64    — L2 node ids aligned with vertices
    or ``None`` on any failure (network error, too few nodes).
    """
    # 1. Get L2 nodes for this root
    l2ids = root_leaves(root_id, stop_layer=2, token=token)
    if l2ids is None or len(l2ids) < 2:
        return None

    rng = np.random.default_rng(seed)
    if len(l2ids) > max_l2_nodes:
        l2ids = l2ids[rng.choice(len(l2ids), max_l2_nodes, replace=False)]

    # 2. Fetch rep_coord_nm from L2 attribute cache in batches
    url = f"{L2_CACHE_SERVER}/l2cache/api/v1/table/{L2_TABLE}/attributes"
    hdr = {**_headers(token), "Content-Type": "application/json"}
    coords: dict[int, np.ndarray] = {}
    try:
        for start in range(0, len(l2ids), _L2_BATCH):
            chunk = l2ids[start:start + _L2_BATCH].tolist()
            body = {"l2_ids": chunk, "attribute_names": ["rep_coord_nm"]}
            time.sleep(_REQUEST_SLEEP)
            resp = requests.post(url, headers=hdr, json=body, timeout=120)
            if resp.status_code != 200:
                return None
            data = resp.json()
            for id_str, attrs in data.items():
                coord = attrs.get("rep_coord_nm")
                if coord is not None and len(coord) == 3:
                    coords[int(id_str)] = np.asarray(coord, dtype=np.float32)
    except Exception:
        return None

    if len(coords) < 2:
        return None

    # 3. Build skeleton: vertices = L2 centroids, edges = MST
    ids_out = np.array(list(coords.keys()), dtype=np.uint64)
    verts = np.stack([coords[int(i)] for i in ids_out], axis=0).astype(np.float32)
    n = len(verts)

    # MST via Kruskal on a proximity graph (k-NN to limit edge count)
    from scipy.spatial import cKDTree as _KDTree
    k = min(6, n - 1)
    tree = _KDTree(verts)
    dists, nbrs = tree.query(verts, k=k + 1)  # includes self at index 0

    # Build edge list with weights
    edge_set: dict[tuple[int, int], float] = {}
    for i in range(n):
        for slot in range(1, k + 1):
            j = int(nbrs[i, slot])
            key = (min(i, j), max(i, j))
            d = float(dists[i, slot])
            if key not in edge_set or edge_set[key] > d:
                edge_set[key] = d

    # Kruskal MST using union-find
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sorted_edges = sorted(edge_set.items(), key=lambda kv: kv[1])
    mst: list[tuple[int, int]] = []
    for (u, v), _ in sorted_edges:
        pu, pv = _find(u), _find(v)
        if pu != pv:
            parent[pu] = pv
            mst.append((u, v))
            if len(mst) == n - 1:
                break

    if not mst:
        return None

    edges = np.array(mst, dtype=np.int64)
    return {
        "vertices_nm": verts,
        "edges": edges,
        "radii_nm": np.full(n, 200.0, dtype=np.float32),
        "l2_ids": ids_out,
    }


__all__ = [
    "DATASTACK",
    "V117_TIMESTAMP",
    "SYNAPSE_TABLE",
    "SYNAPSE_VOXEL_NM",
    "L2_CACHE_SERVER",
    "L2_TABLE",
    "version_timestamp",
    "list_versions",
    "root_leaves",
    "roots_at",
    "root_at_version",
    "fragment_breakdown",
    "fetch_synapses",
    "fetch_region_synapses",
    "l2_skeleton",
]
