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


__all__ = [
    "DATASTACK",
    "V117_TIMESTAMP",
    "SYNAPSE_TABLE",
    "version_timestamp",
    "list_versions",
    "root_leaves",
    "roots_at",
    "root_at_version",
    "fragment_breakdown",
    "fetch_synapses",
]
