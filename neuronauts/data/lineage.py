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

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from .loaders import DEFAULT_TOKEN

DATASTACK = "minnie65_public"
CG_SERVER = "https://minnie.microns-daf.com"
SEG_TABLE = "minnie65_public"
MAT_SERVER = "https://minnie.microns-daf.com"
SYNAPSE_TABLE = "synapses_pni_2"

# ---------------------------------------------------------------------------
# Synapse-fetch cache
# ---------------------------------------------------------------------------
# The materialization query applies a server-side ``limit`` with no stable sort
# order, so a bbox holding more than ``limit`` synapses returns a DIFFERENT
# arbitrary subset on each call.  That makes any multi-run comparison (e.g.
# adding a training region, sweeping a hyperparameter) invalid, because the
# training/eval data silently changes underneath the experiment.
#
# To make fetches reproducible we cache each result to disk keyed on the query
# (datastack, table, version, side, bbox, limit).  Set the cache directory via
# NEURONAUTS_SYNAPSE_CACHE_DIR; caching is enabled by default.  Set
# NEURONAUTS_SYNAPSE_CACHE_DIR="" (or "0"/"off") to disable.

def _synapse_cache_dir() -> Optional[Path]:
    val = os.environ.get("NEURONAUTS_SYNAPSE_CACHE_DIR", "_DEFAULT_")
    if val in ("", "0", "off", "OFF", "none", "None"):
        return None
    if val == "_DEFAULT_":
        val = "/tmp/neuronauts_synapse_cache"
    p = Path(val)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return p


# ---------------------------------------------------------------------------
# Cache provenance
# ---------------------------------------------------------------------------
# Caches are git-lfs-tracked and shared across runs/machines, so each entry must
# be self-describing: which CAVE datastack/table, which materialization version,
# which algorithm + params, and which code produced it.  We embed a JSON
# provenance blob INTO each .npz under a reserved key (stripped on load so
# callers never see it) and also write a human-readable manifest per cache dir.

_PROV_KEY = "__provenance__"
_CODE_VERSION: Optional[str] = None


def _code_version() -> str:
    """Short git commit of the code that produced a cache entry (or 'unknown')."""
    global _CODE_VERSION
    if _CODE_VERSION is None:
        try:
            _CODE_VERSION = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL, timeout=5).decode().strip() or "unknown"
        except Exception:
            _CODE_VERSION = "unknown"
    return _CODE_VERSION


def _embed_prov(result: dict, prov: dict) -> dict:
    """Return a copy of ``result`` with a JSON provenance blob attached."""
    out = dict(result)
    full = {"code_version": _code_version(),
            "fetched_at": datetime.now(timezone.utc).isoformat(), **prov}
    out[_PROV_KEY] = np.array(json.dumps(full, sort_keys=True))
    return out


def _strip_prov(loaded: dict) -> dict:
    """Drop the provenance blob so callers get only the data arrays."""
    return {k: v for k, v in loaded.items() if k != _PROV_KEY}


def read_cache_provenance(npz_path) -> Optional[dict]:
    """Read the provenance dict embedded in a cache .npz, or None if absent.

    Lets you audit a git-lfs-tracked cache file: which CAVE version, table,
    algorithm, and code commit produced it.
    """
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if _PROV_KEY in z.files:
                return json.loads(str(z[_PROV_KEY]))
    except Exception:
        return None
    return None


def _write_cache_manifest(cdir: Path, manifest: dict) -> None:
    """Write a human-readable PROVENANCE.json into a cache dir (idempotent)."""
    f = cdir / "PROVENANCE.json"
    if f.exists():
        return
    try:
        manifest = {"code_version": _code_version(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    **manifest}
        f.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    except Exception:
        pass


def _synapse_cache_key(bbox_nm, *, version, side, limit) -> str:
    (x0, y0, z0), (x1, y1, z1) = bbox_nm
    # Round to 1 nm to avoid float-repr drift across callers.
    parts = (DATASTACK, SYNAPSE_TABLE, int(version), side, int(limit),
             round(x0), round(y0), round(z0), round(x1), round(y1), round(z1))
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _synapse_cache_load(key: str) -> Optional[dict]:
    cdir = _synapse_cache_dir()
    if cdir is None:
        return None
    f = cdir / f"{key}.npz"
    if not f.exists():
        return None
    try:
        with np.load(f, allow_pickle=False) as z:
            return _strip_prov({k: z[k] for k in z.files})
    except Exception:
        return None


def _synapse_cache_save(key: str, result: dict, *, prov: Optional[dict] = None) -> None:
    cdir = _synapse_cache_dir()
    if cdir is None:
        return
    _write_cache_manifest(cdir, {
        "cache_kind": "synapse_fetch",
        "datastack": DATASTACK,
        "synapse_table": SYNAPSE_TABLE,
        "synapse_voxel_nm": list(SYNAPSE_VOXEL_NM),
        "v117_timestamp": V117_TIMESTAMP,
        "sort": "stable by CAVE synapse id; lexsort (z,y,x) fallback when ids absent",
        "key_fields": ["datastack", "synapse_table", "version", "side", "limit", "bbox_nm"],
        "schema": "<sha1>.npz arrays: positions_nm, supervoxel_ids, root_ids, "
                  "other_*; per-file provenance under '__provenance__'.",
    })
    f = cdir / f"{key}.npz"
    try:
        payload = _embed_prov(result, prov) if prov is not None else result
        # Atomic-ish: write to a tmp .npz then rename. (np.savez appends ".npz"
        # if the path lacks it, so the tmp name must already end in ".npz".)
        tmp = cdir / f"{key}.tmp{os.getpid()}.npz"
        np.savez(tmp, **payload)
        os.replace(tmp, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# L2-skeleton cache
# ---------------------------------------------------------------------------
# ``l2_skeleton`` makes 2+ network roundtrips per fragment (root_leaves + L2
# attribute batches) with a throttle sleep, so building a region with 1500+
# fragments takes hours.  The result is a pure function of the v117 root_id
# (a globally unique, immutable CAVE id) plus max_l2_nodes/seed, so it caches
# cleanly to disk: pay the fetch once, reuse forever across train/eval runs.
# Set the dir via NEURONAUTS_L2_CACHE_DIR (default /tmp/neuronauts_l2_cache);
# set "" / "0" / "off" to disable.  Only successful skeletons are cached, so a
# transient network failure retries on the next call instead of poisoning the
# cache with a permanent None.

def _l2_cache_dir() -> Optional[Path]:
    val = os.environ.get("NEURONAUTS_L2_CACHE_DIR", "_DEFAULT_")
    if val in ("", "0", "off", "OFF", "none", "None"):
        return None
    if val == "_DEFAULT_":
        val = "/tmp/neuronauts_l2_cache"
    p = Path(val)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return p


def _l2_cache_key(root_id: int, *, max_l2_nodes: int, seed: int) -> str:
    raw = f"{DATASTACK}|{L2_TABLE}|{int(root_id)}|{int(max_l2_nodes)}|{int(seed)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def _l2_cache_load(key: str) -> Optional[dict]:
    cdir = _l2_cache_dir()
    if cdir is None:
        return None
    f = cdir / f"{key}.npz"
    if not f.exists():
        return None
    try:
        with np.load(f, allow_pickle=False) as z:
            return _strip_prov({k: z[k] for k in z.files})
    except Exception:
        return None


def _l2_cache_save(key: str, result: dict, *, prov: Optional[dict] = None) -> None:
    cdir = _l2_cache_dir()
    if cdir is None:
        return
    _write_cache_manifest(cdir, {
        "cache_kind": "l2_skeleton",
        "datastack": DATASTACK,
        "l2_table": L2_TABLE,
        "algorithm": "rep_coord_nm per L2 node → kNN(k=6) proximity graph → "
                     "Kruskal MST; radius proxy=200nm constant",
        "note": "L2 skeletons reflect the live root segmentation at fetch time; "
                "keyed on the immutable v117 fragment root_id.",
        "key_fields": ["datastack", "l2_table", "root_id", "max_l2_nodes", "seed"],
        "schema": "<sha1>.npz arrays: vertices_nm, edges, radii_nm, l2_ids; "
                  "per-file provenance under '__provenance__'.",
    })
    f = cdir / f"{key}.npz"
    try:
        payload = _embed_prov(result, prov) if prov is not None else result
        tmp = cdir / f"{key}.tmp{os.getpid()}.npz"
        np.savez(tmp, **payload)
        os.replace(tmp, f)
    except Exception:
        pass


# synapses_pni_2 positions are stored in 4×4×40 nm voxels.
SYNAPSE_VOXEL_NM = (4.0, 4.0, 40.0)

# v117 = 2021-06-11T08:10:00Z (the early/raw segmentation state).
V117_TIMESTAMP = 1623399000

_REQUEST_SLEEP = 0.25  # be gentle with the chunkedgraph
_ROOTS_BATCH = 4000    # max supervoxels per roots_binary POST

_version_ts_cache: dict[int, int] = {117: V117_TIMESTAMP}


def _headers(token: Optional[str]) -> dict:
    if not token:
        from neuronauts.auth import cave_token
        token = cave_token(required=True)
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
    bounds: Optional[str] = None,
    token: str = DEFAULT_TOKEN,
) -> Optional[np.ndarray]:
    """Return the leaf ids of a root.

    ``stop_layer=1`` → supervoxels (finest, timestamp-invariant).
    ``stop_layer=2`` → L2 nodes (a coarser, cheaper proxy for size).
    ``bounds``, e.g. ``"lo_x-hi_x_lo_y-hi_y_lo_z-hi_z"`` in segmentation
    coordinates (see ``harness.substrate.region_bounds``), restricts the
    enumeration to a region. Without it this fetches *every* leaf of the
    root, which is fine for a small object but not for one that has already
    absorbed a large amount of tissue -- a real root from a merge-tree walk
    returned 689,734 supervoxels unbounded. Pass ``bounds`` whenever the
    caller only needs to know about one region, not the whole object.
    Returns a uint64 ndarray, or ``None`` on failure.
    """
    url = f"{CG_SERVER}/segmentation/api/v1/table/{SEG_TABLE}/node/{root_id}/leaves"
    params = {"stop_layer": stop_layer}
    if bounds is not None:
        params["bounds"] = bounds
    try:
        time.sleep(_REQUEST_SLEEP)
        resp = requests.get(url, headers=_headers(token),
                            params=params, timeout=120)
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
        other_root_ids : [N] uint64 — v{version} root id at the OTHER synapse endpoint
        other_positions_nm   : [N, 3] float32 — OTHER endpoint position (nm)
        other_supervoxel_ids : [N] uint64 — OTHER endpoint supervoxel id
        synapse_ids    : [N] int64  — CAVE synapse-table id (-1 if column absent).
                          Shared across pre/post fetches, so it joins a synapse's two
                          observations.
    or ``None`` on failure.
    """
    if side not in ("pre", "post"):
        raise ValueError("side must be 'pre' or 'post'")

    # Reproducibility: serve from the on-disk cache when available so repeated
    # runs (different training configs, hyperparameter sweeps) see IDENTICAL
    # data. The server-side ``limit`` has no stable order, so without this an
    # over-limit bbox returns a different subset every call.
    _ck = _synapse_cache_key(bbox_nm, version=version, side=side, limit=limit)
    _cached = _synapse_cache_load(_ck)
    if _cached is not None:
        return _cached

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
            "other_positions_nm": np.zeros((0, 3), dtype=np.float32),
            "other_supervoxel_ids": np.zeros(0, dtype=np.uint64),
            "synapse_ids": np.zeros(0, dtype=np.int64),
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
    # Real CAVE synapse-table id — shared across pre/post fetches; -1 if absent.
    synapse_ids = (np.asarray(d["id"], dtype=np.int64)
                   if "id" in d else np.full(n, -1, dtype=np.int64))
    # The OTHER endpoint's position + supervoxel, from the SAME rows. This lets a
    # single fetch yield both sides of the same synapses (a guaranteed join), which
    # two independent spatial fetches cannot provide once either is subsampled.
    ox, oy, oz = (f"{other_side}_pt_position_{a}" for a in "xyz")
    if ox in d and f"{other_side}_pt_supervoxel_id" in d:
        other_positions_nm = np.stack([
            np.asarray(d[ox], dtype=np.float64) * vx,
            np.asarray(d[oy], dtype=np.float64) * vy,
            np.asarray(d[oz], dtype=np.float64) * vz,
        ], axis=1).astype(np.float32)
        other_supervoxel_ids = np.asarray(d[f"{other_side}_pt_supervoxel_id"],
                                          dtype=np.uint64)
    else:
        other_positions_nm = np.zeros((n, 3), dtype=np.float32)
        other_supervoxel_ids = np.zeros(n, dtype=np.uint64)
    result = {
        "positions_nm": pos,
        "supervoxel_ids": np.asarray(d[f"{side}_pt_supervoxel_id"], dtype=np.uint64),
        "root_ids": np.asarray(d[f"{side}_pt_root_id"], dtype=np.uint64),
        "other_root_ids": other_root_ids,
        "other_positions_nm": other_positions_nm,
        "other_supervoxel_ids": other_supervoxel_ids,
        "synapse_ids": synapse_ids,
    }

    # Canonical, deterministic row order: sort by real CAVE synapse id when
    # present (falls back to a stable position sort if ids are absent). This
    # ensures the cached array — and any downstream seeded subsample — is
    # reproducible regardless of the order the server happened to return.
    if np.any(synapse_ids >= 0):
        order = np.argsort(synapse_ids, kind="stable")
    else:
        order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
    result = {k: v[order] for k, v in result.items()}

    _synapse_cache_save(_ck, result, prov={
        "cache_kind": "synapse_fetch",
        "datastack": DATASTACK,
        "synapse_table": SYNAPSE_TABLE,
        "materialization_version": int(version),
        "side": side,
        "limit": int(limit),
        "bbox_nm": [list(bbox_nm[0]), list(bbox_nm[1])],
    })
    return result


def fetch_region_synapses_tiled(
    bbox_nm: tuple,
    *,
    version: int = 1718,
    side: str = "pre",
    tile_x_nm: float = 40_000,
    per_tile_limit: int = 200_000,
    token: str = DEFAULT_TOKEN,
) -> Optional[dict]:
    """Fetch all synapses in a bbox by splitting into x-axis tiles.

    CAVE's spatial query API caps at ~250k rows per request. Tiling bypasses
    this by fetching one 40 µm x-tile at a time (~100k synapses/tile), each
    safely under the cap. Every tile is individually cached via
    ``fetch_region_synapses``; the combined result is not (a second call with
    the same arguments yields the same data via tile-level cache hits).

    Parameters
    ----------
    tile_x_nm:
        Width of each x-tile in nm (default 40,000 = 40 µm; ~5 tiles per
        200 µm training region).
    per_tile_limit:
        Max synapses per tile request (default 200,000 — 2× headroom over the
        ~100k synapses expected per 40 µm tile at MICrONS density).

    Returns
    -------
    Same dict format as ``fetch_region_synapses``, or ``None`` if all tiles fail.
    """
    (x0, y0, z0), (x1, y1, z1) = bbox_nm

    # Build half-open x-tile boundaries [cur_x, next_x)
    tile_bboxes = []
    cur_x = float(x0)
    while cur_x < float(x1):
        next_x = min(cur_x + tile_x_nm, float(x1))
        tile_bboxes.append(((cur_x, y0, z0), (next_x, y1, z1)))
        cur_x = next_x

    parts: dict[str, list] = {
        "positions_nm": [], "supervoxel_ids": [], "root_ids": [],
        "other_root_ids": [], "other_positions_nm": [],
        "other_supervoxel_ids": [], "synapse_ids": [],
    }
    n_tiles_ok = 0
    for tile_bbox in tile_bboxes:
        tile = fetch_region_synapses(
            tile_bbox, version=version, side=side,
            limit=per_tile_limit, token=token)
        if tile is None:
            continue
        n_tiles_ok += 1
        for key in parts:
            parts[key].append(tile[key])

    if n_tiles_ok == 0:
        return None

    result: dict[str, np.ndarray] = {}
    for key, arrays in parts.items():
        if arrays:
            result[key] = np.concatenate(arrays, axis=0)
        elif key.endswith("_nm"):
            result[key] = np.zeros((0, 3), dtype=np.float32)
        elif key == "synapse_ids":
            result[key] = np.zeros(0, dtype=np.int64)
        else:
            result[key] = np.zeros(0, dtype=np.uint64)

    # Deduplicate synapses that straddle a tile boundary (CAVE inclusive bbox).
    # Synapse ids are globally unique, so a duplicate can only appear once.
    syn_ids = result["synapse_ids"]
    if np.any(syn_ids >= 0):
        # np.unique returns the FIRST occurrence of each id — preserves tile order
        _, first_idx = np.unique(syn_ids, return_index=True)
        result = {k: v[first_idx] for k, v in result.items()}
        syn_ids = result["synapse_ids"]

    # Final canonical sort (same as single-tile fetch)
    if np.any(syn_ids >= 0):
        order = np.argsort(syn_ids, kind="stable")
    else:
        pos = result["positions_nm"]
        order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
    return {k: v[order] for k, v in result.items()}


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
    # Serve from the on-disk cache when available (skeletons are immutable for
    # a given v117 root_id, so this is the same data every time).
    _ck = _l2_cache_key(root_id, max_l2_nodes=max_l2_nodes, seed=seed)
    _cached = _l2_cache_load(_ck)
    if _cached is not None:
        return _cached

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
    result = {
        "vertices_nm": verts,
        "edges": edges,
        "radii_nm": np.full(n, 200.0, dtype=np.float32),
        "l2_ids": ids_out,
    }
    _l2_cache_save(_ck, result, prov={
        "cache_kind": "l2_skeleton",
        "datastack": DATASTACK,
        "l2_table": L2_TABLE,
        "root_id": int(root_id),
        "max_l2_nodes": int(max_l2_nodes),
        "seed": int(seed),
        "n_vertices": int(n),
    })
    return result


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
    "read_cache_provenance",
]
