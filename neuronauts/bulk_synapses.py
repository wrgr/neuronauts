"""GET-based bulk synapse fetch for MICrONS minnie65.

Bypasses the CAVE materialize ``/query`` POST endpoint entirely. That endpoint
(the one ``synapse_query``/``query_table`` route through) can be down server-side
— it accepts the connection then never sends a response body — while GETs to the
same host stay healthy. This module reconstructs the same data from two healthy,
GET-reachable sources:

1. **Positions + supervoxel IDs + synapse_id** — the public Delta Lake (parquet)
   export of ``synapses_pni_2`` on GCS, in its Morton/Z-ordered-by-center variant::

       gs://mat_dbs/public/deltalake_exports/minnie65_phase3_v1/v1507/synapses_pni_2/ctr_pt_position_morton/

   It is spatially clustered (Z-order on the center point), and the Delta
   ``_delta_log`` stores per-file ctr_pt min/max stats. We replay the log,
   prune to the parquet files whose ctr_pt bbox overlaps the query box, download
   only those, and filter rows exactly. A 20 µm box typically touches ~3 of ~84
   files. Supervoxel IDs and ``id`` are time-invariant, so the export version
   (v1507) does not matter for those fields.

2. **Root IDs at the requested materialization version** — the chunkedgraph
   ``roots_binary`` batch endpoint, a *different, healthy* service from the dead
   materialize query backend. Querying it at the version's materialization
   timestamp yields the true root IDs for that version (verified to match
   ``caveclient.chunkedgraph.get_roots`` at the same timestamp, and to differ
   from "latest" for neurons edited after the version — i.e. genuinely
   time-resolved, not an echo of latest).

Dependencies: ``requests``, ``numpy``, ``pyarrow`` (and ``caveclient`` only to
read the version timestamp). ``REQUESTS_CA_BUNDLE`` must point at the environment
CA bundle (set automatically here); raw ``urllib`` does not work through the
proxy — use ``requests``.
"""

from __future__ import annotations

import datetime
import io
import json
import logging
import struct
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
GCS_BUCKET = "mat_dbs"
# Spatially (Morton-on-ctr) ordered Delta Lake export of synapses_pni_2. v1507 is
# the only spatially-organised synapse export published; positions/supervoxels in
# it are time-invariant so this is fine as the geometry source for any version.
DELTA_TABLE_PREFIX = (
    "public/deltalake_exports/minnie65_phase3_v1/v1507/"
    "synapses_pni_2/ctr_pt_position_morton"
)

CG_ROOTS_BINARY_URL = (
    "https://minnie.microns-daf.com/segmentation/api/v1/"
    "table/minnie65_public/roots_binary"
)

# minnie65 synapse/EM base voxel size in nm (positions in the export are in voxels).
VOXEL_NM = np.array([4.0, 4.0, 40.0])

# Fallback v117 materialization timestamp, used only if the live version metadata
# cannot be read. Verified equal (to the microsecond) to the official value from
# GET /materialize/api/v3/datastack/minnie65_public/version/117.
_V117_TIMESTAMP_UTC = datetime.datetime(
    2021, 6, 11, 8, 10, 0, 215114, tzinfo=datetime.timezone.utc
)


# --------------------------------------------------------------------------- #
# Version timestamp (drives time-resolved root lookup)
# --------------------------------------------------------------------------- #
def _version_timestamp(version: int, token: Optional[str]) -> datetime.datetime:
    """Materialization timestamp for ``version`` via the healthy metadata GET."""
    try:
        from caveclient import CAVEclient

        c = CAVEclient("minnie65_public", auth_token=token)
        ts = c.materialize.get_version_metadata(int(version))["time_stamp"]
        if isinstance(ts, datetime.datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=datetime.timezone.utc)
        return datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception as exc:  # metadata GET should be healthy; fall back for v117
        if int(version) == 117:
            log.warning("version metadata fetch failed (%s); using known v117 timestamp", exc)
            return _V117_TIMESTAMP_UTC
        raise


# --------------------------------------------------------------------------- #
# Low-level GCS helpers (JSON API + ?alt=media object download)
# --------------------------------------------------------------------------- #
def _gcs_media_url(object_path: str) -> str:
    enc = urllib.parse.quote(object_path, safe="")
    return f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o/{enc}?alt=media"


def _gcs_list_url(prefix: str, max_results: int = 1000) -> str:
    enc = urllib.parse.quote(prefix, safe="")
    return (
        f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o"
        f"?prefix={enc}&maxResults={max_results}"
    )


def _get_bytes(session, object_path: str, timeout: float = 120.0, tries: int = 3) -> bytes:
    last: Optional[Exception] = None
    for _ in range(tries):
        try:
            r = session.get(_gcs_media_url(object_path), timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0)
    assert last is not None
    raise last


def _list_objects(session, prefix: str, timeout: float = 30.0) -> list[str]:
    """List all object names under ``prefix`` (handles pagination)."""
    names: list[str] = []
    page_token = None
    while True:
        url = _gcs_list_url(prefix)
        if page_token:
            url += f"&pageToken={urllib.parse.quote(page_token, safe='')}"
        d = session.get(url, timeout=timeout).json()
        names.extend(i["name"] for i in d.get("items", []))
        page_token = d.get("nextPageToken")
        if not page_token:
            return names


# --------------------------------------------------------------------------- #
# Delta Lake transaction-log replay + spatial pruning
# --------------------------------------------------------------------------- #
def _live_files_with_stats(session) -> dict[str, Optional[str]]:
    """Replay the Delta ``_delta_log`` to get {relative_parquet_path: stats_json}.

    No checkpoint exists for this table, so the live state is the union of all
    ``add`` actions minus all ``remove`` actions across commits, in order.
    """
    log_prefix = f"{DELTA_TABLE_PREFIX}/_delta_log/"
    commits = sorted(n for n in _list_objects(session, log_prefix) if n.endswith(".json"))

    bodies: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        for name, body in zip(commits, ex.map(lambda c: _get_bytes(session, c, 30), commits)):
            bodies[name] = body

    live: dict[str, Optional[str]] = {}
    for c in commits:
        for line in bodies[c].decode().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "add" in obj:
                live[obj["add"]["path"]] = obj["add"].get("stats")
            elif "remove" in obj:
                live.pop(obj["remove"]["path"], None)
    return live


def _file_overlaps_bbox(stats_json: Optional[str], lo_vox: np.ndarray, hi_vox: np.ndarray) -> bool:
    """True if a parquet file's ctr_pt stats overlap the bbox. Missing stats -> read it."""
    if not stats_json:
        return True
    s = json.loads(stats_json)
    mn, mx = s.get("minValues", {}), s.get("maxValues", {})
    for ax, key in enumerate(("ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z")):
        if key in mn and key in mx:
            if mx[key] < lo_vox[ax] or mn[key] > hi_vox[ax]:
                return False
    return True


# --------------------------------------------------------------------------- #
# WKB point decode (pre_pt_position / post_pt_position are WKB Point Z)
# --------------------------------------------------------------------------- #
def _decode_wkb_points(blobs: Iterable[bytes]) -> np.ndarray:
    """Decode WKB 'Point Z' blobs to an (N,3) float64 array (voxel units)."""
    out = []
    for b in blobs:
        little = b[0] == 1  # byte0 endian; uint32 geomtype; then 3 doubles at offset 5
        fmt = "<ddd" if little else ">ddd"
        x, y, z = struct.unpack_from(fmt, b, 5)
        out.append((x, y, z))
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, 3), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Root-id recovery at a version timestamp via the (healthy) chunkedgraph
# --------------------------------------------------------------------------- #
def _supervoxels_to_roots(
    session,
    supervoxel_ids: np.ndarray,
    token: str,
    timestamp: datetime.datetime,
    batch: int = 100_000,
    timeout: float = 60.0,
) -> np.ndarray:
    """Map supervoxel ids -> root ids at ``timestamp`` via chunkedgraph roots_binary.

    Supervoxel id 0 (no segmentation) maps to root 0. Returns int64 aligned with input.
    """
    sv = np.asarray(supervoxel_ids, dtype=np.uint64)
    out = np.zeros(len(sv), dtype=np.uint64)
    nz = np.flatnonzero(sv != 0)
    params = {"timestamp": timestamp.timestamp()}
    headers = {"Authorization": f"Bearer {token}"}
    for start in range(0, len(nz), batch):
        idx = nz[start:start + batch]
        r = session.post(CG_ROOTS_BINARY_URL, data=sv[idx].tobytes(),
                         params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        out[idx] = np.frombuffer(r.content, dtype=np.uint64)
    return out.astype(np.int64)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def fetch_synapses_bulk(
    bbox_nm: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    token: str,
    *,
    version: int = 117,
    use_version_roots: bool = True,
    session=None,
    max_download_workers: int = 8,
) -> dict:
    """Fetch synapses whose center point falls within ``bbox_nm`` over GET only.

    Parameters
    ----------
    bbox_nm : ((x0,y0,z0),(x1,y1,z1)) inclusive bounding box in GLOBAL nanometers.
    token : CAVE auth token (used for the chunkedgraph root lookup + metadata).
    version : materialization version whose root IDs to reconstruct (default 117).
    use_version_roots : if True (default), remap supervoxel ids to root ids at the
        ``version`` timestamp (true version roots). If False, use the root ids baked
        into the v1507 parquet export (faster, but v1507 roots).

    Returns
    -------
    dict of numpy arrays:
        pre_pt_nm, post_pt_nm, ctr_pt_nm : (N,3) float64 GLOBAL nanometers
        pre_root_id, post_root_id, synapse_id : (N,) int64
        pre_supervoxel_id, post_supervoxel_id : (N,) int64
        size : (N,) float64
        root_id_version : str
    """
    import requests

    own_session = session is None
    session = session or requests.Session()
    try:
        (x0, y0, z0), (x1, y1, z1) = bbox_nm
        lo_nm = np.array([min(x0, x1), min(y0, y1), min(z0, z1)], dtype=np.float64)
        hi_nm = np.array([max(x0, x1), max(y0, y1), max(z0, z1)], dtype=np.float64)
        lo_vox = np.floor(lo_nm / VOXEL_NM).astype(np.int64)
        hi_vox = np.ceil(hi_nm / VOXEL_NM).astype(np.int64)

        # 1) replay log + spatial prune
        live = _live_files_with_stats(session)
        candidates = [p for p, s in live.items() if _file_overlaps_bbox(s, lo_vox, hi_vox)]
        log.info("bulk synapses: %d/%d parquet files overlap the box", len(candidates), len(live))

        # 2) download candidate parquet files (in parallel) and filter rows exactly
        import pyarrow.parquet as pq

        def _load(rel_path: str):
            raw = _get_bytes(session, f"{DELTA_TABLE_PREFIX}/{rel_path}", timeout=180)
            if raw[:4] != b"PAR1":
                return None
            return pq.read_table(
                io.BytesIO(raw),
                columns=[
                    "id", "valid", "pre_pt_position", "post_pt_position",
                    "ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z",
                    "size", "pre_pt_supervoxel_id", "post_pt_supervoxel_id",
                    "pre_pt_root_id", "post_pt_root_id",
                ],
            ).to_pandas()

        frames = []
        with ThreadPoolExecutor(max_workers=max_download_workers) as ex:
            for df in ex.map(_load, candidates):
                if df is None or len(df) == 0:
                    continue
                m = (
                    (df.ctr_pt_position_x >= lo_vox[0]) & (df.ctr_pt_position_x <= hi_vox[0])
                    & (df.ctr_pt_position_y >= lo_vox[1]) & (df.ctr_pt_position_y <= hi_vox[1])
                    & (df.ctr_pt_position_z >= lo_vox[2]) & (df.ctr_pt_position_z <= hi_vox[2])
                )
                sub = df[m]
                if len(sub):
                    frames.append(sub)

        if not frames:
            empty3 = np.zeros((0, 3), dtype=np.float64)
            empty1 = np.zeros((0,), dtype=np.int64)
            return {
                "pre_pt_nm": empty3, "post_pt_nm": empty3, "ctr_pt_nm": empty3,
                "pre_root_id": empty1, "post_root_id": empty1, "synapse_id": empty1,
                "pre_supervoxel_id": empty1, "post_supervoxel_id": empty1,
                "size": np.zeros((0,), dtype=np.float64),
                "root_id_version": f"v{version}" if use_version_roots else "v1507",
            }

        import pandas as pd

        out = pd.concat(frames, ignore_index=True)
        if "valid" in out.columns:  # match CAVE filtered-view semantics
            out = out[out["valid"] == True].reset_index(drop=True)  # noqa: E712

        pre_vox = _decode_wkb_points(out["pre_pt_position"].values)
        post_vox = _decode_wkb_points(out["post_pt_position"].values)
        ctr_vox = out[["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]].to_numpy(np.float64)
        pre_sv = out["pre_pt_supervoxel_id"].to_numpy(dtype=np.int64)
        post_sv = out["post_pt_supervoxel_id"].to_numpy(dtype=np.int64)

        # 3) root ids
        if use_version_roots:
            ts = _version_timestamp(version, token)
            pre_root = _supervoxels_to_roots(session, pre_sv, token, ts)
            post_root = _supervoxels_to_roots(session, post_sv, token, ts)
            ver = f"v{version}"
        else:
            pre_root = out["pre_pt_root_id"].to_numpy(dtype=np.int64)
            post_root = out["post_pt_root_id"].to_numpy(dtype=np.int64)
            ver = "v1507"

        return {
            "pre_pt_nm": pre_vox * VOXEL_NM,
            "post_pt_nm": post_vox * VOXEL_NM,
            "ctr_pt_nm": ctr_vox * VOXEL_NM,
            "pre_root_id": pre_root,
            "post_root_id": post_root,
            "synapse_id": out["id"].to_numpy(dtype=np.int64),
            "pre_supervoxel_id": pre_sv,
            "post_supervoxel_id": post_sv,
            "size": out["size"].to_numpy(dtype=np.float64),
            "root_id_version": ver,
        }
    finally:
        if own_session:
            session.close()
