"""Probe the MICrONS minnie65_public CAVE / PyChunkedGraph HTTP API *without*
caveclient (which will not install in this container).

Everything here is plain ``requests`` against the public CAVE servers, using the
same bearer token pattern as ``neuronauts/data/loaders.py``.

Goal: determine which HTTP endpoints we can reach for mapping the base
segmentation (v117) to the proofread label version (v1412), and what the version
numbers actually denote (materialization versions -> timestamps).

Run:
    python scripts/probe_seg_mapping.py

It is deliberately gentle: small samples, short sleeps, no concurrency.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Constants (mirrors neuronauts/data/loaders.py + fetch.py)
# ---------------------------------------------------------------------------
from neuronauts.data.auth import cave_token  # token must come from the environment
TOKEN = cave_token()
DATASTACK = "minnie65_public"
GLOBAL_SERVER = "https://global.daf-apis.com"

HEADERS = {"Authorization": f"Bearer {TOKEN}"}
SLEEP = 0.3  # seconds between requests, be gentle

# Filled in at runtime from the datastack info response.
CG_SERVER: Optional[str] = None
SEG_TABLE: Optional[str] = None


def _get(url: str, **kw) -> requests.Response:
    time.sleep(SLEEP)
    return requests.get(url, headers=HEADERS, timeout=60, **kw)


def _post(url: str, **kw) -> requests.Response:
    time.sleep(SLEEP)
    # Merge any caller-supplied headers with the auth header (caller wins on
    # collisions) so passing headers=... in kw does not double-bind the arg.
    merged = {**HEADERS, **kw.pop("headers", {})}
    return requests.post(url, headers=merged, timeout=60, **kw)


def _status(label: str, resp: requests.Response) -> None:
    body_preview = ""
    try:
        body_preview = resp.text[:200].replace("\n", " ")
    except Exception:
        pass
    print(f"[{resp.status_code}] {label}\n       {resp.request.method} {resp.url}")
    if resp.status_code != 200:
        print(f"       body: {body_preview}")


# ---------------------------------------------------------------------------
# 1. Datastack info -> discover cg_server_address + segmentation table id
# ---------------------------------------------------------------------------
def probe_datastack_info() -> Optional[dict]:
    global CG_SERVER, SEG_TABLE
    url = f"{GLOBAL_SERVER}/info/api/v2/datastack/full/{DATASTACK}"
    resp = _get(url)
    _status("datastack info", resp)
    if resp.status_code != 200:
        return None
    info = resp.json()
    # cg url looks like graphene://https://minnie.microns-daf.com/segmentation/table/minnie65_public
    seg_source = info.get("segmentation_source", "") or info.get("flat_segmentation_source", "")
    print(f"       segmentation_source: {seg_source}")
    # Parse graphene://https://HOST/segmentation/table/TABLE
    cg = seg_source.replace("graphene://", "")
    if "/segmentation/table/" in cg:
        host, table = cg.split("/segmentation/table/")
        CG_SERVER = host
        SEG_TABLE = table.strip("/")
    print(f"       => CG_SERVER={CG_SERVER}  SEG_TABLE={SEG_TABLE}")
    # Also report materialization endpoint + viewer_resolution if present
    for k in ("local_server", "aligned_volume", "viewer_resolution_x"):
        if k in info:
            print(f"       info[{k}]={info[k]!r}")
    return info


# ---------------------------------------------------------------------------
# 2. Materialization versions -> timestamps (what is v117? v1412?)
# ---------------------------------------------------------------------------
def probe_materialization_versions() -> None:
    # NOTE: the materialization service lives on the datastack's *local_server*
    # (minnie.microns-daf.com), NOT global.daf-apis.com. The global host 404s.
    base = CG_SERVER or "https://minnie.microns-daf.com"
    for apiver in ("v2", "v3"):
        url = f"{base}/materialize/api/{apiver}/datastack/{DATASTACK}/versions"
        resp = _get(url)
        _status(f"materialization {apiver} versions list", resp)
        if resp.status_code == 200:
            try:
                versions = resp.json()
                print(f"       available versions: {versions}")
            except Exception:
                pass
            # Pull metadata (timestamp) for the two we care about
            for v in (117, 1412):
                murl = f"{base}/materialize/api/{apiver}/datastack/{DATASTACK}/version/{v}"
                mr = _get(murl)
                _status(f"version {v} metadata ({apiver})", mr)
                if mr.status_code == 200:
                    try:
                        meta = mr.json()
                        print(f"       v{v} time_stamp={meta.get('time_stamp')!r} "
                              f"valid={meta.get('valid')} is_merged={meta.get('is_merged')}")
                    except Exception:
                        pass
            return  # first working api version is enough


# ---------------------------------------------------------------------------
# 3. ChunkedGraph: root -> leaves (L2) ; supervoxel -> root@timestamp
# ---------------------------------------------------------------------------
def probe_leaves(root_id: int) -> Optional[list[int]]:
    """Get the L2 node ids of a v1412 root (stop_layer=2)."""
    if CG_SERVER is None or SEG_TABLE is None:
        print("       (skip leaves: no CG_SERVER/SEG_TABLE)")
        return None
    url = f"{CG_SERVER}/segmentation/api/v1/table/{SEG_TABLE}/node/{root_id}/leaves"
    resp = _get(url, params={"stop_layer": 2})
    _status(f"leaves(stop_layer=2) root={root_id}", resp)
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
        leaves = data.get("leaf_ids", data) if isinstance(data, dict) else data
        print(f"       n_L2_nodes={len(leaves)}")
        return list(leaves)
    except Exception as e:
        print(f"       parse error: {e}")
        return None


def probe_root_of_supervoxel(svid: int, timestamp: Optional[str] = None) -> Optional[int]:
    """Map a supervoxel id -> root id, optionally at a past UNIX timestamp."""
    if CG_SERVER is None or SEG_TABLE is None:
        print("       (skip root: no CG_SERVER/SEG_TABLE)")
        return None
    url = f"{CG_SERVER}/segmentation/api/v1/table/{SEG_TABLE}/node/{svid}/root"
    params: dict[str, Any] = {}
    if timestamp is not None:
        params["timestamp"] = timestamp
    resp = _get(url, params=params)
    label = f"root(svid={svid}" + (f", ts={timestamp})" if timestamp else ")")
    _status(label, resp)
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
        root = data.get("root_id", data) if isinstance(data, dict) else data
        print(f"       root_id={root}")
        return int(root) if not isinstance(root, list) else root
    except Exception as e:
        print(f"       parse error: {e}")
        return None


def get_roots_at(svids: list[int], timestamp: Optional[str] = None):
    """Batch map supervoxel ids -> root ids via the binary roots endpoint.

    This is the workhorse for merge/split analysis: map a v1412 neuron's
    supervoxels back to their root ids at the v117 timestamp.

    Body is a little-endian uint64 array of supervoxel ids; the response is a
    little-endian uint64 array of root ids (same length / order).
    """
    import numpy as np
    if CG_SERVER is None or SEG_TABLE is None:
        return None
    url = f"{CG_SERVER}/segmentation/api/v1/table/{SEG_TABLE}/roots_binary"
    if timestamp is not None:
        url += f"?timestamp={timestamp}"
    body = np.array(svids, dtype="<u8").tobytes()
    resp = _post(url, data=body,
                 headers={**HEADERS, "Content-Type": "application/octet-stream"})
    if resp.status_code != 200:
        _status(f"roots_binary(n={len(svids)}, ts={timestamp})", resp)
        return None
    return np.frombuffer(resp.content, dtype="<u8")


def probe_split_structure(root_id: int, v117_ts: str = "1623399000",
                          max_sv: int = 4000) -> None:
    """For one v1412 neuron, count how many distinct v117 roots its
    supervoxels trace back to. >1 distinct v117 root == proofreader split-fix
    (a false-merge in v117 that was separated)... actually the reverse: many
    v117 roots stitched into one v1412 neuron == split-error fix.
    """
    import numpy as np
    from collections import Counter
    if CG_SERVER is None:
        return
    r = _get(f"{CG_SERVER}/segmentation/api/v1/table/{SEG_TABLE}/node/{root_id}/leaves",
             params={"stop_layer": 1})
    if r.status_code != 200:
        _status(f"leaves(sv) root={root_id}", r)
        return
    svs = np.array(r.json()["leaf_ids"], dtype="<u8")
    if len(svs) > max_sv:
        svs = np.random.RandomState(0).choice(svs, max_sv, replace=False)
    past = get_roots_at(svs.tolist(), v117_ts)
    if past is None:
        return
    past = past[past > 0]
    c = Counter(past.tolist())
    top = c.most_common(5)
    shares = [f"{v / len(past):.2f}" for _, v in top]
    print(f"   v1412 {root_id}: {len(c)} distinct v117 roots; "
          f"top-5 mass share = {shares}")


# ---------------------------------------------------------------------------
# Helper: a couple of v1412 root ids from the public nucleus table
# ---------------------------------------------------------------------------
def sample_v1412_roots(n: int = 5) -> list[int]:
    """Grab a few v1412 root ids straight from the public nucleus CSV.

    Avoids importing the package; just downloads & parses inline.
    """
    import gzip
    import io
    url = ("https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/"
           "v1412/nucleus_detection_v0_merged.csv.gz")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    roots: list[int] = []
    with gzip.open(io.BytesIO(resp.content)) as fh:
        for line in fh:
            parts = line.decode().strip().split(",")
            if len(parts) < 4:
                continue
            try:
                rid = int(parts[3])
            except ValueError:
                continue
            if rid != 0:
                roots.append(rid)
            if len(roots) >= n * 50:
                break
    # deterministic spread-out sample
    step = max(1, len(roots) // n)
    return roots[::step][:n]


def main() -> None:
    print("=" * 70)
    print("PROBE: minnie65_public CAVE/PCG HTTP API (no caveclient)")
    print("=" * 70)

    print("\n--- 1. datastack info ---")
    info = probe_datastack_info()

    print("\n--- 2. materialization versions / timestamps ---")
    probe_materialization_versions()

    print("\n--- sampling v1412 roots from nucleus table ---")
    try:
        roots = sample_v1412_roots(5)
        print(f"       sample roots: {roots}")
    except Exception as e:
        print(f"       failed to sample: {e}")
        roots = []

    print("\n--- 3a. root -> L2 leaves (component count per v1412 neuron) ---")
    leaf_counts = {}
    for rid in roots[:5]:
        leaves = probe_leaves(rid)
        if leaves is not None:
            leaf_counts[rid] = len(leaves)

    print("\n--- 3b. supervoxel -> root, current vs v117 timestamp ---")
    # Need a supervoxel id. The leaves(stop_layer=0) would give supervoxels, but
    # is large; instead try leaves with no stop_layer on first root to grab one SV.
    if roots and CG_SERVER:
        first = roots[0]
        url = f"{CG_SERVER}/segmentation/api/v1/table/{SEG_TABLE}/node/{first}/leaves"
        r = _get(url, params={"stop_layer": 1})
        _status(f"leaves(stop_layer=1 -> supervoxels) root={first}", r)
        sv = None
        if r.status_code == 200:
            try:
                d = r.json()
                sv_list = d.get("leaf_ids", d) if isinstance(d, dict) else d
                if len(sv_list):
                    sv = int(sv_list[0])
                    print(f"       picked supervoxel {sv}")
            except Exception as e:
                print(f"       parse error: {e}")
        if sv is not None:
            # current root
            probe_root_of_supervoxel(sv)
            # v117 timestamp: 2021-06-11 ~ unix 1623369600
            probe_root_of_supervoxel(sv, timestamp="1623369600")

    print("\n--- 3c. v117 split structure (svs -> v117 roots) ---")
    # v117 timestamp = 2021-06-11T08:10:00 UTC == unix 1623399000
    for rid in roots[:5]:
        try:
            probe_split_structure(rid)
        except Exception as e:
            print(f"   {rid}: split-structure failed: {e}")

    print("\n--- SUMMARY ---")
    if leaf_counts:
        vals = list(leaf_counts.values())
        print(f"L2 nodes per v1412 neuron (n={len(vals)}): "
              f"min={min(vals)} max={max(vals)} mean={sum(vals)/len(vals):.0f}")
        for rid, c in leaf_counts.items():
            print(f"   {rid}: {c} L2 nodes")


if __name__ == "__main__":
    main()
