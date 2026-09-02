"""L2 geometry for v117 atoms: resumable, tiered, incremental.

For each v117 atom we need its skeleton and its real topology:

  ``lvl2_graph``  -> L2 adjacency edges (the true cut surface; EXP-053B and
                     EXP-056 had to approximate this with a kNN/MST over
                     synapse endpoints and both flagged it as a limitation)
  ``leaves``      -> the L2 node set, needed because an atom of one L2 node
                     has no edges and would otherwise vanish
  L2 attributes   -> rep_coord_nm plus size_nm3, area_nm2, max_dt_nm,
                     mean_dt_nm and pca, pooled across atoms. The dt fields are
                     a distance transform, i.e. caliber, which the
                     caliber-continuity idea needs and which is free here.

Everything is keyed on the atom id, never on the region, so regions compose and
the same cache serves a later scale-up to all somata.

Tiering: run ``>=10`` synapses first, then ``>=5``, then ``>=1``. Each tier
skips atoms already done, so widening costs only the new atoms.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import requests

from neuronauts.data import lineage as L

L2_ATTRS = ["rep_coord_nm", "size_nm3", "area_nm2", "max_dt_nm", "mean_dt_nm"]

#: The l2cache attribute endpoint enforces 600 requests per minute and reports
#: breaches as ``500`` wrapping ``429 Too Many Requests: 600 per 1 minute``.
#: The first tier-10 run drove it at ~748/min with 24 workers and lost exactly
#: 1,318 whole batches (2,636,000 L2 nodes, 14.2%) because the failures were
#: swallowed. Stay under the limit and count what fails.
L2_ATTR_RATE_PER_MIN = 480


class RateLimiter:
    """Token bucket shared across threads, in requests per minute."""

    def __init__(self, per_minute: int):
        self.interval = 60.0 / max(per_minute, 1)
        self._lock = threading.Lock()
        self._next = time.monotonic()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        delay = slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


def _is_rate_limited(resp) -> bool:
    """CAVE reports a throttle as 429, or as 500 with 429 in the body."""
    if resp.status_code == 429:
        return True
    if resp.status_code == 500:
        return "429" in resp.text[:400] and "Too Many Requests" in resp.text[:400]
    return False


# ---------------------------------------------------------------------------
# per-atom topology, sharded + resumable
# ---------------------------------------------------------------------------

class AtomGeometryStore:
    """CSR-style shards of (atom -> L2 nodes, L2 adjacency).

    One file per batch rather than per atom: at 279k atoms a file each would be
    unmanageable, and a single file would forfeit resumability.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_dir = self.root / "shards"
        self.shard_dir.mkdir(exist_ok=True)

    def done_atoms(self) -> set[int]:
        out: set[int] = set()
        for f in sorted(self.shard_dir.glob("*.npz")):
            try:
                with np.load(f, allow_pickle=False) as z:
                    out.update(z["atom_id"].tolist())
            except Exception:
                continue
        return out

    def next_index(self, tag: str) -> int:
        """First unused shard index for ``tag``.

        Shard names must never collide across runs. They used to be
        ``{tag}_{i}`` with ``i`` counted from zero within the *current* todo
        list, so a rerun that had fewer atoms to fetch silently overwrote the
        shards of the first run: a rerun for 350 outstanding atoms replaced a
        2,000-atom shard and destroyed 1,650 atoms' geometry.
        """
        used = {int(f.stem.rsplit("_", 1)[1])
                for f in self.shard_dir.glob(f"{tag}_*.npz")
                if f.stem.rsplit("_", 1)[-1].isdigit()}
        return max(used) + 1 if used else 0

    def write_shard(self, tag: str, records: list[dict]) -> Path:
        atom_id, node_ptr, nodes, edge_ptr, edges = [], [0], [], [0], []
        for r in records:
            atom_id.append(r["atom"])
            nodes.append(r["l2_ids"])
            node_ptr.append(node_ptr[-1] + len(r["l2_ids"]))
            edges.append(r["edges"])
            edge_ptr.append(edge_ptr[-1] + len(r["edges"]))
        path = self.shard_dir / f"{tag}.npz"
        np.savez_compressed(
            path,
            atom_id=np.asarray(atom_id, np.uint64),
            node_ptr=np.asarray(node_ptr, np.int64),
            l2_ids=(np.concatenate(nodes) if nodes else np.zeros(0, np.uint64)),
            edge_ptr=np.asarray(edge_ptr, np.int64),
            edges=(np.concatenate(edges) if edges else np.zeros((0, 2), np.uint64)),
        )
        return path

    def load_all(self) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for f in sorted(self.shard_dir.glob("*.npz")):
            with np.load(f, allow_pickle=False) as z:
                aid, npt, ept = z["atom_id"], z["node_ptr"], z["edge_ptr"]
                l2, ed = z["l2_ids"], z["edges"]
                for i, a in enumerate(aid.tolist()):
                    out[a] = {"l2_ids": l2[npt[i]:npt[i + 1]],
                              "edges": ed[ept[i]:ept[i + 1]]}
        return out


def _fetch_one(atom: int, bstr: str, token: str) -> dict:
    """L2 node set + adjacency for one v117 atom."""
    hdr = L._headers(token)
    base = f"{L.CG_SERVER}/segmentation/api/v1/table/{L.SEG_TABLE}/node/{int(atom)}"
    rec = {"atom": int(atom), "l2_ids": np.zeros(0, np.uint64),
           "edges": np.zeros((0, 2), np.uint64), "error": None}

    edges = np.zeros((0, 2), np.uint64)
    for attempt in range(4):
        try:
            r = requests.get(f"{base}/lvl2_graph", headers=hdr,
                             params={"bounds": bstr}, timeout=120)
            if r.status_code == 200:
                e = r.json().get("edge_graph", [])
                edges = (np.asarray(e, dtype=np.uint64).reshape(-1, 2)
                         if len(e) else np.zeros((0, 2), np.uint64))
                rec["error"] = None      # a retry that succeeds is not an error
                break
            if _is_rate_limited(r):
                rec["error"] = "rate_limited"
                time.sleep(min(30.0, 5.0 * (attempt + 1)))
                continue
            if r.status_code in (400, 404):
                rec["error"] = f"lvl2_graph {r.status_code}"
                break
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}"
        time.sleep(0.5 * (attempt + 1))

    nodes = np.unique(edges) if edges.size else np.zeros(0, np.uint64)
    # Single-L2-node atoms have no edges; recover them via leaves.
    if len(nodes) == 0:
        for attempt in range(4):
            try:
                r = requests.get(f"{base}/leaves", headers=hdr, timeout=120,
                                 params={"stop_layer": 2, "bounds": bstr})
                if r.status_code == 200:
                    nodes = np.asarray(r.json().get("leaf_ids", []), np.uint64)
                    rec["error"] = None
                    break
                if _is_rate_limited(r):
                    rec["error"] = "rate_limited"
                    time.sleep(min(30.0, 5.0 * (attempt + 1)))
                    continue
                if r.status_code in (400, 404):
                    rec["error"] = f"leaves {r.status_code}"
                    break
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}"
            time.sleep(0.5 * (attempt + 1))

    rec["l2_ids"] = nodes.astype(np.uint64)
    rec["edges"] = edges.astype(np.uint64)
    return rec


def fetch_atom_topology(atoms: Iterable[int], seg_bounds, store: AtomGeometryStore,
                        *, token: Optional[str] = None, workers: int = 8,
                        batch: int = 2000, tag: str = "t",
                        verbose: bool = True) -> dict:
    """Fetch topology for atoms not already in the store."""
    token = token or L.DEFAULT_TOKEN
    bstr = "_".join(f"{int(seg_bounds[i][0])}-{int(seg_bounds[i][1])}"
                    for i in range(3))

    have = store.done_atoms()
    todo = [int(a) for a in atoms if int(a) not in have]
    if verbose:
        print(f"  {len(have):,} atoms already cached; {len(todo):,} to fetch",
              flush=True)
    if not todo:
        return {"fetched": 0, "errors": 0, "cached": len(have)}

    t0 = time.time()
    n_done = 0
    retryable: list[int] = []
    absent: list[int] = []
    shard_i = store.next_index(tag)          # never reuse an existing name
    for bi in range(0, len(todo), batch):
        chunk = todo[bi:bi + batch]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            recs = list(ex.map(lambda a: _fetch_one(a, bstr, token), chunk))

        # A record whose fetch failed for a transient reason must NOT be
        # written: the shard is the resume marker, so persisting an empty
        # record would retire the atom as permanently node-less.
        keep = []
        for r in recs:
            err = r["error"]
            if err is None or err.startswith(("lvl2_graph 4", "leaves 4")):
                if err is not None:
                    absent.append(r["atom"])
                keep.append(r)
            else:
                retryable.append(r["atom"])
        if keep:
            store.write_shard(f"{tag}_{shard_i:05d}", keep)
            shard_i += 1
        n_done += len(keep)
        if verbose:
            el = time.time() - t0
            rate = (bi + len(chunk)) / max(el, 1e-9)
            eta = (len(todo) - bi - len(chunk)) / max(rate, 1e-9) / 60
            print(f"    {bi+len(chunk):,}/{len(todo):,} atoms  {rate:.0f}/s  "
                  f"eta {eta:.1f}m  written {n_done:,}  "
                  f"retryable {len(retryable)}  absent {len(absent)}", flush=True)
    if retryable and verbose:
        print(f"  !! {len(retryable)} atoms failed transiently and were NOT "
              f"written; rerun to pick them up", flush=True)
    return {"fetched": n_done, "errors": len(retryable), "absent": len(absent),
            "retryable_atoms": retryable, "cached": len(have)}


# ---------------------------------------------------------------------------
# pooled L2 attributes (coords + caliber)
# ---------------------------------------------------------------------------

def fetch_l2_attributes(l2_ids: np.ndarray, cache_path: Path, *,
                        token: Optional[str] = None, batch: int = 2000,
                        workers: int = 8, verbose: bool = True,
                        rate_per_min: int = L2_ATTR_RATE_PER_MIN,
                        attempts: int = 5) -> dict:
    """Pooled rep_coord_nm + caliber attributes, incremental on disk.

    Rate limited and fail-loud: a dropped batch costs 2,000 L2 nodes and used to
    disappear into a bare ``return {}``.
    """
    token = token or L.DEFAULT_TOKEN
    cache_path = Path(cache_path)
    have_id = np.zeros(0, np.uint64)
    cols: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            have_id = z["l2_id"]
            cols = {k: z[k] for k in z.files if k != "l2_id"}

    missing = np.setdiff1d(np.unique(l2_ids), have_id, assume_unique=False)
    if verbose:
        print(f"  attributes: {len(have_id):,} cached, {len(missing):,} to fetch",
              flush=True)
    if len(missing) == 0:
        return {"l2_id": have_id, **cols}

    url = f"{L.L2_CACHE_SERVER}/l2cache/api/v1/table/{L.L2_TABLE}/attributes"
    hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    chunks = [missing[i:i + batch].tolist() for i in range(0, len(missing), batch)]

    limiter = RateLimiter(rate_per_min)
    failed: list[tuple[list[int], str]] = []
    fail_lock = threading.Lock()

    def go(chunk):
        why = "unknown"
        for attempt in range(attempts):
            limiter.acquire()
            try:
                r = requests.post(url, headers=hdr, timeout=180,
                                  json={"l2_ids": chunk,
                                        "attribute_names": L2_ATTRS})
                if r.status_code == 200:
                    return r.json()
                if _is_rate_limited(r):
                    why = "rate_limited"
                    # Back off past the limiter: the quota is per minute, so a
                    # breach means every worker must idle, not just retry.
                    time.sleep(min(60.0, 5.0 * (attempt + 1)))
                    continue
                why = f"http_{r.status_code}"
            except Exception as exc:  # noqa: BLE001
                why = type(exc).__name__
            time.sleep(1.0 + attempt)
        with fail_lock:
            failed.append((chunk, why))
        return {}

    got: dict[int, dict] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, d in enumerate(ex.map(go, chunks), 1):
            got.update({int(k): v for k, v in d.items()})
            if verbose and i % 25 == 0:
                el = time.time() - t0
                print(f"    {i}/{len(chunks)} batches ({el:.0f}s, "
                      f"{60*i/max(el,1e-9):.0f} req/min, "
                      f"{len(failed)} failed)", flush=True)

    if failed:
        n_lost = sum(len(c) for c, _ in failed)
        reasons: dict[str, int] = {}
        for _, w in failed:
            reasons[w] = reasons.get(w, 0) + 1
        print(f"  !! {len(failed)} batches FAILED ({n_lost:,} L2 nodes "
              f"unfetched): {reasons}", flush=True)

    new_ids = np.asarray(sorted(got.keys()), np.uint64)
    pos = np.full((len(new_ids), 3), np.nan, np.float32)
    scal = {k: np.full(len(new_ids), np.nan, np.float32)
            for k in ("size_nm3", "area_nm2", "max_dt_nm", "mean_dt_nm")}
    for i, k in enumerate(new_ids.tolist()):
        a = got[k]
        c = a.get("rep_coord_nm")
        if c is not None and len(c) == 3:
            pos[i] = c
        for s in scal:
            v = a.get(s)
            if v is not None:
                scal[s][i] = v

    out_id = np.concatenate([have_id, new_ids])
    out = {"l2_id": out_id,
           "pos_nm": np.concatenate([cols.get("pos_nm", np.zeros((0, 3), np.float32)), pos])}
    for s in scal:
        out[s] = np.concatenate([cols.get(s, np.zeros(0, np.float32)), scal[s]])

    order = np.argsort(out["l2_id"])
    out = {k: v[order] for k, v in out.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **out)
    return out
