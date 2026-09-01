"""Timing probe: how fast can we retrieve L2 geometry for many v117 roots?

EXP-053B concluded the complete-root L2 route "is not a practical 1,023-root
dense substrate" after a 10-root probe ran >14 min. That probe used the
per-root path in ``neuronauts.data.lineage.l2_skeleton``, which does
1 root_leaves call + ceil(n_l2/500) attribute calls per root, each preceded by
a 0.25 s sleep, strictly serially.

This script measures three routes on the SAME roots so the comparison is fair:

  A. current  -- lineage.l2_skeleton(), serial, as used by EXP-053B
  B. caveclient -- l2cache.get_l2data(root_ids=[...]) per root
  C. pooled   -- threaded root_leaves, then pool every L2 id across all roots
                 into large batched attribute POSTs with no sleep

Route C is the hypothesis: the attributes endpoint is keyed on l2_ids, not on
root, so per-root batching wastes almost all of the request budget.

Roots come from the offline v117 box cache (no network needed to pick them).
Nothing is written except the shared L2 cache.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.data import lineage  # noqa: E402


def pick_roots(box_dir: str, n_roots: int, min_syn: int) -> tuple[list[int], dict]:
    """Pick the n_roots busiest v117 roots from one cached box. Offline."""
    npz_files = sorted(glob.glob(os.path.join(box_dir, "*.npz")))
    if not npz_files:
        raise SystemExit(f"no npz in {box_dir}")
    path = npz_files[0]
    z = np.load(path)
    counts: collections.Counter = collections.Counter()
    for key in ("pre_root_id", "post_root_id"):
        ids, cts = np.unique(z[key], return_counts=True)
        for i, c in zip(ids.tolist(), cts.tolist()):
            if i:
                counts[i] += c
    eligible = [r for r, c in counts.most_common() if c >= min_syn]
    meta = {
        "box": os.path.basename(path),
        "n_synapses": int(len(z["pre_root_id"])),
        "n_roots_total": len(counts),
        "n_roots_eligible": len(eligible),
    }
    return eligible[:n_roots], meta


def route_a_current(roots: list[int], token: str) -> dict:
    """Serial lineage.l2_skeleton, exactly as EXP-053B used it."""
    t0 = time.time()
    ok = 0
    verts = 0
    for r in roots:
        sk = lineage.l2_skeleton(r, token=token)
        if sk is not None:
            ok += 1
            verts += len(sk["vertices_nm"])
    dt = time.time() - t0
    return {"route": "A_current_serial", "seconds": dt, "n_roots": len(roots),
            "ok": ok, "vertices": verts, "sec_per_root": dt / max(len(roots), 1)}


def route_b_caveclient(roots: list[int]) -> dict:
    """caveclient l2cache.get_l2data, one call per root."""
    from caveclient import CAVEclient

    client = CAVEclient("minnie65_public")
    t0 = time.time()
    ok = 0
    verts = 0
    for r in roots:
        try:
            data = client.l2cache.get_l2data(int(r), attributes=["rep_coord_nm"])
        except Exception as exc:  # noqa: BLE001
            return {"route": "B_caveclient_per_root", "error": f"{type(exc).__name__}: {exc}"}
        n = sum(1 for v in data.values() if v.get("rep_coord_nm"))
        if n >= 2:
            ok += 1
        verts += n
    dt = time.time() - t0
    return {"route": "B_caveclient_per_root", "seconds": dt, "n_roots": len(roots),
            "ok": ok, "vertices": verts, "sec_per_root": dt / max(len(roots), 1)}


def route_c_pooled(roots: list[int], token: str, *, leaf_workers: int,
                   attr_workers: int, attr_batch: int) -> dict:
    """Threaded root_leaves, then pooled batched attribute POSTs (no sleep)."""
    t0 = time.time()

    # 1. L2 ids per root, in parallel.
    def leaves(r: int):
        try:
            return r, lineage.root_leaves(int(r), stop_layer=2, token=token)
        except Exception:
            return r, None

    per_root: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=leaf_workers) as ex:
        for r, ids in ex.map(leaves, roots):
            if ids is not None and len(ids) >= 2:
                per_root[r] = ids
    t_leaves = time.time() - t0

    # 2. Pool every L2 id across all roots, dedup, fetch in large batches.
    pool = np.unique(np.concatenate([v for v in per_root.values()])) if per_root else np.array([], dtype=np.uint64)
    url = f"{lineage.L2_CACHE_SERVER}/l2cache/api/v1/table/{lineage.L2_TABLE}/attributes"
    hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    batches = [pool[i:i + attr_batch].tolist() for i in range(0, len(pool), attr_batch)]

    t1 = time.time()
    got = 0
    errors: list[str] = []

    def fetch(chunk):
        body = {"l2_ids": chunk, "attribute_names": ["rep_coord_nm"]}
        try:
            resp = requests.post(url, headers=hdr, json=body, timeout=180)
            if resp.status_code != 200:
                return f"HTTP {resp.status_code}", 0
            return None, len(resp.json())
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}", 0

    with ThreadPoolExecutor(max_workers=attr_workers) as ex:
        for err, n in ex.map(fetch, batches):
            if err:
                errors.append(err)
            got += n
    t_attrs = time.time() - t1
    dt = time.time() - t0

    return {
        "route": "C_pooled_threaded",
        "seconds": dt,
        "n_roots": len(roots),
        "ok": len(per_root),
        "vertices": got,
        "sec_per_root": dt / max(len(roots), 1),
        "pooled_l2_ids": int(len(pool)),
        "n_attr_requests": len(batches),
        "t_root_leaves": t_leaves,
        "t_attributes": t_attrs,
        "errors": errors[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-dir", default="data/boxes_v117")
    ap.add_argument("--n-roots", type=int, default=12)
    ap.add_argument("--min-syn", type=int, default=10)
    ap.add_argument("--leaf-workers", type=int, default=16)
    ap.add_argument("--attr-workers", type=int, default=8)
    ap.add_argument("--attr-batch", type=int, default=2000)
    ap.add_argument("--routes", default="A,B,C")
    ap.add_argument("--out", default="results/probe_l2_throughput.json")
    args = ap.parse_args()

    # Keep the probe out of the shared cache so route A measures real fetches.
    os.environ.setdefault("NEURONAUTS_L2_CACHE_DIR", "/tmp/neuronauts_l2_probe")

    token = lineage.DEFAULT_TOKEN
    if not token:
        raise SystemExit("no CAVE token available")

    roots, meta = pick_roots(args.box_dir, args.n_roots, args.min_syn)
    print(f"box={meta['box']} synapses={meta['n_synapses']} "
          f"roots_total={meta['n_roots_total']} eligible={meta['n_roots_eligible']}")
    print(f"probing {len(roots)} roots\n")

    want = {r.strip().upper() for r in args.routes.split(",")}
    results = []
    if "A" in want:
        print("route A: current serial l2_skeleton ...", flush=True)
        results.append(route_a_current(roots, token))
        print(f"  {results[-1]}\n", flush=True)
    if "B" in want:
        print("route B: caveclient get_l2data per root ...", flush=True)
        results.append(route_b_caveclient(roots))
        print(f"  {results[-1]}\n", flush=True)
    if "C" in want:
        print("route C: pooled threaded ...", flush=True)
        results.append(route_c_pooled(roots, token, leaf_workers=args.leaf_workers,
                                      attr_workers=args.attr_workers,
                                      attr_batch=args.attr_batch))
        print(f"  {results[-1]}\n", flush=True)

    print("=" * 70)
    print(f"{'route':<26}{'sec':>9}{'sec/root':>11}{'ok':>6}{'verts':>10}")
    for r in results:
        if "error" in r:
            print(f"{r['route']:<26}  ERROR {r['error']}")
            continue
        print(f"{r['route']:<26}{r['seconds']:>9.1f}{r['sec_per_root']:>11.2f}"
              f"{r['ok']:>6}{r['vertices']:>10}")

    # Extrapolate to the EXP-053B population.
    print("\nextrapolation to 1,023 roots:")
    for r in results:
        if "error" in r:
            continue
        est = r["sec_per_root"] * 1023
        print(f"  {r['route']:<26}{est / 60:>8.1f} min")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"meta": meta, "roots": [int(r) for r in roots], "results": results,
         "config": vars(args)}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
