"""Fetch and cache CAVE skeletons for all root IDs in cached training boxes.

Reads each box from BoxCache, collects unique pre/post root IDs, and fetches
the skeleton at the box's root_id_version.

WARNING — empirical caveat (2026-04-28)
---------------------------------------
The MICrONS ``minnie65_public`` skeleton service only returns real
skeletons for **proofread** roots.  In our 6 µm training boxes the
vast majority of root IDs are unproofread fragments, for which CAVE
returns a 1-vertex / 0-edge placeholder after ~80 s of latency.  In
a 102-skeleton fetch run, exactly 1 result had >5 vertices.

Use ``scripts/train.py precompute-self-skeletons`` (kimimaro
skeletonization of the BossDB seg volume) instead — it works for both
proofread and unproofread roots.  Keep this script for the case where
you specifically need proofread CAVE skeletons (e.g. for ground-truth
comparison or when re-using existing CAVE-format caches).

Usage:
    python scripts/fetch_skeletons.py --cache-dir data/boxes --output-dir data/skeletons
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", default="data/boxes")
    p.add_argument("--output-dir", default="data/skeletons")
    p.add_argument("--token", default=None, help="CAVE auth token")
    p.add_argument("--skeleton-service-version", type=int, default=4)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from neuronauts.dataset_builder import BoxCache
    from neuronauts.fetch import fetch_root_skeleton

    cache = BoxCache(args.cache_dir)
    records = list(cache.iter_records())
    if not records:
        print(f"No cached boxes in {args.cache_dir}")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect (root_id, version) pairs across all boxes
    # Only keep roots that have >=2 synapses on the same side — singletons
    # can never participate in same-root edges, so we don't need their skeletons.
    root_versions: dict[int, set[int]] = {}
    box_root_map: dict[str, dict[str, list[int]]] = {}

    for rec in records:
        meta_path = Path(args.cache_dir) / f"{rec.box_hash}.json"
        with open(meta_path) as f:
            meta = json.load(f)
        version = int(meta.get("root_id_version", 1412))

        try:
            _, syn = cache.load(rec)
        except Exception:
            continue

        from collections import Counter
        pre_counts = Counter(int(r) for r in syn.pre_root_id.tolist() if int(r) > 0)
        post_counts = Counter(int(r) for r in syn.post_root_id.tolist() if int(r) > 0)
        pre_roots = sorted(r for r, c in pre_counts.items() if c >= 2)
        post_roots = sorted(r for r, c in post_counts.items() if c >= 2)
        box_root_map[rec.box_hash] = {
            "version": version,
            "pre_roots": pre_roots,
            "post_roots": post_roots,
        }
        for r in pre_roots + post_roots:
            root_versions.setdefault(r, set()).add(version)

    n_unique_pairs = sum(len(vs) for vs in root_versions.values())
    print(f"{len(records)} boxes  {len(root_versions)} unique roots  {n_unique_pairs} (root, version) pairs")

    # Fetch each (root_id, version) into per-version subdirectory
    versions_seen: set[int] = set()
    for vs in root_versions.values():
        versions_seen.update(vs)
    for v in versions_seen:
        (out_dir / f"v{v}").mkdir(exist_ok=True)

    # Use a CAVEclient per version so we don't pay setup cost per fetch
    try:
        from caveclient import CAVEclient
    except ImportError:
        print("pip install caveclient")
        return 1

    clients: dict[int, "CAVEclient"] = {}
    from neuronauts.fetch import MICRONS_DATASTACK, CAVE_SERVER, _install_system_trust_store
    _install_system_trust_store()

    for v in versions_seen:
        cl = CAVEclient(MICRONS_DATASTACK, server_address=CAVE_SERVER, auth_token=args.token)
        cl.version = v
        clients[v] = cl

    fetched = 0
    cached_hits = 0
    failed = 0
    t0 = time.time()
    items = sorted(root_versions.items())

    for i, (root_id, vset) in enumerate(items, 1):
        for version in vset:
            cache_path = out_dir / f"v{version}" / f"v{version}_rid{root_id}_skv{args.skeleton_service_version}.npz"
            if cache_path.exists():
                cached_hits += 1
                continue

            backoff = 1.0
            success = False
            for attempt in range(8):
                try:
                    fetch_root_skeleton(
                        root_id,
                        version=version,
                        token=args.token,
                        skeleton_service_version=args.skeleton_service_version,
                        cache_dir=out_dir / f"v{version}",
                        client=clients[version],
                        max_retries=1,
                    )
                    fetched += 1
                    success = True
                    break
                except Exception as exc:
                    msg = str(exc)
                    is_503 = "503" in msg or "Service Unavailable" in msg or "DNS cache overflow" in msg
                    if attempt < 7 and is_503:
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 30.0)
                        # rebuild client on persistent failure
                        if attempt >= 3:
                            try:
                                cl = CAVEclient(MICRONS_DATASTACK, server_address=CAVE_SERVER, auth_token=args.token)
                                cl.version = version
                                clients[version] = cl
                            except Exception:
                                pass
                        continue
                    failed += 1
                    if not is_503:
                        print(f"  [{root_id}] FAILED: {exc!r}")
                    break
            if not success and is_503:
                # Don't waste credits hammering a sick service
                print(f"  [{root_id}] gave up after retries (CAVE service degraded)")

        if i % 10 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(items) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(items)}]  fetched={fetched}  cached={cached_hits}  failed={failed}  {rate:.1f} roots/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s  fetched={fetched}  cached={cached_hits}  failed={failed}")

    # Write a manifest mapping box_hash -> root IDs and version
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(box_root_map, f, indent=2)
    print(f"Manifest written to {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
