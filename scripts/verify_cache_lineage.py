#!/usr/bin/env python3
"""Verify that a box cache lands in proofread regions.

Probes a random sample of supervoxel IDs from the cache, resolves each to its
v117 root via the chunkedgraph, and compares against the cache's stored
``pre_root_id``.  If at least ``--min-edit-fraction`` of svids show lineage
divergence (cache != v117), the cache is in a proofread region and
``fetch-cave-edits-from-cache`` will yield real false-merge / false-split
pairs.  Otherwise exits non-zero so the surrounding pipeline aborts.

See docs/dataset_seeding_for_edit_pairs.md for context.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--n-svids", type=int, default=1000)
    ap.add_argument("--n-boxes", type=int, default=20)
    ap.add_argument("--per-box", type=int, default=50)
    ap.add_argument("--min-edit-fraction", type=float, default=0.05)
    ap.add_argument("--past-timestamp", default="2021-06-11")
    ap.add_argument("--datastack", default="minnie65_phase3_v1")
    args = ap.parse_args()

    from caveclient import CAVEclient

    secret_path = os.path.expanduser("~/.cloudvolume/secrets/cave-secret.json")
    token = json.load(open(secret_path))["token"]
    client = CAVEclient(args.datastack, auth_token=token)

    cache_dir = Path(args.cache_dir)
    files = sorted(p for p in cache_dir.iterdir() if p.suffix == ".npz")
    if not files:
        print(f"No .npz files in {cache_dir}")
        return 1

    rng = random.Random(0)
    sample_files = rng.sample(files, min(args.n_boxes, len(files)))

    svids: list[int] = []
    cache_roots: list[int] = []
    for f in sample_files:
        npz = np.load(f, allow_pickle=True)
        if "pre_seg_id" not in npz:
            continue
        n = len(npz["pre_seg_id"])
        idx = np.random.RandomState(0).choice(n, size=min(args.per_box, n), replace=False)
        svids.extend(int(s) for s in npz["pre_seg_id"][idx])
        cache_roots.extend(int(r) for r in npz["pre_root_id"][idx])
        if len(svids) >= args.n_svids:
            break

    svids = svids[: args.n_svids]
    cache_roots = cache_roots[: args.n_svids]
    if not svids:
        print("No svids sampled")
        return 1

    past_dt = dt.datetime.fromisoformat(args.past_timestamp).replace(
        tzinfo=dt.timezone.utc
    )
    print(
        f"Probing {len(svids):,} svids from {len(sample_files)} boxes "
        f"in {cache_dir} (past={args.past_timestamp})"
    )
    v117_roots = [int(r) for r in client.chunkedgraph.get_roots(svids, timestamp=past_dt)]

    n_diff = sum(int(c != v) for c, v in zip(cache_roots, v117_roots))
    frac = n_diff / len(svids)
    print(f"  cache != v117 root: {n_diff} / {len(svids)} ({100*frac:.1f}%)")
    print(f"  unique cache roots: {len(set(cache_roots))}")
    print(f"  unique v117  roots: {len(set(v117_roots))}")

    if frac < args.min_edit_fraction:
        print(
            f"FAIL: lineage edit fraction {frac:.3f} < threshold "
            f"{args.min_edit_fraction:.3f}.  The cache is not in a proofread "
            f"region; fetch-cave-edits-from-cache will yield no pairs."
        )
        return 1

    print(f"OK: lineage edit fraction {frac:.3f} >= {args.min_edit_fraction:.3f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
