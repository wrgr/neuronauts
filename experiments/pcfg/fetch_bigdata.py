#!/usr/bin/env python3
"""Track B: pull a bigger proofread-column sample, in the background.

Tiles a denser 27-box grid (3x3x3 @ 40um) over the column center, builds a larger SideTable
(synapses + supervoxel->v1718 root mapping), then fetches the v117 skeletons of its false-merge
roots and the v1718 skeletons of its cells. Output: data/sidetable_big.npz + grown skel caches.
Run in the background while Phases 1-4 build on the current cache.

    CAVE_TOKEN=... python -m experiments.pcfg_synapse_partitions.fetch_bigdata --n-boxes 27
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg_synapse_partitions.run_synapse_correction import fetch_side_table  # noqa: E402
from experiments.pcfg_synapse_partitions.skeleton_topology_merge import fetch_skeletons  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--token", default=os.environ.get("CAVE_TOKEN"))
    ap.add_argument("--later-version", type=int, default=1718)
    ap.add_argument("--n-boxes", type=int, default=27)
    ap.add_argument("--side-um", type=float, default=40.0)
    ap.add_argument("--out", default="data/sidetable_big.npz")
    ap.add_argument("--min-syn", type=int, default=8)
    args = ap.parse_args()
    if not args.token:
        ap.error("CAVE token required (--token or CAVE_TOKEN)")

    t0 = time.time()
    print(f"[bigdata] fetching {args.n_boxes}-box SideTable (side {args.side_um}um)...", flush=True)
    tab = fetch_side_table(args.token, later_version=args.later_version, n_boxes=args.n_boxes,
                           side_um=args.side_um, sides="both", seed=0)
    np.savez(args.out, syn_id=tab.syn_id, side=tab.side, pt=tab.pt,
             root_v117=tab.root_v117, root_later=tab.root_later)
    print(f"[bigdata] saved {args.out}: {len(tab):,} sides in {(time.time()-t0)/60:.1f}min", flush=True)

    valid = tab.root_later > 0
    by_v117 = collections.defaultdict(set)
    cnt_v117, cnt_later = collections.Counter(), collections.Counter()
    cells = set()
    for a, b in zip(tab.root_v117[valid].tolist(), tab.root_later[valid].tolist()):
        by_v117[a].add(b); cnt_v117[a] += 1; cnt_later[b] += 1; cells.add(b)
    fm = sorted(a for a, s in by_v117.items() if len(s) >= 2 and cnt_v117[a] >= args.min_syn)
    cell_ids = sorted(b for b in cells if cnt_later[b] >= 15)
    print(f"[bigdata] false-merge v117 roots = {len(fm)}  cells(>=15 syn) = {len(cell_ids)}", flush=True)

    print("[bigdata] fetching v117 skeletons for false-merge roots (slow long-pole)...", flush=True)
    g117 = fetch_skeletons(fm, 117, args.token, "data/skel_v117", workers=20)
    print(f"[bigdata] v117 skeletons: {len(g117)} ok", flush=True)

    print("[bigdata] fetching v1718 cell skeletons...", flush=True)
    g1718 = fetch_skeletons(cell_ids[:3000], args.later_version, args.token, "data/skel_v1718", workers=20)
    print(f"[bigdata] v1718 skeletons: {len(g1718)} ok; total {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
