#!/usr/bin/env python3
"""Track B: bigger proofread-column sample, fully CHECKPOINTED + resumable.

The agent proxy restarts (new port) kill long network jobs. Everything here checkpoints to
disk so a relaunch resumes instead of restarting:
  * per-box synapse cache      data/bigdata/box_NN.npz   (skip if present)
  * incremental sv->v1718 map  data/bigdata/svmap.npz    (saved every chunk)
  * SideTable                  data/sidetable_big.npz
  * skeletons                  data/skel_v117 / data/skel_v1718 (per-file cache via fetch)
Pair with a relaunch loop (run_bigdata.sh) that re-invokes until sidetable_big.npz exists.

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

from experiments.pcfg_synapse_partitions.skeleton_topology_merge import fetch_skeletons  # noqa: E402

CENTER = (733_592, 513_592, 595_640)
SYN_VOX = np.array([4.0, 4.0, 40.0])


def grid_centers(n, side_um):
    step = side_um * 1000.0
    offs = sorted(((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)),
                  key=lambda o: abs(o[0]) + abs(o[1]) + abs(o[2]))
    return [np.array(CENTER, float) + np.array(o, float) * step for o in offs[:n]]


def fetch_box(client, c, side_um, out_path):
    half = side_um * 1000.0 / 2.0
    lo = ((c - half) / SYN_VOX).astype(np.int64); hi = ((c + half) / SYN_VOX).astype(np.int64)
    df = client.materialize.query_table("synapses_pni_2",
                                        filter_spatial_dict={"ctr_pt_position": [lo.tolist(), hi.tolist()]},
                                        split_positions=False)
    # CAVE caps query_table at LIMIT 500000 (seen in the server SQL). A 40um box returns ~36k, so
    # we are well clear -- but flag it loudly rather than silently lose synapses if a dense box hits
    # the cap. (Server-side 500/503 crashes are handled by the watcher, not here.)
    if len(df) >= 500_000:
        print(f"  !! WARNING box returned {len(df)} rows -- at/over CAVE's 500k cap; "
              f"shrink --side-um and re-fetch this box.", flush=True)
        return -1
    if len(df) == 0:
        np.savez(out_path, empty=True); return 0
    np.savez(out_path,
             syn_id=df.index.values.astype(np.int64),
             pre_pt=np.stack(df["pre_pt_position"].values) * SYN_VOX,
             post_pt=np.stack(df["post_pt_position"].values) * SYN_VOX,
             pre_sv=df["pre_pt_supervoxel_id"].values.astype(np.int64),
             post_sv=df["post_pt_supervoxel_id"].values.astype(np.int64),
             pre_rv=df["pre_pt_root_id"].values.astype(np.int64),
             post_rv=df["post_pt_root_id"].values.astype(np.int64))
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("CAVE_TOKEN"))
    ap.add_argument("--later-version", type=int, default=1718)
    ap.add_argument("--n-boxes", type=int, default=27)
    ap.add_argument("--side-um", type=float, default=40.0)
    ap.add_argument("--out", default="data/sidetable_big.npz")
    ap.add_argument("--min-syn", type=int, default=8)
    args = ap.parse_args()
    if not args.token:
        ap.error("CAVE token required")
    import datetime as dt
    from caveclient import CAVEclient

    bd = Path("data/bigdata"); bd.mkdir(parents=True, exist_ok=True)
    client = CAVEclient("minnie65_public", auth_token=args.token); client.version = 117
    later_ts = client.materialize.get_timestamp(args.later_version)
    if later_ts.tzinfo is None:
        later_ts = later_ts.replace(tzinfo=dt.timezone.utc)

    # --- 1. per-box synapse fetch (checkpointed) ---
    centers = grid_centers(args.n_boxes, args.side_um)
    missing = False
    for i, c in enumerate(centers):
        bp = bd / f"box_{i:02d}.npz"
        if bp.exists():
            continue
        n = fetch_box(client, c, args.side_um, bp)
        if n < 0:                      # truncated at the cap -- left uncached on purpose
            missing = True; continue
        print(f"[box {i+1}/{len(centers)}] {n} synapses cached", flush=True)
    if missing:
        print("[abort] some boxes exceeded the 200k cap; shrink --side-um. Not building SideTable.",
              flush=True)
        return

    # --- 2. concat boxes ---
    cols = collections.defaultdict(list)
    for i in range(len(centers)):
        bp = bd / f"box_{i:02d}.npz"
        if not bp.exists():            # gap in the grid (e.g. still-failing box); resume later
            print(f"[concat] box {i:02d} not cached yet; will resume on next pass", flush=True)
            return
        b = np.load(bp, allow_pickle=True)
        if "empty" in b:
            continue
        for k in ("syn_id", "pre_pt", "post_pt", "pre_sv", "post_sv", "pre_rv", "post_rv"):
            cols[k].append(b[k])
    A = {k: np.concatenate(v) for k, v in cols.items()}
    print(f"[concat] {len(A['syn_id']):,} synapses across {len(centers)} boxes", flush=True)

    # --- 3. sv -> v1718 root, chunked + checkpointed ---
    smap_p = bd / "svmap.npz"
    svmap = {}
    if smap_p.exists():
        z = np.load(smap_p)
        svmap = dict(zip(z["sv"].tolist(), z["root"].tolist()))
    all_sv = np.unique(np.concatenate([A["pre_sv"], A["post_sv"]]))
    all_sv = all_sv[all_sv > 0]
    todo = np.array([s for s in all_sv.tolist() if s not in svmap], dtype=np.int64)
    print(f"[svmap] {len(svmap):,} cached, {len(todo):,} to map", flush=True)
    chunk = 50_000
    for s in range(0, len(todo), chunk):
        batch = todo[s:s + chunk]
        roots = client.chunkedgraph.get_roots(batch.tolist(), timestamp=later_ts)
        svmap.update({int(k): int(v) for k, v in zip(batch.tolist(), roots)})
        sv = np.array(list(svmap.keys()), np.int64); rt = np.array(list(svmap.values()), np.int64)
        np.savez(smap_p, sv=sv, root=rt)               # checkpoint every chunk
        print(f"[svmap] mapped {min(s+chunk, len(todo)):,}/{len(todo):,}", flush=True)

    def lat(sv):
        return np.array([svmap.get(int(x), 0) for x in sv], np.int64)

    # --- 4. build SideTable (both sides), save ---
    keep_pre = A["pre_rv"] > 0; keep_post = A["post_rv"] > 0
    syn_id = np.concatenate([A["syn_id"][keep_pre], A["syn_id"][keep_post]])
    side = np.concatenate([np.zeros(keep_pre.sum(), np.int8), np.ones(keep_post.sum(), np.int8)])
    pt = np.concatenate([A["pre_pt"][keep_pre], A["post_pt"][keep_post]]).astype(np.float64)
    root_v117 = np.concatenate([A["pre_rv"][keep_pre], A["post_rv"][keep_post]])
    root_later = np.concatenate([lat(A["pre_sv"][keep_pre]), lat(A["post_sv"][keep_post])])
    np.savez(args.out, syn_id=syn_id, side=side, pt=pt, root_v117=root_v117, root_later=root_later)
    print(f"[sidetable] saved {args.out}: {len(syn_id):,} sides", flush=True)

    # --- 5. skeletons (per-file cache -> resumable) ---
    valid = root_later > 0
    by = collections.defaultdict(set); cv = collections.Counter(); cl = collections.Counter()
    for a, b in zip(root_v117[valid].tolist(), root_later[valid].tolist()):
        by[a].add(b); cv[a] += 1; cl[b] += 1
    fm = sorted(a for a, s in by.items() if len(s) >= 2 and cv[a] >= args.min_syn)
    cells = sorted(b for b in cl if cl[b] >= 15)
    print(f"[skeletons] false-merge v117 roots={len(fm)}  cells={len(cells)}", flush=True)
    fetch_skeletons(fm, 117, args.token, "data/skel_v117", workers=16)
    fetch_skeletons(cells[:4000], args.later_version, args.token, "data/skel_v1718", workers=16)
    print("[done] Track B complete", flush=True)


if __name__ == "__main__":
    main()
