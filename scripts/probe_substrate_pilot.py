"""Pilot the proofread-cell-first substrate on a handful of cells.

The planned substrate is: proofread cell at v1822 -> real L2 adjacency graph
(``level2_chunk_graph``) -> each L2 node labelled with the v117 root it belonged
to (``roots_at`` at the v117 timestamp) -> v117 fragments become the atoms the
grammar assembles, with the proofread cell id as ground truth.

That whole design rests on one unverified assumption: that ``roots_at`` returns
a sensible v117 root for a *current* L2 node id. L2 nodes are chunk-level
objects that edits can create or destroy, so current L2 ids need not have
existed at v117. ``build_region_world_l2`` assumes this works, but it swallows
failures, so a silent zero rate would look like sparse data.

This measures, on a few cells: fetch cost, node counts, v117 resolution rate,
and how many v117 fragments each proofread cell breaks into (the merge signal
we need to exist for the task to be learnable at all).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.data import lineage as L  # noqa: E402

GT_DIR = Path("data/gt_manifest")
PT_VOXEL_NM = np.asarray([4.0, 4.0, 40.0])
SEG_VOXEL_NM = np.asarray([8.0, 8.0, 40.0])


def load_proofread(version: int) -> pd.DataFrame:
    df = pd.read_csv(GT_DIR / f"proofreading_status_v{version}.csv.gz")
    df["x_nm"] = df["pt_position_x"] * PT_VOXEL_NM[0]
    df["y_nm"] = df["pt_position_y"] * PT_VOXEL_NM[1]
    df["z_nm"] = df["pt_position_z"] * PT_VOXEL_NM[2]
    return df


def select_cells(df, centre_um, side_um, tier):
    centre = np.asarray(centre_um, float) * 1000.0
    half = side_um * 1000.0 / 2.0
    pos = df[["x_nm", "y_nm", "z_nm"]].to_numpy()
    m = np.all(np.abs(pos - centre) <= half, axis=1)
    m &= df["status_dendrite"].astype(bool).to_numpy()
    m &= df["status_axon"].astype(bool).to_numpy()
    m &= (df["strategy_dendrite"].astype(str) == "dendrite_extended").to_numpy()
    if tier == "gold":
        m &= (df["strategy_axon"].astype(str) == "axon_fully_extended").to_numpy()
    return df.loc[m]


def l2_graph(root_id: int, bounds_seg_vox):
    """Real L2 adjacency for a root, restricted to a bounds box."""
    from caveclient import CAVEclient
    client = CAVEclient("minnie65_public")
    t0 = time.time()
    edges = client.chunkedgraph.level2_chunk_graph(int(root_id),
                                                   bounds=bounds_seg_vox)
    return np.asarray(edges, dtype=np.uint64), time.time() - t0


def fetch_coords(l2_ids: np.ndarray, token: str, batch: int = 2000,
                 workers: int = 8) -> dict[int, np.ndarray]:
    url = f"{L.L2_CACHE_SERVER}/l2cache/api/v1/table/{L.L2_TABLE}/attributes"
    hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    chunks = [l2_ids[i:i + batch].tolist() for i in range(0, len(l2_ids), batch)]

    def go(chunk):
        try:
            r = requests.post(url, headers=hdr, timeout=180,
                              json={"l2_ids": chunk,
                                    "attribute_names": ["rep_coord_nm"]})
            if r.status_code != 200:
                return {}
            return {int(k): np.asarray(v["rep_coord_nm"], np.float32)
                    for k, v in r.json().items() if v.get("rep_coord_nm")}
        except Exception:
            return {}

    out: dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d in ex.map(go, chunks):
            out.update(d)
    return out


def map_to_v117(l2_ids: np.ndarray, token: str, workers: int = 8) -> np.ndarray:
    """roots_at at the v117 timestamp, batched and threaded."""
    batch = L._ROOTS_BATCH
    chunks = [l2_ids[i:i + batch] for i in range(0, len(l2_ids), batch)]

    def go(chunk):
        try:
            r = L.roots_at(chunk, L.V117_TIMESTAMP, token=token)
            return r if r is not None else np.zeros(len(chunk), np.uint64)
        except Exception:
            return np.zeros(len(chunk), np.uint64)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return np.concatenate(list(ex.map(go, chunks)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1822)
    ap.add_argument("--centre-um", type=float, nargs=3, default=[663, 591, 860])
    ap.add_argument("--side-um", type=float, default=200.0)
    ap.add_argument("--tier", default="gold", choices=["gold", "silver"])
    ap.add_argument("--n-cells", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="results/probe_substrate_pilot.json")
    args = ap.parse_args()

    token = L.DEFAULT_TOKEN
    df = load_proofread(args.version)
    sel = select_cells(df, args.centre_um, args.side_um, args.tier)
    print(f"proofread cells in {args.side_um:g} um cube "
          f"[{args.tier}]: {len(sel)}", flush=True)
    if len(sel) == 0:
        raise SystemExit("no cells selected")

    centre = np.asarray(args.centre_um, float) * 1000.0
    half = args.side_um * 1000.0 / 2.0
    lo, hi = centre - half, centre + half
    # The chunkedgraph parses each bound with int(), so these must be whole
    # numbers -- passing floats yields a 500 with "invalid literal for int()".
    bounds = np.array([[lo[i] / SEG_VOXEL_NM[i], hi[i] / SEG_VOXEL_NM[i]]
                       for i in range(3)], dtype=int)
    print(f"bounds (seg voxels):\n{bounds.astype(int)}\n", flush=True)

    cells = sel.head(args.n_cells)
    recs = []
    t_start = time.time()

    def do_cell(row):
        root = int(row.pt_root_id)
        try:
            edges, dt = l2_graph(root, bounds)
        except Exception as exc:  # noqa: BLE001
            return {"root": root, "error": f"{type(exc).__name__}: {exc}"}
        ids = np.unique(edges) if edges.size else np.zeros(0, np.uint64)
        return {"root": root, "t_graph_s": dt, "n_edges": int(len(edges)),
                "n_l2": int(len(ids)), "l2_ids": ids}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        raw = list(ex.map(do_cell, [r for _, r in cells.iterrows()]))
    t_graphs = time.time() - t_start
    print(f"[1] L2 adjacency graphs: {t_graphs:.1f}s for {len(raw)} cells", flush=True)
    for r in raw:
        if "error" in r:
            print(f"    root {r['root']}: ERROR {r['error']}")
        else:
            print(f"    root {r['root']}: {r['n_l2']:>6} L2 nodes, "
                  f"{r['n_edges']:>6} edges, {r['t_graph_s']:.1f}s")

    ok = [r for r in raw if "error" not in r and r["n_l2"] > 0]
    if not ok:
        raise SystemExit("no L2 graphs retrieved")

    pool = np.unique(np.concatenate([r["l2_ids"] for r in ok]))
    print(f"\n[2] pooled unique L2 ids: {len(pool):,}", flush=True)

    t0 = time.time()
    coords = fetch_coords(pool, token)
    t_coords = time.time() - t0
    print(f"    rep_coord_nm: {len(coords):,}/{len(pool):,} "
          f"({len(coords)/len(pool):.1%}) in {t_coords:.1f}s", flush=True)

    t0 = time.time()
    v117 = map_to_v117(pool, token)
    t_v117 = time.time() - t0
    nz = int((v117 > 0).sum())
    print(f"\n[3] roots_at(v117): {nz:,}/{len(pool):,} resolved "
          f"({nz/len(pool):.1%}) in {t_v117:.1f}s", flush=True)
    print("    ^^ this is the assumption the whole substrate design rests on")

    l2_to_v117 = dict(zip(pool.tolist(), v117.tolist()))
    print(f"\n[4] v117 fragmentation per proofread cell:")
    frag_counts = []
    for r in ok:
        frs = [l2_to_v117.get(int(i), 0) for i in r["l2_ids"]]
        frs = [f for f in frs if f]
        c = collections.Counter(frs)
        frag_counts.append(len(c))
        big = sum(1 for v in c.values() if v >= 5)
        print(f"    root {r['root']}: {len(c):>5} v117 fragments "
              f"({big} with >=5 L2 nodes) over {len(frs)} labelled nodes")

    # Do fragments get shared between cells? A shared fragment is a real v117
    # false merge -- the split signal. Zero would mean no atomization signal.
    owner: dict[int, set] = collections.defaultdict(set)
    for r in ok:
        for i in r["l2_ids"]:
            f = l2_to_v117.get(int(i), 0)
            if f:
                owner[f].add(r["root"])
    shared = {f: s for f, s in owner.items() if len(s) > 1}
    print(f"\n[5] v117 fragments spanning >1 proofread cell (false merges): "
          f"{len(shared)} of {len(owner)}")

    per_cell = t_graphs / max(len(raw), 1)
    print("\n" + "=" * 62)
    print(f"cost per cell: graph {per_cell:.1f}s | "
          f"coords {t_coords/len(ok):.1f}s | v117 {t_v117/len(ok):.1f}s")
    total = per_cell + (t_coords + t_v117) / len(ok)
    for n in (50, 150, 250):
        print(f"  extrapolated {n:>3} cells: {total*n/60:5.1f} min")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "config": vars(args),
        "n_cells_in_region": int(len(sel)),
        "cells": [{k: v for k, v in r.items() if k != "l2_ids"} for r in raw],
        "pooled_l2": int(len(pool)),
        "coord_rate": len(coords) / len(pool),
        "v117_resolve_rate": nz / len(pool),
        "frag_counts": frag_counts,
        "n_shared_fragments": len(shared),
        "timing": {"graphs_s": t_graphs, "coords_s": t_coords, "v117_s": t_v117},
    }, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
