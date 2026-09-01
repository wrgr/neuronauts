"""Extract synapses for one or more regions from the local static table.

One streaming pass over ``synapses_pni_2_v1_filtered_view.csv.gz`` (337M rows,
~3.2 min) writes a per-region NPZ. Every region is collected in the same pass,
so adding regions is nearly free -- the cost is the scan, not the box.

Why this table rather than a CAVE spatial query: it is already on disk, and it
carries ``pre/post_pt_supervoxel_id``. Supervoxel ids are timestamp-independent,
so a region extracted once can be relabelled to *any* materialization later with
a batched ``roots_at`` call. The stored ``*_root_id`` columns are v1078 and are
kept only for reference.

Positions are (4, 4, 40) nm voxels in the source and are converted to nm here so
that everything downstream is in one frame.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

GZ = Path("data/microns_static/v1078/synapses_pni_2_v1_filtered_view.csv.gz")
OUT_DIR = Path("data/regions")
TOTAL_ROWS = 337_312_428
PT_VOXEL_NM = np.asarray([4.0, 4.0, 40.0])

COLS = ["id", "pre_x", "pre_y", "pre_z", "ctr_x", "ctr_y", "ctr_z",
        "post_x", "post_y", "post_z", "size",
        "pre_sv", "pre_root", "post_sv", "post_root"]
USE = ["id", "ctr_x", "ctr_y", "ctr_z", "pre_x", "pre_y", "pre_z",
       "post_x", "post_y", "post_z", "size", "pre_sv", "post_sv"]

# Chosen from data/gt_manifest/region_candidates.json (proofread soma density).
# One slab wide enough in x to carve train | seam buffer | val.
DEFAULT_REGIONS = {
    # 400 x 350 x 350 um covering both the densest gold (663,591,860) and
    # silver (763,658,925) proofread cubes.
    "dense_v1": {"centre_um": [713, 625, 892], "side_um": [400, 350, 350]},
}


def region_bounds_nm(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    centre = np.asarray(spec["centre_um"], dtype=np.float64) * 1000.0
    side = np.asarray(spec["side_um"], dtype=np.float64) * 1000.0
    return centre - side / 2.0, centre + side / 2.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions-json", default=None,
                    help="JSON mapping name -> {centre_um, side_um}")
    ap.add_argument("--chunk-rows", type=int, default=4_000_000)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--limit-chunks", type=int, default=0,
                    help="stop early (smoke test)")
    args = ap.parse_args()

    regions = (json.loads(Path(args.regions_json).read_text())
               if args.regions_json else DEFAULT_REGIONS)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = {name: region_bounds_nm(spec) for name, spec in regions.items()}
    for name, (lo, hi) in bounds.items():
        print(f"region {name}: {lo/1000} -> {hi/1000} um")
    print()

    buf: dict[str, list[pd.DataFrame]] = {name: [] for name in regions}
    reader = pd.read_csv(
        GZ, header=None, names=COLS, usecols=USE,
        dtype={c: np.int64 for c in ("id", "ctr_x", "ctr_y", "ctr_z", "pre_x",
                                     "pre_y", "pre_z", "post_x", "post_y",
                                     "post_z", "size")}
        | {"pre_sv": np.uint64, "post_sv": np.uint64},
        chunksize=args.chunk_rows,
    )

    t0 = time.time()
    rows = 0
    for i, chunk in enumerate(reader):
        rows += len(chunk)
        ctr = chunk[["ctr_x", "ctr_y", "ctr_z"]].to_numpy(np.float64) * PT_VOXEL_NM
        for name, (lo, hi) in bounds.items():
            m = np.all((ctr >= lo) & (ctr < hi), axis=1)
            if m.any():
                buf[name].append(chunk.loc[m])
        if i % 10 == 0:
            dt = time.time() - t0
            got = {n: sum(len(d) for d in v) for n, v in buf.items()}
            print(f"  {rows/1e6:6.1f}M rows  {dt:5.1f}s  "
                  f"({rows/dt/1e6:.2f}M rows/s)  hits={got}", flush=True)
        if args.limit_chunks and i + 1 >= args.limit_chunks:
            print("  (early stop: --limit-chunks)")
            break

    print(f"\nscanned {rows:,} rows in {time.time()-t0:.1f}s\n")

    manifest = {}
    for name, parts in buf.items():
        if not parts:
            print(f"region {name}: NO SYNAPSES -- check bounds")
            continue
        df = pd.concat(parts, ignore_index=True)
        lo, hi = bounds[name]
        payload = {
            "synapse_id": df["id"].to_numpy(np.int64),
            "ctr_nm": (df[["ctr_x", "ctr_y", "ctr_z"]].to_numpy(np.float64)
                       * PT_VOXEL_NM).astype(np.float32),
            "pre_nm": (df[["pre_x", "pre_y", "pre_z"]].to_numpy(np.float64)
                       * PT_VOXEL_NM).astype(np.float32),
            "post_nm": (df[["post_x", "post_y", "post_z"]].to_numpy(np.float64)
                        * PT_VOXEL_NM).astype(np.float32),
            "size": df["size"].to_numpy(np.int32),
            "pre_sv": df["pre_sv"].to_numpy(np.uint64),
            "post_sv": df["post_sv"].to_numpy(np.uint64),
        }
        path = out_dir / f"{name}_synapses.npz"
        np.savez_compressed(path, **payload)
        n_sv = len(np.unique(np.concatenate([payload["pre_sv"], payload["post_sv"]])))
        manifest[name] = {
            "spec": regions[name],
            "lower_nm": lo.tolist(), "upper_nm": hi.tolist(),
            "n_synapses": int(len(df)),
            "n_unique_supervoxels": int(n_sv),
            "source_table": str(GZ), "source_version": 1078,
            "position_units": "nm", "path": str(path),
        }
        print(f"region {name}: {len(df):,} synapses, {n_sv:,} unique supervoxels "
              f"-> {path}")

    (out_dir / "regions_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out_dir}/regions_manifest.json")


if __name__ == "__main__":
    main()
