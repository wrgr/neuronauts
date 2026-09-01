"""Time a chunked scan of the 20 GB static synapse table and extrapolate.

The static v1078 table carries positions *and* supervoxel ids. Supervoxel ids
are timestamp-independent, so a region extracted here can be relabelled to any
CAVE materialization with a batched ``roots_at`` call. That makes a one-time
full scan the cheapest route to an offline region substrate -- if the scan is
fast enough. This measures the rate instead of guessing at it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

GZ = "data/microns_static/v1078/synapses_pni_2_v1_filtered_view.csv.gz"
COLS = ["id", "pre_x", "pre_y", "pre_z", "ctr_x", "ctr_y", "ctr_z",
        "post_x", "post_y", "post_z", "size",
        "pre_sv", "pre_root", "post_sv", "post_root"]
USE = ["ctr_x", "ctr_y", "ctr_z", "pre_sv", "post_sv", "pre_root", "post_root", "size"]
TOTAL_ROWS = 337_312_428


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--n-chunks", type=int, default=4)
    args = ap.parse_args()

    reader = pd.read_csv(
        GZ, header=None, names=COLS, usecols=USE,
        dtype={"ctr_x": np.int32, "ctr_y": np.int32, "ctr_z": np.int32,
               "pre_sv": np.uint64, "post_sv": np.uint64,
               "pre_root": np.uint64, "post_root": np.uint64,
               "size": np.int32},
        chunksize=args.chunk_rows,
    )

    t0 = time.time()
    rows = 0
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for i, chunk in enumerate(reader):
        rows += len(chunk)
        xyz = chunk[["ctr_x", "ctr_y", "ctr_z"]].to_numpy()
        mins = np.minimum(mins, xyz.min(0))
        maxs = np.maximum(maxs, xyz.max(0))
        dt = time.time() - t0
        print(f"chunk {i}: rows={rows:,} elapsed={dt:.1f}s "
              f"rate={rows/dt:,.0f} rows/s", flush=True)
        if i + 1 >= args.n_chunks:
            break

    dt = time.time() - t0
    rate = rows / dt
    print(f"\nrows scanned : {rows:,}")
    print(f"rate         : {rate:,.0f} rows/s")
    print(f"full scan est: {TOTAL_ROWS / rate / 60:.1f} min "
          f"for all {TOTAL_ROWS:,} rows")
    print(f"\nctr position range seen (voxels): min={mins} max={maxs}")
    print(f"  in um assuming (4,4,40) nm/voxel: "
          f"min={mins * np.array([4, 4, 40]) / 1000} "
          f"max={maxs * np.array([4, 4, 40]) / 1000}")


if __name__ == "__main__":
    main()
