"""Re-warm the synapse cache at 1M cap (no L2 skeletons) to get honest full coverage.

Synapses only: fast (~10 min/region). After this, run warm_cache.py to fetch L2
skeletons for any newly discovered fragments not in the current 11k-entry cache.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("NEURONAUTS_SYNAPSE_CACHE_DIR", str(REPO / "cache" / "synapse"))
os.environ.setdefault("NEURONAUTS_L2_CACHE_DIR", str(REPO / "cache" / "l2_skeleton"))

import numpy as np  # noqa: E402
from neuronauts.data import lineage as L  # noqa: E402
from treestitch.realworld import build_region_world  # noqa: E402

y0, y1, z0, z1 = 930_000, 1_000_000, 780_000, 880_000
buf = 50_000
REGIONS = {
    "T1": ((1_150_000, y0, z0), (1_350_000, y1, z1)),
    "T2": ((550_000, y0, z0), (750_000, y1, z1)),
    "T3": ((1_150_000, 870_000, z0), (1_350_000, 940_000, z1)),
    "T4": ((1_150_000, 1_000_000, z0), (1_350_000, 1_070_000, z1)),
    "A":  ((750_000, y0, z0), (950_000, y1, z1)),
    "B":  ((950_000, y0, z0), (1_150_000 - buf, y1, z1)),
    "C":  ((1_350_000 + buf, y0, z0), (1_550_000, y1, z1)),
    "D":  ((1_550_000 + buf, y0, z0), (1_750_000, y1, z1)),
    "E":  ((750_000, 1_000_000, z0), (950_000, 1_070_000, z1)),
}
MAX_SYN = 1_000_000


def main() -> int:
    args = sys.argv[1:]
    only = args if args else list(REGIONS)
    print(f"Warming synapse cache at {MAX_SYN:,} cap for: {', '.join(only)}", flush=True)
    print(f"Synapse cache → {L._synapse_cache_dir()}", flush=True)
    for name in only:
        bbox = REGIONS[name]
        t0 = time.time()
        try:
            # l2_skeletons=False: synapse fetch only (instant cache hit if already cached)
            frags, region, _ = build_region_world(
                bbox, version=1718, side="pre", max_synapses=MAX_SYN,
                min_syn_per_fragment=5, seed=0, verbose=False, l2_skeletons=False,
                tile_x_nm=40_000, per_tile_limit=200_000)
        except Exception as exc:
            print(f"[{name}] ERROR: {exc}", flush=True)
            continue
        dt = time.time() - t0
        n = len(frags)
        nsyn = len(region.obs_positions) if hasattr(region, 'obs_positions') else -1
        syn_cache = L._synapse_cache_dir()
        nent = len(list(syn_cache.glob("*.npz"))) if syn_cache else 0
        print(f"[{name}] {n} fragments, {dt/60:.1f} min — syn cache now {nent} entries", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
