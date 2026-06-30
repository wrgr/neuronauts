"""Warm the L2-skeleton + synapse caches region-by-region, into the git-lfs repo
cache, committing+pushing after each region so the (expensive) fetch is durably
persisted the moment it completes — surviving container reclaim.

Usage (serial, commit-per-region):
    PYTHONPATH=. python scripts/warm_cache.py [REGION ...]

Usage (parallel — no git, caller commits):
    PYTHONPATH=. python scripts/warm_cache.py --no-git [REGION ...]

With no args, warms all eval (T1-T4) + train (A-E) regions. Each region build
calls l2_skeleton per fragment; results land in cache/l2_skeleton/ (with
provenance) and are reused by every subsequent run.
"""
from __future__ import annotations

import os
import subprocess
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
    "A": ((750_000, y0, z0), (950_000, y1, z1)),
    "B": ((950_000, y0, z0), (1_150_000 - buf, y1, z1)),
    "C": ((1_350_000 + buf, y0, z0), (1_550_000, y1, z1)),
    "D": ((1_550_000 + buf, y0, z0), (1_750_000, y1, z1)),
    "E": ((750_000, 1_000_000, z0), (950_000, 1_070_000, z1)),
}
BRANCH = "claude/tree-dna-phase-1-G1DNn"
MAX_SYN = 1_000_000


def _git(*args: str) -> int:
    return subprocess.call(["git", *args], cwd=str(REPO))


def _commit_push(region: str, nfrag: int, dt: float) -> None:
    _git("add", "cache/l2_skeleton", "cache/synapse")
    if subprocess.call(["git", "diff", "--cached", "--quiet"], cwd=str(REPO)) == 0:
        print(f"  [{region}] no new cache entries to commit", flush=True)
        return
    msg = f"cache: warm L2 skeletons for region {region} ({nfrag} frags, {dt/60:.1f} min)"
    _git("commit", "-q", "-m", msg)
    for attempt in range(4):
        if _git("push", "origin", BRANCH) == 0:
            print(f"  [{region}] pushed", flush=True)
            return
        time.sleep(2 ** (attempt + 1))
    print(f"  [{region}] PUSH FAILED after retries", flush=True)


def main() -> int:
    args = sys.argv[1:]
    no_git = "--no-git" in args
    args = [a for a in args if a != "--no-git"]
    only = args if args else list(REGIONS)

    cdir = L._l2_cache_dir()
    print(f"Warming regions: {', '.join(only)} (no-git={no_git})", flush=True)
    print(f"L2 cache → {cdir}", flush=True)
    print(f"Synapse cache → {L._synapse_cache_dir()}", flush=True)
    for name in only:
        bbox = REGIONS[name]
        t0 = time.time()
        try:
            frags, region, _ = build_region_world(
                bbox, version=1718, side="pre", max_synapses=MAX_SYN,
                min_syn_per_fragment=5, seed=0, verbose=False, l2_skeletons=True,
                tile_x_nm=40_000, per_tile_limit=200_000)
        except Exception as exc:
            print(f"[{name}] ERROR: {exc}", flush=True)
            continue
        dt = time.time() - t0
        n = len(frags)
        nent = len(list(cdir.glob("*.npz"))) if cdir else 0
        print(f"[{name}] {n} fragments in {dt/60:.1f} min "
              f"({n/dt:.2f} frag/s) — cache now {nent} entries", flush=True)
        if not no_git:
            _commit_push(name, n, dt)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
