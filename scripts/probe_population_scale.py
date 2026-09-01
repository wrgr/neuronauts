"""Size the label-blind atom population for candidate region sizes.

The atom population must be enumerated without using ground truth, or the task
is rigged. The GT-free filter is the one described for this work: every v117
fragment carrying at least k synapses in the region. Ground truth is attached
afterward, only where it happens to exist.

The cost driver is mapping synapse supervoxels to v117 roots, so this reports,
per region size: synapse count, unique supervoxel count, and the projected
mapping time at the measured batched ``roots_at`` rate. It also runs a real
timed sample so the projection is anchored to a measurement rather than a guess.

Reads the already-extracted region NPZ; no full table rescan.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.data import lineage as L  # noqa: E402

REGION_NPZ = Path("data/regions/dense_v1_synapses.npz")


def timed_roots_at(svids: np.ndarray, workers: int) -> tuple[float, float]:
    """Return (seconds, resolved_fraction) for a real batched+threaded call."""
    batch = L._ROOTS_BATCH
    chunks = [svids[i:i + batch] for i in range(0, len(svids), batch)]

    def go(chunk):
        try:
            r = L.roots_at(chunk, L.V117_TIMESTAMP)
            return r if r is not None else np.zeros(len(chunk), np.uint64)
        except Exception:
            return np.zeros(len(chunk), np.uint64)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = np.concatenate(list(ex.map(go, chunks)))
    return time.time() - t0, float((out > 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--centre-um", type=float, nargs=3, default=[663, 591, 860])
    ap.add_argument("--sides-um", type=float, nargs="+", default=[60, 100, 150, 200])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample-sv", type=int, default=40_000,
                    help="supervoxels to actually map, for a timed rate")
    ap.add_argument("--out", default="results/probe_population_scale.json")
    args = ap.parse_args()

    print(f"loading {REGION_NPZ} ...", flush=True)
    t0 = time.time()
    with np.load(REGION_NPZ, allow_pickle=False) as z:
        ctr = z["ctr_nm"]
        pre_sv, post_sv = z["pre_sv"], z["post_sv"]
    print(f"  {len(ctr):,} synapses loaded in {time.time()-t0:.1f}s\n", flush=True)

    centre = np.asarray(args.centre_um, float) * 1000.0
    rows = []
    for side in args.sides_um:
        half = side * 1000.0 / 2.0
        m = np.all(np.abs(ctr - centre) <= half, axis=1)
        n_syn = int(m.sum())
        sv = np.unique(np.concatenate([pre_sv[m], post_sv[m]]))
        sv = sv[sv > 0]
        rows.append({"side_um": side, "n_synapses": n_syn,
                     "n_unique_sv": int(len(sv)), "sv": sv})
        print(f"{side:>5.0f} um cube: {n_syn:>12,} synapses  "
              f"{len(sv):>12,} unique supervoxels", flush=True)

    # Anchor the projection on a real timed call.
    biggest = max(rows, key=lambda r: r["n_unique_sv"])
    sample = biggest["sv"][:args.sample_sv]
    print(f"\ntiming roots_at on {len(sample):,} real supervoxels "
          f"({args.workers} workers) ...", flush=True)
    dt, frac = timed_roots_at(sample, args.workers)
    rate = len(sample) / dt
    print(f"  {dt:.1f}s  ->  {rate:,.0f} supervoxels/s, "
          f"{frac:.1%} resolved to a v117 root\n", flush=True)

    print(f"{'side':>6}{'synapses':>14}{'uniq sv':>13}{'v117 map':>12}")
    for r in rows:
        est = r["n_unique_sv"] / rate / 60
        print(f"{r['side_um']:>6.0f}{r['n_synapses']:>14,}"
              f"{r['n_unique_sv']:>13,}{est:>10.1f}m")
        r.pop("sv")
        r["est_v117_map_min"] = est

    print("\nNote: each synapse endpoint is a distinct supervoxel almost "
          "always, so unique-sv ~= 2x synapses. Subsampling synapses is the "
          "lever if the mapping cost is too high.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"centre_um": args.centre_um, "rate_sv_per_s": rate,
         "resolve_fraction": frac, "rows": rows}, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
