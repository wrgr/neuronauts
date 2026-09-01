"""Fetch L2 geometry for the label-blind atom population, in widening tiers.

Runs ``>=10``, then ``>=5``, then ``>=1`` synapses. Each tier only fetches
atoms the previous tiers did not cover, so the sequence costs the same as going
straight to ``>=1`` while producing a usable substrate after the first tier and
letting us look before committing to the next.

Geometry is always fetched against the *outer* region bounds so one cache
serves every nested region and, later, a scale-up to all somata.

Safe to interrupt and rerun; progress is on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.harness.geometry import (  # noqa: E402
    AtomGeometryStore, fetch_atom_topology, fetch_l2_attributes,
)
from neuronauts.harness.population import load_population  # noqa: E402
from neuronauts.harness.substrate import region_bounds  # noqa: E402
from neuronauts.report.provenance import write_result  # noqa: E402


def _coverage(attrs, pool, col):
    """Fraction of *this tier's* L2 nodes that actually have the attribute.

    Measured against the tier's own node set, not against the rows the cache
    happens to hold: the latter is ~100% by construction and hid a 14% loss
    when whole request batches were dropped.
    """
    ids = attrs.get("l2_id")
    v = attrs.get(col)
    if ids is None or v is None or not len(ids) or not len(pool):
        return 0.0
    order = np.argsort(ids, kind="stable")
    srt = ids[order]
    p = np.clip(np.searchsorted(srt, pool), 0, len(srt) - 1)
    ok = srt[p] == pool
    val = v[order[p[ok]]]
    finite = np.isfinite(val[:, 0] if val.ndim == 2 else val)
    return float(finite.sum() / len(pool))


def tier_report(store, attrs, atoms_in_tier, pool, k, elapsed):
    geom = store.load_all()
    sizes = np.asarray([len(geom[a]["l2_ids"]) for a in atoms_in_tier
                        if a in geom])
    have = int((sizes > 0).sum())
    edges = np.asarray([len(geom[a]["edges"]) for a in atoms_in_tier
                        if a in geom])
    covered = _coverage(attrs, pool, "pos_nm")
    cal_cov = _coverage(attrs, pool, "mean_dt_nm")

    print(f"\n  --- tier >={k} summary ---")
    print(f"  atoms in tier          : {len(atoms_in_tier):,}")
    print(f"  atoms with >=1 L2 node : {have:,} ({have/max(len(atoms_in_tier),1):.1%})")
    if len(sizes):
        print(f"  L2 nodes per atom      : median {int(np.median(sizes))}, "
              f"p90 {int(np.percentile(sizes,90))}, max {int(sizes.max())}")
        print(f"  total L2 nodes         : {int(sizes.sum()):,}")
        print(f"  atoms with adjacency   : {int((edges>0).sum()):,}")
    print(f"  rep_coord coverage     : {covered:.1%}")
    print(f"  caliber (mean_dt)      : {cal_cov:.1%}")
    print(f"  elapsed                : {elapsed/60:.1f} min")
    return {"k": k, "n_atoms": int(len(atoms_in_tier)), "n_with_geom": have,
            "total_l2_nodes": int(sizes.sum()) if len(sizes) else 0,
            "coord_coverage": covered, "caliber_coverage": cal_cov,
            "elapsed_min": elapsed / 60}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="data/substrate/c100um/population.npz")
    ap.add_argument("--geom-dir", default="data/substrate/geom")
    ap.add_argument("--bounds-centre-um", type=float, nargs=3,
                    default=[663, 591, 860])
    ap.add_argument("--bounds-side-um", type=float, default=200.0,
                    help="outer bounds; keep fixed so caches compose")
    ap.add_argument("--tiers", type=int, nargs="+", default=[10, 5, 1])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--max-atoms-per-tier", type=int, default=0)
    ap.add_argument("--out", default="results/atom_geometry_tiers.json")
    args = ap.parse_args()

    pop = load_population(args.population)
    _lo, _hi, seg = region_bounds(args.bounds_centre_um, args.bounds_side_um)
    store = AtomGeometryStore(args.geom_dir)
    attr_cache = Path(args.geom_dir) / "l2_attributes.npz"

    print(f"population : {args.population}")
    print(f"atoms      : {len(pop.atom_id):,}")
    print(f"bounds     : {args.bounds_side_um:g} um @ {args.bounds_centre_um}")
    print(f"tiers      : {args.tiers}\n", flush=True)

    summaries = []
    for k in args.tiers:
        t0 = time.time()
        sel = pop.atom_id[pop.n_synapses >= k]
        if args.max_atoms_per_tier:
            sel = sel[:args.max_atoms_per_tier]
        print(f"{'='*64}\nTIER >={k} synapses : {len(sel):,} atoms", flush=True)

        info = fetch_atom_topology(sel, seg, store, workers=args.workers,
                                   batch=args.batch, tag=f"k{k}")
        print(f"  written {info['fetched']:,}, no-geometry {info.get('absent',0):,}, "
              f"retryable {info['errors']:,}", flush=True)

        geom = store.load_all()
        pool = [geom[int(a)]["l2_ids"] for a in sel.tolist() if int(a) in geom]
        pool = (np.unique(np.concatenate(pool)) if pool
                else np.zeros(0, np.uint64))
        print(f"  pooled L2 nodes: {len(pool):,}", flush=True)
        attrs = fetch_l2_attributes(pool, attr_cache, workers=args.workers)

        summaries.append(tier_report(store, attrs, [int(a) for a in sel.tolist()],
                                     pool, k, time.time() - t0))
        write_result(args.out, {"tiers": summaries},
                     inputs=[args.population],
                     params={"tiers": list(args.tiers),
                             "bounds_centre_um": list(args.bounds_centre_um),
                             "bounds_side_um": args.bounds_side_um,
                             "workers": args.workers, "batch": args.batch},
                     quick_hash=True, synthetic_fallback=False)
        print(f"  wrote {args.out}", flush=True)

    print(f"\n{'='*64}\nall tiers complete")


if __name__ == "__main__":
    main()
