"""Build and characterise the label-blind v117 atom population for a region.

Reports the atom-size distribution so the ``min_synapses`` filter can be chosen
on evidence: it sets how many atoms we then pay to fetch geometry for.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.harness.population import build_population  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-npz", default="data/regions/dense_v1_synapses.npz")
    ap.add_argument("--centre-um", type=float, nargs=3, default=[663, 591, 860])
    ap.add_argument("--side-um", type=float, default=100.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--cache-root", default="data/substrate")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    name = args.name or f"c{int(args.side_um)}um"
    cache = Path(args.cache_root) / name

    pop = build_population(args.region_npz, args.centre_um, args.side_um,
                           cache_dir=cache, workers=args.workers)

    print(f"\n{'='*60}")
    print(f"region      : {args.side_um:g} um cube @ {args.centre_um} um")
    print(f"synapses    : {pop.meta['n_synapses']:,}")
    print(f"supervoxels : {pop.meta['n_supervoxels']:,}")
    print(f"v117 atoms  : {len(pop.atom_id):,}  (label-blind)")

    print(f"\n{'min synapses':>13}{'atoms kept':>13}{'cum. synapses':>16}"
          f"{'% of synapse mass':>20}")
    total = float(pop.n_synapses.sum())
    for k in (1, 2, 3, 5, 10, 20, 50, 100):
        m = pop.n_synapses >= k
        s = float(pop.n_synapses[m].sum())
        print(f"{k:>13}{int(m.sum()):>13,}{int(s):>16,}{s/total:>19.1%}")

    q = np.percentile(pop.n_synapses, [50, 75, 90, 95, 99])
    print(f"\nsynapses/atom percentiles (50/75/90/95/99): "
          f"{[int(v) for v in q]}  max={int(pop.n_synapses.max())}")
    print(f"\ncached under {cache}")


if __name__ == "__main__":
    main()
