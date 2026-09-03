#!/usr/bin/env python3
"""Characterize the real v117 → v1718 fragmentation structure.

Samples proofread somata (from the public nucleus table), carries each forward
to its v1718 root via chunkedgraph lineage, and breaks it into the v117 roots
its supervoxels trace back to.  Each distinct v117 root is a real *fragment*.

Reports the real distribution we need to make benchmarks realistic:
  - how many neurons are already a single v117 root (v117 got them right)
  - among the rest, the "trunk + slivers" structure (dominant mass share)
  - how many fragments are *substantial* (≥ a supervoxel-count threshold) vs
    tiny slivers — i.e. how often the partition problem is genuinely non-trivial.

Usage
-----
  python attic/one_off_analyses/characterize_v117_to_v1718.py --n 40 --version 1718 --max-sv 1500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=40, help="number of somata to sample")
    p.add_argument("--version", type=int, default=1718, help="proofread target version")
    p.add_argument("--max-sv", type=int, default=1500,
                   help="supervoxels sampled per neuron for the breakdown")
    p.add_argument("--substantial-sv", type=int, default=30,
                   help="a fragment is 'substantial' if it holds >= this many "
                        "sampled supervoxels (else it is a sliver)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from neuronauts.data import lineage as L
    from neuronauts.data.loaders import sample_neurons

    ts = L.version_timestamp(args.version)
    print(f"versions available: {L.list_versions()}")
    print(f"target v{args.version} timestamp: {ts}  | v117: {L.V117_TIMESTAMP}\n")

    candidates = sample_neurons(args.n * 2, seed=args.seed)
    n_frags: list[int] = []
    n_substantial: list[int] = []
    dom_share: list[float] = []
    processed = 0

    for nuc_root in candidates:
        if processed >= args.n:
            break
        target = L.root_at_version(nuc_root, args.version)
        if target is None:
            continue
        fb = L.fragment_breakdown(target, max_sv=args.max_sv, seed=args.seed)
        if not fb:
            continue
        processed += 1
        counts = np.array(list(fb.values()), dtype=np.int64)
        total = int(counts.sum())
        nf = len(counts)
        nsub = int((counts >= args.substantial_sv).sum())
        n_frags.append(nf)
        n_substantial.append(nsub)
        dom_share.append(float(counts.max() / total))
        flag = "  << multi-substantial" if nsub >= 2 else ""
        print(f"  v{args.version} {target}: {nf:3d} v117 frags  "
              f"{nsub} substantial  dom_share={counts.max()/total:.3f}{flag}")

    if not n_frags:
        print("No neurons processed — check network/token.")
        return 1

    n_frags_a = np.array(n_frags)
    n_sub_a = np.array(n_substantial)
    dom_a = np.array(dom_share)
    single = int((n_frags_a == 1).sum())
    multi_sub = int((n_sub_a >= 2).sum())

    print(f"\n{'='*60}\nSUMMARY  (n={len(n_frags)} neurons, v117 → v{args.version})\n{'='*60}")
    print(f"  already a single v117 root:      {single}/{len(n_frags)} "
          f"({100*single/len(n_frags):.0f}%)")
    print(f"  >=2 SUBSTANTIAL v117 fragments:  {multi_sub}/{len(n_frags)} "
          f"({100*multi_sub/len(n_frags):.0f}%)  "
          f"(>= {args.substantial_sv} svs each — the non-trivial split cases)")
    print(f"  fragments/neuron:  median={int(np.median(n_frags_a))}  "
          f"mean={n_frags_a.mean():.1f}  max={n_frags_a.max()}")
    print(f"  dominant-fragment mass share:  median={np.median(dom_a):.3f}  "
          f"min={dom_a.min():.3f}")
    print(f"{'='*60}")
    print("\nReading: high single-root % and high dominant share confirm that for\n"
          "soma-anchored neurons the real v117 split structure is 'one trunk +\n"
          "slivers'. The genuinely hard partition cases are the multi-substantial\n"
          "minority (and merge errors / axonal fragments not sampled here).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
