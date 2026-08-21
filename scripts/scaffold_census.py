#!/usr/bin/env python3
"""Scaffold census: can label-free gates select a high-purity trunk scaffold?

Experiment 1 of the scaffold-first (coarse-to-fine) design in
``docs/tree_assembly_algorithm.md`` §5. For a real region world it measures,
for several gate settings, the properties of the fragment set that PASSES the
gates (the candidate verified scaffold):

  - fragment purity      — fraction of scaffold v117 roots mapping to exactly
                           one v1718 neuron (the load-bearing number)
  - synapse purity       — scaffold synapses whose label equals their root's
                           majority label
  - mass coverage        — fraction of ALL synapses sitting on scaffold roots
  - fk exclusion         — fraction of real frankenmerge roots the gates keep
                           OUT of the scaffold

Gates are label-free: skeleton oddness (long-bridge / multi-component,
``treestitch.atomize.oddness_scores``) at loose and strict thresholds,
minimum observation count, and ≤1 nucleus soma near the fragment.

Usage
-----
  python scripts/scaffold_census.py --bbox 950000 930000 780000 1150000 1000000 880000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def count_somata_near(fragments, somas_nm, margin_nm: float = 10_000.0):
    """[F] number of nucleus somata inside each fragment's bbox + margin."""
    import numpy as np

    out = []
    sx, sy, sz = somas_nm[:, 0], somas_nm[:, 1], somas_nm[:, 2]
    for f in fragments:
        v = np.asarray(f.vertices_nm, dtype=np.float64)
        lo = v.min(0) - margin_nm
        hi = v.max(0) + margin_nm
        m = ((sx >= lo[0]) & (sx <= hi[0]) &
             (sy >= lo[1]) & (sy <= hi[1]) &
             (sz >= lo[2]) & (sz <= hi[2]))
        out.append(int(m.sum()))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", type=float, nargs=6, required=True,
                   metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"))
    p.add_argument("--version", type=int, default=1718)
    p.add_argument("--max-synapses", type=int, default=20_000)
    p.add_argument("--min-syn-per-fragment", type=int, default=5)
    p.add_argument("--tile-x-nm", type=float, default=0)
    p.add_argument("--per-tile-limit", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nucleus-cache", default=None,
                   help="npz cache path for the nucleus table")
    args = p.parse_args()

    import numpy as np

    from treestitch.atomize import oddness_scores
    from treestitch.realworld import build_region_world

    lo3, hi3 = tuple(args.bbox[:3]), tuple(args.bbox[3:])
    fragments, region, root_label_map = build_region_world(
        (lo3, hi3), version=args.version, max_synapses=args.max_synapses,
        min_syn_per_fragment=args.min_syn_per_fragment, seed=args.seed,
        tile_x_nm=args.tile_x_nm, per_tile_limit=args.per_tile_limit)

    frag_ids = np.asarray(region.pre_seg_id, dtype=np.int64)
    labels = np.asarray(region.pre_root_id, dtype=np.int64)
    n_obs_total = len(frag_ids)

    # Per-fragment facts
    facts = {}
    for f in fragments:
        rid = int(f.base_root_id)
        m = frag_ids == rid
        labs = root_label_map.get(rid, set())
        facts[rid] = {
            "n_obs": int(m.sum()),
            "pure": len(labs) == 1,
            "is_fk": len(labs) > 1,
            "odd_loose": oddness_scores(
                f, long_edge_factor=4.0, long_edge_min_nm=10_000.0)["is_odd"],
            "odd_strict": oddness_scores(
                f, long_edge_factor=8.0, long_edge_min_nm=20_000.0)["is_odd"],
        }

    # Soma counts (optional — skip gracefully if the nucleus fetch fails)
    soma_counts = None
    try:
        from treestitch.realworld import _load_nucleus_somas
        somas = _load_nucleus_somas(cache_path=args.nucleus_cache)
        somas_nm = np.stack([somas["x_nm"], somas["y_nm"], somas["z_nm"]], axis=1)
        counts = count_somata_near(fragments, somas_nm)
        soma_counts = {int(f.base_root_id): c
                       for f, c in zip(fragments, counts)}
        print(f"nucleus somata loaded: {len(somas_nm)} "
              f"(fragments with ≥1 nearby soma: "
              f"{sum(1 for c in counts if c >= 1)}/{len(fragments)})")
    except Exception as exc:  # network-dependent; census still meaningful
        print(f"nucleus soma gate unavailable ({type(exc).__name__}: {exc}) — skipping")

    n_fk_total = sum(1 for v in facts.values() if v["is_fk"])
    print(f"\nWorld: {len(fragments)} fragments, {n_obs_total} synapses, "
          f"{n_fk_total} frankenmerges")

    def census(name, keep_fn):
        keep = [rid for rid, v in facts.items() if keep_fn(rid, v)]
        if not keep:
            print(f"  {name:34s} — empty scaffold")
            return
        n = len(keep)
        pure = sum(1 for r in keep if facts[r]["pure"])
        mass = sum(facts[r]["n_obs"] for r in keep)
        fk_in = sum(1 for r in keep if facts[r]["is_fk"])
        fk_excluded = ((n_fk_total - fk_in) / n_fk_total
                       if n_fk_total else float("nan"))
        # synapse purity: obs on pure roots + majority share on impure roots
        syn_pure = 0
        for r in keep:
            m = frag_ids == r
            labs = labels[m]
            labs = labs[labs != 0]
            if len(labs) == 0:
                continue
            _, c = np.unique(labs, return_counts=True)
            syn_pure += int(c.max())
        print(f"  {name:34s} frags={n:5d} ({n/len(fragments):5.1%})  "
              f"purity={pure/n:6.3f}  syn_purity={syn_pure/max(mass,1):6.3f}  "
              f"mass={mass/n_obs_total:6.1%}  fk_excluded={fk_excluded:5.1%}")

    print("\nGate census (scaffold = fragments passing the gate):")
    census("all fragments (no gate)", lambda r, v: True)
    census("not odd (loose 4x/10um)", lambda r, v: not v["odd_loose"])
    census("not odd (strict 8x/20um)", lambda r, v: not v["odd_strict"])
    census("n_obs >= 10", lambda r, v: v["n_obs"] >= 10)
    census("not odd strict + n_obs >= 10",
           lambda r, v: not v["odd_strict"] and v["n_obs"] >= 10)
    if soma_counts is not None:
        census("<= 1 soma", lambda r, v: soma_counts.get(r, 0) <= 1)
        census("FULL: strict-odd + n>=10 + <=1 soma",
               lambda r, v: (not v["odd_strict"] and v["n_obs"] >= 10
                             and soma_counts.get(r, 0) <= 1))
    census("LOOSE-FULL: loose-odd + n>=10",
           lambda r, v: not v["odd_loose"] and v["n_obs"] >= 10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
