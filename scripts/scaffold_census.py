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


def mst_odd(points_nm, *, long_edge_factor: float = 4.0,
            long_edge_min_nm: float = 10_000.0, knn: int = 6) -> bool:
    """Oddness for point-cloud fragments: long-bridge test on the cloud's MST.

    The k-NN graphs of ``_cloud_fragment`` are the wrong substrate for edge-
    length gates (their long edges are sampling artifacts); the MST recovers
    the trunk-vs-bridge structure the L2-skeleton gate relies on."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import cKDTree

    pts = np.asarray(points_nm, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return False
    k = min(knn + 1, n)
    d, nbr = cKDTree(pts).query(pts, k=k)
    rows = np.repeat(np.arange(n), k - 1)
    cols = nbr[:, 1:].ravel()
    vals = d[:, 1:].ravel()
    mst = minimum_spanning_tree(
        coo_matrix((vals, (rows, cols)), shape=(n, n))).tocoo()
    lens = mst.data
    if len(lens) == 0:
        return True          # k-NN graph disconnected — suspicious by itself
    med = float(np.median(lens))
    thresh = max(long_edge_min_nm, long_edge_factor * med)
    # a disconnected k-NN graph (MST is a forest) is also odd
    return bool((lens > thresh).any() or len(lens) < n - 1)


def somata_contained(fragments, region_bbox, *, version_ts, token,
                     nucleus_cache):
    """{v117_root: n_somata} by LINEAGE CONTAINMENT: a soma belongs to a
    fragment iff its nucleus supervoxel resolves to that v117 root.  Exact —
    unlike any proximity test, which over-counts for large arbors."""
    import numpy as np

    from neuronauts.data import lineage as L
    from treestitch.realworld import _load_nucleus_somas

    somas = _load_nucleus_somas(cache_path=nucleus_cache)
    (x0, y0, z0), (x1, y1, z1) = region_bbox
    pad = 50_000.0
    m = ((somas["x_nm"] >= x0 - pad) & (somas["x_nm"] < x1 + pad) &
         (somas["y_nm"] >= y0 - pad) & (somas["y_nm"] < y1 + pad) &
         (somas["z_nm"] >= z0 - pad) & (somas["z_nm"] < z1 + pad))
    sv = somas["sv"][m].astype(np.uint64)
    sv = sv[sv > 0]
    if len(sv) == 0:
        return {}
    roots = L.roots_at(sv, version_ts, token=token)
    if roots is None:
        raise RuntimeError("roots_at failed for soma supervoxels")
    counts: dict[int, int] = {}
    for r in roots[roots > 0]:
        counts[int(r)] = counts.get(int(r), 0) + 1
    return counts


def count_somata_near(fragments, somas_nm, radius_nm: float = 5_000.0,
                      max_verts: int = 300):
    """[F] number of nucleus somata within ``radius_nm`` of each fragment's
    skeleton vertices.

    Distance-to-skeleton, NOT bounding box: a long axon's bbox covers hundreds
    of unrelated somata, which made the bbox version flag every fragment."""
    import numpy as np
    from scipy.spatial import cKDTree

    tree = cKDTree(somas_nm)
    out = []
    rng = np.random.default_rng(0)
    for f in fragments:
        v = np.asarray(f.vertices_nm, dtype=np.float64)
        if len(v) > max_verts:
            v = v[rng.choice(len(v), max_verts, replace=False)]
        hits: set[int] = set()
        for lst in tree.query_ball_point(v, r=radius_nm):
            hits.update(lst)
        out.append(len(hits))
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
    p.add_argument("--l2-substrate", action="store_true",
                   help="observations = L2 nodes (mass ∝ arbor; soma-seeded "
                        "neurons only) instead of subsampled synapses — the "
                        "honest substrate for the mass-coverage claim")
    p.add_argument("--l2-cache", default=None,
                   help="npz cache path for the L2-substrate world arrays")
    p.add_argument("--max-neurons", type=int, default=0,
                   help="(l2 substrate) cap seed neurons; 0 = all")
    args = p.parse_args()

    import numpy as np

    from treestitch.atomize import oddness_scores
    from treestitch.realworld import build_region_world, build_region_world_l2

    lo3, hi3 = tuple(args.bbox[:3]), tuple(args.bbox[3:])
    if args.l2_substrate:
        fragments, region, root_label_map = build_region_world_l2(
            (lo3, hi3), version=args.version, seed=args.seed,
            max_neurons=args.max_neurons,
            nucleus_cache_path=args.nucleus_cache,
            cache_path=args.l2_cache)
    else:
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
        }
        if args.l2_substrate:
            # cloud fragments: MST-based long-bridge gate (k-NN edge stats
            # flag everything — the substrate lesson from census v2)
            facts[rid]["odd_loose"] = mst_odd(
                f.vertices_nm, long_edge_factor=4.0, long_edge_min_nm=10_000.0)
            facts[rid]["odd_strict"] = mst_odd(
                f.vertices_nm, long_edge_factor=8.0, long_edge_min_nm=20_000.0)
        else:
            facts[rid]["odd_loose"] = oddness_scores(
                f, long_edge_factor=4.0, long_edge_min_nm=10_000.0)["is_odd"]
            facts[rid]["odd_strict"] = oddness_scores(
                f, long_edge_factor=8.0, long_edge_min_nm=20_000.0)["is_odd"]

    # Soma counts by lineage containment (exact; skip gracefully on failure)
    soma_counts = None
    try:
        from neuronauts.data import lineage as L
        from neuronauts.data.loaders import DEFAULT_TOKEN
        soma_counts = somata_contained(
            fragments, (lo3, hi3), version_ts=L.V117_TIMESTAMP,
            token=DEFAULT_TOKEN, nucleus_cache=args.nucleus_cache)
        n_with = sum(1 for f in fragments
                     if soma_counts.get(int(f.base_root_id), 0) >= 1)
        n_multi = sum(1 for f in fragments
                      if soma_counts.get(int(f.base_root_id), 0) >= 2)
        print(f"soma containment (lineage): {n_with}/{len(fragments)} fragments "
              f"contain ≥1 soma, {n_multi} contain ≥2")
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
        census("<= 1 contained soma", lambda r, v: soma_counts.get(r, 0) <= 1)
        census("FULL: strict-odd + n>=10 + <=1 soma",
               lambda r, v: (not v["odd_strict"] and v["n_obs"] >= 10
                             and soma_counts.get(r, 0) <= 1))
        census("FULL-loose: loose-odd + <=1 soma",
               lambda r, v: (not v["odd_loose"]
                             and soma_counts.get(r, 0) <= 1))
    census("LOOSE-FULL: loose-odd + n>=10",
           lambda r, v: not v["odd_loose"] and v["n_obs"] >= 10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
