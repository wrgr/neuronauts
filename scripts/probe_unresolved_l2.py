"""Why do 3.6% of L2 nodes have no v117 root?

Three candidate explanations, with different consequences:

  A. the L2 node was created by a post-v117 edit (a split makes new within-chunk
     components), but the supervoxels underneath did exist at v117
     -> the cable is attributable; assign via supervoxel majority
  B. the underlying supervoxels were unsegmented at v117
     -> genuinely unattributable; must be excluded and declared in the ceiling
  C. transient API failure
     -> a bug on our side; retry

The test drops one level: for L2 nodes that return no v117 root, fetch their
supervoxels and ask *those* for a v117 root. Supervoxel ids are immutable, so
they discriminate A from B directly.

A labelled control group (L2 nodes that DID resolve) is run through the same
path, so we can tell a real effect from a broken query.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.data import lineage as L  # noqa: E402
from neuronauts.harness.substrate import (  # noqa: E402
    fetch_l2_coords, fetch_l2_graphs, fetch_v117_map, region_bounds,
)


def supervoxels_of(l2_id: int, token: str):
    """Supervoxels under one L2 node (stop_layer=1)."""
    url = (f"{L.CG_SERVER}/segmentation/api/v1/table/{L.SEG_TABLE}"
           f"/node/{int(l2_id)}/leaves")
    try:
        r = requests.get(url, headers=L._headers(token),
                         params={"stop_layer": 1}, timeout=120)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}: {r.text[:110]}"
        return np.asarray(r.json().get("leaf_ids", []), dtype=np.uint64), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def analyse(l2_ids, token, workers, label):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = list(ex.map(lambda i: (i,) + supervoxels_of(i, token), l2_ids))

    errs = [g[2] for g in got if g[2]]
    ok = [(i, sv) for i, sv, e in got if sv is not None and len(sv)]
    if not ok:
        print(f"  [{label}] no supervoxels retrieved; errors: {errs[:2]}")
        return {}

    allsv = np.unique(np.concatenate([sv for _, sv in ok]))
    roots = L.roots_at(allsv[:4000], L.V117_TIMESTAMP, token=token)
    sv2v117 = dict(zip(allsv[:4000].tolist(),
                       roots.tolist() if roots is not None else []))

    per_node = []
    for i, sv in ok:
        vals = [sv2v117.get(int(s), 0) for s in sv.tolist() if int(s) in sv2v117]
        if not vals:
            continue
        nz = [v for v in vals if v]
        per_node.append({"l2": int(i), "n_sv": len(sv), "n_checked": len(vals),
                         "frac_sv_with_v117": len(nz) / len(vals),
                         "n_distinct_v117": len(set(nz))})

    if not per_node:
        print(f"  [{label}] no supervoxels were in the sampled mapping window")
        return {}

    frac = np.array([p["frac_sv_with_v117"] for p in per_node])
    nsv = np.array([p["n_sv"] for p in per_node])
    print(f"  [{label}] n={len(per_node)} L2 nodes, "
          f"median {int(np.median(nsv))} supervoxels each")
    print(f"    supervoxels WITH a v117 root: median {np.median(frac):.1%}, "
          f"mean {frac.mean():.1%}")
    print(f"    nodes where >=50% of supervoxels have a v117 root: "
          f"{float((frac >= 0.5).mean()):.1%}")
    if errs:
        print(f"    ({len(errs)} fetch errors, e.g. {errs[0][:90]})")
    return {"n": len(per_node), "median_frac": float(np.median(frac)),
            "mean_frac": float(frac.mean()),
            "frac_nodes_majority_resolved": float((frac >= 0.5).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--centre-um", type=float, nargs=3, default=[663, 591, 860])
    ap.add_argument("--side-um", type=float, default=200.0)
    ap.add_argument("--cache", default="data/substrate/viz_check")
    ap.add_argument("--n-sample", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/probe_unresolved_l2.json")
    args = ap.parse_args()

    token = L.DEFAULT_TOKEN
    cache = Path(args.cache)
    _lo, _hi, seg = region_bounds(args.centre_um, args.side_um)

    import glob
    roots = [int(Path(p).stem) for p in glob.glob(str(cache / "l2_graphs/*.npz"))]
    if not roots:
        raise SystemExit(f"no cached L2 graphs under {cache}; "
                         "run scripts/viz_verify_substrate.py first")
    graphs = fetch_l2_graphs(roots, seg, cache / "l2_graphs", 6, verbose=False)
    pool = np.unique(np.concatenate([np.unique(e) for e in graphs.values()]))
    coords = fetch_l2_coords(pool, token, cache / "l2_coords.npz", verbose=False)
    v117 = fetch_v117_map(pool, token, cache / "l2_v117.npz", verbose=False)

    ids = np.array([i for i in pool.tolist() if i in coords], np.uint64)
    lab = np.array([v117.get(int(i), 0) for i in ids], np.uint64)
    unresolved = ids[lab == 0]
    resolved = ids[lab > 0]
    print(f"L2 nodes: {len(ids):,}  unresolved: {len(unresolved):,} "
          f"({len(unresolved)/len(ids):.1%})\n")

    rng = np.random.default_rng(0)
    us = unresolved[rng.choice(len(unresolved),
                               min(args.n_sample, len(unresolved)), replace=False)]
    rs = resolved[rng.choice(len(resolved),
                             min(args.n_sample, len(resolved)), replace=False)]

    print("dropping to supervoxels and re-asking for a v117 root:")
    out_u = analyse(us, token, args.workers, "unresolved L2")
    out_r = analyse(rs, token, args.workers, "resolved L2 (control)")

    print("\ninterpretation:")
    mf = out_u.get("median_frac")
    if mf is None:
        print("  inconclusive - supervoxel route returned nothing")
    elif mf >= 0.5:
        print("  A: voxels DID exist at v117; these L2 nodes are post-v117")
        print("     creations. The cable is attributable -- assign each node")
        print("     the majority v117 root of its supervoxels.")
    elif mf <= 0.1:
        print("  B: the underlying voxels were unsegmented at v117. Genuinely")
        print("     unattributable -- exclude, and state the ceiling explicitly.")
    else:
        print("  mixed; handle per node by supervoxel majority, excluding the "
              "rest and declaring them.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"n_l2": int(len(ids)), "n_unresolved": int(len(unresolved)),
         "unresolved": out_u, "control": out_r}, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
