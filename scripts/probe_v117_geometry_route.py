"""Can we get real L2 geometry for a *stale* v117 root?

With a label-blind population the atoms are v117 roots enumerated from synapses,
not from proofread cells, so we must fetch geometry keyed on the v117 root
itself. The catch: a v117 root that was later edited is no longer a live root in
the current chunkedgraph. Endpoints may or may not serve historical ids.

This distinguishes two classes explicitly:
  current v117 roots -- never edited, so no merge signal
  stale   v117 roots -- merged/split since, i.e. exactly the interesting atoms

and tries three routes on each: /root, /leaves?stop_layer=2, and lvl2_graph
(true adjacency). Errors are reported, never swallowed.

Stale roots are sourced by walking a proofread cell's L2 nodes back to v117,
which guarantees they participated in a real edit.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.data import lineage as L  # noqa: E402
from neuronauts.harness.substrate import (  # noqa: E402
    fetch_l2_coords, load_proofread_table, region_bounds, select_cells,
)


def try_routes(root: int, seg_bounds, token: str) -> dict:
    hdr = L._headers(token)
    base = f"{L.CG_SERVER}/segmentation/api/v1/table/{L.SEG_TABLE}/node/{int(root)}"
    rec: dict = {"root": int(root)}

    try:
        r = requests.get(f"{base}/root", headers=hdr, timeout=60)
        rec["is_current"] = (r.status_code == 200
                             and str(r.json().get("root_id")) == str(root))
        rec["root_status"] = r.status_code
    except Exception as exc:  # noqa: BLE001
        rec["is_current"], rec["root_status"] = False, -1
        rec["root_error"] = f"{type(exc).__name__}: {exc}"

    bstr = "_".join(f"{int(seg_bounds[i][0])}-{int(seg_bounds[i][1])}"
                    for i in range(3))
    try:
        t0 = time.time()
        r = requests.get(f"{base}/leaves", headers=hdr, timeout=120,
                         params={"stop_layer": 2, "bounds": bstr})
        rec["leaves_status"] = r.status_code
        rec["leaves_s"] = time.time() - t0
        if r.status_code == 200:
            ids = np.asarray(r.json().get("leaf_ids", []), dtype=np.uint64)
            rec["n_l2_leaves"] = int(len(ids))
            rec["_l2_ids"] = ids
        else:
            rec["n_l2_leaves"] = 0
            rec["leaves_error"] = r.text[:150]
    except Exception as exc:  # noqa: BLE001
        rec["leaves_status"], rec["n_l2_leaves"] = -1, 0
        rec["leaves_error"] = f"{type(exc).__name__}: {exc}"

    try:
        t0 = time.time()
        r = requests.get(f"{base}/lvl2_graph", headers=hdr, timeout=120,
                         params={"bounds": bstr})
        rec["lvl2graph_status"] = r.status_code
        rec["lvl2graph_s"] = time.time() - t0
        if r.status_code == 200:
            e = np.asarray(r.json().get("edge_graph", []), dtype=np.uint64)
            rec["n_adjacency_edges"] = int(len(e))
        else:
            rec["n_adjacency_edges"] = 0
            rec["lvl2graph_error"] = r.text[:150]
    except Exception as exc:  # noqa: BLE001
        rec["lvl2graph_status"], rec["n_adjacency_edges"] = -1, 0
        rec["lvl2graph_error"] = f"{type(exc).__name__}: {exc}"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1822)
    ap.add_argument("--centre-um", type=float, nargs=3, default=[663, 591, 860])
    ap.add_argument("--side-um", type=float, default=200.0)
    ap.add_argument("--n-seed-cells", type=int, default=2)
    ap.add_argument("--n-test-roots", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/probe_v117_geometry_route.json")
    args = ap.parse_args()

    token = L.DEFAULT_TOKEN
    _lo, _hi, seg = region_bounds(args.centre_um, args.side_um)

    # Source guaranteed-stale v117 roots: walk proofread cells back to v117.
    from caveclient import CAVEclient
    client = CAVEclient("minnie65_public")
    df = load_proofread_table(args.version)
    sel = select_cells(df, args.centre_um, args.side_um, "gold").head(args.n_seed_cells)
    print(f"seeding from {len(sel)} proofread cells", flush=True)

    l2_all = []
    for root in sel["pt_root_id"].to_numpy():
        e = np.asarray(client.chunkedgraph.level2_chunk_graph(int(root), bounds=seg),
                       dtype=np.uint64)
        if e.size:
            l2_all.append(np.unique(e))
    pool = np.unique(np.concatenate(l2_all))
    print(f"  {len(pool):,} L2 nodes", flush=True)

    v117 = L.roots_at(pool[:4000], L.V117_TIMESTAMP, token=token)
    frag_counts = collections.Counter(int(v) for v in v117 if v)
    # Prefer substantial fragments; those are the ones we would actually keep.
    stale_candidates = [r for r, c in frag_counts.most_common() if c >= 5]
    print(f"  {len(frag_counts)} distinct v117 fragments, "
          f"{len(stale_candidates)} with >=5 L2 nodes", flush=True)

    test = stale_candidates[:args.n_test_roots]
    print(f"\nprobing {len(test)} v117 roots ...\n", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(lambda r: try_routes(r, seg, token), test))

    cur = [r for r in recs if r.get("is_current")]
    stale = [r for r in recs if not r.get("is_current")]
    print(f"{'group':<16}{'n':>4}{'leaves ok':>12}{'lvl2graph ok':>15}"
          f"{'med L2':>9}")
    for name, grp in (("current", cur), ("stale", stale)):
        if not grp:
            print(f"{name:<16}{0:>4}{'-':>12}{'-':>15}{'-':>9}")
            continue
        lo = sum(1 for r in grp if r.get("leaves_status") == 200)
        go = sum(1 for r in grp if r.get("lvl2graph_status") == 200)
        med = int(np.median([r.get("n_l2_leaves", 0) for r in grp]))
        print(f"{name:<16}{len(grp):>4}{lo:>7}/{len(grp):<4}"
              f"{go:>10}/{len(grp):<4}{med:>9}")

    for tag in ("leaves", "lvl2graph"):
        codes = collections.Counter(r.get(f"{tag}_status") for r in recs)
        print(f"\n{tag} status: {dict(codes)}")
        errs = [r.get(f"{tag}_error") for r in recs if r.get(f"{tag}_error")]
        for e in list(dict.fromkeys(errs))[:2]:
            print(f"  e.g. {e[:200]}")

    # If leaves works, confirm coordinates resolve for those nodes.
    ids = np.unique(np.concatenate(
        [r["_l2_ids"] for r in recs if r.get("n_l2_leaves", 0) > 0]
        or [np.zeros(0, np.uint64)]))
    if len(ids):
        coords = fetch_l2_coords(ids, token, Path("/tmp/probe_geom_coords.npz"),
                                 verbose=False)
        print(f"\nrep_coord_nm for v117-fragment L2 nodes: "
              f"{len(coords):,}/{len(ids):,} ({len(coords)/len(ids):.1%})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in recs],
        indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
