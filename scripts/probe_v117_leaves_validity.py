"""Diagnose the EXP-053B L2 coverage failure (only 27.8% of v117 roots covered).

Hypothesis under test: ``_l2_nodes_with_coords`` asks the chunkedgraph for
``node/{v117_root}/leaves`` at the *current* graph state. v117 root ids are
historical. A v117 root that was later edited is no longer a live node, so the
call fails -- and the call site swallows every exception and returns an empty
array, which the benchmark then reports as "no L2 geometry".

If true, coverage is biased *against* exactly the roots that carry the merge
signal, because a true merge pair is by definition a pair of v117 roots that
were subsequently joined. That would make the 27.8% coverage number an artifact
of our own call, not a property of the L2 substrate.

This probe does not swallow errors: it records HTTP status and response body
for each root, and splits roots by whether they are still current.

Roots are drawn from the offline v117 box cache. Read-only apart from stdout
and the JSON report.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.data import lineage as L  # noqa: E402

SEG_VOXEL = np.asarray([8.0, 8.0, 40.0])
# Cached box points are box-local coordinates in (32, 32, 40) nm voxels: for a
# 40 um box the observed extent is ~1276 x 1274 x 1022, and 1276*32 ~= 40 um.
BOX_VOXEL_NM = np.asarray([32.0, 32.0, 40.0])


def sample_roots(box_dir: str, n: int, min_syn: int, seed: int):
    """Sample v117 roots from one cached box, stratified by synapse count.

    Returns centroids in **absolute nm**, converting from the box-local voxel
    frame the cache stores.
    """
    path = sorted(glob.glob(os.path.join(box_dir, "*.npz")))[0]
    box_hash = os.path.basename(path)[:-4]
    index = json.load(open(os.path.join(box_dir, "index.json")))
    rec = next(r for r in index if r["box_hash"] == box_hash)
    origin_nm = np.asarray(rec["center_nm"], dtype=np.float64) - \
        (rec["side_um"] * 1000.0) / 2.0

    z = np.load(path)
    counts: collections.Counter = collections.Counter()
    pts: dict[int, list] = collections.defaultdict(list)
    for side in ("pre", "post"):
        ids = z[f"{side}_root_id"]
        xyz = np.asarray(z[f"{side}_pt"], dtype=np.float64) * BOX_VOXEL_NM + origin_nm
        for i, p in zip(ids.tolist(), xyz):
            if i:
                counts[i] += 1
                if len(pts[i]) < 8:
                    pts[i].append(p)
    eligible = [(r, c) for r, c in counts.items() if c >= min_syn]
    eligible.sort(key=lambda kv: -kv[1])
    rng = np.random.default_rng(seed)
    if len(eligible) > n:
        idx = rng.choice(len(eligible), n, replace=False)
        eligible = [eligible[i] for i in sorted(idx)]
    centroid = {r: np.mean(np.asarray(pts[r], dtype=np.float64), axis=0)
                for r, _ in eligible}
    return eligible, centroid, os.path.basename(path)


def probe_root(root: int, count: int, centre, token: str, halo_nm: float):
    """Return a record with explicit status for bounded + unbounded leaves."""
    hdr = L._headers(token)
    base = f"{L.CG_SERVER}/segmentation/api/v1/table/{L.SEG_TABLE}/node/{int(root)}"
    rec = {"root": int(root), "n_synapses": int(count)}

    # Is this v117 root still a live root in the current graph?
    try:
        r = requests.get(f"{base}/root", headers=hdr, timeout=60)
        rec["root_status"] = r.status_code
        rec["is_current_root"] = (r.status_code == 200
                                  and str(r.json().get("root_id")) == str(root))
        if r.status_code != 200:
            rec["root_error"] = r.text[:180]
    except Exception as exc:  # noqa: BLE001
        rec["root_status"] = -1
        rec["root_error"] = f"{type(exc).__name__}: {exc}"

    lower = np.asarray(centre) - halo_nm
    upper = np.asarray(centre) + halo_nm
    bounds = "_".join(f"{int(lower[i] / SEG_VOXEL[i])}-{int(upper[i] / SEG_VOXEL[i])}"
                      for i in range(3))

    rec["bounds"] = bounds
    leaf_ids: dict[str, np.ndarray] = {}
    for tag, params in (("bounded", {"stop_layer": 2, "bounds": bounds}),
                        ("unbounded", {"stop_layer": 2})):
        try:
            r = requests.get(f"{base}/leaves", headers=hdr, params=params, timeout=120)
            rec[f"{tag}_status"] = r.status_code
            if r.status_code == 200:
                ids = np.asarray(r.json().get("leaf_ids", []), dtype=np.uint64)
                leaf_ids[tag] = ids
                rec[f"{tag}_n_l2"] = int(len(ids))
            else:
                rec[f"{tag}_n_l2"] = 0
                rec[f"{tag}_error"] = r.text[:180]
        except Exception as exc:  # noqa: BLE001
            rec[f"{tag}_status"] = -1
            rec[f"{tag}_n_l2"] = 0
            rec[f"{tag}_error"] = f"{type(exc).__name__}: {exc}"

    # Ground-truth check on the coordinate transform: the synapse centroid we
    # derived from the cache should sit inside this root's real L2 point cloud.
    # If the transform is wrong, this distance blows up and every bounds box we
    # build is in the wrong place.
    ids = leaf_ids.get("unbounded")
    if ids is not None and len(ids):
        sub = ids[:400].tolist()
        try:
            url = f"{L.L2_CACHE_SERVER}/l2cache/api/v1/table/{L.L2_TABLE}/attributes"
            r = requests.post(url, headers={**hdr, "Content-Type": "application/json"},
                              json={"l2_ids": sub, "attribute_names": ["rep_coord_nm"]},
                              timeout=120)
            if r.status_code == 200:
                pts = np.asarray([v["rep_coord_nm"] for v in r.json().values()
                                  if v.get("rep_coord_nm")], dtype=np.float64)
                if len(pts):
                    d = np.linalg.norm(pts - np.asarray(centre), axis=1)
                    rec["dist_centroid_to_nearest_l2_nm"] = float(d.min())
                    rec["l2_cloud_centre_nm"] = pts.mean(0).round(0).tolist()
        except Exception:
            pass
    rec["synapse_centroid_nm"] = list(np.asarray(centre).round(0))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-dir", default="data/boxes_v117")
    ap.add_argument("--n-roots", type=int, default=40)
    ap.add_argument("--min-syn", type=int, default=5)
    ap.add_argument("--halo-nm", type=float, default=25_000.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/probe_v117_leaves_validity.json")
    args = ap.parse_args()

    token = L.DEFAULT_TOKEN
    if not token:
        raise SystemExit("no CAVE token")

    eligible, centroid, box = sample_roots(args.box_dir, args.n_roots,
                                           args.min_syn, args.seed)
    print(f"box={box}  probing {len(eligible)} v117 roots "
          f"(>= {args.min_syn} synapses)\n", flush=True)

    def work(item):
        r, c = item
        return probe_root(r, c, centroid[r], token, args.halo_nm)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(work, eligible))

    n = len(recs)
    current = [r for r in recs if r.get("is_current_root")]
    stale = [r for r in recs if not r.get("is_current_root")]

    def cov(rs, tag):
        if not rs:
            return 0.0
        return sum(1 for r in rs if r.get(f"{tag}_n_l2", 0) >= 2) / len(rs)

    print("=" * 68)
    print(f"roots probed                : {n}")
    print(f"still a current root        : {len(current)} ({len(current)/n:.1%})")
    print(f"stale (edited since v117)   : {len(stale)} ({len(stale)/n:.1%})")
    print()
    print(f"{'group':<22}{'n':>5}{'bounded>=2':>13}{'unbounded>=2':>15}")
    print(f"{'all':<22}{n:>5}{cov(recs,'bounded'):>12.1%}{cov(recs,'unbounded'):>15.1%}")
    print(f"{'current roots':<22}{len(current):>5}{cov(current,'bounded'):>12.1%}"
          f"{cov(current,'unbounded'):>15.1%}")
    print(f"{'stale roots':<22}{len(stale):>5}{cov(stale,'bounded'):>12.1%}"
          f"{cov(stale,'unbounded'):>15.1%}")

    codes = collections.Counter(r.get("bounded_status") for r in recs)
    print(f"\nbounded leaves HTTP status : {dict(codes)}")
    codes_u = collections.Counter(r.get("unbounded_status") for r in recs)
    print(f"unbounded leaves HTTP status: {dict(codes_u)}")

    errs = [r.get("bounded_error") for r in recs if r.get("bounded_error")]
    if errs:
        print("\nsample bounded errors:")
        for e in list(dict.fromkeys(errs))[:3]:
            print(f"  {e}")

    zero = [r for r in recs if r.get("bounded_status") == 200
            and r.get("bounded_n_l2", 0) < 2]
    print(f"\nHTTP 200 but <2 L2 in bounds: {len(zero)} "
          f"({len(zero)/n:.1%}) -- genuine geometry misses, not errors")

    # Transform sanity: synapse centroid should land inside the root's L2 cloud.
    d = np.asarray([r["dist_centroid_to_nearest_l2_nm"] for r in recs
                    if "dist_centroid_to_nearest_l2_nm" in r])
    if len(d):
        print(f"\ncoordinate-transform check ({len(d)} roots):")
        print(f"  synapse centroid -> nearest L2 node, nm: "
              f"median={np.median(d):,.0f}  p90={np.percentile(d, 90):,.0f}  "
              f"max={d.max():,.0f}")
        print("  (should be within a few um if the box-local -> nm transform "
              "is right)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"box": box, "config": vars(args), "records": recs}, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
