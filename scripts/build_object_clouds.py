"""Object point clouds straight from the segmentation volume — no skeletons.

The harness's geometry path is skeleton-first: fetch ``lvl2_graph`` per object,
contract it, extract endpoints, and measure between tips. Three experiments were
built on that surface and EXP-070 showed the tip metric was the wrong one;
EXP-071 then showed the objects that matter were never in the population at all.

This takes the other route. The segmentation volume already *is* the object
partition: read it once, and every object's extent falls out with no graph, no
skeleton and no per-object request. Two things recommend it here beyond the cost:

**Anisotropy.** The level-2 chunk is 2048 x 2048 x 20480 nm — 10:1 longer in z —
so a hop count inherits a direction-dependent scale. A mip-5 voxel is
256 x 256 x 160 nm, 1.6:1. Measuring between voxel-derived clouds is far closer
to measuring in real space than measuring between chunk-derived skeleton nodes.

**It covers the new objects for free.** ``enumerate_region_objects.py`` already
resolves every supervoxel in the region to its v117 root; this pass reuses that
map, so the objects with no synapse — the ones EXP-071 found holding all the
connective cable — get geometry on exactly the same footing as the rest.

One point is emitted per supervoxel (its voxel centroid), so an object's cloud is
as dense as its supervoxel decomposition. That is coarser than an L2 node cloud
and deliberately so: the question these clouds answer is how close two objects
come to each other, and for that the sampling only has to be fine relative to the
gaps being measured.

The per-point voxel count is saved as ``n_voxels_per_point`` -- not ``n_voxels``,
which ``enumerate_region_objects.py`` already uses for its per-OBJECT voxel
count in a different file; the two must not be confused. Clouds built before
this key was renamed still load (``exp072_object_proposal.load_clouds`` reads
either), and are otherwise unaffected.

Clouds built before the half-voxel centring fix below are offset from the
independent objgeom coordinates by half a voxel -- 128/128/80 nm at mip 5,
16/16/20 nm at mip 2 -- toward the voxel corner rather than its centre. That
offset is uniform across every point, so it is harmless for cloud-vs-cloud
distance measurements (EXP-072/073) and only matters when comparing against
another coordinate source.

    python scripts/build_object_clouds.py --side-um 100 --mip 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

V117_TIMESTAMP = 1623399000


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centre-um", type=float, nargs=3, default=[663, 591, 860])
    ap.add_argument("--side-um", type=float, default=100.0)
    ap.add_argument("--mip", type=int, default=5)
    ap.add_argument("--slab-voxels", type=int, default=40_000_000)
    ap.add_argument("--svmap", default=None,
                    help="supervoxel -> v117 root, from enumerate_region_objects")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import caveclient
    from cloudvolume import CloudVolume

    name = f"c{int(args.side_um)}um"
    svmap = Path(args.svmap or
                 f"data/substrate/{name}/objects_v117_mip{args.mip}_svmap.npz")
    out = Path(args.out or f"data/substrate/{name}/object_clouds_mip{args.mip}.npz")
    if not svmap.exists():
        raise SystemExit(f"missing {svmap}; run enumerate_region_objects.py first")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.with_suffix("")
    work.mkdir(parents=True, exist_ok=True)

    path = caveclient.CAVEclient("minnie65_public").chunkedgraph.cloudvolume_path
    cv = CloudVolume(path, mip=args.mip, use_https=True, progress=False,
                     fill_missing=True, agglomerate=False)
    res = np.asarray(cv.resolution, float)
    centre = np.asarray(args.centre_um, float) * 1000.0
    lo = np.floor((centre - args.side_um * 500.0) / res).astype(int)
    hi = np.ceil((centre + args.side_um * 500.0) / res).astype(int)
    shape = hi - lo

    plane = int(shape[0]) * int(shape[1])
    zstep = max(1, min(int(shape[2]), args.slab_voxels // max(plane, 1)))
    zs = list(range(int(lo[2]), int(hi[2]), zstep))
    print(f"volume : mip {args.mip}, {res.tolist()} nm, {shape.tolist()} voxels")
    print(f"slabs  : {len(zs)} x {zstep} z-planes")

    # --- pass over the volume, accumulating a coordinate sum per supervoxel ---
    t0 = time.time()
    for i, z0 in enumerate(zs):
        f = work / f"cent_{i:04d}.npz"
        if f.exists():
            print(f"  slab {i+1}/{len(zs)}: cached"); continue
        z1 = min(z0 + zstep, int(hi[2]))
        ts = time.time()
        arr = np.asarray(cv[lo[0]:hi[0], lo[1]:hi[1], z0:z1])[..., 0]
        flat = arr.reshape(-1)
        nz = flat > 0
        sv, inv = np.unique(flat[nz], return_inverse=True)
        gx, gy, gz = np.meshgrid(np.arange(shape[0], dtype=np.int64),
                                 np.arange(shape[1], dtype=np.int64),
                                 np.arange(z1 - z0, dtype=np.int64),
                                 indexing="ij")
        n = np.bincount(inv, minlength=len(sv))
        sx = np.bincount(inv, weights=gx.reshape(-1)[nz], minlength=len(sv))
        sy = np.bincount(inv, weights=gy.reshape(-1)[nz], minlength=len(sv))
        sz = np.bincount(inv, weights=gz.reshape(-1)[nz] + (z0 - int(lo[2])),
                         minlength=len(sv))
        del arr, flat, nz, inv, gx, gy, gz
        np.savez(f, sv=sv.astype(np.uint64), n=n.astype(np.int64),
                 sx=sx, sy=sy, sz=sz)
        print(f"  slab {i+1}/{len(zs)}  {time.time()-ts:5.1f}s  "
              f"{len(sv):,} supervoxels", flush=True)

    # --- merge slabs: one centroid per supervoxel -----------------------------
    SV, N, SX, SY, SZ = [], [], [], [], []
    for i in range(len(zs)):
        with np.load(work / f"cent_{i:04d}.npz", allow_pickle=False) as z:
            SV.append(z["sv"]); N.append(z["n"])
            SX.append(z["sx"]); SY.append(z["sy"]); SZ.append(z["sz"])
    sv = np.concatenate(SV); n = np.concatenate(N)
    sx, sy, sz = np.concatenate(SX), np.concatenate(SY), np.concatenate(SZ)
    del SV, N, SX, SY, SZ
    o = np.argsort(sv)
    sv, n, sx, sy, sz = sv[o], n[o], sx[o], sy[o], sz[o]
    usv, starts = np.unique(sv, return_index=True)
    n = np.add.reduceat(n, starts)
    sx = np.add.reduceat(sx, starts)
    sy = np.add.reduceat(sy, starts)
    sz = np.add.reduceat(sz, starts)
    print(f"\n{len(usv):,} supervoxels with a centroid "
          f"({(time.time()-t0)/60:.1f} min)")

    # --- join to the v117 root of each supervoxel -----------------------------
    with np.load(svmap, allow_pickle=False) as z:
        m_sv, m_root = z["sv"], z["root"]
    order = np.argsort(m_sv)
    m_sv, m_root = m_sv[order], m_root[order]
    j = np.clip(np.searchsorted(m_sv, usv), 0, max(len(m_sv) - 1, 0))
    hit = m_sv[j] == usv
    root = np.where(hit, m_root[j], np.uint64(0))
    print(f"joined to v117 roots: {int(hit.sum()):,}/{len(usv):,} "
          f"({hit.mean():.1%})")

    keep = hit & (n > 0)
    root = root[keep]
    # +0.5: sx/n etc. are a mean voxel-grid INDEX, so without the half-voxel
    # shift the result lands on a voxel corner, not its centre.
    pts_nm = np.stack([
        (sx[keep] / n[keep] + lo[0] + 0.5) * res[0],
        (sy[keep] / n[keep] + lo[1] + 0.5) * res[1],
        (sz[keep] / n[keep] + lo[2] + 0.5) * res[2]], axis=1).astype(np.float32)
    nvox = n[keep].astype(np.int64)

    # --- CSR by object --------------------------------------------------------
    o2 = np.argsort(root, kind="stable")
    root, pts_nm, nvox = root[o2], pts_nm[o2], nvox[o2]
    obj, starts = np.unique(root, return_index=True)
    ptr = np.append(starts, len(root)).astype(np.int64)

    meta = {
        "centre_um": list(args.centre_um), "side_um": args.side_um,
        "mip": args.mip, "resolution_nm": res.tolist(),
        "timestamp": V117_TIMESTAMP, "base_version": 117,
        "point": "one per supervoxel, its voxel centroid in nm; "
                 "n_voxels_per_point holds that supervoxel's voxel count "
                 "(per POINT, not per object -- see "
                 "enumerate_region_objects.py's per-object n_voxels)",
        "n_objects": int(len(obj)), "n_points": int(len(pts_nm)),
        "svmap": str(svmap),
        "note": "label-blind; synapse-free objects included. Sampling is the "
                "supervoxel decomposition, coarser than an L2 node cloud and "
                "chosen because the question is object-to-object closest "
                "approach, not skeleton structure. A mip-5 voxel is "
                "256x256x160 nm (1.6:1) against the L2 chunk's 10:1, so these "
                "clouds are less direction-distorted than the skeleton path.",
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    np.savez_compressed(out, object_id=obj, node_ptr=ptr, pos_nm=pts_nm,
                        n_voxels_per_point=nvox,
                        meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
    for f in sorted(work.glob("cent_*.npz")):
        f.unlink()
    work.rmdir()

    per = np.diff(ptr)
    print(f"\n{'='*60}")
    print(f"objects : {len(obj):,}")
    print(f"points  : {len(pts_nm):,}  (median {np.median(per):.0f}/object, "
          f"p90 {np.percentile(per, 90):.0f})")
    print(f"wrote {out}  ({meta['elapsed_min']:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
