"""Enumerate EVERY v117 object in a region, including those with no synapse.

``build_population.py`` enumerates objects *synapse-first*: a v117 root enters
the population by owning a synapse whose centre lies in the region. That rule is
label-blind and cheap, and it silently drops the connective cable — a passing
stretch of neurite with no synapse of its own.

That omission is not marginal. Walking the real level-2 graph of proofread cells
(2026-09-02 probe) found their labelled fragments a median of **2 hops** apart,
and every one of the 297 objects holding the material in between was absent from
the population. The proximity experiments were measuring the distance across a
hole the substrate had made.

This script closes it, still label-blind: read the segmentation volume over the
region, take the distinct **supervoxels**, and map those to v117 roots in one
batched pass. No synapse table, no labels, no ground truth.

Reading with ``agglomerate=True`` would hand back v117 roots directly and is the
obvious first thing to try; it was tried and abandoned, deliberately. That flag
makes CloudVolume resolve every supervoxel to a root *during* the read, so the
cost tracks distinct supervoxels rather than bytes, and a full-width slab of
this cube stalled for minutes with no way to see progress. Splitting the two
steps makes the volume read a pure download and turns the mapping into one
batched job whose rate is known and measured (~5,800 supervoxels/s, from
``results/probe_population_scale.json``).

**Resolution is a correctness parameter, and it was chosen by measurement.** A
downsampled read finds large objects and can drop small ones, and the connective
objects are small (median 5 level-2 nodes, against 4 for a typical population
atom). Recall was measured per size bucket against the known population in a
12 um cube: objects with two or more level-2 nodes are recovered at **100% from
mip 2 through mip 5**; only single-node dust falls off (97-98% at mip 4-5). Mip 4
is the default here because it holds that margin at a fraction of mip 2's cost.
Anything coarser should be re-validated the same way before it is trusted.

The read streams in slabs and keeps only the running id->voxel-count tally, so
memory is bounded by the number of distinct objects rather than by the volume.
Every slab is checkpointed, so an interrupted run resumes where it stopped.

    python scripts/enumerate_region_objects.py --side-um 100 --mip 4
    python scripts/enumerate_region_objects.py --side-um 100 --mip 4 --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuronauts.harness.population import load_population  # noqa: E402

#: The materialization the atoms are defined at. Matches population.npz meta.
V117_TIMESTAMP = 1623399000


def _slabs(lo, hi, axis, step):
    a, b = int(lo[axis]), int(hi[axis])
    for s in range(a, b, step):
        yield s, min(s + step, b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--centre-um", type=float, nargs=3, default=[663, 591, 860])
    ap.add_argument("--side-um", type=float, default=100.0)
    ap.add_argument("--mip", type=int, default=4,
                    help="4 is validated; coarser needs re-validation")
    ap.add_argument("--slab-voxels", type=int, default=40_000_000,
                    help="target voxels per read; sets memory use")
    ap.add_argument("--timestamp", type=int, default=V117_TIMESTAMP)
    ap.add_argument("--out", default=None)
    ap.add_argument("--population",
                    default="data/substrate/c100um/population.npz")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    import caveclient
    from cloudvolume import CloudVolume

    name = f"c{int(args.side_um)}um"
    out = Path(args.out or f"data/substrate/{name}/objects_v117_mip{args.mip}.npz")
    ckpt = out.with_suffix(".partial.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    client = caveclient.CAVEclient("minnie65_public")
    path = client.chunkedgraph.cloudvolume_path
    # agglomerate=False: raw supervoxel ids, a pure download. Roots come later,
    # in one batched pass -- see the module docstring.
    cv = CloudVolume(path, mip=args.mip, use_https=True, progress=False,
                     fill_missing=True, agglomerate=False)
    res = np.asarray(cv.resolution, float)

    centre = np.asarray(args.centre_um, float) * 1000.0
    lo_nm = centre - args.side_um * 500.0
    hi_nm = centre + args.side_um * 500.0
    lo = np.floor(lo_nm / res).astype(int)
    hi = np.ceil(hi_nm / res).astype(int)
    shape = hi - lo
    total = int(np.prod(shape))

    plane = int(shape[0]) * int(shape[1])
    zstep = max(1, min(int(shape[2]), args.slab_voxels // max(plane, 1)))
    slabs = list(_slabs(lo, hi, 2, zstep))

    print(f"region     : {args.side_um:g} um cube @ {args.centre_um} um")
    print(f"volume     : mip {args.mip}, {res.tolist()} nm/voxel, "
          f"{shape.tolist()} voxels = {total:,}")
    print(f"timestamp  : {args.timestamp} (v117)")
    print(f"slabs      : {len(slabs)} x {zstep} z-planes "
          f"(~{plane * zstep:,} voxels each)")

    # Each slab's uniques are written to their own small file and merged once at
    # the end. Carrying a running dict instead would cost a rewrite of the whole
    # tally per slab -- quadratic, and several GB of Python ints at this scale.
    slab_dir = ckpt.with_suffix("")
    slab_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    if args.resume:
        while (slab_dir / f"slab_{done:04d}.npz").exists():
            done += 1
        print(f"resuming   : {done}/{len(slabs)} slabs already on disk")

    t0 = time.time()
    for i, (z0, z1) in enumerate(slabs):
        if i < done:
            continue
        tstart = time.time()
        arr = np.asarray(cv[lo[0]:hi[0], lo[1]:hi[1], z0:z1])[..., 0]
        ids, counts = np.unique(arr, return_counts=True)
        del arr
        keep = ids > 0
        np.savez(slab_dir / f"slab_{i:04d}.npz",
                 sv=ids[keep].astype(np.uint64),
                 n=counts[keep].astype(np.int64))
        el = time.time() - t0
        rate = (i + 1 - done) / max(el, 1e-9)
        print(f"  slab {i+1:>3}/{len(slabs)}  z {z0}-{z1}  "
              f"{time.time()-tstart:5.1f}s  {int(keep.sum()):>9,} supervoxels  "
              f"eta {(len(slabs)-i-1)/max(rate,1e-9)/60:5.1f}m", flush=True)

    # Merge by hash bucket, never all at once. At mip 2 the 100 um cube is 417
    # slabs of ~1.5M supervoxels each -- concatenating them is ~10 GB before the
    # argsort, which is where a first version of this would have died. Bucketing
    # on `sv % N_BUCKETS` bounds memory by the largest bucket and lets each one
    # be deduplicated independently.
    N_BUCKETS = 64
    bdir = slab_dir / "buckets"
    bdir.mkdir(exist_ok=True)
    for i in range(len(slabs)):
        with np.load(slab_dir / f"slab_{i:04d}.npz", allow_pickle=False) as z:
            s, c = z["sv"], z["n"]
        b = (s % np.uint64(N_BUCKETS)).astype(np.int64)
        for k in range(N_BUCKETS):
            m = b == k
            if m.any():
                with open(bdir / f"b{k:02d}.bin", "ab") as fh:
                    fh.write(np.stack([s[m].view(np.int64), c[m]], axis=1)
                             .astype(np.int64).tobytes())
    sv_parts, n_parts = [], []
    for k in range(N_BUCKETS):
        f = bdir / f"b{k:02d}.bin"
        if not f.exists():
            continue
        raw = np.fromfile(f, dtype=np.int64).reshape(-1, 2)
        s, c = raw[:, 0].view(np.uint64), raw[:, 1]
        o = np.argsort(s)
        s, c = s[o], c[o]
        u, st = np.unique(s, return_index=True)
        sv_parts.append(u)
        n_parts.append(np.add.reduceat(c, st))
        f.unlink()
    bdir.rmdir()
    sv = np.concatenate(sv_parts) if sv_parts else np.zeros(0, np.uint64)
    svc = np.concatenate(n_parts) if n_parts else np.zeros(0, np.int64)
    del sv_parts, n_parts
    print(f"\nvolume read done: {len(sv):,} distinct supervoxels "
          f"({(time.time()-t0)/60:.1f} min)")

    # --- supervoxel -> v117 root, batched ---------------------------------
    from datetime import datetime, timezone
    ts = datetime.fromtimestamp(args.timestamp, tz=timezone.utc)
    t1 = time.time()
    roots = np.zeros(len(sv), np.uint64)
    B = 50_000
    for i in range(0, len(sv), B):
        chunk = sv[i:i + B]
        roots[i:i + B] = np.asarray(
            client.chunkedgraph.get_roots(chunk, timestamp=ts), np.uint64)
        el = time.time() - t1
        done = min(i + B, len(sv))
        print(f"  roots {done:>9,}/{len(sv):,}  "
              f"{done/max(el,1e-9):,.0f} sv/s  "
              f"eta {(len(sv)-done)/max(done/max(el,1e-9),1e-9)/60:5.1f}m",
              flush=True)
    ok = roots > 0
    print(f"resolved {int(ok.sum()):,}/{len(sv):,} supervoxels "
          f"({ok.mean():.1%}) in {(time.time()-t1)/60:.1f} min")

    # Keep the supervoxel-level map, not just the object aggregate. Geometry for
    # the newly-found objects is then one batched ``get_roots(stop_layer=2)``
    # over their supervoxels, instead of one lvl2_graph request per object --
    # the difference between minutes and a per-object fetch of the same scale as
    # the original 279,075-atom build.
    svmap = out.with_name(out.stem + "_svmap.npz")
    np.savez_compressed(svmap, sv=sv[ok], root=roots[ok],
                        n_voxels=svc[ok])
    print(f"wrote {svmap} ({int(ok.sum()):,} supervoxel -> v117 root)")

    # voxels per v117 object, summed over its supervoxels
    order0 = np.argsort(roots[ok])
    r_s = roots[ok][order0]
    c_s = svc[ok][order0]
    oid, starts = np.unique(r_s, return_index=True)
    cnt = np.add.reduceat(c_s, starts) if len(oid) else np.zeros(0, np.int64)
    order = np.argsort(-cnt)
    oid, cnt = oid[order], cnt[order]
    vox_nm3 = float(np.prod(res))

    pop_path = Path(args.population)
    known = np.zeros(len(oid), bool)
    n_pop = 0
    if pop_path.exists():
        pop = load_population(pop_path)
        n_pop = int(len(pop.atom_id))
        known = np.isin(oid, pop.atom_id)

    meta = {
        "centre_um": list(args.centre_um), "side_um": args.side_um,
        "mip": args.mip, "resolution_nm": res.tolist(),
        "timestamp": args.timestamp, "base_version": 117,
        "voxels_read": total, "voxel_nm3": vox_nm3,
        "n_supervoxels": int(len(sv)),
        "n_supervoxels_resolved": int(ok.sum()),
        "frac_supervoxels_resolved": round(float(ok.mean()), 4),
        "selection": "label-blind: every v117 object with a voxel in the region, "
                     "synapse-free objects included",
        "n_objects": int(len(oid)),
        "n_in_population": int(known.sum()),
        "n_new": int((~known).sum()),
        "population": str(pop_path), "n_population": n_pop,
        "recall_validation": "objects with >=2 L2 nodes recovered at 100% from "
                             "mip 2 through mip 5 on a 12 um cube; single-node "
                             "objects 97-98% at mip 4-5",
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    np.savez_compressed(out, object_id=oid, n_voxels=cnt, in_population=known,
                        meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))
    for f in sorted(slab_dir.glob("slab_*.npz")):
        f.unlink()
    slab_dir.rmdir()

    vol = cnt * vox_nm3 / 1e9
    print(f"\n{'='*64}")
    print(f"objects found        : {len(oid):,}")
    if n_pop:
        print(f"already in population: {int(known.sum()):,} of {n_pop:,}")
        print(f"NEW (no synapse here): {int((~known).sum()):,}")
        for t in (0.05, 0.1, 0.5, 1.0):
            print(f"   new, >= {t:>4} um^3   : {int((vol[~known] >= t).sum()):,}")
        print(f"new share of segmented volume: "
              f"{vol[~known].sum() / vol.sum():.1%}")
    print(f"wrote {out}  ({meta['elapsed_min']:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
