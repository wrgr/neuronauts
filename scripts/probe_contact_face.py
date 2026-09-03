"""Contact-face descriptors: not how NEAR a candidate is, but how much of it
meets the seed.

EXP-077 named the residual difficulty precisely: with correct object identity
the median true gap is 32 nm, 56 of 66 partners simply touch their seed, and
"many candidates touch the seed at a single voxel, and distance cannot order a
tie". ``gap_nm`` is a minimum over voxel pairs, so a glancing brush and a full
severed cross-section score the same 32 nm.

A severed process meets its continuation across a whole cut FACE -- an area of
order pi * caliber^2. A passing process touches at a point. That difference is
an area, and area is exactly the quantity the eroded substrate destroyed: with
one centroid per mip-5-visible supervoxel, no object has a face at all, so this
descriptor could never have been measured before the identity fix.

Computed on the same box the panel used (same centre rule, same mip 2,
``agglomerate=True`` at the v117 timestamp), so rows join to the panel by
object id. Per candidate:

  n_touch1/2   its voxels inside a 1- and 2-voxel dilation of the seed
               (26-connectivity; 1 iteration reaches 32 nm in x/y, 40 in z)
  touch_rms    root-mean-square spread of that contact patch about its
               centroid, nm -- a disc of radius R gives R/sqrt(2)
  touch_cc     connected components of the contact patch: a real cut face is
               one coherent patch, a process running alongside makes several

The same box also yields the descriptor the panel could not carry. A panel's
``along`` is |cos| between the seed's local axis and the direction to the
candidate, so it is UNSIGNED: a candidate lying back along the parent, on the
soma side of the cut, scores exactly as high as its true continuation lying
ahead of it. Orienting that axis needs the direction the seed's cable was
heading when it ran out, and a local axis fitted to a fifth of an object's
voxels is not that object's axis -- which is why EXP-077 found collinearity
useless on the eroded build and useful on the correct one.

  oproj        signed cos between the seed's outward heading and the direction
               from the contact to the candidate's local centroid
  d_ahead      how far along that heading the candidate sits, nm (signed)

    python scripts/probe_contact_face.py --limit 4
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

R = Path(__file__).resolve().parents[1]
V117_TS = datetime.fromtimestamp(1623399000, tz=timezone.utc)
HALF_NM = 4000.0
LOCAL_NM = 1500.0          # the panel builder's local window, reused verbatim


def axis_of(P):
    if len(P) < 3:
        return None
    return np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]


def outward_heading(S, ctr):
    """Which way the seed's cable was heading when it ran out at ``ctr``.

    The principal axis of the seed's voxels near the contact, oriented to point
    from their centroid toward the contact -- i.e. outward, away from the body
    of the seed, which is the direction a continuation must lie in.
    """
    loc = S[np.linalg.norm(S - ctr, axis=1) < LOCAL_NM]
    a = axis_of(loc)
    if a is None:
        return None
    return a if float(a @ (ctr - loc.mean(0))) >= 0 else -a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = every panel")
    ap.add_argument("--out", default=str(R / "data/external/contact_face"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    con = json.load(open(R / "data/external/exp079_contacts.json"))
    import caveclient
    from cloudvolume import CloudVolume

    cl = caveclient.CAVEclient("minnie65_public")
    cv = CloudVolume(cl.chunkedgraph.cloudvolume_path, mip=2, use_https=True,
                     progress=False, fill_missing=True, agglomerate=True,
                     timestamp=V117_TS)
    res = np.asarray(cv.resolution, float)

    files = sorted(glob.glob(str(R / "data/external/panels/*.npz")))
    if args.limit:
        files = files[: args.limit]
    for f in files:
        key = Path(f).stem.split("_")[1]
        dest = out / f"face_{key}.npz"
        if dest.exists():
            print(f"  {key}: exists", flush=True)
            continue
        c = con.get(key)
        if c is None:
            print(f"  {key}: no contact record", flush=True)
            continue
        t0 = time.time()
        try:
            seed = int(c["seed"])
            ctr = np.asarray(c["ctr"], float)
            lo = np.floor((ctr - HALF_NM) / res).astype(int)
            hi = np.ceil((ctr + HALF_NM) / res).astype(int)
            vol = np.asarray(cv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])[..., 0]
            sm = vol == seed
            if sm.sum() < 10:
                print(f"  {key}: seed absent from box", flush=True)
                continue
            st = np.ones((3, 3, 3), bool)          # 26-connectivity
            rows = {}
            for it, tag in ((1, 1), (2, 2)):
                dil = ndimage.binary_dilation(sm, structure=st, iterations=it)
                shell = dil & ~sm & (vol > 0)
                lab = vol[shell]
                if tag == 1:
                    # coordinates of every contact voxel, for patch geometry
                    idx = np.nonzero(shell)
                    pts = (np.stack(idx, 1) + lo) * res
                    order = np.argsort(lab, kind="stable")
                    ls, ps = lab[order], pts[order]
                    uq, starts = np.unique(ls, return_index=True)
                    stops = np.append(starts[1:], len(ls))
                    for o, a_, b_ in zip(uq.tolist(), starts, stops):
                        P = ps[a_:b_]
                        rms = float(np.sqrt(np.mean(np.sum(
                            (P - P.mean(0)) ** 2, axis=1)))) if len(P) > 1 else 0.0
                        rows.setdefault(int(o), {})["n1"] = int(b_ - a_)
                        rows[int(o)]["rms"] = rms
                    # how many separate patches does each object make?
                    cc, _ = ndimage.label(shell, structure=st)
                    cs = cc[shell][order]
                    for o, a_, b_ in zip(uq.tolist(), starts, stops):
                        rows[int(o)]["ncc"] = int(len(np.unique(cs[a_:b_])))
                else:
                    uq, cnt = np.unique(lab, return_counts=True)
                    for o, n in zip(uq.tolist(), cnt.tolist()):
                        rows.setdefault(int(o), {})["n2"] = n
            # --- signed heading, over EVERY object in the box -------------
            # not only the touching ones, so these rows join to the whole panel
            nz = np.nonzero(vol)
            P2 = (np.stack(nz, 1) + lo) * res
            R2 = vol[nz]
            h = outward_heading(P2[R2 == seed], ctr)
            near = np.linalg.norm(P2 - ctr, axis=1) < LOCAL_NM
            oproj, ahead = {}, {}
            if h is not None:
                Rn, Pn = R2[near], P2[near]
                order = np.argsort(Rn, kind="stable")
                Rs, Ps = Rn[order], Pn[order]
                uq, starts = np.unique(Rs, return_index=True)
                stops = np.append(starts[1:], len(Rs))
                for o, a_, b_ in zip(uq.tolist(), starts, stops):
                    v = Ps[a_:b_].mean(0) - ctr
                    n = float(np.linalg.norm(v))
                    oproj[int(o)] = float(v @ h / n) if n > 0 else 0.0
                    ahead[int(o)] = float(v @ h)
            for o in oproj:
                rows.setdefault(o, {})

            objs = np.array(sorted(rows), dtype=np.uint64)
            np.savez(dest, obj=objs,
                     oproj=np.array([oproj.get(int(o), np.nan) for o in objs], np.float32),
                     d_ahead=np.array([ahead.get(int(o), np.nan) for o in objs], np.float32),
                     heading=np.asarray(h if h is not None else [np.nan] * 3, np.float32),
                     n_touch1=np.array([rows[int(o)].get("n1", 0) for o in objs], np.int64),
                     n_touch2=np.array([rows[int(o)].get("n2", 0) for o in objs], np.int64),
                     touch_rms=np.array([rows[int(o)].get("rms", 0.0) for o in objs], np.float32),
                     touch_cc=np.array([rows[int(o)].get("ncc", 0) for o in objs], np.int64),
                     seed=np.uint64(seed))
            print(f"  {key}: {len(objs)} objects touch the seed  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        except Exception as exc:                       # noqa: BLE001
            import os, traceback
            if os.environ.get("FACE_DEBUG"):
                traceback.print_exc()
            print(f"  {key}: {type(exc).__name__}: {str(exc)[:90]}", flush=True)


if __name__ == "__main__":
    main()
