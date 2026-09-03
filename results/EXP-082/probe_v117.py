"""EXP-082 Q4: for a sample of post-v117 human merges, read the v117 substrate
around the two clicked points and ask what a candidate generator would have
seen: same object? two objects in contact? or two objects with other tissue
in between (a skip)."""
import json, glob, sys, time, numpy as np
from datetime import datetime, timezone
R = "/Users/wgray13/projects/neuronauts"
V117_TS = datetime.fromtimestamp(1623399000, tz=timezone.utc)
V117_MS = 1623399000 * 1000
PAD_NM = 1000.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = 0
import os

merges = []
for f in sorted(glob.glob(f"{R}/data/external/edit_history/*.json")):
    d = json.load(open(f))
    for o in d["ops"]:
        if o["is_merge"] and o["timestamp_ms"] > V117_MS and len(o["edit_points_nm"]) == 2:
            p = np.asarray(o["edit_points_nm"], float)
            merges.append((d["root"], o["operation_id"], p, float(np.linalg.norm(p[0]-p[1]))))
print(f"post-v117 merges with 2 points: {len(merges)}", flush=True)
rng = np.random.default_rng(SEED)
span = np.array([m[3] for m in merges])
# stratified: 2/3 uniform random, 1/3 from the >1um tail
idx_all = rng.choice(len(merges), size=int(N*2/3), replace=False)
tail = np.flatnonzero(span > 1000.)
idx_tail = rng.choice(tail, size=min(N - len(idx_all), len(tail)), replace=False)
sample = list(idx_all) + list(idx_tail)
strat = ["uniform"]*len(idx_all) + ["tail>1um"]*len(idx_tail)

import caveclient
from cloudvolume import CloudVolume
cl = caveclient.CAVEclient("minnie65_public")
cv = CloudVolume(cl.chunkedgraph.cloudvolume_path, mip=2, use_https=True,
                 progress=False, fill_missing=True, agglomerate=True, timestamp=V117_TS)
res = np.asarray(cv.resolution, float)
bmin = np.asarray(cv.bounds.minpt, int); bmax = np.asarray(cv.bounds.maxpt, int)
print("mip2 resolution", res, flush=True)

def adj_ids(vol, a):
    """Distinct object ids 6-adjacent to object a inside vol (0 excluded)."""
    A = vol == a; out = set()
    for ax in range(3):
        sa = [slice(None)]*3; sb = [slice(None)]*3
        sa[ax] = slice(1, None); sb[ax] = slice(None, -1)
        out.update(np.unique(vol[tuple(sb)][A[tuple(sa)]]).tolist())
        out.update(np.unique(vol[tuple(sa)][A[tuple(sb)]]).tolist())
    out.discard(0); out.discard(int(a))
    return out


def neighbors_touch(vol, a, b):
    """True if object a and object b are 6-adjacent anywhere in vol."""
    A = vol == a; B = vol == b
    for ax in range(3):
        sa = [slice(None)]*3; sb = [slice(None)]*3
        sa[ax] = slice(1, None); sb[ax] = slice(None, -1)
        if (A[tuple(sa)] & B[tuple(sb)]).any(): return True
        if (B[tuple(sa)] & A[tuple(sb)]).any(): return True
    return False

rows = []; t0 = time.time()
for n, (si, st) in enumerate(zip(sample, strat), 1):
    root, op, p, sp = merges[si]
    lo = np.floor((p.min(0) - PAD_NM) / res).astype(int)
    hi = np.ceil((p.max(0) + PAD_NM) / res).astype(int) + 1
    lo = np.maximum(lo, bmin); hi = np.minimum(hi, bmax)
    if np.any(hi - lo < 2): continue
    try:
        vol = np.asarray(cv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])[..., 0]
    except Exception as e:
        print(f"  read fail {op}: {type(e).__name__} {str(e)[:80]}", flush=True); continue
    vx = np.round(p / res).astype(int) - lo
    vx = np.clip(vx, 0, np.array(vol.shape) - 1)
    ida, idb = int(vol[tuple(vx[0])]), int(vol[tuple(vx[1])])
    # line sample between the two clicks, ~1 voxel steps
    steps = max(2, int(np.ceil(np.linalg.norm(vx[1]-vx[0]))) + 1)
    tt = np.linspace(0, 1, steps)[:, None]
    line = np.round(vx[0] + tt * (vx[1]-vx[0])).astype(int)
    lids = vol[line[:,0], line[:,1], line[:,2]].astype(np.int64)
    same = ida == idb and ida != 0
    touch = (not same) and ida != 0 and idb != 0 and neighbors_touch(vol, ida, idb)
    # tissue strictly between: ids on the line that are neither endpoint id
    mid = lids[1:-1] if len(lids) > 2 else np.array([], dtype=np.int64)
    other = mid[(mid != ida) & (mid != idb)]
    n_other_obj = len(set(other[other != 0].tolist()))
    n_zero = int((other == 0).sum())
    nbr_a = adj_ids(vol, ida) if ida else set()
    nbr_b = adj_ids(vol, idb) if idb else set()
    rows.append(dict(n_adj_a=len(nbr_a), n_adj_b=len(nbr_b),
                     n_obj_in_box=int(len(np.unique(vol)) - (1 if (vol==0).any() else 0)),
                     root=int(root), op=int(op), strat=st, span_nm=sp,
                     id_a=ida, id_b=idb, same=bool(same), touch=bool(touch),
                     n_line=len(lids), n_other_obj=n_other_obj, n_zero_between=n_zero,
                     other_ids=sorted(set(other[other != 0].tolist()))[:5],
                     box_vox=[int(x) for x in (hi-lo)]))
    if n % 25 == 0:
        print(f"  {n}/{len(sample)}  {time.time()-t0:.0f}s", flush=True)
json.dump(rows, open(f"{R}/results/EXP-082/v117_merge_probe.json", "w"), indent=1)
print(f"wrote {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)
