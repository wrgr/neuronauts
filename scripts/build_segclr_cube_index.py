"""Extract SegCLR embeddings inside the 100 um cube, keyed spatially.

The July branch stitched fragments by embedding similarity and reported that
geometry generates candidates while SegCLR selects among them. Our own cache
holds a contradicting measurement (auc_result.json: AUC 0.445 on 34 v117
atoms, different-owner pairs MORE similar than same-owner). Both are small.
This builds the index needed to settle it on 66 real panels.

Embeddings are keyed to the m343 segmentation, so no identifier maps onto v117.
They carry coordinates, so assignment is spatial -- which is what the July code
did too (assign_points_to_vertices).
"""
import csv, io, sys, time, zipfile
from pathlib import Path
import numpy as np

SEG = Path("/Users/wgray13/projects/neuronauts/data/external/segclr")
OUT = Path("/Users/wgray13/projects/neuronauts/data/external/segclr_cube.npz")
C = np.array([663., 591., 860.]) * 1000.0
LO, HI = C - 60000.0, C + 60000.0        # cube plus margin

zips = sorted(SEG.glob("shards/*.zip")) + sorted(SEG.glob("0.zip"))
print(f"{len(zips)} shards", flush=True)
xyz, emb, seg = [], [], []
t0 = time.time()
for zi, zp in enumerate(zips):
    try:
        z = zipfile.ZipFile(zp)
    except Exception as e:
        print(f"  {zp.name}: {type(e).__name__}", flush=True); continue
    kept = 0
    for nm in z.namelist():
        try:
            sid = int(nm[:-4])
        except ValueError:
            continue
        try:
            raw = z.read(nm).decode()
        except Exception:
            continue
        for line in raw.splitlines():
            if not line:
                continue
            p = line.split(",")
            if len(p) < 68:
                continue
            x, y, zc = float(p[1]), float(p[2]), float(p[3])
            if not (LO[0] <= x <= HI[0] and LO[1] <= y <= HI[1] and LO[2] <= zc <= HI[2]):
                continue
            xyz.append((x, y, zc)); emb.append([float(v) for v in p[4:68]]); seg.append(sid)
            kept += 1
    print(f"  [{zi+1}/{len(zips)}] {zp.name}: +{kept:,} in cube  (total {len(xyz):,}, {time.time()-t0:.0f}s)", flush=True)

if not xyz:
    print("NO embedding points inside the cube -- the cached shards cover elsewhere"); sys.exit(0)
X = np.asarray(xyz, np.float32); E = np.asarray(emb, np.float32); S = np.asarray(seg, np.uint64)
E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
np.savez(OUT, xyz_nm=X, emb=E, seg_id=S)
print(f"\nwrote {OUT}: {len(X):,} embedded points, {len(np.unique(S)):,} m343 segments")
print(f"x {X[:,0].min()/1000:.0f}-{X[:,0].max()/1000:.0f} um, "
      f"y {X[:,1].min()/1000:.0f}-{X[:,1].max()/1000:.0f}, z {X[:,2].min()/1000:.0f}-{X[:,2].max()/1000:.0f}")
