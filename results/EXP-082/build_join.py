"""EXP-082 step 1: join every logged proofreader operation onto the final
skeleton of the cell it built. Produces one row per operation endpoint."""
import json, glob, os, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = "/Users/wgray13/projects/neuronauts"
V117_MS = 1623399000 * 1000
out = []
skels = {}
cells = 0
for sf in sorted(glob.glob(f"{ROOT}/data/external/cell_skeletons/*_skv4.npz")):
    r = int(os.path.basename(sf).split("_")[0])
    ef = f"{ROOT}/data/external/edit_history/{r}.json"
    if not os.path.exists(ef):
        continue
    z = np.load(sf, allow_pickle=True)
    V = z["vertices"].astype(np.float64)
    E = z["edges"].astype(np.int64)
    rad = z["radius"].astype(np.float64)
    comp = z["compartment"].astype(np.int64)
    n = len(V)
    w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    A = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(n, n)).tocsr()
    soma = int(np.flatnonzero(comp == 1)[0])
    pathd = dijkstra(A, indices=soma, directed=False)
    # cable length attributed to each vertex (half of each incident edge)
    cable = np.zeros(n)
    np.add.at(cable, E[:, 0], w / 2)
    np.add.at(cable, E[:, 1], w / 2)
    # degree -> tip / branch
    deg = np.bincount(E.ravel(), minlength=n)
    skels[r] = dict(V=V, rad=rad, comp=comp, pathd=pathd, cable=cable, deg=deg,
                    soma_pos=V[soma], n=n)
    tree = cKDTree(V)
    d = json.load(open(ef))
    for o in d["ops"]:
        pts = np.asarray(o["edit_points_nm"], float)
        dd, ii = tree.query(pts)
        for k in range(len(pts)):
            j = int(ii[k])
            out.append((r, o["operation_id"], o["timestamp_ms"], int(o["user_id"]),
                        int(o["is_merge"]), k, len(pts),
                        pts[k, 0], pts[k, 1], pts[k, 2],
                        float(dd[k]), float(rad[j]), int(comp[j]), float(pathd[j]),
                        int(deg[j]), j))
    cells += 1

import numpy as np
dt = np.dtype([("root", "i8"), ("op", "i8"), ("t_ms", "i8"), ("user", "i4"),
               ("is_merge", "i1"), ("pt_idx", "i2"), ("n_pts", "i2"),
               ("x", "f8"), ("y", "f8"), ("z", "f8"),
               ("d_skel", "f8"), ("radius", "f8"), ("comp", "i2"),
               ("path_soma", "f8"), ("deg", "i2"), ("vert", "i8")])
arr = np.array(out, dtype=dt)
np.save("/private/tmp/claude-501/-Users-wgray13-projects-neuronauts/8c2bcfd4-b48d-453f-ae78-fb9ed1b00ae7/scratchpad/edit_join.npy", arr)
np.savez_compressed("/private/tmp/claude-501/-Users-wgray13-projects-neuronauts/8c2bcfd4-b48d-453f-ae78-fb9ed1b00ae7/scratchpad/skel_cache.npz",
                    **{f"{r}_{k}": v for r, s in skels.items() for k, v in s.items()
                       if isinstance(v, np.ndarray)})
print("cells", cells, "endpoint rows", len(arr))
print("merge endpoints", int((arr['is_merge']==1).sum()), "split endpoints", int((arr['is_merge']==0).sum()))
print("dist to final skeleton (nm) percentiles, merge endpoints:")
m = arr[arr['is_merge']==1]['d_skel']
print(" ", np.percentile(m,[10,25,50,75,90,95,99]).round(0))
s = arr[arr['is_merge']==0]['d_skel']
print("split endpoints:", np.percentile(s,[10,25,50,75,90,95,99]).round(0))
