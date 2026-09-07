"""EXP-082 Q2: did proofreaders grow outward from the soma like a frontier
grower, or jump to disconnected places? For each merge in time order, measure
the path distance along the FINAL skeleton from the new merge site to the
nearest site already touched (soma counts as touched at t=0). Compared against
a null in which the same merges happen in shuffled order."""
import numpy as np, glob, os, json
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from pathlib import Path

R = str(Path(__file__).resolve().parents[2])
V117_MS = 1623399000 * 1000
A = np.load(Path(R) / "data/external/edit_join_v082.npz")["edit_join"]
M = A[(A["is_merge"] == 1) & (A["d_skel"] < 2000) & (A["t_ms"] > V117_MS)]
print("post-v117 merge endpoints on the final skeleton:", len(M), flush=True)
rng = np.random.default_rng(0)
obs, null, cells = [], [], 0
for sf in sorted(glob.glob(f"{R}/data/external/cell_skeletons/*_skv4.npz")):
    r = int(os.path.basename(sf).split("_")[0])
    m = M[M["root"] == r]
    if len(m) < 20: continue
    z = np.load(sf, allow_pickle=True)
    V = z["vertices"].astype(float); E = z["edges"].astype(np.int64)
    comp = z["compartment"].astype(int); n = len(V)
    w = np.linalg.norm(V[E[:,0]] - V[E[:,1]], axis=1)
    G = coo_matrix((np.r_[w,w], (np.r_[E[:,0],E[:,1]], np.r_[E[:,1],E[:,0]])), shape=(n,n)).tocsr()
    soma = int(np.flatnonzero(comp == 1)[0])
    sites = np.unique(np.r_[m["vert"], soma])
    D = dijkstra(G, indices=sites, directed=False)          # (len(sites), n)
    row = {int(s): i for i, s in enumerate(sites)}
    # group endpoints by operation, keep time order
    order = np.argsort(m["t_ms"], kind="stable")
    ms = m[order]
    ops, seen_op = [], {}
    for e in ms:
        seen_op.setdefault(int(e["op"]), []).append(int(e["vert"]))
    ops = [(op, vs) for op, vs in seen_op.items()]
    def run(seq):
        claimed = [row[soma]]
        out = []
        for op, vs in seq:
            d = D[claimed][:, vs].min()
            out.append(d)
            claimed.extend(row[v] for v in vs)
        return out
    obs += run(ops)
    perm = list(ops); rng.shuffle(perm)
    null += run(perm)
    cells += 1
obs = np.array(obs) / 1000.; null = np.array(null) / 1000.
print("cells", cells, "merge operations scored", len(obs), flush=True)
def rep(tag, a):
    print(f"{tag}: median {np.median(a):7.2f} um  p25 {np.percentile(a,25):6.2f}  p75 {np.percentile(a,75):7.2f}  p90 {np.percentile(a,90):8.2f}", flush=True)
    for t in [2, 5, 10, 25, 50, 100]:
        print(f"    > {t:3d} um from anything already touched: {100*(a>t).mean():5.1f}%", flush=True)
rep("OBSERVED (real time order)", obs)
rep("NULL (shuffled order)     ", null)
np.savez(f"{R}/results/EXP-082/growth_order.npz", obs=obs, null=null)
