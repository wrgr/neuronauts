"""The real shape of the task: a frontier, not a single contact point.

My panels centred each cell's candidates on the known seed/target contact,
which hands the grower the answer to "where should I look". A grower starting
at a soma has no such information. It has a FRONTIER -- every cut end of the
cable it has claimed -- and must decide at each one whether anything continues.
Most tips are genuine endings.

This measures the real decision load, per cell:
  tips        cut ends on the seed fragment
  live        tips with a target fragment within reach (a real extension site)
  dead        tips where the honest answer is "nothing continues"

Tips are found on the mip-5 cloud, which is adequate at micron scale: a tip is
a point with no cable beyond it along the outward direction.
"""
import numpy as np, glob, json
from scipy.spatial import cKDTree
R="/Users/wgray13/projects/neuronauts/"
z5=np.load(R+"data/substrate/c100um/object_clouds_mip5.npz", allow_pickle=False)
obj,ptr,pos=z5["object_id"],z5["node_ptr"],z5["pos_nm"]
rowi={int(a):k for k,a in enumerate(obj.tolist())}
def pts(a):
    k=rowi.get(int(a)); return pos[int(ptr[k]):int(ptr[k+1])] if k is not None else np.empty((0,3))

def tips(P, soma, nbr_nm=3000.0, beyond_nm=600.0, max_pts=4000):
    """Points with no cable beyond them, walking outward from the soma."""
    if len(P) > max_pts:
        P = P[np.linspace(0, len(P)-1, max_pts).astype(int)]
    tr=cKDTree(P); out=[]
    d_soma=np.linalg.norm(P-soma,axis=1)
    order=np.argsort(-d_soma)                      # distal points first
    claimed=np.zeros(len(P),bool)
    for i in order:
        if claimed[i]: continue
        nb=tr.query_ball_point(P[i], r=nbr_nm)
        if len(nb)<3: continue
        u=P[i]-soma; n=np.linalg.norm(u)
        if n<1: continue
        u=u/n
        if np.any((P[nb]-P[i])@u > beyond_nm): continue
        out.append(P[i])
        for j in tr.query_ball_point(P[i], r=6000.0): claimed[j]=True   # one tip per ending
    return np.asarray(out) if out else np.empty((0,3))

cards=[json.load(open(f)) for f in sorted(glob.glob(R+"data/external/cell_cards/*.json")) if not f.split("/")[-1].startswith("_")]
cards=[c for c in cards if c.get("coverage",{}).get("graph")]
print(f"{'cell':>9} {'tips':>5} {'live':>5} {'dead':>5}  {'target frags':>12}")
rows=[]
for c in cards[:40]:
    seed=int(c["seed"]["v117_fragment"]); soma=np.asarray(c["seed"]["pos_nm"],float)
    P=pts(seed)
    if len(P)<20: continue
    T=tips(P,soma)
    if not len(T): continue
    tgt=set(c["structure"]["seeded_target"])-{seed}
    TP=[pts(x) for x in tgt if len(pts(x))]
    if TP:
        tt=cKDTree(np.vstack(TP))
        live=int(np.sum(tt.query(T,k=1)[0] < 5000.0))
    else:
        live=0
    rows.append((len(T),live,len(T)-live,len(tgt)))
    print(f"{str(c['cell'])[-8:]:>9} {len(T):>5} {live:>5} {len(T)-live:>5}  {len(tgt):>12}")
A=np.array(rows)
print(f"\ncells measured: {len(A)}")
print(f"tips per cell        : median {np.median(A[:,0]):.0f}   total {A[:,0].sum()}")
print(f"live extension sites : median {np.median(A[:,1]):.0f}   total {A[:,1].sum()}")
print(f"dead ends            : median {np.median(A[:,2]):.0f}   total {A[:,2].sum()}")
print(f"\nA grower must decide at {A[:,0].sum()} tips and extend at only {A[:,1].sum()}"
      f" ({100*A[:,1].sum()/max(A[:,0].sum(),1):.1f}%).")
print("That base rate, not the per-panel ranking, is the task.")
