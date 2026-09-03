"""Can anything separate a live extension site from a dead end?

EXP-081: a grower faces median 46 cut ends per cell and should extend at 1.
This scores every tip and asks whether the live ones stand out -- the decision
the grower actually makes, at its real 1.6% base rate.

Coarse by design: features come from the mip-5 clouds, so distances are between
supervoxel centroids rather than surfaces. That understates every feature. The
point is the SHAPE of the answer -- is the frontier separable at all -- not a
final number. If it separates even coarsely, a mip-2 pass is worth its cost.
"""
import numpy as np, glob, json
from scipy.spatial import cKDTree
R="/Users/wgray13/projects/neuronauts/"
z5=np.load(R+"data/substrate/c100um/object_clouds_mip5.npz", allow_pickle=False)
obj,ptr,pos=z5["object_id"],z5["node_ptr"],z5["pos_nm"]
per=np.diff(ptr); owner=np.repeat(obj,per)
tree=cKDTree(pos)
rowi={int(a):k for k,a in enumerate(obj.tolist())}
def pts(a):
    k=rowi.get(int(a)); return pos[int(ptr[k]):int(ptr[k+1])] if k is not None else np.empty((0,3))

def tips(P,soma,nbr=3000.,beyond=600.,maxp=4000):
    if len(P)>maxp: P=P[np.linspace(0,len(P)-1,maxp).astype(int)]
    tr=cKDTree(P); out=[]; claimed=np.zeros(len(P),bool)
    for i in np.argsort(-np.linalg.norm(P-soma,axis=1)):
        if claimed[i]: continue
        nb=tr.query_ball_point(P[i],r=nbr)
        if len(nb)<3: continue
        u=P[i]-soma; n=np.linalg.norm(u)
        if n<1: continue
        u=u/n
        if np.any((P[nb]-P[i])@u>beyond): continue
        loc=P[nb]
        ax=np.linalg.svd(loc-loc.mean(0),full_matrices=False)[2][0]
        if ax@u<0: ax=-ax
        out.append((P[i],u,ax,len(nb)))
        for j in tr.query_ball_point(P[i],r=6000.): claimed[j]=True
    return out

cards=[json.load(open(f)) for f in sorted(glob.glob(R+"data/external/cell_cards/*.json")) if not f.split("/")[-1].startswith("_")]
cards=[c for c in cards if c.get("coverage",{}).get("graph")]
rows=[]
for c in cards[:40]:
    seed=int(c["seed"]["v117_fragment"]); soma=np.asarray(c["seed"]["pos_nm"],float)
    P=pts(seed)
    if len(P)<20: continue
    T=tips(P,soma)
    if not T: continue
    tgt=set(c["structure"]["seeded_target"])-{seed}
    TP=[pts(x) for x in tgt if len(pts(x))]
    tt=cKDTree(np.vstack(TP)) if TP else None
    for tp,u,ax,dens in T:
        live = bool(tt is not None and tt.query(tp[None],k=1)[0][0]<5000.)
        # what does the frontier see at this tip?
        idx=tree.query_ball_point(tp, r=6000.)
        idx=[i for i in idx if int(owner[i])!=seed]
        if not idx:
            rows.append((live,6000.,0.,0.,0)); continue
        Q=pos[idx]; d=np.linalg.norm(Q-tp,axis=1)
        v=(Q-tp)/np.maximum(d[:,None],1.)
        al=v@ax                                   # does it lie along the cable's axis?
        near=int(np.sum(d<2000.))
        best=int(np.argmax(al*np.exp(-d/2000.)))
        rows.append((live,float(d.min()),float(al.max()),float(al[best]),near))
A=np.array([(1 if r[0] else 0,)+r[1:] for r in rows],float)
live=A[:,0]==1
print(f"tips {len(A)}   live {int(live.sum())}   base rate {100*live.mean():.1f}%")
from itertools import product
def auc(x, hi=True):
    a=x[live]; b=x[~live]
    if not len(a) or not len(b): return float("nan")
    s=sum(1.0*(p>q)+0.5*(p==q) for p,q in product(a,b))/(len(a)*len(b))
    return s if hi else 1-s
print(f"\n{'feature':>28}{'AUC':>8}")
for name,col,hi in (("nearest object (closer=live)",1,False),("best along-axis alignment",2,True),
                    ("alignment of best candidate",3,True),("objects within 2um",4,True)):
    print(f"{name:>28}{auc(A[:,col],hi):>8.3f}")
score=A[:,2]*np.exp(-A[:,1]/2000.)
a=auc(score); print(f"{'alignment x proximity':>28}{a:>8.3f}")
# precision at the real base rate
o=np.argsort(-score)
for k in (int(live.sum()), 2*int(live.sum()), 5*int(live.sum())):
    if k<1 or k>len(A): continue
    print(f"  top {k:>4} by that score: {int(live[o][:k].sum())} live  -> precision {100*live[o][:k].mean():.1f}% (base {100*live.mean():.1f}%)")
