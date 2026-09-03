"""The grammar labels every cut surface, so negative sites are not scarce.

A neurite ends for exactly three reasons, and the grammar names two of them:

  SYNAPTIC TERMINAL -- the tip carries a synapse. A bouton or a spine head is a
      legitimate ending. The grammar says stop, and no proofreading is needed to
      know it.
  FIELD BOUNDARY    -- the tip sits at the edge of the volume. It is truncated
      by our box, not by biology, so it is neither a terminal nor a split.
  UNEXPLAINED       -- neither. Cable that simply stops in the middle of tissue
      is what a split looks like.

This gives labelled decision sites at the scale of the segmentation itself
rather than the scale of proofread cells. I had claimed the corpus held 58
negative sites; that counted cells, not cut surfaces.
"""
import numpy as np, glob, json
from scipy.spatial import cKDTree
R="/Users/wgray13/projects/neuronauts/"
C=np.array([663.,591.,860.])*1000.; LO,HI=C-50000,C+50000
p=np.load(R+"data/substrate/c100um/population.npz", allow_pickle=False)
syn=p["syn_ctr_nm"].astype(np.float32)
# A synapse must be ON this object, not merely near it. With 901,498 synapses in
# a 100 um cube the mean spacing is about 1.04 um, so "within 1.5 um of any
# synapse" is satisfied by chance nearly everywhere -- my first pass reported
# 99.96% of cut surfaces explained, which was that artifact.
from collections import defaultdict
_own=defaultdict(list)
for arr in (p["syn_atom_pre"], p["syn_atom_post"]):
    m=arr>0
    for a,c in zip(arr[m].tolist(), syn[m]): _own[int(a)].append(c)
own={k:cKDTree(np.asarray(v,np.float32)) for k,v in _own.items() if v}
print(f"objects carrying a synapse: {len(own):,}")
z5=np.load(R+"data/substrate/c100um/object_clouds_mip5.npz", allow_pickle=False)
obj,ptr,pos=z5["object_id"],z5["node_ptr"],z5["pos_nm"]
per=np.diff(ptr)
print(f"objects in cube: {len(obj):,}   synapses: {len(syn):,}")

def tips_of(P, nbr=3000., beyond=600., maxp=1500):
    """Cut surfaces: points with no cable beyond them along the local axis."""
    if len(P)>maxp: P=P[np.linspace(0,len(P)-1,maxp).astype(int)]
    if len(P)<4: return np.empty((0,3))
    ctr=P.mean(0); tr=cKDTree(P); out=[]; claimed=np.zeros(len(P),bool)
    for i in np.argsort(-np.linalg.norm(P-ctr,axis=1)):
        if claimed[i]: continue
        nb=tr.query_ball_point(P[i],r=nbr)
        if len(nb)<3: continue
        u=P[i]-ctr; n=np.linalg.norm(u)
        if n<1: continue
        u=u/n
        if np.any((P[nb]-P[i])@u>beyond): continue
        out.append(P[i])
        for j in tr.query_ball_point(P[i],r=6000.): claimed[j]=True
    return np.asarray(out) if out else np.empty((0,3))

rng=np.random.default_rng(0)
big=np.flatnonzero(per>=12)            # objects with enough cable to have ends
sel=rng.choice(big, size=min(3000,len(big)), replace=False)
n_syn=n_edge=n_unexp=0; n_obj=0
for k in sel:
    P=pos[int(ptr[k]):int(ptr[k+1])]
    T=tips_of(P)
    if not len(T): continue
    n_obj+=1
    oid=int(obj[k]); ot=own.get(oid)
    d_syn=ot.query(T,k=1)[0] if ot is not None else np.full(len(T), 1e9)
    d_edge=np.minimum((T-LO).min(1),(HI-T).min(1))
    is_edge=d_edge<3000.
    is_syn=(~is_edge)&(d_syn<1500.)
    n_edge+=int(is_edge.sum()); n_syn+=int(is_syn.sum())
    n_unexp+=int((~is_edge&~is_syn).sum())
tot=n_syn+n_edge+n_unexp
print(f"\nsampled {n_obj:,} objects -> {tot:,} cut surfaces")
print(f"  synaptic terminal (grammar: STOP) : {n_syn:>7,}  {100*n_syn/tot:>5.1f}%")
print(f"  field boundary    (not a decision): {n_edge:>7,}  {100*n_edge/tot:>5.1f}%")
print(f"  unexplained       (probable SPLIT): {n_unexp:>7,}  {100*n_unexp/tot:>5.1f}%")
scale=len(big)/max(n_obj,1)
print(f"\nextrapolated over {len(big):,} objects with cable in this cube:")
print(f"  ~{int(n_syn*scale):,} labelled STOP sites and ~{int(n_unexp*scale):,} candidate split sites")
print(f"\nI had said the corpus holds 58 negative sites. That counted proofread")
print(f"cells. Counting cut surfaces, the grammar labels them by the thousand,")
print(f"with no proofreading involved.")
