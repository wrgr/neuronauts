"""Do real MICrONS neurons obey Cajal's conservation laws, and do wrong joins break them?

The Cajal priors are parameter-free biophysics over TREE structure -- caliber
conservation at a branch point, the Hess-Murray optimal bifurcation angle --
which is the grammar level, not the pairwise level. Two questions, in order:

1. Do the laws hold on real proofread cells? If a law does not describe this
   tissue, it cannot police an assembly of it. This has never been checked here.
2. Does a WRONG join break them? A join grafts foreign cable onto the arbor and
   creates a new branch point. If real branch points obey the law and grafted
   ones do not, the grammar detects wrong joins from shape alone -- which is
   exactly what local evidence cannot do.

Murray's law: r_mother^3 = r_d1^3 + r_d2^3, so the exponent p solving
r0^p = r1^p + r2^p is 3 for ideal material conservation.
"""
import numpy as np, glob, json
from scipy.optimize import brentq
R="/Users/wgray13/projects/neuronauts/"
import sys; sys.path.insert(0, R)
from neuronauts.morpho_grammar.cajal_conservation_priors import SantiagoCajalPriors as P

def branch_stats(V, E, rad, max_bp=400):
    from collections import defaultdict
    adj=defaultdict(list)
    for a,b in E: adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    out=[]
    for n,nb in adj.items():
        if len(nb)!=3: continue          # a simple bifurcation
        r=[float(rad[x]) for x in nb]; r0=max(r); ds=sorted(r)[:2]
        if min(ds)<20 or r0<20: continue
        # exponent p with r0^p = r1^p + r2^p
        f=lambda p: ds[0]**p + ds[1]**p - r0**p
        try:
            if f(0.5)*f(8.0) > 0: continue
            p_=brentq(f,0.5,8.0)
        except Exception: continue
        # angle between the two daughters
        vs=[V[x]-V[n] for x in nb]
        order=np.argsort([-np.linalg.norm(rad[x]) if False else -float(rad[x]) for x in nb])
        d1,d2=vs[order[1]],vs[order[2]]
        c=float(d1@d2/max(np.linalg.norm(d1)*np.linalg.norm(d2),1e-9))
        out.append((p_, float(np.arccos(np.clip(c,-1,1))), r0, ds[0], ds[1]))
        if len(out)>=max_bp: break
    return out

real=[]
files=sorted(glob.glob(R+"data/external/cell_skeletons/*_skv4.npz"))
for f in files[:60]:
    z=np.load(f, allow_pickle=False)
    V,E,rad=z["vertices"].astype(float),z["edges"].astype(int),z["radius"].astype(float)
    ok=np.isfinite(V).all(1)
    if not ok.all(): continue
    real += branch_stats(V,E,rad)
A=np.array(real)
print(f"real bifurcations measured: {len(A)} across {len(files[:60])} cells")
print(f"Murray exponent p (ideal 3.0): median {np.median(A[:,0]):.2f}  IQR {np.percentile(A[:,0],25):.2f}-{np.percentile(A[:,0],75):.2f}")
print(f"bifurcation angle (deg)      : median {np.degrees(np.median(A[:,1])):.0f}")

# what the prior scores on real branches vs on random (wrong-join) pairings
rng=np.random.default_rng(0)
sc_real=[P.compute_bifurcation_angle_prior(a[2],a[3],a[4],a[1]) for a in A]
idx=rng.permutation(len(A))
sc_fake=[P.compute_bifurcation_angle_prior(A[i,2],A[j,3],A[j,4],A[j,1]) for i,j in zip(range(len(A)),idx)]
from itertools import product
def auc(a,b,n=200):
    a=np.asarray(a)[:n]; b=np.asarray(b)[:n]
    return float(np.mean([1.0*(x>y)+0.5*(x==y) for x,y in product(a,b)]))
print(f"\nCajal angle prior, real branch vs mismatched caliber/angle: AUC {auc(sc_real,sc_fake):.3f}")
p_real=A[:,0]
p_fake=np.array([brentq(lambda p: A[j,3]**p + A[j,4]**p - A[i,2]**p, 0.5, 8.0)
                 if (A[j,3]**0.5+A[j,4]**0.5-A[i,2]**0.5)*(A[j,3]**8+A[j,4]**8-A[i,2]**8)<0 else np.nan
                 for i,j in zip(range(len(A)),idx)])
p_fake=p_fake[np.isfinite(p_fake)]
print(f"Murray exponent, real: median {np.median(p_real):.2f}   mismatched: median {np.median(p_fake):.2f}")
print(f"  |p-3| real {np.median(np.abs(p_real-3)):.2f}   mismatched {np.median(np.abs(p_fake-3)):.2f}")
print(f"  AUC separating real from mismatched by |p-3|: {auc(-np.abs(p_real-3), -np.abs(p_fake-3)):.3f}")
