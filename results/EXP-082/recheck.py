"""Honesty check on the Q4 skip claim: the pairs called non-adjacent inside a
+-1 um box -- do they touch if the box is widened to +-4 um?"""
import json, numpy as np, time
from datetime import datetime, timezone
from pathlib import Path
R = str(Path(__file__).resolve().parents[2])
V117_TS=datetime.fromtimestamp(1623399000,tz=timezone.utc)
rows=json.load(open(f"{R}/results/EXP-082/v117_merge_probe.json"))
cand=[x for x in rows if (not x['same']) and x['id_a'] and x['id_b'] and not x['touch']]
print('non-adjacent-at-1um pairs:',len(cand),flush=True)
rng=np.random.default_rng(1); idx=rng.choice(len(cand),size=min(70,len(cand)),replace=False)
import caveclient
from cloudvolume import CloudVolume
cl=caveclient.CAVEclient("minnie65_public")
cv=CloudVolume(cl.chunkedgraph.cloudvolume_path,mip=2,use_https=True,progress=False,
               fill_missing=True,agglomerate=True,timestamp=V117_TS)
res=np.asarray(cv.resolution,float)
bmin=np.asarray(cv.bounds.minpt,int); bmax=np.asarray(cv.bounds.maxpt,int)
eh={}
def pts_for(root,op):
    if root not in eh: eh[root]=json.load(open(f"{R}/data/external/edit_history/{root}.json"))
    for o in eh[root]['ops']:
        if o['operation_id']==op: return np.asarray(o['edit_points_nm'],float)
def touch(vol,a,b):
    A=vol==a;B=vol==b
    for ax in range(3):
        sa=[slice(None)]*3;sb=[slice(None)]*3
        sa[ax]=slice(1,None);sb[ax]=slice(None,-1)
        if (A[tuple(sa)]&B[tuple(sb)]).any() or (B[tuple(sa)]&A[tuple(sb)]).any(): return True
    return False
PAD=4000.
n_t=0;n=0;t0=time.time();out=[]
for i in idx:
    x=cand[int(i)]; p=pts_for(x['root'],x['op'])
    lo=np.maximum(np.floor((p.min(0)-PAD)/res).astype(int),bmin)
    hi=np.minimum(np.ceil((p.max(0)+PAD)/res).astype(int)+1,bmax)
    try: vol=np.asarray(cv[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]])[...,0]
    except Exception as e: print('fail',type(e).__name__,flush=True); continue
    t=touch(vol,x['id_a'],x['id_b']); n_t+=t; n+=1
    out.append(dict(op=x['op'],span_nm=x['span_nm'],touch_4um=bool(t)))
    if n%15==0: print(f'  {n}/{len(idx)} touching-at-4um {n_t}  {time.time()-t0:.0f}s',flush=True)
json.dump(out,open(f"{R}/results/EXP-082/v117_recheck_4um.json","w"),indent=1)
print(f'RESULT: of {n} pairs non-adjacent within +-1um, {n_t} ({100*n_t/max(n,1):.1f}%) do touch within +-4um; '
      f'{n-n_t} ({100*(n-n_t)/max(n,1):.1f}%) still do not.',flush=True)
