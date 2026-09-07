import numpy as np, time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from pathlib import Path
_D = Path(__file__).resolve().parents[2] / 'data/external'
A=np.load(_D / 'edit_join_v082.npz')['edit_join']
Z=np.load(_D / 'edit_skel_cache_v082.npz')
roots=sorted({int(k.split('_')[0]) for k in Z.files})
g=lambda k:np.concatenate([Z[f'{r}_{k}'] for r in roots])
cab,rad,comp,pd,deg=g('cable'),g('rad'),g('comp'),g('pathd'),g('deg')
V=np.concatenate([Z[f'{r}_V'] for r in roots])
rt=np.concatenate([np.full(len(Z[f'{r}_cable']),r) for r in roots])
off={}; c=0
for r in roots: off[r]=c; c+=len(Z[f'{r}_cable'])
eu=np.load('eu_vert.npy')
M=A[(A['is_merge']==1)&(A['d_skel']<2000)]
lab=np.zeros(len(rad),bool)
lab[np.array([off[r]+v for r,v in zip(M['root'],M['vert'])])]=True
print('vertices',len(lab),'positive',int(lab.sum()),'base rate %.4f'%lab.mean(),flush=True)
X=np.c_[rad,(comp==2).astype(float),pd/1000.,eu/1000.,deg,V[:,0]/1000.,V[:,1]/1000.,V[:,2]/1000.]
names=['radius_nm','is_axon','path_soma_um','euclid_soma_um','degree','x_um','y_depth_um','z_um']
gkf=GroupKFold(n_splits=5)
def cv(cols,it=150):
    o=np.zeros(len(lab))
    for tr,te in gkf.split(X,lab,groups=rt):
        m=HistGradientBoostingClassifier(max_iter=it,random_state=0,max_bins=64)
        m.fit(X[tr][:,cols],lab[tr]); o[te]=m.predict_proba(X[te][:,cols])[:,1]
    return o
t=time.time(); oof=cv(list(range(len(names))))
auc=roc_auc_score(lab,oof); k=int(lab.sum()); order=np.argsort(-oof)
print(f'FULL heldout-cell AUC {auc:.3f}  precision@top-{k} {lab[order[:k]].mean():.3f} base {lab.mean():.4f} lift {lab[order[:k]].mean()/lab.mean():.2f}x  ({time.time()-t:.0f}s)',flush=True)
for f in [0.02,0.05,0.10,0.20,0.30]:
    kk=int(f*len(lab)); oo=order[:kk]
    print(f'  top {f*100:4.0f}% of cable: recall {lab[oo].sum()/lab.sum():.3f}  precision {lab[oo].mean():.3f}  lift {lab[oo].mean()/lab.mean():.2f}x',flush=True)
np.save('oof.npy',oof); np.save('lab.npy',lab)
for i,n in enumerate(names):
    keep=[j for j in range(len(names)) if j!=i]
    print(f'  ablate {n:16s} AUC {roc_auc_score(lab,cv(keep,80)):.3f}',flush=True)
print(f'  radius ALONE     AUC {roc_auc_score(lab,cv([0],80)):.3f}',flush=True)
print(f'  radius+is_axon   AUC {roc_auc_score(lab,cv([0,1],80)):.3f}',flush=True)
print('DONE',flush=True)
