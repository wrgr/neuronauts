"""Re-cut the 40-cell truth as root processes: for each cached cell whose soma is
an evaluable in-cube seed, the target is the seed's own in-box component."""
import glob, json, sys
import numpy as np
sys.path.insert(0,"/Users/wgray13/projects/neuronauts")
from neuronauts.harness.box_truth import box_components, seeded_target, spanning_target
from neuronauts.harness.labels import TIER_NONE, load_labels
R="/Users/wgray13/projects/neuronauts/"; D=R+"data/external/soma_viz/"
S="/private/tmp/claude-501/-Users-wgray13-projects-neuronauts/8c2bcfd4-b48d-453f-ae78-fb9ed1b00ae7/scratchpad/"
LO=np.array([613000.,541000.,810000.]); HI=np.array([713000.,641000.,910000.])
seeds={int(s["root_v1822"]):int(s["v117_fragment"]) for s in json.load(open(D+"seed_census.json"))["seeds"] if s["evaluable"]}
g=np.load(R+"data/substrate/geom/objgeom_kall.npz",allow_pickle=False)
ol2,opos,oa,optr=g["l2_id"],g["pos_nm"],g["atom_id"],g["node_ptr"]
o=np.argsort(ol2); ol2s,oposs=ol2[o],opos[o]; l2atom=np.repeat(oa,np.diff(optr))[o]
a=np.load(S+"connective_l2_attrs.npz",allow_pickle=False); c=np.argsort(a["l2_id"]); cl2s,cposs=a["l2_id"][c],a["pos_nm"][c]
lab=load_labels(R+"data/substrate/c100um/labels_v1822.npz")
t=np.load(R+"data/substrate/topology/kall.npz",allow_pickle=False); atoms=t["atom_id"]
i=lab.index_of(atoms); has=i>=0
own=np.zeros(len(atoms),np.int64); pure=np.zeros(len(atoms),bool); tr=np.full(len(atoms),TIER_NONE,np.int8)
own[has]=lab.owner[i[has]].astype(np.int64); pure[has]=lab.pure[i[has]]; tr[has]=lab.owner_tier[i[has]]
keep=pure&(tr>TIER_NONE)&(own>0); ids,ow=atoms[keep],own[keep]
rows=[]; n_cached=0
for f in sorted(glob.glob(R+"data/external/cell_l2_graphs/*.npz")):
    cell=int(f.split("/")[-1].split(".")[0]); n_cached+=1
    E=np.load(f,allow_pickle=False)["edges"]
    if not len(E): continue
    members=set(ids[ow==cell].tolist()); nodes=np.unique(E); pos={int(v):k for k,v in enumerate(nodes.tolist())}
    P=np.full((len(nodes),3),np.nan); j=np.clip(np.searchsorted(ol2s,nodes),0,len(ol2s)-1); h=ol2s[j]==nodes; P[h]=oposs[j[h]]
    m=~h
    if m.any():
        k=np.clip(np.searchsorted(cl2s,nodes[m]),0,len(cl2s)-1); h2=cl2s[k]==nodes[m]; P[np.flatnonzero(m)[h2]]=cposs[k[h2]]
    # Refuse to score a cell whose graph we cannot place. Every atom-atom path runs
    # through non-population nodes, so a cell with unpositioned nodes falls apart
    # into singletons and reads as "nothing joinable" -- a data gap, not a result.
    positioned = float(np.isfinite(P).all(axis=1).mean())
    if positioned < 0.95:
        rows.append({"cell":str(cell),"skipped":True,"positioned_frac":round(positioned,4)}); continue
    natom=np.where(h,l2atom[j],np.uint64(0)); frag=np.array([int(x) if int(x) in members else 0 for x in natom.tolist()],np.int64)
    ei=np.array([[pos[int(x)],pos[int(y)]] for x,y in E.tolist()])
    bt=box_components(ei,P,frag,LO,HI)
    seed=seeds.get(cell); tgt=seeded_target(bt,seed) if seed else []
    rows.append({"cell":str(cell),"positioned_frac":round(positioned,4),"fragments":bt.n_fragments,"components":len(bt.components),
                 "soma_in_cube":seed is not None,"seed_fragment":str(seed) if seed else None,
                 "seed_is_labelled_here":bool(seed and any(seed in comp for comp in bt.components)),
                 "seeded_target":len(tgt),"largest":len(bt.largest),
                 "all_components_links":sum(len(x)-1 for x in spanning_target(bt))})
skipped=[r for r in rows if r.get("skipped")]
print(f"cells skipped for unpositioned nodes (<95% placed): {len(skipped)}")
soma=[r for r in rows if not r.get("skipped") and r["soma_in_cube"]]
print(f"cached cells {n_cached}; with an evaluable soma seed in the cube: {len(soma)}")
print(f"{'cell':>20} {'frags':>5} {'comps':>5} {'seeded tgt':>10} {'largest':>7} {'seed==largest':>13}")
for r in soma:
    print(f"{r['cell']:>20} {r['fragments']:>5} {r['components']:>5} {r['seeded_target']:>10} {r['largest']:>7} {str(r['seeded_target']==r['largest']):>13}")
tot=sum(r["fragments"] for r in soma); st=sum(r["seeded_target"] for r in soma); lg=sum(r["largest"] for r in soma)
print(f"\nover soma-seeded cells: fragments {tot}, in seeded target {st} ({st/max(tot,1):.0%}), in largest {lg} ({lg/max(tot,1):.0%})")
print(f"seed's component is the largest in {sum(r['seeded_target']==r['largest'] for r in soma)}/{len(soma)} cells")
json.dump({"n_cached":n_cached,"n_soma_seeded":len(soma),"rows":rows},open(D+"seeded_recut.json","w"),indent=1)
