"""Seed census: which cell bodies are in the cube, which are neurons, and which
give an evaluable root process.

Frames matter here (this repo has three). Annotation tables store pt_position in
4x4x40 nm voxels; the chunkedgraph uses 8x8x40. The spatial filter below is in
the TABLE's voxels, and every returned position is converted to nm and asserted
inside the cube -- the count is validated, not trusted (CLAUDE.md section 2).
"""
import json, sys, time
import numpy as np
from caveclient import CAVEclient
sys.path.insert(0, "/Users/wgray13/projects/neuronauts")
from neuronauts.data import lineage as L
from neuronauts.harness.labels import TIER_NONE, load_labels

OUT = "/Users/wgray13/projects/neuronauts/data/external/soma_viz/seed_census.json"
CENTRE = np.array([663000., 591000., 860000.]); HALF = 50000.
LO, HI = CENTRE - HALF, CENTRE + HALF
V117_TS = 1623399000
c = CAVEclient("minnie65_public")
rec = {"cube_lo_nm": LO.tolist(), "cube_hi_nm": HI.tolist()}

# --- nuclei in the cube, filter in the table's own voxel frame, verify in nm ---
tab = "nucleus_detection_v0"
res = np.array(c.materialize.get_table_metadata(tab).get("voxel_resolution", [4, 4, 40]), float)
rec["nucleus_table"] = tab; rec["nucleus_voxel_nm"] = res.tolist()
lo_v, hi_v = np.floor(LO / res).astype(int), np.ceil(HI / res).astype(int)
t0 = time.time()
df = c.materialize.query_table(tab, filter_spatial_dict={"pt_position": [lo_v.tolist(), hi_v.tolist()]},
                               desired_resolution=[1, 1, 1])
pos = np.stack(df["pt_position"].to_numpy()).astype(float)
inside = np.all((pos >= LO) & (pos <= HI), axis=1)
assert inside.all(), f"{(~inside).sum()} returned nuclei fall outside the cube in nm -- frame bug"
df = df[inside].reset_index(drop=True)
rec["n_nuclei_in_cube"] = int(len(df)); rec["query_s"] = round(time.time() - t0, 1)
print(f"nuclei in cube: {len(df):,} (all verified inside in nm; {rec['query_s']}s)", flush=True)

# --- cell type, if a table exists at this version ---
ct_tables = [t for t in c.materialize.get_tables() if "cell_type" in t or "celltype" in t]
rec["cell_type_tables_available"] = ct_tables
ctype = {}
for t in ct_tables:
    try:
        cdf = c.materialize.query_table(t, filter_in_dict={"pt_root_id": df["pt_root_id"].unique().tolist()[:5000]})
        col = next((x for x in cdf.columns if "class" in x or "cell_type" in x), None)
        if col is None: continue
        for r, v in zip(cdf["pt_root_id"], cdf[col]): ctype.setdefault(int(r), str(v))
        rec["cell_type_table_used"] = t; rec["cell_type_column"] = col
        break
    except Exception as ex:
        rec.setdefault("cell_type_errors", []).append(f"{t}: {type(ex).__name__}")
df["cell_type"] = [ctype.get(int(r), "unknown") for r in df["pt_root_id"]]
from collections import Counter
rec["cell_type_counts"] = dict(Counter(df["cell_type"]))
print("cell types:", rec["cell_type_counts"], flush=True)

# --- soma -> v117 fragment, and is it an evaluable seed? ---
sv = df["pt_supervoxel_id"].astype(np.uint64).to_numpy()
v117 = np.asarray(L.roots_at(sv.tolist(), V117_TS), np.uint64)
df["v117_fragment"] = v117
lab = load_labels("/Users/wgray13/projects/neuronauts/data/substrate/c100um/labels_v1822.npz")
z = np.load("/Users/wgray13/projects/neuronauts/data/substrate/topology/kall.npz", allow_pickle=False)
pop = set(z["atom_id"].tolist())
li = lab.index_of(v117); has = li >= 0
in_pop = np.array([int(x) in pop for x in v117.tolist()])
pure = np.zeros(len(df), bool); tier = np.zeros(len(df), np.int8); owner = np.zeros(len(df), np.uint64)
pure[has] = lab.pure[li[has]]; tier[has] = lab.owner_tier[li[has]]; owner[has] = lab.owner[li[has]]
evaluable = in_pop & pure & (tier > TIER_NONE) & (owner > 0)
owner_matches = evaluable & (owner == df["pt_root_id"].astype(np.uint64).to_numpy())
rec.update(n_resolved_v117=int((v117 > 0).sum()), n_fragment_in_population=int(in_pop.sum()),
           n_evaluable_seed=int(evaluable.sum()),
           n_evaluable_and_owner_is_this_nucleus_root=int(owner_matches.sum()))
for k in ("n_resolved_v117", "n_fragment_in_population", "n_evaluable_seed", "n_evaluable_and_owner_is_this_nucleus_root"):
    print(f"  {k}: {rec[k]:,}", flush=True)
rec["seeds"] = [{"nucleus_id": int(a), "root_v1822": int(b), "v117_fragment": int(f), "cell_type": ct,
                 "pos_nm": p.tolist(), "evaluable": bool(e)}
                for a, b, f, ct, p, e in zip(df["id"], df["pt_root_id"], v117, df["cell_type"], pos[inside], evaluable)]
json.dump(rec, open(OUT, "w"), indent=1); print("wrote", OUT, flush=True)
