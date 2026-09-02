"""pcg_skel skeleton (v4: radius + compartment) for every evaluable seed cell,
cached one file per root, same call the viewer's skeletons came from."""
import json, time
from pathlib import Path
import numpy as np
from caveclient import CAVEclient
D = Path("/Users/wgray13/projects/neuronauts/data/external/cell_skeletons"); D.mkdir(exist_ok=True)
seeds = json.load(open("/Users/wgray13/projects/neuronauts/data/external/soma_viz/seed_census.json"))["seeds"]
roots = sorted({int(s["root_v1822"]) for s in seeds if s["evaluable"]})
c = CAVEclient("minnie65_public"); KEEP = ("vertices", "edges", "radius", "compartment", "lvl2_ids")
ok = fail = 0; t0 = time.time()
for n, r in enumerate(roots, 1):
    f = D / f"{r}_skv4.npz"
    if f.exists(): ok += 1; continue
    try:
        sk = c.skeleton.get_skeleton(r, skeleton_version=4, output_format="dict")
        np.savez_compressed(f, **{k: np.asarray(sk[k]) for k in KEEP if k in sk}); ok += 1
    except Exception as ex:
        fail += 1; print(f"  {r}: {type(ex).__name__} {str(ex)[:100]}", flush=True)
    if n % 10 == 0: print(f"  {n}/{len(roots)}  ok {ok} fail {fail}  {time.time()-t0:.0f}s", flush=True)
print(f"done: ok {ok} fail {fail} in {time.time()-t0:.0f}s", flush=True)
