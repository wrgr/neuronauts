"""Cache the level-2 graph of every evaluable soma-seeded cell, in the same
place and format EXP-071 uses (data/external/cell_l2_graphs/<root>.npz), so
the seeded re-cut can run on all of them, not just the 8 already cached."""
import json, sys, time
from pathlib import Path
import numpy as np, requests
sys.path.insert(0, "/Users/wgray13/projects/neuronauts")
from neuronauts.data import lineage as L
from neuronauts.harness.substrate import region_bounds
D = Path("/Users/wgray13/projects/neuronauts/data/external/cell_l2_graphs"); D.mkdir(exist_ok=True)
_lo, _hi, seg = region_bounds([663.0, 591.0, 860.0], 200.0)
bstr = "_".join(f"{int(seg[i][0])}-{int(seg[i][1])}" for i in range(3))
seeds = json.load(open("/Users/wgray13/projects/neuronauts/data/external/soma_viz/seed_census.json"))["seeds"]
roots = sorted({int(s["root_v1822"]) for s in seeds if s["evaluable"]})
todo = [r for r in roots if not (D / f"{r}.npz").exists()]
print(f"{len(roots)} evaluable seed cells, {len(todo)} not yet cached", flush=True)
ok = fail = 0; t0 = time.time()
for n, cell in enumerate(todo, 1):
    url = f"{L.CG_SERVER}/segmentation/api/v1/table/{L.SEG_TABLE}/node/{cell}/lvl2_graph"
    E = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=L._headers(L.DEFAULT_TOKEN), params={"bounds": bstr}, timeout=180)
            if r.status_code == 200:
                e = r.json().get("edge_graph", [])
                E = np.asarray(e, np.uint64).reshape(-1, 2) if len(e) else np.zeros((0, 2), np.uint64); break
            if r.status_code in (400, 404): break
        except Exception: pass
        time.sleep(2.0 * (attempt + 1))
    if E is None: fail += 1; print(f"  {cell}: FAILED", flush=True); continue
    np.savez_compressed(D / f"{cell}.npz", edges=E); ok += 1
    if n % 10 == 0: print(f"  {n}/{len(todo)}  ok {ok} fail {fail}  {time.time()-t0:.0f}s", flush=True)
print(f"done: ok {ok} fail {fail} in {time.time()-t0:.0f}s", flush=True)
