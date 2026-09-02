"""Where each proofreading edit happened: operation details in batches of 200,
coordinates converted from chunkedgraph voxels (8x8x40 nm) to nm at read time so
the stored file carries one frame only."""
import glob, json, time
from pathlib import Path
import numpy as np
from caveclient import CAVEclient
D = Path("/Users/wgray13/projects/neuronauts/data/external/edit_history")
VOX = np.array([8.0, 8.0, 40.0])
cg = CAVEclient("minnie65_public").chunkedgraph
files = sorted(D.glob("*.json")); t0 = time.time(); done = 0
for f in files:
    rec = json.load(open(f))
    if rec.get("located"): done += 1; continue
    ids = [o["operation_id"] for o in rec["ops"]]
    det = {}
    for i in range(0, len(ids), 200):
        for attempt in range(3):
            try: det.update(cg.get_operation_details(ids[i:i+200])); break
            except Exception as ex:
                if attempt == 2: print(f"  {rec['root']}: batch {i} {type(ex).__name__}", flush=True)
                time.sleep(2.0 * (attempt + 1))
    for o in rec["ops"]:
        d = det.get(o["operation_id"]) or det.get(str(o["operation_id"])) or {}
        pts = (d.get("sink_coords") or []) + (d.get("source_coords") or [])
        o["edit_points_nm"] = (np.asarray(pts, float) * VOX).round().astype(int).tolist() if pts else []
        o["n_removed_edges"] = len(d.get("removed_edges") or []); o["n_added_edges"] = len(d.get("added_edges") or [])
    rec["located"] = True; rec["coord_frame"] = "nm (converted from 8x8x40 chunkedgraph voxels)"
    json.dump(rec, open(f, "w")); done += 1
    if done % 10 == 0: print(f"  {done}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
print(f"done: located {done}/{len(files)} in {time.time()-t0:.0f}s", flush=True)
