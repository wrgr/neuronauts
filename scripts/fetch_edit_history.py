"""Edit history per seed cell: the tabular change log (every merge/split a
proofreader applied to reach this root), cached one file per root."""
import json, sys, time
from pathlib import Path
import numpy as np
from caveclient import CAVEclient
D = Path("/Users/wgray13/projects/neuronauts/data/external/edit_history"); D.mkdir(exist_ok=True)
seeds = json.load(open("/Users/wgray13/projects/neuronauts/data/external/soma_viz/seed_census.json"))["seeds"]
roots = sorted({int(s["root_v1822"]) for s in seeds if s["evaluable"]})
cg = CAVEclient("minnie65_public").chunkedgraph
ok = fail = 0; t0 = time.time()
for n, r in enumerate(roots, 1):
    f = D / f"{r}.json"
    if f.exists(): ok += 1; continue
    try:
        df = cg.get_tabular_change_log([r])[r]
        rows = [{"operation_id": int(a), "timestamp_ms": int(b), "user_id": str(c), "is_merge": bool(d),
                 "before_root_ids": [int(x) for x in e], "after_root_ids": [int(x) for x in g]}
                for a, b, c, d, e, g in zip(df.operation_id, df.timestamp, df.user_id, df.is_merge,
                                             df.before_root_ids, df.after_root_ids)]
        json.dump({"root": r, "n_ops": len(rows), "n_merges": int(df.is_merge.sum()),
                   "n_splits": int((~df.is_merge).sum()), "ops": rows}, open(f, "w"))
        ok += 1
    except Exception as ex:
        fail += 1; print(f"  {r}: {type(ex).__name__} {str(ex)[:120]}", flush=True)
    if n % 10 == 0: print(f"  {n}/{len(roots)}  ok {ok} fail {fail}  {time.time()-t0:.0f}s", flush=True)
print(f"done: ok {ok} fail {fail} in {time.time()-t0:.0f}s", flush=True)
