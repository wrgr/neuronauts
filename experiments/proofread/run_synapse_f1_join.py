"""Driver: synapse-pair F1 before->after joining L2 fragments with the follower."""
from __future__ import annotations
import json, os, datetime as dt
import numpy as np


def relabel(vol, mapping):
    """Relabel a volume of ids via dict mapping (0->0), vectorised."""
    ids = np.unique(vol)
    lut = np.array([mapping.get(int(i), 0) for i in ids], dtype=np.int64)
    idx = np.searchsorted(ids, vol)
    return lut[idx]


def main():
    from caveclient import CAVEclient
    from cloudvolume import CloudVolume
    from experiments.proofread.synapse_f1_join import build_join_edges, synapse_f1_curve
    tok = os.environ["token"]
    cl = CAVEclient("minnie65_public", auth_token=tok)
    seg_src = cl.info.segmentation_source()
    ts = cl.materialize.get_timestamp(1822)
    if ts.tzinfo is None: ts = ts.replace(tzinfo=dt.timezone.utc)

    c = np.array([733592., 513592, 700000.]); half = 5000.0
    bbox = (tuple(c - half), tuple(c + half))
    # supervoxel volume (agglomerate=False)
    cv = CloudVolume(seg_src, mip=2, use_https=True, progress=False, fill_missing=True,
                     secrets={"token": tok}, agglomerate=False)
    vox = tuple(int(x) for x in cv.resolution)
    lo = [int(bbox[0][i] / vox[i]) for i in range(3)]; hi = [int(bbox[1][i] / vox[i]) for i in range(3)]
    b = cv.bounds
    lo = [max(lo[i], int(b.minpt[i])) for i in range(3)]; hi = [min(hi[i], int(b.maxpt[i])) for i in range(3)]
    print("fetching supervoxel volume ...", flush=True)
    sv_vol = np.squeeze(cv[tuple(slice(lo[i], hi[i]) for i in range(3))]).astype(np.int64)
    print("sv volume", sv_vol.shape, flush=True)

    # synapses in box (for the F1 metric)
    cl.version = 117
    syn_vox = np.array([4., 4., 40.]); slo = ((c - half) / syn_vox).astype(int); shi = ((c + half) / syn_vox).astype(int)
    df = cl.materialize.query_table("synapses_pni_2",
        filter_spatial_dict={"ctr_pt_position": [slo.tolist(), shi.tolist()]}, split_positions=False)
    pre_sv = df["pre_pt_supervoxel_id"].values.astype(np.int64)
    post_sv = df["post_pt_supervoxel_id"].values.astype(np.int64)
    print("synapses", len(df), flush=True)

    # map all supervoxels (volume + synapse) -> L2 and -> v1822 root
    allsv = np.unique(np.concatenate([np.unique(sv_vol), pre_sv, post_sv])); allsv = allsv[allsv > 0]
    print(f"mapping {len(allsv)} supervoxels -> L2 and v1822 root ...", flush=True)
    l2_map, root_map = {0: 0}, {0: 0}
    for s in range(0, len(allsv), 50000):
        bb = allsv[s:s + 50000].tolist()
        l2 = cl.chunkedgraph.get_roots(bb, timestamp=ts, stop_layer=2)
        rt = cl.chunkedgraph.get_roots(bb, timestamp=ts)
        l2_map.update({int(k): int(v) for k, v in zip(bb, l2.tolist())})
        root_map.update({int(k): int(v) for k, v in zip(bb, rt.tolist())})

    l2_vol = relabel(sv_vol, l2_map)
    root_of_l2 = {}  # L2 id -> its v1822 root (majority via the sv mapping)
    for sv in allsv.tolist():
        l2 = l2_map.get(sv, 0); rt = root_map.get(sv, 0)
        if l2 > 0 and rt > 0: root_of_l2[l2] = rt

    # synapse endpoints -> L2 and root
    def lab(sv, m): return np.array([m.get(int(x), 0) for x in sv])
    l2_syn = np.concatenate([lab(pre_sv, l2_map), lab(post_sv, l2_map)])
    root_syn = np.concatenate([lab(pre_sv, root_map), lab(post_sv, root_map)])

    edges = build_join_edges(l2_vol, vox, root_of_l2, gaps=(1, 2), traj_k=3, min_area=8)
    rows = synapse_f1_curve(l2_syn, root_syn, edges)
    before = rows[0]
    best = max((r for r in rows if r["stage"] == "after"), key=lambda r: r["F1"])
    # operating point: highest F1 with join_precision >= 0.95
    hp = [r for r in rows if r["stage"] == "after" and r["join_precision"] >= 0.95]
    op = max(hp, key=lambda r: r["F1"]) if hp else None
    print(f"\nBEFORE (L2 fragments):  P={before['P']:.3f} R={before['R']:.3f} F1={before['F1']:.3f}")
    print(f"AFTER  best F1:         F1={best['F1']:.3f} (R={best['R']:.3f} P={best['P']:.3f}) "
          f"thr={best['thr']:.2f} joins={best['n_joins']} join_P={best['join_precision']:.3f}")
    if op:
        print(f"AFTER  join_P>=0.95:    F1={op['F1']:.3f} (R={op['R']:.3f}) thr={op['thr']:.2f} "
              f"joins={op['n_joins']} join_P={op['join_precision']:.3f}")
    os.makedirs("out", exist_ok=True)
    json.dump({"before": before, "best": best, "op_joinP95": op, "rows": rows},
              open("out/synapse_f1_join.json", "w"), indent=2, default=float)
    print("SAVED out/synapse_f1_join.json")


if __name__ == "__main__":
    main()
