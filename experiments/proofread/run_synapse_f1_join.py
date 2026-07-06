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
    rows = synapse_f1_curve(l2_syn, root_syn, edges)          # greedy (threshold union-find)
    before = rows[0]

    # ---- merge-aware constrained joining ----
    from experiments.proofread.merge_aware_join import (
        fragment_types, constrained_union_find, apply_partition)
    from experiments.proofread.synapse_f1_join import _pair_f1
    pre_l2 = lab(pre_sv, l2_map); post_l2 = lab(post_sv, l2_map)   # pre side=axon, post=dend
    types, pc, qc, contam = fragment_types(pre_l2, post_l2, dom=0.6, min_syn=2)
    # per-L2 max cross-section area (µm²) -> caliber + soma flags
    vox_um2 = (vox[0] * vox[1]) / 1e6
    area_um2 = {}
    for z in range(l2_vol.shape[2]):
        ids, cnts = np.unique(l2_vol[:, :, z], return_counts=True)
        for i, cc in zip(ids.tolist(), cnts.tolist()):
            if i > 0:
                area_um2[i] = max(area_um2.get(i, 0.0), cc * vox_um2)
    soma_frags = {f for f, a in area_um2.items() if a > 15.0}
    print(f"[merge-aware] axon={sum(v=='axon' for v in types.values())} "
          f"dend={sum(v=='dend' for v in types.values())} "
          f"contaminated={len(contam)} soma_frags={len(soma_frags)}", flush=True)

    def ma_curve(**flags):
        out = []
        for t in np.linspace(0.05, 0.8, 30):
            dsu, rej = constrained_union_find(
                edges, pre_count=pc, post_count=qc, soma_frags=soma_frags,
                contaminated=contam, area_of=area_um2, threshold=float(t), **flags)
            P, R, F = _pair_f1(apply_partition(dsu, l2_syn), root_syn)
            jp = rej["committed_correct"] / rej["committed"] if rej["committed"] else float("nan")
            out.append({"thr": float(t), "P": P, "R": R, "F1": F, "join_P": jp,
                        "joins": rej["committed"], "rej": rej})
        return out

    full = ma_curve()
    ablations = {
        "no_ad": ma_curve(use_ad=False),
        "no_soma": ma_curve(use_soma=False),
        "no_caliber": ma_curve(use_caliber=False),
        "no_quarantine": ma_curve(use_quarantine=False),
        "no_vetoes": ma_curve(use_ad=False, use_soma=False, use_caliber=False, use_quarantine=False),
    }

    def best(curve, key="F1"):
        return max(curve, key=lambda r: r[key])
    def best_at_joinP(curve, p):
        hp = [r for r in curve if r["join_P"] >= p]
        return max(hp, key=lambda r: r["F1"]) if hp else None

    g_best = max((r for r in rows if r["stage"] == "after"), key=lambda r: r["F1"])
    m_best = best(full)
    print(f"\nBEFORE (L2 fragments):   F1={before['F1']:.3f} (P={before['P']:.3f} R={before['R']:.3f})")
    print(f"GREEDY   best F1:        F1={g_best['F1']:.3f} (R={g_best['R']:.3f}) join_P={g_best['join_precision']:.3f}")
    print(f"MERGE-AWARE best F1:     F1={m_best['F1']:.3f} (R={m_best['R']:.3f}) join_P={m_best['join_P']:.3f} thr={m_best['thr']:.2f}")
    for p in (0.90, 0.95):
        gm = best_at_joinP([{**r, "join_P": r["join_precision"]} for r in rows if r["stage"] == "after"], p)
        mm = best_at_joinP(full, p)
        gs = f"F1={gm['F1']:.3f} R={gm['R']:.3f}" if gm else "unreached"
        ms = f"F1={mm['F1']:.3f} R={mm['R']:.3f}" if mm else "unreached"
        print(f"  join_P>={p}:  greedy {gs:22s}  merge-aware {ms}")
    print("ablation (best F1 with each veto removed):")
    for name, cv in ablations.items():
        b = best(cv); print(f"  {name:14s} F1={b['F1']:.3f} join_P={b['join_P']:.3f} joins={b['joins']}")

    os.makedirs("out", exist_ok=True)
    json.dump({"before": before, "greedy_rows": rows, "merge_aware": full,
               "ablations": ablations,
               "types_summary": {"axon": sum(v == "axon" for v in types.values()),
                                 "dend": sum(v == "dend" for v in types.values()),
                                 "contaminated": len(contam), "soma_frags": len(soma_frags)}},
              open("out/merge_aware_join.json", "w"), indent=2, default=float)
    print("SAVED out/merge_aware_join.json")


if __name__ == "__main__":
    main()
