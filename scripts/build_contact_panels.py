"""Contact panels: the candidate set a soma-seeded grower actually faces.

EXP-074 established that distance narrows the field about tenfold and then
stops -- with correct geometry at 32 nm, some 30-180 objects sit as close to a
seed as its own continuation does. That panel, not the cube, is what a grammar
has to discriminate over.

For each seed this writes one panel: every object with a voxel inside the box,
its true closest approach to the seed at mip 2, and the geometry a grammar
needs to score it.

  gap_nm    closest approach, real voxels at 32x32x40 nm (not centroids)
  along     |cos| between the seed's local axis and the direction to the
            candidate. A severed process lies ALONG the parent's axis; an
            unrelated passing process lies beside it.
  collin    |cos| between the two local axes. A continuation is collinear;
            a crossing process is not.
  cal_seed  caliber (max distance transform, nm) of the seed near the contact
  cal_cand  caliber of the candidate near the contact
  in_target whether the object belongs to the seed's in-box component

Caliber and axis come from the level-2 cache, which stores them computed at the
segmentation's native resolution -- a max_dt of 136 nm is finer than one mip 5
voxel, so these are not derived from the coarse clouds. Supervoxels in the box
map to level-2 nodes in one batched call, so a panel costs seconds rather than
the hours a cube-wide pass would.

Already-whole cells are included deliberately: they have no target, and a
grammar that cannot decline to grow on them is not solving the task.

    python scripts/build_contact_panels.py --cells 24
"""
import argparse, glob, json, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

R = Path("/Users/wgray13/projects/neuronauts")
CUBE_C = np.array([663.0, 591.0, 860.0]) * 1000.0
LO_NM, HI_NM = CUBE_C - 50_000.0, CUBE_C + 50_000.0
HALF_NM, LOCAL_NM, MAXPTS = 4000.0, 1500.0, 20000


def axis_of(P):
    if len(P) < 3:
        return None
    return np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=24)
    ap.add_argument("--out", default=str(R / "data/external/panels"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    z = np.load(R / "data/substrate/c100um/object_clouds_mip5.npz", allow_pickle=False)
    obj, ptr, pos = z["object_id"], z["node_ptr"], z["pos_nm"]
    rowi = {int(a): k for k, a in enumerate(obj.tolist())}

    def pts(a):
        k = rowi.get(int(a))
        return pos[int(ptr[k]):int(ptr[k + 1])] if k is not None else np.empty((0, 3))

    sv = np.load(R / "data/substrate/c100um/objects_v117_mip5_svmap.npz", allow_pickle=False)
    SV, RT = sv["sv"], sv["root"]
    o = np.argsort(SV); SV, RT = SV[o], RT[o]

    def to_root(x):
        i = np.searchsorted(SV, x); i[i >= len(SV)] = 0
        return np.where(SV[i] == x, RT[i], 0)

    import caveclient
    from cloudvolume import CloudVolume
    cl = caveclient.CAVEclient("minnie65_public")
    cv = CloudVolume(cl.chunkedgraph.cloudvolume_path, mip=2, use_https=True,
                     progress=False, fill_missing=True, agglomerate=False)
    res = np.asarray(cv.resolution, float)

    cards = [json.load(open(f)) for f in sorted(glob.glob(str(R / "data/external/cell_cards/*.json")))
             if not Path(f).name.startswith("_")]
    cards = [c for c in cards if c.get("coverage", {}).get("graph")]
    need = [c for c in cards if not c["structure"]["already_whole"]]
    whole = [c for c in cards if c["structure"]["already_whole"]]
    n_w = max(1, args.cells // 3)
    todo = need[: args.cells - n_w] + whole[:n_w]
    print(f"{len(todo)} panels: {len(todo)-n_w} needing joins, {n_w} already whole", flush=True)

    for c in todo:
        cell = str(c["cell"]); key = cell[-8:]
        dest = out / f"panel_{key}.npz"
        if dest.exists():
            print(f"  {key}: exists, skipping", flush=True); continue
        seed = int(c["seed"]["v117_fragment"])
        tgt = set(c["structure"]["seeded_target"]) - {seed}
        Ps = pts(seed)
        if not len(Ps):
            print(f"  {key}: no seed cloud", flush=True); continue
        # centre on the seed/target contact where there is one, else on the soma
        Pt = [pts(t) for t in tgt if len(pts(t))]
        if Pt:
            Pt = np.vstack(Pt)
            d, j = cKDTree(Pt).query(Ps, k=1)
            m = int(np.argmin(d)); ctr = (Ps[m] + Pt[int(j[m])]) / 2.0
        else:
            # An already-whole cell has no target to centre on. Centring on its
            # soma would place its box on 2 um of cell body while every
            # join-needing box sits on a 150-450 nm process -- an abstention
            # test that any caliber threshold passes without learning anything.
            # Centre instead where the grower would actually ask the question:
            # a terminal of the arbor, far from the soma, where the honest
            # answer is that nothing continues.
            # ...and it must be an INTERIOR terminal. The farthest point from
            # the soma is usually where the cell exits the cube, and there a
            # continuation genuinely exists -- it just lies in tissue we never
            # enumerated. Scoring that as "nothing continues here" would label
            # a real join as an abstention. Six of the first eight whole-cell
            # panels sat within 1 um of a cube face this way. Restrict to
            # terminals a full box-width inside every face.
            soma = np.asarray(c["seed"]["pos_nm"], float)
            inside = np.all((Ps - LO_NM > 2 * HALF_NM) & (HI_NM - Ps > 2 * HALF_NM), axis=1)
            if not inside.any():
                print(f"  {key}: no interior terminal, skipping", flush=True)
                continue
            cand_pts = Ps[inside]
            ctr = cand_pts[int(np.argmax(np.linalg.norm(cand_pts - soma, axis=1)))]

        t0 = time.time()
        lo = np.floor((ctr - HALF_NM) / res).astype(int)
        hi = np.ceil((ctr + HALF_NM) / res).astype(int)
        vol = np.asarray(cv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])[..., 0]
        nz = np.nonzero(vol); svid = vol[nz]
        rt = to_root(svid); ok = rt > 0
        P2 = (np.stack(nz, 1)[ok] + lo) * res
        R2 = rt[ok]; SV2 = svid[ok]
        Sv = P2[R2 == seed]
        if len(Sv) < 10:
            print(f"  {key}: seed absent from box", flush=True); continue

        # supervoxel -> level-2 node, one batched call, then caliber per node
        usv = np.unique(SV2)
        l2 = np.zeros(len(usv), dtype=np.uint64)
        for i in range(0, len(usv), 50_000):
            l2[i:i + 50_000] = np.asarray(
                cl.chunkedgraph.get_roots(usv[i:i + 50_000], stop_layer=2), dtype=np.uint64)
        cal = {}
        ul2 = np.unique(l2[l2 > 0]).tolist()
        for i in range(0, len(ul2), 4000):
            for k, v in cl.l2cache.get_l2data([int(x) for x in ul2[i:i + 4000]]).items():
                if v and v.get("max_dt_nm") is not None:
                    cal[int(k)] = float(v["max_dt_nm"])
        sv2l2 = dict(zip(usv.tolist(), l2.tolist()))

        def caliber(mask):
            vals = [cal[sv2l2[s]] for s in np.unique(SV2[mask]).tolist()
                    if sv2l2.get(s) in cal]
            return float(np.median(vals)) if vals else np.nan

        # --- how does the SEED end here? ---
        # EXP-075 showed candidate geometry cannot say whether to stop. This asks
        # a different question, about the seed alone: a severed process keeps its
        # caliber right to the cut face, while a genuine terminal tapers to a tip
        # or closes into a bouton. Caliber is profiled along the seed's own axis
        # out to its last voxel near this point.
        aS_all = axis_of(Sv[np.linalg.norm(Sv - ctr, axis=1) < LOCAL_NM])
        end_ratio = end_drop = np.nan
        if aS_all is not None:
            loc = Sv[np.linalg.norm(Sv - ctr, axis=1) < 2 * LOCAL_NM]
            t = (loc - ctr) @ aS_all
            # the end is whichever side runs out of seed first
            side = 1.0 if abs(t.max()) < abs(t.min()) else -1.0
            t = t * side
            edge = t.max()
            def cal_at(lo_t, hi_t):
                m = (t >= lo_t) & (t < hi_t)
                # voxels in a slab -> equivalent radius of its cross-section
                return np.sqrt(m.sum() * float(np.prod(res)) / max(hi_t - lo_t, 1.0) / np.pi)
            tip = cal_at(edge - 300.0, edge)
            back = cal_at(edge - 1300.0, edge - 1000.0)
            if back > 0:
                end_ratio = float(tip / back)
                end_drop = float(1.0 - tip / back)

        sub = Sv if len(Sv) <= MAXPTS else Sv[:: len(Sv) // MAXPTS][:MAXPTS]
        st = cKDTree(sub)
        others = np.unique(R2[R2 != seed])
        rec = []
        for a in others.tolist():
            mask = R2 == a
            Q = P2[mask]
            Qs = Q if len(Q) <= 4000 else Q[:: len(Q) // 4000][:4000]
            dq, jq = st.query(Qs, k=1)
            m = int(np.argmin(dq)); gap = float(dq[m])
            cp = (Qs[m] + sub[int(jq[m])]) / 2.0
            Sl = sub[np.linalg.norm(sub - cp, axis=1) < LOCAL_NM]
            Cl = Q[np.linalg.norm(Q - cp, axis=1) < LOCAL_NM]
            aS, aC = axis_of(Sl), axis_of(Cl)
            if aS is None or aC is None:
                al = co = 0.0
            else:
                u = Cl.mean(0) - Sl.mean(0); n = np.linalg.norm(u)
                al = abs(float(aS @ (u / n))) if n > 0 else 0.0
                co = abs(float(aS @ aC))
            rec.append((a, gap, al, co, int(mask.sum()), caliber(mask), a in tgt))
        seed_mask = R2 == seed
        np.savez(dest,
                 obj=np.array([r[0] for r in rec], dtype=np.uint64),
                 gap_nm=np.array([r[1] for r in rec], dtype=np.float32),
                 along=np.array([r[2] for r in rec], dtype=np.float32),
                 collin=np.array([r[3] for r in rec], dtype=np.float32),
                 n_vox=np.array([r[4] for r in rec], dtype=np.int64),
                 cal_cand=np.array([r[5] for r in rec], dtype=np.float32),
                 in_target=np.array([r[6] for r in rec], dtype=bool),
                 seed=np.uint64(seed), cal_seed=np.float32(caliber(seed_mask)),
                 end_ratio=np.float32(end_ratio), end_drop=np.float32(end_drop),
                 already_whole=bool(c["structure"]["already_whole"]))
        n_t = sum(r[6] for r in rec)
        print(f"  {key}: {len(rec)} candidates, {n_t} in target, "
              f"seed caliber {caliber(seed_mask):.0f}nm  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
