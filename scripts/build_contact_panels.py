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
import argparse, glob, json, signal, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

class _TimeoutSentinel(Exception):
    """Raised by the per-cell alarm when a volume read hangs."""


R = Path("/Users/wgray13/projects/neuronauts")
# v117 is the base segmentation whose objects we assemble; supervoxel
# identity must be read at its timestamp, not at head.
V117_TS = datetime.fromtimestamp(1623399000, tz=timezone.utc)
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
    ap.add_argument("--only", choices=("both", "cut", "whole"), default="both",
                    help="build only cut-centred or only terminal panels")
    ap.add_argument("--per-cell-timeout", type=int, default=420,
                    help="abandon a cell whose volume read hangs (seconds)")
    ap.add_argument("--out", default=str(R / "data/external/panels"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    z = np.load(R / "data/substrate/c100um/object_clouds_mip5.npz", allow_pickle=False)
    obj, ptr, pos = z["object_id"], z["node_ptr"], z["pos_nm"]
    rowi = {int(a): k for k, a in enumerate(obj.tolist())}

    def pts(a):
        k = rowi.get(int(a))
        return pos[int(ptr[k]):int(ptr[k + 1])] if k is not None else np.empty((0, 3))

    import caveclient
    from cloudvolume import CloudVolume
    cl = caveclient.CAVEclient("minnie65_public")
    # Read OBJECTS, not supervoxels. The volume stores supervoxel ids, but
    # agglomerate=True resolves them to v117 objects server-side: measured at
    # the same cost as doing it ourselves (8.0 s vs 8.1 s on a 12.5M-voxel box)
    # and identical on 100.0000% of voxels. Doing it ourselves is what
    # introduced the 21%-coverage defect, since our supervoxel map came from a
    # mip-5 read and silently dropped every voxel it did not know.
    cv = CloudVolume(cl.chunkedgraph.cloudvolume_path, mip=2, use_https=True,
                     progress=False, fill_missing=True,
                     agglomerate=True, timestamp=V117_TS)
    res = np.asarray(cv.resolution, float)

    cards = [json.load(open(f)) for f in sorted(glob.glob(str(R / "data/external/cell_cards/*.json")))
             if not Path(f).name.startswith("_")]
    cards = [c for c in cards if c.get("coverage", {}).get("graph")]
    need = [c for c in cards if not c["structure"]["already_whole"]]
    whole = [c for c in cards if c["structure"]["already_whole"]]
    n_w = max(1, args.cells // 3)
    todo = need[: args.cells - n_w] + whole[:n_w]
    if args.only == "cut":
        todo = [c for c in todo if not c["structure"]["already_whole"]]
    elif args.only == "whole":
        todo = [c for c in todo if c["structure"]["already_whole"]]
    print(f"{len(todo)} panels: {len(todo)-n_w} needing joins, {n_w} already whole", flush=True)

    for c in todo:
      try:
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
            # The farthest interior point is NOT a terminal -- it is where the
            # interior mask clipped the arbor. EXP-076 measured the seed's cloud
            # continuing past it in 28 of 35 cells by a median of 2,303 nm. A
            # real cable end has nothing beyond it: walking outward from the
            # soma, no seed points lie further along that direction nearby.
            outward = cand_pts - soma      # not `out`: that is the output Path
            nrm = np.linalg.norm(outward, axis=1, keepdims=True)
            dirs = outward / np.maximum(nrm, 1.0)
            tip_ok = np.zeros(len(cand_pts), dtype=bool)
            tree_seed = cKDTree(Ps)
            for i in range(len(cand_pts)):
                nb = Ps[tree_seed.query_ball_point(cand_pts[i], r=3000.0)]
                if len(nb) < 3:
                    continue
                # anything beyond this point, along the outward direction?
                tip_ok[i] = not np.any((nb - cand_pts[i]) @ dirs[i] > 500.0)
            if not tip_ok.any():
                print(f"  {key}: no true cable end interior to the cube, skipping", flush=True)
                continue
            ends = cand_pts[tip_ok]
            ctr = ends[int(np.argmax(np.linalg.norm(ends - soma, axis=1)))]

        t0 = time.time()

        def _alarm(sig, frm):
            raise _TimeoutSentinel()

        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(args.per_cell_timeout)
        lo = np.floor((ctr - HALF_NM) / res).astype(int)
        hi = np.ceil((ctr + HALF_NM) / res).astype(int)
        vol = np.asarray(cv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])[..., 0]
        nz = np.nonzero(vol)
        P2 = (np.stack(nz, 1) + lo) * res
        R2 = vol[nz]                       # already v117 object ids
        Sv = P2[R2 == seed]
        if len(Sv) < 10:
            print(f"  {key}: seed absent from box", flush=True); continue

        # --- how does the SEED end here? ---
        # EXP-075 showed candidate geometry cannot say whether to stop. This asks
        # a different question, about the seed alone: a severed process keeps its
        # caliber right to the cut face, while a genuine terminal tapers to a tip
        # or closes into a bouton. Caliber is profiled along the seed's own axis
        # out to its last voxel near this point.
        aS_all = axis_of(Sv[np.linalg.norm(Sv - ctr, axis=1) < LOCAL_NM])
        end_ratio = np.nan
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
                end_ratio = float(tip / back)   # end_drop was exactly 1-this; dropped

        sub = Sv if len(Sv) <= MAXPTS else Sv[np.linspace(0, len(Sv) - 1, MAXPTS).astype(int)]
        st = cKDTree(sub)
        # Group voxels by object ONCE. Scanning `R2 == a` per
        # candidate is O(candidates x voxels); with identity no longer eroded
        # that is 2,445 x 12.5M and took 630 s a panel.
        order = np.argsort(R2, kind="stable")
        R2s, P2s = R2[order], P2[order]
        uniq, starts = np.unique(R2s, return_index=True)
        stops = np.append(starts[1:], len(R2s))
        span = {int(o): (int(a_), int(b_)) for o, a_, b_ in zip(uniq.tolist(), starts, stops)}

        vox_nm3 = float(np.prod(res))

        def caliber_span(a_, b_):
            """Equivalent radius of the object's local cross-section, in nm.

            Measured from its own voxels. With identity no longer eroded this
            needs no level-2 cache lookup and no supervoxel bookkeeping.
            """
            n = b_ - a_
            if n <= 0:
                return np.nan
            P = P2s[a_:b_]
            ax = axis_of(P)
            if ax is None:
                return float(np.cbrt(n * vox_nm3))
            t = P @ ax
            length = max(float(t.max() - t.min()), float(res[0]))
            return float(np.sqrt(n * vox_nm3 / length / np.pi))

        rec = []
        for a in [int(x) for x in uniq.tolist() if int(x) != seed]:
            a_, b_ = span[a]
            Q = P2s[a_:b_]
            # never Q[::len(Q)//4000][:4000] -- when the stride rounds to 1
            # that keeps the first 4,000 raster-ordered voxels, a slab at the
            # low-x end of the object rather than a sample of it
            Qs = Q if len(Q) <= 4000 else Q[np.linspace(0, len(Q) - 1, 4000).astype(int)]
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
            rec.append((a, gap, al, co, int(b_ - a_), caliber_span(a_, b_), a in tgt))
        s_a, s_b = span.get(int(seed), (0, 0))
        np.savez(dest,
                 obj=np.array([r[0] for r in rec], dtype=np.uint64),
                 gap_nm=np.array([r[1] for r in rec], dtype=np.float32),
                 along=np.array([r[2] for r in rec], dtype=np.float32),
                 collin=np.array([r[3] for r in rec], dtype=np.float32),
                 n_vox=np.array([r[4] for r in rec], dtype=np.int64),
                 cal_cand=np.array([r[5] for r in rec], dtype=np.float32),
                 in_target=np.array([r[6] for r in rec], dtype=bool),
                 seed=np.uint64(seed), cal_seed=np.float32(caliber_span(s_a, s_b)),
                 end_ratio=np.float32(end_ratio),
                 already_whole=bool(c["structure"]["already_whole"]))
        signal.alarm(0)
        n_t = sum(r[6] for r in rec)
        print(f"  {key}: {len(rec)} candidates, {n_t} in target, "
              f"seed caliber {caliber_span(s_a, s_b):.0f}nm  [{time.time()-t0:.0f}s]", flush=True)
      except _TimeoutSentinel:
        signal.alarm(0)
        print(f"  {key}: volume read exceeded {args.per_cell_timeout}s, skipped", flush=True)
      except Exception as exc:
        signal.alarm(0)
        import os, traceback
        if os.environ.get("PANEL_DEBUG"):
            traceback.print_exc()
        print(f"  {key}: {type(exc).__name__}: {str(exc)[:90]}, skipped", flush=True)


if __name__ == "__main__":
    main()
