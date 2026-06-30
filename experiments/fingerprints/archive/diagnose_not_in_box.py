"""Why are ~half the true partners 'not_in_box'?  Dump a concrete site.

`measure_panel_recall` reports that ~92% of recall misses are 'not_in_box': no
historical fragment in the v117-painted box (other than the query's own)
resolves via ``frag2cur`` to the query's current root.  The box is NOT empty --
segmentation is dense -- so this script inspects a real not_in_box site to find
the actual mechanism: resolution miss (frag2cur majority vote lands elsewhere),
sparsity (partner too small to survive), or genuinely-absent.

For each not_in_box site it prints:
  - painted histogram (how many voxels are 0 vs each historical fragment)
  - q_cur (query current root) and frag2cur for the painted fragments
  - what is painted at pos_frag, and what frag2cur maps it to
  - the INDEPENDENT current root of the supervoxel actually at pos_frag
    (queried straight from the chunkedgraph) -- the ground-truth check
"""

from __future__ import annotations

import numpy as np

from ..cutface import v117_error_relink as v
from ..cutface import v117_reconstructed as r


def _sv_and_cur_at(cl, vol, pos_nm, ts, mip=1):
    """Supervoxel id and (historical, current) root at a world point, queried
    directly from the chunkedgraph -- ground truth, independent of the paint."""
    cv = r._sv_volume(cl, mip=mip)
    vox = np.asarray(vol.resolution_nm, float)
    ijk = (np.asarray(pos_nm, float) / vox).astype(int)
    sv = int(np.squeeze(np.asarray(cv[ijk[0]:ijk[0] + 1, ijk[1]:ijk[1] + 1, ijk[2]:ijk[2] + 1])))
    if sv == 0:
        return 0, 0, 0
    hist = int(cl.chunkedgraph.get_roots([sv], timestamp=ts)[0])
    cur = int(cl.chunkedgraph.get_roots([sv])[0])
    return sv, hist, cur


def diagnose(cl, ts, site, *, mip=1, radius_nm=2000.0):
    vol, frag2cur = r.fetch_v117_box(cl, ts, site.pos_main_nm, site.pos_frag_nm, radius_nm, mip)
    seg = vol.seg
    qa_id, _ = v._seg_id_at(vol, site.pos_main_nm)
    if qa_id == 0 or qa_id not in frag2cur:
        return None
    q_cur = frag2cur[qa_id]

    same_roots = [int(s) for s in np.unique(seg)
                  if int(s) not in (0, qa_id) and frag2cur.get(int(s)) == q_cur]
    if same_roots:
        return None                                  # this one IS in box; skip

    ids, counts = np.unique(seg, return_counts=True)
    order = np.argsort(counts)[::-1]
    tot = seg.size
    print(f"\n=== not_in_box site: gap={np.linalg.norm(np.asarray(site.pos_frag_nm)-np.asarray(site.pos_main_nm)):.0f}nm "
          f"box={seg.shape} ({tot} vox) ===")
    print(f"  query: hist={qa_id}  q_cur={q_cur}")
    print(f"  painted fragments (top 8 by voxels):")
    for i in order[:8]:
        sid = int(ids[i])
        tag = "BACKGROUND(0)" if sid == 0 else ("QUERY" if sid == qa_id else "")
        print(f"    hist={sid:<22} vox={int(counts[i]):>8} ({100*counts[i]/tot:4.1f}%)  "
              f"-> cur={frag2cur.get(sid)}  {tag}")
    n_zero = int(counts[ids == 0][0]) if (ids == 0).any() else 0
    print(f"  background(0) voxels: {n_zero} ({100*n_zero/tot:.1f}%)")

    # what is painted at the frag-side point, and the ground-truth current root there
    fseg, _ = v._seg_id_at(vol, site.pos_frag_nm)
    print(f"  painted at pos_frag: hist={fseg} -> cur={frag2cur.get(int(fseg))}")
    try:
        sv, h, c = _sv_and_cur_at(cl, vol, site.pos_frag_nm, ts, mip=mip)
        match = "*** SHARES q_cur ***" if c == q_cur else "different current root"
        print(f"  chunkedgraph at pos_frag: sv={sv} hist={h} cur={c}   ({match})")
    except Exception as e:
        print(f"  chunkedgraph lookup failed: {e}")
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=200)
    ap.add_argument("--neurons", type=int, default=6)
    ap.add_argument("--want", type=int, default=4, help="how many not_in_box sites to dump")
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--radius-nm", type=float, default=2000.0)
    ap.add_argument("--max-sites", type=int, default=10)
    args = ap.parse_args()

    cl = v._client()
    ts = cl.chunkedgraph.get_oldest_timestamp()
    roots, _ = v.find_split_neurons(cl, n_scan=args.n_scan)
    roots = roots[:args.neurons]

    shown = 0
    for rt in roots:
        try:
            ss = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=args.radius_nm, max_sites=args.max_sites)
        except Exception:
            continue
        for s in ss:
            try:
                if diagnose(cl, ts, s, mip=args.mip, radius_nm=args.radius_nm):
                    shown += 1
            except Exception as e:
                print(f"  (site failed: {e})")
            if shown >= args.want:
                return
    print(f"\n[done] dumped {shown} not_in_box sites")


if __name__ == "__main__":
    main()
