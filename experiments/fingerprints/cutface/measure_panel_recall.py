"""Panel recall: how often is the true partner even *in* the candidate panel?

The correction top-1 numbers (geometry 0.645, combiner 0.758) are computed only
over sites that survive ``site_faces_bands`` -- which drops any site where the
true partner is absent from the proximity+cone panel (``v117_artifact_bands.py``
line 98, ``if not any(is_true)``).  That makes them *correction-given-candidates*
numbers, conditional on the candidate generator already succeeding.

This script measures the missing half: over real v117 false-split sites with a
valid query, what fraction have the true partner present in the panel.  That
recall is the ceiling; ``end_to_end = recall * correction_top1`` is the honest
deployed figure.  Every ranker shares the same filtered set, so the *relative*
comparison is unaffected -- only the absolute scale.
"""

from __future__ import annotations

import json

import numpy as np

from . import v117_error_relink as v
from . import v117_reconstructed as r
from .v117_artifact_bands import _band_face


def _nearest_same_root_nm(vol, frag2cur, q_cur, qa_id, pmain):
    """Min distance (nm) from the query point to any *other* fragment that shares
    the current root -- i.e. the closest true-partner voxel anywhere in the box,
    regardless of the proximity radius.  Returns (dist_nm or None, exists).

    Uses the current-root-painted ``curseg`` (direct per-voxel identity) when the
    box carries one; otherwise falls back to historical-root -> majority vote."""
    vox = np.asarray(vol.resolution_nm, float)
    origin = np.asarray(vol.origin_vox, float)
    curseg = getattr(vol, "curseg", None)
    if curseg is not None:
        # partner voxels: current root == q_cur but not the query's own fragment
        coords = np.argwhere((curseg == np.uint64(q_cur)) & (vol.seg != np.uint64(qa_id)))
        if not len(coords):
            return None, False
        pts = (origin + coords + 0.5) * vox
        return float(np.min(np.linalg.norm(pts - pmain, axis=1))), True
    seg = vol.seg
    same_ids = [sid for sid in np.unique(seg)
                if int(sid) != 0 and int(sid) != qa_id and frag2cur.get(int(sid)) == q_cur]
    if not same_ids:
        return None, False
    best = None
    for sid in same_ids:
        coords = np.argwhere(seg == sid)            # voxel indices [N,3]
        if not len(coords):
            continue
        pts = (origin + coords + 0.5) * vox
        dn = float(np.min(np.linalg.norm(pts - pmain, axis=1)))
        best = dn if best is None else min(best, dn)
    return best, True


def site_partner_status(cl, ts, site, *, mip=1, slab=2, radius_nm=2000.0,
                        direction_cone_deg=45.0, min_vox=40, sigma=2.0):
    """Replicate site_faces_bands panel construction; report partner presence
    plus diagnostics on why a partner was missed.

    Returns a dict with 'status' in:
      'no_query'    query supervoxel invalid (no scoreable site)
      'not_a_split' query fragment reaches pos_frag -> no real break (discarded)
      'no_prox'     no proximity candidates at all
      'too_few'     <3 buildable candidate faces (panel too thin to score)
      'present'     true partner is in the panel  (correction gets a chance)
      'absent'      true partner fell outside proximity / unbuildable (recall miss)
    and (when status in present/absent) diagnostic fields:
      site_gap_nm        the L2-graph gap for this site (pos_main -> pos_frag)
      n_faces            buildable candidate faces in the panel
      partner_dist_nm    nearest same-root voxel distance (None if not in box)
      partner_in_box     was any same-root fragment present in the box at all
      miss_reason        for 'absent': 'not_in_box' | 'out_of_radius' | 'unbuildable'

    'not_a_split' sites are excluded from the recall denominator -- they are not
    real false-splits, so failing to "fix" them is not a miss.
    """
    vol, frag2cur = r.fetch_v117_box(cl, ts, site.pos_main_nm, site.pos_frag_nm, radius_nm, mip)
    nz = vol.em.shape[2]
    qa_id, idx_main = v._seg_id_at(vol, site.pos_main_nm)
    if qa_id == 0 or qa_id not in frag2cur:
        return {"status": "no_query"}
    # FIX 1: query identity from the actual scanned root, not a box-local vote.
    q_cur = int(site.root) if getattr(site, "root", None) else frag2cur[qa_id]
    fseg, _ = v._seg_id_at(vol, site.pos_frag_nm)   # what occupies the partner location

    za = max(min(v._z_index(vol, site.pos_main_nm[2]), nz - slab), 0)
    q = _band_face(vol.em, vol.seg, za, za + slab, qa_id, sigma)
    if q is None:
        return {"status": "no_query"}

    prox = v._proximity_candidates(vol, idx_main, radius_nm, qa_id, min_vox)
    if not prox:
        return {"status": "no_prox"}
    vox = np.asarray(vol.resolution_nm, float)
    origin = np.asarray(vol.origin_vox, float)
    pmain = np.asarray(site.pos_main_nm, float)
    pfrag = np.asarray(site.pos_frag_nm, float)
    tangent = np.asarray(site.tangent_nm, float)
    tn = np.linalg.norm(tangent)
    cone_cos = np.cos(np.deg2rad(direction_cone_deg)) if direction_cone_deg else None
    curseg = getattr(vol, "curseg", None)

    def _cur_of(sid, nv):                            # direct per-candidate lookup
        if curseg is not None:
            c = int(curseg[int(nv[0]), int(nv[1]), int(nv[2])])
            if c != 0:
                return c
        return int(frag2cur.get(int(sid), 0))

    n_faces, partner_in = 0, False
    for sid, (nv, _) in prox.items():
        same = _cur_of(sid, nv) == q_cur
        d = (origin + nv + 0.5) * vox - pmain
        dn = float(np.linalg.norm(d))
        if cone_cos is not None and tn > 1e-6 and not same:
            if dn > 1e-6 and abs(float(d @ tangent) / (dn * tn)) < cone_cos:
                continue
        zc = max(min(int(nv[2]) - slab // 2, nz - slab), 0)
        bf = _band_face(vol.em, vol.seg, zc, zc + slab, sid, sigma)
        if bf is None:
            continue
        n_faces += 1
        if same:
            partner_in = True
    if n_faces < 3:
        return {"status": "too_few", "n_faces": n_faces}

    site_gap = float(np.linalg.norm(pfrag - pmain))
    pdist, in_box = _nearest_same_root_nm(vol, frag2cur, q_cur, qa_id, pmain)
    base = {"site_gap_nm": site_gap, "n_faces": n_faces,
            "partner_dist_nm": pdist, "partner_in_box": in_box}
    if partner_in:
        return {"status": "present", **base}
    # no distinct same-root partner.  Query fragment reaching pos_frag => the
    # L2-flagged break was not real -> discard (not a recall miss).
    if int(fseg) == int(qa_id):
        return {"status": "not_a_split", **base}
    # otherwise a genuine miss -- classify why
    if not in_box:
        reason = "not_in_box"
    elif pdist is not None and pdist > radius_nm:
        reason = "out_of_radius"
    else:
        reason = "unbuildable"          # in box & within radius but face/min_vox failed
    return {"status": "absent", "miss_reason": reason, **base}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-scan", type=int, default=300)
    ap.add_argument("--neurons", type=int, default=60)
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--radius-nm", type=float, default=2000.0)
    ap.add_argument("--max-sites", type=int, default=10)
    ap.add_argument("--correction-top1", type=float, default=0.758,
                    help="combiner top-1 to multiply by recall for end-to-end")
    ap.add_argument("--out", default="experiments/fingerprints/panel_recall_metrics.json")
    args = ap.parse_args()

    cl = v._client()
    ts = cl.chunkedgraph.get_oldest_timestamp()
    roots, _ = v.find_split_neurons(cl, n_scan=args.n_scan)
    roots = roots[:args.neurons]
    print(f"[recall] {len(roots)} false-split neurons", flush=True)

    from collections import Counter
    counts = Counter()
    miss_reasons = Counter()
    present_dist, present_gap, absent_dist, absent_gap = [], [], [], []
    for n, rt in enumerate(roots):
        try:
            ss = v.sites_from_l2_graph(cl, rt, ts, max_gap_nm=args.radius_nm, max_sites=args.max_sites)
        except Exception:
            continue
        for s in ss:
            try:
                d = site_partner_status(cl, ts, s, mip=args.mip, radius_nm=args.radius_nm)
            except Exception:
                d = {"status": "no_query"}
            st = d["status"]
            counts[st] += 1
            if st == "present":
                if d.get("partner_dist_nm") is not None:
                    present_dist.append(d["partner_dist_nm"])
                present_gap.append(d["site_gap_nm"])
            elif st == "absent":
                miss_reasons[d["miss_reason"]] += 1
                if d.get("partner_dist_nm") is not None:
                    absent_dist.append(d["partner_dist_nm"])
                absent_gap.append(d["site_gap_nm"])
        if counts:
            print(f"  {n + 1}/{len(roots)}  {dict(counts)}  miss={dict(miss_reasons)}", flush=True)

    present = counts["present"]
    absent = counts["absent"]
    not_a_split = counts["not_a_split"]
    scoreable = present + absent           # real splits with a buildable panel
    recall = present / scoreable if scoreable else 0.0
    end_to_end = recall * args.correction_top1

    def _summ(a):
        a = np.asarray(a, float)
        if not len(a):
            return None
        return {"n": int(len(a)), "median": float(np.median(a)),
                "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
                "max": float(a.max())}

    out = {"mip": args.mip, "radius_nm": args.radius_nm,
           "counts": dict(counts),
           "scoreable_sites": scoreable,
           "not_a_split_discarded": not_a_split,
           "panel_recall": recall,
           "correction_top1": args.correction_top1,
           "end_to_end_top1": end_to_end,
           "miss_reasons": dict(miss_reasons),
           "present_partner_dist_nm": _summ(present_dist),
           "present_site_gap_nm": _summ(present_gap),
           "absent_partner_dist_nm": _summ(absent_dist),
           "absent_site_gap_nm": _summ(absent_gap)}
    print(f"\nnot-a-split discarded (query continuous through pos_frag): {not_a_split}")
    print(f"Panel recall (partner present | real split with panel): {recall:.3f}"
          f"  ({present}/{scoreable})")
    print(f"End-to-end = recall x correction_top1 = {recall:.3f} x {args.correction_top1:.3f}"
          f" = {end_to_end:.3f}")
    print(f"Miss reasons: {dict(miss_reasons)}")
    print(f"  present partner dist nm: {_summ(present_dist)}")
    print(f"  absent  partner dist nm: {_summ(absent_dist)}")
    print(f"  absent  site gap   nm:   {_summ(absent_gap)}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
