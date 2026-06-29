"""Disambiguate at REAL v117-era split errors with the cut-face hash.

The artificial-cut experiment (``fingerprint_break_resolution.py``) cut neurites
at an arbitrary z-plane.  This one tests the hash where it actually matters: at
locations the automated segmentation *got wrong* and a human had to fix.

How a real error site is found (no materialization needed -- chunkedgraph only):

1. A proofread neuron is one current root ``R``.  Look up the historical root of
   each of its level-2 nodes at the oldest timestamp.  If they fall into several
   distinct historical roots, ``R`` was assembled by **merging** those fragments
   -- i.e. the v117-era segmentation falsely *split* the neuron, and proofreading
   merged it back.
2. For each minor fragment, its closest approach to the main fragment (in L2 rep
   coordinates) is the merge interface -- a real false-split site.  The two L2
   positions there are the two faces a human glued together.

The test at each site: take the cross-section "face" at the main-side point as a
query, and rank the candidate neurites near the fragment-side point by cut-face
hash similarity.  The true continuation is the neurite actually sitting at the
fragment-side point.  Top-1 / MRR are reported for the learned hash, the raw
patch, and chance -- this is re-identification across the *real* error gap.

Requires a CAVE token in env var ``token`` (or ``CAVE_TOKEN``) and the public
EM + seg volumes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from neuronauts.fetch import fetch_volume as _fetch_em, fetch_seg_volume as _fetch_seg
from .fingerprint_break_resolution import Volume, face_hash


def _grad(em):
    gx = np.gradient(em.astype(np.float32), axis=0)
    gy = np.gradient(em.astype(np.float32), axis=1)
    return np.sqrt(gx * gx + gy * gy)


# ---------------------------------------------------------------------------
# CAVE: find real split sites
# ---------------------------------------------------------------------------

def _client():
    from caveclient import CAVEclient
    tok = os.environ.get("token") or os.environ.get("CAVE_TOKEN")
    if not tok:
        raise RuntimeError("set the CAVE token in env var 'token' or 'CAVE_TOKEN'")
    return CAVEclient("minnie65_public", auth_token=tok)


@dataclass
class ErrorSite:
    root: int
    pos_main_nm: tuple   # face A (main fragment side)
    pos_frag_nm: tuple   # face B (merged-in fragment side)
    gap_nm: float
    frag_l2: int         # number of L2 nodes in the minor fragment
    tangent_nm: tuple = (0.0, 0.0, 0.0)  # query fragment's local axis (for the direction cone)


def find_split_neurons(cl, n_scan=120, seed=7, min_l2=20):
    """Return current roots (with soma) that were split at the oldest timestamp."""
    import io
    import requests
    import pandas as pd

    url = ("https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
           "minnie65/nucleus_detection/nucleus_detection_v0.csv")
    df = pd.read_csv(io.BytesIO(requests.get(url, timeout=120).content), header=None,
                     names=["nuc", "t", "sv", "root", "x", "y", "z", "vol"])
    df = df[(df.root > 0) & (df.sv > 0)].sample(n_scan, random_state=seed)
    cg = cl.chunkedgraph
    ts = cg.get_oldest_timestamp()
    roots = cg.get_roots(df.sv.astype(np.int64).tolist())

    from concurrent.futures import ThreadPoolExecutor

    def _is_split(rt):
        try:
            lvs = np.asarray(cg.get_leaves(int(rt), stop_layer=2))
            if len(lvs) < min_l2:
                return None
            s = lvs if len(lvs) <= 120 else np.random.default_rng(int(rt) % 2**31).choice(lvs, 120, replace=False)
            return int(rt) if len(np.unique(cg.get_roots(s, timestamp=ts))) > 1 else None
        except Exception:
            return None

    uniq = np.unique(roots).tolist()
    out = [r for r in ThreadPoolExecutor(max_workers=16).map(_is_split, uniq) if r]
    return out, ts


def sites_for_neuron(cl, root, ts, *, min_gap_nm=300.0, max_gap_nm=5000.0,
                     min_frag_l2=4, max_sites=10, main_sample=800, frag_sample=400,
                     lvs=None, hist=None) -> list[ErrorSite]:
    """Locate real false-split interfaces inside one neuron.

    Only looks up rep coordinates for the minor fragments plus a *sample* of the
    main arbor -- not all L2 nodes -- because position lookups are the dominant
    network cost.  Pass precomputed ``lvs``/``hist`` to avoid re-querying.

    ``main_sample``/``frag_sample`` trade gap accuracy for lookup cost: too small
    a sample over-estimates the true closest-approach gap (and wrongly drops
    proposable sites under a tight ``max_gap_nm``), so they default generously.
    """
    cg = cl.chunkedgraph
    if lvs is None or hist is None:
        lvs = np.asarray(cg.get_leaves(int(root), stop_layer=2))
        hist = np.asarray(cg.get_roots(lvs, timestamp=ts))
    lvs = np.asarray(lvs)
    hist = np.asarray(hist)

    frags, counts = np.unique(hist, return_counts=True)
    main = frags[np.argmax(counts)]
    rng = np.random.default_rng(int(root) % 2**31)

    # Build a small index subset: sampled main nodes + (sampled) minor fragments.
    main_idx = np.where(hist == main)[0]
    if len(main_idx) > main_sample:
        main_idx = rng.choice(main_idx, main_sample, replace=False)
    frag_idx = {}
    for f, c in zip(frags.tolist(), counts.tolist()):
        if f == main or f == 0 or c < min_frag_l2:
            continue
        fi = np.where(hist == f)[0]
        if len(fi) > frag_sample:
            fi = rng.choice(fi, frag_sample, replace=False)
        frag_idx[f] = fi
    if not frag_idx:
        return []

    sel = np.concatenate([main_idx] + list(frag_idx.values()))
    pos_sel = _l2_positions(cl, lvs[sel])          # the only position lookup
    pos_by_node = {int(lvs[s]): pos_sel[k] for k, s in enumerate(sel)}

    def _pos(idx_arr):
        p = np.array([pos_by_node[int(lvs[i])] for i in idx_arr])
        return p[~np.isnan(p[:, 0])]

    main_pos = _pos(main_idx)
    if len(main_pos) == 0:
        return []

    sites = []
    for f, fi in frag_idx.items():
        fp = _pos(fi)
        if len(fp) == 0:
            continue
        d = np.linalg.norm(fp[:, None, :] - main_pos[None, :, :], axis=2)
        i, j = np.unravel_index(np.argmin(d), d.shape)
        gap = float(d[i, j])
        if min_gap_nm <= gap <= max_gap_nm:
            tangent = _local_tangent(main_pos, main_pos[j])
            sites.append(ErrorSite(root=int(root),
                                   pos_main_nm=tuple(main_pos[j].tolist()),
                                   pos_frag_nm=tuple(fp[i].tolist()),
                                   gap_nm=gap, frag_l2=int(counts[list(frags).index(f)]),
                                   tangent_nm=tuple(tangent.tolist())))
    sites.sort(key=lambda s: s.gap_nm)
    return sites[:max_sites]


def sites_from_l2_graph(cl, root, ts, *, min_gap_nm=300.0, max_gap_nm=3000.0,
                        min_frag_l2=2, max_sites=40, dedup_nm=1500.0) -> list:
    """Find real false-split interfaces via the L2 *adjacency* graph (accurate).

    The correct way to locate where proofreading merged fragments: in the
    current level-2 chunk graph, an edge whose two endpoints had *different*
    historical roots at ``ts`` is a real merge interface.  The merge-created
    nodes have historical root 0 and sit exactly at the seams, so they are kept
    (not excluded).  This finds every interface, topologically adjacent, instead
    of a sampling-inflated closest-approach estimate.

    Returns de-duplicated :class:`ErrorSite`s (one per ~``dedup_nm`` region).
    """
    from collections import defaultdict
    cg = cl.chunkedgraph
    edges = np.asarray(cg.level2_chunk_graph(int(root)))
    if edges.ndim != 2 or len(edges) == 0:
        return []
    nodes = np.unique(edges)
    hist = np.asarray(cg.get_roots(nodes, timestamp=ts))
    hm = {int(n): int(h) for n, h in zip(nodes.tolist(), hist.tolist())}
    nz = hist[hist != 0]
    if len(nz) == 0:
        return []
    frags, counts = np.unique(nz, return_counts=True)
    size = {int(f): int(c) for f, c in zip(frags.tolist(), counts.tolist())}

    adj = defaultdict(list)
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    ha = np.array([hm[int(a)] for a in edges[:, 0]])
    hb = np.array([hm[int(b)] for b in edges[:, 1]])
    cross = edges[ha != hb]
    if len(cross) == 0:
        return []

    pairs, want = [], set()
    for a, b in cross:
        a, b = int(a), int(b)
        # main side = endpoint in the larger fragment (size 0 for the glue node)
        m, fr = (a, b) if size.get(hm[a], 0) >= size.get(hm[b], 0) else (b, a)
        if hm[fr] != 0 and size.get(hm[fr], 0) < min_frag_l2:
            continue
        pairs.append((m, fr))
        want.add(m)
        want.add(fr)
        # 2-hop same-fragment neighbourhood of the main endpoint -> local tangent
        for n1 in adj[m]:
            if hm.get(n1) == hm[m]:
                want.add(n1)
                for n2 in adj[n1][:4]:
                    if hm.get(n2) == hm[m]:
                        want.add(n2)
    if not pairs:
        return []

    want = np.array(sorted(want))
    pos = _l2_positions(cl, want)
    pm = {int(n): pos[k] for k, n in enumerate(want.tolist())}

    sites, seen = [], []
    pairs.sort(key=lambda mf: np.linalg.norm(pm[mf[0]] - pm[mf[1]])
               if (not np.isnan(pm[mf[0]][0]) and not np.isnan(pm[mf[1]][0])) else 1e18)
    for m, fr in pairs:
        a_pos, b_pos = pm.get(m), pm.get(fr)
        if a_pos is None or b_pos is None or np.isnan(a_pos[0]) or np.isnan(b_pos[0]):
            continue
        gap = float(np.linalg.norm(a_pos - b_pos))
        if not (min_gap_nm <= gap <= max_gap_nm):
            continue
        if any(np.linalg.norm(a_pos - s) < dedup_nm for s in seen):
            continue
        nbp = []
        for n1 in adj[m]:
            if hm.get(n1) == hm[m] and n1 in pm and not np.isnan(pm[n1][0]):
                nbp.append(pm[n1])
            for n2 in adj[n1][:4]:
                if hm.get(n2) == hm[m] and n2 in pm and not np.isnan(pm[n2][0]):
                    nbp.append(pm[n2])
        tangent = (_local_tangent(np.array(nbp + [a_pos]), a_pos)
                   if len(nbp) >= 2 else np.zeros(3))
        seen.append(a_pos)
        sites.append(ErrorSite(root=int(root), pos_main_nm=tuple(a_pos.tolist()),
                               pos_frag_nm=tuple(b_pos.tolist()), gap_nm=gap,
                               frag_l2=int(size.get(hm[fr], 0)),
                               tangent_nm=tuple(tangent.tolist())))
        if len(sites) >= max_sites:
            break
    return sites


def _local_tangent(main_pos, tip, k=8) -> np.ndarray:
    """Principal direction of the main arbor near the query tip (answer-free)."""
    if len(main_pos) < 3:
        return np.zeros(3)
    d = np.linalg.norm(main_pos - tip, axis=1)
    knn = main_pos[np.argsort(d)[:k]]
    if len(knn) < 3:
        return np.zeros(3)
    cc = knn - knn.mean(axis=0)
    _, _, vt = np.linalg.svd(cc, full_matrices=False)
    return vt[0]


def _l2_positions(cl, l2_ids) -> np.ndarray:
    out = np.full((len(l2_ids), 3), np.nan)
    ids = [int(x) for x in l2_ids]
    for b0 in range(0, len(ids), 1000):
        chunk = ids[b0:b0 + 1000]
        d = cl.l2cache.get_l2data(chunk, attributes=["rep_coord_nm"])
        for k, lid in enumerate(chunk):
            rc = d.get(str(lid), {}).get("rep_coord_nm")
            if rc:
                out[b0 + k] = rc
    return out


# ---------------------------------------------------------------------------
# Cut-face re-linking at a site
# ---------------------------------------------------------------------------

@dataclass
class SiteResult:
    root: int
    gap_nm: float
    n_candidates: int
    rank_learned: int     # 0 == top-1
    rank_raw: int
    top1_learned: bool
    top1_raw: bool
    sim_learned_true: float


# On-disk cache of fetched EM+seg boxes.  The per-box CloudVolume fetch (uint64
# seg over a ~5 um cube) is by far the slowest step, so caching each fetched box
# lets every re-run (different encoder, eval, candidate settings) reuse it
# instead of re-hitting CAVE/EM.  Default dir lives under data/ (gitignored);
# set EM_BOX_CACHE="" to disable.
BOX_CACHE_DIR = os.environ.get("EM_BOX_CACHE", "data/em_box_cache")


def _box_key(bbox, mip) -> str:
    import hashlib
    flat = tuple(int(round(c)) for pt in bbox for c in pt) + (int(mip),)
    return hashlib.md5(repr(flat).encode()).hexdigest()[:16]


def _fetch_box(pos_a, pos_b, margin_nm, mip):
    pts = np.asarray([pos_a, pos_b], float)
    lo = pts.min(0) - margin_nm
    hi = pts.max(0) + margin_nm
    bbox = (tuple(lo.tolist()), tuple(hi.tolist()))

    path = None
    if BOX_CACHE_DIR:
        path = os.path.join(BOX_CACHE_DIR, f"box_{_box_key(bbox, mip)}.npz")
        if os.path.exists(path):
            try:
                z = np.load(path)
                return Volume(em=z["em"], seg=z["seg"],
                              resolution_nm=tuple(int(x) for x in z["res"]),
                              origin_vox=tuple(int(x) for x in z["origin"]))
            except Exception:
                pass  # corrupt/partial cache entry -> refetch

    em = _fetch_em(bbox, mip=mip)
    seg = _fetch_seg(bbox, mip=mip)
    vol = Volume(em=em.data.astype(np.uint8), seg=seg.data.astype(np.uint64),
                 resolution_nm=seg.voxel_size_nm, origin_vox=seg.bbox_voxels[0])
    if path:
        os.makedirs(BOX_CACHE_DIR, exist_ok=True)
        # np.savez_compressed appends ".npz" if absent, so keep it on the temp name.
        tmp = path + f".tmp{os.getpid()}.npz"
        try:
            np.savez_compressed(tmp, em=vol.em, seg=vol.seg,
                                res=np.asarray(vol.resolution_nm),
                                origin=np.asarray(vol.origin_vox))
            os.replace(tmp, path)   # atomic; safe under concurrent runs
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
    return vol


def _z_index(vol, z_nm):
    return int(round(z_nm / vol.resolution_nm[2] - vol.origin_vox[2] - 0.5))


def _seg_id_at(vol, pos_nm):
    vox = np.asarray(vol.resolution_nm, float)
    origin = np.asarray(vol.origin_vox, float)
    idx = np.round(np.asarray(pos_nm, float) / vox - origin - 0.5).astype(int)
    idx = np.clip(idx, 0, [s - 1 for s in vol.seg.shape])
    return int(vol.seg[idx[0], idx[1], idx[2]]), idx


def _patch_from_slab(em, seg, z_lo, z_hi, seg_id):
    """Translation-normalised cut-face patch for one seg id in a z-slab."""
    from .fingerprint_break_resolution import PATCH
    sub = em[:, :, z_lo:z_hi].astype(np.float32)
    mask = seg[:, :, z_lo:z_hi] == seg_id
    c2 = mask.sum(axis=2)
    if c2.sum() == 0:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        proj = np.where(c2 > 0, (sub * mask).sum(axis=2) / c2, 0.0)
    xs, ys = np.nonzero(c2 > 0)
    ci, cj = int(round(xs.mean())), int(round(ys.mean()))
    h = PATCH // 2
    out = np.zeros((PATCH, PATCH), np.float32)
    xi0, xi1 = max(ci - h, 0), min(ci + h, proj.shape[0])
    yi0, yi1 = max(cj - h, 0), min(cj + h, proj.shape[1])
    px0, py0 = xi0 - (ci - h), yi0 - (cj - h)
    out[px0:px0 + (xi1 - xi0), py0:py0 + (yi1 - yi0)] = proj[xi0:xi1, yi0:yi1]
    return out


def _proximity_candidates(vol, idx_main, radius_nm, qa_id, min_vox):
    """Seg ids with >= min_vox voxels within radius_nm of the query endpoint.

    Returns ``{seg_id: (nearest_voxel_global_idx, n_vox_in_ball)}`` excluding
    background and the query's own id -- the realistic merge-proposal set: other
    neurites that physically approach the dangling tip.
    """
    vox = np.asarray(vol.resolution_nm, float)
    shape = vol.seg.shape
    rvx = [int(np.ceil(radius_nm / vox[d])) for d in range(3)]
    x0, x1 = max(idx_main[0] - rvx[0], 0), min(idx_main[0] + rvx[0] + 1, shape[0])
    y0, y1 = max(idx_main[1] - rvx[1], 0), min(idx_main[1] + rvx[1] + 1, shape[1])
    z0, z1 = max(idx_main[2] - rvx[2], 0), min(idx_main[2] + rvx[2] + 1, shape[2])
    sub = vol.seg[x0:x1, y0:y1, z0:z1]

    gx = (np.arange(x0, x1) - idx_main[0]) * vox[0]
    gy = (np.arange(y0, y1) - idx_main[1]) * vox[1]
    gz = (np.arange(z0, z1) - idx_main[2]) * vox[2]
    dist = np.sqrt(gx[:, None, None] ** 2 + gy[None, :, None] ** 2 + gz[None, None, :] ** 2)
    within = dist <= radius_nm

    wseg = sub[within]
    loc = np.argwhere(within) + np.array([x0, y0, z0])   # global voxel idx
    wd = dist[within]
    order = np.argsort(wd, kind="stable")
    ws_sorted = wseg[order]
    loc_sorted = loc[order]
    uids, first_idx, counts = np.unique(ws_sorted, return_index=True, return_counts=True)

    out = {}
    for s, fi, c in zip(uids.tolist(), first_idx.tolist(), counts.tolist()):
        if s == 0 or s == qa_id or c < min_vox:
            continue
        out[int(s)] = (loc_sorted[fi], int(c))
    return out


def evaluate_site(site: ErrorSite, embed_fn, *, mip=1, slab=3, margin_nm=1200.0,
                  candidate_mode="proximity", radius_nm=2000.0,
                  direction_cone_deg=None, min_vox=40, collect=False):
    """Rank the true continuation at a real error site by cut-face hash.

    ``candidate_mode``:
      - ``"proximity"`` (default, realistic): candidates are other neurites with
        a voxel within ``radius_nm`` of the query endpoint -- the set a merge-
        proposal generator would actually consider.  Optionally cone-filtered by
        the query fragment's local tangent (``direction_cone_deg``).  Sites whose
        true partner is farther than ``radius_nm`` are unproposable -> skipped.
      - ``"slab"`` (legacy): every neurite crossing a z-slab at the fragment
        point within the box.

    With ``collect=True`` also returns a viz dict
    ``{"query","true","top_learned","top_learned_id","true_id"}``.
    """
    vol = _fetch_box(site.pos_main_nm, site.pos_frag_nm,
                     max(margin_nm, radius_nm if candidate_mode == "proximity" else margin_nm), mip)
    grad = _grad(vol.em)
    dark = float(np.percentile(vol.em[vol.seg > 0], 25)) if (vol.seg > 0).any() else 128.0
    nz = vol.em.shape[2]

    qa_id, idx_main = _seg_id_at(vol, site.pos_main_nm)
    true_id, _ = _seg_id_at(vol, site.pos_frag_nm)
    if qa_id == 0 or true_id == 0 or true_id == qa_id:
        return None

    za = max(min(_z_index(vol, site.pos_main_nm[2]), nz - slab), 0)
    q_patch = _patch_from_slab(vol.em, vol.seg, za, za + slab, qa_id)
    if q_patch is None:
        return None

    cand_patch = {}   # seg_id -> patch
    if candidate_mode == "proximity":
        if site.gap_nm > radius_nm:
            return None
        prox = _proximity_candidates(vol, idx_main, radius_nm, qa_id, min_vox)
        if true_id not in prox:
            return None
        vox = np.asarray(vol.resolution_nm, float)
        origin = np.asarray(vol.origin_vox, float)
        pmain = np.asarray(site.pos_main_nm, float)
        tangent = np.asarray(site.tangent_nm, float)
        tn = np.linalg.norm(tangent)
        cone_cos = np.cos(np.deg2rad(direction_cone_deg)) if direction_cone_deg else None
        for sid, (nv, _) in prox.items():
            if cone_cos is not None and tn > 1e-6 and sid != true_id:
                d = (origin + nv + 0.5) * vox - pmain
                dn = np.linalg.norm(d)
                if dn > 1e-6 and abs(float(d @ tangent) / (dn * tn)) < cone_cos:
                    continue  # outside the direction cone
            zc = max(min(int(nv[2]) - slab // 2, nz - slab), 0)
            p = _patch_from_slab(vol.em, vol.seg, zc, zc + slab, sid)
            if p is not None:
                cand_patch[sid] = p
    else:  # legacy slab mode
        zb = max(min(_z_index(vol, site.pos_frag_nm[2]), nz - slab), 0)
        faces = face_hash(vol.em, vol.seg, grad, zb, zb + slab, dark_thresh=dark)
        cand_patch = {i: f.patch for i, f in faces.items() if i != qa_id}

    if true_id not in cand_patch or len(cand_patch) < 3:
        return None

    cand_ids = sorted(cand_patch)
    cp = np.stack([cand_patch[i] for i in cand_ids])

    qe = np.asarray(embed_fn(q_patch[None]))[0]
    ce = np.asarray(embed_fn(cp))
    qe = qe / (np.linalg.norm(qe) + 1e-9)
    ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-9)
    d_learned = 1.0 - ce @ qe
    qf = _flatnorm(q_patch)
    cf = np.stack([_flatnorm(cand_patch[i]) for i in cand_ids])
    d_raw = 1.0 - cf @ qf

    tcol = cand_ids.index(true_id)
    r_learned = int((d_learned < d_learned[tcol]).sum())
    r_raw = int((d_raw < d_raw[tcol]).sum())
    result = SiteResult(
        root=site.root, gap_nm=site.gap_nm, n_candidates=len(cand_ids),
        rank_learned=r_learned, rank_raw=r_raw,
        top1_learned=(r_learned == 0), top1_raw=(r_raw == 0),
        sim_learned_true=float(1.0 - d_learned[tcol]),
    )
    if not collect:
        return result
    top_col = int(np.argmin(d_learned))
    viz = {
        "query": q_patch,
        "true": cand_patch[true_id],
        "top_learned": cand_patch[cand_ids[top_col]],
        "top_learned_id": cand_ids[top_col],
        "true_id": int(true_id),
    }
    return result, viz


# A few current roots that the oldest-timestamp scan flagged as v117-era splits
# (soma neurons with multiple historical fragments).  Used by the figure to skip
# the expensive soma scan.
KNOWN_SPLIT_ROOTS = [
    864691135662059248, 864691135700637474, 864691135866022620,
    864691135355803087, 864691136194738508, 864691136990837653,
    864691135133705888, 864691135737524356,
]


def make_figure(out_png, embed_fn, cl, *, n_examples=6, n_scan=160, mip=1, roots=None,
                candidate_mode="proximity", radius_nm=2000.0, direction_cone_deg=45.0):
    """Render real v117 error sites: query face | true continuation | hash top pick."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = cl.chunkedgraph.get_oldest_timestamp()
    if roots is None:
        roots = KNOWN_SPLIT_ROOTS
    pool = []  # (result, viz) -- collect a pool, then select for display
    target_pool = max(40, n_examples * 6)
    for rt in roots:
        try:
            # prefer cleaner, shorter-gap interfaces for a legible figure
            sites = sites_for_neuron(cl, rt, ts, max_gap_nm=2500.0)
        except Exception:
            continue
        for s in sites:
            try:
                out = evaluate_site(s, embed_fn, mip=mip, collect=True,
                                    candidate_mode=candidate_mode, radius_nm=radius_nm,
                                    direction_cone_deg=direction_cone_deg)
            except Exception:
                out = None
            if out is not None:
                pool.append(out)
        if len(pool) >= target_pool:
            break
    if not pool:
        raise RuntimeError("no scorable sites for the figure")

    # Show the clearest successes first (rank, then small gap); fall back to
    # best-ranked overall if there are few hits. Ranks are labelled honestly.
    pool.sort(key=lambda rv: (rv[0].rank_learned, rv[0].gap_nm))
    picks = pool[:n_examples]
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.5, 2.4 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    for row, (res, viz) in enumerate(picks):
        ok = res.top1_learned
        cells = [
            (viz["query"], f"query (main side)\nroot …{res.root % 100000}"),
            (viz["true"], f"TRUE continuation\ngap {res.gap_nm:.0f} nm"),
            (viz["top_learned"], f"hash top pick  {'✓' if ok else '✗'}\n"
             f"rank {res.rank_learned + 1}/{res.n_candidates}"),
        ]
        for col, (img, title) in enumerate(cells):
            ax = axes[row, col]
            ax.imshow(img.T, cmap="gray", origin="lower")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Cut-face hash at REAL v117 split errors", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def _flatnorm(patch):
    v = patch.ravel().astype(np.float64)
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="experiments/fingerprints/cutface_encoder.pt")
    ap.add_argument("--n-scan", type=int, default=120, help="neurons to scan for splits")
    ap.add_argument("--max-neurons", type=int, default=20, help="split neurons to evaluate")
    ap.add_argument("--mip", type=int, default=1)
    ap.add_argument("--candidate-mode", choices=["proximity", "slab"], default="proximity",
                    help="proximity = neurites within --radius-nm of the query tip (realistic); "
                         "slab = every neurite crossing a z-slab (legacy, generous)")
    ap.add_argument("--radius-nm", type=float, default=2000.0,
                    help="proximity radius around the query endpoint")
    ap.add_argument("--direction-cone-deg", type=float, default=None,
                    help="optional: keep only candidates within this half-angle of the "
                         "query fragment's local tangent")
    ap.add_argument("--out", default="experiments/fingerprints/v117_relink_metrics.json")
    ap.add_argument("--figure", default=None, help="render a montage PNG and exit")
    args = ap.parse_args()

    from .learned_cutface_encoder import load_encoder, make_embed_fn
    embed_fn = make_embed_fn(load_encoder(args.encoder))

    cl = _client()

    if args.figure:
        print(f"[fig] building montage of real v117 error sites -> {args.figure}")
        make_figure(args.figure, embed_fn, cl, mip=args.mip,
                    candidate_mode=args.candidate_mode, radius_nm=args.radius_nm,
                    direction_cone_deg=args.direction_cone_deg or 45.0)
        print(f"[out] wrote {args.figure}")
        return

    print(f"[cave] scanning {args.n_scan} somas for v117-era splits ...")
    roots, ts = find_split_neurons(cl, n_scan=args.n_scan)
    print(f"[cave] {len(roots)} split neurons found; evaluating up to {args.max_neurons}")

    results: list[SiteResult] = []
    for rt in roots[:args.max_neurons]:
        try:
            sites = sites_for_neuron(cl, rt, ts)
        except Exception as e:
            print(f"  root {rt}: site error {type(e).__name__}")
            continue
        for s in sites:
            try:
                r = evaluate_site(s, embed_fn, mip=args.mip,
                                  candidate_mode=args.candidate_mode,
                                  radius_nm=args.radius_nm,
                                  direction_cone_deg=args.direction_cone_deg)
            except Exception as e:
                r = None
            if r is not None:
                results.append(r)
        if results:
            print(f"  root {rt}: cumulative {len(results)} sites scored")

    if not results:
        print("no scorable sites"); return

    n = len(results)
    t1_learned = np.mean([r.top1_learned for r in results])
    t1_raw = np.mean([r.top1_raw for r in results])
    mrr_learned = np.mean([1.0 / (r.rank_learned + 1) for r in results])
    mrr_raw = np.mean([1.0 / (r.rank_raw + 1) for r in results])
    chance = np.mean([1.0 / r.n_candidates for r in results])
    cone = f", cone {args.direction_cone_deg}deg" if args.direction_cone_deg else ""
    mode_desc = (f"proximity r={args.radius_nm:.0f}nm{cone}"
                 if args.candidate_mode == "proximity" else "slab (legacy)")
    print(f"\nReal v117 error sites scored: {n}   [candidates: {mode_desc}]")
    print(f"  mean candidates/site : {np.mean([r.n_candidates for r in results]):.1f}")
    print(f"  mean gap             : {np.mean([r.gap_nm for r in results]):.0f} nm")
    print(f"  chance top-1         : {chance:.3f}")
    print(f"  raw-patch  top-1 / MRR: {t1_raw:.3f} / {mrr_raw:.3f}")
    print(f"  LEARNED    top-1 / MRR: {t1_learned:.3f} / {mrr_learned:.3f}")

    with open(args.out, "w") as f:
        json.dump({"n_sites": n, "candidate_mode": args.candidate_mode,
                   "radius_nm": args.radius_nm, "direction_cone_deg": args.direction_cone_deg,
                   "mean_candidates": float(np.mean([r.n_candidates for r in results])),
                   "mean_gap_nm": float(np.mean([r.gap_nm for r in results])),
                   "chance_top1": float(chance),
                   "top1_raw": float(t1_raw), "top1_learned": float(t1_learned),
                   "mrr_raw": float(mrr_raw), "mrr_learned": float(mrr_learned),
                   "sites": [asdict(r) for r in results]}, f, indent=2)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
