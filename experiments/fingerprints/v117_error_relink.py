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
                     min_frag_l2=4, max_sites=6, main_sample=300, frag_sample=120,
                     lvs=None, hist=None) -> list[ErrorSite]:
    """Locate real false-split interfaces inside one neuron.

    Only looks up rep coordinates for the minor fragments plus a *sample* of the
    main arbor -- not all L2 nodes -- because position lookups are the dominant
    network cost.  Pass precomputed ``lvs``/``hist`` to avoid re-querying.
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
            sites.append(ErrorSite(root=int(root),
                                   pos_main_nm=tuple(main_pos[j].tolist()),
                                   pos_frag_nm=tuple(fp[i].tolist()),
                                   gap_nm=gap, frag_l2=int(counts[list(frags).index(f)])))
    sites.sort(key=lambda s: s.gap_nm)
    return sites[:max_sites]


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


def _fetch_box(pos_a, pos_b, margin_nm, mip):
    pts = np.asarray([pos_a, pos_b], float)
    lo = pts.min(0) - margin_nm
    hi = pts.max(0) + margin_nm
    bbox = (tuple(lo.tolist()), tuple(hi.tolist()))
    em = _fetch_em(bbox, mip=mip)
    seg = _fetch_seg(bbox, mip=mip)
    return Volume(em=em.data.astype(np.uint8), seg=seg.data.astype(np.uint64),
                  resolution_nm=seg.voxel_size_nm, origin_vox=seg.bbox_voxels[0])


def _z_index(vol, z_nm):
    return int(round(z_nm / vol.resolution_nm[2] - vol.origin_vox[2] - 0.5))


def _seg_id_at(vol, pos_nm):
    vox = np.asarray(vol.resolution_nm, float)
    origin = np.asarray(vol.origin_vox, float)
    idx = np.round(np.asarray(pos_nm, float) / vox - origin - 0.5).astype(int)
    idx = np.clip(idx, 0, [s - 1 for s in vol.seg.shape])
    return int(vol.seg[idx[0], idx[1], idx[2]]), idx


def evaluate_site(site: ErrorSite, embed_fn, *, mip=1, slab=3, margin_nm=1200.0,
                  collect=False):
    """Rank the true continuation at a real error site by cut-face hash.

    With ``collect=True`` also returns a dict of patches for visualisation:
    ``{"query", "true", "top_learned", "top_learned_id", "true_id"}``.
    """
    vol = _fetch_box(site.pos_main_nm, site.pos_frag_nm, margin_nm, mip)
    grad = _grad(vol.em)
    dark = float(np.percentile(vol.em[vol.seg > 0], 25)) if (vol.seg > 0).any() else 128.0

    za = _z_index(vol, site.pos_main_nm[2])
    zb = _z_index(vol, site.pos_frag_nm[2])
    nz = vol.em.shape[2]
    a_lo = max(min(za, nz - slab), 0)
    b_lo = max(min(zb, nz - slab), 0)

    true_id, _ = _seg_id_at(vol, site.pos_frag_nm)
    if true_id == 0:
        return None

    # query = main-side face (its own seg id), candidates = all faces in B-slab
    q_faces = face_hash(vol.em, vol.seg, grad, a_lo, a_lo + slab, dark_thresh=dark)
    qa_id, _ = _seg_id_at(vol, site.pos_main_nm)
    if qa_id not in q_faces:
        return None
    cand = face_hash(vol.em, vol.seg, grad, b_lo, b_lo + slab, dark_thresh=dark)
    if true_id not in cand or len(cand) < 3:
        return None

    cand_ids = sorted(cand)
    qp = q_faces[qa_id].patch[None]
    cp = np.stack([cand[i].patch for i in cand_ids])

    # learned
    qe = np.asarray(embed_fn(qp))[0]
    ce = np.asarray(embed_fn(cp))
    qe = qe / (np.linalg.norm(qe) + 1e-9)
    ce = ce / (np.linalg.norm(ce, axis=1, keepdims=True) + 1e-9)
    d_learned = 1.0 - ce @ qe
    # raw patch
    qf = _flatnorm(q_faces[qa_id].patch)
    cf = np.stack([_flatnorm(cand[i].patch) for i in cand_ids])
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
        "query": q_faces[qa_id].patch,
        "true": cand[true_id].patch,
        "top_learned": cand[cand_ids[top_col]].patch,
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


def make_figure(out_png, embed_fn, cl, *, n_examples=6, n_scan=160, mip=1, roots=None):
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
                out = evaluate_site(s, embed_fn, mip=mip, collect=True)
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
    ap.add_argument("--out", default="experiments/fingerprints/v117_relink_metrics.json")
    ap.add_argument("--figure", default=None, help="render a montage PNG and exit")
    args = ap.parse_args()

    from .learned_cutface_encoder import load_encoder, make_embed_fn
    embed_fn = make_embed_fn(load_encoder(args.encoder))

    cl = _client()

    if args.figure:
        print(f"[fig] building montage of real v117 error sites -> {args.figure}")
        make_figure(args.figure, embed_fn, cl, mip=args.mip)
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
                r = evaluate_site(s, embed_fn, mip=args.mip)
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
    print(f"\nReal v117 error sites scored: {n}")
    print(f"  mean candidates/site : {np.mean([r.n_candidates for r in results]):.1f}")
    print(f"  mean gap             : {np.mean([r.gap_nm for r in results]):.0f} nm")
    print(f"  chance top-1         : {chance:.3f}")
    print(f"  raw-patch  top-1 / MRR: {t1_raw:.3f} / {mrr_raw:.3f}")
    print(f"  LEARNED    top-1 / MRR: {t1_learned:.3f} / {mrr_learned:.3f}")

    with open(args.out, "w") as f:
        json.dump({"n_sites": n, "chance_top1": float(chance),
                   "top1_raw": float(t1_raw), "top1_learned": float(t1_learned),
                   "mrr_raw": float(mrr_raw), "mrr_learned": float(mrr_learned),
                   "sites": [asdict(r) for r in results]}, f, indent=2)
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
