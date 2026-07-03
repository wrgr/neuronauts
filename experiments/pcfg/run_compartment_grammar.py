#!/usr/bin/env python3
"""Compartment-augmented PCFG: pipeline + experiments.

Experiment 0 (``--exp0``) is the SegCLR-only *value probe*: before building the
compartment grammar, measure how well the real Google SegCLR embeddings, **by
themselves**, localize the seam of a false merge (a split error).

Why m343-native (point clouds, no skeletons)?
--------------------------------------------
SegCLR ids are keyed to the ``m343`` segmentation snapshot (circa materialization
343, Nov 2022), which is *not* a queryable public materialization (the public
datastack exposes v1300+).  Mapping m343 -> current roots (to attach current
skeletons + proofreading ground truth) needs the ``seg_m343`` volume or the
chunkedgraph and is the next milestone.  For a quick value signal we stay fully
inside m343: SegCLR ships per-node coordinates + embeddings, which already tile
each neuron, so we build a spatial kNN graph over the embedding points and treat
that as the object graph -- no CAVE, no skeleton service, no cloudvolume.

The value question, made concrete
---------------------------------
Take two real m343 neurons whose arbors pass near each other (a realistic
false-merge candidate).  Union their embedding point clouds and build a spatial
kNN graph.  An edge is a *seam* edge if it joins the two neurons (a cross-cell
contact) and a *within* edge otherwise.  Score every edge purely by SegCLR
discontinuity ``1 - cos(emb_i, emb_j)`` (optionally after local mean-pooling, to
mirror the grammar's windowed pooling).  Then ask:

* AUC of the score separating seam edges from within edges, and
* precision@K / recall@K of the top-K highest-score edges being seam edges,

averaged over many neuron pairs.  Strong numbers => SegCLR is a load-bearing
split-error signal and the full grammar is worth building; weak => it is a minor
term.  We report the number honestly (see CLAUDE.md), no unearned "it works".
"""
from __future__ import annotations

import argparse
import io
import zipfile
from dataclasses import dataclass

import numpy as np
import requests

from neuronauts.segclr import (
    BUCKET_HTTPS,
    VARIANTS,
    SegCLRReader,
    _HttpRangeFile,
    _parse_csv,
)


# --------------------------------------------------------------------------- #
# Discovery: find large (real-neuron) m343 segments via shard directory sizes
# --------------------------------------------------------------------------- #

def discover_large_segments(
    shard: int,
    *,
    variant: str = "nm_coord",
    top: int = 12,
    min_bytes: int = 500_000,
) -> list[tuple[int, int]]:
    """Return ``[(segment_id, uncompressed_bytes), ...]`` for the largest CSVs in
    a shard (a proxy for point count => real neurons, not tiny fragments)."""
    url = f"{BUCKET_HTTPS}/{VARIANTS[variant]}/{shard}.zip"
    f = _HttpRangeFile(url, requests.Session())
    with zipfile.ZipFile(f) as z:
        infos = [i for i in z.infolist() if i.file_size >= min_bytes]
    infos.sort(key=lambda i: i.file_size, reverse=True)
    return [(int(i.filename[:-4]), int(i.file_size)) for i in infos[:top]]


# --------------------------------------------------------------------------- #
# Neuron cloud
# --------------------------------------------------------------------------- #

@dataclass
class Cloud:
    seg_id: int
    points: np.ndarray   # [N,3] nm
    emb: np.ndarray      # [N,64] float32


def load_cloud(reader: SegCLRReader, seg_id: int, *, subsample: int = 40_000,
               rng: np.random.Generator | None = None) -> Cloud | None:
    pts, emb = reader.read_segment(seg_id)
    if len(pts) < 100:
        return None
    if subsample and len(pts) > subsample:
        rng = rng or np.random.default_rng(0)
        sel = rng.choice(len(pts), subsample, replace=False)
        pts, emb = pts[sel], emb[sel]
    return Cloud(int(seg_id), pts, emb.astype(np.float32))


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #

def _local_pool(points: np.ndarray, emb: np.ndarray, radius_nm: float) -> np.ndarray:
    """Mean-pool each point's embedding over spatial neighbors within radius (the
    point-cloud analog of the grammar's geodesic-window pooling)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    nbrs = tree.query_ball_point(points, r=radius_nm)
    out = np.empty_like(emb)
    for i, ns in enumerate(nbrs):
        out[i] = emb[ns].mean(axis=0) if ns else emb[i]
    return out


def _cosine_dissim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return 1.0 - np.einsum("ij,ij->i", an, bn)


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC via rank statistic (labels: 1=positive/seam, 0=within). No sklearn dep."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


@dataclass
class PairResult:
    seg_a: int
    seg_b: int
    n_within: int
    n_seam: int
    n_edges: int
    auc_raw: float
    auc_pooled: float
    best_seam_pct: float   # percentile rank of the best-ranked seam edge (lower=better)
    hit_at: dict           # K -> 1 if a seam edge is in the top-K edges (raw score)
    site_hit_at: dict      # K -> 1 if a seam edge is in the top-K spatial *sites*
    n_sites: int           # number of candidate sites (spatial clusters of top edges)


def evaluate_merge(
    a: Cloud, b: Cloud, *, k: int = 8, pool_radius_nm: float = 3000.0,
    contact_nm: float = 1500.0, hit_ks=(10, 50, 100, 500), site_ks=(1, 3, 5, 10),
    site_link_nm: float = 5000.0, n_top_for_sites: int = 200,
) -> PairResult | None:
    """Synthetic merge: union two neurons' point clouds (labels 0/1) and score."""
    P = np.vstack([a.points, b.points])
    E = np.vstack([a.emb, b.emb]).astype(np.float32)
    lab = np.concatenate([np.zeros(len(a.points), int), np.ones(len(b.points), int)])
    return _seam_metrics(
        P, E, lab, seg_a=a.seg_id, seg_b=b.seg_id, k=k, pool_radius_nm=pool_radius_nm,
        contact_nm=contact_nm, hit_ks=hit_ks, site_ks=site_ks,
        site_link_nm=site_link_nm, n_top_for_sites=n_top_for_sites,
    )


def _seam_metrics(
    P: np.ndarray, E: np.ndarray, lab: np.ndarray, *, seg_a: int, seg_b: int,
    k: int = 8, pool_radius_nm: float = 3000.0, contact_nm: float = 1500.0,
    hit_ks=(10, 50, 100, 500), site_ks=(1, 3, 5, 10),
    site_link_nm: float = 5000.0, n_top_for_sites: int = 200,
) -> PairResult | None:
    """Core scoring: spatial kNN graph over labeled points, score each edge by
    SegCLR discontinuity, and measure how well it localizes the seam.

    ``lab`` is an integer label per point (which physical cell the point belongs
    to).  A *seam* edge joins points with different labels within ``contact_nm``
    -- a real cross-cell contact (the merge boundary).
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(P)
    dist, idx = tree.query(P, k=k + 1)  # includes self
    ei, ej = [], []
    for i in range(len(P)):
        for col in range(1, k + 1):
            j = int(idx[i, col])
            if i < j:
                ei.append(i); ej.append(j)
    ei = np.asarray(ei); ej = np.asarray(ej)
    if len(ei) == 0:
        return None
    edge_len = np.linalg.norm(P[ei] - P[ej], axis=1)
    cross = lab[ei] != lab[ej]
    seam = cross & (edge_len <= contact_nm)
    if seam.sum() == 0:
        return None  # cells never come within contact => not a merge candidate

    Ep = _local_pool(P, E, pool_radius_nm)
    dissim_raw = _cosine_dissim(E[ei], E[ej])
    dissim_pool = _cosine_dissim(Ep[ei], Ep[ej])

    labels = seam.astype(int)
    auc_raw = _roc_auc(dissim_raw, labels)
    auc_pool = _roc_auc(dissim_pool, labels)

    # All ranking uses the raw per-point discontinuity -- the honest "SegCLR
    # alone" signal.  (Euclidean local pooling crosses the seam at a contact and
    # destroys it; the skeleton grammar pools geodesically instead.)
    order = np.argsort(-dissim_raw)
    n_seam = int(seam.sum())
    n_edges = len(order)

    rank = np.empty(n_edges, int)
    rank[order] = np.arange(n_edges)          # 0 = highest score
    seam_ranks = rank[seam]
    best_seam_pct = float(seam_ranks.min()) / n_edges
    hit = {K: int((seam_ranks < K).any()) for K in hit_ks}

    # Candidate SITES: cluster the top edges spatially (by edge midpoint), so a
    # "detection" is a neighbourhood a reviewer would inspect, not a single edge.
    # This matches "top-K split-error candidates" better than single edges.
    from scipy.spatial import cKDTree

    mids = 0.5 * (P[ei] + P[ej])
    top = order[: min(n_top_for_sites, n_edges)]
    tmids = mids[top]
    t_is_seam = seam[top]
    parent = list(range(len(top)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    if len(top) > 1:
        tt = cKDTree(tmids)
        for i2, j2 in tt.query_pairs(site_link_nm):
            parent[find(i2)] = find(j2)
    roots: dict = {}
    for i2 in range(len(top)):
        roots.setdefault(find(i2), []).append(i2)
    sites = sorted(roots.values(), key=lambda idxs: min(idxs))  # by best edge rank
    site_is_seam = [any(t_is_seam[m] for m in idxs) for idxs in sites]
    site_hit = {K: int(any(site_is_seam[:K])) for K in site_ks}

    return PairResult(int(seg_a), int(seg_b), int((~cross).sum()), n_seam, n_edges,
                      auc_raw, auc_pool, best_seam_pct, hit, site_hit, len(sites))


# --------------------------------------------------------------------------- #
# Exp 0 driver
# --------------------------------------------------------------------------- #

def run_exp0(args) -> None:
    rng = np.random.default_rng(args.seed)
    reader = SegCLRReader(variant=args.variant, cache_dir=args.cache_dir)

    print(f"[exp0] discovering large segments in shards {args.shards} ...", flush=True)
    seg_ids: list[int] = []
    for sh in args.shards:
        found = discover_large_segments(sh, variant=args.variant, top=args.n_neurons,
                                        min_bytes=args.min_bytes)
        seg_ids += [s for s, _ in found]
        print(f"  shard {sh}: {len(found)} large segments", flush=True)
    seg_ids = seg_ids[: args.n_neurons]

    print(f"[exp0] loading {len(seg_ids)} neuron clouds (subsample={args.subsample}) ...",
          flush=True)
    clouds: list[Cloud] = []
    for sid in seg_ids:
        c = load_cloud(reader, sid, subsample=args.subsample, rng=rng)
        if c is not None:
            clouds.append(c)
            print(f"  seg {sid}: {len(c.points)} pts, bbox extent nm "
                  f"{np.round(c.points.max(0)-c.points.min(0)).astype(int)}", flush=True)

    # find neuron pairs whose bounding boxes overlap (real spatial adjacency)
    def bbox(c): return (c.points.min(0), c.points.max(0))
    boxes = [bbox(c) for c in clouds]

    def overlap(i, j):
        lo_i, hi_i = boxes[i]; lo_j, hi_j = boxes[j]
        return bool(np.all(hi_i >= lo_j) and np.all(hi_j >= lo_i))

    pairs = [(i, j) for i in range(len(clouds)) for j in range(i + 1, len(clouds))
             if overlap(i, j)]
    print(f"[exp0] {len(pairs)} bbox-overlapping pairs among {len(clouds)} neurons",
          flush=True)

    results: list[PairResult] = []
    for i, j in pairs:
        r = evaluate_merge(clouds[i], clouds[j], k=args.k, pool_radius_nm=args.pool_radius,
                           contact_nm=args.contact_nm)
        if r is not None:
            results.append(r)
            print(f"  merge {r.seg_a} + {r.seg_b}: seam={r.n_seam}/{r.n_edges} edges "
                  f"AUC={r.auc_raw:.3f} best_seam_pct={r.best_seam_pct*100:.2f}% "
                  f"hit@100={r.hit_at[100]} site_hit@3={r.site_hit_at[3]} "
                  f"(of {r.n_sites} sites)", flush=True)

    if not results:
        print("[exp0] no contacting pairs found; increase --n-neurons or --contact-nm.")
        return

    aucs_raw = np.array([r.auc_raw for r in results])
    aucs_pool = np.array([r.auc_pooled for r in results])
    print("\n=== Exp 0 summary: SegCLR-alone seam localization (m343-native) ===")
    print(f"pairs evaluated (real contacts present): {len(results)}")
    print(f"median seam edges / total edges per pair: "
          f"{int(np.median([r.n_seam for r in results]))} / "
          f"{int(np.median([r.n_edges for r in results]))}")
    print(f"AUC (raw per-point)   mean={aucs_raw.mean():.3f}  median={np.median(aucs_raw):.3f}"
          "   <- ranking power of SegCLR discontinuity")
    print(f"AUC (euclid-pooled)   mean={aucs_pool.mean():.3f}  median={np.median(aucs_pool):.3f}"
          "   (euclid pool crosses seam; diagnostic only)")
    bsp = np.array([r.best_seam_pct for r in results]) * 100
    print(f"best-seam-edge percentile  median={np.median(bsp):.2f}%  mean={bsp.mean():.2f}%"
          "   (how deep to find the FIRST true seam edge)")
    print("edge-level hit@K (a seam edge is within top-K of ALL edges):")
    for K in (10, 50, 100, 500):
        print(f"  hit@{K:<4d}={np.mean([r.hit_at[K] for r in results]):.2f}")
    print("SITE-level hit@K (top-K spatial clusters of high-discontinuity edges):")
    for K in (1, 3, 5, 10):
        print(f"  site_hit@{K:<2d}={np.mean([r.site_hit_at[K] for r in results]):.2f}"
              f"   (median #sites/pair={int(np.median([r.n_sites for r in results]))})"
              if K == 1 else
              f"  site_hit@{K:<2d}={np.mean([r.site_hit_at[K] for r in results]):.2f}")


# --------------------------------------------------------------------------- #
# Exp 1: REAL proofread merges (m343 root -> multiple current roots)
# --------------------------------------------------------------------------- #

def label_by_nearest_skeleton(
    points: np.ndarray, skels: list[tuple[int, np.ndarray]], *, cap_nm: float = 2500.0,
) -> np.ndarray:
    """Label each point by the nearest current-descendant skeleton (index into
    ``skels``); points farther than ``cap_nm`` from every skeleton get -1."""
    from scipy.spatial import cKDTree

    if not skels:
        return np.full(len(points), -1, int)
    dists = np.full((len(skels), len(points)), np.inf)
    for si, (_rid, verts) in enumerate(skels):
        if len(verts):
            dists[si] = cKDTree(verts).query(points, k=1)[0]
    lab = np.argmin(dists, axis=0)
    best = dists[lab, np.arange(len(points))]
    lab[best > cap_nm] = -1
    return lab


def evaluate_real_merge(
    reader: SegCLRReader, m343_root: int, current_roots: list[int], *,
    version: int, token: str, skel_cache: str, subsample: int = 40_000,
    min_share: float = 0.15, cap_nm: float = 2500.0, k: int = 8,
    contact_nm: float = 1500.0, min_skel_verts: int = 150,
    rng: np.random.Generator | None = None, client=None,
) -> PairResult | None:
    """Evaluate SegCLR seam localization on a REAL proofread merge: the m343 root
    was later split into ``current_roots``; label its SegCLR points by nearest
    current skeleton (= ground-truth cell), then score the seam."""
    from neuronauts.fetch import fetch_root_skeleton

    # Check descendant skeletons FIRST (cheap, cached) -- most large m343 roots
    # just shed small fragments and have only one substantial descendant.  Only
    # download the (large) SegCLR cloud once we know it's a real >=2-cell merge.
    skels: list[tuple[int, np.ndarray]] = []
    for rid in current_roots:
        try:
            sk = fetch_root_skeleton(int(rid), version=version, token=token,
                                     cache_dir=skel_cache, client=client)
        except Exception:
            continue
        if len(sk.vertices) >= min_skel_verts:   # substantial cell, not a shed fragment
            skels.append((int(rid), sk.vertices.astype(np.float64)))
    if len(skels) < 2:
        return None  # not a merge of >=2 substantial cells

    cloud = load_cloud(reader, m343_root, subsample=subsample, rng=rng)
    if cloud is None:
        return None

    lab = label_by_nearest_skeleton(cloud.points, skels, cap_nm=cap_nm)
    keep = lab >= 0
    if keep.sum() < 100:
        return None
    P, E, lab = cloud.points[keep], cloud.emb[keep], lab[keep]

    # keep only substantial cells (>= min_share of labeled points); need >=2
    uniq, counts = np.unique(lab, return_counts=True)
    frac = counts / counts.sum()
    big = uniq[frac >= min_share]
    if len(big) < 2:
        return None
    m = np.isin(lab, big)
    P, E, lab = P[m], E[m], lab[m]
    # compact labels to 0..m-1
    remap = {v: i for i, v in enumerate(sorted(big.tolist()))}
    lab = np.array([remap[v] for v in lab], int)

    return _seam_metrics(P, E, lab, seg_a=m343_root, seg_b=len(big), k=k,
                         contact_nm=contact_nm)


def run_exp1(args) -> None:
    import os as _os
    from caveclient import CAVEclient

    rng = np.random.default_rng(args.seed)
    reader = SegCLRReader(variant=args.variant, cache_dir=args.cache_dir)
    tok = _os.environ.get("token") or _os.environ.get("CAVE_TOKEN")
    if not tok:
        print("[exp1] no CAVE token in env (token / CAVE_TOKEN); cannot fetch skeletons.")
        return
    client = CAVEclient("minnie65_public", auth_token=tok)
    version = args.version or client.materialize.version
    client.version = int(version)
    cg = client.chunkedgraph
    print(f"[exp1] datastack version {version}", flush=True)

    print(f"[exp1] discovering large m343 roots in shards {args.shards} ...", flush=True)
    ids: list[int] = []
    for sh in args.shards:
        ids += [s for s, _ in discover_large_segments(sh, variant=args.variant,
                                                       top=args.n_neurons, min_bytes=args.min_bytes)]

    print("[exp1] finding real merges (m343 root -> 2..N current roots) ...", flush=True)
    cands: list[tuple[int, list[int]]] = []
    for r in ids:
        try:
            lat = [int(x) for x in np.atleast_1d(cg.get_latest_roots(r))]
        except Exception:
            continue
        if 2 <= len(lat) <= args.max_desc:
            cands.append((r, lat))
    print(f"[exp1] {len(cands)} merge candidates (<= {args.max_desc} descendants)", flush=True)

    results: list[PairResult] = []
    for ci, (r, lat) in enumerate(cands):
        print(f"  [{ci+1}/{len(cands)}] m343 {r} ({len(lat)} descendants) ...", flush=True)
        try:
            res = evaluate_real_merge(
                reader, r, lat, version=version, token=tok, skel_cache=args.skel_cache,
                subsample=args.subsample, min_share=args.min_share, cap_nm=args.cap_nm,
                k=args.k, contact_nm=args.contact_nm, min_skel_verts=args.min_skel_verts,
                rng=rng, client=client)
        except Exception as e:
            print(f"  m343 {r}: ERROR {type(e).__name__} {str(e)[:80]}", flush=True)
            continue
        if res is not None:
            results.append(res)
            print(f"  REAL merge m343 {res.seg_a} ({res.seg_b} cells): "
                  f"seam={res.n_seam}/{res.n_edges} AUC={res.auc_raw:.3f} "
                  f"best_seam_pct={res.best_seam_pct*100:.2f}% hit@100={res.hit_at[100]} "
                  f"site_hit@3={res.site_hit_at[3]} (of {res.n_sites} sites)", flush=True)

    if not results:
        print("[exp1] no evaluable real merges; loosen --min-share / raise --n-neurons.")
        return

    aucs = np.array([r.auc_raw for r in results])
    bsp = np.array([r.best_seam_pct for r in results]) * 100
    print("\n=== Exp 1 summary: SegCLR-alone on REAL proofread merges ===")
    print(f"real merges evaluated: {len(results)}")
    print(f"AUC (raw per-point)   mean={aucs.mean():.3f}  median={np.median(aucs):.3f}")
    print(f"best-seam-edge percentile  median={np.median(bsp):.2f}%  mean={bsp.mean():.2f}%")
    print("edge-level hit@K:")
    for K in (10, 50, 100, 500):
        print(f"  hit@{K:<4d}={np.mean([r.hit_at[K] for r in results]):.2f}")
    print("SITE-level hit@K (top-K high-discontinuity spatial clusters):")
    for K in (1, 3, 5, 10):
        print(f"  site_hit@{K:<2d}={np.mean([r.site_hit_at[K] for r in results]):.2f}")
    print(f"median #sites/merge: {int(np.median([r.n_sites for r in results]))}")


# --------------------------------------------------------------------------- #
# Exp 2: SegCLR top-1 retrieval + local same-vs-different discrimination
# --------------------------------------------------------------------------- #

def _embedding_top1_same_cell(E: np.ndarray, lab: np.ndarray) -> tuple[float, np.ndarray]:
    """Global retrieval: for each node, is its nearest node in EMBEDDING space the
    same cell? (the classic SegCLR 'top-1 same-cell' metric). Returns (acc, correct)."""
    from scipy.spatial import cKDTree

    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    tree = cKDTree(En)
    _, idx = tree.query(En, k=2)          # k=1 is self
    nn = idx[:, 1]
    correct = lab[nn] == lab
    return float(correct.mean()), correct


def _local_top1_same_cell(P, E, lab, *, radius_nm: float) -> dict:
    """Local discrimination: among nodes within ``radius_nm`` of a query node, is the
    top-1 by embedding cosine the same cell?  Restricted to *discriminative* query
    nodes (those that actually have a different-cell node within the radius) — the
    merge-relevant contacts."""
    from scipy.spatial import cKDTree

    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    tree = cKDTree(P)
    neigh = tree.query_ball_point(P, r=radius_nm)
    n_disc = 0
    n_correct = 0
    for i, ns in enumerate(neigh):
        cand = [j for j in ns if j != i]
        if not cand:
            continue
        labs = lab[cand]
        if (labs == lab[i]).all():
            continue  # no different-cell candidate -> not discriminative
        n_disc += 1
        sims = En[cand] @ En[i]
        top = cand[int(np.argmax(sims))]
        if lab[top] == lab[i]:
            n_correct += 1
    acc = n_correct / n_disc if n_disc else float("nan")
    return {"radius_nm": radius_nm, "n_discriminative": n_disc, "top1_same_cell_acc": acc}


def run_exp2(args) -> None:
    import os as _os
    from caveclient import CAVEclient

    reader = SegCLRReader(variant=args.variant, cache_dir=args.cache_dir)
    tok = _os.environ.get("token") or _os.environ.get("CAVE_TOKEN")
    client = CAVEclient("minnie65_public", auth_token=tok) if tok else None
    cg = client.chunkedgraph if client else None

    # --- source: clean cells (unchanged since m343 = stable identity) ------
    print(f"[exp2] discovering clean unchanged m343 cells in shards {args.shards} ...",
          flush=True)
    ids = []
    for sh in args.shards:
        ids += [s for s, _ in discover_large_segments(sh, variant=args.variant,
                                                       top=args.n_neurons, min_bytes=args.min_bytes)]
    clean = []
    for m in ids:
        if cg is None:
            clean.append(m); continue
        try:
            if bool(np.atleast_1d(cg.is_latest_roots([m]))[0]):
                clean.append(m)
        except Exception:
            pass
        if len(clean) >= args.n_neurons:
            break

    clouds = []
    for m in clean:
        pts, emb = reader.read_segment(m)
        if len(pts) >= 500:
            clouds.append((m, pts, emb))
        if len(clouds) >= args.n_neurons:
            break
    if len(clouds) < 2:
        print("[exp2] need >=2 clean cells; loosen filters."); return

    P = np.vstack([p for _, p, _ in clouds])
    E = np.vstack([e for _, _, e in clouds]).astype(np.float32)
    lab = np.concatenate([[i] * len(p) for i, (_, p, _) in enumerate(clouds)])
    from scipy.spatial import cKDTree
    nn_sp = np.median(cKDTree(P).query(P[: min(4000, len(P))], k=2)[0][:, 1])
    print(f"[exp2] pooled {len(clouds)} clean cells, {len(P)} SegCLR nodes; "
          f"median node spacing {nn_sp:.0f} nm", flush=True)

    # --- Metric A: global embedding top-1 same-cell (retrieval) -----------
    accA, correct = _embedding_top1_same_cell(E, lab)
    print("\n=== Exp 2A: SegCLR embedding top-1 retrieval (clean cells) ===")
    print(f"top-1 nearest-embedding node is SAME cell: {accA:.3f}  "
          f"(over {len(P)} nodes, {len(clouds)} cells)")
    print(f"  chance if random among {len(clouds)} cells ≈ "
          f"{max(len(p) for _,p,_ in clouds)/len(P):.3f} (largest-cell share)")

    # --- Metric B: local top-1 same-cell at contacts ----------------------
    print("\n=== Exp 2B: local top-1 same-cell at contacts (does SegCLR keep touching cells apart?) ===")
    print("(node spacing ~1um, so radii below that yield few/no cross-cell candidates)")
    for R in args.radii:
        res = _local_top1_same_cell(P, E, lab, radius_nm=R)
        print(f"  within {int(R):>5d} nm: discriminative_nodes={res['n_discriminative']:>6d}  "
              f"top1_same_cell_acc={res['top1_same_cell_acc']:.3f}")


# --------------------------------------------------------------------------- #
# M1: compartment labeling sanity on real clean neurons
# --------------------------------------------------------------------------- #

def _nucleus_pos_nm_for_root(client, root_id: int) -> np.ndarray | None:
    """Return the nucleus centroid(s) in nm for a root, or None."""
    try:
        df = client.materialize.query_table(
            "nucleus_detection_v0", filter_equal_dict={"pt_root_id": int(root_id)})
    except Exception:
        return None
    if df is None or len(df) == 0 or "pt_position" not in df.columns:
        return None
    vox = np.array([4.0, 4.0, 40.0])  # nucleus table pt_position is (4,4,40) voxels
    pts = np.array([np.asarray(p, float) for p in df["pt_position"].values])
    return pts * vox


def run_m1(args) -> None:
    import os as _os
    from caveclient import CAVEclient
    from neuronauts.fetch import fetch_root_skeleton, fetch_synapses_for_roots
    from experiments.pcfg.compartments import label_compartments, AXON, DEND, SOMA

    tok = _os.environ.get("token") or _os.environ.get("CAVE_TOKEN")
    if not tok:
        print("[m1] no CAVE token in env."); return
    client = CAVEclient("minnie65_public", auth_token=tok)
    version = int(args.version) if args.version else int(client.materialize.version)
    client.version = version
    print(f"[m1] datastack version {version}", flush=True)

    # prefer proofread neurons (clean, complete cells with real skeletons); the
    # nucleus table alone yields many degenerate/non-neuronal 1-vertex roots.
    roots = args.roots
    if not roots:
        try:
            df = client.materialize.query_table("proofreading_status_and_strategy", limit=3000)
            rid = df["pt_root_id"].values
            roots = [int(x) for x in rid if int(x) != 0][: args.n_neurons * 6]
        except Exception:
            df = client.materialize.query_table("nucleus_detection_v0", limit=6000)
            rid = df["pt_root_id"].values
            rid = rid[rid != 0]
            u, c = np.unique(rid, return_counts=True)
            roots = [int(x) for x in u[c == 1][: args.n_neurons * 6]]

    done = 0
    for r in roots:
        if done >= args.n_neurons:
            break
        try:
            sk = fetch_root_skeleton(int(r), version=version, token=tok,
                                     cache_dir=args.skel_cache, client=client)
        except Exception as e:
            continue
        if len(sk.vertices) < 200 or sk.radius is None:
            continue
        syn = fetch_synapses_for_roots([int(r)], version=version, token=tok, mip=2)
        if syn.n_synapses < 20:
            continue
        nuc = _nucleus_pos_nm_for_root(client, int(r))
        labels = label_compartments(sk, syn, root_id=int(r), mip=2, nucleus_pos_nm=nuc)

        # concordance: do PRE synapses land on AXON vertices and POST on DEND?
        from scipy.spatial import cKDTree
        vox = np.array([32.0, 32.0, 40.0])
        tree = cKDTree(labels.vertices_nm)
        pre = np.asarray(syn.pre_pt, float)[np.asarray(syn.pre_root_id) == r] * vox
        post = np.asarray(syn.post_pt, float)[np.asarray(syn.post_root_id) == r] * vox
        def frac_on(pts, comp):
            if len(pts) == 0:
                return float("nan")
            d, i = tree.query(pts, k=1)
            lab = labels.label[i][d <= 1500.0]
            return float(np.mean(lab == comp)) if len(lab) else float("nan")
        pre_axon = frac_on(pre, AXON)
        post_dend = frac_on(post, DEND)

        s = labels.summary()
        # is_tree: connected & acyclic (E == V-1 within each component)
        n_edges = len(labels.edges)
        is_tree = (n_edges == len(labels.vertices_nm) - 1)
        print(f"\nroot {r} (v{version}): {s['n_vertices']} verts, {syn.n_synapses} syn, "
              f"n_soma_clusters={s['n_soma_clusters']}", flush=True)
        print(f"  label verts: soma={s['n_soma_verts']} axon={s['n_axon_verts']} "
              f"dend={s['n_dend_verts']} unknown={s['n_unknown_verts']}")
        print(f"  #pre(axonal)={len(pre)} #post(dendritic)={len(post)}")
        print(f"  PRE→AXON frac={pre_axon:.2f}  POST→DEND frac={post_dend:.2f}  "
              f"(edges==V-1: {is_tree})")
        done += 1
    if done == 0:
        print("[m1] no suitable neurons found; try --roots or raise --n-neurons.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp0", action="store_true",
                    help="SegCLR-only value probe on synthetic merges (m343-native)")
    ap.add_argument("--exp1", action="store_true",
                    help="SegCLR-only probe on REAL proofread merges (m343 -> current split)")
    ap.add_argument("--exp2", action="store_true",
                    help="SegCLR top-1 retrieval + local same-vs-different discrimination")
    ap.add_argument("--radii", type=float, nargs="+", default=[200, 500, 1000, 2000],
                    help="radii (nm) for the exp2 local top-1 test")
    ap.add_argument("--m1", action="store_true",
                    help="compartment-labeling sanity on clean neurons")
    ap.add_argument("--roots", type=int, nargs="*", default=None,
                    help="explicit root ids for --m1")
    ap.add_argument("--variant", default="nm_coord", choices=sorted(VARIANTS))
    ap.add_argument("--cache-dir", default="cache/segclr")
    ap.add_argument("--skel-cache", default="cache/skel_current")
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-neurons", type=int, default=12)
    ap.add_argument("--min-bytes", type=int, default=1_000_000)
    ap.add_argument("--subsample", type=int, default=40_000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pool-radius", type=float, default=3000.0)
    ap.add_argument("--contact-nm", type=float, default=1500.0)
    ap.add_argument("--version", type=int, default=0, help="materialization version (0=latest)")
    ap.add_argument("--max-desc", type=int, default=8, help="max current descendants for a merge")
    ap.add_argument("--min-share", type=float, default=0.15,
                    help="min fraction of labeled points for a cell to count")
    ap.add_argument("--cap-nm", type=float, default=2500.0,
                    help="max point->skeleton distance to accept a label")
    ap.add_argument("--min-skel-verts", type=int, default=150,
                    help="min skeleton vertices for a descendant to count as a real cell")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.m1:
        run_m1(args)
    elif args.exp2:
        run_exp2(args)
    elif args.exp1:
        run_exp1(args)
    elif args.exp0 or True:
        run_exp0(args)


if __name__ == "__main__":
    main()
