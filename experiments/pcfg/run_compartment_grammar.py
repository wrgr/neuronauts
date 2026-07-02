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
    """Build a spatial kNN graph over the union of two neurons' embedding points
    and score every edge by SegCLR discontinuity.  Seam edges = cross-cell kNN
    edges within ``contact_nm`` (realistic false-merge contacts)."""
    from scipy.spatial import cKDTree

    P = np.vstack([a.points, b.points])
    E = np.vstack([a.emb, b.emb]).astype(np.float32)
    lab = np.concatenate([np.zeros(len(a.points), int), np.ones(len(b.points), int)])

    tree = cKDTree(P)
    dist, idx = tree.query(P, k=k + 1)  # includes self
    # collect undirected edges (i<j)
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
    # a seam edge must be a genuine spatial contact between the two cells
    seam = cross & (edge_len <= contact_nm)
    if seam.sum() == 0:
        return None  # these two neurons never come within contact => not a merge candidate

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

    return PairResult(a.seg_id, b.seg_id, int((~cross).sum()), n_seam, n_edges,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    # keep a flat --exp0 flag as the primary entry for now
    ap.add_argument("--exp0", action="store_true", help="run the SegCLR-only value probe")
    ap.add_argument("--variant", default="nm_coord", choices=sorted(VARIANTS))
    ap.add_argument("--cache-dir", default="cache/segclr")
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-neurons", type=int, default=12)
    ap.add_argument("--min-bytes", type=int, default=1_000_000)
    ap.add_argument("--subsample", type=int, default=40_000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--pool-radius", type=float, default=3000.0)
    ap.add_argument("--contact-nm", type=float, default=1500.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.exp0 or args.cmd is None:
        run_exp0(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
