#!/usr/bin/env python3
"""Skeleton-topology global grammar for false merges (the real shape substrate).

The synapse-cloud proof (global_shape_merge.py) hit 41% precision flagging merges from
coarse cloud shape, and argued the residual (intertwined merges) needs the cable tree.
This uses v1718 skeletons -- the actual connected morphology -- and the key construction:

  * clean cell   = one v1718 skeleton (a real neuron)            -> label 0
  * REAL merge   = union of the v1718 cells a v117 split-root combined, BRIDGED at the
                   nearest vertex pair into ONE connected object  -> label 1
  * synth merge  = union of two spatially-near clean cells, likewise bridged -> label 1 (train)

Bridging is essential: an unbridged union has n_components=2, a trivial giveaway. Bridged,
the classifier must use GLOBAL topology -- two somas, two trunks, bimodal cable mass -- the
signal local methods and synapse clouds can't see for intertwined cells.

Topology features (global only): cable length, branch/tip counts, #soma (radius peaks),
radius stats, 2-means cable-mass bimodality, extent, branch density. Grouped-by-cell CV,
size ablation, do-nothing-relevant precision@k. Skeletons cached under --skel-cache.

Usage:
    python -m experiments.pcfg_synapse_partitions.skeleton_topology_merge \
        --sidetable data/sidetable_7box.npz --token $CAVE_TOKEN \
        --skel-cache data/skel_v1718 --sample-clean 250 --max-fm 150
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg_synapse_partitions.synapse_correction import SideTable  # noqa: E402


import threading
_TL = threading.local()


def _client(version, token):
    c = getattr(_TL, "client", None)
    if c is None:
        from caveclient import CAVEclient
        c = CAVEclient("minnie65_public", auth_token=token)
        c.version = int(version)
        _TL.client = c
    return c


def _fetch_one(rid, version, token, cache_dir):
    from neuronauts.fetch import fetch_root_skeleton
    try:
        sk = fetch_root_skeleton(int(rid), version=version, token=token,
                                 cache_dir=cache_dir, client=_client(version, token))
        if len(sk.vertices) < 3:
            return rid, None
        return rid, (sk.vertices.astype(np.float64),
                     sk.edges.astype(np.int64),
                     None if sk.radius is None else sk.radius.astype(np.float64))
    except Exception:
        return rid, None


def fetch_skeletons(ids, version, token, cache_dir, workers=8):
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_one, r, version, token, cache_dir) for r in ids]
        for i, f in enumerate(futs):
            rid, sk = f.result()
            if sk is not None:
                out[rid] = sk
            if (i + 1) % 25 == 0:
                print(f"  fetched {i+1}/{len(ids)} ({len(out)} ok)", flush=True)
    return out


def bridge_union(skels):
    """Concatenate skeletons and add ONE edge between the nearest cross-part vertex pair,
    yielding a single connected merged object (realistic merge, no disconnection giveaway)."""
    Vs, Es, Rs, offs, off = [], [], [], [], 0
    for V, E, R in skels:
        Vs.append(V); Es.append(E + off)
        Rs.append(R if R is not None else np.full(len(V), np.nan))
        offs.append((off, off + len(V))); off += len(V)
    V = np.vstack(Vs); E = np.vstack(Es) if Es else np.zeros((0, 2), int)
    R = np.concatenate(Rs)
    bridges = []
    for k in range(1, len(offs)):
        a0, a1 = offs[0]; b0, b1 = offs[k]
        from scipy.spatial import cKDTree
        t = cKDTree(V[a0:a1]); dd, ii = t.query(V[b0:b1], k=1)
        j = int(np.argmin(dd)); i = int(ii[j])
        bridges.append([a0 + i, b0 + j])
    if bridges:
        E = np.vstack([E, np.array(bridges)])
    return V, E, R


def topo_features(V, E, R):
    """Global topology/shape descriptors (no local junction features)."""
    n = len(V)
    deg = np.bincount(E.flatten(), minlength=n) if len(E) else np.zeros(n, int)
    seg = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1) if len(E) else np.zeros(0)
    cable = float(seg.sum())
    extent = float(np.linalg.norm(V.max(0) - V.min(0))) if n else 0.0
    n_branch = int((deg >= 3).sum())
    n_tips = int((deg == 1).sum())
    rad = R[~np.isnan(R)] if R is not None else np.zeros(0)
    rmax = float(rad.max()) if len(rad) else 0.0
    rmean = float(rad.mean()) if len(rad) else 0.0
    # soma proxy: connected clusters of large-radius vertices
    n_soma = 0
    if len(rad):
        big = np.where((R > 3000) & ~np.isnan(R))[0]
        if len(big):
            from scipy.spatial import cKDTree
            t = cKDTree(V[big]); pairs = t.query_pairs(8000.0)
            parent = {b: b for b in big.tolist()}

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]; x = parent[x]
                return x
            for a, b in pairs:
                parent[find(big[a])] = find(big[b])
            n_soma = len({find(b) for b in big.tolist()})
    # 2-means cable-mass bimodality
    bimod = center_gap = 0.0
    if n >= 6:
        from sklearn.cluster import KMeans
        km1 = KMeans(1, n_init=1, random_state=0).fit(V)
        km2 = KMeans(2, n_init=2, random_state=0).fit(V)
        bimod = float(1 - km2.inertia_ / km1.inertia_) if km1.inertia_ > 0 else 0.0
        center_gap = float(np.linalg.norm(km2.cluster_centers_[0] - km2.cluster_centers_[1]) / (extent + 1))
    return np.array([
        np.log1p(cable), np.log1p(n), np.log1p(extent),
        np.log1p(n_branch), np.log1p(n_tips),
        n_branch / (cable / 1000 + 1), float(n_soma),
        rmax, rmean, bimod, center_gap,
    ])


SIZE_COLS = [0, 1, 2]                         # cable, n, extent
SHAPE_COLS = [3, 4, 5, 6, 7, 8, 9, 10]        # topology/shape, no raw size
FULL = list(range(11))


def _grouped(X, y, groups, cols, folds, n_perm, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xc = X[:, cols]
    gkf = GroupKFold(n_splits=max(2, min(folds, len(np.unique(groups)))))
    res = {}
    for nm, mk in {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "rf": lambda: RandomForestClassifier(300, min_samples_leaf=2, class_weight="balanced", random_state=seed, n_jobs=-1),
    }.items():
        oof = np.full(len(y), np.nan)
        for tr, te in gkf.split(Xc, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            m = mk(); m.fit(Xc[tr], y[tr]); oof[te] = m.predict_proba(Xc[te])[:, 1]
        ok = ~np.isnan(oof)
        auc = roc_auc_score(y[ok], oof[ok])
        rng = np.random.default_rng(seed)
        null = [roc_auc_score(rng.permutation(y[ok]), oof[ok]) for _ in range(n_perm)]
        thr = np.quantile(oof[ok], 0.90)
        fl = oof[ok] >= thr
        prec = (y[ok][fl] == 1).mean() if fl.sum() else float("nan")
        rec = (fl & (y[ok] == 1)).sum() / max(1, (y[ok] == 1).sum())
        res[nm] = (auc, float(np.mean(null)), prec, rec)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sidetable", required=True)
    ap.add_argument("--token", default=os.environ.get("CAVE_TOKEN"))
    ap.add_argument("--skel-cache", default="data/skel_v1718")
    ap.add_argument("--version", type=int, default=1718)
    ap.add_argument("--sample-clean", type=int, default=250)
    ap.add_argument("--max-fm", type=int, default=150)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    d = np.load(args.sidetable)
    tab = SideTable(d["syn_id"], d["side"], d["pt"], d["root_v117"], d["root_later"])
    valid = tab.root_later > 0
    by_v117_later = defaultdict(set)
    by_later_v117 = defaultdict(set)
    cnt = defaultdict(int)
    for i in np.nonzero(valid)[0]:
        a, b = int(tab.root_v117[i]), int(tab.root_later[i])
        by_v117_later[a].add(b); by_later_v117[b].add(a); cnt[b] += 1

    fm_roots = [(a, sorted(s)) for a, s in by_v117_later.items() if len(s) >= 2]
    rng.shuffle(fm_roots); fm_roots = fm_roots[:args.max_fm]
    clean_later = [b for b, s in by_later_v117.items() if len(s) == 1 and cnt[b] >= 20]
    rng.shuffle(clean_later); clean_later = clean_later[:args.sample_clean]

    need = set(clean_later)
    for _, parts in fm_roots:
        need.update(parts)
    print(f"fetching {len(need)} v{args.version} skeletons (clean={len(clean_later)}, "
          f"fm-objects={len(fm_roots)}) -> {args.skel_cache}", flush=True)
    skels = fetch_skeletons(sorted(need), args.version, args.token, args.skel_cache)
    print(f"have {len(skels)} skeletons", flush=True)

    X, y, groups = [], [], []
    gid = {}

    def gget(k):
        return gid.setdefault(k, len(gid))
    # positives: clean cells
    for b in clean_later:
        if b in skels:
            X.append(topo_features(*skels[b])); y.append(0); groups.append(gget(("c", b)))
    # negatives: real merges (bridged union of constituent cells)
    for a, parts in fm_roots:
        ps = [skels[p] for p in parts if p in skels]
        if len(ps) >= 2:
            X.append(topo_features(*bridge_union(ps))); y.append(1)
            groups.append(gget(("m", tuple(parts))))
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    if len(y) < 20 or len(np.unique(y)) < 2:
        print("not enough objects with skeletons; try larger --sample-clean / --max-fm"); return
    print(f"\nobjects={len(y)}  merges={int(y.sum())}  clean={int((y==0).sum())}  "
          f"base={y.mean():.1%}  cells={len(np.unique(groups))}")
    print(f"\n  {'features':10s}{'model':8s}{'AUC':>7s}{'null':>7s}{'prec@10%':>10s}{'rec@10%':>9s}")
    for cols, nm in ((SIZE_COLS, "size"), (SHAPE_COLS, "topology"), (FULL, "full")):
        for mdl, (auc, null, prec, rec) in _grouped(X, y, groups, cols, args.folds, args.n_perm, args.seed).items():
            print(f"  {nm:10s}{mdl:8s}{auc:>7.3f}{null:>7.2f}{prec:>10.2f}{rec:>9.2f}")
    print("\n  topology >> size would confirm real morphology (2 somas/2 trunks) flags merges,")
    print("  vs the synapse-cloud floor (AUC 0.88, prec@2% 0.41). #soma is the key feature.")


if __name__ == "__main__":
    main()
