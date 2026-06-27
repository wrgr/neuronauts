#!/usr/bin/env python3
"""Learn the synapse-level proofreading correction function f(v117) -> proofread partition.

Two data sources:

  * ``--synthetic``  : fabricate a v117/later scenario with known injected false-merges
                       and false-splits (no network, no token).  Used by the smoke test.
  * CAVE (default)   : fetch v117 synapses in a densely-proofread region, map each side's
                       supervoxel to its later-version root via the chunkedgraph, and learn.

Reports grouped-by-cell CV AUC overall and split out by stratum (MERGE = false-split
corrected, SPLIT = false-merge corrected), each against a permutation null.

Examples
--------
    # offline sanity check
    python -m experiments.pcfg_synapse_partitions.run_synapse_correction --synthetic

    # real data (needs a CAVE token)
    python -m experiments.pcfg_synapse_partitions.run_synapse_correction \\
        --token $CAVE_TOKEN --later-version 1718 --n-boxes 6 --side-um 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.pcfg_synapse_partitions.synapse_correction import (  # noqa: E402
    SideTable,
    build_correction_pairs,
    build_side_table,
    summarize_edits,
)

V117_DATE = "2021-06-11"  # v117 materialization date (matches edit-mining docs)
DEFAULT_CENTER_NM = (733_592, 513_592, 595_640)  # validated densely-proofread region


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _oof_predictions(X, y, groups, n_splits, seed):
    """Out-of-fold probabilities from grouped CV for LogReg and RandomForest."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_groups = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    models = {
        "logreg": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
    }
    oof = {name: np.full(len(y), np.nan) for name in models}
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        for name, make in models.items():
            m = make()
            m.fit(X[tr], y[tr])
            oof[name][te] = m.predict_proba(X[te])[:, 1]
    return oof


def evaluate(X, y, groups, strata, *, n_splits=5, n_perm=50, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    oof = _oof_predictions(X, y, groups, n_splits, seed)
    results: dict[str, dict] = {}

    strata_masks = {
        "overall": np.ones(len(y), bool),
        "merge": strata == 1,
        "split": strata == 0,
    }
    for name, p in oof.items():
        scored = ~np.isnan(p)
        for sname, smask in strata_masks.items():
            m = smask & scored
            if m.sum() < 10 or len(np.unique(y[m])) < 2:
                continue
            auc = _auc(y[m], p[m])
            # permutation null: shuffle labels within groups
            null = []
            for _ in range(n_perm):
                yp = y.copy()
                for g in np.unique(groups[m]):
                    gm = m & (groups == g)
                    idx = np.nonzero(gm)[0]
                    yp[idx] = rng.permutation(yp[idx])
                null.append(_auc(yp[m], p[m]))
            null = np.array([v for v in null if not np.isnan(v)])
            nm, ns = (float(null.mean()), float(null.std())) if len(null) else (np.nan, np.nan)
            pval = float((null >= auc).mean()) if len(null) else np.nan
            results[f"{name}/{sname}"] = {
                "auc": auc, "n": int(m.sum()), "pos": int(y[m].sum()),
                "null_mean": nm, "null_std": ns, "p": pval,
            }

    if verbose:
        print("\nSynapse-level correction f(v117) -> proofread partition")
        print(f"  pairs={len(y)}  positives={int(y.sum())} ({y.mean():.1%})  "
              f"groups(cells)={len(np.unique(groups))}")
        print(f"  strata: merge(cross-root)={int((strata==1).sum())}  "
              f"split(within-root)={int((strata==0).sum())}")
        print(f"  {'model/stratum':28s}{'AUC':>7s}{'null':>14s}{'p':>8s}{'n':>9s}{'pos%':>7s}")
        for k, r in results.items():
            print(f"  {k:28s}{r['auc']:>7.3f}"
                  f"{r['null_mean']:>8.2f}±{r['null_std']:.2f}"
                  f"{r['p']:>8.3f}{r['n']:>9d}{r['pos']/max(1,r['n']):>7.1%}")
    return results


# ---------------------------------------------------------------------------
# Synthetic scenario with known injected merges / splits
# ---------------------------------------------------------------------------

def make_synthetic(
    n_cells: int = 40,
    syn_per_cell: int = 14,
    false_merge_frac: float = 0.25,   # fraction of cells fused with a neighbour in v117
    false_split_frac: float = 0.25,   # fraction of cells cut into two roots in v117
    box_um: float = 40.0,
    seed: int = 0,
) -> SideTable:
    """Build a SideTable where the proofread (later) partition is the true cells and the
    v117 partition contains deliberately injected false merges and false splits.

    True cell = an arbor: synapses strung along a random axis (a line with jitter).
    later_root = true cell id.  v117_root corrupted:
      * false split: cell cut into two collinear roots (-> should MERGE)
      * false merge: two nearby cells share one v117 root, end-to-end with a gap+kink
                     (-> should SPLIT)
    """
    rng = np.random.default_rng(seed)
    span = box_um * 1000.0
    pts: list[np.ndarray] = []
    later: list[int] = []
    v117: list[int] = []
    next_v117 = 1

    # lay out arbors as lines
    for cell in range(1, n_cells + 1):
        start = rng.uniform(0, span, size=3)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        length = rng.uniform(8000, 16000)
        t = np.sort(rng.uniform(0, length, size=syn_per_cell))
        arbor = start + np.outer(t, axis) + rng.normal(0, 300, size=(syn_per_cell, 3))

        r = rng.random()
        if r < false_split_frac:
            # cut collinearly: first half = root A, second half = root B; both later=cell
            mid = syn_per_cell // 2
            ra, rb = next_v117, next_v117 + 1
            next_v117 += 2
            roots = np.where(np.arange(syn_per_cell) < mid, ra, rb)
            for p, rr in zip(arbor, roots):
                pts.append(p); later.append(cell); v117.append(int(rr))
        elif r < false_split_frac + false_merge_frac:
            # fuse with a tacked-on neighbour arbor under ONE v117 root; later roots differ
            rfuse = next_v117; next_v117 += 1
            for p in arbor:
                pts.append(p); later.append(cell); v117.append(rfuse)
            # neighbour arbor: starts past a gap, with a directional kink
            gap = rng.uniform(4000, 7000)
            kink = rng.normal(size=3); kink /= np.linalg.norm(kink)
            start2 = arbor[-1] + axis * gap
            t2 = np.sort(rng.uniform(0, length * 0.7, size=syn_per_cell))
            arbor2 = start2 + np.outer(t2, kink) + rng.normal(0, 300, size=(syn_per_cell, 3))
            for p in arbor2:
                pts.append(p); later.append(n_cells + cell); v117.append(rfuse)
        else:
            rclean = next_v117; next_v117 += 1
            for p in arbor:
                pts.append(p); later.append(cell); v117.append(rclean)

    pts = np.array(pts)
    later = np.array(later, np.int64)
    v117 = np.array(v117, np.int64)
    n = len(pts)
    # pre-side stream only (single root space) is enough for the synthetic test
    return build_side_table(
        pre_pt=pts, post_pt=pts,
        pre_root_v117=v117, post_root_v117=np.zeros(n, np.int64),
        pre_root_later=later, post_root_later=np.zeros(n, np.int64),
        syn_id=np.arange(n, dtype=np.int64),
        sides="pre",
    )


# ---------------------------------------------------------------------------
# CAVE fetch -> per-synapse cross-version SideTable
# ---------------------------------------------------------------------------

def fetch_side_table(
    token: str,
    *,
    later_version: int,
    n_boxes: int,
    side_um: float,
    center_nm=DEFAULT_CENTER_NM,
    sides: str = "both",
    seed: int = 0,
) -> SideTable:
    import datetime as dt

    from caveclient import CAVEclient

    rng = np.random.default_rng(seed)
    client = CAVEclient("minnie65_public", auth_token=token)
    client.version = 117
    later_ts = client.materialize.get_timestamp(later_version)
    if later_ts.tzinfo is None:
        later_ts = later_ts.replace(tzinfo=dt.timezone.utc)
    syn_vox = np.array([4.0, 4.0, 40.0])  # synapse table voxel size (nm)
    half = side_um * 1000.0 / 2.0

    pre_pt, post_pt = [], []
    pre_sv, post_sv = [], []
    pre_rv, post_rv = [], []
    syn_ids = []
    # Tile boxes on a compact 3-D grid around the proofread-column center so every box
    # stays inside proofread tissue (random jitter risks wandering into volume with no
    # v117->later divergence, which dilutes the edit signal).
    step = side_um * 1000.0
    offsets = sorted(
        ((dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)),
        key=lambda o: abs(o[0]) + abs(o[1]) + abs(o[2]),
    )
    centers = [np.array(center_nm, float) + np.array(o, float) * step
               for o in offsets[:n_boxes]]

    for ci, c in enumerate(centers):
        lo = ((c - half) / syn_vox).astype(np.int64)
        hi = ((c + half) / syn_vox).astype(np.int64)
        print(f"[box {ci+1}/{len(centers)}] querying v117 synapses around {c.astype(int).tolist()} nm ...")
        df = client.materialize.query_table(
            "synapses_pni_2",
            filter_spatial_dict={"ctr_pt_position": [lo.tolist(), hi.tolist()]},
            split_positions=False,
        )
        if len(df) == 0:
            continue
        syn_ids.append(df.index.values.astype(np.int64))
        pre_pt.append(np.stack(df["pre_pt_position"].values) * syn_vox)
        post_pt.append(np.stack(df["post_pt_position"].values) * syn_vox)
        pre_sv.append(df["pre_pt_supervoxel_id"].values.astype(np.int64))
        post_sv.append(df["post_pt_supervoxel_id"].values.astype(np.int64))
        pre_rv.append(df["pre_pt_root_id"].values.astype(np.int64))
        post_rv.append(df["post_pt_root_id"].values.astype(np.int64))

    if not syn_ids:
        raise RuntimeError("no synapses fetched in any box")
    syn_ids = np.concatenate(syn_ids)
    pre_pt = np.concatenate(pre_pt); post_pt = np.concatenate(post_pt)
    pre_sv = np.concatenate(pre_sv); post_sv = np.concatenate(post_sv)
    pre_rv = np.concatenate(pre_rv); post_rv = np.concatenate(post_rv)

    # map every supervoxel to its later-version root (single-valued, exact)
    all_sv = np.unique(np.concatenate([pre_sv, post_sv]))
    all_sv = all_sv[all_sv > 0]
    print(f"mapping {len(all_sv):,} supervoxels -> v{later_version} roots (ts={later_ts:%Y-%m-%d}) ...")
    sv_to_later: dict[int, int] = {0: 0}
    chunk = 100_000
    for s in range(0, len(all_sv), chunk):
        batch = all_sv[s:s + chunk]
        roots = client.chunkedgraph.get_roots(batch.tolist(), timestamp=later_ts)
        sv_to_later.update({int(k): int(v) for k, v in zip(batch.tolist(), roots)})

    def lat(sv):
        return np.array([sv_to_later.get(int(x), 0) for x in sv], np.int64)

    tab = build_side_table(
        pre_pt=pre_pt, post_pt=post_pt,
        pre_root_v117=pre_rv, post_root_v117=post_rv,
        pre_root_later=lat(pre_sv), post_root_later=lat(post_sv),
        syn_id=syn_ids, sides=sides,
    )
    return tab


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="offline synthetic scenario")
    ap.add_argument("--token", default=None, help="CAVE token (or set CAVE_TOKEN)")
    ap.add_argument("--later-version", type=int, default=1718)
    ap.add_argument("--n-boxes", type=int, default=6)
    ap.add_argument("--side-um", type=float, default=30.0)
    ap.add_argument("--sides", choices=["pre", "post", "both"], default="both")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.synthetic:
        tab = make_synthetic(seed=args.seed)
    else:
        import os
        token = args.token or os.environ.get("CAVE_TOKEN")
        if not token:
            ap.error("a CAVE token is required for real data; pass --token or set CAVE_TOKEN "
                     "(or use --synthetic)")
        tab = fetch_side_table(
            token, later_version=args.later_version, n_boxes=args.n_boxes,
            side_um=args.side_um, sides=args.sides, seed=args.seed,
        )

    print("\nedit summary:", summarize_edits(tab))
    X, y, groups, strata = build_correction_pairs(tab, rng=np.random.default_rng(args.seed))
    if len(y) == 0:
        print("no pairs built (need proofread divergence in the region).")
        return
    evaluate(X, y, groups, strata, n_splits=args.cv_folds, n_perm=args.n_perm, seed=args.seed)


if __name__ == "__main__":
    main()
