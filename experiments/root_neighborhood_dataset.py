#!/usr/bin/env python3
"""Build a synapse-only training cache anchored on *proofread* roots.

This is an alternative to random/synapse-seeded boxes:
- pick "trusted" roots using `proofreading_status_and_strategy`
- locate each root's soma via `nucleus_detection_v0`
- fetch synapses in a local neighborhood around the soma at a chosen materialization
- (optionally) keep only synapses where the anchor root is pre/post/both

The output is a standard `BoxCache` directory (npz/json + index.json), so you can
train with the existing pipeline:

    python scripts/train.py train --cache-dir <cache_dir> ...
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from caveclient import CAVEclient

from neuronauts.dataset_builder import BoxCache, count_positive_pairs
from neuronauts.fetch import RealBoxSpec, fetch_synapses


@dataclass(frozen=True)
class AnchorSpec:
    root_id: int
    center_nm: tuple[int, int, int]


def _client(datastack: str, version: int, token: str | None = None) -> CAVEclient:
    client = CAVEclient(datastack, auth_token=token) if token else CAVEclient(datastack)
    client.version = int(version)
    return client


def sample_proofread_roots(
    *,
    datastack: str,
    version: int,
    n_roots: int,
    seed: int,
    token: str | None = None,
    require_dendrite: bool = True,
    require_axon: bool = False,
    min_synapses_total: int = 0,
) -> list[int]:
    """Sample root IDs from proofreading_status_and_strategy at a given version."""
    client = _client(datastack, version, token=token)
    tbl = "proofreading_status_and_strategy"

    print(f"[CAVE] querying {tbl} at v{version} …")
    # Pull only a few columns to keep the payload small.
    cols = [
        "pt_root_id",
        "status_axon",
        "status_dendrite",
        "strategy_axon",
        "strategy_dendrite",
    ]
    df = client.materialize.query_table(tbl, select_columns=cols)
    if df is None or len(df) == 0:
        raise SystemExit(f"No rows returned from {tbl} at v{version}.")

    # Normalize types and filter.
    df = df.dropna(subset=["pt_root_id"]).copy()
    df["pt_root_id"] = df["pt_root_id"].astype("int64")

    if require_dendrite:
        if "status_dendrite" in df.columns:
            df = df[df["status_dendrite"].astype(bool)]
        if "strategy_dendrite" in df.columns:
            df = df[df["strategy_dendrite"].fillna("none") != "none"]

    if require_axon:
        if "status_axon" in df.columns:
            df = df[df["status_axon"].astype(bool)]
        if "strategy_axon" in df.columns:
            df = df[df["strategy_axon"].fillna("none") != "none"]

    # Optional: filter for roots that have at least some synapses in the latest synapse table.
    # We do this cheaply by just requiring a minimum later during cache build; this param is
    # kept for future expansion.
    if min_synapses_total < 0:
        raise ValueError("--min-synapses-total must be >= 0")

    roots = df["pt_root_id"].drop_duplicates().to_numpy(dtype=np.int64)
    if len(roots) == 0:
        raise SystemExit("No proofread roots matched the requested filters.")

    rng = np.random.default_rng(seed)
    if n_roots >= len(roots):
        sample = roots
    else:
        sample = rng.choice(roots, size=int(n_roots), replace=False)
    out = [int(r) for r in sample.tolist()]
    print(f"[CAVE] sampled {len(out)} roots (from {len(roots)} candidates)")
    return out


def fetch_soma_centers_nm(
    *,
    datastack: str,
    version: int,
    root_ids: list[int],
    token: str | None = None,
) -> dict[int, tuple[int, int, int]]:
    """Fetch soma (nucleus) centers for roots using nucleus_detection_v0.

    Uses pt_position converted to nanometers (desired_resolution=[1,1,1]) and split_positions.
    """
    if not root_ids:
        return {}

    client = _client(datastack, version, token=token)
    tbl = "nucleus_detection_v0"

    print(f"[CAVE] querying {tbl} (soma centers) for {len(root_ids)} roots at v{version} …")
    df = client.materialize.query_table(
        tbl,
        filter_in_dict={"pt_root_id": root_ids},
        split_positions=True,
        desired_resolution=[1, 1, 1],
        select_columns=["pt_root_id", "pt_position"],
    )
    if df is None or len(df) == 0:
        raise SystemExit(f"No nucleus rows returned for requested roots at v{version}.")

    # Expect columns: pt_root_id and pt_position_x/y/z (because split_positions=True).
    root_col = "pt_root_id"
    pos_cols = ["pt_position_x", "pt_position_y", "pt_position_z"]
    missing = [c for c in [root_col, *pos_cols] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns from nucleus table query: {missing}. Got: {df.columns}")

    df = df.dropna(subset=[root_col, *pos_cols]).copy()
    df[root_col] = df[root_col].astype("int64")

    # Some roots can have multiple nucleus detections; use the first row per root.
    centers: dict[int, tuple[int, int, int]] = {}
    for _, row in df.iterrows():
        rid = int(row[root_col])
        if rid == 0 or rid in centers:
            continue
        x = int(float(row[pos_cols[0]]))
        y = int(float(row[pos_cols[1]]))
        z = int(float(row[pos_cols[2]]))
        centers[rid] = (x, y, z)

    print(f"[CAVE] got soma centers for {len(centers)} / {len(root_ids)} roots")
    return centers


def build_root_neighborhood_cache(
    *,
    cache_dir: str,
    datastack: str,
    version: int,
    root_ids: list[int],
    radius_um: float,
    mip: int,
    token: str | None = None,
    anchor_side: str = "both",  # both|pre|post
    min_anchor_synapses: int = 50,
    max_synapses: int = 200_000,
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """Build a synapse-only BoxCache for root neighborhoods."""
    cache = BoxCache(cache_dir)
    centers = fetch_soma_centers_nm(
        datastack=datastack,
        version=version,
        root_ids=root_ids,
        token=token,
    )

    rng = np.random.default_rng(seed)
    side_um = float(radius_um) * 2.0

    wrote = 0
    skipped_missing_soma = 0
    skipped_too_few = 0

    for root_id in root_ids:
        center = centers.get(int(root_id))
        if center is None:
            skipped_missing_soma += 1
            if verbose:
                print(f"[W] root {root_id} has no soma center; skipping")
            continue

        spec = RealBoxSpec(center_nm=center, side_um=side_um, mip=int(mip))
        if cache.contains(spec):
            if verbose:
                print(f"[I] already cached neighborhood for soma at {center}; skipping")
            continue

        if verbose:
            print(f"[CAVE] fetching synapses: root={root_id} center_nm={center} side_um={side_um:g} v{version}")
        syn = fetch_synapses(spec.bbox_nm, mip=spec.mip, version=version)
        if syn is None or len(syn.pre_pt) == 0:
            skipped_too_few += 1
            if verbose:
                print(f"[I] root {root_id} box has no synapses; skipping")
            continue

        pre_roots = np.asarray(syn.pre_root_id, dtype=np.int64)
        post_roots = np.asarray(syn.post_root_id, dtype=np.int64)
        rid = int(root_id)

        if anchor_side == "pre":
            keep = pre_roots == rid
        elif anchor_side == "post":
            keep = post_roots == rid
        else:
            keep = (pre_roots == rid) | (post_roots == rid)

        if int(np.sum(keep)) < int(min_anchor_synapses):
            skipped_too_few += 1
            if verbose:
                print(
                    f"[I] root {root_id} has only {int(np.sum(keep))} anchor synapses "
                    f"(min {min_anchor_synapses}); skipping"
                )
            continue

        # Apply mask.
        def _mask_or_none(arr):
            if arr is None:
                return None
            return np.asarray(arr)[keep]

        syn_filtered = syn.__class__(
            pre_pt=np.asarray(syn.pre_pt)[keep],
            post_pt=np.asarray(syn.post_pt)[keep],
            pre_root_id=pre_roots[keep],
            post_root_id=post_roots[keep],
            synapse_id=np.asarray(syn.synapse_id)[keep],
            pre_seg_id=_mask_or_none(syn.pre_seg_id),
            post_seg_id=_mask_or_none(syn.post_seg_id),
        )

        # Optional: cap for safety on huge neighborhoods.
        if len(syn_filtered.pre_pt) > int(max_synapses):
            idx = rng.permutation(len(syn_filtered.pre_pt))[: int(max_synapses)]
            idx = np.sort(idx)
            syn_filtered = syn_filtered.__class__(
                pre_pt=np.asarray(syn_filtered.pre_pt)[idx],
                post_pt=np.asarray(syn_filtered.post_pt)[idx],
                pre_root_id=np.asarray(syn_filtered.pre_root_id)[idx],
                post_root_id=np.asarray(syn_filtered.post_root_id)[idx],
                synapse_id=np.asarray(syn_filtered.synapse_id)[idx],
                pre_seg_id=None if syn_filtered.pre_seg_id is None else np.asarray(syn_filtered.pre_seg_id)[idx],
                post_seg_id=None if syn_filtered.post_seg_id is None else np.asarray(syn_filtered.post_seg_id)[idx],
            )

        n_pos = count_positive_pairs(syn_filtered)
        rec = cache.save_synapse_only(
            spec,
            syn_filtered,
            n_positive_pairs=int(n_pos),
            root_id_version=int(version),
        )

        # Attach anchor-specific metadata (safe: loader ignores unknown keys).
        meta_path = Path(cache_dir) / f"{rec.box_hash}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(
            {
                "anchor_root_id": int(root_id),
                "anchor_side": anchor_side,
                "radius_um": float(radius_um),
                "materialization_version": int(version),
                "datastack": datastack,
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        wrote += 1
        if verbose:
            print(
                f"[OK] root {root_id} -> {rec.box_hash[:8]} "
                f"syn={rec.n_synapses:,} pos_pairs={rec.n_positive_pairs:,}"
            )

    print(
        f"Done. wrote={wrote}, skipped_missing_soma={skipped_missing_soma}, skipped_too_few={skipped_too_few}. "
        f"Cache: {cache_dir}"
    )


def prune_incomplete_cache_entries(cache_dir: str) -> None:
    """Remove index entries whose .npz/.json files are missing.

    This can happen if a run is interrupted mid-write.
    """
    cache_path = Path(cache_dir)
    idx_path = cache_path / "index.json"
    if not idx_path.exists():
        return
    try:
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return

    kept = []
    removed = []
    for entry in idx:
        h = entry.get("box_hash")
        if not h:
            continue
        if (cache_path / f"{h}.npz").exists() and (cache_path / f"{h}.json").exists():
            kept.append(entry)
        else:
            removed.append(h)

    if removed:
        idx_path.write_text(json.dumps(kept, indent=2), encoding="utf-8")
        print(f"[I] pruned {len(removed)} incomplete cache entries from index.json")


def _write_roots_tsv(path: str, root_ids: list[int]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["root_id"])
        for rid in root_ids:
            w.writerow([int(rid)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample-roots", help="Sample proofread roots and write a TSV.")
    p_sample.add_argument("--datastack", default="minnie65_public")
    p_sample.add_argument("--version", type=int, required=True)
    p_sample.add_argument("--n-roots", type=int, default=50)
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--cave-token", default=None)
    p_sample.add_argument("--require-dendrite", action="store_true", default=True)
    p_sample.add_argument("--no-require-dendrite", dest="require_dendrite", action="store_false")
    p_sample.add_argument("--require-axon", action="store_true", default=False)
    p_sample.add_argument("--out-tsv", default="run_logs/proofread_roots.tsv")

    p_build = sub.add_parser("build-cache", help="Build a root-neighborhood synapse-only cache.")
    p_build.add_argument("--cache-dir", required=True)
    p_build.add_argument("--datastack", default="minnie65_public")
    p_build.add_argument("--version", type=int, required=True)
    p_build.add_argument("--roots-tsv", type=str, default=None, help="Optional TSV with column root_id.")
    p_build.add_argument("--n-roots", type=int, default=25, help="If --roots-tsv not given, sample this many.")
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument("--cave-token", default=None)
    p_build.add_argument("--radius-um", type=float, default=40.0)
    p_build.add_argument("--mip", type=int, default=2)
    p_build.add_argument("--anchor-side", choices=["both", "pre", "post"], default="both")
    p_build.add_argument("--min-anchor-synapses", type=int, default=50)
    p_build.add_argument("--max-synapses", type=int, default=200_000)
    p_build.add_argument("--require-dendrite", action="store_true", default=True)
    p_build.add_argument("--no-require-dendrite", dest="require_dendrite", action="store_false")
    p_build.add_argument("--require-axon", action="store_true", default=False)
    p_build.add_argument("--verbose", action="store_true", default=True)
    p_build.add_argument("--quiet", dest="verbose", action="store_false")

    args = ap.parse_args(argv)

    if args.cmd == "sample-roots":
        roots = sample_proofread_roots(
            datastack=args.datastack,
            version=args.version,
            n_roots=args.n_roots,
            seed=args.seed,
            token=args.cave_token,
            require_dendrite=args.require_dendrite,
            require_axon=args.require_axon,
        )
        _write_roots_tsv(args.out_tsv, roots)
        print(f"Wrote {len(roots)} roots to {args.out_tsv}")
        return 0

    if args.cmd == "build-cache":
        if args.roots_tsv:
            df = pd.read_csv(args.roots_tsv, sep="\t")
            if "root_id" not in df.columns:
                raise SystemExit(f"--roots-tsv must include column root_id. Found: {df.columns}")
            root_ids = [int(x) for x in df["root_id"].dropna().astype("int64").tolist()]
        else:
            root_ids = sample_proofread_roots(
                datastack=args.datastack,
                version=args.version,
                n_roots=args.n_roots,
                seed=args.seed,
                token=args.cave_token,
                require_dendrite=args.require_dendrite,
                require_axon=args.require_axon,
            )

        build_root_neighborhood_cache(
            cache_dir=args.cache_dir,
            datastack=args.datastack,
            version=args.version,
            root_ids=root_ids,
            radius_um=args.radius_um,
            mip=args.mip,
            token=args.cave_token,
            anchor_side=args.anchor_side,
            min_anchor_synapses=args.min_anchor_synapses,
            max_synapses=args.max_synapses,
            seed=args.seed,
            verbose=args.verbose,
        )
        prune_incomplete_cache_entries(args.cache_dir)
        return 0

    raise SystemExit(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

