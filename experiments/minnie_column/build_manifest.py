#!/usr/bin/env python3
"""Build a nucleus manifest for Minnie Column experiments (bbox query @1718+).

Example::

    python -m experiments.minnie_column.build_manifest \\
        --bbox-json data/minnie_column_bbox_example.json \\
        --version 1718 \\
        --bin-width-um 50 --bin-height-um 100 \\
        --auto-median-test \\
        --out-tsv run_logs/minnie_column_manifest.tsv

Requires network access and ``caveclient``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cave_queries import query_nuclei_in_bbox_nm, query_proofread_for_roots
from .constants import DEFAULT_DATASTACK, DEFAULT_MATERIALIZATION_VERSION
from .paradigm import difficulty_from_proofread_row
from .spatial import assign_bins_xy, bbox_from_json, parse_bbox_nm, train_test_split_by_bin


def build_manifest_df(
    *,
    bbox_nm: tuple[tuple[int, int, int], tuple[int, int, int]],
    datastack: str,
    version: int,
    token: str | None,
    bin_width_um: float,
    bin_height_um: float,
    join_proofread: bool,
    train_bins: set[int] | None,
    test_bins: set[int] | None,
    auto_median_test: bool,
) -> pd.DataFrame:
    df = query_nuclei_in_bbox_nm(bbox_nm, datastack=datastack, version=version, token=token)
    if df is None or len(df) == 0:
        raise SystemExit("No nuclei returned for bbox — check bounds and materialization version.")

    pos_cols = ["pt_position_x", "pt_position_y", "pt_position_z"]
    missing = [c for c in pos_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing position columns {missing}; got {list(df.columns)}")

    df = df.dropna(subset=["pt_root_id", *pos_cols]).copy()
    df["pt_root_id"] = df["pt_root_id"].astype("int64")
    df = df[df["pt_root_id"] != 0]

    x_nm = df["pt_position_x"].to_numpy(dtype=np.float64)
    y_nm = df["pt_position_y"].to_numpy(dtype=np.float64)
    z_nm = df["pt_position_z"].to_numpy(dtype=np.float64)

    bin_id = assign_bins_xy(
        x_nm,
        y_nm,
        bbox_nm,
        bin_width_um=bin_width_um,
        bin_height_um=bin_height_um,
    )
    df["bin_id"] = bin_id

    split = train_test_split_by_bin(
        bin_id,
        train_bins=train_bins,
        test_bins=test_bins,
        auto_median_test=auto_median_test,
    )
    df["split"] = split

    center_nm = np.stack([x_nm, y_nm, z_nm], axis=1).astype(np.int64)
    df["center_x_nm"] = center_nm[:, 0]
    df["center_y_nm"] = center_nm[:, 1]
    df["center_z_nm"] = center_nm[:, 2]

    if join_proofread:
        roots = df["pt_root_id"].drop_duplicates().astype("int64").tolist()
        pr = query_proofread_for_roots(roots, datastack=datastack, version=version, token=token)
        if len(pr) > 0:
            df = df.merge(pr, on="pt_root_id", how="left")
            diffs = []
            for _, row in df.iterrows():
                diffs.append(
                    difficulty_from_proofread_row(
                        status_dendrite=row.get("status_dendrite"),
                        status_axon=row.get("status_axon"),
                        strategy_dendrite=row.get("strategy_dendrite"),
                        strategy_axon=row.get("strategy_axon"),
                    )
                )
            df["difficulty_heuristic"] = diffs
        else:
            df["difficulty_heuristic"] = "unknown"
    else:
        df["difficulty_heuristic"] = "unknown"

    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox-nm", type=str, default=None, help="x0,y0,z0,x1,y1,z1 in nm (overrides --bbox-json)")
    ap.add_argument("--bbox-json", type=str, default=None, help="JSON file with key bbox_nm")
    ap.add_argument("--datastack", default=DEFAULT_DATASTACK)
    ap.add_argument("--version", type=int, default=DEFAULT_MATERIALIZATION_VERSION)
    ap.add_argument("--cave-token", default=None)
    ap.add_argument("--bin-width-um", type=float, default=50.0)
    ap.add_argument("--bin-height-um", type=float, default=100.0)
    ap.add_argument("--train-bins", type=str, default=None, help="Comma-separated bin ids for train")
    ap.add_argument("--test-bins", type=str, default=None, help="Comma-separated bin ids for test")
    ap.add_argument("--auto-median-test", action="store_true", help="Split train/test by median bin_id")
    ap.add_argument("--no-proofread", action="store_true", help="Skip proofreading join")
    ap.add_argument("--out-tsv", type=str, required=True)
    ap.add_argument("--out-meta-json", type=str, default=None, help="Optional sidecar JSON with cli args")
    args = ap.parse_args(argv)

    if args.bbox_nm:
        bbox_nm = parse_bbox_nm(args.bbox_nm)
    elif args.bbox_json:
        bbox_nm = bbox_from_json(args.bbox_json)
    else:
        raise SystemExit("Provide --bbox-nm or --bbox-json")

    train_bins = None
    test_bins = None
    if args.train_bins:
        train_bins = {int(x.strip()) for x in args.train_bins.split(",") if x.strip()}
    if args.test_bins:
        test_bins = {int(x.strip()) for x in args.test_bins.split(",") if x.strip()}

    df = build_manifest_df(
        bbox_nm=bbox_nm,
        datastack=args.datastack,
        version=args.version,
        token=args.cave_token,
        bin_width_um=args.bin_width_um,
        bin_height_um=args.bin_height_um,
        join_proofread=not args.no_proofread,
        train_bins=train_bins,
        test_bins=test_bins,
        auto_median_test=bool(args.auto_median_test),
    )

    out = Path(args.out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {len(df)} nuclei to {out}")

    if args.out_meta_json:
        meta = {
            "bbox_nm": [list(bbox_nm[0]), list(bbox_nm[1])],
            "version": args.version,
            "datastack": args.datastack,
            "bin_width_um": args.bin_width_um,
            "bin_height_um": args.bin_height_um,
        }
        Path(args.out_meta_json).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
