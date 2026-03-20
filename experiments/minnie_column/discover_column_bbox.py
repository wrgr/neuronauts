#!/usr/bin/env python3
"""Discover Minnie Column 3D bbox (nm) from a CAVE **column membership** table.

Uses ``allen_v1_column_types_slanted_ref`` by default (nucleus ``target_id`` list),
loads those rows from ``nucleus_detection_v0``, and writes bbox JSON + optional
manifest-ready bounds with margin.

Requires network + ``caveclient`` + ``pandas``.

Example::

    python -m experiments.minnie_column.discover_column_bbox \\
        --version 1718 \\
        --margin-um 2 \\
        --out-json data/minnie_column_bbox.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cave_queries import (
    get_client,
    query_column_target_ids,
    query_nuclei_by_nucleus_ids,
)
from .constants import DEFAULT_DATASTACK, DEFAULT_MATERIALIZATION_VERSION


def compute_bbox_nm(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    margin_nm: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    m = int(margin_nm)
    return (
        (int(np.floor(x.min())) - m, int(np.floor(y.min())) - m, int(np.floor(z.min())) - m),
        (int(np.ceil(x.max())) + m, int(np.ceil(y.max())) + m, int(np.ceil(z.max())) + m),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datastack", default=DEFAULT_DATASTACK)
    ap.add_argument("--version", type=int, default=DEFAULT_MATERIALIZATION_VERSION)
    ap.add_argument("--cave-token", default=None)
    ap.add_argument(
        "--column-table",
        default="allen_v1_column_types_slanted_ref",
        help="CAVE reference table listing column nuclei via target_id",
    )
    ap.add_argument("--margin-um", type=float, default=2.0, help="Pad bbox on all sides (µm)")
    ap.add_argument("--out-json", required=True, help="Output bbox JSON (bbox_nm key)")
    ap.add_argument("--out-nuclei-tsv", default=None, help="Optional: write nuclei used for bounds")
    args = ap.parse_args(argv)

    margin_nm = int(float(args.margin_um) * 1000.0)

    print(f"[CAVE] column table={args.column_table!r} @ v{args.version} …")
    target_ids = query_column_target_ids(
        table_name=args.column_table,
        datastack=args.datastack,
        version=args.version,
        token=args.cave_token,
    )
    if not target_ids:
        raise SystemExit(
            f"No target_id rows from {args.column_table} at v{args.version}. "
            "Try another --column-table (see docs/minnie_column_downloads.md)."
        )
    print(f"[CAVE] {len(target_ids)} unique nucleus target_ids")

    df = query_nuclei_by_nucleus_ids(
        target_ids,
        datastack=args.datastack,
        version=args.version,
        token=args.cave_token,
    )
    if df is None or len(df) == 0:
        raise SystemExit("nucleus_detection_v0 returned no rows for those ids.")

    pos_cols = ["pt_position_x", "pt_position_y", "pt_position_z"]
    miss = [c for c in pos_cols if c not in df.columns]
    if miss:
        raise SystemExit(f"Missing columns {miss}; got {list(df.columns)}")

    df = df.dropna(subset=pos_cols)
    x = df["pt_position_x"].to_numpy(dtype=np.float64)
    y = df["pt_position_y"].to_numpy(dtype=np.float64)
    z = df["pt_position_z"].to_numpy(dtype=np.float64)

    bbox = compute_bbox_nm(x, y, z, margin_nm=margin_nm)

    payload = {
        "source": "discover_column_bbox.py",
        "column_table": args.column_table,
        "materialization_version": args.version,
        "n_nuclei": int(len(df)),
        "margin_um": float(args.margin_um),
        "bbox_nm": [list(bbox[0]), list(bbox[1])],
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote bbox ({len(df)} nuclei) to {out}")
    print(f"  x: {bbox[0][0]:,} … {bbox[1][0]:,} nm  (~{(bbox[1][0]-bbox[0][0])/1000:.1f} µm)")
    print(f"  y: {bbox[0][1]:,} … {bbox[1][1]:,} nm  (~{(bbox[1][1]-bbox[0][1])/1000:.1f} µm)")
    print(f"  z: {bbox[0][2]:,} … {bbox[1][2]:,} nm  (~{(bbox[1][2]-bbox[0][2])/1000:.1f} µm)")

    if args.out_nuclei_tsv:
        p = Path(args.out_nuclei_tsv)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, sep="\t", index=False)
        print(f"Wrote nuclei subset to {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
