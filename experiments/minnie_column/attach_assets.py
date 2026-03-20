#!/usr/bin/env python3
"""Attach download / API metadata columns to a nucleus manifest TSV.

Does **not** download bytes — it writes paths and URL templates so scripts or
``gsutil``/``curl`` can fetch assets. See ``docs/minnie_column_downloads.md``.

Example::

    python -m experiments.minnie_column.attach_assets \\
        --manifest-tsv run_logs/minnie_column_manifest.tsv \\
        --version 1718 \\
        --out-tsv run_logs/minnie_column_manifest_assets.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .asset_urls import (
    EM_PRECOMPUTED_HTTPS,
    GRAPHENE_MINNIE65_PUBLIC,
    SEG_FLAT_M1300,
    SYNAPSE_CSV_V117_BOSSDB,
    mat_dbs_synapse_gz,
    mat_dbs_synapse_header,
    skeleton_swc_url,
)


def enrich_manifest(df: pd.DataFrame, *, version: int, static_synapse_version: int | None) -> pd.DataFrame:
    out = df.copy()
    syn_ver = int(static_synapse_version) if static_synapse_version is not None else version

    # Same for every row (training code reads once from sidecar JSON if preferred)
    out["asset_em_cloudvolume"] = EM_PRECOMPUTED_HTTPS
    out["asset_seg_graphene"] = GRAPHENE_MINNIE65_PUBLIC
    out["asset_seg_flat_m1300"] = SEG_FLAT_M1300
    out["asset_synapse_static_gz"] = mat_dbs_synapse_gz(syn_ver)
    out["asset_synapse_static_header"] = mat_dbs_synapse_header(syn_ver)
    out["asset_synapse_csv_v117_bossdb"] = SYNAPSE_CSV_V117_BOSSDB

    if "pt_root_id" not in out.columns:
        raise SystemExit("manifest must include column pt_root_id")

    def _swc_proof(r: int) -> str:
        return skeleton_swc_url(int(r), proofread=True)

    def _swc_dend(r: int) -> str:
        return skeleton_swc_url(int(r), proofread=False)

    out["asset_skeleton_swc_proofread_url"] = out["pt_root_id"].map(_swc_proof)
    out["asset_skeleton_swc_dendrite_url"] = out["pt_root_id"].map(_swc_dend)

    out["asset_mesh_note"] = (
        "CAVE: use meshclient on minnie65_public @ materialization "
        f"{version} for pt_root_id (see docs/minnie_column_downloads.md)"
    )
    out["materialization_version"] = int(version)
    out["static_synapse_table_version"] = int(syn_ver)

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-tsv", required=True)
    ap.add_argument("--version", type=int, default=1718, help="CAVE materialization for meshes/labels")
    ap.add_argument(
        "--static-synapse-version",
        type=int,
        default=None,
        help="mat_dbs synapse CSV version (default: same as --version)",
    )
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument(
        "--out-sidecar-json",
        default=None,
        help="Optional JSON with global URLs (one copy for pipelines that prefer a single file)",
    )
    args = ap.parse_args(argv)

    df = pd.read_csv(args.manifest_tsv, sep="\t")
    df2 = enrich_manifest(df, version=args.version, static_synapse_version=args.static_synapse_version)

    p = Path(args.out_tsv)
    p.parent.mkdir(parents=True, exist_ok=True)
    df2.to_csv(p, sep="\t", index=False)
    print(f"Wrote {len(df2)} rows to {p}")

    if args.out_sidecar_json:
        side = {
            "materialization_version": args.version,
            "static_synapse_table_version": args.static_synapse_version or args.version,
            "em_cloudvolume": EM_PRECOMPUTED_HTTPS,
            "seg_graphene": GRAPHENE_MINNIE65_PUBLIC,
            "seg_flat_m1300": SEG_FLAT_M1300,
            "synapse_static_gz": mat_dbs_synapse_gz(args.static_synapse_version or args.version),
            "synapse_static_header": mat_dbs_synapse_header(args.static_synapse_version or args.version),
            "synapse_csv_v117": SYNAPSE_CSV_V117_BOSSDB,
            "skeleton_swc_proofread_prefix": "https://storage.googleapis.com/microns-static-links/skel/swc/proofread/",
        }
        Path(args.out_sidecar_json).write_text(json.dumps(side, indent=2), encoding="utf-8")
        print(f"Wrote sidecar {args.out_sidecar_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
