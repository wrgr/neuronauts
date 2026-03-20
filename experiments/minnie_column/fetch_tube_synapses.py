#!/usr/bin/env python3
"""Fetch synapses inside axis-aligned tube bboxes for each nucleus row in a manifest.

Writes one directory per root under ``--out-dir`` with ``synapses.npz`` + ``meta.json``.
Uses :func:`neuronauts.fetch.fetch_synapses` with a **non-cube** bbox from
:func:`tubes.fetch_bbox_for_tube`.

Example::

    python -m experiments.minnie_column.fetch_tube_synapses \\
        --manifest-tsv run_logs/minnie_column_manifest.tsv \\
        --split train \\
        --radius-xy-um 15 --z-half-extent-um 40 \\
        --version 1718 \\
        --out-dir run_logs/minnie_column_synapses \\
        --max-rows 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from neuronauts.fetch import fetch_synapses

from .tubes import fetch_bbox_for_tube


def _save_synapse_npz(out_dir: Path, syn, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "pre_pt": syn.pre_pt,
        "post_pt": syn.post_pt,
        "pre_root_id": syn.pre_root_id,
        "post_root_id": syn.post_root_id,
        "synapse_id": syn.synapse_id,
    }
    if syn.pre_seg_id is not None:
        arrays["pre_seg_id"] = syn.pre_seg_id
    if syn.post_seg_id is not None:
        arrays["post_seg_id"] = syn.post_seg_id
    np.savez_compressed(out_dir / "synapses.npz", **arrays)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-tsv", type=str, required=True)
    ap.add_argument("--split", type=str, default=None, help="Optional filter: train|test|unassigned")
    ap.add_argument("--radius-xy-um", type=float, default=15.0)
    ap.add_argument("--z-half-extent-um", type=float, default=40.0)
    ap.add_argument("--mip", type=int, default=2)
    ap.add_argument("--version", type=int, default=1718)
    ap.add_argument("--datastack", default="minnie65_public")
    ap.add_argument("--cave-token", default=None)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--max-rows", type=int, default=0, help="Max nuclei to process (0 = all)")
    ap.add_argument("--dedup-index", type=str, default=None, help="Optional path to write synapse dedup keys TSV")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.manifest_tsv, sep="\t")
    if args.split:
        df = df[df["split"].astype(str) == args.split]
    if len(df) == 0:
        raise SystemExit("No rows after split filter.")

    n = len(df) if args.max_rows <= 0 else min(len(df), int(args.max_rows))
    df = df.iloc[:n]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_ids: list[tuple[int, int, int]] = []

    for _, row in df.iterrows():
        rid = int(row["pt_root_id"])
        nid = int(row["id"]) if "id" in row else rid
        cx = int(row["center_x_nm"])
        cy = int(row["center_y_nm"])
        cz = int(row["center_z_nm"])
        center_nm = (cx, cy, cz)

        bbox_nm = fetch_bbox_for_tube(
            center_nm,
            radius_xy_um=float(args.radius_xy_um),
            z_half_extent_um=float(args.z_half_extent_um),
        )

        syn = fetch_synapses(
            bbox_nm,
            mip=int(args.mip),
            version=int(args.version),
            datastack=args.datastack,
            token=args.cave_token,
        )

        if len(syn.pre_pt) > 0:
            for sid in np.asarray(syn.synapse_id).ravel().tolist():
                all_ids.append((int(rid), int(sid), int(nid)))

        sub = out_root / f"root_{rid}_nucleus_{nid}"
        meta = {
            "pt_root_id": rid,
            "nucleus_id": nid,
            "center_nm": list(center_nm),
            "bbox_nm": [list(bbox_nm[0]), list(bbox_nm[1])],
            "radius_xy_um": float(args.radius_xy_um),
            "z_half_extent_um": float(args.z_half_extent_um),
            "version": int(args.version),
            "n_synapses": int(len(syn.pre_pt)),
        }
        _save_synapse_npz(sub, syn, meta)

    if args.dedup_index and all_ids:
        p = Path(args.dedup_index)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write("pt_root_id\tsynapse_id\tnucleus_id\n")
            for a, b, c in all_ids:
                f.write(f"{a}\t{b}\t{c}\n")

    print(f"Wrote {n} tube synapse bundles under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
