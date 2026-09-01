"""Cache the proofreading-status and nucleus tables, then locate dense regions.

Ground-truth quality is the thing that sank the synthetic experiments. The
``proofreading_status_and_strategy`` table names the cells whose current
segmentation has been human-verified, so "same neuron" labels derived from
those roots are trustworthy rather than just "whatever the segmentation says".

This writes a small local manifest (a few MB) that every later stage can read
with no network, and reports the densest cubes of proofread somas so we can
pick a region on evidence instead of by hand.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data/gt_manifest")
NUCLEUS_CSV = Path("data/microns_static/v1078/nucleus_detection_v0.csv")
# nucleus_detection_v0 / proofreading tables store pt_position in (4,4,40) nm voxels.
PT_VOXEL_NM = np.asarray([4.0, 4.0, 40.0])


def fetch_tables(version: int | None) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    from caveclient import CAVEclient

    client = CAVEclient("minnie65_public")
    if version is None:
        version = int(client.materialize.most_recent_version())
    print(f"materialization version: {version}", flush=True)

    pr_path = OUT_DIR / f"proofreading_status_v{version}.csv.gz"
    nuc_path = OUT_DIR / f"nucleus_detection_v{version}.csv.gz"
    if pr_path.exists() and nuc_path.exists():
        print("  (serving both tables from local cache)", flush=True)
        return pd.read_csv(pr_path), pd.read_csv(nuc_path), version

    print("fetching proofreading_status_and_strategy ...", flush=True)
    pr = client.materialize.query_table("proofreading_status_and_strategy",
                                        materialization_version=version)
    print(f"  {len(pr):,} rows, columns: {list(pr.columns)}", flush=True)

    print("fetching nucleus_detection_v0 ...", flush=True)
    nuc = client.materialize.query_table("nucleus_detection_v0",
                                         materialization_version=version)
    print(f"  {len(nuc):,} rows", flush=True)
    return pr, nuc, version


def _positions_voxel(df: pd.DataFrame) -> np.ndarray | None:
    """Extract pt_position as an [N,3] voxel array.

    CAVE returns pt_position as a list per row, but a CSV round-trip turns it
    into a string like "[296464 111200  16770]", so both forms are handled.
    """
    if {"pt_position_x", "pt_position_y", "pt_position_z"} <= set(df.columns):
        return df[["pt_position_x", "pt_position_y",
                   "pt_position_z"]].to_numpy(dtype=np.float64)
    if "pt_position" not in df.columns:
        return None
    raw = df["pt_position"].to_numpy()
    if len(raw) and isinstance(raw[0], str):
        return np.array([np.fromstring(s.strip("[]"), sep=" ") for s in raw],
                        dtype=np.float64)
    return np.vstack(raw).astype(np.float64)


def _positions_nm(df: pd.DataFrame) -> np.ndarray | None:
    vox = _positions_voxel(df)
    return None if vox is None else vox * PT_VOXEL_NM


def densest_cubes(pts_nm: np.ndarray, side_nm: float, top: int) -> list[dict]:
    """Grid the volume at `side_nm` and report the fullest cells.

    A plain histogram on a fixed grid can split a dense cluster across a face,
    so the grid is evaluated at two offsets (0 and half a cell) per axis and the
    best-scoring placement is reported.
    """
    best: list[dict] = []
    for shift in (0.0, 0.5):
        origin = pts_nm.min(0) - shift * side_nm
        idx = np.floor((pts_nm - origin) / side_nm).astype(np.int64)
        keys, counts = np.unique(idx, axis=0, return_counts=True)
        for k, c in zip(keys, counts):
            lo = origin + k * side_nm
            best.append({"count": int(c),
                         "lower_nm": lo.round(0).tolist(),
                         "upper_nm": (lo + side_nm).round(0).tolist(),
                         "centre_nm": (lo + side_nm / 2).round(0).tolist(),
                         "shift": shift})
    best.sort(key=lambda d: -d["count"])

    # Suppress overlapping duplicates from the two offsets.
    kept: list[dict] = []
    for cand in best:
        c = np.asarray(cand["centre_nm"])
        if all(np.abs(c - np.asarray(k["centre_nm"])).max() >= side_nm * 0.5
               for k in kept):
            kept.append(cand)
        if len(kept) >= top:
            break
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--side-um", type=float, nargs="+", default=[100, 200, 300])
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pr, nuc, version = fetch_tables(args.version)

    # Persist positions as explicit x/y/z voxel columns so downstream readers
    # never have to re-parse a stringified array.
    for df, name in ((pr, f"proofreading_status_v{version}.csv.gz"),
                     (nuc, f"nucleus_detection_v{version}.csv.gz")):
        path = OUT_DIR / name
        if path.exists():
            continue
        out = df.copy()
        vox = _positions_voxel(out)
        if vox is not None:
            out = out.drop(columns=["pt_position"], errors="ignore")
            out["pt_position_x"] = vox[:, 0].astype(np.int64)
            out["pt_position_y"] = vox[:, 1].astype(np.int64)
            out["pt_position_z"] = vox[:, 2].astype(np.int64)
        out.to_csv(path, index=False, compression="gzip")

    # Which cells count as usable ground truth?
    status_cols = [c for c in pr.columns if "status" in c or "strategy" in c]
    print("\nstatus/strategy value counts:")
    for c in status_cols:
        print(f"  {c}: {pr[c].value_counts().to_dict()}")

    pts = _positions_nm(pr)
    if pts is None:
        raise SystemExit(f"no position column in proofreading table: {list(pr.columns)}")

    # Quality tiers. status_* are booleans; the strategy_* columns say how far
    # each compartment was taken. Dendrite-extended cells are the safe arbor
    # ground truth; a fully extended axon additionally makes outgoing
    # connectivity trustworthy.
    status_ok = np.ones(len(pr), dtype=bool)
    for col in ("status_dendrite", "status_axon"):
        if col in pr.columns:
            status_ok &= pr[col].astype(bool).to_numpy()
    dend_ext = (pr.get("strategy_dendrite", "").astype(str)
                == "dendrite_extended").to_numpy()
    axon_full = (pr.get("strategy_axon", "").astype(str)
                 == "axon_fully_extended").to_numpy()
    mask = status_ok & dend_ext
    gold = status_ok & dend_ext & axon_full
    n_both, n_gold = int(mask.sum()), int(gold.sum())
    print(f"\nproofread cells                    : {len(pr):,}")
    print(f"  status ok (dendrite+axon)        : {int(status_ok.sum()):,}")
    print(f"  + dendrite_extended  [silver]    : {n_both:,}")
    print(f"  + axon_fully_extended  [gold]    : {n_gold:,}")

    summary = {"version": int(version), "n_proofread": int(len(pr)),
               "n_silver": n_both, "n_gold": n_gold, "regions": {}}

    for side_um in args.side_um:
        side_nm = side_um * 1000.0
        print(f"\n=== densest {side_um:g} um cubes ===")
        for label, subset in (("all_proofread", pts),
                              ("silver_dendrite_extended", pts[mask] if n_both else None),
                              ("gold_axon_fully_extended", pts[gold] if n_gold else None)):
            if subset is None or len(subset) == 0:
                continue
            cubes = densest_cubes(subset, side_nm, args.top)
            print(f"  [{label}]")
            for c in cubes:
                ctr = [f"{v/1000:.0f}" for v in c["centre_nm"]]
                print(f"    n={c['count']:<5} centre_um=({', '.join(ctr)})")
            summary["regions"].setdefault(f"{side_um:g}um", {})[label] = cubes

    (OUT_DIR / "region_candidates.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {OUT_DIR}/region_candidates.json and cached tables")


if __name__ == "__main__":
    main()
