#!/usr/bin/env python3
"""Analyze non-zero box counts in Minnie65 at different box sizes.

Counts how many distinct boxes (non-overlapping grid cells) contain at least
one synapse. Smaller boxes → more boxes → more global coverage from training.

Usage
-----
With CAVE (requires network, caveclient)::

    python attic/one_off_analyses/analyze_minnie65_boxes.py --sample 100000

With static synapse CSV (if already downloaded)::

    python attic/one_off_analyses/analyze_minnie65_boxes.py --static-dir data/microns_static --version 1078 --sample 500000

Estimate-only (no data fetch)::

    python attic/one_off_analyses/analyze_minnie65_boxes.py --estimate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Minnie65 bounds from dataset_builder (nm)
MINNIE65_X_NM = (300_000, 3_800_000)
MINNIE65_Y_NM = (300_000, 2_700_000)
MINNIE65_Z_NM = (50_000, 780_000)

# Synapse voxel size for position conversion (nm)
SYNAPSE_VOX_NM = np.array([4.0, 4.0, 40.0], dtype=np.float64)

BOX_SIZES_UM = [4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100, 150, 200]


def load_positions_from_cave(sample_limit: int) -> np.ndarray:
    """Fetch synapse positions from CAVE. Returns [N, 3] in nm. Uses pre+post for 2× coverage."""
    from neuronauts.fetch import CAVE_SERVER, MICRONS_DATASTACK, SYNAPSE_VOXEL_SIZE_NM

    try:
        from caveclient import CAVEclient
    except ImportError:
        raise SystemExit("pip install caveclient for CAVE-based analysis")

    client = CAVEclient(MICRONS_DATASTACK, server_address=CAVE_SERVER)
    syn_vox = np.array(SYNAPSE_VOXEL_SIZE_NM, dtype=np.float64)

    positions_nm = []
    try:
        df = client.materialize.synapse_query(limit=sample_limit)
        if len(df) == 0:
            raise RuntimeError("synapse_query returned empty")
        # Use both pre and post positions for 2× spatial coverage
        for col in ["pre_pt_position", "post_pt_position"]:
            if col in df.columns:
                for pos in df[col].values:
                    arr = np.asarray(pos, dtype=np.float64)
                    nm = (arr * syn_vox).astype(np.float64)
                    positions_nm.append(nm)
        if not positions_nm:
            # Fallback: ctr_pt_position
            if "ctr_pt_position" in df.columns:
                for pos in df["ctr_pt_position"].values:
                    arr = np.asarray(pos, dtype=np.float64)
                    nm = (arr * syn_vox).astype(np.float64)
                    positions_nm.append(nm)
    except Exception:
        tbl = client.materialize.query_table(
            "synapses_pni_2",
            select_columns=["pre_pt_position", "post_pt_position"],
            limit=sample_limit,
        )
        if len(tbl) > 0:
            for col in ["pre_pt_position", "post_pt_position"]:
                if col in tbl.columns:
                    for pos in tbl[col].values:
                        arr = np.asarray(pos, dtype=np.float64)
                        nm = (arr * syn_vox).astype(np.float64)
                        positions_nm.append(nm)

    if not positions_nm:
        raise RuntimeError("No synapse positions returned from CAVE")
    return np.array(positions_nm, dtype=np.float64)


def load_positions_from_static_csv(
    static_dir: str, version: int, sample_limit: int
) -> np.ndarray:
    """Stream synapse positions from static CSV. Returns [N, 3] in nm."""
    import pandas as pd

    base = Path(static_dir) / f"v{version}"
    syn_csv = base / "synapses_pni_2_v1_filtered_view.csv.gz"
    header_csv = base / "synapses_pni_2_v1_filtered_view_header.csv"
    if not syn_csv.exists() or not header_csv.exists():
        raise FileNotFoundError(
            f"Static files not found. Run synapse_root_counts_static.py or ensure "
            f"{base} contains the synapse CSV and header."
        )

    header_df = pd.read_csv(header_csv, header=None, names=["column", "type"])
    col_map = dict(enumerate(header_df["column"].tolist()))
    # Position columns: ctr_pt_position is typically index for center, or pre/post pos
    # The static CSV may use different column names. Check for common variants.
    idx_x = idx_y = idx_z = None
    for i, name in col_map.items():
        if "ctr_pt" in name.lower() and "x" in name.lower():
            idx_x = i
        elif "ctr_pt" in name.lower() and "y" in name.lower():
            idx_y = i
        elif "ctr_pt" in name.lower() and "z" in name.lower():
            idx_z = i
    if idx_x is None:
        # Try pre_pt_position as fallback (one of the two sides)
        for i, name in col_map.items():
            if "pre_pt_position_x" == name or "ctr_pt_position_x" == name:
                idx_x = i
            elif "pre_pt_position_y" == name or "ctr_pt_position_y" == name:
                idx_y = i
            elif "pre_pt_position_z" == name or "ctr_pt_position_z" == name:
                idx_z = i
    if idx_x is None:
        raise KeyError(
            f"Could not find position columns in {header_csv}. "
            f"Columns: {list(col_map.values())[:20]}..."
        )

    # Read in chunks; positions are in voxel units at SYNAPSE_VOX_NM
    usecols = [idx_x, idx_y, idx_z]
    positions_nm_list = []
    n_read = 0
    for chunk in pd.read_csv(
        syn_csv, compression="gzip", header=None, usecols=usecols, chunksize=100_000
    ):
        vox = chunk.values.astype(np.float64)
        nm = vox * SYNAPSE_VOX_NM
        positions_nm_list.append(nm)
        n_read += len(chunk)
        if n_read >= sample_limit:
            break
    if not positions_nm_list:
        raise RuntimeError("No rows in synapse CSV")
    arr = np.vstack(positions_nm_list)[:sample_limit]
    return arr.astype(np.float64)


def count_unique_boxes(positions_nm: np.ndarray, box_side_um: float) -> int:
    """Count unique non-overlapping grid cells that contain at least one synapse."""
    box_nm = box_side_um * 1000.0
    cells = np.floor(positions_nm / box_nm).astype(np.int64)
    unique = np.unique(cells.view(np.dtype((np.void, cells.dtype.itemsize * 3))))
    return len(unique)


def geometric_estimate(extent_um: tuple[float, float, float], box_side_um: float) -> int:
    """Estimate box count for a cuboid (non-overlapping grid)."""
    lx, ly, lz = extent_um
    nx = max(1, int(lx / box_side_um))
    ny = max(1, int(ly / box_side_um))
    nz = max(1, int(lz / box_side_um))
    return nx * ny * nz


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=50_000, help="Max synapse positions to use")
    ap.add_argument("--estimate-only", action="store_true", help="Skip fetch; print geometric estimates only")
    ap.add_argument("--static-dir", default="data/microns_static", help="Static data dir for CSV")
    ap.add_argument("--version", type=int, default=1078, help="Static data version")
    ap.add_argument("--source", choices=["cave", "static"], default="cave", help="Data source")
    args = ap.parse_args()

    print("Minnie65 non-zero box count analysis")
    print("=" * 60)

    # Extents in µm
    extent_full_nm = (
        MINNIE65_X_NM[1] - MINNIE65_X_NM[0],
        MINNIE65_Y_NM[1] - MINNIE65_Y_NM[0],
        MINNIE65_Z_NM[1] - MINNIE65_Z_NM[0],
    )
    extent_full_um = tuple(n / 1000 for n in extent_full_nm)
    # Synapse-covered neuropil core ~1 mm³ (from MICrONS docs)
    extent_core_um = (1000.0, 1000.0, 1000.0)

    if args.estimate_only:
        print("\nGeometric estimates (non-overlapping grid)")
        print("-" * 50)
        print(f"{'Box (µm)':<10} {'Full extent':>14} {'Core ~1mm³':>14}")
        print(f"{'':10} {extent_full_um}  {extent_core_um}")
        print("-" * 50)
        for s in BOX_SIZES_UM:
            n_full = geometric_estimate(extent_full_um, float(s))
            n_core = geometric_estimate(extent_core_um, float(s))
            print(f"{s:<10} {n_full:>14,} {n_core:>14,}")
        print("\nNote: 'Full extent' uses dataset_builder bounds (3.5×2.4×0.73 mm).")
        print("'Core ~1mm³' assumes proofread neuropil. Actual non-zero boxes")
        print("require synapse positions; run without --estimate-only.")
        return 0

    # Load positions
    print(f"\nLoading up to {args.sample:,} synapse positions from {args.source}...")
    try:
        if args.source == "cave":
            positions = load_positions_from_cave(args.sample)
        else:
            positions = load_positions_from_static_csv(
                args.static_dir, args.version, args.sample
            )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    n_actual = len(positions)
    print(f"Loaded {n_actual:,} positions")

    # Spatial extent of sample
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    span_nm = hi - lo
    span_um = span_nm / 1000
    print(f"Sample span: {span_um[0]:.0f} × {span_um[1]:.0f} × {span_um[2]:.0f} µm (nm units)")

    # Count unique boxes per size
    print(f"\nUnique non-zero boxes (from sample of {n_actual:,} synapses)")
    print("-" * 50)
    print(f"{'Box (µm)':<10} {'Unique boxes':>14} {'Boxes/synapse':>14}")
    print("-" * 50)

    for s in BOX_SIZES_UM:
        n_boxes = count_unique_boxes(positions, float(s))
        ratio = n_boxes / n_actual if n_actual > 0 else 0
        print(f"{s:<10} {n_boxes:>14,} {ratio:>14.4f}")

    # Geometric max for sample span
    print("\nGeometric max (non-overlapping grid over sample span):")
    for s in [6, 15, 30]:
        n_geom = geometric_estimate(tuple(span_um), float(s))
        print(f"  {s} µm: {n_geom:,}")

    print("\n--- Implications ---")
    print("Smaller boxes → more distinct training examples from the same volume.")
    print("6 µm vs 30 µm: ~127× more boxes (core); see docs/minnie65_box_analysis.md")
    print("Trade-off: smaller boxes have fewer synapses per box; min_positive_pairs")
    print("may need adjustment (e.g. 2 for 6µm, 5 for 30µm).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
