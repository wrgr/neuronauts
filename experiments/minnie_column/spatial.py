"""2D bin assignment in the Minnie Column xy plane and train/test splits."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

BBoxNm = tuple[tuple[int, int, int], tuple[int, int, int]]


def parse_bbox_nm(s: str) -> BBoxNm:
    """Parse ``x0,y0,z0,x1,y1,z1`` (nm) into a pair of corners."""
    parts = [int(x.strip()) for x in s.replace(" ", "").split(",")]
    if len(parts) != 6:
        raise ValueError(f"Expected 6 comma-separated integers, got {len(parts)}: {s!r}")
    x0, y0, z0, x1, y1, z1 = parts
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        raise ValueError("bbox must have x1>x0, y1>y0, z1>z0")
    return ((x0, y0, z0), (x1, y1, z1))


def bbox_from_json(path: str | Path) -> BBoxNm:
    """Load bbox from JSON with key ``bbox_nm``: [[x0,y0,z0],[x1,y1,z1]]."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    if "bbox_nm" in raw:
        a, b = raw["bbox_nm"]
        return (tuple(int(x) for x in a), tuple(int(x) for x in b))
    raise KeyError("JSON must contain key 'bbox_nm': [[x0,y0,z0],[x1,y1,z1]]")


def assign_bins_xy(
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    bbox_nm: BBoxNm,
    *,
    bin_width_um: float,
    bin_height_um: float,
    origin_x_um: float | None = None,
    origin_y_um: float | None = None,
) -> np.ndarray:
    """Assign each nucleus to a non-overlapping **integer** bin id in the xy plane.

    Bins tile a rectangle anchored at (origin_x, origin_y) in the **bbox** min corner
    by default, stepping **bin_width_um** × **bin_height_um** in microns.

    Parameters
    ----------
    x_nm, y_nm
        Shape (N,) soma positions in nanometers.
    bbox_nm
        Column (or ROI) bounds; used only to set default origin at ``bbox[0]``.
    bin_width_um, bin_height_um
        Bin size in **microns** (e.g. 50 and 100 for 50×100 µm bins).
    origin_x_um, origin_y_um
        Optional origin for bin indexing in **µm** relative to the same coordinate
        system as nm. If None, use ``bbox_nm[0]`` converted to µm.

    Returns
    -------
    bin_id : int64 array of shape (N,)
    """
    x_nm = np.asarray(x_nm, dtype=np.float64).ravel()
    y_nm = np.asarray(y_nm, dtype=np.float64).ravel()
    if x_nm.shape != y_nm.shape:
        raise ValueError("x_nm and y_nm must have the same shape")

    x0_nm, y0_nm, _ = bbox_nm[0]
    if origin_x_um is None:
        ox_um = x0_nm / 1000.0
    else:
        ox_um = float(origin_x_um)
    if origin_y_um is None:
        oy_um = y0_nm / 1000.0
    else:
        oy_um = float(origin_y_um)

    w_um = float(bin_width_um)
    h_um = float(bin_height_um)
    if w_um <= 0 or h_um <= 0:
        raise ValueError("bin width/height must be positive")

    x_um = x_nm / 1000.0
    y_um = y_nm / 1000.0

    ix = np.floor((x_um - ox_um) / w_um).astype(np.int64)
    iy = np.floor((y_um - oy_um) / h_um).astype(np.int64)
    # Single linear index (row-major: iy varies slow if desired — here pack ix,iy uniquely)
    # Use a 2D grid code that is stable: bin_id = iy * max_ix_range + ix; for simplicity use pairing:
    # bin_id = iy * 10000 + ix (assumes ix,iy small); cleaner: np.ravel_multi_index after shift
    ix_min = int(ix.min()) if len(ix) else 0
    iy_min = int(iy.min()) if len(iy) else 0
    ix_rel = ix - ix_min
    iy_rel = iy - iy_min
    nx = int(ix_rel.max()) + 1 if len(ix_rel) else 1
    bin_id = iy_rel * nx + ix_rel
    return bin_id.astype(np.int64)


def train_test_split_by_bin(
    bin_id: np.ndarray,
    *,
    train_bins: set[int] | None = None,
    test_bins: set[int] | None = None,
    auto_median_test: bool = False,
) -> np.ndarray:
    """Return split label per nucleus: ``train`` | ``test`` | ``unassigned``.

    * If ``train_bins`` / ``test_bins`` are provided, assign those bins (disjoint).
    * If ``auto_median_test`` is True, bins with ``bin_id > median(unique)`` are
      ``test``; the rest ``train`` (smoke tests only).

    Rows in no set remain ``unassigned`` unless ``auto_median_test`` applies.
    """
    n = len(bin_id)
    out = np.array(["unassigned"] * n, dtype=object)
    bid = np.asarray(bin_id, dtype=np.int64)

    if auto_median_test:
        uniq = np.unique(bid)
        if len(uniq) < 2:
            out[:] = "train"
            return out
        mid = float(np.median(uniq))
        out[bid <= mid] = "train"
        out[bid > mid] = "test"
        return out

    if train_bins is not None:
        out[np.isin(bid, list(train_bins))] = "train"
    if test_bins is not None:
        out[np.isin(bid, list(test_bins))] = "test"

    return out
