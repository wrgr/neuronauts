"""Geometric train/eval split over the proofread MICrONS minnie65 column.

The column (``allen_v1_column_types_slanted_ref``) is a densely proofread ROI:
~1355 neurons, essentially all proofread in the latest CAVE.  We split them into
**non-overlapping** train / eval regions by a *tangential* axis (z), so both
regions span the full cortical depth (y) — no layer bias — with a gap band
between them so no neuron's soma sits on the boundary.  Assigning each neuron to a
region by its soma guarantees no cell is in both sets (cell-identity leakage-safe,
in the spirit of the pcfg ``cell_components`` grouping).

Metrics are computed only on proofread roots (the validated ground truth); the
detector is still *run* densely on every root in the region.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

NUC_VOXEL_NM = np.array([4.0, 4.0, 40.0])  # nucleus_detection_v0 pt_position units
COLUMN_TABLE = "allen_v1_column_types_slanted_ref"
PROOFREAD_TABLE = "proofreading_status_and_strategy"


@dataclass
class ColumnNeuron:
    root_id: int
    soma_nm: np.ndarray   # [3]
    cell_type: str
    proofread: bool


@dataclass
class ColumnSplit:
    train: list[ColumnNeuron]
    eval: list[ColumnNeuron]
    gap: list[ColumnNeuron]
    axis: object          # int (raw axis) or "pca-tangential"
    lo: float
    hi: float
    direction: np.ndarray | None = None   # projection direction used

    def summary(self) -> str:
        def pf(xs):
            return sum(1 for n in xs if n.proofread)
        return (f"split on {self.axis}: train={len(self.train)} (pf {pf(self.train)}) | "
                f"gap={len(self.gap)} | eval={len(self.eval)} (pf {pf(self.eval)}) | "
                f"boundary {self.lo/1000:.0f}–{self.hi/1000:.0f} µm (proj)")


def load_column_neurons(client) -> list[ColumnNeuron]:
    """Load column neurons with soma position (nm), cell type, and proofread flag."""
    col = client.materialize.query_table(COLUMN_TABLE, split_positions=True)
    col = col[col["pt_root_id"] != 0]
    proof = set(
        int(x) for x in
        client.materialize.query_table(PROOFREAD_TABLE, limit=5000)["pt_root_id"].values
        if int(x) != 0
    )
    out = []
    for _, r in col.iterrows():
        soma = np.array([r["pt_position_x"], r["pt_position_y"], r["pt_position_z"]],
                        dtype=np.float64) * NUC_VOXEL_NM
        out.append(ColumnNeuron(
            root_id=int(r["pt_root_id"]), soma_nm=soma,
            cell_type=str(r.get("cell_type", "")),
            proofread=int(r["pt_root_id"]) in proof,
        ))
    return out


def _tangential_direction(soma: np.ndarray) -> np.ndarray:
    """Return the tangential axis (PCA minor component) of the soma cloud.

    PC1 of column somas ≈ the pia→white-matter depth gradient; splitting along it
    would separate cortical layers.  The minor component (PC3) is tangential, so a
    split along it keeps the same layer composition on both sides.
    """
    c = soma - soma.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return vt[-1]  # smallest-variance (most tangential) direction


def make_split(neurons: list[ColumnNeuron], *, axis="pca-tangential",
               train_frac: float = 0.55, gap_um: float = 20.0) -> ColumnSplit:
    """Split neurons into train (low side) / gap / eval (high side) along ``axis``.

    ``axis`` is 0/1/2 for a raw axis, or ``"pca-tangential"`` (default) to project
    onto the tangential PCA direction so layer composition is balanced across
    train/eval.  ``train_frac`` sets the split quantile; ``gap_um`` is the excluded
    band width (µm) so train and eval regions don't touch.
    """
    soma = np.array([n.soma_nm for n in neurons])
    direction = None
    if axis == "pca-tangential":
        direction = _tangential_direction(soma)
        coord = soma @ direction
    else:
        coord = soma[:, int(axis)]
    lo = float(np.quantile(coord, train_frac))
    hi = lo + gap_um * 1000.0
    train = [n for n, c in zip(neurons, coord) if c <= lo]
    gap = [n for n, c in zip(neurons, coord) if lo < c < hi]
    ev = [n for n, c in zip(neurons, coord) if c >= hi]
    return ColumnSplit(train=train, eval=ev, gap=gap, axis=axis, lo=lo, hi=hi,
                       direction=direction)
