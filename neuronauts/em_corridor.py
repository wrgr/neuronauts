"""EM corridor infrastructure for targeted boundary-edge connectivity resolution.

This module supports the Neuronauts pipeline's post-inference stage where ~5-15
"boundary edges" per box sit in an ambiguous similarity band after CellGNN
inference.  For each such edge we fetch a thin EM volume between the two synapse
positions and analyse whether a continuous neural process (axon / dendrite)
connects them.

Typical usage
-------------
>>> specs = corridors_from_boundary_edges(syn_positions_nm, boundary_edges)
>>> scores = batch_score_boundary_edges(syn_positions_nm, boundary_edges)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

# Re-export constants and VolumeChunk from fetch so callers only need one import.
from .fetch import MICRONS_EM_PATH, MICRONS_SEG_PATH, MIP_VOXEL_SIZES, VolumeChunk

__all__ = [
    "CorridorSpec",
    "fetch_corridor",
    "fetch_corridor_seg",
    "corridor_mask",
    "corridor_intensity_stats",
    "corridor_seg_connectivity_score",
    "corridors_from_boundary_edges",
    "score_corridor_connectivity",
    "batch_score_boundary_edges",
    "batch_score_seg_connectivity",
    "fetch_seg_ids_at_points",
    "batch_score_seg_connectivity_fast",
    "cross_section_patch",
    "cutface_similarity",
    "batch_cutface_similarity",
]

# Default cross-section patch geometry -- must match the trained cut-face encoder
# (see experiments/fingerprints/learned_cutface_encoder.py, PATCH = 48).
CUTFACE_PATCH = 48
CUTFACE_SLAB = 3


# ---------------------------------------------------------------------------
# CorridorSpec
# ---------------------------------------------------------------------------

@dataclass
class CorridorSpec:
    """Specification of a cylindrical EM corridor between two synapse positions.

    Parameters
    ----------
    pos_a_nm:
        First synapse position in nanometres, shape ``(3,)``.
    pos_b_nm:
        Second synapse position in nanometres, shape ``(3,)``.
    radius_nm:
        Cylinder radius around the connecting line segment (nm).  Default 1500 nm
        (1.5 µm) captures the likely process plus some context.
    mip:
        CloudVolume MIP level.  MIP 2 gives 32 × 32 × 40 nm/voxel, which is
        sufficient for identifying axon / dendrite cross-sections.
    edge_key:
        ``(syn_i, syn_j)`` index pair used for bookkeeping in the pipeline.
    """

    pos_a_nm: np.ndarray  # shape (3,) — first synapse position in nm
    pos_b_nm: np.ndarray  # shape (3,) — second synapse position in nm
    radius_nm: float = 1500.0  # cylinder radius around the connecting line
    mip: int = 2              # resolution level (MIP 2 = 32×32×40 nm/vox)
    edge_key: tuple = field(default_factory=tuple)  # (syn_i, syn_j) — bookkeeping

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def length_nm(self) -> float:
        """Euclidean distance between pos_a_nm and pos_b_nm in nanometres."""
        return float(np.linalg.norm(np.asarray(self.pos_b_nm) - np.asarray(self.pos_a_nm)))

    @property
    def bbox_nm(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Axis-aligned bounding box of the corridor cylinder, expanded by ``radius_nm``.

        Returns
        -------
        ``((x0, y0, z0), (x1, y1, z1))`` in nanometres.
        """
        a = np.asarray(self.pos_a_nm, dtype=np.float64)
        b = np.asarray(self.pos_b_nm, dtype=np.float64)
        lo = np.minimum(a, b) - self.radius_nm
        hi = np.maximum(a, b) + self.radius_nm
        return (
            (float(lo[0]), float(lo[1]), float(lo[2])),
            (float(hi[0]), float(hi[1]), float(hi[2])),
        )


# ---------------------------------------------------------------------------
# fetch_corridor
# ---------------------------------------------------------------------------

def fetch_corridor(
    spec: CorridorSpec,
    em_path: str = MICRONS_EM_PATH,
) -> VolumeChunk:
    """Fetch the EM volume that encompasses the corridor defined by *spec*.

    The fetched region is the axis-aligned bounding box of the cylinder (i.e.
    the corridor expanded by ``spec.radius_nm`` on all sides).

    Parameters
    ----------
    spec:
        Corridor specification.
    em_path:
        CloudVolume-style path to the EM dataset.

    Returns
    -------
    VolumeChunk
        Raw uint8 EM data with voxel size and bounding-box metadata.

    Raises
    ------
    ValueError
        If ``spec.length_nm > 20_000`` to prevent accidentally fetching huge
        volumes.
    """
    if spec.length_nm > 20_000.0:
        raise ValueError(
            f"Corridor length {spec.length_nm:.1f} nm exceeds the 20 000 nm safety "
            "limit.  Use a shorter edge or increase the limit explicitly."
        )

    # Lazy import — cloudvolume is a heavy dependency; keep it out of module-level
    # imports so the rest of the module can be used without it installed.
    from .fetch import fetch_volume  # noqa: PLC0415

    return fetch_volume(spec.bbox_nm, mip=spec.mip, em_path=em_path)


# ---------------------------------------------------------------------------
# corridor_mask
# ---------------------------------------------------------------------------

def corridor_mask(spec: CorridorSpec, volume: VolumeChunk) -> np.ndarray:
    """Build a boolean mask selecting voxels inside the corridor cylinder.

    The cylinder axis is the line segment from ``spec.pos_a_nm`` to
    ``spec.pos_b_nm``.  A voxel is included when its centre lies within
    ``spec.radius_nm`` of the nearest point on that segment.

    This is a pure-numpy implementation; no network access required.

    Parameters
    ----------
    spec:
        Corridor specification (positions in nm, radius in nm).
    volume:
        ``VolumeChunk`` whose ``data`` shape we want to replicate.  The
        ``bbox_voxels`` and ``voxel_size_nm`` fields are used to convert voxel
        indices to physical nm coordinates.

    Returns
    -------
    np.ndarray
        Boolean array with the same shape as ``volume.data``, True inside the
        cylinder.
    """
    vox_size = np.asarray(volume.voxel_size_nm, dtype=np.float64)  # (3,) — (sx, sy, sz)
    bbox_origin = np.asarray(volume.bbox_voxels[0], dtype=np.float64)  # (x0, y0, z0) voxels

    shape = volume.data.shape  # (X, Y, Z)

    # Build coordinate grids for voxel *centres* in nm.
    # volume.data axes are ordered X, Y, Z matching the voxel index order.
    ix = np.arange(shape[0], dtype=np.float64)
    iy = np.arange(shape[1], dtype=np.float64)
    iz = np.arange(shape[2], dtype=np.float64)

    # Centre of voxel (i,j,k) in nm = (bbox_origin + [i,j,k] + 0.5) * vox_size
    cx = (bbox_origin[0] + ix + 0.5) * vox_size[0]
    cy = (bbox_origin[1] + iy + 0.5) * vox_size[1]
    cz = (bbox_origin[2] + iz + 0.5) * vox_size[2]

    # Broadcast to (X, Y, Z, 3).
    gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
    coords = np.stack([gx, gy, gz], axis=-1)  # (X, Y, Z, 3)

    # Segment endpoints in nm.
    a = np.asarray(spec.pos_a_nm, dtype=np.float64)
    b = np.asarray(spec.pos_b_nm, dtype=np.float64)
    ab = b - a
    ab_sq = float(np.dot(ab, ab))

    if ab_sq < 1e-12:
        # Degenerate: zero-length segment; distance to the single point.
        diff = coords - a  # (X, Y, Z, 3)
        dist_sq = np.sum(diff ** 2, axis=-1)
    else:
        # t = dot(p - a, ab) / |ab|^2, clamped to [0, 1]
        pa = coords - a  # (X, Y, Z, 3)
        t = np.sum(pa * ab, axis=-1) / ab_sq  # (X, Y, Z)
        t = np.clip(t, 0.0, 1.0)
        # Nearest point on segment: a + t * ab
        nearest = a + t[..., np.newaxis] * ab  # (X, Y, Z, 3)
        diff = coords - nearest
        dist_sq = np.sum(diff ** 2, axis=-1)

    return dist_sq <= (spec.radius_nm ** 2)


# ---------------------------------------------------------------------------
# corridor_intensity_stats
# ---------------------------------------------------------------------------

def corridor_intensity_stats(spec: CorridorSpec, volume: VolumeChunk) -> dict:
    """Compute intensity statistics inside the corridor cylinder mask.

    These statistics are the raw features for a future connectivity classifier.

    Parameters
    ----------
    spec:
        Corridor specification.
    volume:
        EM volume (uint8 data).

    Returns
    -------
    dict with keys:
        ``mean``          – mean uint8 intensity inside the mask.
        ``std``           – standard deviation of intensity.
        ``n_voxels``      – number of voxels in the mask.
        ``fraction_bright`` – fraction of masked voxels with intensity > 200.
          Axon interiors tend to be darker; myelin sheaths are bright.
        ``length_nm``     – corridor length in nm (for convenience).
    """
    mask = corridor_mask(spec, volume)
    voxels = volume.data[mask].astype(np.float64)
    n = int(mask.sum())

    if n == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "n_voxels": 0,
            "fraction_bright": 0.0,
            "length_nm": spec.length_nm,
        }

    return {
        "mean": float(voxels.mean()),
        "std": float(voxels.std()),
        "n_voxels": n,
        "fraction_bright": float((voxels > 200).sum() / n),
        "length_nm": spec.length_nm,
    }


# ---------------------------------------------------------------------------
# corridors_from_boundary_edges
# ---------------------------------------------------------------------------

def corridors_from_boundary_edges(
    syn_positions_nm: np.ndarray,           # [N, 3] synapse positions in nm
    boundary_edges: list[tuple[int, int]],  # (i, j) ambiguous edge pairs
    *,
    radius_nm: float = 1500.0,
    mip: int = 2,
    max_length_nm: float = 15_000.0,        # skip edges that are too far apart
) -> list[CorridorSpec]:
    """Create one :class:`CorridorSpec` per boundary edge.

    Edges whose synapse-pair distance exceeds *max_length_nm* are silently
    skipped (the caller should assign those a neutral score of 0.5).

    Parameters
    ----------
    syn_positions_nm:
        Array of shape ``[N, 3]`` giving every synapse's nm position.
    boundary_edges:
        List of ``(i, j)`` index pairs into ``syn_positions_nm``.
    radius_nm:
        Cylinder radius passed to each :class:`CorridorSpec`.
    mip:
        CloudVolume MIP level.
    max_length_nm:
        Maximum allowed corridor length.  Edges longer than this are dropped.

    Returns
    -------
    list[CorridorSpec]
        One spec per kept edge, in the same order as *boundary_edges* (minus
        skipped entries).
    """
    positions = np.asarray(syn_positions_nm, dtype=np.float64)
    specs: list[CorridorSpec] = []
    for i, j in boundary_edges:
        pos_a = positions[i]
        pos_b = positions[j]
        length = float(np.linalg.norm(pos_b - pos_a))
        if length > max_length_nm:
            continue
        specs.append(
            CorridorSpec(
                pos_a_nm=pos_a.copy(),
                pos_b_nm=pos_b.copy(),
                radius_nm=radius_nm,
                mip=mip,
                edge_key=(i, j),
            )
        )
    return specs


# ---------------------------------------------------------------------------
# score_corridor_connectivity  (placeholder for future ML)
# ---------------------------------------------------------------------------

def score_corridor_connectivity(stats: dict) -> float:
    """Return a heuristic connectivity score in ``[0, 1]`` for a corridor.

    .. note::
        **Placeholder** — this function implements a simple first-pass
        heuristic and is intended to be replaced by a trained CNN that operates
        directly on the raw voxel data.  Do not tune the thresholds here for
        production use; train a proper classifier instead.

    Heuristic intuition
    -------------------
    * High ``fraction_bright`` combined with low ``std`` → likely a myelin
      sheath wrapping a continuous axon → high connectivity score.
    * Very low mean intensity (dark interior) → likely cellular cytoplasm;
      ambiguous evidence → score ≈ 0.5.

    Score formula
    -------------
    ``score = min(1.0, fraction_bright * 2.0)``

    This maps ``fraction_bright = 0.5`` → ``score = 1.0``.

    Parameters
    ----------
    stats:
        Dict as returned by :func:`corridor_intensity_stats`.

    Returns
    -------
    float
        Value in ``[0.0, 1.0]``.  Higher means more likely connected.
    """
    fraction_bright = float(stats.get("fraction_bright", 0.0))
    score = min(1.0, fraction_bright * 2.0)
    return float(score)


# ---------------------------------------------------------------------------
# batch_score_boundary_edges
# ---------------------------------------------------------------------------

def batch_score_boundary_edges(
    syn_positions_nm: np.ndarray,
    boundary_edges: list[tuple[int, int]],
    *,
    radius_nm: float = 1500.0,
    mip: int = 2,
    max_length_nm: float = 15_000.0,
    em_path: str = MICRONS_EM_PATH,
    verbose: bool = False,
) -> dict[tuple[int, int], float]:
    """End-to-end scoring of ambiguous boundary edges via EM corridor analysis.

    For each boundary edge the pipeline is:
    ``CorridorSpec`` → ``fetch_corridor`` → ``corridor_intensity_stats`` →
    ``score_corridor_connectivity``.

    Edges that exceed *max_length_nm* receive a neutral score of 0.5 (no
    evidence either way).  Network / fetch errors are caught, a warning is
    emitted, and a score of 0.5 is assigned so the pipeline can continue.

    Parameters
    ----------
    syn_positions_nm:
        Array of shape ``[N, 3]`` giving every synapse's nm position.
    boundary_edges:
        List of ``(i, j)`` index pairs — the ambiguous edges from CellGNN
        post-inference.
    radius_nm:
        Cylinder radius for corridor specs.
    mip:
        CloudVolume MIP level for EM fetches.
    max_length_nm:
        Edges longer than this skip the EM fetch (score 0.5).
    em_path:
        CloudVolume-style path to the EM dataset.
    verbose:
        If True, print progress information to stdout.

    Returns
    -------
    dict[tuple[int, int], float]
        Mapping ``(i, j)`` → connectivity score in ``[0, 1]``.
    """
    positions = np.asarray(syn_positions_nm, dtype=np.float64)
    scores: dict[tuple[int, int], float] = {}

    for i, j in boundary_edges:
        edge_key = (i, j)
        pos_a = positions[i]
        pos_b = positions[j]
        length = float(np.linalg.norm(pos_b - pos_a))

        if length > max_length_nm:
            if verbose:
                print(
                    f"[em_corridor] edge {edge_key}: length {length:.0f} nm > "
                    f"max {max_length_nm:.0f} nm — skipping (score=0.5)"
                )
            scores[edge_key] = 0.5
            continue

        spec = CorridorSpec(
            pos_a_nm=pos_a.copy(),
            pos_b_nm=pos_b.copy(),
            radius_nm=radius_nm,
            mip=mip,
            edge_key=edge_key,
        )

        try:
            volume = fetch_corridor(spec, em_path=em_path)
            stats = corridor_intensity_stats(spec, volume)
            score = score_corridor_connectivity(stats)
            if verbose:
                print(
                    f"[em_corridor] edge {edge_key}: length {length:.0f} nm, "
                    f"mean={stats['mean']:.1f}, fraction_bright={stats['fraction_bright']:.3f}, "
                    f"score={score:.3f}"
                )
            scores[edge_key] = score
        except Exception as exc:  # pragma: no cover — requires live network
            warnings.warn(
                f"[em_corridor] Failed to fetch corridor for edge {edge_key}: {exc!r}. "
                "Assigning neutral score 0.5.",
                stacklevel=2,
            )
            scores[edge_key] = 0.5

    return scores


# ---------------------------------------------------------------------------
# fetch_corridor_seg
# ---------------------------------------------------------------------------


def fetch_corridor_seg(
    spec: CorridorSpec,
    seg_path: str = MICRONS_SEG_PATH,
) -> VolumeChunk:
    """Fetch the segmentation volume (uint64 neurite IDs) for a corridor.

    Uses the same bounding box as :func:`fetch_corridor` but fetches from the
    neurite segmentation volume instead of the raw EM.  The returned
    ``VolumeChunk.data`` array is dtype ``uint64``.

    Parameters
    ----------
    spec:
        Corridor specification (positions in nm, radius in nm, mip level).
    seg_path:
        CloudVolume-style path to the segmentation dataset.

    Raises
    ------
    ValueError
        If ``spec.length_nm > 20_000`` (same guard as :func:`fetch_corridor`).
    """
    if spec.length_nm > 20_000.0:
        raise ValueError(
            f"Corridor length {spec.length_nm:.1f} nm exceeds the 20 000 nm safety "
            "limit.  Use a shorter edge or increase the limit explicitly."
        )

    from .fetch import fetch_seg_volume  # noqa: PLC0415

    return fetch_seg_volume(spec.bbox_nm, mip=spec.mip, seg_path=seg_path)


# ---------------------------------------------------------------------------
# corridor_seg_connectivity_score
# ---------------------------------------------------------------------------

def corridor_seg_connectivity_score(spec: CorridorSpec, seg_volume: VolumeChunk) -> float:
    """Return a connectivity score in [0, 1] based on neurite seg IDs at corridor endpoints.

    The score indicates whether the two synapse positions lie on the same
    neurite in the MICrONS segmentation:

    * **1.0** — both endpoints map to the same non-zero seg ID (same neurite).
    * **0.0** — both endpoints map to different non-zero seg IDs (different neurites).
    * **0.5** — at least one endpoint falls on background (seg ID = 0); no evidence.

    Parameters
    ----------
    spec:
        Corridor specification; ``pos_a_nm`` and ``pos_b_nm`` are the positions
        whose seg IDs we compare.
    seg_volume:
        Segmentation ``VolumeChunk`` (uint64) covering at least the corridor
        bounding box.

    Returns
    -------
    float
        Connectivity score in ``[0.0, 0.5, 1.0]``.
    """
    vox_size = np.asarray(seg_volume.voxel_size_nm, dtype=np.float64)
    bbox_origin = np.asarray(seg_volume.bbox_voxels[0], dtype=np.float64)
    shape = seg_volume.data.shape  # (X, Y, Z)

    def _nm_to_idx(pos_nm: np.ndarray) -> tuple[int, int, int]:
        # Voxel centre i has nm coordinate: (bbox_origin[ax] + i + 0.5) * vox_size[ax]
        # Solving for i: i = pos_nm / vox_size - bbox_origin - 0.5
        idx_f = np.asarray(pos_nm, dtype=np.float64) / vox_size - bbox_origin - 0.5
        idx = np.clip(np.round(idx_f).astype(int), 0,
                      [shape[0] - 1, shape[1] - 1, shape[2] - 1])
        return int(idx[0]), int(idx[1]), int(idx[2])

    ia, ja, ka = _nm_to_idx(spec.pos_a_nm)
    ib, jb, kb = _nm_to_idx(spec.pos_b_nm)

    seg_a = int(seg_volume.data[ia, ja, ka])
    seg_b = int(seg_volume.data[ib, jb, kb])

    if seg_a == 0 or seg_b == 0:
        return 0.5  # background — no evidence either way
    if seg_a == seg_b:
        return 1.0  # same neurite
    return 0.0      # different neurites


# ---------------------------------------------------------------------------
# batch_score_seg_connectivity
# ---------------------------------------------------------------------------

def batch_score_seg_connectivity(
    syn_positions_nm: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    radius_nm: float = 1500.0,
    mip: int = 3,
    max_length_nm: float = 15_000.0,
    seg_path: str = MICRONS_SEG_PATH,
    verbose: bool = False,
) -> dict[tuple[int, int], float]:
    """Score edges by neurite segmentation connectivity along the corridor.

    For each edge the pipeline is:
    ``CorridorSpec`` → ``fetch_corridor_seg`` → ``corridor_seg_connectivity_score``.

    This gives a hard morphological signal: 1.0 = same neurite, 0.0 = different
    neurites, 0.5 = background / unknown.  The result is suitable as an edge
    feature in :func:`~neuronauts.cell_graph.build_synapse_graph` via the
    ``seg_connectivity_scores`` parameter.

    Parameters
    ----------
    syn_positions_nm:
        Array of shape ``[N, 3]`` giving every synapse's nm position.
    edges:
        List of ``(i, j)`` index pairs into ``syn_positions_nm``.
    radius_nm:
        Cylinder radius for corridor specs (nm).
    mip:
        CloudVolume MIP level for seg fetches.  MIP 3 (64×64×40 nm/vox) is
        the default — small volumes, sufficient for endpoint lookups.
    max_length_nm:
        Edges longer than this skip the seg fetch and receive score 0.5.
    seg_path:
        CloudVolume-style path to the neurite segmentation.
    verbose:
        If True, print per-edge progress.

    Returns
    -------
    dict[tuple[int, int], float]
        Mapping ``(i, j)`` → score in ``{0.0, 0.5, 1.0}``.
    """
    positions = np.asarray(syn_positions_nm, dtype=np.float64)
    scores: dict[tuple[int, int], float] = {}

    for i, j in edges:
        edge_key = (i, j)
        pos_a = positions[i]
        pos_b = positions[j]
        length = float(np.linalg.norm(pos_b - pos_a))

        if length > max_length_nm:
            if verbose:
                print(
                    f"[em_corridor.seg] edge {edge_key}: length {length:.0f} nm > "
                    f"max {max_length_nm:.0f} nm — skipping (score=0.5)"
                )
            scores[edge_key] = 0.5
            continue

        spec = CorridorSpec(
            pos_a_nm=pos_a.copy(),
            pos_b_nm=pos_b.copy(),
            radius_nm=radius_nm,
            mip=mip,
            edge_key=edge_key,
        )

        try:
            seg_vol = fetch_corridor_seg(spec, seg_path=seg_path)
            score = corridor_seg_connectivity_score(spec, seg_vol)
            if verbose:
                print(
                    f"[em_corridor.seg] edge {edge_key}: length {length:.0f} nm, "
                    f"score={score:.1f}"
                )
            scores[edge_key] = score
        except Exception as exc:  # pragma: no cover — requires live network
            warnings.warn(
                f"[em_corridor] Failed to fetch seg corridor for edge {edge_key}: {exc!r}. "
                "Assigning neutral score 0.5.",
                stacklevel=2,
            )
            scores[edge_key] = 0.5

    return scores


# ---------------------------------------------------------------------------
# fetch_seg_ids_at_points  (fast point-query approach)
# ---------------------------------------------------------------------------

def fetch_seg_ids_at_points(
    positions_nm: np.ndarray,
    *,
    mip: int = 3,
    seg_path: str = MICRONS_SEG_PATH,
    margin_nm: float = 200.0,
) -> np.ndarray:
    """Fetch proofread seg IDs at each point using a single bounding-box fetch.

    Fetches the smallest bounding volume that covers all ``positions_nm`` (plus
    ``margin_nm`` padding) at the requested MIP level, then queries each point
    from the cached array.  This is orders of magnitude faster than the corridor
    approach because it issues one CloudVolume request per batch instead of one
    per edge.

    Parameters
    ----------
    positions_nm:
        Array of shape ``[N, 3]`` giving absolute nm coordinates.
    mip:
        CloudVolume MIP level.  MIP 3 (~64×64×80 nm/vox) is the default —
        the volume is small and each synapse fits comfortably within one voxel.
    seg_path:
        CloudVolume-style path to the neurite segmentation.
    margin_nm:
        Padding added around the bounding box to avoid boundary clipping.

    Returns
    -------
    np.ndarray of shape ``[N]``, dtype ``uint64``.
        Proofread seg ID at each position.  0 = background (no neurite).
    """
    from .fetch import fetch_seg_volume  # noqa: PLC0415

    positions = np.asarray(positions_nm, dtype=np.float64)
    min_nm = positions.min(axis=0) - margin_nm
    max_nm = positions.max(axis=0) + margin_nm
    bbox_nm = (tuple(min_nm.tolist()), tuple(max_nm.tolist()))

    seg_vol = fetch_seg_volume(bbox_nm, mip=mip, seg_path=seg_path)

    vox_size = np.asarray(seg_vol.voxel_size_nm, dtype=np.float64)
    bbox_origin = np.asarray(seg_vol.bbox_voxels[0], dtype=np.float64)
    shape = seg_vol.data.shape  # (X, Y, Z)
    max_idx = np.array([shape[0] - 1, shape[1] - 1, shape[2] - 1], dtype=int)

    seg_ids = np.zeros(len(positions), dtype=np.uint64)
    for k, pos in enumerate(positions):
        idx_f = pos / vox_size - bbox_origin - 0.5
        idx = np.clip(np.round(idx_f).astype(int), 0, max_idx)
        seg_ids[k] = seg_vol.data[idx[0], idx[1], idx[2]]

    return seg_ids


# ---------------------------------------------------------------------------
# batch_score_seg_connectivity_fast
# ---------------------------------------------------------------------------

def batch_score_seg_connectivity_fast(
    syn_positions_nm: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    mip: int = 3,
    seg_path: str = MICRONS_SEG_PATH,
    margin_nm: float = 200.0,
    verbose: bool = False,
) -> dict[tuple[int, int], float]:
    """Score edges by seg ID comparison using a single bounding-box fetch.

    Unlike :func:`batch_score_seg_connectivity`, this function issues **one**
    CloudVolume request for the entire batch: it fetches the seg volume covering
    all synapse positions, reads seg IDs from the cached array, then scores each
    edge by comparing IDs.

    Scores are identical in semantics to the corridor approach:

    * **1.0** — both endpoints share the same non-zero proofread root ID.
    * **0.0** — endpoints have different non-zero root IDs.
    * **0.5** — at least one endpoint is background (ID = 0).

    Parameters
    ----------
    syn_positions_nm:
        Array of shape ``[N, 3]`` giving every synapse's absolute nm position.
    edges:
        List of ``(i, j)`` index pairs into ``syn_positions_nm``.
    mip:
        CloudVolume MIP level for the seg fetch.
    seg_path:
        CloudVolume-style path to the neurite segmentation.
    margin_nm:
        Padding added around the position bounding box.
    verbose:
        If True, print a progress line.

    Returns
    -------
    dict[tuple[int, int], float]
        Mapping ``(i, j)`` → score in ``{0.0, 0.5, 1.0}``.
    """
    positions = np.asarray(syn_positions_nm, dtype=np.float64)

    if verbose:
        print(
            f"[em_corridor.fast] fetching MIP{mip} seg for {len(positions)} positions … ",
            end="",
            flush=True,
        )

    seg_ids = fetch_seg_ids_at_points(positions, mip=mip, seg_path=seg_path, margin_nm=margin_nm)

    if verbose:
        n_nonzero = int((seg_ids > 0).sum())
        print(f"done  ({n_nonzero}/{len(positions)} non-background)")

    scores: dict[tuple[int, int], float] = {}
    for i, j in edges:
        a, b = int(seg_ids[i]), int(seg_ids[j])
        if a == 0 or b == 0:
            scores[(i, j)] = 0.5
        elif a == b:
            scores[(i, j)] = 1.0
        else:
            scores[(i, j)] = 0.0

    return scores


# ---------------------------------------------------------------------------
# Cut-face fingerprint scoring (learned hash edge feature)
# ---------------------------------------------------------------------------
#
# These helpers turn the contrastive cut-face encoder from
# ``experiments/fingerprints/learned_cutface_encoder.py`` into a boundary-edge
# feature.  The premise (see that experiment): slicing is self-inflicted to make
# imaging possible, so every true split is a cut through one continuous process;
# the local ultrastructure on the two faces was continuous, so a learned hash of
# the cross-section can re-link them.
#
# To keep this module free of a torch dependency, the encoder is injected as a
# generic ``embed_fn``: a callable mapping a float32 array ``[N, P, P]`` of
# cross-section patches to an ``[N, D]`` embedding matrix.  The experiment module
# provides one backed by the trained network.

def cross_section_patch(
    em_vol: VolumeChunk,
    seg_vol: VolumeChunk,
    pos_nm,
    *,
    patch: int = CUTFACE_PATCH,
    slab: int = CUTFACE_SLAB,
) -> np.ndarray:
    """Extract the translation-normalised cross-section patch at ``pos_nm``.

    Finds the neurite seg id at the point, masks it in a thin z-slab, projects
    the mean EM intensity over that slab, and crops a ``patch x patch`` window
    centred on the masked footprint.  Returns zeros if the point is background.

    ``em_vol`` and ``seg_vol`` must share a voxel grid (true for MICrONS MIP 0-2,
    where EM and seg resolutions match).
    """
    vox = np.asarray(seg_vol.voxel_size_nm, dtype=np.float64)
    origin = np.asarray(seg_vol.bbox_voxels[0], dtype=np.float64)
    shape = seg_vol.data.shape
    idx = np.round(np.asarray(pos_nm, dtype=np.float64) / vox - origin - 0.5).astype(int)
    idx = np.clip(idx, 0, [shape[0] - 1, shape[1] - 1, shape[2] - 1])

    sid = int(seg_vol.data[idx[0], idx[1], idx[2]])
    out = np.zeros((patch, patch), dtype=np.float32)
    if sid == 0:
        return out

    z = idx[2]
    z0 = max(z - slab // 2, 0)
    z1 = min(z0 + slab, shape[2])
    mask = seg_vol.data[:, :, z0:z1] == sid
    em_sub = em_vol.data[:, :, z0:z1].astype(np.float32)
    count2d = mask.sum(axis=2)
    if not count2d.any():
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        proj = np.where(count2d > 0, (em_sub * mask).sum(axis=2) / count2d, 0.0)

    xs, ys = np.nonzero(count2d > 0)
    ci, cj = int(round(xs.mean())), int(round(ys.mean()))
    h = patch // 2
    xi0, xi1 = max(ci - h, 0), min(ci + h, proj.shape[0])
    yi0, yi1 = max(cj - h, 0), min(cj + h, proj.shape[1])
    px0, py0 = xi0 - (ci - h), yi0 - (cj - h)
    out[px0:px0 + (xi1 - xi0), py0:py0 + (yi1 - yi0)] = proj[xi0:xi1, yi0:yi1]
    return out


def cutface_similarity(
    pos_a_nm,
    pos_b_nm,
    embed_fn,
    *,
    mip: int = 1,
    patch: int = CUTFACE_PATCH,
    slab: int = CUTFACE_SLAB,
    margin_nm: float = 1000.0,
    em_path: str = MICRONS_EM_PATH,
    seg_path: str = MICRONS_SEG_PATH,
) -> float:
    """Cut-face hash similarity (cosine in embedding space) between two points.

    Fetches one EM + seg volume covering both points, extracts each point's
    cross-section patch, embeds both via ``embed_fn`` and returns their cosine
    similarity in ``[-1, 1]`` -- higher means the two faces look like the same
    continuous process, i.e. the boundary edge is more likely a true merge.
    """
    from .fetch import fetch_volume, fetch_seg_volume  # noqa: PLC0415

    pts = np.asarray([pos_a_nm, pos_b_nm], dtype=np.float64)
    lo = pts.min(axis=0) - margin_nm
    hi = pts.max(axis=0) + margin_nm
    bbox = ((float(lo[0]), float(lo[1]), float(lo[2])),
            (float(hi[0]), float(hi[1]), float(hi[2])))

    em_vol = fetch_volume(bbox, mip=mip, em_path=em_path)
    seg_vol = fetch_seg_volume(bbox, mip=mip, seg_path=seg_path)

    pa = cross_section_patch(em_vol, seg_vol, pos_a_nm, patch=patch, slab=slab)
    pb = cross_section_patch(em_vol, seg_vol, pos_b_nm, patch=patch, slab=slab)
    emb = np.asarray(embed_fn(np.stack([pa, pb]).astype(np.float32)))
    a, b = emb[0], emb[1]
    return float(a @ b / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))


def batch_cutface_similarity(
    syn_positions_nm: np.ndarray,
    edges: list,
    embed_fn,
    *,
    mip: int = 1,
    patch: int = CUTFACE_PATCH,
    slab: int = CUTFACE_SLAB,
    margin_nm: float = 1000.0,
    em_path: str = MICRONS_EM_PATH,
    seg_path: str = MICRONS_SEG_PATH,
    verbose: bool = False,
) -> dict:
    """Score boundary ``edges`` by cut-face hash similarity with one bulk fetch.

    Fetches a single EM + seg volume covering all referenced synapse positions,
    extracts every needed cross-section patch once, embeds them in a single
    ``embed_fn`` call, then returns ``{(i, j): cosine_similarity}``.  Suitable as
    an edge feature in :func:`~neuronauts.cell_graph.build_synapse_graph`.
    """
    from .fetch import fetch_volume, fetch_seg_volume  # noqa: PLC0415

    positions = np.asarray(syn_positions_nm, dtype=np.float64)
    used = sorted({k for e in edges for k in e})
    if not used:
        return {}
    pts = positions[used]
    lo = pts.min(axis=0) - margin_nm
    hi = pts.max(axis=0) + margin_nm
    bbox = ((float(lo[0]), float(lo[1]), float(lo[2])),
            (float(hi[0]), float(hi[1]), float(hi[2])))

    if verbose:
        print(f"[cutface] fetching MIP{mip} EM+seg for {len(used)} points ...")
    em_vol = fetch_volume(bbox, mip=mip, em_path=em_path)
    seg_vol = fetch_seg_volume(bbox, mip=mip, seg_path=seg_path)

    patches = {k: cross_section_patch(em_vol, seg_vol, positions[k], patch=patch, slab=slab)
               for k in used}
    emb = np.asarray(embed_fn(np.stack([patches[k] for k in used]).astype(np.float32)))
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    row = {k: emb[c] for c, k in enumerate(used)}

    scores = {}
    for i, j in edges:
        scores[(i, j)] = float(row[i] @ row[j])
    return scores
