"""Caliber measured on v117 fragment geometry, not on a proofread skeleton.

Every caliber number this repository has reported so far -- EXP-082's
where-to-edit prior (held-out-by-cell area under the curve 0.779, dominated by
radius; radius alone 0.750) and EXP-084's Murray-law conservation prior (area
under the curve 0.675) -- reads ``radius`` out of a **proofread** skeleton
(``data/external/cell_skeletons/*_skv4.npz``). A grower does not have that. It
has v117 fragments. Both experiments name the same unverified step: recompute
caliber from v117 geometry and check the signal survives. This module is that
step, written once so EXP-082 and EXP-088 measure the same quantity the same
way rather than each rolling its own.

Two estimators, because the two callers need different spatial supports.

1. :class:`VoxelCaliber` -- **distance transform of the segmentation itself.**
   Given one read box of v117 labels (``CloudVolume(..., agglomerate=True,
   timestamp=V117_TIMESTAMP)``, the same read EXP-082's ``probe_v117.py``
   made), the radius of object ``o`` at a point ``p`` is

       max over voxels v of object o with |centre(v) - p| <= local_nm
           of  EDT_o(v)

   where ``EDT_o`` is the Euclidean distance transform, in nanometres, of the
   binary mask ``labels == o``, computed with ``sampling=resolution_nm``. That
   is: the largest sphere inscribed in the object that is centred within
   ``local_nm`` of the query point. It is the same convention the level-2
   cache's ``max_dt_nm`` uses (a maximum of a distance transform over a
   neighbourhood), so the two estimators below are commensurable, and it is
   what makes a caliber defined at a point that need not sit on the medial
   axis.

   **Resolution is never assumed.** ``resolution_nm`` must be the volume's own
   ``cv.resolution`` at the mip actually read, passed in by the caller and
   recorded by the caller. The repository contains two different claims about
   what "mip 2" means -- ``scripts/build_object_clouds.py`` says 16/16/20 nm,
   the registry note on EXP-072 says 32/32/40 nm -- because they are different
   volumes. Hardcoding either would be a units bug of exactly the kind that
   silently changes a result, so this module refuses to guess. What it does
   assume is that ``labels`` is indexed ``[x, y, z]`` in the same axis order as
   ``resolution_nm`` and ``origin_vox``, which is how CloudVolume returns a
   cutout.

   Two failure modes are reported rather than hidden, because both bias the
   radius *downwards* and a quiet under-estimate would look like a real
   caliber mismatch:

   * ``truncated`` -- the halo sub-box was clipped by the edge of the read box,
     so tissue outside the read is being treated as background.
   * ``saturated`` -- the measured radius reached ``halo_nm``, so the object is
     at least that thick and the true radius is unknown. Radii above the halo
     cannot be measured from a halo-sized window.

   Cost is why there is a halo at all: the transform runs on a sub-box around
   the query point, not on the whole read box, so measuring five objects in one
   box costs five small transforms instead of five large ones.

2. :class:`L2Caliber` -- **the level-2 cache's distance transform.**
   ``neuronauts.harness.geometry.fetch_l2_attributes`` already pulls
   ``max_dt_nm`` and ``mean_dt_nm`` per level-2 node into
   ``data/substrate/geom/l2_attributes.npz``; ``objgeom.ObjectGeometry.radii``
   already exposes ``max_dt_nm`` as "local radius". This is the route EXP-082
   needs, because its unit is a skeleton vertex and it has 650,200 of them: a
   pcg_skel v4 skeleton carries ``lvl2_ids`` alongside ``vertices``, so a
   vertex maps to a level-2 node and a level-2 node carries a caliber, with no
   volume reads at all.

   Its spatial support is a level-2 chunk, which is far coarser than a voxel
   neighbourhood -- ``max_dt_nm`` is a maximum over the whole chunk, so a chunk
   containing a bouton or a spine head over-reports the shaft it also contains,
   and ``mean_dt_nm`` under-reports for the same reason. Both are returned;
   the caller states which it used. Coverage is whatever has been fetched, and
   the fetch is region-scoped, so ``radius_nm`` returns NaN for level-2 nodes
   outside it rather than a number.

**Neither estimator is validated here.** Nothing in this module knows whether
its output agrees with the proofread ``radius`` it is meant to replace. A
caller must measure that agreement on sites where both exist and report it --
correlation and median ratio against the skeleton radius at the same place --
before treating a v117 caliber as a substitute. EXP-088 does this and puts the
numbers in its outcome; a future caller should do the same rather than inherit
the assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

#: Half-width of the sub-box the distance transform runs on, in nm. Radii at or
#: above this cannot be measured from a window this size and come back
#: ``saturated``. 1 um comfortably covers axonal and thin-dendritic caliber
#: (EXP-082 measured a proofread-skeleton axon radius range of roughly 100-300
#: nm) without paying for a transform over a whole read box.
DEFAULT_HALO_NM = 1000.0
#: How far from the query point an inscribed sphere's centre may sit and still
#: count as "the caliber here". Too small and the estimate collapses onto the
#: distance transform at one voxel, which under-reports off the medial axis;
#: too large and it wanders onto neighbouring structure.
DEFAULT_LOCAL_NM = 300.0


@dataclass
class CaliberEstimate:
    """One radius measurement and everything that could be wrong with it."""

    radius_nm: float                 #: NaN when the object has no voxel in range
    n_voxels: int                    #: object voxels inside ``local_nm``
    saturated: bool                  #: hit ``halo_nm``; true radius is >= this
    truncated: bool                  #: halo clipped by the read box edge

    @property
    def usable(self) -> bool:
        return bool(np.isfinite(self.radius_nm)
                    and not self.saturated and not self.truncated)


@dataclass
class VoxelCaliber:
    """One cutout of v117 labels, with caliber measured by distance transform.

    ``labels`` is the raw ``[x, y, z]`` segmentation cutout, ``origin_vox`` its
    lower corner in voxels, and ``resolution_nm`` the volume's own resolution
    at the mip that was read -- see the module docstring on why that is a
    required argument and not a constant.
    """

    labels: np.ndarray
    origin_vox: np.ndarray
    resolution_nm: np.ndarray

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels)
        if self.labels.ndim != 3:
            raise ValueError(f"labels must be [x,y,z], got {self.labels.shape}")
        self.origin_vox = np.asarray(self.origin_vox, np.int64).reshape(3)
        self.resolution_nm = np.asarray(self.resolution_nm, np.float64).reshape(3)
        if not np.all(self.resolution_nm > 0):
            raise ValueError(f"resolution_nm must be positive, got "
                             f"{self.resolution_nm.tolist()}")

    # -- frames ------------------------------------------------------------
    @property
    def shape_vox(self) -> np.ndarray:
        return np.asarray(self.labels.shape, np.int64)

    def to_vox(self, p_nm) -> np.ndarray:
        """Nearest voxel index inside this box, unclipped (may be negative)."""
        p = np.asarray(p_nm, np.float64)
        return np.rint(p / self.resolution_nm).astype(np.int64) - self.origin_vox

    def contains(self, p_nm, margin_nm: float = 0.0) -> bool:
        v = self.to_vox(p_nm)
        m = np.ceil(margin_nm / self.resolution_nm).astype(np.int64)
        return bool(np.all(v >= m) and np.all(v < self.shape_vox - m))

    def object_at(self, p_nm) -> int:
        """Object id at ``p_nm``; 0 for background or outside the box."""
        v = self.to_vox(p_nm)
        if np.any(v < 0) or np.any(v >= self.shape_vox):
            return 0
        return int(self.labels[v[0], v[1], v[2]])

    # -- neighbourhood -----------------------------------------------------
    def _window(self, p_nm, half_nm: float):
        """Sub-box slices around ``p_nm``, plus whether it had to be clipped."""
        v = self.to_vox(p_nm)
        h = np.ceil(np.asarray(half_nm, np.float64) / self.resolution_nm
                    ).astype(np.int64)
        lo_want, hi_want = v - h, v + h + 1
        lo = np.maximum(lo_want, 0)
        hi = np.minimum(hi_want, self.shape_vox)
        clipped = bool(np.any(lo != lo_want) or np.any(hi != hi_want))
        if np.any(hi <= lo):
            return None, clipped
        return (slice(int(lo[0]), int(hi[0])),
                slice(int(lo[1]), int(hi[1])),
                slice(int(lo[2]), int(hi[2]))), clipped

    def _centres_nm(self, sl) -> np.ndarray:
        """Voxel-centre coordinates of a sub-box, ``[nx,ny,nz,3]`` nm."""
        ax = [(np.arange(s.start, s.stop, dtype=np.float64)
               + self.origin_vox[i]) * self.resolution_nm[i]
              for i, s in enumerate(sl)]
        g = np.meshgrid(*ax, indexing="ij")
        return np.stack(g, axis=-1)

    def objects_within(self, p_nm, radius_nm: float, *,
                       exclude: Iterable[int] = ()) -> list[tuple[int, float, np.ndarray]]:
        """``(object_id, gap_nm, nearest_point_nm)`` for objects within reach.

        ``gap_nm`` is the distance from ``p_nm`` to that object's nearest voxel
        centre -- a centre-to-centre gap at voxel resolution, not a
        surface-to-surface one -- and ``nearest_point_nm`` is that voxel's
        centre. Sorted nearest first; background (0) dropped.
        """
        sl, _ = self._window(p_nm, radius_nm)
        if sl is None:
            return []
        sub = self.labels[sl]
        centres = self._centres_nm(sl)
        d = np.linalg.norm(centres - np.asarray(p_nm, np.float64), axis=-1)
        keep = (d <= float(radius_nm)) & (sub != 0)
        if not keep.any():
            return []
        ids = sub[keep].astype(np.int64)
        dd = d[keep]
        pp = centres[keep]
        order = np.argsort(ids, kind="stable")
        ids, dd, pp = ids[order], dd[order], pp[order]
        uniq, start = np.unique(ids, return_index=True)
        ends = np.r_[start[1:], len(ids)]
        drop = set(int(x) for x in exclude)
        out = []
        for i, s, e in zip(uniq.tolist(), start, ends):
            if int(i) in drop:
                continue
            k = int(np.argmin(dd[s:e]))
            out.append((int(i), float(dd[s:e][k]), pp[s:e][k].copy()))
        out.sort(key=lambda t: t[1])
        return out

    def point_at_range(self, obj: int, p_nm, direction, range_nm: float, *,
                       cone_cos: float = 0.5, tol_nm: float = 500.0
                       ) -> Optional[np.ndarray]:
        """A voxel centre of ``obj`` about ``range_nm`` from ``p_nm``, in ``direction``.

        Used to sample a candidate object at the same stand-off distance as the
        skeleton arm it is competing with, so a caliber comparison is not
        secretly a comparison of where the two were measured. Returns None when
        the object has no voxel in the cone at that range.
        """
        sl, _ = self._window(p_nm, float(range_nm) + float(tol_nm))
        if sl is None:
            return None
        mask = self.labels[sl] == np.asarray(obj, self.labels.dtype)
        if not mask.any():
            return None
        centres = self._centres_nm(sl)
        v = centres - np.asarray(p_nm, np.float64)
        d = np.linalg.norm(v, axis=-1)
        u = np.asarray(direction, np.float64)
        nu = float(np.linalg.norm(u))
        if nu <= 0:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = (v @ (u / nu)) / np.where(d > 0, d, np.nan)
        ok = mask & np.isfinite(cos) & (cos >= float(cone_cos)) \
            & (np.abs(d - float(range_nm)) <= float(tol_nm))
        if not ok.any():
            return None
        err = np.where(ok, np.abs(d - float(range_nm)), np.inf)
        k = np.unravel_index(int(np.argmin(err)), err.shape)
        return centres[k].copy()

    # -- the measurement ---------------------------------------------------
    def radius(self, obj: int, p_nm, *, local_nm: float = DEFAULT_LOCAL_NM,
               halo_nm: float = DEFAULT_HALO_NM) -> CaliberEstimate:
        """Caliber of ``obj`` at ``p_nm``, in nm. See the module docstring."""
        if int(obj) == 0:
            return CaliberEstimate(float("nan"), 0, False, False)
        sl, clipped = self._window(p_nm, float(local_nm) + float(halo_nm))
        if sl is None:
            return CaliberEstimate(float("nan"), 0, False, True)
        mask = self.labels[sl] == np.asarray(obj, self.labels.dtype)
        if not mask.any():
            return CaliberEstimate(float("nan"), 0, False, clipped)
        dt = distance_transform_edt(mask, sampling=self.resolution_nm)
        d = np.linalg.norm(self._centres_nm(sl) - np.asarray(p_nm, np.float64),
                           axis=-1)
        cand = mask & (d <= float(local_nm))
        if not cand.any():
            return CaliberEstimate(float("nan"), 0, False, clipped)
        r = float(dt[cand].max())
        sat = r >= float(halo_nm) - float(self.resolution_nm.max())
        return CaliberEstimate(r, int(cand.sum()), bool(sat), bool(clipped))


def read_v117_box(cv, lo_nm, hi_nm, *, pad_nm: float = 0.0,
                  max_voxels: Optional[int] = None) -> Optional[VoxelCaliber]:
    """Read one v117 cutout covering ``[lo_nm, hi_nm]`` padded by ``pad_nm``.

    ``cv`` is an already-configured CloudVolume -- this module deliberately does
    not build one, so the caller owns the mip, the timestamp and the
    ``agglomerate`` flag and can record them. Returns ``None`` when the request
    would exceed ``max_voxels`` or degenerates to nothing, so a caller can
    count skipped sites instead of discovering a memory blow-up mid-run.
    """
    res = np.asarray(cv.resolution, np.float64).reshape(3)
    lo = np.floor((np.asarray(lo_nm, np.float64) - pad_nm) / res).astype(np.int64)
    hi = np.ceil((np.asarray(hi_nm, np.float64) + pad_nm) / res).astype(np.int64) + 1
    bmin = np.asarray(cv.bounds.minpt, np.int64)
    bmax = np.asarray(cv.bounds.maxpt, np.int64)
    lo = np.maximum(lo, bmin)
    hi = np.minimum(hi, bmax)
    if np.any(hi - lo < 2):
        return None
    if max_voxels is not None and float(np.prod(hi - lo)) > float(max_voxels):
        return None
    vol = np.asarray(cv[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])[..., 0]
    return VoxelCaliber(labels=vol, origin_vox=lo, resolution_nm=res)


# ---------------------------------------------------------------------------
# level-2 cache route
# ---------------------------------------------------------------------------

#: Attribute names the level-2 cache serves for caliber, in
#: ``data/substrate/geom/l2_attributes.npz`` as written by
#: :func:`neuronauts.harness.geometry.fetch_l2_attributes`.
L2_CALIBER_STATS = ("max_dt_nm", "mean_dt_nm")


@dataclass
class L2Caliber:
    """Per-level-2-node caliber from the level-2 cache's distance transform."""

    l2_id: np.ndarray                  #: sorted uint64
    stats: dict                        #: name -> [N] float32, NaN where unfetched

    def radius_nm(self, l2_ids, stat: str = "max_dt_nm") -> np.ndarray:
        """Caliber for each level-2 id, NaN for ids this cache does not hold."""
        if stat not in self.stats:
            raise KeyError(f"{stat!r} not in this cache; have "
                           f"{sorted(self.stats)}")
        q = np.asarray(l2_ids, np.uint64)
        pos = np.searchsorted(self.l2_id, q)
        pos = np.clip(pos, 0, max(len(self.l2_id) - 1, 0))
        hit = (len(self.l2_id) > 0) & (self.l2_id[pos] == q)
        out = np.full(len(q), np.nan, np.float64)
        out[hit] = self.stats[stat][pos[hit]]
        return out

    @property
    def coverage(self) -> int:
        return len(self.l2_id)


def load_l2_caliber(path) -> L2Caliber:
    """Load ``l2_attributes.npz``; keeps only the id and the caliber columns."""
    with np.load(Path(path), allow_pickle=False) as z:
        l2 = np.asarray(z["l2_id"], np.uint64)
        order = np.argsort(l2, kind="stable")
        stats = {k: np.asarray(z[k], np.float64)[order]
                 for k in L2_CALIBER_STATS if k in z.files}
    if not stats:
        raise ValueError(f"{path}: none of {L2_CALIBER_STATS} present; this is "
                         "not a level-2 attribute cache")
    return L2Caliber(l2_id=l2[order], stats=stats)


def vertex_radii_from_l2(lvl2_ids, cal: L2Caliber, *, n_vertices: Optional[int] = None,
                         stat: str = "max_dt_nm") -> np.ndarray:
    """Per-skeleton-vertex v117 caliber, for the EXP-082 recomputation.

    A pcg_skel v4 skeleton stores ``lvl2_ids`` alongside ``vertices``; this
    maps each vertex to its level-2 node's caliber. ``n_vertices`` is checked
    when given rather than assumed: nothing in this repository yet consumes
    ``lvl2_ids``, so the one-id-per-vertex correspondence is stated by the
    skeleton service and has not been verified here. A caller that cannot show
    the lengths match should not use this.
    """
    ids = np.asarray(lvl2_ids, np.uint64).ravel()
    if n_vertices is not None and len(ids) != int(n_vertices):
        raise ValueError(f"lvl2_ids has {len(ids)} entries for "
                         f"{int(n_vertices)} vertices; the per-vertex "
                         "correspondence does not hold for this skeleton")
    return cal.radius_nm(ids, stat=stat)
