"""Loader for the public Google SegCLR embeddings (MICrONS minnie65, ``m343``).

SegCLR (Segmentation-Guided Contrastive Learning of Representations) released
64-dim per-node embeddings for the minnie65 dataset, published as CSV files
packed into sharded ZIP archives on a **public** GCS bucket::

    gs://iarpa_microns/minnie/minnie65/embeddings_m343/<variant>/<shard>.zip

Each ``<shard>.zip`` contains one ``<segment_id>.csv`` per segment, whose rows
carry a node coordinate followed by the 64 embedding values.  A segment id maps
to its shard via ``md5_shard`` (10000 shards, 8-byte little-endian key) -- the
exact scheme used by ``connectomics.segclr.reader`` / ``connectomics.common.sharding``,
reimplemented here so we take **no** dependency on the ``connectomics`` package.

Two facts drive the rest of the pipeline (see CLAUDE.md on the version footgun):

* Embeddings are keyed to the **m343 base segmentation**, *not* current CAVE
  root ids.  Segment ids share the ``864691...`` format but are not identical to
  live roots -- so downstream we assign embeddings to skeleton vertices
  **spatially** (nearest point within a cap), needing only coverage, not an id map.
* We read the ``nm_coord_public_offset`` variant, whose coordinates are already
  in **nanometres** (matching ``SkeletonData.vertices``), avoiding a voxel frame.

Access is over HTTPS with byte-range requests, so reading one segment downloads
only the ZIP central directory + that segment's CSV (a few KB), never the whole
~220 MB shard.
"""
from __future__ import annotations

import hashlib
import io
import os
import zipfile
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import requests

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

BUCKET_HTTPS = "https://storage.googleapis.com/iarpa_microns/minnie/minnie65/embeddings_m343"

# variant key -> GCS subdirectory. "nm_coord" is the recommended default: its
# coordinates come back in nm (no voxel conversion).
VARIANTS = {
    "voxel": "segclr_csvzips",
    "nm_coord": "segclr_nm_coord_public_offset_csvzips",
    "agg25um": "segclr_aggregated_25um_csvzips",
    "nm_coord_agg25um": "segclr_nm_coord_public_offset_aggregated_25um_csvzips",
    "nm_coord_agg10um": "segclr_nm_coord_public_offset_aggregated_10um_csvzips",
}

# bytewidth is 64 *bytes* (not bits): the released microns/H01 data uses the
# ``DATA_URL_FROM_KEY_BYTEWIDTH64`` scheme, verified empirically against the
# actual shard membership of the public bucket.
NUM_SHARDS = 10_000
BYTEWIDTH = 64
EMB_DIM = 64

_DEFAULT_CACHE = os.path.join("cache", "segclr")


# --------------------------------------------------------------------------- #
# Sharding (verbatim scheme from connectomics.common.sharding.md5_shard)
# --------------------------------------------------------------------------- #

def md5_shard(
    segment_id: int,
    num_shards: int = NUM_SHARDS,
    byteorder: str = "little",
    bytewidth: int = BYTEWIDTH,
) -> int:
    """Map a segment id to its shard index via md5 hashing."""
    md5 = hashlib.md5()
    md5.update(int(segment_id).to_bytes(bytewidth, byteorder))
    return int.from_bytes(md5.digest(), byteorder) % num_shards


# --------------------------------------------------------------------------- #
# Minimal seekable HTTP file (byte-range) so zipfile reads only what it needs
# --------------------------------------------------------------------------- #

class _HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file-like object backed by HTTP range requests.

    Enough of the file protocol for ``zipfile.ZipFile`` to read the End-Of-
    Central-Directory record, the central directory, and individual members.
    """

    def __init__(self, url: str, session: requests.Session, timeout: float = 60.0):
        self._url = url
        self._session = session
        self._timeout = timeout
        self._pos = 0
        self._size = self._head_size()

    def _head_size(self) -> int:
        r = self._session.head(self._url, timeout=self._timeout, allow_redirects=True)
        if r.status_code == 200 and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
        # Fall back to a 0-0 range GET to learn the total from Content-Range.
        r = self._session.get(
            self._url, headers={"Range": "bytes=0-0"}, timeout=self._timeout
        )
        r.raise_for_status()
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            return int(cr.rsplit("/", 1)[1])
        raise OSError(f"cannot determine size of {self._url!r} (Content-Range={cr!r})")

    # --- io protocol ---
    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self._pos = offset
        elif whence == os.SEEK_CUR:
            self._pos += offset
        elif whence == os.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self._size - self._pos
        if size <= 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + size, self._size) - 1
        headers = {"Range": f"bytes={self._pos}-{end}"}
        r = self._session.get(self._url, headers=headers, timeout=self._timeout)
        r.raise_for_status()
        data = r.content
        self._pos += len(data)
        return data


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #

@dataclass
class SegCLRAssignment:
    """SegCLR embeddings assigned to a specific set of skeleton vertices."""

    vertices_nm: np.ndarray      # [V, 3] the vertices assigned against
    embedding: np.ndarray        # [V, EMB_DIM] float32; NaN rows = uncovered
    covered: np.ndarray          # [V] bool
    coverage_fraction: float
    n_points: int                # embedding points fetched (before assignment)
    variant: str


class SegCLRReader:
    """Reads per-segment SegCLR embeddings from the public sharded ZIPs.

    Parameters
    ----------
    variant:
        Key into :data:`VARIANTS`.  Default ``"nm_coord"`` (coords in nm).
    cache_dir:
        Directory for per-segment ``.npz`` caches.  ``None`` disables caching.
    """

    def __init__(self, variant: str = "nm_coord", cache_dir: str | None = _DEFAULT_CACHE):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
        self.variant = variant
        self._zipdir = f"{BUCKET_HTTPS}/{VARIANTS[variant]}"
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._session = requests.Session()

    # --- low level ---
    def _cache_path(self, seg_id: int) -> str:
        return os.path.join(self.cache_dir or "", f"{self.variant}_{int(seg_id)}.npz")

    def _shard_url(self, seg_id: int) -> str:
        return f"{self._zipdir}/{md5_shard(seg_id)}.zip"

    def read_segment(self, seg_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(points_nm [P,3], embeddings [P, EMB_DIM])`` for one segment.

        Missing segments (not present in their shard) return empty arrays.
        """
        seg_id = int(seg_id)
        if self.cache_dir:
            cp = self._cache_path(seg_id)
            if os.path.exists(cp):
                with np.load(cp) as z:
                    return z["points"], z["emb"]
        pts, emb = self._read_segment_remote(seg_id)
        if self.cache_dir:
            np.savez_compressed(self._cache_path(seg_id), points=pts, emb=emb)
        return pts, emb

    def _read_segment_remote(self, seg_id: int) -> tuple[np.ndarray, np.ndarray]:
        url = self._shard_url(seg_id)
        member = f"{seg_id}.csv"
        f = _HttpRangeFile(url, self._session)
        with zipfile.ZipFile(f) as z:
            try:
                raw = z.read(member)
            except KeyError:
                return (np.zeros((0, 3), np.float64), np.zeros((0, EMB_DIM), np.float32))
        return _parse_csv(raw)

    # --- high level ---
    def assign_to_skeleton(
        self,
        vertices_nm: np.ndarray,
        segment_ids: Iterable[int],
        *,
        max_dist_nm: float = 2000.0,
    ) -> SegCLRAssignment:
        """Fetch embeddings for ``segment_ids`` and assign to each skeleton vertex.

        Each vertex takes the embedding of its nearest embedding point within
        ``max_dist_nm``; vertices with no point in range are left NaN/uncovered.
        Purely spatial -- tolerant of the m343-vs-CAVE id mismatch.
        """
        vertices_nm = np.asarray(vertices_nm, np.float64)
        pts_list, emb_list = [], []
        for sid in segment_ids:
            p, e = self.read_segment(sid)
            if len(p):
                pts_list.append(p)
                emb_list.append(e)
        V = len(vertices_nm)
        if not pts_list:
            return SegCLRAssignment(
                vertices_nm, np.full((V, EMB_DIM), np.nan, np.float32),
                np.zeros(V, bool), 0.0, 0, self.variant,
            )
        points = np.vstack(pts_list)
        embs = np.vstack(emb_list).astype(np.float32)
        return assign_points_to_vertices(
            vertices_nm, points, embs, max_dist_nm=max_dist_nm, variant=self.variant
        )


def assign_points_to_vertices(
    vertices_nm: np.ndarray,
    points_nm: np.ndarray,
    embeddings: np.ndarray,
    *,
    max_dist_nm: float = 2000.0,
    variant: str = "nm_coord",
) -> SegCLRAssignment:
    """Nearest-embedding-point per vertex within ``max_dist_nm`` (cKDTree)."""
    from scipy.spatial import cKDTree

    vertices_nm = np.asarray(vertices_nm, np.float64)
    V = len(vertices_nm)
    emb = np.full((V, embeddings.shape[1]), np.nan, np.float32)
    covered = np.zeros(V, bool)
    if len(points_nm):
        tree = cKDTree(points_nm)
        dist, idx = tree.query(vertices_nm, k=1)
        hit = dist <= max_dist_nm
        emb[hit] = embeddings[idx[hit]]
        covered = hit
    return SegCLRAssignment(
        vertices_nm, emb, covered, float(covered.mean()) if V else 0.0,
        int(len(points_nm)), variant,
    )


def _parse_csv(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Parse a segment CSV into ``(points [P,3], embeddings [P, EMB_DIM])``.

    Row layout is ``<leading cols...>, x, y, z, e0..e63`` -- i.e. the last
    :data:`EMB_DIM` values are the embedding and the three before them are the
    coordinate.  We locate columns by count rather than assuming a fixed number
    of leading id columns.
    """
    text = raw.decode("utf-8")
    pts, embs = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < EMB_DIM + 3:
            continue
        vals = [float(x) for x in parts[-(EMB_DIM + 3):]]
        pts.append(vals[:3])
        embs.append(vals[3:])
    if not pts:
        return (np.zeros((0, 3), np.float64), np.zeros((0, EMB_DIM), np.float32))
    return np.asarray(pts, np.float64), np.asarray(embs, np.float32)
