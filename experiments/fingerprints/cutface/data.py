"""Data substrate: client, box fetch (EM + seg + current-root paint), disk cache.

CONTRIBUTION 0 (the IO layer every application needs).

A "box" is a small EM+segmentation cube around a candidate site.  ``fetch_box``
returns a :class:`Volume` carrying:
  * ``em``     -- uint8 EM intensity [X,Y,Z]
  * ``seg``    -- uint64 *historical-root*-painted segmentation (the error-era ids)
  * ``curseg`` -- uint64 *current-root*-painted segmentation (per-voxel, vote-free)
  * resolution / origin metadata
plus ``frag2cur`` (historical fragment -> majority current root, legacy fallback).

To run on a DIFFERENT application, this is the main thing to swap: point
``fetch_box`` at your volume + segmentation and supply whatever "id at a voxel"
your task needs.  Everything downstream is resolution- and dataset-agnostic.

Caches: set env ``EM_BOX_CACHE`` / ``V117_BOX_CACHE`` to disk dirs (default under
``data/``, gitignored); per-box fetches are the dominant cost, so caching makes
re-runs with different encoders/evals nearly free.
"""

from __future__ import annotations

from .fingerprint_break_resolution import Volume, PATCH
from .v117_error_relink import _client as client, _box_key as box_key
from .v117_reconstructed import fetch_v117_box as fetch_box, _sv_volume, _ensure_secret

__all__ = ["Volume", "PATCH", "client", "box_key", "fetch_box",
           "_sv_volume", "_ensure_secret"]
