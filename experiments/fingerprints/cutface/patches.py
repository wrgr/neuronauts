"""CONTRIBUTION 2 -- how to compute the patch (the cut-face representation).

Given a box and a query/candidate location, build the masked cross-section
"face" that the hash compares.  Layers, cheapest first:

* `seg_id_at` / `z_index`  -- locate the fragment id and z-section at a world point.
* `patch_from_slab`        -- raw masked, translation-normalized PATCH x PATCH face.
* `band_face`              -- split a face into a **bio** (low-pass, shape) band and
  an **art** (high-pass, texture) band; the low band is what transfers across a
  real cut.
* `depth_stack`            -- N z-sections marching *away from the cut* (don't cross
  the gap), kept as channels so trajectory / caliber-taper survive.

Panel assemblers (face for the query + every candidate, with is_true + geometry):
* `face_panel`        -- single-slab bio/art faces  (`site_faces_bands`).
* `face_panel_depth`  -- depth-stack faces, identify at one mip, sample at another
  (`site_faces_bands_depth`).

Verdict from the experiment: 16 nm single-slab is the sweet spot -- depth ties it,
8 nm hurts.  ``require_true``/``use_site_root`` and the not-a-split drop live here
(see `face_panel`), so this module also encodes "what counts as a real site".
"""

from __future__ import annotations

from .v117_error_relink import (
    _seg_id_at as seg_id_at,
    _z_index as z_index,
    _patch_from_slab as patch_from_slab,
)
from .v117_artifact_bands import _band_face as band_face, site_faces_bands as face_panel
from .band_faces_depth import _stack as depth_stack, site_faces_bands_depth as face_panel_depth

__all__ = ["seg_id_at", "z_index", "patch_from_slab", "band_face", "depth_stack",
           "face_panel", "face_panel_depth"]
