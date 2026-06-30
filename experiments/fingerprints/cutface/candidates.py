"""CONTRIBUTION 1 -- how to get the candidates.

Given a segmentation with known errors, produce, per error site, a *query point*
and a *candidate panel*: the neurites that physically approach the dangling tip,
filtered to a realistic merge-proposal set.  Two stages:

* **Find error sites** (`find_split_neurons` + `find_error_sites`).  v117-specific:
  a proofread root assembled by merging several historical fragments was
  *falsely split*; each fragment interface is a real error a human fixed.  This is
  the part you SWAP for a new application -- substitute any source of
  ``ErrorSite(pos_main_nm, pos_frag_nm, tangent_nm, root)`` (e.g. flagged
  low-confidence edges, synthetic cuts, a different proofreading diff).
* **Build the panel** (`proximity_candidates` + a direction `cone` + `local_tangent`).
  Application-agnostic: within a radius of the query tip, every other fragment of
  >= ``min_vox`` voxels, optionally restricted to a forward cone along the arbor's
  local tangent.  Flooding the ball (~256 distractors) drowns correction; the
  cone trims it to ~45 -- evaluate location (recall of the panel) and correction
  (top-1 given the panel) separately.
"""

from __future__ import annotations

from .v117_error_relink import (
    ErrorSite,
    find_split_neurons,
    sites_from_l2_graph as find_error_sites,
    _proximity_candidates as proximity_candidates,
    _local_tangent as local_tangent,
    _l2_positions as l2_positions,
)

__all__ = ["ErrorSite", "find_split_neurons", "find_error_sites",
           "proximity_candidates", "local_tangent", "l2_positions"]
