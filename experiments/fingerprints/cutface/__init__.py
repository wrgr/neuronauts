"""cutface -- a modular cut-face re-identification library.

Re-link the two faces of a segmentation break by hashing local EM ultrastructure,
arbitrated against spatial proximity.  Built for MICrONS v117 false-splits but
organized so the pieces can be reused on other re-identification / proofreading
tasks.  The five contributions, each a submodule with a docstring on what to swap
for a new application:

    cutface.data        IO substrate -- client, box fetch (EM+seg+curseg), cache
    cutface.candidates  CONTRIBUTION 1 -- find error sites + build the panel
    cutface.patches     CONTRIBUTION 2 -- the masked cut-face (bands / depth stack)
    cutface.features    CONTRIBUTION 3 -- learned embedding + combiner features
    cutface.evaluate    CONTRIBUTION 4 -- panel recall, combiner top-1, abstention

Typical pipeline (see README):
    from experiments.fingerprints import cutface as cf
    cl = cf.data.client(); ts = cl.chunkedgraph.get_oldest_timestamp()
    roots, _ = cf.candidates.find_split_neurons(cl, n_scan=200)
    sites = cf.candidates.find_error_sites(cl, roots[0], ts)
    panel = cf.patches.face_panel(cl, ts, sites[0])          # query + candidates
    bio, art = cf.features.load_encoder(...), cf.features.load_encoder(...)
    # -> combiner_features -> train_combiner -> combiner_top1 / abstention_curve

Headline result (N~73 real held-out sites): geometry alone 0.65; the learned
combiner over geometry + fine-tuned bio band reaches 0.77 (panel recall ~1.0
after the location-layer fixes).  16 nm single-slab is the representation sweet
spot (depth ties, 8 nm hurts).  See README for the full story; superseded
drivers and intermediate artifacts are under ``../archive/``.
"""

from __future__ import annotations

from . import data, candidates, patches, features, evaluate

__all__ = ["data", "candidates", "patches", "features", "evaluate"]
