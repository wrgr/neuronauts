"""CONTRIBUTION 3 -- how to compute features (turn faces into a decision signal).

Two halves:

**Learned cut-face embedding** -- a small contrastive CNN that maps a face to a
unit vector; cosine in that space is the hash.  Pulls the two faces of one
fragment (sampled at z-separated sections, a synthetic "cut") together, pushes
different fragments apart (NT-Xent / InfoNCE).
* 2D faces:  `build_encoder` / `embed_patches` / `make_embed_fn` / `load_encoder`,
  trained/fine-tuned with `finetune`.
* depth stacks: `build_depth_encoder` / `embed_stacks` / `make_stack_embed_fn` /
  `load_depth_encoder`, trained with `finetune_depth`.
Train at scale on unlimited *synthetic* same-fragment pairs (`mine_box` /
`mine_from_cache`), then adapt to the real break distribution on the scarce real
pairs (`collect_real_band_pairs`).  Finding: the low-pass *bio* band is what
transfers across a real cut; the synthetic *art*-band advantage does not.

**Per-candidate feature vector + combiner** -- geometry alone already gets ~0.65,
so don't replace it: `combiner_features` builds, per candidate, the feature row
``[geom z, art z, bio z, art, bio, is-geom-nearest, is-art-best]`` and
`train_combiner` learns when to trust the hash over distance (a tiny MLP).  This
learned arbitration is what turns the complementary hash signal into a top-1 win.
"""

from __future__ import annotations

# --- learned embedding (2D faces) ---
from .learned_cutface_encoder import (
    build_encoder, embed_patches, make_embed_fn, load_encoder, _normalize_patches,
)
from .train_real_cutface import finetune

# --- learned embedding (depth stacks) ---
from .train_depth_bands import (
    build_depth_encoder, embed_stacks, make_stack_embed_fn, load_depth_encoder,
    finetune_depth,
)

# --- synthetic + real training pairs ---
from .train_synthetic_skeleton import (
    mine_box, mine_from_cache, collect_real_band_pairs, _fragment_z_extents,
)

# --- per-candidate features + the confidence combiner ---
from .train_combiner import (
    site_features as combiner_features,
    train_mlp as train_combiner,
    _score as combiner_score,
    _sims as band_sims,
    _z as zscore,
)

__all__ = [
    "build_encoder", "embed_patches", "make_embed_fn", "load_encoder", "finetune",
    "build_depth_encoder", "embed_stacks", "make_stack_embed_fn", "load_depth_encoder",
    "finetune_depth", "mine_box", "mine_from_cache", "collect_real_band_pairs",
    "combiner_features", "train_combiner", "combiner_score", "band_sims", "zscore",
]
