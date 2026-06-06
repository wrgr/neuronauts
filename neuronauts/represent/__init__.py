"""Representation stage: Fragment → learned tree-DNA embedding."""

from .dna import (
    TreeDNAEncoder,
    encode_fragments,
    featurize_fragment,
    sample_tree_paths,
    train_dna_encoder,
)

__all__ = [
    "TreeDNAEncoder",
    "encode_fragments",
    "featurize_fragment",
    "sample_tree_paths",
    "train_dna_encoder",
]
