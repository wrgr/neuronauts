"""Representation stage: Fragment → learned tree-DNA embedding."""

from .dna import (
    TreeDNAEncoder,
    encode_fragments,
    featurize_fragment,
    sample_tree_paths,
    train_dna_encoder,
)
from .enrich import (
    build_synapse_dna_matrix,
    evaluate_dna_auc,
    spatial_proximity_scores,
    synapse_pair_dna_scores,
)

__all__ = [
    "TreeDNAEncoder",
    "encode_fragments",
    "featurize_fragment",
    "sample_tree_paths",
    "train_dna_encoder",
    "build_synapse_dna_matrix",
    "evaluate_dna_auc",
    "spatial_proximity_scores",
    "synapse_pair_dna_scores",
]
