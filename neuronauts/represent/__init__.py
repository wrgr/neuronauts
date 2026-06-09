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
from .skeleton_gnn import (
    SkeletonGNN,
    encode_fragments_gnn,
    fragment_to_tensors,
    train_skeleton_gnn,
)
from .path_grammar import (
    PathGrammarReranker,
    fragment_to_intrinsic_paths,
    path_to_intrinsic,
    score_fragment_pairs,
    train_path_grammar_reranker,
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
    "SkeletonGNN",
    "encode_fragments_gnn",
    "fragment_to_tensors",
    "train_skeleton_gnn",
    "PathGrammarReranker",
    "fragment_to_intrinsic_paths",
    "path_to_intrinsic",
    "score_fragment_pairs",
    "train_path_grammar_reranker",
]
