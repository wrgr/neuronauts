"""Representation stage: Fragment → learned DNA embedding (SkeletonGNN)."""

from .skeleton_gnn import (
    SkeletonGNN,
    encode_fragments_gnn,
    fragment_to_tensors,
    train_skeleton_gnn,
)

__all__ = [
    "SkeletonGNN",
    "encode_fragments_gnn",
    "fragment_to_tensors",
    "train_skeleton_gnn",
]
