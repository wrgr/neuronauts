"""Fragment embedding: encode tree fragments as fixed-size vectors.

FragmentEncoder (= SkeletonGNN) takes a tree fragment (vertices, edges,
radii) and produces a single embedding vector [D].  Fragments from the
same parent tree should end up close; fragments from different trees
should be far apart.

Node features: centroid-normalised (x-cx, y-cy, z-cz, radius) — four
scalars per vertex.  Centroid normalisation removes global position and
orientation so the encoder focuses on morphological shape.

Usage
-----
    from treestitch.embed import FragmentEncoder, encode_fragments, train_fragment_encoder
    from treestitch import Fragment

    encoder = FragmentEncoder(d_model=64, output_dim=32)
    encoder = train_fragment_encoder(encoder, [fragments], root_label_map=label_map)
    fragments_with_embeddings = encode_fragments(encoder, fragments)
"""

from __future__ import annotations

from typing import Any

# FragmentEncoder is SkeletonGNN with a generic alias.
# The architecture (centroid-normalised GNN → L2-norm output) is domain-agnostic:
# it works for any tree-structured 3-D fragment regardless of the physical domain.
from neuronauts.represent.skeleton_gnn import SkeletonGNN as FragmentEncoder
from neuronauts.represent.skeleton_gnn import encode_fragments_gnn as encode_fragments
from neuronauts.represent.skeleton_gnn import train_skeleton_gnn as train_fragment_encoder

__all__ = ["FragmentEncoder", "encode_fragments", "train_fragment_encoder"]
