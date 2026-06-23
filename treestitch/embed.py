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

from typing import Any, Sequence

import numpy as np

# FragmentEncoder is SkeletonGNN with a generic alias.
# The architecture (centroid-normalised GNN → L2-norm output) is domain-agnostic:
# it works for any tree-structured 3-D fragment regardless of the physical domain.
from neuronauts.represent.skeleton_gnn import SkeletonGNN as FragmentEncoder
from neuronauts.represent.skeleton_gnn import encode_fragments_gnn as encode_fragments
from neuronauts.represent.skeleton_gnn import train_skeleton_gnn as train_fragment_encoder


def encode_fragments_morphological(
    fragments: Sequence,
    *,
    device: str = "cpu",
    output_dim: int = 32,
) -> list:
    """Deterministic morphological descriptors — no training required, no collapse.

    Computes a fixed descriptor from each fragment's point cloud: PCA spread
    (eigenvalue spectrum), elongation, centroid-distance radii stats, and log
    node count.  L2-normalised to output_dim (padded with zeros).

    The descriptor encodes fragment SIZE and SHAPE but not global position,
    making it translation-invariant and directly comparable across regions.
    The partition GNN uses it alongside spatial position and edge type — even
    a coarse shape signal improves cross-region generalisation over a collapsed
    GNN encoder (which reduces to a constant vector for all inputs).
    """
    from neuronauts.schemas import Fragment

    SCALE = 1e4  # 10 µm reference scale (nm)

    result = []
    for frag in fragments:
        pts = frag.vertices_nm.astype(np.float64)
        n = len(pts)
        centered = pts - pts.mean(0)

        # PCA of the point cloud
        if n >= 3:
            cov = (centered.T @ centered) / max(n - 1, 1)
            eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]   # descending
            spread = np.sqrt(np.maximum(eigvals, 0))            # std along each axis (nm)
        elif n == 2:
            d = np.linalg.norm(centered[0] - centered[1])
            spread = np.array([d / 2, 0.0, 0.0])
        else:
            spread = np.zeros(3)

        # Centroid-distance radii (set by _cloud_fragment as |xi - cx|)
        r = frag.radius_nm.astype(np.float64)
        r_mean = r.mean()
        r_max = r.max() if n > 0 else 0.0
        r_std = r.std() if n > 1 else 0.0

        # Endpoint distance (longest axis extent)
        if len(frag.endpoints_nm) == 2:
            ep_dist = float(np.linalg.norm(
                frag.endpoints_nm[0] - frag.endpoints_nm[1]))
        else:
            ep_dist = spread[0] * 2

        # Elongation: log ratio of 1st to 3rd PCA axis
        elongation = np.log(spread[0] / (spread[2] + 1.0) + 1.0)
        # Axis ratio 1st/2nd (planarity)
        axis_ratio = spread[0] / (spread[1] + 1e-3)

        feat = np.zeros(output_dim, dtype=np.float32)
        raw = np.array([
            np.log(n + 1),           # 0: log node count
            r_mean / SCALE,          # 1: mean centroid distance
            r_max / SCALE,           # 2: max centroid distance
            r_std / SCALE,           # 3: std centroid distance
            spread[0] / SCALE,       # 4: major PCA axis
            spread[1] / SCALE,       # 5: second PCA axis
            spread[2] / SCALE,       # 6: minor PCA axis
            elongation,              # 7: elongation ratio
            ep_dist / SCALE,         # 8: endpoint distance
            float(axis_ratio),       # 9: 1st/2nd axis ratio
        ], dtype=np.float32)
        feat[:len(raw)] = raw

        norm = float(np.linalg.norm(feat))
        if norm > 1e-8:
            feat = feat / norm

        result.append(Fragment(
            fragment_id=frag.fragment_id,
            region_id=frag.region_id,
            base_root_id=frag.base_root_id,
            vertices_nm=frag.vertices_nm,
            edges=frag.edges,
            endpoints_nm=frag.endpoints_nm,
            radius_nm=frag.radius_nm,
            synapse_indices=frag.synapse_indices,
            dna=feat,
        ).validate())

    return result


__all__ = [
    "FragmentEncoder", "encode_fragments", "train_fragment_encoder",
    "encode_fragments_morphological",
]
