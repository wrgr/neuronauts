"""
Typed schemas and contracts for the Next-Gen Global Merge & Assembly Engine.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class EdgeType(str, Enum):
    SAME_SEGMENT = "same_segment"
    TANGENT_FLOW = "tangent_flow"
    SPATIAL_KNN = "spatial_knn"
    EDIT_SPLIT_HARD_NEG = "edit_split_hard_neg"


@dataclass
class EndpointTangent:
    """Represents a skeleton leaf/endpoint with 3D orientation and caliber."""
    fragment_id: str
    vertex_idx: int
    coord_nm: np.ndarray  # shape (3,)
    tangent: np.ndarray   # unit vector pointing outward along arbor, shape (3,)
    radius_nm: float
    curvature: float = 0.0

    def to_dict(self) -> dict:
        return {
            "fragment_id": self.fragment_id,
            "vertex_idx": int(self.vertex_idx),
            "coord_nm": self.coord_nm.tolist() if isinstance(self.coord_nm, np.ndarray) else list(self.coord_nm),
            "tangent": self.tangent.tolist() if isinstance(self.tangent, np.ndarray) else list(self.tangent),
            "radius_nm": float(self.radius_nm),
            "curvature": float(self.curvature),
        }


@dataclass
class SegmentFragment:
    """Atomic fragment of a neuron (kimimaro skeleton or point cloud cluster)."""
    fragment_id: str
    segment_id: int
    vertices_nm: np.ndarray  # shape (N, 3)
    radii_nm: np.ndarray     # shape (N,)
    edges: np.ndarray        # shape (E, 2)
    endpoints: List[EndpointTangent] = field(default_factory=list)
    synapse_ids: List[int] = field(default_factory=list)
    synapse_coords_nm: Optional[np.ndarray] = None  # shape (S, 3)
    synapse_types: Optional[np.ndarray] = None      # shape (S,) 0=pre, 1=post
    synapse_partner_ids: Optional[np.ndarray] = None# shape (S,) partner root IDs
    point_cloud_nm: Optional[np.ndarray] = None     # shape (P, 3)
    is_soma: bool = False
    soma_confidence: float = 0.0
    dna_embedding: Optional[np.ndarray] = None      # shape (D,)

    @property
    def path_length_nm(self) -> float:
        if len(self.edges) == 0 or len(self.vertices_nm) < 2:
            return 0.0
        v_src = self.vertices_nm[self.edges[:, 0]]
        v_dst = self.vertices_nm[self.edges[:, 1]]
        return float(np.sum(np.linalg.norm(v_dst - v_src, axis=1)))

    @property
    def centroid_nm(self) -> np.ndarray:
        if len(self.vertices_nm) == 0:
            return np.zeros(3)
        return np.mean(self.vertices_nm, axis=0)


@dataclass
class AssemblyEdge:
    """An edge between two fragments in the global assembly graph."""
    src_id: str
    dst_id: str
    edge_type: EdgeType
    distance_nm: float
    collinearity_score: float = 0.0
    dna_similarity: float = 0.0
    synapse_coassign_score: float = 0.0
    synapse_polarity_score: float = 0.0
    weight: float = 0.0
    is_hard_negative: bool = False
    metadata: Dict = field(default_factory=dict)


@dataclass
class NeuronHypothesis:
    """A reconstructed global neuron assembled from connected fragments."""
    neuron_id: str
    fragment_ids: List[str]
    total_path_length_nm: float
    synapse_count: int
    has_soma: bool = False
    is_valid_tree: bool = True
    confidence_score: float = 1.0
    pooled_dna: Optional[np.ndarray] = None


@dataclass
class GlobalAssemblyResult:
    """Output of the global merge & assembly pipeline."""
    neurons: List[NeuronHypothesis]
    fragment_to_neuron: Dict[str, str]
    num_merges: int
    num_splits_prevented: int
    metrics: Dict[str, float] = field(default_factory=dict)
