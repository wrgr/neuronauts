"""Abstract schemas for the global tree stitching problem.

These are thin re-exports of the neuronauts schemas with domain-agnostic
aliases and a lightweight ObservationGraph dataclass that mirrors
HalfSynapseGraph without neuro-specific field names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Re-export Fragment and NeuronHypothesis under generic names.
# Fragment already uses abstract field names (vertices_nm, edges, endpoints_nm,
# base_root_id, dna) — it maps cleanly to the abstract problem.
from neuronauts.schemas import Fragment  # noqa: F401
from neuronauts.schemas import NeuronHypothesis as ObjectHypothesis  # noqa: F401


@dataclass
class ObservationGraph:
    """Typed-edge graph of observations for partition learning.

    This is the domain-agnostic equivalent of HalfSynapseGraph.  One node per
    observation (e.g. a synapse, a GPS ping, a cell-division event).  Typed
    edges connect observations that share evidence for co-membership in the
    same parent tree.

    Attributes
    ----------
    node_feat : [N, 3+D]
        Node features: concat(normalised position [3], fragment embedding [D]).
        The embedding is zero for observations whose fragment has no embedding.
    node_pos : [N, 3]
        Raw nm-scale positions of observations.
    edge_src : [E] int64
        Source node indices.
    edge_dst : [E] int64
        Destination node indices.
    edge_type : [E] int64
        0 = same-fragment  — strong (possibly noisy) co-membership evidence.
        1 = spatial k-NN   — weak proximity evidence.
        2 = endpoint-adj   — fragment-level topological evidence (optional).
    edge_feat : [E, 3] float32
        [is_same_frag, is_spatial, embed_cos_sim].
    labels : [N] int64
        Ground-truth parent-tree IDs (supervision only, NOT input features).
        0 = unlabelled / held-out.
    fragment_id : [N] int64
        Which fragment each observation belongs to (noisy evidence channel).
    side : str
        Arbitrary tag for multi-sided observations (e.g. "pre" / "post" /
        "source" / "target").  Empty string when not applicable.
    """

    node_feat: np.ndarray
    node_pos: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_type: np.ndarray
    edge_feat: np.ndarray
    labels: np.ndarray
    fragment_id: np.ndarray
    side: str = ""

    @property
    def n_nodes(self) -> int:
        return len(self.node_feat)

    @property
    def n_edges(self) -> int:
        return len(self.edge_src)

    @property
    def node_dim(self) -> int:
        return self.node_feat.shape[1]

    @property
    def embed_dim(self) -> int:
        return self.node_dim - 3

    @classmethod
    def from_half_synapse_graph(cls, g) -> "ObservationGraph":
        """Wrap a HalfSynapseGraph as an ObservationGraph."""
        return cls(
            node_feat=g.node_feat,
            node_pos=g.node_pos,
            edge_src=g.edge_src,
            edge_dst=g.edge_dst,
            edge_type=g.edge_type,
            edge_feat=g.edge_feat,
            labels=g.labels,
            fragment_id=g.seg_id,
            side=g.side,
        )
