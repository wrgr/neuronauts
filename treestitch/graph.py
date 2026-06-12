"""Build the typed-edge observation graph for partition learning.

build_observation_graph wraps build_half_synapse_graph with domain-agnostic
parameter names.  It takes:
  - A Region (the observation container — positions + fragment_ids + labels)
  - A list of Fragments (with embeddings filled by FragmentEncoder)
And produces an ObservationGraph with three edge types:
  0  same-fragment  — all pairs of observations on the same fragment
  1  spatial k-NN   — k nearest observations in position space
  2  endpoint-adj   — observations on fragments whose endpoints are within
                      endpoint_radius_nm of each other (optional, powerful)

The endpoint-adjacent edges are the key insight: two fragments whose
endpoints are close are likely adjacent pieces of the same parent tree.
Connecting their observations gives the PartitionGNN direct topological
evidence for merging even when spatial proximity is uninformative.

Usage
-----
    from treestitch.graph import build_observation_graph

    graph = build_observation_graph(
        region, fragments,
        side="pre",
        k_spatial=8,
        endpoint_radius_nm=10_000,   # nm — connect adjacent piece endpoints
    )
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from treestitch.schemas import ObservationGraph


def build_observation_graph(
    region,
    fragments: Sequence,
    *,
    side: str = "pre",
    k_spatial: int = 8,
    max_dist_nm: float | None = None,
    pos_scale_nm: float = 50_000.0,
    max_same_fragment_pairs: int = 200,
    endpoint_radius_nm: float | None = None,
    max_endpoint_pairs: int = 200,
) -> ObservationGraph:
    """Build a typed-edge observation graph.

    Parameters
    ----------
    region:
        Container of observations (positions + fragment IDs + ground-truth
        object IDs).  In the neuro domain this is a Region with synapse data.
    fragments:
        List of Fragment objects with embeddings filled.
    side:
        Which side of directed observations to build the graph for.
        Use ``"pre"`` / ``"post"`` for synapses, or any tag for other domains.
    k_spatial:
        Number of spatial nearest-neighbour edges per node (type 1).
    max_dist_nm:
        Prune spatial edges beyond this distance.  ``None`` = no limit.
    pos_scale_nm:
        Divisor for position normalisation in node features.
        Default 50 µm keeps position and embedding features on similar scales.
    max_same_fragment_pairs:
        Cap on directed same-fragment pairs per fragment, preventing O(N²)
        blowup from large merged fragments.
    endpoint_radius_nm:
        Radius (nm) for endpoint-adjacent edges (type 2).
        ``None`` (default) disables endpoint edges entirely.
        Set to ~10× typical skeleton step size to capture adjacent piece
        endpoints from the same parent tree.
    max_endpoint_pairs:
        Cap on directed endpoint-adjacent pairs per fragment pair.

    Returns
    -------
    ObservationGraph
        Typed-edge graph ready for PartitionGNN training.
    """
    from neuronauts.assemble.half_synapse_graph import build_half_synapse_graph

    hsg = build_half_synapse_graph(
        region,
        fragments,
        side=side,
        k_spatial=k_spatial,
        max_dist_nm=max_dist_nm,
        pos_scale_nm=pos_scale_nm,
        max_same_seg_pairs=max_same_fragment_pairs,
        endpoint_radius_nm=endpoint_radius_nm,
        max_endpoint_pairs=max_endpoint_pairs,
    )
    return ObservationGraph.from_half_synapse_graph(hsg)


__all__ = ["build_observation_graph"]
