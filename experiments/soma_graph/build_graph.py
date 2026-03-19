"""Build a soma-level neuron × neuron graph from synapse tables.

Nodes = root IDs (neurons). Edges = pre→post connections. Reuses
GlobalAssemblyGAT from neuronauts for message passing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SomaGraph:
    """Neuron × neuron graph built from synapse (pre_root, post_root) pairs.

    Attributes
    ----------
    node_ids : np.ndarray
        Sorted unique root IDs; shape [N].
    node_features : np.ndarray
        Per-neuron features; shape [N, feat_dim].
    src : np.ndarray
        Source node indices for each directed edge; shape [E].
    dst : np.ndarray
        Destination node indices; shape [E].
    edge_synapse_count : np.ndarray, optional
        Number of synapses per (pre, post) pair; shape [E].
    """

    node_ids: np.ndarray
    node_features: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    edge_synapse_count: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_edges(self) -> int:
        return len(self.src)


def build_soma_graph_from_synapses(
    pre_root_ids: np.ndarray,
    post_root_ids: np.ndarray,
    *,
    node_feat_dim: int = 32,
    feature_seed: int | None = 42,
) -> SomaGraph:
    """Build a soma graph from (pre_root_id, post_root_id) synapse arrays.

    Filters out root_id == 0 (invalid). Aggregates multiple synapses between
    the same (pre, post) into a single directed edge. Node features are
    simple placeholders (random or zeros) for now — in production these would
    be pooled synapse/fragment embeddings from the grammar.

    Parameters
    ----------
    pre_root_ids, post_root_ids : np.ndarray
        Int64 arrays of shape [n_synapses].
    node_feat_dim : int
        Dimensionality of node features (must match GAT node_dim).
    feature_seed : int or None
        If set, use seeded random features; else zeros.

    Returns
    -------
    SomaGraph
    """
    pre = np.asarray(pre_root_ids, dtype=np.int64).ravel()
    post = np.asarray(post_root_ids, dtype=np.int64).ravel()
    assert len(pre) == len(post), "pre and post must have same length"

    # Drop invalid roots
    valid = (pre > 0) & (post > 0) & (pre != post)
    pre = pre[valid]
    post = post[valid]

    all_roots = np.unique(np.concatenate([pre, post]))
    id_to_idx = {int(r): i for i, r in enumerate(all_roots)}

    # Aggregate edges: (pre_idx, post_idx) -> count
    edge_counts: dict[tuple[int, int], int] = {}
    for p, q in zip(pre, post):
        i, j = id_to_idx[int(p)], id_to_idx[int(q)]
        key = (i, j)
        edge_counts[key] = edge_counts.get(key, 0) + 1

    src_list = [k[0] for k in edge_counts]
    dst_list = [k[1] for k in edge_counts]
    counts = [edge_counts[(i, j)] for i, j in zip(src_list, dst_list)]

    # Node features: placeholder
    n = len(all_roots)
    if feature_seed is not None:
        rng = np.random.default_rng(feature_seed)
        node_features = rng.standard_normal((n, node_feat_dim)).astype(np.float32)
    else:
        node_features = np.zeros((n, node_feat_dim), dtype=np.float32)

    return SomaGraph(
        node_ids=all_roots,
        node_features=node_features,
        src=np.array(src_list, dtype=np.int64),
        dst=np.array(dst_list, dtype=np.int64),
        edge_synapse_count=np.array(counts, dtype=np.int64),
    )
