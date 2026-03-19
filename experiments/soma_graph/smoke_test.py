"""Smoke test: build a synthetic soma graph and run GAT forward pass.

Run with:
    python experiments/soma_graph/smoke_test.py

Or via pytest:
    pytest tests/test_soma_graph_experiment.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing from project root
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from experiments.soma_graph.build_graph import build_soma_graph_from_synapses

try:
    import torch  # noqa: F401
    from neuronauts.shared_grammar_model import GlobalAssemblyGAT

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    GlobalAssemblyGAT = None  # type: ignore[misc, assignment]


def _add_self_loops_and_bidirectional(
    src: np.ndarray, dst: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Add self-loops and reverse edges for GAT (matches assembly._build_gat_edges)."""
    edge_set: set[tuple[int, int]] = set()
    for i in range(n_nodes):
        edge_set.add((i, i))
    for i, j in zip(src, dst):
        edge_set.add((int(i), int(j)))
        edge_set.add((int(j), int(i)))
    pairs = sorted(edge_set)
    return (
        np.array([p[0] for p in pairs], dtype=np.int64),
        np.array([p[1] for p in pairs], dtype=np.int64),
    )


def run_smoke_test() -> bool:
    """Build synthetic soma graph, run GAT forward + edge scoring. Returns True on success."""
    # Synthetic synapses: 5 neurons, ~20 edges
    # 0->1, 0->2, 1->2, 1->3, 2->3, 3->4, 4->0 (small cycle)
    pre_root = np.array([100, 100, 100, 200, 200, 300, 400, 500], dtype=np.int64)
    post_root = np.array([200, 300, 200, 300, 400, 400, 500, 100], dtype=np.int64)

    graph = build_soma_graph_from_synapses(
        pre_root, post_root, node_feat_dim=32, feature_seed=42
    )

    assert graph.n_nodes == 5
    assert graph.n_edges >= 4  # aggregated, so some duplicates collapsed
    assert graph.node_features.shape == (5, 32)

    # GAT expects self-loops + bidirectional for message passing
    src_gat, dst_gat = _add_self_loops_and_bidirectional(
        graph.src, graph.dst, graph.n_nodes
    )

    if not _HAS_TORCH:
        print("Skip: torch not installed (pip install -e '.[topology]')")
        return True  # build_graph part passed

    import torch  # used for from_numpy, sigmoid (already imported if _HAS_TORCH)

    gat = GlobalAssemblyGAT(node_dim=32, gat_dim=64, n_heads=4, n_layers=2)
    x = torch.from_numpy(graph.node_features)
    src_t = torch.from_numpy(src_gat)
    dst_t = torch.from_numpy(dst_gat)

    # Forward pass
    h = gat(x, src_t, dst_t)
    assert h.shape == (5, 64)

    # Score a subset of edges (original directed synapse edges)
    syn_src = torch.from_numpy(graph.src)
    syn_dst = torch.from_numpy(graph.dst)
    logits = gat.score_edges(h, syn_src, syn_dst)
    assert logits.shape == (graph.n_edges,)

    probs = torch.sigmoid(logits)
    assert (probs >= 0).all() and (probs <= 1).all()

    return True


def main() -> int:
    print("Soma graph smoke test …")
    ok = run_smoke_test()
    if ok:
        print("OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
