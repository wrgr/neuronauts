"""Lightweight Dijkstra-style bridge search over neurite fragments.

This module provides a small, dependency-minimal graph helper that can be used
by the scaffolded grammar stack to propose candidate bridges across voids or
under-segmented regions. It intentionally stays numpy-only and does not depend
on the training stack so it can be imported in evaluation and tooling code
without pulling in ``torch``.

The core abstraction is ``BridgeGraph``, a thin wrapper around an adjacency
list with non-negative edge weights. The API is deliberately general: callers
are responsible for mapping fragment endpoints, skeleton nodes, or segment IDs
onto integer node indices and for choosing an appropriate cost metric
(e.g. geometric distance, curvature deviation, or learned merge penalty).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import heapq
import math


NodeId = int


@dataclass(frozen=True)
class BridgePath:
    """A candidate bridge between two nodes.

    Attributes
    ----------
    cost:
        Total additive cost along the path.
    nodes:
        Sequence of node ids including both endpoints.
    """

    cost: float
    nodes: Tuple[NodeId, ...]


class BridgeGraph:
    """Minimal undirected weighted graph for bridge search.

    The representation is a simple adjacency list mapping each node id to a list
    of (neighbor_id, weight) pairs. All weights must be finite and
    non-negative so that Dijkstra's algorithm is well-defined.
    """

    def __init__(self) -> None:
        self._adj: Dict[NodeId, List[Tuple[NodeId, float]]] = {}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def add_edge(self, u: NodeId, v: NodeId, weight: float) -> None:
        """Add an undirected edge between ``u`` and ``v`` with the given cost."""
        if weight < 0 or not math.isfinite(weight):
            raise ValueError(f"edge weight must be finite and non-negative, got {weight!r}")
        if u == v:
            return
        self._adj.setdefault(u, []).append((v, float(weight)))
        self._adj.setdefault(v, []).append((u, float(weight)))

    def add_path(self, nodes: Sequence[NodeId], weight: float) -> None:
        """Add a chain of edges along ``nodes`` with shared ``weight`` per hop.

        This is a convenience for wiring up a polyline skeleton or mesh trace.
        """
        if len(nodes) < 2:
            return
        for a, b in zip(nodes[:-1], nodes[1:]):
            self.add_edge(int(a), int(b), float(weight))

    # ------------------------------------------------------------------
    # Dijkstra search
    # ------------------------------------------------------------------
    def dijkstra(
        self,
        sources: Iterable[NodeId],
        targets: Optional[Iterable[NodeId]] = None,
        max_cost: Optional[float] = None,
    ) -> Mapping[NodeId, BridgePath]:
        """Run multi-source Dijkstra to all nodes (or until hitting ``targets``).

        Parameters
        ----------
        sources:
            One or more starting node ids.
        targets:
            Optional collection of target node ids. If provided, the search
            terminates once all reachable targets have been settled.
        max_cost:
            Optional ceiling on path cost. Nodes whose cheapest path would
            exceed this threshold are ignored.

        Returns
        -------
        Mapping[NodeId, BridgePath]
            Best path discovered for each reachable node, keyed by the node id.
            Sources will have a zero-cost singleton path.
        """
        # Normalize inputs.
        src_list = [int(s) for s in sources]
        if not src_list:
            return {}

        if targets is not None:
            target_set = {int(t) for t in targets}
        else:
            target_set = None

        # Standard Dijkstra with path reconstruction.
        dist: Dict[NodeId, float] = {}
        parent: Dict[NodeId, Optional[NodeId]] = {}
        heap: List[Tuple[float, NodeId]] = []

        for s in src_list:
            if s not in self._adj:
                continue
            dist[s] = 0.0
            parent[s] = None
            heapq.heappush(heap, (0.0, s))

        settled_targets = 0
        total_targets = len(target_set) if target_set is not None else 0

        while heap:
            cost, u = heapq.heappop(heap)
            if u in dist and cost > dist[u] + 1e-9:
                continue
            if max_cost is not None and cost > max_cost:
                break
            if target_set is not None and u in target_set:
                settled_targets += 1
                if settled_targets >= total_targets:
                    break
            for v, w in self._adj.get(u, ()):
                new_cost = cost + w
                if max_cost is not None and new_cost > max_cost:
                    continue
                if v not in dist or new_cost < dist[v] - 1e-9:
                    dist[v] = new_cost
                    parent[v] = u
                    heapq.heappush(heap, (new_cost, v))

        def _reconstruct(node: NodeId) -> BridgePath:
            path_nodes: List[NodeId] = []
            cur: Optional[NodeId] = node
            while cur is not None:
                path_nodes.append(cur)
                cur = parent.get(cur)
            path_nodes.reverse()
            return BridgePath(cost=dist[node], nodes=tuple(path_nodes))

        return {node: _reconstruct(node) for node in dist.keys()}

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def best_bridge(
        self,
        sources: Iterable[NodeId],
        targets: Iterable[NodeId],
        max_cost: Optional[float] = None,
    ) -> Optional[BridgePath]:
        """Return the cheapest bridge path connecting any source to any target.

        This is a small wrapper around ``dijkstra`` that only materializes paths
        for nodes that are themselves in ``targets``.
        """
        target_set = {int(t) for t in targets}
        if not target_set:
            return None
        all_paths = self.dijkstra(sources=sources, targets=target_set, max_cost=max_cost)
        candidates = [p for node, p in all_paths.items() if node in target_set]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p.cost)


__all__ = ["NodeId", "BridgePath", "BridgeGraph"]

