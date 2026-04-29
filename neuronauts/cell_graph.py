"""Synapse-level cell graph for global topological merge.

**Core idea — reachability not pairwise decisions**

The existing pipeline makes a sequence of local, pairwise merge decisions
(grammar score → beam search) and only then assembles full cells.  The problem
is that each decision is made without seeing the rest of the cell.

This module inverts the framing:

1. Build a *synapse-level evidence graph* where nodes are synapses and edges
   carry all available noisy evidence: scaffold group membership, spatial
   proximity, pairwise grammar scores, shared agent visits.
2. Run a learned GNN with K message-passing rounds.  After round K, each
   synapse's embedding contains information from its K-hop neighbours — this
   is a direct implementation of the "reachability" argument: two synapses
   that are K-hop connected through high-confidence evidence will have similar
   embeddings.
3. Cluster embeddings → full cell assignments.  The pairwise grammar scores
   become *edge features* (not decisions); the GNN learns which evidence paths
   are trustworthy in context.

Empirical K-hop behaviour
-------------------------
Earlier docstrings claimed a K=1/2/3+ hierarchy
(scaffold-group → branch → arbor).  The K-hop ablation
(``models/cell_gnn_seg_K{1..5}.pt``, all seed=42) does not
support that story:

  K=1: 0.176   K=2: 0.197   K=3: 0.194   K=4: 0.195   K=5: 0.185

K=1 underfits; K=2..4 are flat within ±0.003.  K=2 is the
recommended default for the current dataset; the on-disk default
of ``CellGNNConfig.n_layers=3`` is preserved for backwards
compatibility but produces no measurable F1 gain over K=2.

Architecture
------------
1. ``build_synapse_graph``           -- weighted evidence graph from all available signals
2. ``CellGNN``                       -- edge-conditioned message-passing GNN
                                        (optionally consumes per-edge skeleton
                                        path embeddings via ``PathEdgeEncoder``)
3. ``partition_from_embeddings``     -- cluster embeddings → cell labels
4. ``cell_graph_train_step``         -- contrastive pull/push against CAVE root IDs
5. ``train_cell_gnn``                -- epoch loop over a BoxCache
6. ``infer_cells``                   -- inference → per-synapse cell labels
7. ``connectivity_graph_from_cell_labels`` -- labels → ConnectivityGraph for F1 eval

Pre-computed edge-feature caches
--------------------------------
* ``precompute_seg_scores_fast``      -- BossDB seg ID match per edge
                                        (one bbox fetch per box)
* ``precompute_self_skeletons_for_cache``
                                      -- kimimaro self-skeletonization of the
                                        BossDB seg volume per box
* ``precompute_skeleton_paths_for_cache``
                                      -- Dijkstra paths through self- or CAVE-
                                        skeletons; feeds ``PathEdgeEncoder``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dataclasses import dataclass as _dataclass

from .grammar import PATH_ISO, DEFAULT_PATH_FEATURE_MODE, featurize_path_points, _require_torch

if TYPE_CHECKING:
    from .fetch import SynapseTable
    from .merge import ConnectivityGraph


# ---------------------------------------------------------------------------
# 1. Synapse evidence graph construction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SynapseEdge:
    """A weighted edge between two synapses carrying evidence of same-cell membership."""
    src: int          # synapse index
    dst: int          # synapse index
    distance: float   # spatial distance (isotropic nm)
    same_scaffold: float  # 1.0 if same scaffold group, 0.0 otherwise
    grammar_score: float  # pairwise grammar merge score (if available), else 0.0
    shared_agents: int    # number of agents that visited both synapses
    shared_partners: int = 0  # shared connectivity targets on the opposite side
    seg_connectivity: float = 0.5  # EM seg corridor score: 1.0=same neurite, 0.0=different, 0.5=unknown


@dataclass
class SynapseGraph:
    """A graph over synapses with evidence edges and node features.

    This is the input to the CellGNN.  Each node is a synapse; each edge
    carries evidence about whether the two synapses belong to the same cell.

    Attributes
    ----------
    n_synapses : int
        Number of synapse nodes.
    role : str
        Which side this graph represents ("pre" or "post").
    node_positions : ndarray [N, 3]
        Isotropic synapse positions.
    node_scaffold_ids : ndarray [N]
        Scaffold group ID per synapse (from CAVE seg IDs).  -1 if unknown.
    edges : list[SynapseEdge]
        Evidence edges.
    root_ids : ndarray [N] or None
        Ground-truth root IDs (for training).  None at inference.
    """
    n_synapses: int
    role: str
    node_positions: np.ndarray
    node_scaffold_ids: np.ndarray
    edges: list[SynapseEdge]
    root_ids: np.ndarray | None = None
    # Optional: ``edge_path_features[(i, j)]`` is a ``[T, path_input_dim]``
    # float32 array of per-step skeleton-path features; absent or empty
    # means "no path".  Populated by callers that have run the
    # skeleton-path precompute (see ``precompute_skeleton_paths_for_cache``).
    edge_path_features: dict[tuple[int, int], np.ndarray] | None = None
    # Optional boolean mask: True = core node (used in loss), False = halo
    # node included only as message-passing context.  None means all core.
    core_mask: np.ndarray | None = None


def build_synapse_graph(
    synapses: "SynapseTable",
    role: str,
    *,
    synapse_hits: np.ndarray | None = None,
    scaffold_groups: dict[int, list[int]] | None = None,
    grammar_scores: dict[tuple[int, int], float] | None = None,
    proximity_radius_nm: float = 5000.0,
    max_edges_per_node: int = 32,
    partner_seg_ids: np.ndarray | None = None,
    seg_connectivity_scores: dict[tuple[int, int], float] | None = None,
) -> SynapseGraph:
    """Build a synapse-level evidence graph for one side (pre or post).

    Parameters
    ----------
    synapses :
        SynapseTable from fetch.py.
    role :
        "pre" or "post" -- which side of the synapse to group.
    synapse_hits :
        Bool array [n_agents, n_synapses] from agent simulation.  Used to
        compute shared-agent edges.  Optional.
    scaffold_groups :
        Mapping from scaffold group ID to list of synapse indices.  Synapses
        in the same group get same_scaffold=1.0 edges.  Optional.
    grammar_scores :
        Pairwise grammar merge scores keyed by (syn_i, syn_j) with i < j.
        Optional.
    proximity_radius_nm :
        Maximum distance (in isotropic nm) for creating proximity edges.
    max_edges_per_node :
        Cap on edges per synapse to keep the graph sparse.
    partner_seg_ids :
        Seg IDs from the *opposite* side (post_seg_id when role="pre", or
        pre_seg_id when role="post").  Used to compute shared-partner counts:
        how many connectivity targets two synapse groups share, which is a
        strong signal that they belong to the same cell.
    seg_connectivity_scores :
        EM segmentation corridor scores keyed by ``(min(i,j), max(i,j))``.
        Produced by :func:`~neuronauts.em_corridor.batch_score_seg_connectivity`.
        Values: 1.0 = confirmed same neurite, 0.0 = confirmed different neurites,
        0.5 = unknown.  If None, all edges default to 0.5 (neutral).
    """
    if role == "pre":
        positions = synapses.pre_pt.copy().astype(np.float32)
        root_ids = synapses.pre_root_id.copy()
        seg_ids = getattr(synapses, "pre_seg_id", None)
    else:
        positions = synapses.post_pt.copy().astype(np.float32)
        root_ids = synapses.post_root_id.copy()
        seg_ids = getattr(synapses, "post_seg_id", None)

    n = len(positions)
    # Apply isotropic scaling
    iso_positions = positions * PATH_ISO[np.newaxis, :]

    # Build scaffold ID per synapse
    node_scaffold = np.full(n, -1, dtype=np.int64)
    if seg_ids is not None:
        node_scaffold = seg_ids.copy()
    elif scaffold_groups is not None:
        for gid, syn_indices in scaffold_groups.items():
            for si in syn_indices:
                if 0 <= si < n:
                    node_scaffold[si] = gid

    # Build scaffold membership lookup
    scaffold_membership: dict[int, set[int]] = {}
    if seg_ids is not None:
        for i, sid in enumerate(seg_ids):
            sid_int = int(sid)
            if sid_int > 0:
                scaffold_membership.setdefault(sid_int, set()).add(i)
    elif scaffold_groups is not None:
        for gid, syn_indices in scaffold_groups.items():
            scaffold_membership[gid] = set(syn_indices)

    # Shared-agent lookup
    agent_covisit: dict[tuple[int, int], int] = {}
    if synapse_hits is not None:
        # For each pair of synapses, count agents that visited both
        # Efficient: iterate agents, record which synapses each visited
        for agent_idx in range(synapse_hits.shape[0]):
            visited = np.flatnonzero(synapse_hits[agent_idx])
            if len(visited) < 2:
                continue
            # Only consider up to 20 visited synapses per agent to avoid O(n^2)
            if len(visited) > 20:
                visited = visited[:20]
            for i in range(len(visited)):
                for j in range(i + 1, len(visited)):
                    key = (int(min(visited[i], visited[j])),
                           int(max(visited[i], visited[j])))
                    agent_covisit[key] = agent_covisit.get(key, 0) + 1

    # Shared-partner lookup: per scaffold group → set of partner seg_ids
    # Two synapses from the same pre-cell should connect to overlapping post targets.
    synapse_to_group: dict[int, int] = {}
    group_partners: dict[int, set[int]] = {}
    if partner_seg_ids is not None and len(partner_seg_ids) == n:
        # Build group membership for the current side
        if seg_ids is not None:
            for i, sid in enumerate(seg_ids):
                synapse_to_group[i] = int(sid)
        elif scaffold_groups is not None:
            for gid, syn_indices in scaffold_groups.items():
                for si in syn_indices:
                    synapse_to_group[si] = gid
        else:
            # No grouping info — treat each synapse as its own group
            for i in range(n):
                synapse_to_group[i] = i

        for i in range(n):
            gid = synapse_to_group.get(i, i)
            pid = int(partner_seg_ids[i])
            if pid > 0:
                group_partners.setdefault(gid, set()).add(pid)

    # Build edges via spatial proximity (KD-tree)
    from ._scipy_compat import cKDTree

    tree = cKDTree(iso_positions)
    edge_dict: dict[tuple[int, int], SynapseEdge] = {}

    def _shared_partner_count(a: int, b: int) -> int:
        if not group_partners:
            return 0
        ga = synapse_to_group.get(a, a)
        gb = synapse_to_group.get(b, b)
        pa = group_partners.get(ga, set())
        pb = group_partners.get(gb, set())
        return len(pa & pb)

    # Proximity edges — use K-NN (k=max_edges_per_node+1) to keep edge count
    # O(N·K) regardless of density.  query_pairs with a fixed radius explodes
    # to O(N²) in dense 30µm boxes (8K+ synapses, 160+ neighbours per node).
    k = min(max_edges_per_node + 1, len(iso_positions))
    _, nn_indices = tree.query(iso_positions, k=k, workers=-1)
    pair_set: set[tuple[int, int]] = set()
    for i, neighbours in enumerate(nn_indices):
        for j in neighbours[1:]:  # skip self (index 0)
            d = float(np.linalg.norm(iso_positions[i] - iso_positions[j]))
            if d <= proximity_radius_nm:
                pair_set.add((min(i, j), max(i, j)))
    if pair_set:
        for a, b in pair_set:
            a, b = int(a), int(b)
            key = (min(a, b), max(a, b))
            dist = float(np.linalg.norm(iso_positions[a] - iso_positions[b]))
            same_scaf = 1.0 if (node_scaffold[a] == node_scaffold[b]
                                and node_scaffold[a] > 0) else 0.0
            gs = 0.0
            if grammar_scores is not None:
                gs = grammar_scores.get(key, 0.0)
            sa = agent_covisit.get(key, 0)
            sp = _shared_partner_count(key[0], key[1])
            sc = seg_connectivity_scores.get(key, 0.5) if seg_connectivity_scores else 0.5
            edge_dict[key] = SynapseEdge(
                src=key[0], dst=key[1],
                distance=dist, same_scaffold=same_scaf,
                grammar_score=gs, shared_agents=sa,
                shared_partners=sp,
                seg_connectivity=sc,
            )

    # Scaffold edges (ensure all same-scaffold synapses are connected)
    for gid, members in scaffold_membership.items():
        member_list = sorted(members)
        if len(member_list) > max_edges_per_node:
            member_list = member_list[:max_edges_per_node]
        for i in range(len(member_list)):
            for j in range(i + 1, len(member_list)):
                key = (member_list[i], member_list[j])
                if key not in edge_dict:
                    dist = float(np.linalg.norm(
                        iso_positions[key[0]] - iso_positions[key[1]]))
                    gs = grammar_scores.get(key, 0.0) if grammar_scores else 0.0
                    sa = agent_covisit.get(key, 0)
                    sp = _shared_partner_count(key[0], key[1])
                    sc = seg_connectivity_scores.get(key, 0.5) if seg_connectivity_scores else 0.5
                    edge_dict[key] = SynapseEdge(
                        src=key[0], dst=key[1],
                        distance=dist, same_scaffold=1.0,
                        grammar_score=gs, shared_agents=sa,
                        shared_partners=sp,
                        seg_connectivity=sc,
                    )

    return SynapseGraph(
        n_synapses=n,
        role=role,
        node_positions=iso_positions,
        node_scaffold_ids=node_scaffold,
        edges=list(edge_dict.values()),
        root_ids=root_ids,
    )


def subdivide_synapse_graph(
    graph: SynapseGraph,
    n_nodes: int = 5000,
    n_subgraphs: int | None = None,
    *,
    halo_hops: int = 0,
    rng: "np.random.Generator | None" = None,
    min_nodes: int = 50,
) -> "list[SynapseGraph]":
    """Extract connected sub-graphs from a large SynapseGraph via BFS sampling.

    Instead of randomly dropping nodes (which destroys chain paths and spatial
    structure), this grows a BFS neighbourhood from a random seed node and
    returns the induced sub-graph.  A 50µm box with 40K synapses yields ~20
    non-overlapping sub-graphs of 2K nodes, each preserving local topology.

    Parameters
    ----------
    graph : SynapseGraph
        Full graph (may be very large).
    n_nodes : int
        Target number of *core* nodes per sub-graph.  Core nodes are used in
        the contrastive loss; they have at least ``halo_hops`` hops of context.
    n_subgraphs : int or None
        How many sub-graphs to extract.  Defaults to
        ``max(1, graph.n_synapses // n_nodes)``.
    halo_hops : int
        After collecting ``n_nodes`` core nodes, expand BFS by this many
        additional hops.  The extra "halo" nodes are included in the graph for
        message-passing context but are masked out of the contrastive loss via
        ``SynapseGraph.core_mask``.  Use ``halo_hops = n_layers`` so every
        core node has a full receptive field.
    rng : np.random.Generator or None
        RNG for seed selection.
    min_nodes : int
        Discard sub-graphs with fewer than this many *core* nodes.

    Returns
    -------
    list of SynapseGraph, each a connected induced sub-graph with
    ``core_mask`` set when ``halo_hops > 0``.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = graph.n_synapses
    if N == 0:
        return []

    if n_subgraphs is None:
        n_subgraphs = max(1, N // n_nodes)

    # Build adjacency list once
    adj: list[list[int]] = [[] for _ in range(N)]
    for e in graph.edges:
        adj[e.src].append(e.dst)
        adj[e.dst].append(e.src)

    results: list[SynapseGraph] = []
    used = np.zeros(N, dtype=bool)

    for _ in range(n_subgraphs):
        # Pick a random unused seed
        available = np.flatnonzero(~used)
        if len(available) == 0:
            break
        seed = int(rng.choice(available))

        # Phase 1: BFS up to n_nodes core nodes
        core: list[int] = []
        in_visited = np.zeros(N, dtype=bool)
        queue = [seed]
        in_visited[seed] = True
        head = 0
        while head < len(queue) and len(core) < n_nodes:
            node = queue[head]; head += 1
            core.append(node)
            order = rng.permutation(len(adj[node]))
            for idx in order:
                nb = adj[node][idx]
                if not in_visited[nb] and len(core) + (len(queue) - head) < n_nodes:
                    in_visited[nb] = True
                    queue.append(nb)

        if len(core) < min_nodes:
            continue

        # Mark core nodes as used for future sub-graphs
        if N > n_nodes:
            for v in core:
                used[v] = True

        # Phase 2: expand halo_hops steps from the core frontier.
        # Cap at n_nodes // 4 to prevent dense graphs from ballooning the halo
        # to full-box size (which would defeat the purpose of subdivision).
        halo: list[int] = []
        if halo_hops > 0:
            max_halo = n_nodes // 4
            frontier = set(core)
            halo_set: set[int] = set()
            for _ in range(halo_hops):
                if len(halo_set) >= max_halo:
                    break
                next_frontier: set[int] = set()
                for v in frontier:
                    for nb in adj[v]:
                        if nb not in in_visited:
                            in_visited[nb] = True
                            halo_set.add(nb)
                            next_frontier.add(nb)
                            if len(halo_set) >= max_halo:
                                break
                    if len(halo_set) >= max_halo:
                        break
                frontier = next_frontier
                if not frontier:
                    break
            halo = list(halo_set)

        all_nodes = core + halo
        old_to_new = {v: i for i, v in enumerate(all_nodes)}
        node_set = set(all_nodes)

        # Induced sub-graph over core + halo
        sub_edges = []
        sub_edge_path: dict[tuple[int, int], np.ndarray] = {}
        for e in graph.edges:
            if e.src in node_set and e.dst in node_set:
                new_src = old_to_new[e.src]
                new_dst = old_to_new[e.dst]
                sub_edges.append(SynapseEdge(
                    src=new_src, dst=new_dst,
                    distance=e.distance,
                    same_scaffold=e.same_scaffold,
                    grammar_score=e.grammar_score,
                    shared_agents=e.shared_agents,
                    shared_partners=e.shared_partners,
                    seg_connectivity=e.seg_connectivity,
                ))
                if graph.edge_path_features:
                    old_key = (min(e.src, e.dst), max(e.src, e.dst))
                    new_key = (min(new_src, new_dst), max(new_src, new_dst))
                    arr = graph.edge_path_features.get(old_key)
                    if arr is not None:
                        sub_edge_path[new_key] = arr

        idx = np.array(all_nodes, dtype=np.int64)
        n_total = len(all_nodes)

        # core_mask: True for core nodes, False for halo context nodes
        core_mask: np.ndarray | None = None
        if halo:
            core_mask = np.zeros(n_total, dtype=bool)
            core_mask[:len(core)] = True

        results.append(SynapseGraph(
            n_synapses=n_total,
            role=graph.role,
            node_positions=graph.node_positions[idx],
            node_scaffold_ids=graph.node_scaffold_ids[idx],
            edges=sub_edges,
            root_ids=graph.root_ids[idx] if graph.root_ids is not None else None,
            edge_path_features=sub_edge_path if sub_edge_path else None,
            core_mask=core_mask,
        ))

    return results


def build_synapse_chain_paths(
    graph: SynapseGraph,
    labels: "np.ndarray | None" = None,
    *,
    mode: str = DEFAULT_PATH_FEATURE_MODE,
) -> "dict[tuple[int, int], np.ndarray]":
    """Build skeleton-like path features from synapse position chains.

    Groups synapses by cell label (uses ``graph.root_ids`` when ``labels``
    is None), sorts each group's positions along its principal axis, and
    traces nearest-neighbour chains through the sorted positions.  For every
    within-cell proximity-graph edge (i, j) the path is the chain segment
    from node i to node j through their shared group.

    This is the **synapse-link skeleton** strategy: no segmentation is needed,
    only synapse positions.  During training supply ``labels=None`` to use
    ground-truth root IDs.  At inference, pass the output of a first-pass
    ``infer_cells`` call.

    Parameters
    ----------
    graph : SynapseGraph
        Must have ``node_positions`` populated.
    labels : int64 ndarray [N] or None
        Cell assignment per synapse.  ``None`` falls back to ``graph.root_ids``
        (training mode).  Root IDs of 0 are treated as unknown and excluded.
    mode : str
        Feature mode forwarded to ``featurize_path_points``.

    Returns
    -------
    dict mapping ``(i, j)`` (i < j) → float32 ``[T, D]`` feature array.
    """
    if labels is None:
        if graph.root_ids is None:
            return {}
        labels = graph.root_ids.astype(np.int64)

    pos = graph.node_positions  # [N, 3] isotropic nm

    # Group synapse indices by cell label (skip unknown/0)
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, lbl in enumerate(labels):
        if int(lbl) != 0:
            groups[int(lbl)].append(idx)

    # Build per-group chains and a global node→(chain_pos, ordered_pts) lookup.
    # Then iterate edges ONCE (O(E)) instead of O(groups × E).
    node_chain_pos: dict[int, int] = {}       # node index → position in its group chain
    node_chain_pts: dict[int, np.ndarray] = {}  # node index → ordered_pts array for group

    for lbl, members in groups.items():
        if len(members) < 2:
            continue
        pts = pos[members]  # [M, 3]
        M = len(pts)

        # Sort along the principal component to get a consistent chain order
        centred = pts - pts.mean(axis=0)
        if M >= 2:
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
            proj = centred @ Vt[0]
            order = np.argsort(proj)
        else:
            order = np.arange(M)
        ordered_members = [members[o] for o in order]
        ordered_pts = pts[order]  # [M, 3]

        for k, m in enumerate(ordered_members):
            node_chain_pos[m] = k
            node_chain_pts[m] = ordered_pts

    # Single O(E) pass over edges
    result: dict[tuple[int, int], np.ndarray] = {}
    for e in graph.edges:
        i, j = e.src, e.dst
        if int(labels[i]) == 0 or int(labels[i]) != int(labels[j]):
            continue
        key = (min(i, j), max(i, j))
        if key in result:
            continue
        ki = node_chain_pos.get(i)
        kj = node_chain_pos.get(j)
        if ki is None or kj is None:
            continue
        ordered_pts = node_chain_pts[i]  # same array for both (same group)
        a, b = (ki, kj) if ki <= kj else (kj, ki)
        segment = ordered_pts[a : b + 1]
        if len(segment) < 2:
            continue
        arr = featurize_path_points(segment, mode=mode)
        if arr.shape[0] > 0:
            result[key] = arr

    return result


# ---------------------------------------------------------------------------
# 2. CellGNN -- sparse message-passing for cell membership
# ---------------------------------------------------------------------------

_EDGE_FEAT_DIM = 6  # distance, same_scaffold, grammar_score, shared_agents, shared_partners, seg_connectivity
EDGE_FEATURE_NAMES = (
    "distance", "same_scaffold", "grammar_score",
    "shared_agents", "shared_partners", "seg_connectivity",
)
# Ablation hook: when set to an int in [0, _EDGE_FEAT_DIM), _graph_to_tensors
# zeros out that column in the edge feature matrix. Used by the per-feature
# ablation in scripts/train.py.
_ABLATE_FEATURE_IDX: int | None = None
# Below this synapse count use exact O(N²) similarity; above use ANN sparse path.
_ANN_PARTITION_THRESHOLD = 500


def _graph_to_tensors(
    graph: SynapseGraph,
    *,
    return_paths: bool = False,
    path_max_len: int = 64,
    path_feat_dim: int = 6,
):
    """Convert a SynapseGraph to tensors for the GNN.

    Returns
    -------
    node_feat : Tensor [N, 3]  (isotropic position, centered)
    edge_src, edge_dst : Tensor [2E]  (bidirectional)
    edge_feat : Tensor [2E, 6]  (distance, same_scaffold, grammar_score,
                                  shared_agents, shared_partners,
                                  seg_connectivity)

    When ``return_paths`` is True, also returns
    ``(path_seq, path_mask, has_path)``:

    path_seq : Tensor [2E, T_max, path_feat_dim] float32
    path_mask : Tensor [2E, T_max] bool   (True = padded position)
    has_path : Tensor [2E] bool           (True if real path)

    Path features come from ``graph.edge_path_features`` (a
    ``dict[(i, j)] -> [T, path_feat_dim]`` array).  Edges without an
    entry get an all-padding row and ``has_path=False``.  Self-loops
    always get ``has_path=False``.
    """
    torch, _ = _require_torch()
    N = graph.n_synapses

    # Node features: centered isotropic positions
    pos = graph.node_positions.copy()
    pos -= pos.mean(axis=0, keepdims=True)
    # Normalize to unit scale
    scale = max(np.abs(pos).max(), 1e-6)
    pos /= scale
    node_feat = torch.from_numpy(pos).float()

    if not graph.edges:
        empty_edges_feat = torch.zeros(0, _EDGE_FEAT_DIM, dtype=torch.float32)
        if return_paths:
            return (
                node_feat,
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                empty_edges_feat,
                torch.zeros(0, path_max_len, path_feat_dim, dtype=torch.float32),
                torch.ones(0, path_max_len, dtype=torch.bool),
                torch.zeros(0, dtype=torch.bool),
            )
        return (
            node_feat,
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            empty_edges_feat,
        )

    # Build bidirectional edges
    src_list, dst_list, feat_list = [], [], []
    # Per-bidirectional-edge path lookup keys (canonical (min, max))
    path_keys: list[tuple[int, int] | None] = []
    for e in graph.edges:
        canonical_key = (min(e.src, e.dst), max(e.src, e.dst))
        for s, d in [(e.src, e.dst), (e.dst, e.src)]:
            src_list.append(s)
            dst_list.append(d)
            feat_list.append([
                e.distance / max(scale, 1e-6),  # normalized distance
                e.same_scaffold,
                e.grammar_score,
                min(e.shared_agents, 10) / 10.0,   # clamp & normalize
                min(e.shared_partners, 5) / 5.0,   # clamp & normalize
                e.seg_connectivity,                 # EM seg endpoint ID match
            ])
            path_keys.append(canonical_key)

    # Self-loops: seg_connectivity=0.5 (neutral; self-edges don't resolve connectivity)
    for i in range(N):
        src_list.append(i)
        dst_list.append(i)
        feat_list.append([0.0, 1.0, 1.0, 0.0, 0.0, 0.5])
        path_keys.append(None)  # self-loops never have a path

    edge_src = torch.tensor(src_list, dtype=torch.long)
    edge_dst = torch.tensor(dst_list, dtype=torch.long)
    edge_feat = torch.tensor(feat_list, dtype=torch.float32)

    if _ABLATE_FEATURE_IDX is not None and 0 <= _ABLATE_FEATURE_IDX < _EDGE_FEAT_DIM:
        edge_feat[:, _ABLATE_FEATURE_IDX] = 0.0

    if not return_paths:
        return node_feat, edge_src, edge_dst, edge_feat

    # Build padded path tensors aligned with the bidirectional edge order.
    E_total = len(src_list)
    path_seq = np.zeros((E_total, path_max_len, path_feat_dim), dtype=np.float32)
    path_mask = np.ones((E_total, path_max_len), dtype=bool)
    has_path = np.zeros((E_total,), dtype=bool)

    feat_lookup = graph.edge_path_features or {}
    if feat_lookup:
        for k, key in enumerate(path_keys):
            if key is None:
                continue
            arr = feat_lookup.get(key)
            if arr is None or len(arr) == 0:
                continue
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != path_feat_dim:
                continue
            T = min(arr.shape[0], path_max_len)
            path_seq[k, :T, :] = arr[:T]
            path_mask[k, :T] = False
            has_path[k] = True

    return (
        node_feat,
        edge_src,
        edge_dst,
        edge_feat,
        torch.from_numpy(path_seq),
        torch.from_numpy(path_mask),
        torch.from_numpy(has_path),
    )


class CellGNN:
    """Factory for a synapse-level GNN that learns cell membership embeddings.

    Architecture
    ------------
    1. Node input projection: synapse position -> d_model
    2. Edge feature projection: evidence features -> d_model
    3. K message-passing rounds with edge-conditioned attention:
       - Each round: node attends to neighbours, weighted by learned
         edge-evidence compatibility.  Residual + LayerNorm.
       - After K rounds, each node's embedding reflects K-hop reachability.
    4. Output projection -> embedding_dim

    The embedding space is trained so that same-cell synapses cluster together
    (contrastive loss against ground-truth root IDs).

    At inference, embeddings are clustered (e.g., by cosine similarity
    thresholding or agglomerative clustering) to produce cell assignments.
    """

    def __new__(
        cls,
        *,
        node_input_dim: int = 3,
        edge_input_dim: int = _EDGE_FEAT_DIM,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        embedding_dim: int = 32,
        path_emb_dim: int = 0,
        path_input_dim: int = 6,
        path_d_model: int = 32,
        path_n_heads: int = 2,
        path_n_layers: int = 2,
        path_max_len: int = 64,
    ):
        torch, nn = _require_torch()
        import torch.nn.functional as F
        assert d_model % n_heads == 0
        head_dim = d_model // n_heads

        # When path encoding is enabled, the edge projection sees
        # [scalar edge feats || path embedding].
        effective_edge_input = edge_input_dim + (path_emb_dim if path_emb_dim > 0 else 0)

        class _CellGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self._init_kwargs = {
                    "node_input_dim": node_input_dim,
                    "edge_input_dim": edge_input_dim,
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "dropout": dropout,
                    "embedding_dim": embedding_dim,
                    "path_emb_dim": path_emb_dim,
                    "path_input_dim": path_input_dim,
                    "path_d_model": path_d_model,
                    "path_n_heads": path_n_heads,
                    "path_n_layers": path_n_layers,
                    "path_max_len": path_max_len,
                }
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = head_dim
                self.embedding_dim = embedding_dim
                self.path_emb_dim = path_emb_dim

                # Optional path encoder for skeleton-path edge features.
                if path_emb_dim > 0:
                    from .path_edge_encoder import PathEdgeEncoder
                    self.path_encoder = PathEdgeEncoder(
                        input_dim=path_input_dim,
                        d_model=path_d_model,
                        n_heads=path_n_heads,
                        n_layers=path_n_layers,
                        output_dim=path_emb_dim,
                        max_len=path_max_len,
                        dropout=dropout,
                    )
                else:
                    self.path_encoder = None

                # Input projections
                self.node_proj = nn.Linear(node_input_dim, d_model)
                self.edge_proj = nn.Linear(effective_edge_input, d_model)

                # Message-passing layers
                self.msg_linears = nn.ModuleList([
                    nn.Linear(d_model * 3, d_model)  # [src, dst, edge] -> message
                    for _ in range(n_layers)
                ])
                self.update_linears = nn.ModuleList([
                    nn.Linear(d_model * 2, d_model)  # [node, aggregated_msg] -> updated
                    for _ in range(n_layers)
                ])
                self.norms = nn.ModuleList([
                    nn.LayerNorm(d_model) for _ in range(n_layers)
                ])
                self.drop = nn.Dropout(dropout)

                # Output projection
                self.output_proj = nn.Linear(d_model, embedding_dim)

            def forward(
                self, node_feat, edge_src, edge_dst, edge_feat,
                path_seq=None, path_mask=None, has_path=None,
            ):
                """Run message passing and return per-synapse embeddings.

                Parameters
                ----------
                node_feat : Tensor [N, node_input_dim]
                edge_src, edge_dst : Tensor [E]
                edge_feat : Tensor [E, edge_input_dim]
                path_seq : Tensor [E, T, path_input_dim] or None
                path_mask : Tensor [E, T] bool or None  (True = padding)
                has_path : Tensor [E] bool or None      (True = real path)

                Returns
                -------
                Tensor [N, embedding_dim]
                """
                N = node_feat.size(0)
                h = self.node_proj(node_feat)         # [N, d_model]

                # If path encoder enabled, compute path embeddings and
                # concatenate with the scalar edge features before projection.
                if self.path_encoder is not None:
                    if path_seq is None or path_mask is None or has_path is None:
                        raise ValueError(
                            "CellGNN was constructed with path_emb_dim > 0 but "
                            "forward was called without path inputs."
                        )
                    path_emb = self.path_encoder(path_seq, path_mask, has_path)
                    edge_feat = torch.cat([edge_feat, path_emb], dim=-1)

                e = self.edge_proj(edge_feat)          # [E, d_model]

                for msg_lin, upd_lin, norm in zip(
                    self.msg_linears, self.update_linears, self.norms
                ):
                    # Compute messages: f(h_src, h_dst, e_ij)
                    src_h = h[edge_src]                # [E, d_model]
                    dst_h = h[edge_dst]                # [E, d_model]
                    msg_input = torch.cat([src_h, dst_h, e], dim=-1)  # [E, 3*d_model]
                    messages = F.relu(msg_lin(msg_input))              # [E, d_model]
                    messages = self.drop(messages)

                    # Aggregate messages per destination node (mean)
                    agg = torch.zeros(N, self.d_model, device=h.device, dtype=h.dtype)
                    count = torch.zeros(N, 1, device=h.device, dtype=h.dtype)
                    agg.scatter_add_(0, edge_dst.unsqueeze(1).expand_as(messages), messages)
                    count.scatter_add_(0, edge_dst.unsqueeze(1)[:, :1],
                                       torch.ones(edge_dst.size(0), 1,
                                                  device=h.device, dtype=h.dtype))
                    agg = agg / count.clamp_min(1.0)

                    # Update node embeddings
                    h_new = F.relu(upd_lin(torch.cat([h, agg], dim=-1)))
                    h = norm(h + h_new)  # residual + norm

                return self.output_proj(h)  # [N, embedding_dim]

        return _CellGNN()


# ---------------------------------------------------------------------------
# 3. Partition from embeddings
# ---------------------------------------------------------------------------


class _UF:
    """Minimal union-find (path-compressed, no external deps)."""

    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, x: int) -> int:
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]  # path halving
            x = self._p[x]
        return x

    def union(self, x: int, y: int) -> None:
        self._p[self.find(x)] = self.find(y)

    def copy(self) -> "_UF":
        uf = _UF.__new__(_UF)
        uf._p = self._p[:]
        return uf

    def labels(self) -> list[int]:
        """Return contiguous 0-based labels [N]."""
        roots = [self.find(i) for i in range(len(self._p))]
        unique_roots = sorted(set(roots))
        remap = {r: idx for idx, r in enumerate(unique_roots)}
        return [remap[r] for r in roots]

def partition_from_embeddings(
    embeddings: np.ndarray,
    *,
    threshold: float = 0.5,
    method: str = "complete",
) -> np.ndarray:
    """Cluster synapse embeddings into cell assignments.

    Parameters
    ----------
    embeddings : ndarray [N, D]
        Per-synapse embedding vectors from CellGNN.
    threshold : float
        Cosine similarity threshold for same-cell assignment.
    method : str
        "complete" (default) — complete-linkage: every pair within a cluster
        must individually have cosine sim >= threshold.  Prevents single-linkage
        chaining where one high-sim bridge contaminates unrelated synapses.
        "agglomerative" — legacy single-linkage (deprecated; chaining artefacts).
        "greedy" — greedy assignment from first unassigned node.

    Returns
    -------
    labels : ndarray [N]
        Integer cell assignment per synapse.
    """
    N = len(embeddings)
    if N == 0:
        return np.array([], dtype=np.int64)
    if N == 1:
        return np.array([0], dtype=np.int64)

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms

    if method == "greedy":
        # Simple greedy assignment
        labels = np.full(N, -1, dtype=np.int64)
        next_label = 0
        for i in range(N):
            if labels[i] >= 0:
                continue
            labels[i] = next_label
            for j in range(i + 1, N):
                if labels[j] >= 0:
                    continue
                sim = float(normed[i] @ normed[j])
                if sim >= threshold:
                    labels[j] = next_label
            next_label += 1
        return labels

    if method == "complete":
        # Complete-linkage: merge clusters only when ALL cross-cluster pairs
        # meet the threshold.  Equivalent to finding cliques in the sim graph.
        # For N <= _ANN_PARTITION_THRESHOLD use the full similarity matrix.
        if N <= _ANN_PARTITION_THRESHOLD:
            sim = normed @ normed.T  # [N, N]
            upper_tri = np.triu_indices(N, k=1)
            sims = sim[upper_tri]
            order = np.argsort(-sims)
            cluster = np.arange(N, dtype=np.int64)  # cluster[i] = cluster id of i

            for idx in order:
                if sims[idx] < threshold:
                    break
                i, j = int(upper_tri[0][idx]), int(upper_tri[1][idx])
                ci, cj = cluster[i], cluster[j]
                if ci == cj:
                    continue
                mi = np.where(cluster == ci)[0]
                mj = np.where(cluster == cj)[0]
                # All cross-cluster pairs must have sim >= threshold
                if np.all(sim[np.ix_(mi, mj)] >= threshold):
                    cluster[cluster == cj] = ci

            unique = np.unique(cluster)
            remap = {int(old): new for new, old in enumerate(unique)}
            return np.array([remap[int(c)] for c in cluster], dtype=np.int64)
        else:
            # For large N fall back to single-linkage ANN (still fast, may chain)
            from ._scipy_compat import cKDTree
            from .helpers import UnionFind
            max_dist = float(np.sqrt(max(0.0, 2.0 * (1.0 - threshold))))
            tree = cKDTree(normed)
            pairs = tree.query_pairs(r=max_dist, output_type="ndarray")
            uf = UnionFind(N)
            if len(pairs) > 0:
                for a, b in pairs:
                    uf.union(int(a), int(b))
            labels = np.array([uf.find(i) for i in range(N)], dtype=np.int64)
            unique_roots = sorted(set(labels.tolist()))
            remap2 = {r: idx for idx, r in enumerate(unique_roots)}
            return np.array([remap2[l] for l in labels], dtype=np.int64)

    # "agglomerative" — legacy single-linkage (kept for backward compat)
    from .helpers import UnionFind
    from ._scipy_compat import cKDTree

    uf = UnionFind(N)

    if N > _ANN_PARTITION_THRESHOLD:
        # Sparse ANN path: for normalized embeddings, cosine_sim ≥ t ↔ L2 ≤ sqrt(2*(1-t)).
        # query_pairs returns only close pairs, keeping memory O(N·k) instead of O(N²).
        max_dist = float(np.sqrt(max(0.0, 2.0 * (1.0 - threshold))))
        tree = cKDTree(normed)
        pairs = tree.query_pairs(r=max_dist, output_type="ndarray")
        if len(pairs) > 0:
            for a, b in pairs:
                uf.union(int(a), int(b))
    else:
        # Exact path: full pairwise matrix, fine for small N.
        sim_matrix = normed @ normed.T
        upper_tri = np.triu_indices(N, k=1)
        sims = sim_matrix[upper_tri]
        order = np.argsort(-sims)
        for idx in order:
            if sims[idx] < threshold:
                break
            i, j = int(upper_tri[0][idx]), int(upper_tri[1][idx])
            uf.union(i, j)

    # Convert to contiguous labels
    labels = np.array([uf.find(i) for i in range(N)], dtype=np.int64)
    unique_roots = sorted(set(labels.tolist()))
    remap = {r: idx for idx, r in enumerate(unique_roots)}
    return np.array([remap[l] for l in labels], dtype=np.int64)


# ---------------------------------------------------------------------------
# 4. Config
# ---------------------------------------------------------------------------

@_dataclass(frozen=True)
class CellGNNConfig:
    """Hyperparameters for CellGNN architecture and training."""
    # Architecture
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    embedding_dim: int = 32
    # Path-edge encoder (Option 2 — set path_emb_dim > 0 to enable)
    path_emb_dim: int = 0
    path_input_dim: int = 6
    path_d_model: int = 32
    path_n_heads: int = 2
    path_n_layers: int = 2
    path_max_len: int = 64
    # Training
    epochs: int = 50
    learning_rate: float = 1e-3
    margin: float = 0.5          # cosine similarity target separation for negatives
    max_pairs_per_box: int = 2048
    max_synapses_per_box: int = 2000  # randomly subsample graphs above this size
    proximity_radius_nm: float = 5000.0
    partition_threshold: float = 0.5
    partition_method: str = "agglomerative"
    seed: int = 42


# ---------------------------------------------------------------------------
# 5. Training: contrastive cell membership loss
# ---------------------------------------------------------------------------

def _sample_contrastive_pairs(
    root_ids: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
    *,
    core_mask: "np.ndarray | None" = None,
    min_group_size: int = 3,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Sample balanced positive (same root) and negative (different root) pairs.

    Parameters
    ----------
    core_mask : bool array [N] or None
        If provided, only nodes where ``core_mask[i]`` is True are eligible
        for pair sampling.  Halo context nodes are excluded from the loss.
    min_group_size : int
        Minimum number of visible synapses a root ID must have to be used as
        a positive anchor.  Cells with fewer synapses are likely box-boundary
        fragments with incomplete local evidence.
    """
    root_groups: dict[int, list[int]] = {}
    for i, rid in enumerate(root_ids):
        if core_mask is not None and not core_mask[i]:
            continue
        rid_int = int(rid)
        if rid_int > 0:
            root_groups.setdefault(rid_int, []).append(i)

    pos_pairs: list[tuple[int, int]] = []
    for members in root_groups.values():
        if len(members) < min_group_size:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pos_pairs.append((members[i], members[j]))
                if len(pos_pairs) >= max_pairs:
                    break
            if len(pos_pairs) >= max_pairs:
                break

    # Negatives can still sample from any core node, regardless of group size
    all_root_ids: list[int] = []
    all_root_nodes: dict[int, list[int]] = {}
    for i, rid in enumerate(root_ids):
        if core_mask is not None and not core_mask[i]:
            continue
        rid_int = int(rid)
        if rid_int > 0:
            if rid_int not in all_root_nodes:
                all_root_ids.append(rid_int)
            all_root_nodes.setdefault(rid_int, []).append(i)

    neg_pairs: list[tuple[int, int]] = []
    n_neg = min(len(pos_pairs) * 2, max_pairs)
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 4:
        attempts += 1
        if len(all_root_ids) < 2:
            break
        r1, r2 = rng.choice(len(all_root_ids), size=2, replace=False)
        i = int(rng.choice(all_root_nodes[all_root_ids[r1]]))
        j = int(rng.choice(all_root_nodes[all_root_ids[r2]]))
        neg_pairs.append((i, j))

    return pos_pairs, neg_pairs


def _featurize_path_lookup(
    raw: "dict[tuple[int, int], dict]",
    mode: str = DEFAULT_PATH_FEATURE_MODE,
) -> "dict[tuple[int, int], np.ndarray]":
    """Convert raw path-vertex dicts (from skeleton_path_cache) to feature arrays.

    ``raw`` maps ``(i, j)`` to ``{"path": [[x,y,z], ...], ...}``.  Returns a
    new dict mapping the same keys to float32 ``[T, D]`` feature arrays, ready
    to assign to ``SynapseGraph.edge_path_features``.  Empty paths are dropped
    (the encoder will use its ``no_path_embedding`` for missing entries).
    """
    out: dict[tuple[int, int], np.ndarray] = {}
    for key, val in raw.items():
        verts = val.get("path", [])
        if len(verts) < 2:
            continue
        arr = featurize_path_points(np.array(verts, dtype=np.float32), mode=mode)
        if arr.shape[0] > 0:
            out[key] = arr
    return out


def cell_graph_train_step(
    model,
    optimizer,
    graph: SynapseGraph,
    *,
    margin: float = 0.5,
    max_pairs: int = 2048,
    rng: "np.random.Generator | None" = None,
    edit_positive_pairs: "list[tuple[int, int]] | None" = None,
    edit_negative_pairs: "list[tuple[int, int]] | None" = None,
    edit_weight: float = 2.0,
    hard_neg_mining: bool = True,
    hard_neg_threshold: float = 0.7,
    hard_neg_weight: float = 3.0,
    max_hard_negs: int = 64,
) -> dict[str, float]:
    """One gradient step: contrastive loss over synapse embeddings.

    Loss has two terms:
    - **Positive pull**: ``1 - cosine_sim`` for same-root pairs → drives
      same-cell synapses to identical embeddings.
    - **Negative push**: ``relu(cosine_sim - (1 - margin))`` for different-root
      pairs → penalises any pair whose similarity exceeds ``(1 - margin)``.

    With ``margin=0.5`` negatives must be below 0.5 cosine similarity; positives
    aim for 1.0.  The resulting 0.5-unit gap is the learned cell boundary.

    Parameters
    ----------
    model : CellGNN
    optimizer : torch optimizer
    graph : SynapseGraph with root_ids populated
    margin : desired cosine-similarity separation between pos and neg clusters
    max_pairs : cap on pairs sampled per call (controls memory and speed)
    rng : numpy RNG for pair sampling; defaults to a fresh unseeded generator
    edit_positive_pairs : extra positive pairs from proofreader edit history
    edit_negative_pairs : extra negative (split) pairs from proofreader edit history
    edit_weight : loss multiplier for edit-derived pairs (default 2.0)
    """
    torch, _ = _require_torch()
    import torch.nn.functional as F

    if graph.root_ids is None:
        raise ValueError("SynapseGraph.root_ids must be set for training")
    if rng is None:
        rng = np.random.default_rng()

    model.train()
    optimizer.zero_grad()

    _use_paths = getattr(model, "path_emb_dim", 0) > 0
    if _use_paths:
        node_feat, edge_src, edge_dst, edge_feat, path_seq, path_mask, has_path = (
            _graph_to_tensors(graph, return_paths=True)
        )
        embeddings = model(
            node_feat, edge_src, edge_dst, edge_feat,
            path_seq=path_seq, path_mask=path_mask, has_path=has_path,
        )
    else:
        node_feat, edge_src, edge_dst, edge_feat = _graph_to_tensors(graph)
        embeddings = model(node_feat, edge_src, edge_dst, edge_feat)
    emb_norm = F.normalize(embeddings, p=2, dim=-1)

    pos_pairs, neg_pairs = _sample_contrastive_pairs(
        graph.root_ids, max_pairs, rng, core_mask=graph.core_mask
    )

    # Merge in edit-history pairs (filtering out-of-range indices)
    N = graph.n_synapses
    if edit_positive_pairs:
        for p in edit_positive_pairs:
            if p[0] < N and p[1] < N:
                pos_pairs.append(p)
    if edit_negative_pairs:
        for p in edit_negative_pairs:
            if p[0] < N and p[1] < N:
                neg_pairs.append(p)

    if not pos_pairs and not neg_pairs:
        return {"loss": 0.0, "pos_sim": 0.0, "neg_sim": 0.0, "n_pos": 0, "n_neg": 0,
                "hard_neg_sim": 0.0, "n_hard_neg": 0}

    # Online hard negative mining: find different-root pairs with surprisingly high sim.
    # For small graphs (N <= 300) score all O(N²) pairs; for large graphs sample
    # _HARD_NEG_SAMPLE_PAIRS random cross-root pairs and take the hardest among those.
    _HARD_NEG_SAMPLE_PAIRS = 4096
    hard_neg_pairs: list[tuple[int, int]] = []
    if hard_neg_mining and graph.root_ids is not None:
        root_arr = graph.root_ids
        if N <= 300:
            with torch.no_grad():
                sim_matrix = emb_norm @ emb_norm.T  # [N, N]
            ui, uj = torch.triu_indices(N, N, offset=1)
            pair_sims = sim_matrix[ui, uj]
            ri = torch.tensor(root_arr[ui.numpy()], dtype=torch.long)
            rj = torch.tensor(root_arr[uj.numpy()], dtype=torch.long)
            valid = (ri > 0) & (rj > 0)
            different = ri != rj
            hard = pair_sims > hard_neg_threshold
            mask = valid & different & hard
            hard_indices = mask.nonzero(as_tuple=False).view(-1)
            if len(hard_indices) > 0:
                sorted_by_sim = hard_indices[pair_sims[hard_indices].argsort(descending=True)]
                for k in sorted_by_sim[:max_hard_negs]:
                    hard_neg_pairs.append((int(ui[k]), int(uj[k])))
        else:
            # Large graph: sample random pairs and score only those (O(sample) not O(N²))
            cand_i = rng.integers(0, N, size=_HARD_NEG_SAMPLE_PAIRS)
            cand_j = rng.integers(0, N, size=_HARD_NEG_SAMPLE_PAIRS)
            # Keep i < j, distinct, different root, both known
            keep = (cand_i < cand_j) & (root_arr[cand_i] != 0) & (root_arr[cand_j] != 0) & (root_arr[cand_i] != root_arr[cand_j])
            cand_i, cand_j = cand_i[keep], cand_j[keep]
            if len(cand_i) > 0:
                with torch.no_grad():
                    ti = torch.tensor(cand_i, dtype=torch.long)
                    tj = torch.tensor(cand_j, dtype=torch.long)
                    sims = (emb_norm[ti] * emb_norm[tj]).sum(dim=-1)
                hard_mask = sims > hard_neg_threshold
                hard_idx = hard_mask.nonzero(as_tuple=False).view(-1)
                if len(hard_idx) > 0:
                    sorted_hard = hard_idx[sims[hard_idx].argsort(descending=True)]
                    for k in sorted_hard[:max_hard_negs]:
                        hard_neg_pairs.append((int(cand_i[k]), int(cand_j[k])))

    loss_terms = []
    pos_sim_val = 0.0
    neg_sim_val = 0.0

    if pos_pairs:
        pi = torch.tensor([p[0] for p in pos_pairs], dtype=torch.long)
        pj = torch.tensor([p[1] for p in pos_pairs], dtype=torch.long)
        pos_sims = (emb_norm[pi] * emb_norm[pj]).sum(dim=-1)
        loss_terms.append((1.0 - pos_sims).mean())
        pos_sim_val = float(pos_sims.detach().mean())

    if neg_pairs:
        ni = torch.tensor([p[0] for p in neg_pairs], dtype=torch.long)
        nj = torch.tensor([p[1] for p in neg_pairs], dtype=torch.long)
        neg_sims = (emb_norm[ni] * emb_norm[nj]).sum(dim=-1)
        # Hinge: penalise if neg_sim > (1 - margin).
        loss_terms.append(F.relu(neg_sims - (1.0 - margin)).mean())
        neg_sim_val = float(neg_sims.detach().mean())

    if hard_neg_pairs:
        hni = torch.tensor([p[0] for p in hard_neg_pairs], dtype=torch.long)
        hnj = torch.tensor([p[1] for p in hard_neg_pairs], dtype=torch.long)
        hn_sims = (emb_norm[hni] * emb_norm[hnj]).sum(dim=-1)
        loss_terms.append(hard_neg_weight * F.relu(hn_sims - (1.0 - margin)).mean())
        hn_sim_val = float(hn_sims.detach().mean())
    else:
        hn_sim_val = 0.0

    loss = sum(loss_terms)  # type: ignore[arg-type]
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "pos_sim": pos_sim_val,
        "neg_sim": neg_sim_val,
        "n_pos": len(pos_pairs),
        "n_neg": len(neg_pairs),
        "hard_neg_sim": hn_sim_val,
        "n_hard_neg": len(hard_neg_pairs),
    }


# ---------------------------------------------------------------------------
# 6. Full training loop over a BoxCache
# ---------------------------------------------------------------------------

def train_cell_gnn(
    model,
    cache,
    *,
    config: "CellGNNConfig | None" = None,
    val_cache=None,
    edit_pairs: "list | None" = None,
    edit_weight: float = 2.0,
    hard_neg_mining: bool = True,
    hard_neg_threshold: float = 0.7,
    hard_neg_weight: float = 3.0,
    seg_score_cache: "dict | None" = None,
    skeleton_path_cache: "dict | None" = None,
    path_feature_mode: str = DEFAULT_PATH_FEATURE_MODE,
    checkpoint_every: int = 0,
    checkpoint_path_template: "str | None" = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Train CellGNN over a BoxCache for ``config.epochs`` epochs.

    For each epoch:
      - Shuffle cached boxes.
      - For each box, build pre- and post-side SynapseGraphs and run one
        gradient step per side.
      - Optionally evaluate on ``val_cache`` at the end of each epoch.

    No agent simulation is required; supervision comes entirely from the
    cached synapse root IDs (same workflow as grammar training).

    Parameters
    ----------
    model : CellGNN instance
    cache : BoxCache (from dataset_builder)
    config : CellGNNConfig (defaults used if None)
    val_cache : optional BoxCache for validation
    edit_pairs : optional list of EditPair from edit_history module
    edit_weight : loss multiplier for edit-derived pairs (default 2.0)
    skeleton_path_cache : optional dict from ``load_skeleton_path_cache``.
        When provided, per-step skeleton-path features are attached to each
        training graph before the forward pass so ``PathEdgeEncoder`` receives
        real path data.  Edges without a cached path use the learned
        ``no_path_embedding``.  Format:
        ``{box_hash: {"pre": {(i,j): {"path": [...], ...}}, "post": ...}}``
    path_feature_mode : featurization mode forwarded to ``featurize_path_points``
    verbose : print per-epoch summary

    Returns
    -------
    dict with keys ``train_loss``, ``train_pos_sim``, ``train_neg_sim``,
    and (if val_cache provided) ``val_loss``.
    """
    torch, _ = _require_torch()
    from .edit_history import edit_pairs_to_contrastive

    cfg = config or CellGNNConfig()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    rng = np.random.default_rng(cfg.seed)

    # Pre-compute edit-history contrastive pairs per role
    _edit_pos: dict[str, list[tuple[int, int]]] = {"pre": [], "post": []}
    _edit_neg: dict[str, list[tuple[int, int]]] = {"pre": [], "post": []}
    if edit_pairs:
        for role in ("pre", "post"):
            pos, neg = edit_pairs_to_contrastive(edit_pairs, role)
            _edit_pos[role] = pos
            _edit_neg[role] = neg
        if verbose:
            print(
                f"Edit-history pairs: "
                f"pre +{len(_edit_pos['pre'])}/-{len(_edit_neg['pre'])}  "
                f"post +{len(_edit_pos['post'])}/-{len(_edit_neg['post'])}"
            )

    history: dict[str, list[float]] = {
        "train_loss": [], "train_pos_sim": [], "train_neg_sim": [],
    }
    if val_cache is not None:
        history["val_loss"] = []

    for epoch in range(cfg.epochs):
        import time as _time
        _epoch_start = _time.monotonic()
        epoch_metrics: dict[str, list[float]] = {
            "loss": [], "pos_sim": [], "neg_sim": [],
            "hard_neg_sim": [], "n_hard_neg": [],
        }

        for record in cache.iter_records(shuffle=True, rng=rng):
            if record.n_positive_pairs == 0 and record.n_synapses < 4:
                continue
            try:
                _, synapses = cache.load(record, load_volume=False)
            except Exception:
                continue

            # Look up pre-computed seg connectivity scores for this box
            _box_seg = seg_score_cache.get(record.box_hash, {}) if seg_score_cache else {}
            # Look up pre-computed skeleton path features for this box
            _box_paths = skeleton_path_cache.get(record.box_hash, {}) if skeleton_path_cache else {}

            for role in ("pre", "post"):
                # Convert string keys from JSON back to tuple[int,int] if needed
                _side_seg_raw = _box_seg.get(role, {})
                _side_seg: dict[tuple[int, int], float] = {
                    (k if isinstance(k, tuple) else tuple(int(x) for x in str(k).strip("() ").split(","))): float(v)
                    for k, v in _side_seg_raw.items()
                } if _side_seg_raw else {}

                full_graph = build_synapse_graph(
                    synapses, role,
                    proximity_radius_nm=cfg.proximity_radius_nm,
                    seg_connectivity_scores=_side_seg or None,
                )
                if full_graph.n_synapses < 2:
                    continue

                # Attach skeleton path features to the full graph before subdivision
                _side_raw_paths = _box_paths.get(role, {})
                if _side_raw_paths:
                    full_graph.edge_path_features = _featurize_path_lookup(
                        _side_raw_paths, mode=path_feature_mode
                    )

                # Subdivide large graphs into connected BFS sub-graphs rather than
                # randomly dropping nodes.  BFS subdivision preserves chain paths
                # and local topology; random dropping destroys both.
                if full_graph.n_synapses > cfg.max_synapses_per_box:
                    graphs_to_train = subdivide_synapse_graph(
                        full_graph,
                        n_nodes=cfg.max_synapses_per_box,
                        halo_hops=1,
                        rng=rng,
                    )
                else:
                    graphs_to_train = [full_graph]

                for graph in graphs_to_train:
                    # Augment with synapse-chain paths built from ground-truth root IDs.
                    if getattr(model, "path_emb_dim", 0) > 0:
                        chain = build_synapse_chain_paths(graph, mode=path_feature_mode)
                        if chain:
                            merged = dict(chain)
                            if graph.edge_path_features:
                                merged.update(graph.edge_path_features)
                            graph.edge_path_features = merged

                    m = cell_graph_train_step(
                        model, optimizer, graph,
                        margin=cfg.margin,
                        max_pairs=cfg.max_pairs_per_box,
                        rng=rng,
                        edit_positive_pairs=_edit_pos[role] or None,
                        edit_negative_pairs=_edit_neg[role] or None,
                        edit_weight=edit_weight,
                        hard_neg_mining=hard_neg_mining,
                        hard_neg_threshold=hard_neg_threshold,
                        hard_neg_weight=hard_neg_weight,
                    )
                    epoch_metrics["loss"].append(m["loss"])
                    epoch_metrics["pos_sim"].append(m["pos_sim"])
                    epoch_metrics["neg_sim"].append(m["neg_sim"])
                    epoch_metrics["hard_neg_sim"].append(m.get("hard_neg_sim", 0.0))
                    epoch_metrics["n_hard_neg"].append(float(m.get("n_hard_neg", 0)))

        mean_loss = float(np.mean(epoch_metrics["loss"])) if epoch_metrics["loss"] else 0.0
        mean_pos = float(np.mean(epoch_metrics["pos_sim"])) if epoch_metrics["pos_sim"] else 0.0
        mean_neg = float(np.mean(epoch_metrics["neg_sim"])) if epoch_metrics["neg_sim"] else 0.0
        mean_hn_sim = float(np.mean(epoch_metrics["hard_neg_sim"])) if epoch_metrics["hard_neg_sim"] else 0.0
        total_hn = int(sum(epoch_metrics["n_hard_neg"]))
        history["train_loss"].append(mean_loss)
        history["train_pos_sim"].append(mean_pos)
        history["train_neg_sim"].append(mean_neg)

        if val_cache is not None:
            import torch.nn.functional as F
            val_losses = []
            for record in val_cache.iter_records():
                try:
                    _, synapses = val_cache.load(record, load_volume=False)
                except Exception:
                    continue
                for role in ("pre", "post"):
                    full_val_graph = build_synapse_graph(
                        synapses, role,
                        proximity_radius_nm=cfg.proximity_radius_nm,
                    )
                    if full_val_graph.n_synapses < 2:
                        continue
                    # Use same subdivision as training for a consistent distribution
                    if full_val_graph.n_synapses > cfg.max_synapses_per_box:
                        val_subgraphs = subdivide_synapse_graph(
                            full_val_graph,
                            n_nodes=cfg.max_synapses_per_box,
                            halo_hops=1,
                            rng=rng,
                        )
                    else:
                        val_subgraphs = [full_val_graph]
                    for graph in val_subgraphs:
                        if graph.n_synapses < 2:
                            continue
                        model.eval()
                        with torch.no_grad():
                            if getattr(model, "path_emb_dim", 0) > 0:
                                nf, es, ed, ef, ps, pm, hp = _graph_to_tensors(graph, return_paths=True)
                                emb = F.normalize(
                                    model(nf, es, ed, ef, path_seq=ps, path_mask=pm, has_path=hp),
                                    p=2, dim=-1,
                                )
                            else:
                                node_feat, es, ed, ef = _graph_to_tensors(graph)
                                emb = F.normalize(
                                    model(node_feat, es, ed, ef), p=2, dim=-1
                                )
                        model.train()
                        val_pos, _ = _sample_contrastive_pairs(
                            graph.root_ids, cfg.max_pairs_per_box, rng,
                            core_mask=graph.core_mask,
                        )
                        if val_pos:
                            pi = [p[0] for p in val_pos]
                            pj = [p[1] for p in val_pos]
                            pos_sims = (emb[pi] * emb[pj]).sum(dim=-1)
                            val_losses.append(float((1.0 - pos_sims).mean()))
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            history["val_loss"].append(val_loss)

        if verbose:
            _epoch_wall = _time.monotonic() - _epoch_start
            val_str = ""
            if val_cache is not None:
                val_str = f"  val_loss={history['val_loss'][-1]:.4f}"
            hn_str = ""
            if total_hn > 0:
                hn_str = f"  hn_sim={mean_hn_sim:.3f}  n_hn={total_hn}"
            print(
                f"Epoch {epoch + 1}/{cfg.epochs}  "
                f"loss={mean_loss:.4f}  "
                f"pos_sim={mean_pos:.3f}  neg_sim={mean_neg:.3f}"
                f"{hn_str}{val_str}"
                f"  wall={_epoch_wall:.0f}s",
                flush=True,
            )

        # Periodic checkpoint (epoch is 0-indexed; save at epoch+1)
        if (checkpoint_every > 0
                and checkpoint_path_template
                and (epoch + 1) % checkpoint_every == 0):
            ckpt_path = checkpoint_path_template.format(epoch=epoch + 1)
            save_cell_gnn(ckpt_path, model)
            if verbose:
                print(f"  [checkpoint] Saved epoch {epoch + 1} → {ckpt_path}")

    return history


# ---------------------------------------------------------------------------
# 7. Inference: labels + ConnectivityGraph
# ---------------------------------------------------------------------------

def infer_cells(
    model,
    graph: SynapseGraph,
    *,
    threshold: float = 0.5,
    method: str = "complete",
) -> np.ndarray:
    """Run CellGNN on one SynapseGraph and return integer cell labels [N].

    Parameters
    ----------
    model : CellGNN (eval mode recommended)
    graph : SynapseGraph (root_ids not needed)
    threshold : cosine similarity cutoff for merging two synapses into one cell
    method : "agglomerative" or "greedy"

    Returns
    -------
    labels : int64 ndarray [N_synapses]
    """
    torch, _ = _require_torch()
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        if getattr(model, "path_emb_dim", 0) > 0:
            node_feat, edge_src, edge_dst, edge_feat, path_seq, path_mask, has_path = (
                _graph_to_tensors(graph, return_paths=True)
            )
            embeddings = model(
                node_feat, edge_src, edge_dst, edge_feat,
                path_seq=path_seq, path_mask=path_mask, has_path=has_path,
            )
        else:
            node_feat, edge_src, edge_dst, edge_feat = _graph_to_tensors(graph)
            embeddings = model(node_feat, edge_src, edge_dst, edge_feat)
        embeddings_np = F.normalize(embeddings, p=2, dim=-1).cpu().numpy()

    return partition_from_embeddings(embeddings_np, threshold=threshold, method=method)


def infer_cells_two_pass(
    model,
    graph: SynapseGraph,
    *,
    threshold: float = 0.5,
    method: str = "complete",
    path_feature_mode: str = DEFAULT_PATH_FEATURE_MODE,
) -> np.ndarray:
    """Two-pass inference using synapse-chain paths derived from the first pass.

    Pass 1: run ``infer_cells`` with whatever features are in ``graph``
    (no path features required).

    Pass 2: build synapse-chain skeleton paths from the pass-1 cell
    assignments (``build_synapse_chain_paths``), attach them to the graph,
    and re-run inference.  The path encoder now sees geometry-derived paths
    for every within-cell edge — no seg needed.

    Requires ``model.path_emb_dim > 0``; falls back to single-pass
    ``infer_cells`` otherwise.
    """
    if getattr(model, "path_emb_dim", 0) == 0:
        return infer_cells(model, graph, threshold=threshold, method=method)

    # Pass 1: scalar-only inference to get initial hypotheses
    # Temporarily clear path features so pass 1 uses no_path_embedding for all edges
    saved = graph.edge_path_features
    graph.edge_path_features = None
    labels_pass1 = infer_cells(model, graph, threshold=threshold, method=method)
    graph.edge_path_features = saved

    # Pass 2: build chain paths from pass-1 hypotheses and re-run
    chain_paths = build_synapse_chain_paths(
        graph, labels=labels_pass1, mode=path_feature_mode
    )
    graph.edge_path_features = chain_paths
    labels_pass2 = infer_cells(model, graph, threshold=threshold, method=method)
    graph.edge_path_features = saved  # restore original state
    return labels_pass2


def _score_partition(normed: np.ndarray, uf: "_UF") -> float:
    """Score a partition by mean within-cell cosine similarity.

    Each cell contributes its mean pairwise dot product (cosine sim for
    normalised embeddings).  Singletons contribute 0.0 so that merging two
    high-similarity nodes always improves the score over leaving them
    separate.  The final score is a size-weighted mean over all cells.
    """
    labels = uf.labels()
    # Group indices by cell
    cells: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        cells.setdefault(lbl, []).append(i)

    total_weight = 0.0
    total_score = 0.0
    for members in cells.values():
        size = len(members)
        if size == 1:
            # Singletons score 0.0: they are neither good nor bad merges.
            # This ensures that merging two high-similarity nodes raises the
            # score above the all-singletons baseline.
            total_score += 0.0
        else:
            emb = normed[members]              # [K, D]
            sim_sum = float((emb @ emb.T).sum()) - size  # subtract diagonal
            n_pairs = size * (size - 1)
            mean_sim = sim_sum / n_pairs
            total_score += mean_sim * size
        total_weight += size

    return total_score / total_weight if total_weight > 0 else 0.0


def boundary_partition_search(
    model,
    graph: SynapseGraph,
    *,
    low_sim: float = 0.93,
    high_sim: float = 0.99,
    max_boundary_edges: int = 12,
    beam_width: int = 8,
    corridor_scores: "dict[tuple[int,int], float] | None" = None,
    corridor_accept_threshold: float = 0.8,
    corridor_reject_threshold: float = 0.2,
) -> np.ndarray:
    """Beam search over uncertain boundary edges to find the best partition.

    Parameters
    ----------
    model : CellGNN (eval mode recommended)
    graph : SynapseGraph
    low_sim : lower bound of the ambiguous similarity band (inclusive)
    high_sim : upper bound of the ambiguous similarity band (exclusive)
    max_boundary_edges : cap on number of boundary edges to explore
    beam_width : number of best states to keep at each beam step
    corridor_scores :
        Optional dict mapping ``(i, j)`` → EM connectivity score in ``[0, 1]``
        from :func:`neuronauts.em_corridor.batch_score_boundary_edges`.
        Scores above *corridor_accept_threshold* force-accept the merge before
        beam search; scores below *corridor_reject_threshold* force-reject.
        Edges with intermediate scores (or missing from the dict) still go
        through the beam search as normal.
    corridor_accept_threshold :
        EM score above which a merge is force-accepted (default 0.8).
    corridor_reject_threshold :
        EM score below which a merge is force-rejected (default 0.2).

    Returns
    -------
    labels : int64 ndarray [N_synapses]
    """
    torch, _ = _require_torch()
    import torch.nn.functional as F

    N = graph.n_synapses
    if N == 0:
        return np.array([], dtype=np.int64)
    if N == 1:
        return np.array([0], dtype=np.int64)

    # --- Step 1: run GNN inference once ---
    model.eval()
    with torch.no_grad():
        if getattr(model, "path_emb_dim", 0) > 0:
            nf, es, ed, ef, ps, pm, hp = _graph_to_tensors(graph, return_paths=True)
            raw = model(nf, es, ed, ef, path_seq=ps, path_mask=pm, has_path=hp)
        else:
            node_feat, edge_src, edge_dst, edge_feat = _graph_to_tensors(graph)
            raw = model(node_feat, edge_src, edge_dst, edge_feat)
        normed = F.normalize(raw, p=2, dim=-1).cpu().numpy()  # [N, D]

    # --- Step 2: find boundary edges (only over graph edges) ---
    midpoint = (low_sim + high_sim) / 2.0
    boundary: list[tuple[float, int, int]] = []  # (distance_to_mid, i, j)

    seen_pairs: set[tuple[int, int]] = set()
    for e in graph.edges:
        i, j = e.src, e.dst
        key = (min(i, j), max(i, j))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        sim = float(normed[i] @ normed[j])
        if low_sim <= sim < high_sim:
            boundary.append((abs(sim - midpoint), i, j))

    # --- Step 3: cap at max_boundary_edges most uncertain ---
    boundary.sort(key=lambda x: x[0])          # ascending: closest to midpoint first
    boundary = boundary[:max_boundary_edges]
    boundary_edges = [(i, j) for (_, i, j) in boundary]

    # --- Step 4: base partition from high-confidence edges only ---
    base_labels = partition_from_embeddings(normed, threshold=high_sim)
    base_uf = _UF(N)
    for n_idx in range(N):
        if base_labels[n_idx] != n_idx:
            # Find the representative for this label
            pass
    # Re-build union-find from the base partition labels
    label_to_rep: dict[int, int] = {}
    for n_idx in range(N):
        lbl = int(base_labels[n_idx])
        if lbl not in label_to_rep:
            label_to_rep[lbl] = n_idx
        else:
            base_uf.union(n_idx, label_to_rep[lbl])

    # --- Step 4.5: apply EM corridor overrides ---
    # Force-accept/reject edges whose EM score is decisive; the rest go to beam.
    if corridor_scores:
        beam_edges: list[tuple[int, int]] = []
        for ei, ej in boundary_edges:
            key = (min(ei, ej), max(ei, ej))
            score = corridor_scores.get(key, corridor_scores.get((ei, ej), None))
            if score is None:
                beam_edges.append((ei, ej))
            elif score >= corridor_accept_threshold:
                base_uf.union(ei, ej)   # definitive merge
            elif score <= corridor_reject_threshold:
                pass                    # definitive reject — drop from search
            else:
                beam_edges.append((ei, ej))  # still ambiguous
        boundary_edges = beam_edges

    # --- Step 5: beam search ---
    if not boundary_edges:
        # No uncertain edges — return the high-confidence partition
        return np.array(base_uf.labels(), dtype=np.int64)

    # Beam: list of (score, uf_state)
    # Compute initial score
    init_score = _score_partition(normed, base_uf)
    beam: list[tuple[float, "_UF"]] = [(-init_score, base_uf)]

    for (ei, ej) in boundary_edges:
        next_beam: list[tuple[float, "_UF"]] = []
        for (neg_score, uf_state) in beam:
            # Branch A: reject merge (keep as-is)
            next_beam.append((neg_score, uf_state))

            # Branch B: accept merge
            new_uf = uf_state.copy()
            new_uf.union(ei, ej)
            new_score = _score_partition(normed, new_uf)
            next_beam.append((-new_score, new_uf))

        # Keep best beam_width states (lowest neg_score = highest score)
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]

    # Pick the highest-scoring final state
    best_neg_score, best_uf = beam[0]
    return np.array(best_uf.labels(), dtype=np.int64)


def infer_cells_with_search(
    model,
    graph: SynapseGraph,
    *,
    high_threshold: float = 0.99,
    low_sim: float = 0.93,
    max_boundary_edges: int = 12,
    beam_width: int = 8,
    corridor_scores: "dict[tuple[int,int], float] | None" = None,
    corridor_accept_threshold: float = 0.8,
    corridor_reject_threshold: float = 0.2,
) -> np.ndarray:
    """Run CellGNN with boundary-edge partition search for better accuracy.

    Calls :func:`boundary_partition_search` to resolve ambiguous merge
    decisions in the similarity band ``[low_sim, high_threshold)``.

    Parameters
    ----------
    model : CellGNN
    graph : SynapseGraph
    high_threshold : cosine similarity above which merges are accepted without search
    low_sim : lower bound for the ambiguous band
    max_boundary_edges : cap on uncertain edges explored in the search
    beam_width : beam width for the search
    corridor_scores :
        Optional per-edge EM connectivity scores from
        :func:`neuronauts.em_corridor.batch_score_boundary_edges`.
    corridor_accept_threshold :
        EM score above which a merge is force-accepted (default 0.8).
    corridor_reject_threshold :
        EM score below which a merge is force-rejected (default 0.2).

    Returns
    -------
    labels : int64 ndarray [N_synapses]
    """
    return boundary_partition_search(
        model,
        graph,
        low_sim=low_sim,
        high_sim=high_threshold,
        max_boundary_edges=max_boundary_edges,
        beam_width=beam_width,
        corridor_scores=corridor_scores,
        corridor_accept_threshold=corridor_accept_threshold,
        corridor_reject_threshold=corridor_reject_threshold,
    )


def _get_boundary_edges(
    model,
    graph: SynapseGraph,
    *,
    low_sim: float = 0.93,
    high_sim: float = 0.999,
    max_boundary_edges: int = 40,
) -> "list[tuple[int, int]]":
    """Run GNN inference and return the list of ambiguous boundary edge pairs.

    These are the edges whose cosine similarity falls in ``[low_sim, high_sim)``,
    capped at ``max_boundary_edges`` (selecting those closest to the midpoint of
    the band first).

    Parameters
    ----------
    model : CellGNN
    graph : SynapseGraph
    low_sim : lower bound of the ambiguous similarity band (inclusive)
    high_sim : upper bound of the ambiguous similarity band (exclusive)
    max_boundary_edges : maximum number of boundary edges to return

    Returns
    -------
    list of ``(i, j)`` index pairs (undirected, i <= j)
    """
    torch, _ = _require_torch()
    import torch.nn.functional as F

    N = graph.n_synapses
    if N <= 1:
        return []

    model.eval()
    with torch.no_grad():
        if getattr(model, "path_emb_dim", 0) > 0:
            nf, es, ed, ef, ps, pm, hp = _graph_to_tensors(graph, return_paths=True)
            raw = model(nf, es, ed, ef, path_seq=ps, path_mask=pm, has_path=hp)
        else:
            node_feat, edge_src, edge_dst, edge_feat = _graph_to_tensors(graph)
            raw = model(node_feat, edge_src, edge_dst, edge_feat)
        normed = F.normalize(raw, p=2, dim=-1).cpu().numpy()  # [N, D]

    midpoint = (low_sim + high_sim) / 2.0
    boundary: list[tuple[float, int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for e in graph.edges:
        i, j = e.src, e.dst
        key = (min(i, j), max(i, j))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        sim = float(normed[i] @ normed[j])
        if low_sim <= sim < high_sim:
            boundary.append((abs(sim - midpoint), key[0], key[1]))

    boundary.sort(key=lambda x: x[0])
    boundary = boundary[:max_boundary_edges]
    return [(i, j) for (_, i, j) in boundary]


def score_cell_quality(
    model,
    graph: SynapseGraph,
    labels: np.ndarray,
    *,
    topology_validator=None,
) -> dict[int, float]:
    """Score each inferred cell's structural coherence.

    Uses the CellGNN embeddings for each cell's synapses, then optionally
    runs them through an ``AttentionArborValidator`` to get a [0, 1]
    plausibility score.  Without a validator, returns cosine cohesion
    (mean pairwise similarity within each cell).

    Parameters
    ----------
    model : CellGNN
    graph : SynapseGraph used for ``infer_cells``
    labels : int64 [N] cell assignments from ``infer_cells``
    topology_validator : optional ``AttentionArborValidator`` checkpoint or module

    Returns
    -------
    dict mapping cell_id → quality score in [0, 1]
    """
    torch, _ = _require_torch()
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        if getattr(model, "path_emb_dim", 0) > 0:
            nf, es, ed, ef, ps, pm, hp = _graph_to_tensors(graph, return_paths=True)
            embeddings = model(nf, es, ed, ef, path_seq=ps, path_mask=pm, has_path=hp)
        else:
            node_feat, edge_src, edge_dst, edge_feat = _graph_to_tensors(graph)
            embeddings = model(node_feat, edge_src, edge_dst, edge_feat)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

    # Load validator if path given
    validator = None
    if isinstance(topology_validator, str):
        from .topology_model import load_validator
        validator = load_validator(topology_validator)
    elif topology_validator is not None:
        validator = topology_validator

    unique_cells = np.unique(labels)
    scores: dict[int, float] = {}

    for cell_id in unique_cells:
        mask = labels == cell_id
        cell_emb = embeddings[mask]  # [K, D]

        if cell_emb.shape[0] <= 1:
            scores[int(cell_id)] = 1.0
            continue

        if validator is not None:
            # AttentionArborValidator expects [B, K, D]
            with torch.no_grad():
                prob = validator(cell_emb.unsqueeze(0))
                scores[int(cell_id)] = float(prob.squeeze().cpu())
        else:
            # Fallback: mean pairwise cosine similarity
            sim = (cell_emb @ cell_emb.T).cpu().numpy()
            n = sim.shape[0]
            triu = np.triu_indices(n, k=1)
            scores[int(cell_id)] = float(np.mean(sim[triu])) if len(triu[0]) > 0 else 1.0

    return scores


def connectivity_graph_from_cell_labels(
    pre_labels: np.ndarray,
    post_labels: np.ndarray,
    synapses: "SynapseTable",
) -> "ConnectivityGraph":
    """Build a ConnectivityGraph from per-synapse cell labels.

    Parameters
    ----------
    pre_labels : int64 [N]
        Cell assignment for each synapse on the pre side.
    post_labels : int64 [N]
        Cell assignment for each synapse on the post side.
    synapses : SynapseTable

    Returns
    -------
    ConnectivityGraph with neurons (pre + post cells) and synapse edges.
    """
    from .merge import ConnectivityGraph, MergedNeuron

    N = len(pre_labels)
    neurons: dict[int, MergedNeuron] = {}
    neuron_id = 0

    # Pre-side cells
    pre_cell_map: dict[int, int] = {}  # label -> neuron_id
    pre_groups: dict[int, list[int]] = {}
    for i, lbl in enumerate(pre_labels):
        pre_groups.setdefault(int(lbl), []).append(i)
    for lbl, indices in pre_groups.items():
        pts = synapses.pre_pt[indices].astype(np.float32)
        neurons[neuron_id] = MergedNeuron(
            neuron_id=neuron_id,
            agent_ids=[],
            path_points=pts,
            synapse_indices=sorted(indices),
            role="pre",
        )
        pre_cell_map[lbl] = neuron_id
        neuron_id += 1

    # Post-side cells
    post_cell_map: dict[int, int] = {}
    post_groups: dict[int, list[int]] = {}
    for i, lbl in enumerate(post_labels):
        post_groups.setdefault(int(lbl), []).append(i)
    for lbl, indices in post_groups.items():
        pts = synapses.post_pt[indices].astype(np.float32)
        neurons[neuron_id] = MergedNeuron(
            neuron_id=neuron_id,
            agent_ids=[],
            path_points=pts,
            synapse_indices=sorted(indices),
            role="post",
        )
        post_cell_map[lbl] = neuron_id
        neuron_id += 1

    # Build synapse edges: (pre_neuron_id, post_neuron_id, syn_idx)
    edges: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for syn_idx in range(N):
        pre_nid = pre_cell_map[int(pre_labels[syn_idx])]
        post_nid = post_cell_map[int(post_labels[syn_idx])]
        key = (pre_nid, post_nid, syn_idx)
        if key not in seen:
            seen.add(key)
            edges.append(key)

    return ConnectivityGraph(
        neurons=neurons,
        edges=edges,
        unresolved_synapse_indices=[],
    )


# ---------------------------------------------------------------------------
# 8. Persistence
# ---------------------------------------------------------------------------

def save_cell_gnn(path, model) -> None:
    """Save a CellGNN checkpoint."""
    torch, _ = _require_torch()
    from pathlib import Path as _P
    p = _P(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "init_kwargs": dict(getattr(model, "_init_kwargs", {})),
    }, p)


def load_cell_gnn(path):
    """Load a CellGNN checkpoint."""
    torch, _ = _require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CellGNN(**ckpt.get("init_kwargs", {}))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def precompute_seg_scores_for_cache(
    cache: "BoxCache",
    records=None,
    *,
    proximity_radius_nm: float = 5000.0,
    radius_nm: float = 1500.0,
    mip: int = 2,
    max_length_nm: float = 15_000.0,
    verbose: bool = True,
) -> dict[str, dict[tuple[int, int], float]]:
    """Pre-compute EM segmentation corridor scores for all edges in a BoxCache.

    For each box, builds the proximity graph to enumerate candidate edges, converts
    synapse positions from MIP-2 box-relative voxels to absolute nanometres, and
    calls :func:`~neuronauts.em_corridor.batch_score_seg_connectivity` to score
    every edge against the live MICrONS segmentation.

    Results are keyed by ``record.box_hash`` so they can be looked up quickly
    during training via :func:`train_cell_gnn`.

    Parameters
    ----------
    cache:
        BoxCache containing cached synapse data.
    records:
        Iterable of BoxRecord to process.  If None, uses ``cache.iter_records()``.
    proximity_radius_nm:
        Spatial radius used to enumerate candidate edges.
    radius_nm, mip, max_length_nm:
        Passed through to ``batch_score_seg_connectivity``.
    verbose:
        Print per-box progress.

    Returns
    -------
    dict mapping ``box_hash`` → ``{(i, j): score}`` for pre-side synapses.
    (Post side is handled implicitly — same scores are indexed by the same
    synapse indices, which are shared across sides within one box.)
    """
    import json as _json
    from .em_corridor import batch_score_seg_connectivity
    from .fetch import make_cube_bbox_nm

    _MIP2_VOX = np.array([32.0, 32.0, 40.0])

    if records is None:
        records = list(cache.iter_records())

    seg_cache: dict[str, dict[tuple[int, int], float]] = {}

    for rec in records:
        box_hash = rec.box_hash
        try:
            _, synapses = cache.load(rec)
        except Exception:
            continue
        if len(synapses.pre_pt) < 2:
            continue

        # Convert pre_pt (MIP2 box-relative voxels) → absolute nm
        bbox = make_cube_bbox_nm(tuple(rec.center_nm), rec.side_um)
        box_origin_nm = np.array(bbox[0], dtype=np.float64)
        pre_nm = synapses.pre_pt.astype(np.float64) * _MIP2_VOX + box_origin_nm
        post_nm = synapses.post_pt.astype(np.float64) * _MIP2_VOX + box_origin_nm

        # Enumerate proximity edges for both sides combined
        from ._scipy_compat import cKDTree as _cKDTree

        all_edges: set[tuple[int, int]] = set()
        for positions in (pre_nm, post_nm):
            tree = _cKDTree(positions)
            pairs = tree.query_pairs(r=proximity_radius_nm, output_type="ndarray")
            for a, b in pairs:
                key = (int(min(a, b)), int(max(a, b)))
                all_edges.add(key)

        if not all_edges:
            continue

        edge_list = sorted(all_edges)

        # Score pre-side (post side shares same synapse indices in this box)
        if verbose:
            print(f"  [{box_hash[:8]}] {len(edge_list)} edges to score …", end=" ", flush=True)

        pre_scores = batch_score_seg_connectivity(
            pre_nm, edge_list,
            radius_nm=radius_nm,
            mip=mip,
            max_length_nm=max_length_nm,
            verbose=False,
        )
        post_scores = batch_score_seg_connectivity(
            post_nm, edge_list,
            radius_nm=radius_nm,
            mip=mip,
            max_length_nm=max_length_nm,
            verbose=False,
        )

        # Store both side scores; build_synapse_graph uses whichever matches
        seg_cache[box_hash] = {
            "pre": {str(k): v for k, v in pre_scores.items()},
            "post": {str(k): v for k, v in post_scores.items()},
        }

        n_signal = sum(1 for v in pre_scores.values() if v != 0.5)
        if verbose:
            print(f"done  {n_signal}/{len(edge_list)} have signal (≠0.5)")

    return seg_cache


def precompute_seg_scores_fast(
    cache: "BoxCache",
    records=None,
    *,
    proximity_radius_nm: float = 5000.0,
    mip: int = 3,
    margin_nm: float = 200.0,
    verbose: bool = True,
) -> dict[str, dict[tuple[int, int], float]]:
    """Pre-compute EM seg scores using one bbox fetch per box (fast).

    Instead of fetching a corridor volume per edge (the slow approach in
    :func:`precompute_seg_scores_for_cache`), this function fetches one
    bounding-box seg volume per box, covering all synapse positions, and reads
    the proofread seg ID at each synapse from the cached array.

    At MIP 3 (~64×64×80 nm/vox) a 6 µm box is ~94×94×75 voxels ≈ 5 MB.
    For 37 boxes that is ~185 MB total vs ~74 GB for the corridor approach.

    Parameters
    ----------
    cache:
        BoxCache containing cached synapse data.
    records:
        Iterable of BoxRecord to process.  If None, uses ``cache.iter_records()``.
    proximity_radius_nm:
        Spatial radius used to enumerate candidate edges.
    mip:
        CloudVolume MIP level for seg fetches (default 3 for speed).
    margin_nm:
        Padding around the synapse bounding box when fetching the seg volume.
    verbose:
        Print per-box progress.

    Returns
    -------
    dict mapping ``box_hash`` → ``{"pre": {(i,j): score}, "post": {(i,j): score}}``.
    """
    from .em_corridor import batch_score_seg_connectivity_fast
    from .fetch import make_cube_bbox_nm

    _MIP2_VOX = np.array([32.0, 32.0, 40.0])

    if records is None:
        records = list(cache.iter_records())

    seg_cache: dict[str, dict] = {}

    for rec_idx, rec in enumerate(records):
        box_hash = rec.box_hash
        try:
            _, synapses = cache.load(rec)
        except Exception:
            continue
        if len(synapses.pre_pt) < 2:
            continue

        # Convert pre_pt and post_pt (MIP2 box-relative voxels) → absolute nm
        bbox = make_cube_bbox_nm(tuple(rec.center_nm), rec.side_um)
        box_origin_nm = np.array(bbox[0], dtype=np.float64)
        pre_nm = synapses.pre_pt.astype(np.float64) * _MIP2_VOX + box_origin_nm
        post_nm = synapses.post_pt.astype(np.float64) * _MIP2_VOX + box_origin_nm

        # Enumerate proximity edges
        from ._scipy_compat import cKDTree as _cKDTree

        pre_edges: list[tuple[int, int]] = []
        post_edges: list[tuple[int, int]] = []
        seen_pre: set[tuple[int, int]] = set()
        seen_post: set[tuple[int, int]] = set()

        pre_tree = _cKDTree(pre_nm)
        for a, b in pre_tree.query_pairs(r=proximity_radius_nm, output_type="ndarray"):
            key = (int(min(a, b)), int(max(a, b)))
            if key not in seen_pre:
                seen_pre.add(key)
                pre_edges.append(key)

        post_tree = _cKDTree(post_nm)
        for a, b in post_tree.query_pairs(r=proximity_radius_nm, output_type="ndarray"):
            key = (int(min(a, b)), int(max(a, b)))
            if key not in seen_post:
                seen_post.add(key)
                post_edges.append(key)

        if not pre_edges and not post_edges:
            continue

        n_pre = len(pre_edges)
        n_post = len(post_edges)
        if verbose:
            print(
                f"  [{rec_idx+1}/{len(records)}] {box_hash[:8]}  "
                f"{n_pre} pre-edges  {n_post} post-edges … ",
                end="",
                flush=True,
            )

        try:
            pre_scores = batch_score_seg_connectivity_fast(
                pre_nm, pre_edges, mip=mip, margin_nm=margin_nm, verbose=False,
            ) if pre_edges else {}
            post_scores = batch_score_seg_connectivity_fast(
                post_nm, post_edges, mip=mip, margin_nm=margin_nm, verbose=False,
            ) if post_edges else {}
        except Exception as exc:
            if verbose:
                print(f"FAILED ({exc!r})")
            continue

        seg_cache[box_hash] = {
            "pre": {str(k): v for k, v in pre_scores.items()},
            "post": {str(k): v for k, v in post_scores.items()},
        }

        n_signal = sum(1 for v in pre_scores.values() if v != 0.5) + \
                   sum(1 for v in post_scores.values() if v != 0.5)
        if verbose:
            print(f"done  {n_signal}/{n_pre + n_post} have signal (≠0.5)")

    return seg_cache


# ---------------------------------------------------------------------------
# Option-3 data plane: self-skeletonization from BossDB seg volumes
# ---------------------------------------------------------------------------

def precompute_self_skeletons_for_cache(
    cache: "BoxCache",
    output_dir: str,
    *,
    records=None,
    mip: int = 3,
    margin_nm: float = 200.0,
    teasar_const: float = 5.0,
    teasar_scale: float = 1.5,
    min_vertices: int = 5,
    verbose: bool = True,
) -> dict[str, str]:
    """Self-skeletonize all roots in each box from the BossDB seg volume.

    For each cached box: fetches the seg volume covering all synapse positions
    (single CloudVolume request, ~5 MB at MIP 3), runs kimimaro on the labeled
    volume to produce a skeleton per non-zero seg ID, and saves the result as
    one ``.npz`` per (box_hash, root_id) pair.

    Unlike the CAVE skeleton service, this works for **all** roots — proofread
    or not — because it skeletonizes whatever connected segmentation exists in
    BossDB.  The catch: in a 6 µm box, most roots are partial fragments whose
    skeletons end at the box boundary.  That's OK for path tracing as long as
    the box contains enough of the neurite to connect the synapse pair.

    Parameters
    ----------
    cache : BoxCache
    output_dir : str
        Directory in which to write per-box skeleton archives.
    records : iterable, optional
    mip : int
        CloudVolume MIP for the seg fetch (default 3 = ~64×64×80 nm/vox).
    margin_nm : float
        Padding around synapse bounding box for the seg fetch.
    teasar_const, teasar_scale : float
        Kimimaro TEASAR parameters.  Defaults follow the kimimaro README.
    min_vertices : int
        Skeletons with fewer vertices than this are discarded (likely noise).
    verbose : bool

    Returns
    -------
    dict[str, str]
        Mapping ``box_hash`` -> path to its skeleton archive.
    """
    from pathlib import Path as _P
    from .fetch import fetch_seg_volume, make_cube_bbox_nm

    try:
        import kimimaro
    except ImportError as exc:  # pragma: no cover - env issue
        raise ImportError(
            "kimimaro not installed.  pip install kimimaro crackle-codec"
        ) from exc

    if records is None:
        records = list(cache.iter_records())

    out_root = _P(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    _MIP2_VOX = np.array([32.0, 32.0, 40.0])
    manifest: dict[str, str] = {}

    for rec_idx, rec in enumerate(records):
        box_hash = rec.box_hash
        out_path = out_root / f"{box_hash}.npz"
        if out_path.exists():
            manifest[box_hash] = str(out_path)
            if verbose:
                print(f"  [{rec_idx+1}/{len(records)}] {box_hash[:8]}: cached, skip")
            continue

        try:
            _, syn = cache.load(rec)
        except Exception:
            continue

        # Synapse bounding box in nm (covers both pre and post)
        bbox = make_cube_bbox_nm(tuple(rec.center_nm), rec.side_um)
        box_origin_nm = np.array(bbox[0], dtype=np.float64)
        pre_nm = syn.pre_pt.astype(np.float64) * _MIP2_VOX + box_origin_nm
        post_nm = syn.post_pt.astype(np.float64) * _MIP2_VOX + box_origin_nm
        all_nm = np.vstack([pre_nm, post_nm])
        min_nm = all_nm.min(axis=0) - margin_nm
        max_nm = all_nm.max(axis=0) + margin_nm
        seg_bbox = (tuple(min_nm.tolist()), tuple(max_nm.tolist()))

        if verbose:
            print(
                f"  [{rec_idx+1}/{len(records)}] {box_hash[:8]}: "
                f"fetching seg ({seg_bbox[1][0]-seg_bbox[0][0]:.0f} x "
                f"{seg_bbox[1][1]-seg_bbox[0][1]:.0f} x "
                f"{seg_bbox[1][2]-seg_bbox[0][2]:.0f} nm) ...",
                end=" ", flush=True,
            )
        try:
            vol = fetch_seg_volume(seg_bbox, mip=mip)
        except Exception as exc:
            if verbose:
                print(f"FAILED ({exc!r})")
            continue

        if verbose:
            print(f"got {vol.data.shape}, skeletonizing ...", end=" ", flush=True)

        try:
            skels = kimimaro.skeletonize(
                vol.data.astype(np.uint64),
                teasar_params={
                    "scale": float(teasar_scale),
                    "const": float(teasar_const),
                    "pdrf_scale": 100,
                    "pdrf_exponent": 4,
                },
                anisotropy=tuple(int(v) for v in vol.voxel_size_nm),
                fix_branching=True,
                fix_borders=True,
                progress=False,
            )
        except Exception as exc:
            if verbose:
                print(f"FAILED ({exc!r})")
            continue

        # Translate skeleton vertices into the absolute-nm frame.
        # kimimaro returns vertices in voxel*anisotropy coords relative to the
        # input volume's origin (voxel 0,0,0).  We need to add the seg volume's
        # origin in nm to get absolute nm coordinates.
        seg_origin_nm = np.array([
            vol.bbox_voxels[0][i] * vol.voxel_size_nm[i] for i in range(3)
        ], dtype=np.float64)

        kept_root_ids: list[int] = []
        kept_offsets: list[int] = []  # cumulative vertex offset per root
        kept_n_verts: list[int] = []
        kept_n_edges: list[int] = []
        all_verts: list[np.ndarray] = []
        all_edges: list[np.ndarray] = []

        v_offset = 0
        for label, sk in skels.items():
            if int(label) == 0:
                continue
            verts = np.asarray(sk.vertices, dtype=np.float32)
            edges = np.asarray(sk.edges, dtype=np.int64)
            if len(verts) < min_vertices:
                continue
            # Add seg volume origin to translate kimimaro's relative coords
            # into absolute nm.
            verts_abs = verts + seg_origin_nm.astype(np.float32)
            kept_root_ids.append(int(label))
            kept_offsets.append(v_offset)
            kept_n_verts.append(len(verts_abs))
            kept_n_edges.append(len(edges))
            all_verts.append(verts_abs)
            # Edges are local to this skeleton; re-index globally
            all_edges.append(edges + v_offset)
            v_offset += len(verts_abs)

        if not kept_root_ids:
            if verbose:
                print(f"no skeletons (all <{min_vertices} verts)")
            continue

        merged_verts = np.concatenate(all_verts, axis=0).astype(np.float32)
        merged_edges = np.concatenate(all_edges, axis=0).astype(np.int64)

        np.savez_compressed(
            out_path,
            root_ids=np.asarray(kept_root_ids, dtype=np.int64),
            v_offsets=np.asarray(kept_offsets + [v_offset], dtype=np.int64),
            n_verts=np.asarray(kept_n_verts, dtype=np.int64),
            n_edges=np.asarray(kept_n_edges, dtype=np.int64),
            vertices=merged_verts,
            edges=merged_edges,
            voxel_size_nm=np.asarray(vol.voxel_size_nm, dtype=np.float32),
        )
        manifest[box_hash] = str(out_path)
        if verbose:
            print(
                f"saved {len(kept_root_ids)} skeletons "
                f"(total {len(merged_verts)} verts, {len(merged_edges)} edges)"
            )

    return manifest


def load_self_skeleton_archive(path: str) -> dict[int, "tuple"]:
    """Load a per-box self-skeleton archive saved by
    :func:`precompute_self_skeletons_for_cache`.

    Returns
    -------
    dict[int, (vertices_nm, edges)]
        Mapping root_id -> (float32 [V, 3], int64 [E, 2])
    """
    d = np.load(path, allow_pickle=False)
    root_ids = d["root_ids"].astype(np.int64)
    v_offsets = d["v_offsets"].astype(np.int64)
    n_edges = d["n_edges"].astype(np.int64)
    vertices = d["vertices"].astype(np.float32)
    edges = d["edges"].astype(np.int64)

    out: dict[int, tuple] = {}
    e_offset = 0
    for i, rid in enumerate(root_ids):
        v_lo = int(v_offsets[i])
        v_hi = int(v_offsets[i + 1])
        ne = int(n_edges[i])
        sub_v = vertices[v_lo:v_hi]
        sub_e = edges[e_offset:e_offset + ne] - v_lo  # rebase to local indices
        out[int(rid)] = (sub_v, sub_e)
        e_offset += ne
    return out


# ---------------------------------------------------------------------------
# Option-2 data plane: skeleton-path precompute
# ---------------------------------------------------------------------------

def _skeleton_to_csr(skel) -> "tuple":
    """Build a symmetric CSR adjacency matrix from a SkeletonData.

    Edge weight = Euclidean distance between connected vertices.

    Returns
    -------
    (csr, vertices_nm) where csr is the sparse adjacency and vertices_nm is
    a [V, 3] float32 array.
    """
    from scipy.sparse import csr_matrix

    verts = np.asarray(skel.vertices, dtype=np.float32)
    edges = np.asarray(skel.edges, dtype=np.int64)
    if len(verts) == 0 or len(edges) == 0:
        return None, verts

    # Filter out self-loops and out-of-range edges
    valid = (edges[:, 0] != edges[:, 1]) & \
            (edges[:, 0] >= 0) & (edges[:, 0] < len(verts)) & \
            (edges[:, 1] >= 0) & (edges[:, 1] < len(verts))
    edges = edges[valid]
    if len(edges) == 0:
        return None, verts

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    weights = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    data = np.concatenate([weights, weights]).astype(np.float64)

    csr = csr_matrix((data, (rows, cols)), shape=(len(verts), len(verts)))
    return csr, verts


def _nearest_vertex_idx(verts: np.ndarray, point_nm: np.ndarray) -> int:
    """Return index of the skeleton vertex nearest a 3D point (Euclidean)."""
    if len(verts) == 0:
        return -1
    diffs = verts.astype(np.float64) - np.asarray(point_nm, dtype=np.float64)
    d2 = (diffs * diffs).sum(axis=1)
    return int(np.argmin(d2))


def _skeleton_path_points(
    csr,
    verts: np.ndarray,
    src_idx: int,
    dst_idx: int,
    *,
    max_path_nm: float = 50_000.0,
) -> np.ndarray:
    """Return the ordered path vertices (in nm) from src_idx to dst_idx via Dijkstra.

    If no path exists or the path exceeds ``max_path_nm``, returns empty
    ``[0, 3]`` array.
    """
    from scipy.sparse.csgraph import dijkstra

    if csr is None or src_idx < 0 or dst_idx < 0:
        return np.zeros((0, 3), dtype=np.float32)
    if src_idx == dst_idx:
        return verts[[src_idx]].astype(np.float32, copy=False)

    distances, predecessors = dijkstra(
        csr, indices=src_idx, return_predecessors=True, unweighted=False,
    )
    if not np.isfinite(distances[dst_idx]) or distances[dst_idx] > max_path_nm:
        return np.zeros((0, 3), dtype=np.float32)

    # Reconstruct path
    path = [dst_idx]
    cur = dst_idx
    while cur != src_idx:
        cur = int(predecessors[cur])
        if cur < 0:
            return np.zeros((0, 3), dtype=np.float32)
        path.append(cur)
    path.reverse()
    return verts[path].astype(np.float32, copy=False)


def precompute_skeleton_paths_for_cache(
    cache: "BoxCache",
    skeleton_dir: str,
    *,
    records=None,
    proximity_radius_nm: float = 5000.0,
    max_path_nm: float = 50_000.0,
    skeleton_service_version: int = 4,
    verbose: bool = True,
) -> dict[str, dict]:
    """Pre-compute skeleton paths between proximity-graph synapse pairs.

    For each cached box and each side ("pre", "post"), enumerate proximity
    edges between synapses.  When both endpoints share a CAVE root ID and the
    skeleton for that root is cached at ``skeleton_dir/v<version>/``, trace the
    Dijkstra-shortest path through the skeleton and store its vertex sequence
    (ordered, nm coordinates).  Cross-root edges are stored with an empty path.

    Parameters
    ----------
    cache : BoxCache
    skeleton_dir : str
        Root directory of cached skeletons (per ``scripts/fetch_skeletons.py``).
        Subdirs of the form ``v<materialization_version>/`` contain
        ``v<v>_rid<root_id>_skv<sk_v>.npz`` files.
    records : iterable of BoxRecord, optional
    proximity_radius_nm : float
    max_path_nm : float
        Skeleton paths longer than this are dropped (treated as no-path).
    skeleton_service_version : int
    verbose : bool

    Returns
    -------
    dict mapping ``box_hash`` -> {
        "pre":  {(i, j): {"path": [[x,y,z], ...], "len_nm": float, "same_root": bool}},
        "post": ...
    }
    """
    import json as _json
    from pathlib import Path as _P
    from .fetch import make_cube_bbox_nm, SkeletonData
    from ._scipy_compat import cKDTree as _cKDTree

    _MIP2_VOX = np.array([32.0, 32.0, 40.0])

    if records is None:
        records = list(cache.iter_records())

    out: dict[str, dict] = {}

    for rec_idx, rec in enumerate(records):
        box_hash = rec.box_hash
        meta_path = _P(cache.cache_dir) / f"{box_hash}.json"
        try:
            with open(meta_path) as f:
                meta = _json.load(f)
            version = int(meta.get("root_id_version", 1412))
        except Exception:
            continue

        try:
            _, syn = cache.load(rec)
        except Exception:
            continue
        if len(syn.pre_pt) < 2:
            continue

        bbox = make_cube_bbox_nm(tuple(rec.center_nm), rec.side_um)
        box_origin_nm = np.array(bbox[0], dtype=np.float64)
        pre_nm = syn.pre_pt.astype(np.float64) * _MIP2_VOX + box_origin_nm
        post_nm = syn.post_pt.astype(np.float64) * _MIP2_VOX + box_origin_nm

        # Load skeletons for unique roots present on either side
        pre_roots = syn.pre_root_id.astype(np.int64)
        post_roots = syn.post_root_id.astype(np.int64)
        unique_roots = sorted({int(r) for r in pre_roots.tolist() + post_roots.tolist() if int(r) > 0})

        skel_cache: dict[int, tuple] = {}  # root_id -> (csr, verts)

        # Two supported layouts:
        #   1. Per-(root, version) files at <skeleton_dir>/v<version>/v<v>_rid<rid>_skv<skv>.npz
        #      (the CAVE skeleton service convention)
        #   2. Per-box archives at <skeleton_dir>/<box_hash>.npz
        #      (kimimaro self-skeletonization output, all roots in one file)
        # We try the per-box archive first; if absent, fall back to per-root files.
        self_archive = _P(skeleton_dir) / f"{box_hash}.npz"
        if self_archive.exists():
            try:
                box_skels = load_self_skeleton_archive(str(self_archive))
            except Exception:
                box_skels = {}
            for root_id, (verts, edges) in box_skels.items():
                if root_id not in unique_roots:
                    continue
                sk = SkeletonData(
                    root_id=root_id,
                    materialization_version=version,
                    vertices=verts,
                    edges=edges,
                    radius=None,
                )
                skel_cache[root_id] = _skeleton_to_csr(sk)
        else:
            skel_dir_v = _P(skeleton_dir) / f"v{version}"
            for root_id in unique_roots:
                skel_path = skel_dir_v / f"v{version}_rid{root_id}_skv{skeleton_service_version}.npz"
                if not skel_path.exists():
                    continue
                try:
                    d = np.load(skel_path, allow_pickle=False)
                    sk = SkeletonData(
                        root_id=root_id,
                        materialization_version=version,
                        vertices=d["vertices"].astype(np.float32),
                        edges=d["edges"].astype(np.int64),
                        radius=None,
                    )
                    skel_cache[root_id] = _skeleton_to_csr(sk)
                except Exception:
                    continue

        side_results = {"pre": {}, "post": {}}
        for side, positions, roots in (
            ("pre", pre_nm, pre_roots),
            ("post", post_nm, post_roots),
        ):
            tree = _cKDTree(positions)
            pairs = tree.query_pairs(r=proximity_radius_nm, output_type="ndarray")

            n_same_root = 0
            n_traced = 0
            n_total = 0
            for a, b in pairs:
                i, j = (int(a), int(b)) if a < b else (int(b), int(a))
                n_total += 1
                ra = int(roots[i])
                rb = int(roots[j])
                same_root = (ra > 0 and ra == rb)
                if same_root:
                    n_same_root += 1
                    if ra in skel_cache:
                        csr, verts = skel_cache[ra]
                        si = _nearest_vertex_idx(verts, positions[i])
                        di = _nearest_vertex_idx(verts, positions[j])
                        path = _skeleton_path_points(csr, verts, si, di, max_path_nm=max_path_nm)
                        if len(path) > 0:
                            length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                            side_results[side][(i, j)] = {
                                "path": path.tolist(),
                                "len_nm": length,
                                "same_root": True,
                            }
                            n_traced += 1
                            continue
                # Either cross-root, or same-root but no skeleton cached / no path
                side_results[side][(i, j)] = {
                    "path": [],
                    "len_nm": 0.0,
                    "same_root": same_root,
                }

            if verbose:
                print(
                    f"  [{rec_idx+1}/{len(records)}] {box_hash[:8]} {side}: "
                    f"{n_total} edges  {n_same_root} same-root  {n_traced} traced"
                )

        out[box_hash] = side_results

    return out


def save_skeleton_path_cache(cache: dict, path: str) -> None:
    """Save skeleton-path cache as a compressed numpy file (.npz with pickled blob).

    JSON would blow up — paths can be hundreds of vertices each.  We pickle
    the dict into a single numpy object so the file stays compact.
    """
    import pickle
    from pathlib import Path as _P
    _P(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Skeleton-path cache saved → {path}")


def load_skeleton_path_cache(path: str) -> dict:
    """Load a skeleton-path cache previously saved by :func:`save_skeleton_path_cache`."""
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def load_seg_score_cache(path: str) -> dict:
    """Load a seg-score cache from a JSON file saved by :func:`save_seg_score_cache`."""
    import json as _json
    with open(path) as f:
        raw = _json.load(f)
    # Convert "(i, j)" string keys back to tuple[int, int]
    result = {}
    for box_hash, sides in raw.items():
        result[box_hash] = {}
        for side, scores in sides.items():
            result[box_hash][side] = {}
            for k_str, v in scores.items():
                # k_str is "(i, j)" or "i,j"
                k_str = k_str.strip("() ")
                parts = [int(x) for x in k_str.split(",")]
                result[box_hash][side][(parts[0], parts[1])] = float(v)
    return result


def save_seg_score_cache(cache: dict, path: str) -> None:
    """Save a seg-score cache dict to JSON."""
    import json as _json
    from pathlib import Path as _P
    _P(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        _json.dump(cache, f)
    print(f"Seg score cache saved → {path}")


# ---------------------------------------------------------------------------
# 9. Sampling strategy: proofread-core with tangledness prioritisation
# ---------------------------------------------------------------------------

def score_box_tangledness(
    cache: "BoxCache",
    record: "BoxRecord",
) -> dict[str, float]:
    """Score a cached box for how "tangled" its root-ID structure is.

    Tangled boxes have many distinct roots sharing synapses in a small volume —
    exactly the hard cases where the CellGNN needs to learn.  Proofreader-edited
    regions (the proofread core) are rich in these.

    Metrics returned:
    - ``n_pre_roots``: number of distinct pre-side root IDs
    - ``n_post_roots``: number of distinct post-side root IDs
    - ``root_density``: (n_pre_roots + n_post_roots) / n_synapses
    - ``max_root_size``: largest root group (synapses) on either side
    - ``multi_root_fraction``: fraction of roots that own ≥2 synapses (on
      either side).  Higher = more mergeable structure.
    - ``tangledness``: composite score = multi_root_fraction * root_density * 100
    """
    try:
        _, synapses = cache.load(record, load_volume=False)
    except Exception:
        return {"tangledness": 0.0}

    pre_roots = np.asarray(synapses.pre_root_id, dtype=np.int64)
    post_roots = np.asarray(synapses.post_root_id, dtype=np.int64)
    n = len(pre_roots)
    if n == 0:
        return {"tangledness": 0.0}

    from collections import Counter
    pre_counts = Counter(pre_roots.tolist())
    post_counts = Counter(post_roots.tolist())

    n_pre = len(pre_counts)
    n_post = len(post_counts)
    root_density = (n_pre + n_post) / max(n, 1)

    max_root_size = max(
        max(pre_counts.values(), default=0),
        max(post_counts.values(), default=0),
    )

    # Fraction of roots with ≥2 synapses (mergeable groups)
    multi_pre = sum(1 for c in pre_counts.values() if c >= 2)
    multi_post = sum(1 for c in post_counts.values() if c >= 2)
    total_roots = n_pre + n_post
    multi_root_fraction = (multi_pre + multi_post) / max(total_roots, 1)

    tangledness = multi_root_fraction * root_density * 100.0

    return {
        "n_pre_roots": float(n_pre),
        "n_post_roots": float(n_post),
        "root_density": root_density,
        "max_root_size": float(max_root_size),
        "multi_root_fraction": multi_root_fraction,
        "tangledness": tangledness,
    }


def rank_boxes_by_tangledness(
    cache: "BoxCache",
    records: "list[BoxRecord] | None" = None,
    *,
    min_synapses: int = 10,
    min_positive_pairs: int = 2,
) -> "list[tuple[BoxRecord, dict[str, float]]]":
    """Rank cached boxes by tangledness, most tangled first.

    Returns list of (record, metrics) tuples sorted by descending tangledness.
    Filters out boxes with too few synapses or positive pairs.
    """
    if records is None:
        records = cache.all_records()

    scored = []
    for rec in records:
        if rec.n_synapses < min_synapses:
            continue
        if rec.n_positive_pairs < min_positive_pairs:
            continue
        metrics = score_box_tangledness(cache, rec)
        scored.append((rec, metrics))

    scored.sort(key=lambda x: x[1].get("tangledness", 0.0), reverse=True)
    return scored


def spatial_train_val_test_split(
    cache: "BoxCache",
    records: "list[BoxRecord]",
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    axis: int = 0,
    seed: int = 42,
) -> "dict[str, list[BoxRecord]]":
    """Split BoxRecords into train/val/test by spatial binning.

    Assigns each box to a spatial bin along ``axis`` (0=x, 1=y, 2=z) based
    on its center coordinate, then allocates bins to splits.  This ensures
    that nearby boxes (which may share neurons) stay in the same split,
    preventing data leakage.

    Falls back to shuffled splitting if there are too few distinct bins.

    Parameters
    ----------
    cache : BoxCache
    records : list of BoxRecord to split
    val_fraction, test_fraction : target fractions for val and test
    axis : spatial axis to bin along (0=x, 1=y, 2=z)
    seed : RNG seed for bin assignment when bins are tied

    Returns
    -------
    dict with keys "train", "val", "test", each a list of BoxRecord.
    """
    rng = np.random.default_rng(seed)

    if not records:
        return {"train": [], "val": [], "test": []}

    # Get center coordinates along the chosen axis
    centers = []
    for rec in records:
        c = rec.center_nm
        if isinstance(c, (list, tuple)) and len(c) > axis:
            centers.append(float(c[axis]))
        else:
            centers.append(0.0)
    centers = np.array(centers)

    # Determine number of bins (at least 3 for a proper split)
    n = len(records)
    n_bins = max(3, min(10, n // 3))

    # Assign boxes to bins via quantiles
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(centers, percentiles)
    bin_ids = np.digitize(centers, bin_edges[1:-1])  # [0, n_bins-1]

    # Assign bins to splits
    unique_bins = sorted(set(bin_ids.tolist()))
    rng.shuffle(unique_bins)

    n_val_bins = max(1, round(len(unique_bins) * val_fraction))
    n_test_bins = max(1, round(len(unique_bins) * test_fraction))
    n_train_bins = len(unique_bins) - n_val_bins - n_test_bins

    if n_train_bins < 1:
        # Fallback: shuffled split
        order = rng.permutation(n)
        n_val = max(1, int(n * val_fraction))
        n_test = max(1, int(n * test_fraction))
        return {
            "val": [records[i] for i in order[:n_val]],
            "test": [records[i] for i in order[n_val:n_val + n_test]],
            "train": [records[i] for i in order[n_val + n_test:]],
        }

    val_bins = set(unique_bins[:n_val_bins])
    test_bins = set(unique_bins[n_val_bins:n_val_bins + n_test_bins])

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for i, rec in enumerate(records):
        b = bin_ids[i]
        if b in val_bins:
            splits["val"].append(rec)
        elif b in test_bins:
            splits["test"].append(rec)
        else:
            splits["train"].append(rec)

    return splits


def select_cell_gnn_training_boxes(
    cache: "BoxCache",
    *,
    max_train: int = 200,
    max_val: int = 30,
    min_tangledness: float = 0.0,
    min_synapses: int = 10,
    min_positive_pairs: int = 2,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> "dict[str, list[BoxRecord]]":
    """Select and split boxes for CellGNN training, prioritising tangled ones.

    Pipeline:
    1. Score all boxes for tangledness.
    2. Filter by min_tangledness (0.0 keeps all).
    3. Spatial train/val/test split to prevent leakage.
    4. Cap train set at ``max_train`` (keeping most tangled).
    5. Cap val set at ``max_val``.

    Returns dict with "train", "val", "test" lists of BoxRecord.
    """
    ranked = rank_boxes_by_tangledness(
        cache,
        min_synapses=min_synapses,
        min_positive_pairs=min_positive_pairs,
    )

    if min_tangledness > 0:
        ranked = [(r, m) for r, m in ranked if m.get("tangledness", 0) >= min_tangledness]

    records = [r for r, _m in ranked]
    if not records:
        return {"train": [], "val": [], "test": []}

    splits = spatial_train_val_test_split(
        cache, records,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    # Sort each split by tangledness (most tangled first), then cap
    def _tangle_key(rec):
        for r, m in ranked:
            if r.box_hash == rec.box_hash:
                return m.get("tangledness", 0.0)
        return 0.0

    splits["train"].sort(key=_tangle_key, reverse=True)
    splits["val"].sort(key=_tangle_key, reverse=True)

    if max_train > 0 and len(splits["train"]) > max_train:
        splits["train"] = splits["train"][:max_train]
    if max_val > 0 and len(splits["val"]) > max_val:
        splits["val"] = splits["val"][:max_val]

    return splits


# ---------------------------------------------------------------------------
# 10. Grammar score extraction — bridge from pairwise grammar to edge features
# ---------------------------------------------------------------------------

def extract_grammar_scores(
    synapses: "SynapseTable",
    role: str,
    grammar_score_fn,
    *,
    proximity_radius_nm: float = 5000.0,
    path_feature_mode: str = DEFAULT_PATH_FEATURE_MODE,
) -> dict[tuple[int, int], float]:
    """Score nearby scaffold-group pairs with the grammar model.

    Groups synapses by scaffold (seg_id), featurizes each group's synapse
    positions into a path sequence, then scores all spatially nearby group
    pairs.  Returns a dict keyed by canonical synapse-index pair ``(i, j)``
    with ``i < j``, valued at the grammar merge logit.

    This is the bridge between the pairwise grammar and the CellGNN: grammar
    scores become edge features in ``build_synapse_graph``.

    Parameters
    ----------
    synapses : SynapseTable
    role : "pre" or "post"
    grammar_score_fn :
        Callable ``(left_seq, right_seq) -> float`` — the grammar model's
        merge scorer (e.g. from ``_load_shared_merge_score_fn``).
    proximity_radius_nm :
        Only score group pairs whose centroids are within this distance.
    path_feature_mode :
        Feature mode for ``featurize_path_points``.
    """
    if role == "pre":
        positions = synapses.pre_pt.copy().astype(np.float32)
        seg_ids = getattr(synapses, "pre_seg_id", None)
    else:
        positions = synapses.post_pt.copy().astype(np.float32)
        seg_ids = getattr(synapses, "post_seg_id", None)

    # Group synapses by scaffold seg_id
    groups: dict[int, list[int]] = {}
    if seg_ids is not None:
        for i, sid in enumerate(seg_ids):
            sid_int = int(sid)
            if sid_int > 0:
                groups.setdefault(sid_int, []).append(i)
    if not groups:
        return {}

    # Compute per-group centroids and sequences
    group_ids = sorted(groups.keys())
    centroids = {}
    sequences = {}
    iso_positions = positions * PATH_ISO[np.newaxis, :]
    for gid in group_ids:
        indices = groups[gid]
        pts = iso_positions[indices]
        centroids[gid] = pts.mean(axis=0)
        # Order points along principal axis for consistent featurization
        if len(pts) >= 2:
            centered = pts - pts.mean(axis=0, keepdims=True)
            try:
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                order = np.argsort(centered @ vh[0])
                pts = pts[order]
            except np.linalg.LinAlgError:
                pass
        sequences[gid] = featurize_path_points(
            pts, mode=path_feature_mode, iso_scale=PATH_ISO,
        )

    # Score nearby group pairs
    group_scores: dict[tuple[int, int], float] = {}
    for i_idx, gid_a in enumerate(group_ids):
        for gid_b in group_ids[i_idx + 1:]:
            dist = float(np.linalg.norm(centroids[gid_a] - centroids[gid_b]))
            if dist > proximity_radius_nm:
                continue
            seq_a = sequences[gid_a]
            seq_b = sequences[gid_b]
            if len(seq_a) == 0 or len(seq_b) == 0:
                continue
            score = float(grammar_score_fn(seq_a, seq_b))
            group_scores[(gid_a, gid_b)] = score

    # Map group-pair scores to synapse-pair scores
    synapse_scores: dict[tuple[int, int], float] = {}
    for (gid_a, gid_b), score in group_scores.items():
        for syn_a in groups[gid_a]:
            for syn_b in groups[gid_b]:
                key = (min(syn_a, syn_b), max(syn_a, syn_b))
                # Keep the best score if multiple group pairs cover the same synapse pair
                if key not in synapse_scores or score > synapse_scores[key]:
                    synapse_scores[key] = score

    return synapse_scores


# ---------------------------------------------------------------------------
# 11. CellGNN assembly — full alternative to beam-search pipeline
# ---------------------------------------------------------------------------

def cell_gnn_assembly(
    synapses: "SynapseTable",
    model,
    *,
    grammar_score_fn=None,
    synapse_hits: np.ndarray | None = None,
    proximity_radius_nm: float = 5000.0,
    partition_threshold: float = 0.5,
    partition_method: str = "agglomerative",
    path_feature_mode: str = DEFAULT_PATH_FEATURE_MODE,
    verbose: bool = False,
    use_boundary_search: bool = False,
    high_sim: float = 0.999,
    low_sim: float = 0.93,
    max_boundary_edges: int = 40,
    beam_width: int = 8,
    use_em_corridors: bool = False,
    corridor_radius_nm: float = 1500.0,
    corridor_accept_threshold: float = 0.8,
    corridor_reject_threshold: float = 0.2,
    seg_connectivity_scores: "dict[tuple[int, int], float] | None" = None,
) -> "ConnectivityGraph":
    """Run CellGNN to produce a ConnectivityGraph — drop-in alternative to _build_graph.

    Pipeline:
    1. Optionally extract grammar pairwise scores from scaffold groups.
    2. Build pre- and post-side SynapseGraphs with all available evidence.
    3. Run CellGNN inference to get per-synapse cell labels.
    4. Convert to ConnectivityGraph for F1 evaluation.

    Parameters
    ----------
    synapses : SynapseTable
    model : CellGNN (eval mode)
    grammar_score_fn :
        Optional grammar merge scorer.  When provided, grammar pairwise scores
        are computed between scaffold groups and fed as edge features.
    synapse_hits :
        Optional agent hit matrix [n_agents, n_synapses] for shared-agent edges.
    proximity_radius_nm :
        Spatial radius for evidence graph construction.
    partition_threshold :
        Cosine similarity threshold for clustering embeddings into cells.
    partition_method :
        "agglomerative" or "greedy".
    path_feature_mode :
        Feature mode for grammar score extraction.
    verbose :
        Print progress.
    use_boundary_search :
        If True, use :func:`boundary_partition_search` instead of
        :func:`infer_cells`.  Resolves ambiguous merge decisions in the
        similarity band ``[low_sim, high_sim)``.  Default False preserves
        backward-compatible behaviour.
    high_sim :
        Upper bound (exclusive) of the ambiguous similarity band used by
        boundary search.  Edges above this are unconditionally merged.
    low_sim :
        Lower bound (inclusive) of the ambiguous band.
    max_boundary_edges :
        Maximum number of boundary edges explored by the beam search.
    beam_width :
        Beam width for the beam search.
    use_em_corridors :
        If True, fetch EM corridor volumes for boundary edges and use the
        connectivity scores to force-accept or force-reject decisive cases
        before beam search.  Requires network access (CloudVolume).  Errors
        are caught and logged as warnings; the pipeline falls back to beam
        search without corridor scores.  Ignored when
        ``use_boundary_search=False``.
    corridor_radius_nm :
        Cylinder radius (nm) for EM corridor specifications.
    corridor_accept_threshold :
        EM score above which a boundary-edge merge is force-accepted.
    corridor_reject_threshold :
        EM score below which a boundary-edge merge is force-rejected.
    seg_connectivity_scores :
        Pre-computed EM segmentation corridor scores keyed by
        ``(min(i,j), max(i,j))``.  Produced by
        :func:`~neuronauts.em_corridor.batch_score_seg_connectivity`.
        When provided, injected as the ``seg_connectivity`` edge feature
        directly into the CellGNN input graph.  If None, all edges default
        to 0.5 (neutral).
    """
    import time as _time

    t0 = _time.time()

    # Extract grammar scores if model available
    pre_grammar_scores = None
    post_grammar_scores = None
    if grammar_score_fn is not None:
        if verbose:
            print("  CellGNN: extracting grammar scores …")
        pre_grammar_scores = extract_grammar_scores(
            synapses, "pre", grammar_score_fn,
            proximity_radius_nm=proximity_radius_nm,
            path_feature_mode=path_feature_mode,
        )
        post_grammar_scores = extract_grammar_scores(
            synapses, "post", grammar_score_fn,
            proximity_radius_nm=proximity_radius_nm,
            path_feature_mode=path_feature_mode,
        )
        if verbose:
            print(
                f"    pre: {len(pre_grammar_scores)} scored pairs  "
                f"post: {len(post_grammar_scores)} scored pairs"
            )

    # Build synapse graphs — pass opposite-side seg_ids for shared-partner feature
    post_seg_ids = getattr(synapses, "post_seg_id", None)
    pre_seg_ids = getattr(synapses, "pre_seg_id", None)
    pre_graph = build_synapse_graph(
        synapses, "pre",
        synapse_hits=synapse_hits,
        grammar_scores=pre_grammar_scores,
        proximity_radius_nm=proximity_radius_nm,
        partner_seg_ids=post_seg_ids,
        seg_connectivity_scores=seg_connectivity_scores,
    )
    post_graph = build_synapse_graph(
        synapses, "post",
        synapse_hits=synapse_hits,
        grammar_scores=post_grammar_scores,
        proximity_radius_nm=proximity_radius_nm,
        partner_seg_ids=pre_seg_ids,
        seg_connectivity_scores=seg_connectivity_scores,
    )

    if verbose:
        print(
            f"  CellGNN: pre graph {pre_graph.n_synapses} nodes / {len(pre_graph.edges)} edges  "
            f"post graph {post_graph.n_synapses} nodes / {len(post_graph.edges)} edges"
        )

    # Infer cell labels
    if not use_boundary_search:
        # --- Backward-compatible path ---
        pre_labels = infer_cells(
            model, pre_graph,
            threshold=partition_threshold,
            method=partition_method,
        )
        post_labels = infer_cells(
            model, post_graph,
            threshold=partition_threshold,
            method=partition_method,
        )
    else:
        # --- Boundary search path ---
        def _run_side(graph: SynapseGraph) -> np.ndarray:
            corridor_scores = None
            if use_em_corridors:
                boundary_edges = _get_boundary_edges(
                    model, graph,
                    low_sim=low_sim,
                    high_sim=high_sim,
                    max_boundary_edges=max_boundary_edges,
                )
                if boundary_edges:
                    try:
                        from .em_corridor import batch_score_boundary_edges as _bsbe
                        corridor_scores = _bsbe(
                            graph.node_positions,
                            boundary_edges,
                            radius_nm=corridor_radius_nm,
                            verbose=verbose,
                        )
                        if verbose:
                            print(
                                f"  [em_corridor] scored {len(corridor_scores)} "
                                f"boundary edges for {graph.role} graph"
                            )
                    except Exception as exc:
                        import warnings as _warnings
                        _warnings.warn(
                            f"[cell_gnn_assembly] EM corridor scoring failed for "
                            f"{graph.role} graph: {exc!r}. "
                            "Falling back to beam search without corridor scores.",
                            stacklevel=3,
                        )
                        corridor_scores = None
            return infer_cells_with_search(
                model, graph,
                high_threshold=high_sim,
                low_sim=low_sim,
                max_boundary_edges=max_boundary_edges,
                beam_width=beam_width,
                corridor_scores=corridor_scores,
                corridor_accept_threshold=corridor_accept_threshold,
                corridor_reject_threshold=corridor_reject_threshold,
            )

        pre_labels = _run_side(pre_graph)
        post_labels = _run_side(post_graph)

    n_pre_cells = len(set(pre_labels.tolist()))
    n_post_cells = len(set(post_labels.tolist()))

    # Convert to ConnectivityGraph
    cg = connectivity_graph_from_cell_labels(pre_labels, post_labels, synapses)

    if verbose:
        print(
            f"  CellGNN: {n_pre_cells} pre cells + {n_post_cells} post cells  "
            f"{len(cg.neurons)} neurons  {len(cg.edges)} edges  "
            f"{_time.time() - t0:.2f}s"
        )

    return cg
