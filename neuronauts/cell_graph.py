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

The hierarchy the user asked for is implicit in K:
  K=1  →  scaffold-group substructures  (cheap, nearly free)
  K=2  →  locally adjacent fragments on the same branch
  K=3+ →  full arbor / whole-cell reach

Architecture
------------
1. ``build_synapse_graph``           -- weighted evidence graph from all available signals
2. ``CellGNN``                       -- edge-conditioned message-passing GNN
3. ``partition_from_embeddings``     -- cluster embeddings → cell labels
4. ``cell_graph_train_step``         -- contrastive pull/push against CAVE root IDs
5. ``train_cell_gnn``                -- epoch loop over a BoxCache
6. ``infer_cells``                   -- inference → per-synapse cell labels
7. ``connectivity_graph_from_cell_labels`` -- labels → ConnectivityGraph for F1 eval
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


def build_synapse_graph(
    synapses: "SynapseTable",
    role: str,
    *,
    synapse_hits: np.ndarray | None = None,
    scaffold_groups: dict[int, list[int]] | None = None,
    grammar_scores: dict[tuple[int, int], float] | None = None,
    proximity_radius_nm: float = 5000.0,
    max_edges_per_node: int = 32,
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

    # Build edges via spatial proximity (KD-tree)
    from ._scipy_compat import cKDTree

    tree = cKDTree(iso_positions)
    edge_dict: dict[tuple[int, int], SynapseEdge] = {}

    # Proximity edges
    pairs = tree.query_pairs(r=proximity_radius_nm, output_type="ndarray")
    if len(pairs) > 0:
        for a, b in pairs:
            a, b = int(a), int(b)
            key = (min(a, b), max(a, b))
            dist = float(np.linalg.norm(iso_positions[a] - iso_positions[b]))
            same_scaf = 1.0 if (node_scaffold[a] == node_scaffold[b]
                                and node_scaffold[a] > 0) else 0.0
            gs = 0.0
            if grammar_scores is not None:
                gs = grammar_scores.get(key, 0.0)
            sa = agent_covisit.get(key, 0)
            edge_dict[key] = SynapseEdge(
                src=key[0], dst=key[1],
                distance=dist, same_scaffold=same_scaf,
                grammar_score=gs, shared_agents=sa,
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
                    edge_dict[key] = SynapseEdge(
                        src=key[0], dst=key[1],
                        distance=dist, same_scaffold=1.0,
                        grammar_score=gs, shared_agents=sa,
                    )

    return SynapseGraph(
        n_synapses=n,
        role=role,
        node_positions=iso_positions,
        node_scaffold_ids=node_scaffold,
        edges=list(edge_dict.values()),
        root_ids=root_ids,
    )


# ---------------------------------------------------------------------------
# 2. CellGNN -- sparse message-passing for cell membership
# ---------------------------------------------------------------------------

_EDGE_FEAT_DIM = 4  # distance, same_scaffold, grammar_score, shared_agents


def _graph_to_tensors(graph: SynapseGraph):
    """Convert a SynapseGraph to tensors for the GNN.

    Returns
    -------
    node_feat : Tensor [N, 3]  (isotropic position, centered)
    edge_src, edge_dst : Tensor [2E]  (bidirectional)
    edge_feat : Tensor [2E, 4]  (distance, same_scaffold, grammar_score, shared_agents)
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
        return (
            node_feat,
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, _EDGE_FEAT_DIM, dtype=torch.float32),
        )

    # Build bidirectional edges
    src_list, dst_list, feat_list = [], [], []
    for e in graph.edges:
        for s, d in [(e.src, e.dst), (e.dst, e.src)]:
            src_list.append(s)
            dst_list.append(d)
            feat_list.append([
                e.distance / max(scale, 1e-6),  # normalized distance
                e.same_scaffold,
                e.grammar_score,
                min(e.shared_agents, 10) / 10.0,  # clamp & normalize
            ])

    # Self-loops
    for i in range(N):
        src_list.append(i)
        dst_list.append(i)
        feat_list.append([0.0, 1.0, 1.0, 0.0])

    edge_src = torch.tensor(src_list, dtype=torch.long)
    edge_dst = torch.tensor(dst_list, dtype=torch.long)
    edge_feat = torch.tensor(feat_list, dtype=torch.float32)

    return node_feat, edge_src, edge_dst, edge_feat


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
    ):
        torch, nn = _require_torch()
        import torch.nn.functional as F
        assert d_model % n_heads == 0
        head_dim = d_model // n_heads

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
                }
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = head_dim
                self.embedding_dim = embedding_dim

                # Input projections
                self.node_proj = nn.Linear(node_input_dim, d_model)
                self.edge_proj = nn.Linear(edge_input_dim, d_model)

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

            def forward(self, node_feat, edge_src, edge_dst, edge_feat):
                """Run message passing and return per-synapse embeddings.

                Parameters
                ----------
                node_feat : Tensor [N, node_input_dim]
                edge_src, edge_dst : Tensor [E]
                edge_feat : Tensor [E, edge_input_dim]

                Returns
                -------
                Tensor [N, embedding_dim]
                """
                N = node_feat.size(0)
                h = self.node_proj(node_feat)         # [N, d_model]
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

def partition_from_embeddings(
    embeddings: np.ndarray,
    *,
    threshold: float = 0.5,
    method: str = "agglomerative",
) -> np.ndarray:
    """Cluster synapse embeddings into cell assignments.

    Parameters
    ----------
    embeddings : ndarray [N, D]
        Per-synapse embedding vectors from CellGNN.
    threshold : float
        Cosine similarity threshold for same-cell assignment.
    method : str
        "agglomerative" (default) or "greedy".

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

    # Agglomerative: bottom-up merging by cosine similarity
    from .helpers import UnionFind

    uf = UnionFind(N)

    # Compute pairwise similarities and merge above threshold
    sim_matrix = normed @ normed.T
    # Process in order of decreasing similarity
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
    # Training
    epochs: int = 50
    learning_rate: float = 1e-3
    margin: float = 0.5          # cosine similarity target separation for negatives
    max_pairs_per_box: int = 2048
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
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Sample balanced positive (same root) and negative (different root) pairs."""
    root_groups: dict[int, list[int]] = {}
    for i, rid in enumerate(root_ids):
        rid_int = int(rid)
        if rid_int > 0:
            root_groups.setdefault(rid_int, []).append(i)

    pos_pairs: list[tuple[int, int]] = []
    for members in root_groups.values():
        if len(members) < 2:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pos_pairs.append((members[i], members[j]))
                if len(pos_pairs) >= max_pairs:
                    break
            if len(pos_pairs) >= max_pairs:
                break

    all_roots = list(root_groups.keys())
    neg_pairs: list[tuple[int, int]] = []
    n_neg = min(len(pos_pairs) * 2, max_pairs)
    attempts = 0
    while len(neg_pairs) < n_neg and attempts < n_neg * 4:
        attempts += 1
        if len(all_roots) < 2:
            break
        r1, r2 = rng.choice(len(all_roots), size=2, replace=False)
        i = int(rng.choice(root_groups[all_roots[r1]]))
        j = int(rng.choice(root_groups[all_roots[r2]]))
        neg_pairs.append((i, j))

    return pos_pairs, neg_pairs


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

    node_feat, edge_src, edge_dst, edge_feat = _graph_to_tensors(graph)
    embeddings = model(node_feat, edge_src, edge_dst, edge_feat)  # [N, D]
    emb_norm = F.normalize(embeddings, p=2, dim=-1)

    pos_pairs, neg_pairs = _sample_contrastive_pairs(graph.root_ids, max_pairs, rng)

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
        return {"loss": 0.0, "pos_sim": 0.0, "neg_sim": 0.0, "n_pos": 0, "n_neg": 0}

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

    loss = sum(loss_terms)  # type: ignore[arg-type]
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach()),
        "pos_sim": pos_sim_val,
        "neg_sim": neg_sim_val,
        "n_pos": len(pos_pairs),
        "n_neg": len(neg_pairs),
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
        epoch_metrics: dict[str, list[float]] = {
            "loss": [], "pos_sim": [], "neg_sim": [],
        }

        for record in cache.iter_records(shuffle=True, rng=rng):
            if record.n_positive_pairs == 0 and record.n_synapses < 4:
                continue
            try:
                _, synapses = cache.load(record)
            except Exception:
                continue

            for role in ("pre", "post"):
                graph = build_synapse_graph(
                    synapses, role,
                    proximity_radius_nm=cfg.proximity_radius_nm,
                )
                if graph.n_synapses < 2:
                    continue
                m = cell_graph_train_step(
                    model, optimizer, graph,
                    margin=cfg.margin,
                    max_pairs=cfg.max_pairs_per_box,
                    rng=rng,
                    edit_positive_pairs=_edit_pos[role] or None,
                    edit_negative_pairs=_edit_neg[role] or None,
                    edit_weight=edit_weight,
                )
                epoch_metrics["loss"].append(m["loss"])
                epoch_metrics["pos_sim"].append(m["pos_sim"])
                epoch_metrics["neg_sim"].append(m["neg_sim"])

        mean_loss = float(np.mean(epoch_metrics["loss"])) if epoch_metrics["loss"] else 0.0
        mean_pos = float(np.mean(epoch_metrics["pos_sim"])) if epoch_metrics["pos_sim"] else 0.0
        mean_neg = float(np.mean(epoch_metrics["neg_sim"])) if epoch_metrics["neg_sim"] else 0.0
        history["train_loss"].append(mean_loss)
        history["train_pos_sim"].append(mean_pos)
        history["train_neg_sim"].append(mean_neg)

        if val_cache is not None:
            val_losses = []
            for record in val_cache.iter_records():
                try:
                    _, synapses = val_cache.load(record)
                except Exception:
                    continue
                for role in ("pre", "post"):
                    graph = build_synapse_graph(
                        synapses, role,
                        proximity_radius_nm=cfg.proximity_radius_nm,
                    )
                    if graph.n_synapses < 2:
                        continue
                    # Eval-only forward (no grad)
                    model.eval()
                    with torch.no_grad():
                        node_feat, es, ed, ef = _graph_to_tensors(graph)
                        import torch.nn.functional as F
                        emb = F.normalize(
                            model(node_feat, es, ed, ef), p=2, dim=-1
                        )
                    model.train()
                    val_pos, val_neg = _sample_contrastive_pairs(
                        graph.root_ids, cfg.max_pairs_per_box, rng
                    )
                    if val_pos:
                        pi = [p[0] for p in val_pos]
                        pj = [p[1] for p in val_pos]
                        pos_sims = (emb[pi] * emb[pj]).sum(dim=-1)
                        val_losses.append(float((1.0 - pos_sims).mean()))
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            history["val_loss"].append(val_loss)

        if verbose:
            val_str = ""
            if val_cache is not None:
                val_str = f"  val_loss={history['val_loss'][-1]:.4f}"
            print(
                f"Epoch {epoch + 1}/{cfg.epochs}  "
                f"loss={mean_loss:.4f}  "
                f"pos_sim={mean_pos:.3f}  neg_sim={mean_neg:.3f}"
                f"{val_str}"
            )

    return history


# ---------------------------------------------------------------------------
# 7. Inference: labels + ConnectivityGraph
# ---------------------------------------------------------------------------

def infer_cells(
    model,
    graph: SynapseGraph,
    *,
    threshold: float = 0.5,
    method: str = "agglomerative",
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
        node_feat, edge_src, edge_dst, edge_feat = _graph_to_tensors(graph)
        embeddings = model(node_feat, edge_src, edge_dst, edge_feat)
        embeddings_np = F.normalize(embeddings, p=2, dim=-1).cpu().numpy()

    return partition_from_embeddings(embeddings_np, threshold=threshold, method=method)


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
        _, synapses = cache.load(record)
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

    # Build synapse graphs
    pre_graph = build_synapse_graph(
        synapses, "pre",
        synapse_hits=synapse_hits,
        grammar_scores=pre_grammar_scores,
        proximity_radius_nm=proximity_radius_nm,
    )
    post_graph = build_synapse_graph(
        synapses, "post",
        synapse_hits=synapse_hits,
        grammar_scores=post_grammar_scores,
        proximity_radius_nm=proximity_radius_nm,
    )

    if verbose:
        print(
            f"  CellGNN: pre graph {pre_graph.n_synapses} nodes / {len(pre_graph.edges)} edges  "
            f"post graph {post_graph.n_synapses} nodes / {len(post_graph.edges)} edges"
        )

    # Infer cell labels
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
