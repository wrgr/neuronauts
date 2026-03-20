"""Shared multitask grammar model and training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from .merge import ConnectivityGraph

from .grammar import (
    DEFAULT_PATH_FEATURE_MODE,
    LEGACY_PATH_FEATURE_MODE,
    TorchArborEncoder,
    TorchMergeScorer,
    TorchPathEncoder,
    _require_torch,
    path_feature_dim,
)


class SharedGrammarModel:
    """Factory for a shared encoder with merge, atomicity, and bridge heads.

    The path encoder is now a Transformer-based model (see
    ``TorchPathEncoder``).  ``path_d_model`` controls the internal transformer
    width and replaces the old ``path_hidden_dim`` argument.  The legacy
    ``path_hidden_dim`` keyword is still accepted and forwarded as
    ``path_d_model`` so that existing call-sites and older checkpoints
    continue to work.

    Bridge head
    -----------
    ``predict_bridge(left_x, left_mask, right_x, right_mask)`` takes two
    padded path sequences and returns a ``[B, 6]`` tensor whose columns are:

        [:, :3]  predicted 3-D midpoint of the bridge in fragment-relative
                 coordinates (the point where the neurite crosses the void)
        [:, 3:]  predicted unit-direction tangent at the bridge midpoint

    The 6D output is supervised at training time with a self-supervised loss
    computed from the true adjacent-segment geometry (see
    ``multitask_train_step``).  At inference time the predicted midpoint and
    direction are used to score or parameterise Dijkstra bridge candidates
    built by ``neuronauts.dijkstra.BridgeGraph``.
    """

    def __new__(
        cls,
        *,
        input_dim: int | None = 3,
        path_d_model: int = 64,
        path_n_heads: int = 4,
        path_n_layers: int = 2,
        path_ffn_dim: int = 128,
        path_dropout: float = 0.1,
        embedding_dim: int = 32,
        merge_hidden_dim: int = 64,
        arbor_hidden_dim: int = 64,
        arbor_output_dim: int = 64,
        atomicity_hidden_dim: int = 64,
        bridge_hidden_dim: int = 64,
        path_feature_mode: str = DEFAULT_PATH_FEATURE_MODE,
        # Legacy alias: older checkpoints / scripts may pass path_hidden_dim.
        path_hidden_dim: int | None = None,
    ):
        torch, nn = _require_torch()
        _input_dim = int(path_feature_dim(path_feature_mode) if input_dim is None else input_dim)

        _path_d_model = int(path_d_model if path_hidden_dim is None else path_hidden_dim)

        class _SharedGrammarModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._init_kwargs = {
                    "input_dim": _input_dim,
                    "path_d_model": _path_d_model,
                    "path_n_heads": int(path_n_heads),
                    "path_n_layers": int(path_n_layers),
                    "path_ffn_dim": int(path_ffn_dim),
                    "path_dropout": float(path_dropout),
                    "embedding_dim": int(embedding_dim),
                    "merge_hidden_dim": int(merge_hidden_dim),
                    "arbor_hidden_dim": int(arbor_hidden_dim),
                    "arbor_output_dim": int(arbor_output_dim),
                    "atomicity_hidden_dim": int(atomicity_hidden_dim),
                    "bridge_hidden_dim": int(bridge_hidden_dim),
                    "path_feature_mode": str(path_feature_mode),
                }
                self.path_feature_mode = str(path_feature_mode)
                self.path_encoder = TorchPathEncoder(
                    input_dim=_input_dim,
                    d_model=_path_d_model,
                    n_heads=path_n_heads,
                    n_layers=path_n_layers,
                    ffn_dim=path_ffn_dim,
                    dropout=path_dropout,
                    output_dim=embedding_dim,
                )
                self.path_encoder.path_feature_mode = self.path_feature_mode
                self.merge_scorer = TorchMergeScorer(
                    embedding_dim=embedding_dim,
                    hidden_dim=merge_hidden_dim,
                )
                self.arbor_encoder = TorchArborEncoder(
                    embedding_dim=embedding_dim,
                    hidden_dim=arbor_hidden_dim,
                    output_dim=arbor_output_dim,
                )
                self.atomicity_head = nn.Sequential(
                    nn.Linear(arbor_output_dim, atomicity_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(atomicity_hidden_dim, 1),
                )
                # Bridge head: two fragment embeddings → 6D (midpoint + direction).
                # Input is the concatenation of both embeddings plus their
                # element-wise difference, giving the head explicit asymmetric
                # context about both sides of the potential void.
                self.bridge_head = nn.Sequential(
                    nn.Linear(embedding_dim * 3, int(bridge_hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(bridge_hidden_dim), int(bridge_hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(bridge_hidden_dim), 6),
                )

            def encode_paths(self, x, mask):
                return self.path_encoder(x, mask=mask)

            def score_merge(self, left_x, left_mask, right_x, right_mask):
                left = self.path_encoder(left_x, mask=left_mask)
                right = self.path_encoder(right_x, mask=right_mask)
                return self.merge_scorer(left, right)

            def score_atomicity(self, branch_x, branch_sequence_mask, branch_mask):
                batch_size, max_branches, max_steps, feat_dim = branch_x.shape
                flat_x = branch_x.view(batch_size * max_branches, max_steps, feat_dim)
                flat_mask = branch_sequence_mask.view(batch_size * max_branches, max_steps)
                branch_embeddings = self.path_encoder(flat_x, mask=flat_mask)
                branch_embeddings = branch_embeddings.view(batch_size, max_branches, -1)
                arbor_repr = self.arbor_encoder(branch_embeddings, mask=branch_mask)
                return self.atomicity_head(arbor_repr).squeeze(-1)

            def predict_bridge(self, left_x, left_mask, right_x, right_mask):
                """Predict a 6D bridge descriptor between two path fragments.

                Parameters
                ----------
                left_x, right_x:
                    Float tensors of shape ``[B, T, input_dim]``.
                left_mask, right_mask:
                    Bool tensors of shape ``[B, T]`` where ``True`` = PAD.

                Returns
                -------
                torch.Tensor
                    Shape ``[B, 6]``.  Columns ``[:, :3]`` are the predicted
                    bridge midpoint; ``[:, 3:]`` are the raw (un-normalised)
                    tangent direction.  Callers that need a unit vector should
                    apply ``F.normalize`` to the last 3 columns.
                """
                left = self.path_encoder(left_x, mask=left_mask)
                right = self.path_encoder(right_x, mask=right_mask)
                combined = torch.cat([left, right, left - right], dim=-1)
                return self.bridge_head(combined)

        return _SharedGrammarModel()


@dataclass(frozen=True)
class SharedTrainingConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    merge_loss_weight: float = 1.0
    atomicity_loss_weight: float = 1.0
    bridge_loss_weight: float = 0.5
    seed: int = 42


def multitask_train_step(
    model,
    optimizer,
    *,
    merge_batch: dict,
    topology_batch: dict,
    bridge_batch: dict | None = None,
    merge_loss_weight: float = 1.0,
    atomicity_loss_weight: float = 1.0,
    bridge_loss_weight: float = 0.5,
) -> dict[str, float]:
    """Run one multitask gradient step over merge, atomicity, and bridge tasks.

    Bridge batch format
    -------------------
    ``bridge_batch`` is optional.  When supplied it must contain:

    - ``"left_x"``  / ``"left_mask"``  : left fragment sequence + pad mask
    - ``"right_x"`` / ``"right_mask"`` : right fragment sequence + pad mask
    - ``"target_midpoint"``            : float ``[B, 3]`` ground-truth midpoint
    - ``"target_direction"``           : float ``[B, 3]`` ground-truth unit tangent

    The bridge loss is the sum of:

    1. Mean-squared error between predicted and true midpoints.
    2. 1 − cosine similarity between predicted and true directions
       (a smooth proxy for angular deviation that vanishes when the vectors
       are aligned).

    This is self-supervised in the sense that both targets can be derived
    from the geometry of known adjacent-segment pairs without additional
    human annotation.
    """
    import torch
    import torch.nn.functional as F

    model.train()
    optimizer.zero_grad()

    merge_logits = model.score_merge(
        merge_batch["left_x"],
        merge_batch["left_mask"],
        merge_batch["right_x"],
        merge_batch["right_mask"],
    )
    merge_loss = F.binary_cross_entropy_with_logits(merge_logits, merge_batch["y"].float())

    atomicity_logits = model.score_atomicity(
        topology_batch["branch_x"],
        topology_batch["branch_sequence_mask"],
        topology_batch["branch_mask"],
    )
    atomicity_loss = F.binary_cross_entropy_with_logits(atomicity_logits, topology_batch["y"].float())

    loss = merge_loss_weight * merge_loss + atomicity_loss_weight * atomicity_loss
    bridge_loss_val = 0.0

    if bridge_batch is not None and bridge_loss_weight > 0.0:
        bridge_pred = model.predict_bridge(
            bridge_batch["left_x"],
            bridge_batch["left_mask"],
            bridge_batch["right_x"],
            bridge_batch["right_mask"],
        )
        pred_midpoint = bridge_pred[:, :3]
        pred_direction = bridge_pred[:, 3:]

        target_midpoint = bridge_batch["target_midpoint"].float()
        target_direction = bridge_batch["target_direction"].float()

        midpoint_loss = F.mse_loss(pred_midpoint, target_midpoint)
        # Direction loss: 1 - cos_sim, averaged over batch.
        cos_sim = F.cosine_similarity(pred_direction, target_direction, dim=-1)
        direction_loss = (1.0 - cos_sim).mean()

        bridge_loss = midpoint_loss + direction_loss
        bridge_loss_val = float(bridge_loss.detach().cpu())
        loss = loss + bridge_loss_weight * bridge_loss

    loss.backward()
    optimizer.step()

    merge_acc = float(
        ((merge_logits.detach() >= 0.0).long() == merge_batch["y"].long())
        .float().mean().cpu()
    )
    atomicity_acc = float(
        ((atomicity_logits.detach() >= 0.0).long() == topology_batch["y"].long())
        .float().mean().cpu()
    )

    metrics: dict[str, float] = {
        "loss": float(loss.detach().cpu()),
        "merge_loss": float(merge_loss.detach().cpu()),
        "atomicity_loss": float(atomicity_loss.detach().cpu()),
        "bridge_loss": bridge_loss_val,
        "merge_accuracy": merge_acc,
        "atomicity_accuracy": atomicity_acc,
    }
    return metrics


class _SparseGATLayer:
    """Factory returning a single sparse Graph Attention layer (no PyG required).

    Implements the original GAT attention from Veličković et al. 2018 using
    scatter-add over an edge list, so it works on graphs of any size without
    materialising an N×N attention matrix.
    """

    def __new__(cls, in_dim: int, out_dim: int, n_heads: int, dropout: float = 0.1):
        torch, nn = _require_torch()
        import torch.nn.functional as F
        assert out_dim % n_heads == 0
        head_dim = out_dim // n_heads

        class _GAT(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = n_heads
                self.head_dim = head_dim
                # Shared linear projection.
                self.W = nn.Linear(in_dim, out_dim, bias=False)
                # Attention vector per head: applied to [Wh_i || Wh_j].
                self.attn = nn.Linear(2 * head_dim, 1, bias=False)
                self.dropout = nn.Dropout(dropout)

            def forward(self, h, src, dst):
                N = h.size(0)
                H, d = self.n_heads, self.head_dim

                Wh = self.W(h).view(N, H, d)       # [N, H, d]
                src_feat = Wh[src]                  # [E, H, d]
                dst_feat = Wh[dst]                  # [E, H, d]

                # Attention logits: LeakyReLU(a^T [Wh_src || Wh_dst]).
                e = F.leaky_relu(
                    self.attn(torch.cat([src_feat, dst_feat], dim=-1)).squeeze(-1),
                    negative_slope=0.2,
                )  # [E, H]

                # Numerically stable scatter-softmax: clamp to prevent overflow.
                exp_e = torch.exp(torch.clamp(e, max=20.0))
                sum_exp = torch.zeros(N, H, device=h.device, dtype=h.dtype)
                sum_exp.scatter_add_(0, dst.unsqueeze(1).expand(-1, H), exp_e)
                alpha = exp_e / (sum_exp[dst] + 1e-8)   # [E, H]
                alpha = self.dropout(alpha)

                # Weighted neighbour aggregation.
                weighted = alpha.unsqueeze(-1) * src_feat   # [E, H, d]
                out = torch.zeros(N, H, d, device=h.device, dtype=h.dtype)
                out.scatter_add_(
                    0,
                    dst.unsqueeze(1).unsqueeze(2).expand_as(weighted),
                    weighted,
                )
                return F.elu(out.view(N, H * d))    # [N, out_dim]

        return _GAT()


class GlobalAssemblyGAT:
    """Factory for a Global Assembly GAT replacing local beam search.

    Architecture
    ------------
    A 2-layer sparse GAT that takes one node per ``MergedNeuron`` fragment
    and an edge list of candidate merges.  Two graph-attention passes let
    every fragment gather global evidence before edge scores are computed.

    Pipeline
    --------
    1. ``input_proj``: project path-encoder embeddings to ``gat_dim``.
    2. N × ``_SparseGATLayer``: update node embeddings with neighbourhood
       attention.  Residual + LayerNorm applied after each layer.
    3. ``edge_scorer``: MLP over ``[h_u || h_v || h_u - h_v]`` → scalar
       logit for each candidate edge.  Sigmoid of the logit = merge prob.

    The module exposes two entry points:

    ``forward(x, src, dst) → h``
        Run GAT message-passing, return updated node embeddings ``[N, gat_dim]``.

    ``score_edges(h, src, dst) → logits``
        Score each edge in ``[E]`` given updated embeddings.

    Parameters
    ----------
    node_dim:
        Dimensionality of input node features (= path-encoder output_dim).
    gat_dim:
        Hidden width of each GAT layer.  Must be divisible by ``n_heads``.
    n_heads:
        Number of attention heads per GAT layer.
    n_layers:
        Number of stacked GAT layers.
    dropout:
        Dropout probability on attention weights.
    edge_score_hidden:
        Width of the hidden layer in the edge-scoring MLP.
    """

    def __new__(
        cls,
        *,
        node_dim: int = 32,
        gat_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        edge_score_hidden: int = 64,
    ):
        torch, nn = _require_torch()

        class _AssemblyGAT(nn.Module):
            def __init__(self):
                super().__init__()
                self._init_kwargs = {
                    "node_dim": node_dim,
                    "gat_dim": gat_dim,
                    "n_heads": n_heads,
                    "n_layers": n_layers,
                    "dropout": dropout,
                    "edge_score_hidden": edge_score_hidden,
                }
                self.input_proj = nn.Linear(node_dim, gat_dim)
                self.gat_layers = nn.ModuleList([
                    _SparseGATLayer(gat_dim, gat_dim, n_heads, dropout)
                    for _ in range(n_layers)
                ])
                self.norms = nn.ModuleList([
                    nn.LayerNorm(gat_dim) for _ in range(n_layers)
                ])
                self.edge_scorer = nn.Sequential(
                    nn.Linear(gat_dim * 3, edge_score_hidden),
                    nn.ReLU(),
                    nn.Linear(edge_score_hidden, 1),
                )

            def forward(self, x, src, dst):
                """Run GAT message passing.

                Parameters
                ----------
                x : Tensor [N, node_dim]
                    Initial node features (path-encoder embeddings).
                src, dst : Tensor [E] int64
                    Edge endpoints (directed; add self-loops externally if
                    desired).

                Returns
                -------
                Tensor [N, gat_dim]
                """
                h = self.input_proj(x)
                for layer, norm in zip(self.gat_layers, self.norms):
                    h = norm(h + layer(h, src, dst))
                return h

            def score_edges(self, h, src, dst):
                """Score each edge given updated node embeddings.

                Parameters
                ----------
                h : Tensor [N, gat_dim]
                src, dst : Tensor [E] int64

                Returns
                -------
                Tensor [E]  — raw logits (apply sigmoid for probabilities).
                """
                cat = torch.cat([h[src], h[dst], h[src] - h[dst]], dim=-1)
                return self.edge_scorer(cat).squeeze(-1)

        return _AssemblyGAT()


def save_global_assembly_gat(path: str | Path, model) -> None:
    """Persist a ``GlobalAssemblyGAT`` to disk."""
    torch, _ = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "init_kwargs": dict(getattr(model, "_init_kwargs", {})),
        },
        path,
    )


def load_global_assembly_gat(path: str | Path):
    """Load a ``GlobalAssemblyGAT`` checkpoint and return the model in eval mode."""
    torch, _ = _require_torch()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = GlobalAssemblyGAT(**checkpoint.get("init_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def save_shared_grammar_model(path: str | Path, model) -> None:
    torch, _ = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "init_kwargs": dict(getattr(model, "_init_kwargs", {})),
        },
        path,
    )


def load_shared_grammar_model(path: str | Path):
    torch, _ = _require_torch()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    init_kwargs = dict(checkpoint.get("init_kwargs", {}))
    if "path_feature_mode" not in init_kwargs:
        # Historical checkpoints with input_dim=3 were trained before feature
        # modes were explicit; assume the original heuristic triplet.
        if int(init_kwargs.get("input_dim", 3)) == 3:
            init_kwargs["path_feature_mode"] = LEGACY_PATH_FEATURE_MODE
        else:
            init_kwargs["path_feature_mode"] = DEFAULT_PATH_FEATURE_MODE
    model = SharedGrammarModel(**init_kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# GAT training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GATTrainingConfig:
    """Hyper-parameters for training a ``GlobalAssemblyGAT``.

    Data generation
    ---------------
    Training examples are synthetic ``ConnectivityGraph`` instances produced
    by running the agent simulation on small random volumes.  ``n_examples``
    and ``val_fraction`` control the total pool size.  ``volume_shape``
    should be small enough that each example generates quickly on CPU
    (32³ voxels takes ~ 0.5 s).

    Optimisation
    ------------
    The per-edge loss is a weighted combination of binary cross-entropy
    (penalises individual edge mistakes) and a differentiable soft-F1
    surrogate (aligns the optimisation objective directly with the terminal
    metric).  ``soft_f1_weight`` in ``[0, 1]`` trades off between the two.

    The path encoder is always frozen during GAT training — it acts as a
    feature extractor and its weights are not updated.
    """

    epochs: int = 30
    n_examples: int = 200
    val_fraction: float = 0.15
    learning_rate: float = 3e-4
    soft_f1_weight: float = 0.5
    seed: int = 42
    # Small volumes keep each example cheap (~ 0.5 s on CPU at 40³).
    # Must be large enough to accommodate ``anchor_margin`` (default 12) on
    # each side, so minimum useful value is 28 per axis.
    volume_shape: tuple = (40, 40, 40)
    n_synapses: int = 12


def gat_train_step(
    gat_model,
    path_encoder,
    optimizer,
    *,
    graph: "ConnectivityGraph",
    pre_root_ids: "np.ndarray",
    post_root_ids: "np.ndarray",
    soft_f1_weight: float = 0.5,
) -> dict[str, float]:
    """One gradient step for the GlobalAssemblyGAT on a single labeled graph.

    The path encoder is run in ``eval()`` mode and its parameters are not
    updated — it acts as a frozen feature extractor.  Only the GAT weights
    receive gradients.

    Loss
    ----
    ``total = (1 - w) * BCE  +  w * (1 - soft_F1)``

    where the soft-F1 is computed from sigmoid probabilities:

    ``soft_F1 = 2·TP / (2·TP + FP + FN + ε)``

    Parameters
    ----------
    gat_model:
        A ``GlobalAssemblyGAT`` module (in training mode).
    path_encoder:
        A ``TorchPathEncoder`` module — frozen, used only for node features.
    optimizer:
        A ``torch.optim`` optimizer managing only the GAT parameters.
    graph:
        A ``ConnectivityGraph`` produced by ``_build_graph``.
    pre_root_ids, post_root_ids:
        Ground-truth root ID arrays for synapse labelling.
    soft_f1_weight:
        Weight ``w`` of the soft-F1 term in the combined loss.

    Returns
    -------
    dict with keys ``loss``, ``bce_loss``, ``f1_loss``, ``n_edges``,
    ``n_pos``, ``pred_f1`` (hard threshold F1 on predicted probabilities).
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .assembly import _build_gat_edges, _encode_neurons, label_graph_edges
    from .merge import ConnectivityGraph

    if not graph.edges:
        return {"loss": 0.0, "bce_loss": 0.0, "f1_loss": 0.0,
                "n_edges": 0, "n_pos": 0, "pred_f1": 0.0}

    gat_model.train()
    path_encoder.eval()
    optimizer.zero_grad()

    # 1. Frozen node features from path encoder.
    with torch.no_grad():
        node_ids, h = _encode_neurons(graph.neurons, path_encoder)

    if len(node_ids) == 0:
        return {"loss": 0.0, "bce_loss": 0.0, "f1_loss": 0.0,
                "n_edges": 0, "n_pos": 0, "pred_f1": 0.0}

    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # 2. Build GAT edge list (self-loops + bidirectional synapse edges).
    src, dst, _ = _build_gat_edges(node_ids, graph)

    # 3. GAT forward pass — gradients flow here.
    h_gat = gat_model(h, src, dst)

    # 4. Batch-score all synapse edges in one call.
    edge_src = torch.tensor(
        [id_to_idx[pre_nid] for pre_nid, _, _ in graph.edges], dtype=torch.long
    )
    edge_dst = torch.tensor(
        [id_to_idx[post_nid] for _, post_nid, _ in graph.edges], dtype=torch.long
    )
    logits = gat_model.score_edges(h_gat, edge_src, edge_dst)   # [E]

    # 5. Ground-truth labels.
    labels_np = label_graph_edges(graph, pre_root_ids, post_root_ids)
    labels = torch.from_numpy(labels_np).float()
    n_pos = int(labels.sum().item())

    # 6. Binary cross-entropy.
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

    # 7. Differentiable soft-F1 surrogate.
    probs = torch.sigmoid(logits)
    tp = (probs * labels).sum()
    fp = (probs * (1.0 - labels)).sum()
    fn = ((1.0 - probs) * labels).sum()
    soft_f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    f1_loss = 1.0 - soft_f1

    total = (1.0 - soft_f1_weight) * bce_loss + soft_f1_weight * f1_loss
    total.backward()
    optimizer.step()

    # Hard-threshold F1 for logging (not used for gradients).
    with torch.no_grad():
        hard_pred = (probs >= 0.5).float()
        h_tp = float((hard_pred * labels).sum())
        h_fp = float((hard_pred * (1.0 - labels)).sum())
        h_fn = float(((1.0 - hard_pred) * labels).sum())
        pred_f1 = 2.0 * h_tp / max(2.0 * h_tp + h_fp + h_fn, 1e-8)

    return {
        "loss": float(total.detach()),
        "bce_loss": float(bce_loss.detach()),
        "f1_loss": float(f1_loss.detach()),
        "n_edges": len(graph.edges),
        "n_pos": n_pos,
        "pred_f1": pred_f1,
    }


def _generate_gat_example(
    volume_shape: tuple,
    n_synapses: int,
    seed: int,
):
    """Generate one labeled (ConnectivityGraph, pre_root_ids, post_root_ids) tuple.

    Uses the full synthetic benchmark pipeline on a small volume so examples
    are cheap to produce (< 1 s on CPU).
    """
    import numpy as np
    from .fetch import SyntheticBenchmarkConfig, make_test_volume
    from .run import HeuristicConfig, _build_graph, simulate_paths_and_hits
    from .fields import compute_membrane_field

    min_side = min(volume_shape)
    # Shrink anchor_margin so synapses fit inside small volumes.
    anchor_margin = max(2, min_side // 6)
    config = SyntheticBenchmarkConfig(
        shape=tuple(volume_shape),          # type: ignore[arg-type]
        n_synapses=n_synapses,
        anchor_margin=anchor_margin,
        min_neuron_groups=2,
        max_neuron_groups=5,
    )
    chunk, synapses = make_test_volume(config=config, seed=seed)
    mf = compute_membrane_field(chunk.data)
    path_arr, synapse_hits, path_lengths, _ = simulate_paths_and_hits(
        chunk.data,
        synapses.pre_pt,
        synapses.post_pt,
        seed=seed,
        verbose=False,
        membrane_field_override=mf,
    )
    # Use a wide-open HeuristicConfig so the graph captures as many candidate
    # edges as possible — the GAT will learn to filter them.
    graph = _build_graph(
        path_arr=path_arr,
        path_lengths=path_lengths,
        synapse_hits=synapse_hits,
        pre_pts=synapses.pre_pt,
        post_pts=synapses.post_pt,
        pre_seg_ids=synapses.pre_seg_id,
        post_seg_ids=synapses.post_seg_id,
        heuristic_config=HeuristicConfig.learned(),
    )
    return graph, synapses.pre_root_id, synapses.post_root_id


def train_global_assembly_gat(
    path_encoder,
    gat_model,
    output_path: "str | Path",
    config: "GATTrainingConfig | None" = None,
) -> dict[str, list[float]]:
    """Train a ``GlobalAssemblyGAT`` using synthetic ConnectivityGraph examples.

    The path encoder is frozen throughout.  A pool of synthetic examples is
    generated once before training begins; training then iterates over this
    pool for ``config.epochs`` passes.

    Checkpointing
    -------------
    The model with the best validation soft-F1 is saved to ``output_path``
    using ``save_global_assembly_gat``.

    Parameters
    ----------
    path_encoder:
        A ``TorchPathEncoder`` (frozen feature extractor).
    gat_model:
        A ``GlobalAssemblyGAT`` module to train.
    output_path:
        Path for the best-validation checkpoint.
    config:
        Training hyper-parameters.  Defaults to ``GATTrainingConfig()``.

    Returns
    -------
    dict with keys ``train_loss``, ``val_loss``, ``train_f1``, ``val_f1`` —
    one float per epoch.
    """
    import numpy as np
    import torch

    config = config or GATTrainingConfig()
    torch_mod, _ = _require_torch()
    rng = np.random.default_rng(config.seed)

    # ── 1. Generate example pool ──────────────────────────────────────────
    print(f"Generating {config.n_examples} synthetic training examples …")
    examples = []
    for i in range(config.n_examples):
        seed_i = int(rng.integers(0, 2**31))
        try:
            graph, pre_root_ids, post_root_ids = _generate_gat_example(
                config.volume_shape, config.n_synapses, seed_i
            )
            if graph.edges:
                examples.append((graph, pre_root_ids, post_root_ids))
        except Exception:
            pass  # skip failed examples silently

    if not examples:
        raise RuntimeError("No valid training examples could be generated.")

    n_val = max(1, int(len(examples) * config.val_fraction))
    rng.shuffle(examples)          # type: ignore[arg-type]
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    print(f"  {len(train_examples)} train / {len(val_examples)} val examples "
          f"({sum(len(g.edges) for g, *_ in train_examples)} train edges)")

    optimizer = torch_mod.optim.Adam(gat_model.parameters(), lr=config.learning_rate)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []
    }
    best_val_f1 = -1.0

    for epoch in range(config.epochs):
        # ── Training pass ─────────────────────────────────────────────────
        rng.shuffle(train_examples)   # type: ignore[arg-type]
        train_metrics: list[dict] = []
        for graph, pre_root_ids, post_root_ids in train_examples:
            m = gat_train_step(
                gat_model, path_encoder, optimizer,
                graph=graph,
                pre_root_ids=pre_root_ids,
                post_root_ids=post_root_ids,
                soft_f1_weight=config.soft_f1_weight,
            )
            if m["n_edges"] > 0:
                train_metrics.append(m)

        # ── Validation pass ───────────────────────────────────────────────
        gat_model.eval()
        path_encoder.eval()
        val_metrics: list[dict] = []
        with torch_mod.no_grad():
            for graph, pre_root_ids, post_root_ids in val_examples:
                if not graph.edges:
                    continue
                from .assembly import _build_gat_edges, _encode_neurons, label_graph_edges
                import torch.nn.functional as F
                node_ids, h = _encode_neurons(graph.neurons, path_encoder)
                if not node_ids:
                    continue
                id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
                src, dst, _ = _build_gat_edges(node_ids, graph)
                h_gat = gat_model(h, src, dst)
                edge_src = torch_mod.tensor(
                    [id_to_idx[pre_nid] for pre_nid, _, _ in graph.edges],
                    dtype=torch_mod.long,
                )
                edge_dst = torch_mod.tensor(
                    [id_to_idx[post_nid] for _, post_nid, _ in graph.edges],
                    dtype=torch_mod.long,
                )
                logits = gat_model.score_edges(h_gat, edge_src, edge_dst)
                labels_np = label_graph_edges(graph, pre_root_ids, post_root_ids)
                labels = torch_mod.from_numpy(labels_np).float()

                bce = float(F.binary_cross_entropy_with_logits(logits, labels))
                probs = torch_mod.sigmoid(logits)
                tp = float((probs * labels).sum())
                fp = float((probs * (1.0 - labels)).sum())
                fn = float(((1.0 - probs) * labels).sum())
                sf1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1e-8)
                val_metrics.append({"loss": bce, "f1": sf1, "n_edges": len(graph.edges)})

        def _mean(ms, key):
            vals = [m[key] for m in ms if m.get("n_edges", 1) > 0]
            return float(np.mean(vals)) if vals else 0.0

        t_loss = _mean(train_metrics, "loss")
        t_f1   = _mean(train_metrics, "pred_f1")
        v_loss = _mean(val_metrics,   "loss")
        v_f1   = _mean(val_metrics,   "f1")

        history["train_loss"].append(t_loss)
        history["train_f1"].append(t_f1)
        history["val_loss"].append(v_loss)
        history["val_f1"].append(v_f1)

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            save_global_assembly_gat(output_path, gat_model)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}/{config.epochs} | "
                  f"train loss={t_loss:.4f} f1={t_f1:.3f} | "
                  f"val loss={v_loss:.4f} f1={v_f1:.3f} | "
                  f"best_val_f1={best_val_f1:.3f}")

        gat_model.train()

    print(f"Training complete. Best val soft-F1={best_val_f1:.4f} → {output_path}")
    return history
