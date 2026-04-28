"""Path-based edge feature encoder for the CellGNN (Option 2 model plane).

The CellGNN currently consumes 6 scalar edge features
(distance, same_scaffold, grammar_score, shared_agents,
shared_partners, seg_connectivity).  The scalar ``grammar_score``
collapses an entire skeleton path between two synapses to a single
float — losing the topological character of the path (does it bend
through a branch point?  does it stay straight?  is it long-and-curved?).

This module provides :class:`PathEdgeEncoder`, a Transformer-based encoder
that consumes the **full skeleton path between two synapses** as a sequence
of per-step features, producing a fixed-size edge embedding.  The embedding
captures tree-topology cues that a scalar score cannot:

* short, straight path → embedding near a "same branch" prototype
* long path passing through a branch-point detour → distinct embedding
* empty path (cross-cell or skeleton missing) → encoded via the
  ``no_path_token`` learnable parameter

The encoder is designed to be trained jointly with the CellGNN under
the existing root-ID contrastive loss — no separate pre-training.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .grammar import _require_torch


# Per-step path feature dimension matches `featurize_path_points`'s
# default mode "raw_delta3+skeleton" (3 delta dims + 3 skeleton dims = 6).
PATH_STEP_FEAT_DIM = 6


class PathEdgeEncoder:
    """Factory for a Transformer-based edge encoder over skeleton paths.

    Architecture
    ------------
    1. Linear projection of per-step features (``input_dim``) to ``d_model``.
    2. Learned ``[CLS]`` token prepended to each sequence.
    3. Sinusoidal positional encodings.
    4. ``n_layers`` of ``nn.TransformerEncoderLayer``.
    5. Output head: ``[CLS]`` -> ``output_dim`` linear projection.
    6. **Empty-path handling**: when an edge has no skeleton path
       (``[T = 0]`` or all-padding), the encoder returns a learned
       ``no_path_embedding`` vector instead of running the Transformer.
       This keeps gradient flow well-behaved and lets the model learn
       a distinct representation for "no skeleton evidence" edges.

    The output is meant to be **concatenated** with the scalar edge
    features (or replace ``grammar_score``) before the CellGNN's
    ``edge_proj`` linear layer.

    Parameters
    ----------
    input_dim:
        Per-step feature dimension.  Default 6
        (``raw_delta3+skeleton`` from ``featurize_path_points``).
    d_model:
        Internal transformer dimension.
    n_heads:
        Attention heads.
    n_layers:
        Stacked encoder layers.
    output_dim:
        Output edge embedding size.  Concatenated alongside the existing
        6 scalar edge features in CellGNN.
    max_len:
        Max path length (positional-encoding cap).  Longer paths are
        truncated by the caller.
    dropout:
        Dropout probability inside the transformer.
    """

    def __new__(
        cls,
        *,
        input_dim: int = PATH_STEP_FEAT_DIM,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        output_dim: int = 16,
        max_len: int = 64,
        dropout: float = 0.1,
        ffn_dim: int = 64,
    ):
        torch, nn = _require_torch()
        import math

        # Clamp n_heads to divide d_model
        _n_heads = int(n_heads)
        _d_model = int(d_model)
        while _d_model % _n_heads != 0 and _n_heads > 1:
            _n_heads -= 1

        class _PathEdgeEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.d_model = _d_model
                self.output_dim = int(output_dim)
                self.max_len = int(max_len)
                self._init_kwargs = {
                    "input_dim": self.input_dim,
                    "d_model": self.d_model,
                    "n_heads": _n_heads,
                    "n_layers": int(n_layers),
                    "output_dim": self.output_dim,
                    "max_len": self.max_len,
                    "dropout": float(dropout),
                    "ffn_dim": int(ffn_dim),
                }

                self.input_proj = nn.Linear(self.input_dim, self.d_model)

                # Learned [CLS] for global summary.
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
                nn.init.trunc_normal_(self.cls_token, std=0.02)

                # Sinusoidal positional encodings (CLS at position 0).
                pe = torch.zeros(self.max_len + 1, self.d_model)
                position = torch.arange(0, self.max_len + 1, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, self.d_model, 2, dtype=torch.float)
                    * (-math.log(10000.0) / self.d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term[: self.d_model // 2])
                self.register_buffer("pos_enc", pe.unsqueeze(0))

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=_n_heads,
                    dim_feedforward=int(ffn_dim),
                    dropout=float(dropout),
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=int(n_layers),
                    enable_nested_tensor=False,
                )
                self.output_proj = nn.Linear(self.d_model, self.output_dim)

                # Learned embedding returned for edges with no skeleton path.
                self.no_path_embedding = nn.Parameter(torch.zeros(self.output_dim))
                nn.init.trunc_normal_(self.no_path_embedding, std=0.02)

            def forward(
                self,
                path_seq,        # [E, T, input_dim] float
                path_mask,       # [E, T] bool — True = padding
                has_path,        # [E] bool — True if the edge has any non-pad steps
            ):
                """Encode each edge's skeleton path into a fixed-size vector.

                Parameters
                ----------
                path_seq : Tensor [E, T, input_dim] float
                path_mask : Tensor [E, T] bool — True at padded positions
                has_path : Tensor [E] bool — True if the edge has a real path

                Returns
                -------
                Tensor [E, output_dim]
                """
                E, T, _ = path_seq.shape

                # Start with the learned no-path embedding for all edges.
                # Only edges with real paths go through the transformer,
                # keeping batch size at ~P (typically <<E) rather than E.
                result = self.no_path_embedding.unsqueeze(0).expand(E, -1).contiguous()

                path_indices = has_path.nonzero(as_tuple=False).view(-1)
                if len(path_indices) == 0 or T == 0:
                    return result

                P = path_indices.shape[0]
                ps = path_seq[path_indices]    # [P, T, input_dim]
                pm = path_mask[path_indices]   # [P, T]

                tokens = self.input_proj(ps.float())          # [P, T, d_model]
                cls = self.cls_token.expand(P, -1, -1)        # [P, 1, d_model]
                tokens = torch.cat([cls, tokens], dim=1)      # [P, T+1, d_model]
                tokens = tokens + self.pos_enc[:, : T + 1, :]

                cls_mask = torch.zeros(P, 1, dtype=torch.bool, device=ps.device)
                full_mask = torch.cat([cls_mask, pm], dim=1)  # [P, T+1]

                out = self.transformer(tokens, src_key_padding_mask=full_mask)
                cls_out = out[:, 0, :]                        # [P, d_model]
                emb = self.output_proj(cls_out)               # [P, output_dim]

                # Scatter back: only the P path-having edges are updated
                result = result.clone()
                result[path_indices] = emb
                return result

        return _PathEdgeEncoder()


def pad_path_sequences(
    paths: list[np.ndarray],
    *,
    max_len: int = 64,
    feat_dim: int = PATH_STEP_FEAT_DIM,
) -> "tuple":
    """Pad a list of variable-length path-feature sequences for a batch.

    Empty paths (length 0) become all-padding rows; ``has_path`` is False
    for those rows.

    Parameters
    ----------
    paths:
        List of arrays of shape ``[T_i, feat_dim]`` or empty ``[0, feat_dim]``.
    max_len:
        Truncate each path to this length (taking first ``max_len`` steps).
    feat_dim:
        Per-step feature dim (must match every non-empty path).

    Returns
    -------
    (path_seq, path_mask, has_path)
        path_seq : np.ndarray [E, max_len, feat_dim] float32
        path_mask : np.ndarray [E, max_len] bool   (True = padding)
        has_path : np.ndarray [E] bool             (True if any real step)
    """
    E = len(paths)
    path_seq = np.zeros((E, max_len, feat_dim), dtype=np.float32)
    path_mask = np.ones((E, max_len), dtype=bool)
    has_path = np.zeros((E,), dtype=bool)

    for i, p in enumerate(paths):
        if p is None or len(p) == 0:
            continue
        arr = np.asarray(p, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != feat_dim:
            continue
        T = min(arr.shape[0], max_len)
        path_seq[i, :T, :] = arr[:T]
        path_mask[i, :T] = False
        has_path[i] = True

    return path_seq, path_mask, has_path
