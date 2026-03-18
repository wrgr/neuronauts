"""Shared connectome grammar models.

This module holds the main coordinate-free representation surfaces used by the
topology dataset path and, eventually, the global assembly path:

- ``PathEncoder``: sequential path-profile encoder
- ``MergeScorer``: pairwise fragment compatibility
- ``ArborEncoder``: cluster/arbor summarizer over multiple fragments

The implementation here remains lightweight and numpy-based so the package
stays runnable without optional training dependencies. The topology training
stack in ``neuronauts.topology_model`` consumes the embeddings exported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PathBatch:
    """Coordinate-free path descriptors for one or more candidate fragments.

    The core fields are `(edge_len, radius, curvature)` sequences. Optional
    `skeleton_feat` and `mesh_feat` arrays allow future callers to attach
    richer per-step descriptors without changing the baseline encoder API.
    """

    edge_len: np.ndarray
    radius: np.ndarray
    curvature: np.ndarray
    skeleton_feat: np.ndarray | None = None
    mesh_feat: np.ndarray | None = None


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("pip install torch or pip install -e .[topology]") from exc
    return torch, nn


class PathEncoder:
    """Sequential path encoder over raw path descriptors.

    Rather than collapsing an entire path into one global mean/std summary, the
    encoder keeps coarse beginning/middle/end structure by splitting the
    sequence into thirds and summarizing each segment independently. It also
    preserves explicit endpoint state so merge and atomicity heads can access
    terminal cues instead of reconstructing them from pooled chunk statistics.
    A compact whole-path summary is retained as well so downstream consumers do
    not need to infer global caliber/curvature trends only from chunked views.
    """

    def __init__(self, output_dim: int = 32) -> None:
        self.output_dim = int(output_dim)

    def encode(self, batch: PathBatch) -> np.ndarray:
        if batch.edge_len.size == 0:
            return np.zeros(self.output_dim, dtype=np.float32)

        stacked = np.stack([batch.edge_len, batch.radius, batch.curvature], axis=-1)
        global_mean = stacked.mean(axis=0).astype(np.float32, copy=False)
        global_std = stacked.std(axis=0).astype(np.float32, copy=False)
        parts = np.array_split(stacked, 3, axis=0)
        features = []
        features.append(global_mean)
        features.append(global_std)
        for part in parts:
            if len(part) == 0:
                features.append(np.zeros(3, dtype=np.float32))
                features.append(np.zeros(3, dtype=np.float32))
                continue
            features.append(part.mean(axis=0).astype(np.float32, copy=False))
            features.append(part.std(axis=0).astype(np.float32, copy=False))
        features.append(stacked[0].astype(np.float32, copy=False))
        features.append(stacked[-1].astype(np.float32, copy=False))

        feature_vec = np.concatenate(features, axis=0)
        if feature_vec.size >= self.output_dim:
            return feature_vec[: self.output_dim].astype(np.float32, copy=False)
        padded = np.zeros(self.output_dim, dtype=np.float32)
        padded[: feature_vec.size] = feature_vec.astype(np.float32, copy=False)
        return padded


class MergeScorer:
    """Baseline pairwise compatibility scorer over two path embeddings."""

    def score(self, left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=np.float32)
        right = np.asarray(right, dtype=np.float32)
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 0.0:
            return 0.0
        return float(np.dot(left, right) / denom)


class ArborEncoder:
    """Cluster-level summarizer for global atomicity decisions.

    Mean pooling keeps the dominant structural trend while max pooling preserves
    sharper fragment-level motifs. This stays permutation-invariant and avoids
    introducing a quadratic attention dependency into the export path itself.
    """

    def __init__(self, output_dim: int = 64) -> None:
        self.output_dim = int(output_dim)

    def encode(self, embeddings: Sequence[np.ndarray]) -> np.ndarray:
        if not embeddings:
            return np.zeros(self.output_dim, dtype=np.float32)
        matrix = np.stack([np.asarray(item, dtype=np.float32) for item in embeddings], axis=0)
        mean_pool = matrix.mean(axis=0)
        max_pool = matrix.max(axis=0)
        feature_vec = np.concatenate([mean_pool, max_pool], axis=0)
        if feature_vec.size >= self.output_dim:
            return feature_vec[: self.output_dim].astype(np.float32, copy=False)
        padded = np.zeros(self.output_dim, dtype=np.float32)
        padded[: feature_vec.size] = feature_vec.astype(np.float32, copy=False)
        return padded


class TorchPathEncoder:
    """Factory for a Transformer-based multi-modal path encoder.

    Architecture
    ------------
    1. A linear input projection maps each per-step feature vector to a
       ``d_model``-dimensional token embedding.
    2. A learned ``[CLS]`` token is prepended to the sequence.  It is the sole
       slot that attends globally without being masked.
    3. Sinusoidal positional encodings are added before the transformer stack.
    4. A ``nn.TransformerEncoder`` (default: 2 layers, 4 heads) contextualizes
       the token sequence.  The padding mask passed by callers prevents
       attention from flowing into PAD positions; the ``[CLS]`` position is
       always unmasked.
    5. The ``[CLS]`` output is passed through a final linear head to produce
       the ``output_dim``-dimensional fragment embedding.

    Parameters
    ----------
    input_dim:
        Feature dimension of each time-step.  For the original
        ``(edge_len, radius, curvature)`` representation this is 3.  Callers
        that also supply skeleton or mesh features should set this to the full
        concatenated feature width.
    d_model:
        Internal transformer dimension.  Must be divisible by ``n_heads``.
    n_heads:
        Number of attention heads inside each ``TransformerEncoderLayer``.
    n_layers:
        Number of stacked transformer layers.
    ffn_dim:
        Feed-forward expansion dimension inside each layer.
    dropout:
        Dropout probability applied inside the transformer.
    output_dim:
        Dimension of the returned fragment embedding.
    max_len:
        Maximum sequence length supported by sinusoidal positional encodings.
    """

    def __new__(
        cls,
        input_dim: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        output_dim: int = 32,
        max_len: int = 512,
        # Legacy alias kept so old checkpoints that stored hidden_dim still
        # deserialize correctly.  If both d_model and hidden_dim are given,
        # d_model takes precedence.
        hidden_dim: int | None = None,
    ):
        torch, nn = _require_torch()
        import math

        # Resolve d_model vs legacy hidden_dim.
        _d_model = int(d_model if hidden_dim is None else hidden_dim)

        # n_heads must divide d_model.  Clamp gracefully rather than crash.
        _n_heads = int(n_heads)
        while _d_model % _n_heads != 0 and _n_heads > 1:
            _n_heads -= 1

        class _TorchPathEncoder(nn.Module):
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
                    "ffn_dim": int(ffn_dim),
                    "dropout": float(dropout),
                    "output_dim": self.output_dim,
                    "max_len": int(max_len),
                }
                # Project raw per-step features into the transformer space.
                self.input_proj = nn.Linear(self.input_dim, self.d_model)

                # Learned [CLS] token (shape: [1, 1, d_model]).
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
                nn.init.trunc_normal_(self.cls_token, std=0.02)

                # Sinusoidal positional encoding buffer (not a parameter).
                pe = torch.zeros(int(max_len) + 1, self.d_model)
                position = torch.arange(0, int(max_len) + 1, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, self.d_model, 2, dtype=torch.float)
                    * (-math.log(10000.0) / self.d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term[: self.d_model // 2])
                self.register_buffer("pos_enc", pe.unsqueeze(0))  # [1, max_len+1, d_model]

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

            def forward(self, x, mask=None):
                """Encode a batch of padded path sequences.

                Parameters
                ----------
                x:
                    Float tensor of shape ``[B, T, input_dim]``.
                mask:
                    Bool tensor of shape ``[B, T]`` where ``True`` means PAD.
                    If ``None``, no positions are masked.

                Returns
                -------
                torch.Tensor
                    Shape ``[B, output_dim]``.
                """
                x = x.float()
                batch_size, seq_len, _ = x.shape

                if mask is None:
                    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)

                # Project input features.
                tokens = self.input_proj(x)  # [B, T, d_model]

                # Expand [CLS] across batch and prepend.
                cls = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
                tokens = torch.cat([cls, tokens], dim=1)          # [B, T+1, d_model]

                # Positional encodings (CLS gets position 0, steps get 1..T).
                tokens = tokens + self.pos_enc[:, : seq_len + 1, :]

                # Build the key-padding mask: False for CLS, then the caller's mask.
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                full_mask = torch.cat([cls_mask, mask], dim=1)  # [B, T+1]

                # Run transformer; output: [B, T+1, d_model].
                out = self.transformer(tokens, src_key_padding_mask=full_mask)

                # CLS token at position 0 carries the global summary.
                cls_out = out[:, 0, :]  # [B, d_model]
                return self.output_proj(cls_out)  # [B, output_dim]

        return _TorchPathEncoder()


class TorchMergeScorer:
    """Factory for a torch-native pairwise merge scorer."""

    def __new__(cls, embedding_dim: int = 32, hidden_dim: int = 64):
        torch, nn = _require_torch()

        class _TorchMergeScorer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding_dim = int(embedding_dim)
                self.hidden_dim = int(hidden_dim)
                self._init_kwargs = {
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
                }
                self.net = nn.Sequential(
                    nn.Linear(self.embedding_dim * 4, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, 1),
                )

            def forward(self, left, right):
                features = torch.cat([left, right, torch.abs(left - right), left * right], dim=-1)
                return self.net(features).squeeze(-1)

        return _TorchMergeScorer()


class TorchArborEncoder:
    """Factory for a torch-native permutation-invariant arbor summarizer."""

    def __new__(cls, embedding_dim: int = 32, hidden_dim: int = 64, output_dim: int = 64):
        torch, nn = _require_torch()

        class _TorchArborEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding_dim = int(embedding_dim)
                self.hidden_dim = int(hidden_dim)
                self.output_dim = int(output_dim)
                self._init_kwargs = {
                    "embedding_dim": self.embedding_dim,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                }
                self.proj = nn.Sequential(
                    nn.Linear(self.embedding_dim * 2, self.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.hidden_dim, self.output_dim),
                )

            def forward(self, x, mask=None):
                x = x.float()
                if mask is None:
                    mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
                valid = (~mask).float().unsqueeze(-1)
                denom = valid.sum(dim=1).clamp_min(1.0)
                mean = (x * valid).sum(dim=1) / denom
                max_ready = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
                max_pool = max_ready.max(dim=1).values
                max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
                features = torch.cat([mean, max_pool], dim=-1)
                return self.proj(features)

        return _TorchArborEncoder()


def save_torch_grammar_component(path: str | Path, model) -> None:
    torch, _, = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_name": model.__class__.__name__,
            "init_kwargs": dict(getattr(model, "_init_kwargs", {})),
        },
        path,
    )


def load_torch_grammar_component(path: str | Path, factory):
    torch, _ = _require_torch()
    checkpoint = torch.load(path, map_location="cpu")
    model = factory(**checkpoint.get("init_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build_path_batch(
    edge_len: Iterable[float],
    radius: Iterable[float],
    curvature: Iterable[float],
) -> PathBatch:
    return PathBatch(
        edge_len=np.asarray(list(edge_len), dtype=np.float32),
        radius=np.asarray(list(radius), dtype=np.float32),
        curvature=np.asarray(list(curvature), dtype=np.float32),
    )


def build_multimodal_path_sequence(
    batch: PathBatch,
    *,
    skeleton_feat: np.ndarray | None = None,
    mesh_feat: np.ndarray | None = None,
) -> np.ndarray:
    """Fuse a ``PathBatch`` with optional per-step skeleton and mesh features.

    Returns a float32 array of shape ``[T, D]`` where ``D`` is the total
    feature width, suitable for passing directly into ``TorchPathEncoder``.

    The core ``(edge_len, radius, curvature)`` triplet is always present.
    Optional modalities are appended per-step when provided:

    - ``skeleton_feat``: shape ``[T, D_s]`` or ``[T]`` (broadcast to ``[T, 1]``).
      Typical content: tortuosity, mean radius, branch angle.
    - ``mesh_feat``: shape ``[T, D_m]`` or ``[T]`` (broadcast to ``[T, 1]``).
      Typical content: volume/surface ratio, mean curvature from mesh.

    When an optional modality is ``None``, it is omitted from the output so
    that the returned ``D`` matches the ``input_dim`` the encoder was built
    with.

    Parameters
    ----------
    batch:
        Core path descriptors from ``build_path_batch``.
    skeleton_feat:
        Optional per-step skeleton features.  Must have the same leading
        dimension as ``batch.edge_len``.
    mesh_feat:
        Optional per-step mesh features.  Same constraint.

    Returns
    -------
    np.ndarray
        Shape ``[T, D]``, dtype float32.
    """
    T = len(batch.edge_len)
    parts: list[np.ndarray] = [
        np.stack([batch.edge_len, batch.radius, batch.curvature], axis=-1).astype(np.float32),
    ]

    for extra, label in ((skeleton_feat, "skeleton_feat"), (mesh_feat, "mesh_feat")):
        if extra is None:
            continue
        arr = np.asarray(extra, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] != T:
            raise ValueError(
                f"{label} leading dimension {arr.shape[0]} does not match "
                f"path length {T}"
            )
        parts.append(arr)

    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
