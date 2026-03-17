"""Attention-based topology validation models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("pip install torch or pip install -e .[topology]") from exc
    return torch, nn, F


class AttentionArborValidator:
    """Factory wrapper returning a torch attention model.

    The project keeps torch optional, so this class is written as a small
    factory. Instantiating it returns a real ``torch.nn.Module``.
    """

    def __new__(
        cls,
        *,
        embed_dim: int = 32,
        num_heads: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        _, nn, _ = _require_torch()

        class _AttentionArborValidator(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = int(embed_dim)
                self.num_heads = int(num_heads)
                self.hidden_dim = int(hidden_dim)
                self.dropout = float(dropout)
                self.input_proj = nn.Linear(self.embed_dim, self.embed_dim)
                self.attention = nn.MultiheadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                self.classifier = nn.Sequential(
                    nn.Linear(self.embed_dim, self.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_dim, 1),
                    nn.Sigmoid(),
                )

            def forward(self, x, mask=None):
                x = self.input_proj(x)
                attended, _ = self.attention(x, x, x, key_padding_mask=mask)
                if mask is not None:
                    weights = (~mask).float().unsqueeze(-1)
                    denom = weights.sum(dim=1).clamp_min(1.0)
                    pooled = (attended * weights).sum(dim=1) / denom
                else:
                    pooled = attended.mean(dim=1)
                return self.classifier(pooled)

        return _AttentionArborValidator()


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    seed: int = 42


def train_iteration(model, optimizer, batch_x, batch_y, *, mask=None) -> float:
    """One optimization step for the attention validator."""
    torch, _, F = _require_torch()
    del torch
    model.train()
    optimizer.zero_grad()
    probs = model(batch_x, mask=mask).squeeze(-1)
    loss = F.binary_cross_entropy(probs, batch_y.float())
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def save_validator(path: str | Path, model, *, embed_dim: int) -> None:
    torch, _, _ = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embed_dim": int(embed_dim),
        },
        path,
    )


def load_validator(path: str | Path):
    torch, _, _ = _require_torch()
    checkpoint = torch.load(path, map_location="cpu")
    model = AttentionArborValidator(embed_dim=int(checkpoint["embed_dim"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
