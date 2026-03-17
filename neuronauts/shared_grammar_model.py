"""Shared multitask grammar model and training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .grammar import TorchArborEncoder, TorchMergeScorer, TorchPathEncoder, _require_torch


class SharedGrammarModel:
    """Factory for a shared encoder with merge and atomicity heads."""

    def __new__(
        cls,
        *,
        input_dim: int = 3,
        path_hidden_dim: int = 64,
        embedding_dim: int = 32,
        merge_hidden_dim: int = 64,
        arbor_hidden_dim: int = 64,
        arbor_output_dim: int = 64,
        atomicity_hidden_dim: int = 64,
    ):
        torch, nn = _require_torch()

        class _SharedGrammarModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self._init_kwargs = {
                    "input_dim": int(input_dim),
                    "path_hidden_dim": int(path_hidden_dim),
                    "embedding_dim": int(embedding_dim),
                    "merge_hidden_dim": int(merge_hidden_dim),
                    "arbor_hidden_dim": int(arbor_hidden_dim),
                    "arbor_output_dim": int(arbor_output_dim),
                    "atomicity_hidden_dim": int(atomicity_hidden_dim),
                }
                self.path_encoder = TorchPathEncoder(
                    input_dim=input_dim,
                    hidden_dim=path_hidden_dim,
                    output_dim=embedding_dim,
                )
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

        return _SharedGrammarModel()


@dataclass(frozen=True)
class SharedTrainingConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    merge_loss_weight: float = 1.0
    atomicity_loss_weight: float = 1.0
    seed: int = 42


def multitask_train_step(
    model,
    optimizer,
    *,
    merge_batch: dict,
    topology_batch: dict,
    merge_loss_weight: float = 1.0,
    atomicity_loss_weight: float = 1.0,
) -> dict[str, float]:
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
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "merge_loss": float(merge_loss.detach().cpu()),
        "atomicity_loss": float(atomicity_loss.detach().cpu()),
    }


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
    checkpoint = torch.load(path, map_location="cpu")
    model = SharedGrammarModel(**checkpoint.get("init_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
