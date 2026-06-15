"""Model checkpoint save/load for NeuronautS.

Persists FragmentEncoder + EdgePartitionGNN weights so the spatial variance
study can train once and evaluate many test bboxes without retraining.

Usage
-----
    from treestitch.checkpoint import save_checkpoint, load_checkpoint
    from treestitch.embed import FragmentEncoder
    from neuronauts.assemble.edge_partition import EdgePartitionGNN

    # After training:
    save_checkpoint("run.pt", encoder, gnn,
                    encoder_kwargs={"node_input_dim": 4, "d_model": 64, "output_dim": 32},
                    gnn_kwargs={"input_dim": 35, "d_model": 64, "n_edge_types": 3})

    # Before inference:
    encoder, gnn = load_checkpoint("run.pt")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    encoder,
    gnn,
    *,
    encoder_kwargs: dict | None = None,
    gnn_kwargs: dict | None = None,
    extra: dict | None = None,
) -> None:
    """Save encoder + GNN state dicts and their constructor kwargs.

    Parameters
    ----------
    path:
        File path (will be created/overwritten).
    encoder:
        `FragmentEncoder` (SkeletonGNN) nn.Module instance.
    gnn:
        `EdgePartitionGNN` nn.Module instance.
    encoder_kwargs:
        Dict of kwargs passed to `FragmentEncoder(...)`.  Stored so the
        checkpoint is self-contained — caller must pass them explicitly since
        the factory `__new__` pattern doesn't expose them on the object.
    gnn_kwargs:
        Dict of kwargs passed to `EdgePartitionGNN(...)`.
    extra:
        Optional dict of any extra metadata to embed (bbox coords, epoch, etc.).
    """
    torch.save({
        "encoder_state": encoder.state_dict(),
        "gnn_state": gnn.state_dict(),
        "encoder_kwargs": encoder_kwargs or {},
        "gnn_kwargs": gnn_kwargs or {},
        "extra": extra or {},
    }, path)


def load_checkpoint(path: str | Path) -> tuple[Any, Any]:
    """Load encoder + GNN from a checkpoint file.

    Returns
    -------
    (encoder, gnn) — both nn.Module instances ready for inference (eval mode).
    """
    from treestitch.embed import FragmentEncoder
    from neuronauts.assemble.edge_partition import EdgePartitionGNN

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    enc_kwargs = ckpt["encoder_kwargs"]
    gnn_kwargs = ckpt["gnn_kwargs"]

    encoder = FragmentEncoder(**enc_kwargs)
    encoder.load_state_dict(ckpt["encoder_state"])
    encoder.eval()

    gnn = EdgePartitionGNN(**gnn_kwargs)
    gnn.load_state_dict(ckpt["gnn_state"])
    gnn.eval()

    return encoder, gnn


def checkpoint_extra(path: str | Path) -> dict:
    """Return only the `extra` metadata from a checkpoint without loading weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("extra", {})
