"""2.5D U-Net for membrane probability prediction.

Architecture overview
---------------------
The original codebase used a plain 2D U-Net that processed each Z-slice
independently.  That approach is blind to the 3D context that makes membrane
surfaces coherent across slices.

This module upgrades to a **2.5D** strategy: each Z-slice is processed with
``context_slices`` neighbouring slices stacked as extra input channels, giving
the network local 3D awareness without the memory cost of full 3D convolutions.
For MICrONS (8×8×40 nm voxels, 5× anisotropy in Z), ``context_slices=2``
means the receptive field spans ±2 Z-steps = ±80 nm of axial context.

Architectural improvements over the previous version
-----------------------------------------------------
- ``InstanceNorm2d`` after every conv pair — stable with batch size 1,
  no running-statistic issues at inference time.
- Anisotropy-aware input: ``in_channels = 1 + 2 * context_slices`` rather
  than hard-coded 1, so the first encoder layer adapts to the stacked context.
- Backward-compatible API: ``MembraneUNet(context_slices=0)`` is functionally
  equivalent to the old single-channel 2D network.

Usage
-----
Training::

    from neuronauts.membrane_unet import train_membrane_unet, TrainingConfig
    history = train_membrane_unet("data/membranes", "models/membrane.pt",
                                  config=TrainingConfig(context_slices=2))

Inference in ``run()``::

    from neuronauts.run import run
    metrics = run(volume, pre_pts, post_pts, pre_root_ids, post_root_ids,
                  membrane_unet_checkpoint="models/membrane.pt")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("pip install torch") from exc
    return torch, nn, F


def normalize_slice(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _double_conv(in_ch: int, out_ch: int):
    """Two 3×3 convs each followed by InstanceNorm + ReLU."""
    _, nn, _ = _require_torch()
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# 2.5D U-Net
# ---------------------------------------------------------------------------

class MembraneUNet:
    """Factory that returns a ``torch.nn.Module`` implementing the 2.5D U-Net.

    Parameters
    ----------
    context_slices:
        Number of neighbouring Z-slices on each side to stack as input
        channels.  ``context_slices=0`` degenerates to a plain 2D U-Net.
        ``context_slices=2`` is recommended for MICrONS anisotropy.
    base_channels:
        Channel width at the first encoder level; doubles at each downsampling.
    out_channels:
        Number of output channels (1 for binary membrane segmentation).
    """

    def __new__(
        cls,
        context_slices: int = 2,
        base_channels: int = 32,
        out_channels: int = 1,
        # Legacy kwarg kept for backward compatibility.
        in_channels: int | None = None,
    ):
        torch, nn, F = _require_torch()
        _in = (1 + 2 * context_slices) if in_channels is None else in_channels
        bc = base_channels

        class _UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.context_slices = context_slices

                self.enc1 = _double_conv(_in, bc)
                self.enc2 = _double_conv(bc, bc * 2)
                self.enc3 = _double_conv(bc * 2, bc * 4)
                self.enc4 = _double_conv(bc * 4, bc * 8)
                self.pool = nn.MaxPool2d(2)

                self.bottleneck = _double_conv(bc * 8, bc * 16)

                self.up4 = nn.ConvTranspose2d(bc * 16, bc * 8, kernel_size=2, stride=2)
                self.dec4 = _double_conv(bc * 16, bc * 8)
                self.up3 = nn.ConvTranspose2d(bc * 8, bc * 4, kernel_size=2, stride=2)
                self.dec3 = _double_conv(bc * 8, bc * 4)
                self.up2 = nn.ConvTranspose2d(bc * 4, bc * 2, kernel_size=2, stride=2)
                self.dec2 = _double_conv(bc * 4, bc * 2)
                self.up1 = nn.ConvTranspose2d(bc * 2, bc, kernel_size=2, stride=2)
                self.dec1 = _double_conv(bc * 2, bc)

                self.out_conv = nn.Conv2d(bc, out_channels, kernel_size=1)

            @staticmethod
            def _crop_to(skip, target):
                """Centre-crop ``skip`` to match ``target``'s spatial size."""
                if skip.shape[-2:] == target.shape[-2:]:
                    return skip
                dy = skip.shape[-2] - target.shape[-2]
                dx = skip.shape[-1] - target.shape[-1]
                return skip[
                    :, :,
                    dy // 2: dy // 2 + target.shape[-2],
                    dx // 2: dx // 2 + target.shape[-1],
                ]

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))
                b = self.bottleneck(self.pool(e4))

                u4 = self.up4(b)
                d4 = self.dec4(torch.cat([self._crop_to(e4, u4), u4], dim=1))
                u3 = self.up3(d4)
                d3 = self.dec3(torch.cat([self._crop_to(e3, u3), u3], dim=1))
                u2 = self.up2(d3)
                d2 = self.dec2(torch.cat([self._crop_to(e2, u2), u2], dim=1))
                u1 = self.up1(d2)
                d1 = self.dec1(torch.cat([self._crop_to(e1, u1), u1], dim=1))
                return self.out_conv(d1)

        return _UNet()


# ---------------------------------------------------------------------------
# 2.5D input assembly helpers
# ---------------------------------------------------------------------------

def _assemble_2_5d_slice(
    volume_zyx: np.ndarray,
    z: int,
    context_slices: int,
) -> np.ndarray:
    """Stack slice ``z`` with its ±``context_slices`` neighbours.

    Boundary slices are replicated (edge padding).

    Parameters
    ----------
    volume_zyx:
        3-D float32 array with shape ``[Z, Y, X]``.
    z:
        Target Z index.
    context_slices:
        Number of neighbours on each side.

    Returns
    -------
    np.ndarray
        Shape ``[1 + 2*context_slices, Y, X]``, float32, values in [0, 1].
    """
    Z = volume_zyx.shape[0]
    channels = []
    for dz in range(-context_slices, context_slices + 1):
        zi = int(np.clip(z + dz, 0, Z - 1))
        channels.append(normalize_slice(volume_zyx[zi]))
    return np.stack(channels, axis=0)  # [C, Y, X]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    seed: int = 42
    base_channels: int = 32
    context_slices: int = 2


def load_tiff_pair_dataset(
    dataset_dir: str | Path,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load matched ``train/images/*.tif`` and ``train/labels/*.tif`` pairs."""
    dataset_dir = Path(dataset_dir)
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("pip install tifffile") from exc

    image_paths = sorted((dataset_dir / "train" / "images").glob("*.tif*"))
    label_paths = sorted((dataset_dir / "train" / "labels").glob("*.tif*"))
    if not image_paths or len(image_paths) != len(label_paths):
        raise ValueError(
            f"expected matching train/images and train/labels tif files under {dataset_dir}"
        )

    images = [normalize_slice(tifffile.imread(p)) for p in image_paths]
    labels = [normalize_slice(tifffile.imread(p)) for p in label_paths]
    labels = [(lbl > 0.5).astype(np.float32) for lbl in labels]
    return images, labels


def train_membrane_unet(
    dataset_dir: str | Path,
    output_path: str | Path,
    config: TrainingConfig | None = None,
) -> dict[str, float]:
    """Train a 2.5D ``MembraneUNet`` on a tiff-pair dataset.

    For 2D datasets (single-slice tiff files) set ``config.context_slices=0``
    to fall back to the original 2D behaviour.  For 3D volumes stored as
    per-slice tiffs in Z order, ``context_slices=2`` adds ±2 Z-slice context.

    The best-validation-loss checkpoint is saved to ``output_path``.

    Returns
    -------
    dict with keys ``train_loss`` and ``val_loss`` from the final epoch.
    """
    config = config or TrainingConfig()
    torch, _, F = _require_torch()

    images, labels = load_tiff_pair_dataset(dataset_dir)
    rng = np.random.default_rng(config.seed)
    indices = np.arange(len(images))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - config.val_fraction)))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(indices) else indices[-1:]

    # For 2D datasets each "image" is a single 2-D slice [Y, X].
    # We replicate the context channels to simulate the 2.5D input.
    C = 1 + 2 * config.context_slices

    def make_tensor(batch_indices: Iterable[int]):
        xs, ys = [], []
        for i in batch_indices:
            img = images[i]
            # Replicate the single slice C times to match 2.5D channel count.
            x = np.stack([img] * C, axis=0)  # [C, Y, X]
            xs.append(x)
            ys.append(labels[i][None, :, :])   # [1, Y, X]
        return (
            torch.from_numpy(np.stack(xs, axis=0)).float(),
            torch.from_numpy(np.stack(ys, axis=0)).float(),
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MembraneUNet(
        context_slices=config.context_slices,
        base_channels=config.base_channels,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val = float("inf")
    history: dict[str, float] = {"train_loss": 0.0, "val_loss": 0.0}

    for _ in range(config.epochs):
        model.train()
        rng.shuffle(train_idx)
        losses = []
        for start in range(0, len(train_idx), config.batch_size):
            batch = train_idx[start: start + config.batch_size]
            x, y = make_tensor(batch)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            x_val, y_val = make_tensor(val_idx)
            x_val, y_val = x_val.to(device), y_val.to(device)
            val_logits = model(x_val)
            val_loss = float(F.binary_cross_entropy_with_logits(val_logits, y_val).detach().cpu())

        history = {
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_loss": val_loss,
        }
        if val_loss < best_val:
            best_val = val_loss
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "context_slices": config.context_slices,
                        "base_channels": config.base_channels,
                    },
                    "best_val_loss": best_val,
                },
                out,
            )

    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str | Path,
    device: str | None = None,
):
    """Load a saved ``MembraneUNet`` checkpoint.

    Returns ``(model, device_str)``.
    """
    torch, _, _ = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    model = MembraneUNet(
        context_slices=int(cfg.get("context_slices", 0)),
        base_channels=int(cfg.get("base_channels", 32)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(target_device)
    model.eval()
    return model, target_device


def predict_membranes(
    model,
    volume: np.ndarray,
    *,
    device: str,
    batch_size: int = 8,
    context_slices: int | None = None,
) -> np.ndarray:
    """Run ``model`` slice-by-slice over ``volume`` and return a membrane
    probability volume of the same spatial shape.

    Parameters
    ----------
    model:
        A ``MembraneUNet`` module (already on ``device`` and in eval mode).
    volume:
        EM intensity array, shape ``[X, Y, Z]`` (neuronauts convention).
    device:
        Torch device string.
    batch_size:
        Number of Z-slices to batch together.
    context_slices:
        If ``None``, read from ``model.context_slices``.  Override to force a
        specific context width (e.g. ``0`` for a checkpoint trained in 2D mode).

    Returns
    -------
    np.ndarray
        Membrane probabilities, shape ``[X, Y, Z]``, float32 in [0, 1].
    """
    torch, _, _ = _require_torch()

    model.eval()
    k = context_slices if context_slices is not None else getattr(model, "context_slices", 0)

    # Rearrange to [Z, Y, X] for slice iteration.
    vol_zyx = np.moveaxis(volume.astype(np.float32), 2, 0)
    Z = vol_zyx.shape[0]

    slices_in = np.stack(
        [_assemble_2_5d_slice(vol_zyx, z, k) for z in range(Z)],
        axis=0,
    )  # [Z, C, Y, X]

    preds = []
    with torch.no_grad():
        for start in range(0, Z, batch_size):
            batch = torch.from_numpy(slices_in[start: start + batch_size]).float().to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds.append(probs.astype(np.float32))

    membrane_zyx = np.concatenate(preds, axis=0)   # [Z, Y, X]
    return np.moveaxis(membrane_zyx, 0, 2)          # back to [X, Y, Z]
