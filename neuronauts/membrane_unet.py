"""Small 2D U-Net utilities for membrane prediction."""

from __future__ import annotations

from dataclasses import dataclass
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


class DoubleConv:
    def __new__(cls, in_channels: int, out_channels: int):
        torch, nn, _ = _require_torch()
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


class MembraneUNet:
    def __new__(cls, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        torch, nn, F = _require_torch()

        class _UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = DoubleConv(in_channels, base_channels)
                self.enc2 = DoubleConv(base_channels, base_channels * 2)
                self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
                self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)
                self.pool = nn.MaxPool2d(2)
                self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)
                self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
                self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
                self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
                self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
                self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
                self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
                self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
                self.dec1 = DoubleConv(base_channels * 2, base_channels)
                self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

            @staticmethod
            def _match(skip, upsampled):
                if skip.shape[-2:] == upsampled.shape[-2:]:
                    return skip
                dy = skip.shape[-2] - upsampled.shape[-2]
                dx = skip.shape[-1] - upsampled.shape[-1]
                return skip[:, :, dy // 2 : dy // 2 + upsampled.shape[-2], dx // 2 : dx // 2 + upsampled.shape[-1]]

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                e4 = self.enc4(self.pool(e3))
                b = self.bottleneck(self.pool(e4))

                u4 = self.up4(b)
                d4 = self.dec4(torch.cat([self._match(e4, u4), u4], dim=1))
                u3 = self.up3(d4)
                d3 = self.dec3(torch.cat([self._match(e3, u3), u3], dim=1))
                u2 = self.up2(d3)
                d2 = self.dec2(torch.cat([self._match(e2, u2), u2], dim=1))
                u1 = self.up1(d2)
                d1 = self.dec1(torch.cat([self._match(e1, u1), u1], dim=1))
                return self.out(d1)

        return _UNet()


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    seed: int = 42
    base_channels: int = 32


def load_tiff_pair_dataset(dataset_dir: str | Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    dataset_dir = Path(dataset_dir)
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("pip install tifffile") from exc

    image_paths = sorted((dataset_dir / "train" / "images").glob("*.tif*"))
    label_paths = sorted((dataset_dir / "train" / "labels").glob("*.tif*"))
    if not image_paths or len(image_paths) != len(label_paths):
        raise ValueError(f"expected matching train/images and train/labels tif files under {dataset_dir}")

    images = [normalize_slice(tifffile.imread(path)) for path in image_paths]
    labels = [normalize_slice(tifffile.imread(path)) for path in label_paths]
    labels = [(label > 0.5).astype(np.float32) for label in labels]
    return images, labels


def train_membrane_unet(
    dataset_dir: str | Path,
    output_path: str | Path,
    config: TrainingConfig | None = None,
) -> dict[str, float]:
    config = config or TrainingConfig()
    torch, _, F = _require_torch()

    images, labels = load_tiff_pair_dataset(dataset_dir)
    rng = np.random.default_rng(config.seed)
    indices = np.arange(len(images))
    rng.shuffle(indices)
    split = max(1, int(len(indices) * (1.0 - config.val_fraction)))
    train_idx = indices[:split]
    val_idx = indices[split:] if split < len(indices) else indices[-1:]

    def make_tensor(batch_indices: Iterable[int]) -> tuple[object, object]:
        x = np.stack([images[i] for i in batch_indices], axis=0)[:, None, :, :]
        y = np.stack([labels[i] for i in batch_indices], axis=0)[:, None, :, :]
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MembraneUNet(base_channels=config.base_channels).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val = float("inf")
    history = {"train_loss": 0.0, "val_loss": 0.0}

    for _ in range(config.epochs):
        model.train()
        rng.shuffle(train_idx)
        losses = []
        for start in range(0, len(train_idx), config.batch_size):
            batch = train_idx[start : start + config.batch_size]
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

        history = {"train_loss": float(np.mean(losses)) if losses else 0.0, "val_loss": val_loss}
        if val_loss < best_val:
            best_val = val_loss
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "best_val_loss": best_val,
                },
                output_path,
            )

    return history


def load_model(checkpoint_path: str | Path, device: str | None = None):
    torch, _, _ = _require_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    config = checkpoint.get("config", {})
    model = MembraneUNet(base_channels=int(config.get("base_channels", 32)))
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
) -> np.ndarray:
    torch, _, _ = _require_torch()
    slices = np.moveaxis(volume.astype(np.float32), 2, 0)
    slices = np.stack([normalize_slice(slice_) for slice_ in slices], axis=0)
    preds = []
    with torch.no_grad():
        for start in range(0, len(slices), batch_size):
            batch = torch.from_numpy(slices[start : start + batch_size, None, :, :]).float().to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds.append(probs.astype(np.float32))
    membrane = np.concatenate(preds, axis=0)
    return np.moveaxis(membrane, 0, 2)
