#!/usr/bin/env python3
"""Fetch one MICRONS box, predict membranes with the trained U-Net, and cache the result."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuronauts.fetch import RealBoxSpec, fetch_volume, save_cached_membrane
from neuronauts.membrane_unet import load_model, predict_membranes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/membrane_unet.pt", help="Trained membrane U-Net checkpoint.")
    parser.add_argument("--cache-dir", default="cache/membranes", help="Cache directory for membrane .npy volumes.")
    parser.add_argument("--center-nm", required=True, help="Comma-separated center in nm: x,y,z")
    parser.add_argument("--side-um", type=float, default=6.0)
    parser.add_argument("--mip", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu or cuda.")
    return parser.parse_args()


def parse_center(center_text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in center_text.split(",")]
    if len(parts) != 3:
        raise ValueError("--center-nm must be x,y,z")
    return tuple(parts)  # type: ignore[return-value]


def main() -> int:
    args = parse_args()
    box = RealBoxSpec(center_nm=parse_center(args.center_nm), side_um=args.side_um, mip=args.mip)
    chunk = fetch_volume(box.bbox_nm, mip=box.mip)
    model, device = load_model(args.checkpoint, device=args.device)
    membrane = predict_membranes(model, chunk.data, device=device, batch_size=args.batch_size)
    output_path = save_cached_membrane(
        box,
        args.cache_dir,
        membrane,
        source=str(Path(args.checkpoint)),
        extra_metadata={"device": device},
    )
    print(f"saved membrane cache: {output_path}")
    print(f"shape={membrane.shape} min={float(membrane.min()):.4f} max={float(membrane.max()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
