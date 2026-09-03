#!/usr/bin/env python3
"""Generate reliability diagram (calibration plot) from a saved checkpoint.

Loads the EdgePartitionGNN checkpoint, runs reliability_diagram() on the training
graph, and outputs a PDF/PNG showing predicted confidence vs fraction of true
positive edges per confidence bin.

Usage
-----
    python scripts/plot_calibration.py
    python scripts/plot_calibration.py --checkpoint /tmp/neuronauts_variance.pt
    python scripts/plot_calibration.py --out /tmp/calib.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="/tmp/neuronauts_variance.pt")
    p.add_argument("--out", default=str(_ROOT / "docs/latex/figures/reliability_diagram.pdf"))
    p.add_argument("--version", type=int, default=1718)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from treestitch.calibration import (
        expected_calibration_error, fit_temperature, reliability_diagram,
    )
    from treestitch.checkpoint import load_checkpoint
    from treestitch.embed import encode_fragments
    from treestitch.graph import build_observation_graph
    from treestitch.realworld import build_region_world

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found at {ckpt_path}. Run scripts/spatial_variance.py first.")
        return 1

    print(f"Loading checkpoint from {ckpt_path} …")
    encoder, model = load_checkpoint(ckpt_path)

    # Use Train A bbox (same as spatial_variance.py)
    bbox_a = ((750_000, 930_000, 780_000), (950_000, 1_000_000, 880_000))
    print("Building train-A world for calibration …")
    frags, region, _ = build_region_world(
        bbox_a, version=args.version, seed=args.seed, verbose=True)

    frags_enc = encode_fragments(encoder, frags, device=args.device)
    graph = build_observation_graph(region, frags_enc, side="pre", k_spatial=8)

    print("Fitting temperature T …")
    T = fit_temperature(model, graph, bias=-2.0, device=args.device)
    print(f"  T = {T:.4f}")

    diag = reliability_diagram(model, graph, T, bias=-2.0,
                               n_bins=args.n_bins, device=args.device)
    ece = expected_calibration_error(diag)
    print(f"  ECE = {ece:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(5, 5))
    valid = ~__import__("numpy").isnan(diag["frac_pos"])
    mc = diag["mean_conf"][valid]
    fp = diag["frac_pos"][valid]
    ct = diag["counts"][valid]

    # Bar chart (gap calibration style)
    ax.bar(mc, fp, width=0.08, alpha=0.7, color="#4C72B0", label="Fraction positive")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    # Size-coded scatter
    sizes = 20 + 80 * ct / ct.max()
    ax.scatter(mc, fp, s=sizes, color="#4C72B0", zorder=3, alpha=0.9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Fraction of true positive edges")
    ax.set_title(f"Reliability diagram  (T={T:.3f}, ECE={ece:.3f})\n"
                 f"EdgePartitionGNN after temperature scaling", fontsize=10)
    ax.legend(fontsize=9)

    # ECE annotation
    ax.text(0.05, 0.92, f"ECE = {ece:.3f}", transform=ax.transAxes,
            fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out} and {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
