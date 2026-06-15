#!/usr/bin/env python3
"""Generate bar chart comparing ARI and merge_P across in-column test locations.

Uses the hard-coded results from the spatial variance study (Phase 2.12/2.13).
Outputs docs/latex/figures/variance_barchart.pdf and .png.

Usage
-----
    python scripts/plot_variance.py
    python scripts/plot_variance.py --out /tmp/variance.pdf
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
    p.add_argument("--out", default=str(_ROOT / "docs/latex/figures/variance_barchart.pdf"))
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Results from the spatial variance study (Phase 2.12)
    locations = [
        "T1\nx=1150-1350k\n(reference)",
        "T2\nx=550-750k\n(west)",
        "T3\ny=870-940k\n(south)",
        "T4\ny=1000-1070k\n(north, 6× denser)",
    ]
    ari      = [0.613, 0.877, 0.829, 0.287]
    merge_p  = [0.977, 0.972, 0.991, 0.958]

    x = np.arange(len(locations))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - width / 2, ari,     width, label="ARI",     color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x + width / 2, merge_p, width, label="merge_P", color="#DD8452", alpha=0.85)

    # Annotate bars
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # Mean lines
    ax.axhline(np.mean(ari),     color="#4C72B0", linestyle="--", linewidth=1.2,
               alpha=0.6, label=f"ARI mean={np.mean(ari):.3f}")
    ax.axhline(np.mean(merge_p), color="#DD8452", linestyle="--", linewidth=1.2,
               alpha=0.6, label=f"merge_P mean={np.mean(merge_p):.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(locations, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Spatial variance: ARI and merge_P across 4 in-column test bboxes\n"
                 "(same A/B/C-trained model, 50 µm seam buffer, b = −2.0)",
                 fontsize=10)
    ax.legend(fontsize=9, loc="lower right")
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(3.6, 0.951, "merge_P=0.95 threshold", fontsize=7, color="gray", va="bottom")

    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    # Also save PNG alongside
    fig.savefig(out.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out} and {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
