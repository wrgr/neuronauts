#!/usr/bin/env python3
"""Plot iteration-level F1 / precision / recall from iterative_loop output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_tsv", help="Path to iteration_summary.tsv")
    parser.add_argument("--output", default=None, help="Optional output image path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary_tsv)
    rows = list(csv.DictReader(summary_path.open(encoding="utf-8"), delimiter="\t"))
    if not rows:
        raise SystemExit("no iteration rows found")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting: pip install matplotlib") from exc

    x = [int(row["iteration"]) for row in rows]
    mean_f1 = [float(row["mean_f1"]) for row in rows]
    best_f1 = [float(row["best_f1"]) for row in rows]
    mean_precision = [float(row["mean_precision"]) for row in rows]
    mean_recall = [float(row["mean_recall"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, mean_f1, label="Mean F1", marker="o")
    ax.plot(x, best_f1, label="Best F1", linestyle="--", marker="o")
    ax.plot(x, mean_precision, label="Mean Precision", marker="o")
    ax.plot(x, mean_recall, label="Mean Recall", marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_title("Iteration Metrics")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(args.output)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
