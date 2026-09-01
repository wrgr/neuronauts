#!/usr/bin/env python3
"""EXP-054: scorer bake-off with a fail-closed candidate-panel gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-result", type=Path,
                        default=ROOT / "results/exp053b_l2_candidate_panel.json")
    parser.add_argument("--min-covered-positives", type=int, default=10)
    parser.add_argument("--min-panel-recall", type=float, default=.9)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/exp054_fixed_panel_scorers.json")
    args = parser.parse_args()

    panel = json.loads(args.panel_result.read_text())
    population = panel["population"]
    best = panel["best_recall_configuration"]
    covered = int(population["l2_covered_true_pairs"])
    recall = float(best["recall_all_true_pairs"])
    failures = []
    if covered < args.min_covered_positives:
        failures.append(f"covered positives {covered} < {args.min_covered_positives}")
    if recall < args.min_panel_recall:
        failures.append(f"candidate recall {recall:.3f} < {args.min_panel_recall:.3f}")

    result = {
        "experiment": "EXP-054 fixed-panel scorer bake-off",
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "input": str(args.panel_result.relative_to(ROOT)),
            "synthetic_fallback": False,
        },
        "prerequisite_gate": {
            "required_covered_positives": args.min_covered_positives,
            "required_candidate_recall": args.min_panel_recall,
            "observed_covered_positives": covered,
            "observed_candidate_recall": recall,
            "passed": not failures,
            "failures": failures,
        },
        "status": "ready" if not failures else "prerequisite_failed",
        "scorer_metrics": {},
        "interpretation": (
            "Panel is valid; scorer execution may proceed."
            if not failures else
            "No scorer comparison was run because a panel without sufficient "
            "positive coverage cannot estimate discrimination or calibration."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["prerequisite_gate"], sort_keys=True))
    print(f"complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
