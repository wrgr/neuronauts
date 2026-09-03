#!/usr/bin/env python3
"""EXP-055: conservative soma forest with a fail-closed scorer gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorer-result", type=Path,
                        default=ROOT / "results/exp054_fixed_panel_scorers.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/exp055_conservative_soma_forest.json")
    args = parser.parse_args()

    scorer = json.loads(args.scorer_result.read_text())
    scorer_ready = scorer.get("status") == "completed"
    result = {
        "experiment": "EXP-055 conservative soma-seeded forest",
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "input": str(args.scorer_result.relative_to(ROOT)),
            "synthetic_fallback": False,
        },
        "prerequisite_gate": {
            "required_scorer_status": "completed",
            "observed_scorer_status": scorer.get("status"),
            "passed": scorer_ready,
        },
        "status": "ready" if scorer_ready else "prerequisite_failed",
        "assembly_metrics": {},
        "interpretation": (
            "Scorer is valid; forest execution may proceed."
            if scorer_ready else
            "No forest was assembled because joining roots with an unvalidated "
            "or empty scorer would reproduce the EXP-053A collapse failure."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["prerequisite_gate"], sort_keys=True))
    print(f"complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
