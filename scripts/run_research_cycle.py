#!/usr/bin/env python3
"""Run the canonical Neuronauts v1 research cycle (export → train → validate).

NOTE: For Neuronauts v2 the recommended training path is ``scripts/train.py``,
which handles real-data box caching, shared grammar training, and GAT training
in a single CLI without requiring pre-exported dataset files::

    python scripts/train.py run \\
        --cache-dir data/boxes \\
        --n-boxes 100 \\
        --grammar-output models/shared_grammar.pt \\
        --gat-output models/gat.pt \\
        --epochs 50 \\
        --train-gat

This script remains functional for the v1 offline-export pipeline (useful for
ablation studies or when datasets must be shared across multiple training runs).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from neuronauts.experiment_driver import ResearchCycleConfig, run_research_cycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--merge-dataset", default="data/merge_dataset_smoke.npz")
    parser.add_argument("--topology-dataset", default="data/topology_dataset_smoke.npz")
    parser.add_argument("--shared-model", default="models/shared_grammar_smoke.pt")
    parser.add_argument("--assembly-dataset", default="data/assembly_ranking_smoke.npz")
    parser.add_argument("--assembly-reranker", default="models/assembly_reranker_smoke.npz")
    parser.add_argument("--export-boxes", default="0,1,2")
    parser.add_argument("--assembly-cases", type=int, default=3)
    parser.add_argument("--thresholds", default="-0.5,0.0,0.5")
    parser.add_argument("--beam-widths", default="1,2,4")
    parser.add_argument("--selection-box-indices", default="0,1,2")
    parser.add_argument("--holdout-box-indices", default="3,4,5")
    parser.add_argument("--run-data-mode", choices=["synthetic", "real"], default="real")
    parser.add_argument("--real-boxes-per-eval", type=int, default=3)
    parser.add_argument("--real-min-synapses", type=int, default=50)
    parser.add_argument("--membrane-source", default="auto")
    parser.add_argument("--membrane-cache-dir", default="cache/membranes")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    summary = run_research_cycle(
        ResearchCycleConfig(
            repo_root=repo_root,
            python_bin=args.python_bin,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            merge_dataset=args.merge_dataset,
            topology_dataset=args.topology_dataset,
            shared_model=args.shared_model,
            assembly_dataset=args.assembly_dataset,
            assembly_reranker=args.assembly_reranker,
            export_boxes=args.export_boxes,
            assembly_cases=args.assembly_cases,
            thresholds=args.thresholds,
            beam_widths=args.beam_widths,
            selection_box_indices=args.selection_box_indices,
            holdout_box_indices=args.holdout_box_indices,
            run_data_mode=args.run_data_mode,
            real_boxes_per_eval=args.real_boxes_per_eval,
            real_min_synapses=args.real_min_synapses,
            membrane_source=args.membrane_source,
            membrane_cache_dir=args.membrane_cache_dir,
            quiet=args.quiet,
        ),
        env=os.environ.copy(),
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
