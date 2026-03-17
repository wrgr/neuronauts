#!/usr/bin/env python3
"""Export top-K box-level assembly hypotheses for reranker training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuronauts.assembly_dataset import build_hypothesis_examples, hypothesis_features, save_hypothesis_examples_npz
from neuronauts.fetch import make_test_volume
from neuronauts.line_graph import evaluate
from neuronauts.run import BENCHMARK_CONFIG, build_graph_hypotheses, simulate_paths_and_hits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/assembly_ranking_dataset.npz", help="Output dataset path.")
    parser.add_argument("--cases", type=int, default=3, help="Number of synthetic boxes to sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thresholds", default="-0.5,0.0,0.5", help="Comma-separated learned merge thresholds.")
    parser.add_argument("--beam-widths", default="1,2,4", help="Comma-separated beam widths.")
    parser.add_argument("--shared-grammar-checkpoint", default=None, help="Optional shared grammar checkpoint.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    beam_widths = [int(item.strip()) for item in args.beam_widths.split(",") if item.strip()]

    all_examples = []
    manifest = []
    for case_idx in range(args.cases):
        chunk, synapses = make_test_volume(config=BENCHMARK_CONFIG, seed=args.seed + case_idx)
        path_arr, synapse_hits, path_lengths, _ = simulate_paths_and_hits(
            chunk.data,
            synapses.pre_pt,
            synapses.post_pt,
            seed=args.seed + case_idx,
            verbose=False,
        )
        hypotheses = []
        for threshold, beam_width, graph in build_graph_hypotheses(
            path_arr,
            path_lengths,
            synapse_hits,
            synapses.pre_pt,
            synapses.post_pt,
            thresholds=thresholds,
            beam_widths=beam_widths,
            shared_grammar_checkpoint=args.shared_grammar_checkpoint,
        ):
            metrics = evaluate(graph, synapses.pre_root_id, synapses.post_root_id)
            features = hypothesis_features(
                graph,
                merge_threshold=threshold,
                beam_width=beam_width,
                n_synapses=len(synapses.pre_pt),
            )
            hypothesis_id = f"thr={threshold:.3f}|beam={beam_width}"
            hypotheses.append((hypothesis_id, features, metrics))

        box_id = f"synthetic_case_{case_idx:03d}"
        examples = build_hypothesis_examples(box_id, hypotheses)
        all_examples.extend(examples)
        manifest.append({"box_id": box_id, "hypotheses": len(examples)})

    save_hypothesis_examples_npz(args.output, all_examples)
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved dataset: {args.output}")
    print(f"examples={len(all_examples)}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
