#!/usr/bin/env python3
"""EXP-053A: checkpoint bake-off on one fixed edit-bearing v117 population."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuronauts.data import lineage as L
from neuronauts.grammar import featurize_path_points
from neuronauts.real_dense_soma import (
    Fragment, assert_real_root_ids, build_candidate_edges_batched,
    partition_metrics, single_soma_compliance, skeleton_from_observed_points,
    soma_seeded_assemble, true_merge_pair_count,
)
from neuronauts.shared_grammar_model import load_shared_grammar_model
from scripts.benchmark_exp051_real_dense_soma_grammar import (
    circuit_f1, exact_soma_lineage, fetch_candidates,
)

BBOX = ((844776.0, 700136.0, 875040.0),
        (874776.0, 730136.0, 905040.0))
ANCHOR_SOMA_NM = (859776.0, 715136.0, 890040.0)
ANCHOR_TARGET_ROOT = 864691135106016333
CHECKPOINTS = (
    "shared_grammar_raw_skel_50e.pt",
    "shared_grammar_raw_skel_gat50e.pt",
    "shared_grammar_real.pt",
    "shared_grammar_root_neighborhood_run001.pt",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cluster_summary(prediction: dict[int, int]) -> dict:
    sizes = sorted(Counter(prediction.values()).values(), reverse=True)
    return {
        "n_clusters_all_roots": len(sizes),
        "largest_cluster_roots": sizes[0] if sizes else 0,
        "top_10_cluster_sizes": sizes[:10],
        "predicted_join_pairs": int(sum(n * (n - 1) // 2 for n in sizes)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-version", type=int, default=1412)
    parser.add_argument("--min-root-observations", type=int, default=10)
    parser.add_argument("--max-path-points", type=int, default=96)
    parser.add_argument("--max-distance-nm", type=float, default=2500.0)
    parser.add_argument("--thresholds", default="0,1,2,3,4,5,6")
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/exp053a_checkpoint_bakeoff.json")
    args = parser.parse_args()

    token = sys.stdin.readline().strip() if args.token_stdin else L.DEFAULT_TOKEN
    if not token:
        raise SystemExit("CAVE token is required")
    started = time.time()
    target_timestamp = L.version_timestamp(args.target_version, token=token)
    if target_timestamp is None:
        raise SystemExit("target timestamp unavailable")

    print("[1/5] loading fixed real edit-bearing population", flush=True)
    point_map, label_map, context = fetch_candidates(
        BBOX, target_timestamp=target_timestamp, token=token)
    all_roots = sorted(point_map)
    assert_real_root_ids(all_roots)
    soma_counts, target_soma_counts = exact_soma_lineage(
        BBOX, target_timestamp=target_timestamp, token=token)
    seeds = sorted(set(all_roots) & set(soma_counts))
    if not seeds or ANCHOR_TARGET_ROOT not in target_soma_counts:
        raise SystemExit("fixed proofread soma anchor failed validation")
    if any(soma_counts[root] > 1 for root in seeds):
        raise SystemExit("multi-soma v117 root cannot be assembled atomically")

    selected = sorted({root for root in all_roots
                       if len(point_map[root]) >= args.min_root_observations}
                      | set(seeds))
    fragments = []
    contaminated = 0
    for root in selected:
        vertices, skel_edges = skeleton_from_observed_points(
            point_map[root], max_points=args.max_path_points)
        if len(skel_edges) == 0:
            continue
        label, purity, n_labels = label_map.get(root, (0, 0.0, 0))
        contaminated += int(n_labels > 1)
        fragments.append(Fragment(root, vertices, skel_edges,
                                  soma_counts.get(root, 0), label, purity))
    true_pairs = true_merge_pair_count(fragments)
    if true_pairs < 10:
        raise SystemExit(f"edit-signal gate failed: {true_pairs} < 10")
    print(f"      synapses={len(context['synapse_ids'])}; all_roots={len(all_roots)}; "
          f"active={len(fragments)}; true_pairs={true_pairs}", flush=True)

    baseline = partition_metrics(
        fragments, {fragment.root_id: i for i, fragment in enumerate(fragments)})
    thresholds = [float(value) for value in args.thresholds.split(",") if value]

    def with_confusers(active: dict[int, int]) -> dict[int, int]:
        prediction = dict(active)
        next_cluster = max(prediction.values(), default=-1) + 1
        for root in all_roots:
            if root not in prediction:
                prediction[root] = next_cluster
                next_cluster += 1
        return prediction

    runs = {}
    for index, name in enumerate(CHECKPOINTS, start=1):
        print(f"[{index + 1}/5] scoring {name}", flush=True)
        checkpoint = ROOT / "models" / name
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint missing: {checkpoint}")
        model = load_shared_grammar_model(checkpoint)
        mode = model.path_feature_mode
        mip2_nm = np.asarray([32.0, 32.0, 40.0], dtype=np.float32)
        featurize = lambda points, _mode=mode: featurize_path_points(
            points / mip2_nm, mode=_mode)
        candidate_edges = build_candidate_edges_batched(
            fragments, model, featurize,
            max_distance_nm=args.max_distance_nm)
        sweep = {}
        for threshold in thresholds:
            prediction = with_confusers(soma_seeded_assemble(
                fragments, candidate_edges, min_score=threshold))
            metrics = partition_metrics(fragments, prediction)
            metrics.update(single_soma_compliance(fragments, prediction))
            metrics.update(circuit_f1(context, prediction))
            metrics.update(cluster_summary(prediction))
            sweep[str(threshold)] = metrics
        scores = [edge.score for edge in candidate_edges]
        runs[name] = {
            "checkpoint_sha256": file_sha256(checkpoint),
            "path_feature_mode": mode,
            "candidate_edges": len(candidate_edges),
            "score_quantiles": ({str(q): float(np.quantile(scores, q))
                                  for q in (0.0, .25, .5, .75, .9, .99, 1.0)}
                                 if scores else {}),
            "threshold_sweep": sweep,
        }

    result = {
        "experiment": "EXP-053A checkpoint bake-off",
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "bbox_nm": [list(BBOX[0]), list(BBOX[1])],
            "anchor_soma_nm": list(ANCHOR_SOMA_NM),
            "anchor_target_root": ANCHOR_TARGET_ROOT,
            "target_version": args.target_version,
            "target_timestamp": target_timestamp,
            "ground_truth_used_during_inference": False,
            "benchmark_selection_used_target_lineage": True,
            "synthetic_fallback": False,
            "thresholds_are_post_hoc": True,
        },
        "population": {
            "synapses": len(context["synapse_ids"]),
            "synapse_bearing_v117_roots": len(all_roots),
            "active_path_roots": len(fragments),
            "singleton_confusers": len(all_roots) - len(fragments),
            "soma_seeds": len(seeds),
            "true_merge_pairs": true_pairs,
            "mixed_lineage_roots": contaminated,
        },
        "untouched_v117_baseline": baseline,
        "checkpoints": runs,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
