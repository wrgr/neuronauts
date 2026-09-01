#!/usr/bin/env python3
"""EXP-051: real synapse-dense v117 soma-seeded grammar assembly.

Candidates are real v117 roots at either endpoint of real synapses centered in
the box. There is no synthetic fracture, frankenmerge injection, truth-selected
candidate list, or synthetic fallback.
"""

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
    Fragment,
    assert_real_root_ids,
    build_candidate_edges_batched,
    partition_metrics,
    single_soma_compliance,
    skeleton_from_observed_points,
    soma_seeded_assemble,
)
from neuronauts.shared_grammar_model import load_shared_grammar_model
from treestitch.realworld import _load_nucleus_somas


def parse_bbox(value: str) -> tuple:
    values = tuple(float(item) for item in value.split(","))
    if len(values) != 6:
        raise argparse.ArgumentTypeError("bbox requires x0,y0,z0,x1,y1,z1")
    return values[:3], values[3:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_candidates(
    bbox: tuple,
    *,
    target_timestamp: int,
    token: str,
) -> tuple[dict[int, np.ndarray], dict[int, tuple[int, float, int]], dict]:
    """Fetch public Delta synapses and resolve exact v117/v1412 lineage."""
    from neuronauts.bulk_synapses import fetch_synapses_bulk

    cache_key = hashlib.sha1(json.dumps(bbox).encode()).hexdigest()[:16]
    cache_path = Path("/tmp") / f"neuronauts_exp051_synapses_{cache_key}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            bulk = {key: cached[key] for key in cached.files}
    else:
        raw = fetch_synapses_bulk(
            bbox, token, version=117, use_version_roots=True)
        keys = (
            "pre_pt_nm", "post_pt_nm", "pre_root_id", "post_root_id",
            "pre_supervoxel_id", "post_supervoxel_id", "synapse_id",
        )
        bulk = {key: np.asarray(raw[key]) for key in keys}
        np.savez_compressed(cache_path, **bulk)

    mapped = {}
    points: dict[int, list[np.ndarray]] = {}
    votes: dict[int, list[int]] = {}
    role_counts: dict[int, Counter] = {}
    for side, other in (("pre", "post"), ("post", "pre")):
        roots = np.asarray(bulk[f"{side}_root_id"], dtype=np.uint64)
        supervoxels = np.asarray(bulk[f"{side}_supervoxel_id"], dtype=np.uint64)
        labels = L.roots_at(supervoxels, target_timestamp, token=token)
        if labels is None:
            raise RuntimeError(f"{side}-side v1412 lineage mapping failed")
        mapped[side] = {
            "v117": roots,
            "target": labels,
            "other_supervoxels": bulk[f"{other}_supervoxel_id"],
        }
        for root, label, position in zip(roots, labels, bulk[f"{side}_pt_nm"]):
            root_id, target_id = int(root), int(label)
            if root_id <= 0:
                continue
            points.setdefault(root_id, []).append(np.asarray(position, dtype=np.float32))
            role_counts.setdefault(root_id, Counter())[side] += 1
            if target_id > 0:
                votes.setdefault(root_id, []).append(target_id)

    label_map = {}
    for root, labels in votes.items():
        unique, counts = np.unique(labels, return_counts=True)
        best = int(np.argmax(counts))
        label_map[root] = (
            int(unique[best]), float(counts[best] / counts.sum()), int(len(unique)))
    point_map = {
        root: np.unique(np.stack(values), axis=0).astype(np.float32, copy=False)
        for root, values in points.items()
    }
    context = {
        "synapse_ids": bulk["synapse_id"],
        "pre_v117": mapped["pre"]["v117"],
        "post_v117": mapped["post"]["v117"],
        "pre_target": mapped["pre"]["target"],
        "post_target": mapped["post"]["target"],
        "role_counts": {root: dict(counts) for root, counts in role_counts.items()},
    }
    return point_map, label_map, context


def exact_soma_counts(bbox: tuple, *, token: str) -> dict[int, int]:
    somas = _load_nucleus_somas()
    lower, upper = np.asarray(bbox[0]), np.asarray(bbox[1])
    positions = np.stack(
        [somas["x_nm"], somas["y_nm"], somas["z_nm"]], axis=1)
    inside = np.all((positions >= lower) & (positions < upper), axis=1)
    roots = L.roots_at(somas["sv"][inside], L.V117_TIMESTAMP, token=token)
    if roots is None:
        raise RuntimeError("exact nucleus-supervoxel to v117 mapping failed")
    return dict(Counter(int(root) for root in roots if int(root) > 0))


def circuit_f1(context: dict, prediction: dict[int, int]) -> dict:
    """Pairwise joint pre/post F1 using linear-memory contingency counts."""
    pre, post = context["pre_v117"], context["post_v117"]
    pre_target, post_target = context["pre_target"], context["post_target"]
    keep = np.asarray([
        int(left) in prediction and int(right) in prediction
        and int(left_target) > 0 and int(right_target) > 0
        for left, right, left_target, right_target
        in zip(pre, post, pre_target, post_target)
    ])
    if not np.any(keep):
        return {"circuit_f1": None, "n_circuit_synapses": 0}
    predicted_joint = np.column_stack([
        [prediction[int(root)] for root in pre[keep]],
        [prediction[int(root)] for root in post[keep]],
    ])
    true_joint = np.column_stack([pre_target[keep], post_target[keep]])
    _, true_inverse = np.unique(true_joint, axis=0, return_inverse=True)
    _, pred_inverse = np.unique(predicted_joint, axis=0, return_inverse=True)
    n_pred = int(pred_inverse.max()) + 1
    joint = np.bincount(true_inverse.astype(np.int64) * n_pred + pred_inverse)
    true_counts, pred_counts = np.bincount(true_inverse), np.bincount(pred_inverse)
    choose2 = lambda values: float(np.sum(values * (values - 1) // 2))
    true_positive = choose2(joint)
    true_pairs, predicted_pairs = choose2(true_counts), choose2(pred_counts)
    precision = true_positive / predicted_pairs if predicted_pairs else 1.0
    recall = true_positive / true_pairs if true_pairs else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)
    return {
        "circuit_f1": float(f1),
        "circuit_precision": float(precision),
        "circuit_recall": float(recall),
        "n_circuit_synapses": int(keep.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bbox-nm", type=parse_bbox,
        default=parse_bbox("718592,498592,580640,748592,528592,610640"),
    )
    parser.add_argument("--target-version", type=int, default=1412)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "models/shared_grammar_raw_skel_50e.pt",
    )
    parser.add_argument("--min-root-observations", type=int, default=10)
    parser.add_argument("--max-path-points", type=int, default=96)
    parser.add_argument("--max-distance-nm", type=float, default=2500.0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--score-sweep", default="0,1,2,3,4,5,6")
    parser.add_argument("--max-fragments", type=int, default=0,
                        help="debug-only cap; zero uses every eligible root")
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/exp051_real_dense.json",
    )
    args = parser.parse_args()

    token = sys.stdin.readline().strip() if args.token_stdin else L.DEFAULT_TOKEN
    if not token:
        raise SystemExit("CAVE token is required")
    started = time.time()
    target_timestamp = L.version_timestamp(args.target_version, token=token)
    if target_timestamp is None:
        raise SystemExit("target timestamp unavailable; refusing label substitution")
    if not args.checkpoint.is_file():
        raise SystemExit(f"trained grammar checkpoint missing: {args.checkpoint}")

    print("[1/6] fetching real synapses and exact endpoint lineage", flush=True)
    point_map, label_map, context = fetch_candidates(
        args.bbox_nm, target_timestamp=target_timestamp, token=token)
    all_roots = sorted(point_map)
    assert_real_root_ids(all_roots)
    print(f"      synapses={len(context['synapse_ids'])}; roots={len(all_roots)}", flush=True)

    print("[2/6] resolving exact soma seeds", flush=True)
    soma_counts = exact_soma_counts(args.bbox_nm, token=token)
    seeds = sorted(set(all_roots) & set(soma_counts))
    if not seeds:
        raise SystemExit("no exact soma seed intersects the candidate population")
    if any(soma_counts[root] > 1 for root in seeds):
        raise SystemExit("multi-soma v117 root cannot be assembled atomically")

    selected = sorted(
        {root for root in all_roots
         if len(point_map[root]) >= args.min_root_observations} | set(seeds))
    if args.max_fragments:
        nonseeds = sorted(
            (root for root in selected if root not in seeds),
            key=lambda root: (-len(point_map[root]), root),
        )
        selected = sorted(seeds + nonseeds[:max(0, args.max_fragments - len(seeds))])
    print(f"      soma seeds={len(seeds)}; path roots={len(selected)}", flush=True)

    print("[3/6] building real observation path graphs", flush=True)
    fragments = []
    contaminated = 0
    for root in selected:
        vertices, edges = skeleton_from_observed_points(
            point_map[root], max_points=args.max_path_points)
        if len(edges) == 0:
            continue
        label, purity, n_labels = label_map.get(root, (0, 0.0, 0))
        contaminated += int(n_labels > 1)
        fragments.append(Fragment(
            root, vertices, edges, soma_counts.get(root, 0), label, purity))
    if not (set(seeds) & {fragment.root_id for fragment in fragments}):
        raise SystemExit("no soma seed has enough real observations for a path")

    print("[4/6] scoring joins with the trained grammar", flush=True)
    grammar = load_shared_grammar_model(args.checkpoint)
    mode = grammar.path_feature_mode
    mip2_nm = np.asarray([32.0, 32.0, 40.0], dtype=np.float32)
    featurize = lambda points: featurize_path_points(points / mip2_nm, mode=mode)
    edges = build_candidate_edges_batched(
        fragments, grammar, featurize, max_distance_nm=args.max_distance_nm)
    print(f"      candidate joins={len(edges)}", flush=True)

    def with_confusers(active: dict[int, int]) -> dict[int, int]:
        prediction = dict(active)
        next_cluster = max(prediction.values(), default=-1) + 1
        for root in all_roots:
            if root not in prediction:
                prediction[root] = next_cluster
                next_cluster += 1
        return prediction

    print("[5/6] growing competing soma-seeded pathways", flush=True)
    prediction = with_confusers(soma_seeded_assemble(
        fragments, edges, min_score=args.min_score))
    metrics = partition_metrics(fragments, prediction)
    metrics.update(single_soma_compliance(fragments, prediction))
    metrics.update(circuit_f1(context, prediction))

    baseline = partition_metrics(
        fragments, {fragment.root_id: i for i, fragment in enumerate(fragments)})
    sweep = {}
    for threshold in [float(value) for value in args.score_sweep.split(",") if value]:
        sweep_prediction = with_confusers(soma_seeded_assemble(
            fragments, edges, min_score=threshold))
        sweep[str(threshold)] = partition_metrics(fragments, sweep_prediction)

    target_counts = Counter(
        fragment.gt_label for fragment in fragments if fragment.gt_label > 0)
    true_merge_pairs = int(sum(
        count * (count - 1) // 2 for count in target_counts.values()))
    side_um = (np.asarray(args.bbox_nm[1]) - np.asarray(args.bbox_nm[0])) / 1000.0
    volume_um3 = float(np.prod(side_um))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = {
        "experiment": "EXP-051 real synapse-dense v117 soma-seeded grammar",
        "provenance": {
            "git_commit": commit,
            "bbox_nm": [list(args.bbox_nm[0]), list(args.bbox_nm[1])],
            "segmentation_version": 117,
            "v117_timestamp": L.V117_TIMESTAMP,
            "target_version": args.target_version,
            "target_timestamp": target_timestamp,
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256(args.checkpoint),
            "path_feature_mode": mode,
            "candidate_policy": (
                "all v117 roots at either endpoint of every real synapse whose "
                "center lies in bbox; public Delta export"),
            "seed_policy": "exact nucleus supervoxel lineage containment",
            "min_root_observations": args.min_root_observations,
            "ground_truth_used_during_inference": False,
            "synthetic_fallback": False,
            "score_sweep_is_post_hoc": True,
        },
        "population": {
            "volume_um3": volume_um3,
            "synapses": len(context["synapse_ids"]),
            "synapse_bearing_v117_roots": len(all_roots),
            "selected_roots": len(selected),
            "singleton_confuser_roots": len(all_roots) - len(fragments),
            "usable_path_graphs": len(fragments),
            "exact_soma_seeds": len(seeds),
            "contaminated_v117_roots": contaminated,
            "fragment_density_per_1000_um3": 1000.0 * len(all_roots) / volume_um3,
            "v1412_label_coverage": len(label_map) / max(len(all_roots), 1),
            "n_v1412_target_roots_active": len(target_counts),
            "true_fragment_merge_pairs_active": true_merge_pairs,
        },
        "assembly": {
            "candidate_edges": len(edges),
            "accepted_non_singleton_fragments": int(sum(
                Counter(prediction.values())[cluster] > 1
                for cluster in prediction.values())),
            "min_score": args.min_score,
            "max_distance_nm": args.max_distance_nm,
            "score_quantiles": (
                {str(q): float(np.quantile([edge.score for edge in edges], q))
                 for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)}
                if edges else {}),
        },
        "metrics": metrics,
        "untouched_v117_baseline": baseline,
        "post_hoc_threshold_sweep": sweep,
        "predicted_cluster_by_v117_root": {
            str(root): int(cluster) for root, cluster in prediction.items()},
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("[6/6] complete", flush=True)
    print(json.dumps({"population": result["population"], "metrics": metrics}, indent=2))
    print(f"result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
