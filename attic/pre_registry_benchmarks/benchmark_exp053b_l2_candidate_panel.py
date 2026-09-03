#!/usr/bin/env python3
"""EXP-053B: candidate-panel recall from bounded real v117 L2 geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuronauts.data import lineage as L
from neuronauts.l2_candidate_panel import (
    candidate_endpoint_pairs, endpoint_records, filter_candidate_pairs,
    panel_sizes,
)
from neuronauts.real_dense_soma import (
    Fragment, assert_real_root_ids, skeleton_from_observed_points,
    true_merge_pair_count,
)
from scripts.benchmark_exp051_real_dense_soma_grammar import (
    exact_soma_lineage, fetch_candidates,
)
from treestitch.realworld import _l2_nodes_with_coords

BBOX = ((844776.0, 700136.0, 875040.0),
        (874776.0, 730136.0, 905040.0))
ANCHOR_TARGET_ROOT = 864691135106016333


def true_pairs(fragments):
    groups = {}
    for fragment in fragments:
        if fragment.gt_label > 0:
            groups.setdefault(int(fragment.gt_label), []).append(fragment.root_id)
    return {(left, right) for roots in groups.values()
            for i, left in enumerate(sorted(set(roots)))
            for right in sorted(set(roots))[i + 1:]}


def panel_summary(values):
    return ({"median": float(np.median(values)),
             "p90": float(np.quantile(values, .9)),
             "max": int(np.max(values))}
            if len(values) else {"median": 0.0, "p90": 0.0, "max": 0})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-version", type=int, default=1412)
    parser.add_argument("--min-root-observations", type=int, default=10)
    parser.add_argument("--max-l2-points", type=int, default=512)
    parser.add_argument("--geometry-halo-nm", type=float, default=10_000.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--radii-um", default="0.5,1,2,2.5,5,10")
    parser.add_argument("--cones-deg", default="30,45,60,90,180")
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--cache-dir", type=Path,
                        default=ROOT / "cache/exp053b_l2_roots")
    parser.add_argument("--panel-cache", type=Path,
                        default=ROOT / "cache/exp053b_candidate_panel.npz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/exp053b_l2_candidate_panel.json")
    args = parser.parse_args()

    token = sys.stdin.readline().strip() if args.token_stdin else L.DEFAULT_TOKEN
    if not token:
        raise SystemExit("CAVE token is required")
    started = time.time()
    target_timestamp = L.version_timestamp(args.target_version, token=token)
    if target_timestamp is None:
        raise SystemExit("target timestamp unavailable")

    print("[1/4] loading fixed real edit-bearing population", flush=True)
    point_map, label_map, context = fetch_candidates(
        BBOX, target_timestamp=target_timestamp, token=token)
    all_roots = sorted(point_map)
    assert_real_root_ids(all_roots)
    soma_counts, target_soma_counts = exact_soma_lineage(
        BBOX, target_timestamp=target_timestamp, token=token)
    if ANCHOR_TARGET_ROOT not in target_soma_counts:
        raise SystemExit("fixed proofread soma anchor failed validation")
    selected = sorted({root for root in all_roots
                       if len(point_map[root]) >= args.min_root_observations}
                      | (set(all_roots) & set(soma_counts)))

    reference = []
    for root in selected:
        vertices, edges = skeleton_from_observed_points(point_map[root])
        if len(edges) == 0:
            continue
        label, purity, _ = label_map.get(root, (0, 0.0, 0))
        reference.append(Fragment(root, vertices, edges,
                                  soma_counts.get(root, 0), label, purity))
    expected = true_pairs(reference)
    if len(expected) != 14 or true_merge_pair_count(reference) != 14:
        raise SystemExit(f"edit-signal gate failed: expected 14, got {len(expected)}")
    selected = sorted(fragment.root_id for fragment in reference)
    print(f"      active={len(selected)}; true_pairs={len(expected)}", flush=True)

    voxel = np.asarray([8.0, 8.0, 40.0])
    lower = np.asarray(BBOX[0]) - args.geometry_halo_nm
    upper = np.asarray(BBOX[1]) + args.geometry_halo_nm
    bounds = tuple((float(lower[i] / voxel[i]), float(upper[i] / voxel[i]))
                   for i in range(3))
    cache_key = hashlib.sha1(json.dumps({
        "bbox": BBOX, "halo": args.geometry_halo_nm,
        "max_l2_points": args.max_l2_points}, sort_keys=True).encode()
    ).hexdigest()[:12]
    cache_dir = args.cache_dir / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_root(root):
        path = cache_dir / f"{root}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                return root, cached["points_nm"].astype(np.float32)
        _ids, points = _l2_nodes_with_coords(
            root, token=token, bounds_seg_vox=bounds)
        points = np.unique(points[np.all((points >= lower) & (points < upper),
                                         axis=1)], axis=0).astype(np.float32)
        if len(points) > args.max_l2_points:
            points = points[np.linspace(0, len(points) - 1,
                                        args.max_l2_points, dtype=int)]
        np.savez_compressed(path, points_nm=points)
        return root, points

    print("[2/4] fetching bounded real L2 coordinates", flush=True)
    l2_points = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(fetch_root, root) for root in selected]
        for done, future in enumerate(as_completed(futures), start=1):
            root, points = future.result()
            l2_points[root] = points
            if done % 100 == 0 or done == len(futures):
                usable = sum(len(points) >= 2 for points in l2_points.values())
                print(f"      {done}/{len(futures)} roots; usable={usable}", flush=True)

    fragments = []
    for root in selected:
        vertices, edges = skeleton_from_observed_points(
            l2_points.get(root, np.zeros((0, 3), np.float32)),
            max_points=args.max_l2_points)
        if len(edges) == 0:
            continue
        label, purity, _ = label_map.get(root, (0, 0.0, 0))
        fragments.append(Fragment(root, vertices, edges,
                                  soma_counts.get(root, 0), label, purity))
    covered_roots = {fragment.root_id for fragment in fragments}
    covered_expected = {pair for pair in expected
                        if pair[0] in covered_roots and pair[1] in covered_roots}

    print("[3/4] building label-blind endpoint candidate panels", flush=True)
    records = endpoint_records(fragments)
    max_radius_nm = max(map(float, args.radii_um.split(","))) * 1000
    raw_pairs = candidate_endpoint_pairs(records,
                                         max_distance_nm=max_radius_nm)
    args.panel_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.panel_cache,
        left_root=np.asarray([min(p.left_root, p.right_root) for p in raw_pairs]),
        right_root=np.asarray([max(p.left_root, p.right_root) for p in raw_pairs]),
        distance_nm=np.asarray([p.distance_nm for p in raw_pairs]),
        facing=np.asarray([p.facing for p in raw_pairs]),
        tangent_opposition=np.asarray([p.tangent_opposition for p in raw_pairs]),
        same_target=np.asarray([tuple(sorted((p.left_root, p.right_root))) in expected
                                for p in raw_pairs], dtype=bool))

    grid = {}
    best = None
    roots_for_sizes = sorted(covered_roots)
    for radius_um in map(float, args.radii_um.split(",")):
        for cone in map(float, args.cones_deg.split(",")):
            kept = filter_candidate_pairs(raw_pairs,
                                          max_distance_nm=radius_um * 1000,
                                          cone_degrees=cone)
            keys = {tuple(sorted((p.left_root, p.right_root))) for p in kept}
            recovered = len(expected & keys)
            recovered_covered = len(covered_expected & keys)
            sizes = panel_sizes(roots_for_sizes, kept)
            metrics = {
                "candidate_pairs": len(kept),
                "true_pairs_recovered": recovered,
                "recall_all_true_pairs": recovered / len(expected),
                "recall_l2_covered_true_pairs": (recovered_covered /
                                                   len(covered_expected)
                                                   if covered_expected else 0.0),
                "panel_size": panel_summary(sizes),
            }
            grid[f"r{radius_um:g}_cone{cone:g}"] = metrics
            objective = (metrics["recall_all_true_pairs"],
                         -metrics["panel_size"]["median"],
                         -metrics["candidate_pairs"])
            if best is None or objective > best[0]:
                best = (objective, radius_um, cone, metrics)

    print("[4/4] writing aggregate evaluation", flush=True)
    result = {
        "experiment": "EXP-053B real-L2 candidate-panel recall",
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "bbox_nm": [list(BBOX[0]), list(BBOX[1])],
            "geometry_halo_nm": args.geometry_halo_nm,
            "target_version": args.target_version,
            "target_timestamp": target_timestamp,
            "root_selection": "synapse-table v117 roots with >=10 observations, plus soma roots",
            "candidate_generation_used_target_lineage": False,
            "labels_used_only_for_evaluation": True,
            "synthetic_fallback": False,
            "l2_topology": "MST over real bounded L2 rep_coord_nm points",
        },
        "population": {
            "synapses": len(context["synapse_ids"]),
            "eligible_roots": len(selected),
            "l2_covered_roots": len(covered_roots),
            "endpoint_paths": len(records),
            "true_merge_pairs": len(expected),
            "l2_covered_true_pairs": len(covered_expected),
            "raw_pairs_within_max_radius": len(raw_pairs),
        },
        "best_recall_configuration": {
            "radius_um": best[1], "cone_degrees": best[2], **best[3]},
        "success_criterion": {
            "required_recall": .9,
            "max_median_panel_size": 20,
            "passed": bool(best[3]["recall_all_true_pairs"] >= .9 and
                           best[3]["panel_size"]["median"] <= 20),
        },
        "grid": grid,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
