#!/usr/bin/env python3
"""EXP-056: label-blind atomization of real mixed-lineage v117 roots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neuronauts.atomization import (
    cut_components, euclidean_mst, metrics_from_counts, pair_counts,
)
from neuronauts.bulk_synapses import fetch_synapses_bulk
from neuronauts.data import lineage as L

BBOX = ((844776.0, 700136.0, 875040.0),
        (874776.0, 730136.0, 905040.0))


def add_counts(total, counts):
    for key, value in counts.items():
        total[key] += int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-version", type=int, default=1412)
    parser.add_argument("--min-observations", type=int, default=10)
    parser.add_argument("--max-observations", type=int, default=512)
    parser.add_argument("--token-stdin", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/exp056_real_root_atomization.json")
    args = parser.parse_args()

    token = sys.stdin.readline().strip() if args.token_stdin else L.DEFAULT_TOKEN
    if not token:
        raise SystemExit("CAVE token is required")
    started = time.time()
    target_timestamp = L.version_timestamp(args.target_version, token=token)
    if target_timestamp is None:
        raise SystemExit("target timestamp unavailable")

    print("[1/3] loading real synapse observations and lineage", flush=True)
    raw = fetch_synapses_bulk(BBOX, token, version=117,
                              use_version_roots=True)
    observations = defaultdict(list)
    for side in ("pre", "post"):
        roots = np.asarray(raw[f"{side}_root_id"], dtype=np.uint64)
        supervoxels = np.asarray(raw[f"{side}_supervoxel_id"], dtype=np.uint64)
        points = np.asarray(raw[f"{side}_pt_nm"], dtype=np.float32)
        labels = L.roots_at(supervoxels, target_timestamp, token=token)
        if labels is None:
            raise SystemExit(f"{side} target lineage resolution failed")
        for root, point, label in zip(roots, points, labels):
            if int(root) > 0 and int(label) > 0:
                observations[int(root)].append((point, int(label)))

    mixed = []
    for root, values in observations.items():
        labels = {label for _point, label in values}
        if len(values) >= args.min_observations and len(labels) > 1:
            mixed.append((root, values))
    if len(mixed) != 116:
        raise SystemExit(f"mixed-root validity gate failed: expected 116, got {len(mixed)}")
    print(f"      mixed roots={len(mixed)}", flush=True)

    prepared = []
    for _root, values in mixed:
        if len(values) > args.max_observations:
            take = np.linspace(0, len(values) - 1,
                               args.max_observations, dtype=int)
            values = [values[index] for index in take]
        points = np.stack([point for point, _label in values])
        labels = np.asarray([label for _point, label in values], dtype=np.int64)
        edges, lengths = euclidean_mst(points)
        if len(edges):
            prepared.append((points, labels, edges, lengths))
    print(f"[2/3] sweeping label-blind cuts on {len(prepared)} roots", flush=True)

    rules = {"atomic": lambda lengths: float("inf")}
    for threshold_um in (.25, .5, 1, 2, 5, 10):
        rules[f"absolute_{threshold_um:g}um"] = (
            lambda lengths, value=threshold_um: value * 1000)
    for quantile in (.5, .75, .9, .95, .99):
        rules[f"quantile_{quantile:g}"] = (
            lambda lengths, value=quantile: float(np.quantile(lengths, value)))
    for robust_z in (2, 3, 4, 6):
        def robust(lengths, value=robust_z):
            median = float(np.median(lengths))
            mad = float(np.median(np.abs(lengths - median)))
            return median + value * 1.4826 * mad
        rules[f"robust_z{robust_z:g}"] = robust

    sweep = {}
    for name, cutoff in rules.items():
        totals = {key: 0 for key in ("tp", "fp", "fn", "tn")}
        macro_f1 = []
        perfect = 0
        components = []
        for points, labels, edges, lengths in prepared:
            prediction = cut_components(
                len(points), edges, lengths, cutoff(lengths))
            counts = pair_counts(labels, prediction)
            add_counts(totals, counts)
            metrics = metrics_from_counts(counts)
            macro_f1.append(metrics["pair_f1"])
            perfect += int(metrics["pair_precision"] == 1.0 and
                           metrics["pair_recall"] == 1.0)
            components.append(len(np.unique(prediction)))
        sweep[name] = {
            **metrics_from_counts(totals),
            "macro_root_pair_f1": float(np.mean(macro_f1)),
            "perfect_roots": perfect,
            "perfect_root_fraction": perfect / len(prepared),
            "median_components_per_root": float(np.median(components)),
            "pair_counts": totals,
        }

    candidates = [(metrics["pair_f1"], metrics["pair_recall"], name)
                  for name, metrics in sweep.items()
                  if name != "atomic" and metrics["pair_recall"] >= .9]
    best_name = max(candidates)[2] if candidates else None
    best = sweep.get(best_name) if best_name else None
    baseline = sweep["atomic"]

    print("[3/3] writing aggregate evaluation", flush=True)
    result = {
        "experiment": "EXP-056 real-root atomization",
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "bbox_nm": [list(BBOX[0]), list(BBOX[1])],
            "target_version": args.target_version,
            "target_timestamp": target_timestamp,
            "substrate": "real pre/post synapse endpoint observations",
            "cut_inference_used_target_lineage": False,
            "labels_used_only_for_evaluation": True,
            "synthetic_fallback": False,
            "sweep_is_post_hoc": True,
        },
        "population": {
            "mixed_lineage_roots": len(mixed),
            "evaluated_roots": len(prepared),
            "min_observations": args.min_observations,
            "max_observations_per_root": args.max_observations,
        },
        "atomic_baseline": baseline,
        "best_rule_with_pair_recall_at_least_0.9": (
            {"rule": best_name, **best} if best else None),
        "success_criterion": {
            "required_pair_recall": .9,
            "required_cross_lineage_split_recall": .5,
            "passed": bool(best and best["cross_lineage_split_recall"] >= .5),
        },
        "sweep": sweep,
        "elapsed_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
