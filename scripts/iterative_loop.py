#!/usr/bin/env python3
"""Run repeated benchmark iterations and log per-run and per-iteration metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=5.0, help="Wall-clock budget for each iteration.")
    parser.add_argument("--iterations", type=int, default=None, help="Optional number of 5-minute iterations to run.")
    parser.add_argument(
        "--repeat-until-interrupt",
        action="store_true",
        help="Keep starting new iterations until Ctrl+C.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable to use.")
    parser.add_argument(
        "--cmd",
        default="-m neuronauts.run",
        help="Command passed to the selected Python executable.",
    )
    parser.add_argument("--log-dir", default="run_logs/latest", help="Top-level directory for iteration logs.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first non-zero return code.")
    parser.add_argument(
        "--config-path",
        default="neuronauts/run.py",
        help="File containing the experiment config block to snapshot.",
    )
    parser.add_argument(
        "--benchmark-mode",
        default="random",
        choices=["random", "fixed_validation"],
        help="Benchmark sampling policy.",
    )
    parser.add_argument(
        "--data-mode",
        default="synthetic",
        choices=["synthetic", "real"],
        help="Whether each run evaluates synthetic cases or real MICrONS boxes.",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=5,
        help="Synthetic cases per evaluation run.",
    )
    parser.add_argument("--real-boxes-per-eval", type=int, default=3, help="Real boxes to average per evaluation run.")
    parser.add_argument("--real-min-synapses", type=int, default=50, help="Minimum synapses required for a real box to count.")
    parser.add_argument(
        "--membrane-source",
        default="auto",
        choices=["auto", "cache", "sobel"],
        help="Membrane field source for real-data runs.",
    )
    parser.add_argument(
        "--membrane-cache-dir",
        default="cache/membranes",
        help="Directory containing cached membrane .npy volumes.",
    )
    return parser.parse_args()


def extract_metric(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def extract_metrics(text: str) -> dict[str, str | None]:
    return {
        "mode": extract_metric(r"mode=([a-z_]+)", text),
        "val_f1": extract_metric(r"val_f1\s*=\s*([0-9.]+)", text),
        "precision": extract_metric(r"P=([0-9.]+)", text),
        "recall": extract_metric(r"R=([0-9.]+)", text),
        "tp": extract_metric(r"TP=([0-9]+)", text),
        "fp": extract_metric(r"FP=([0-9]+)", text),
        "fn": extract_metric(r"FN=([0-9]+)", text),
        "neurons": extract_metric(r"\|\s*([0-9]+) neurons,", text),
        "edges": extract_metric(r"neurons,\s*([0-9]+) edges,", text),
        "unresolved": extract_metric(r"edges,\s*([0-9]+) unresolved", text),
        "case_results": str(len(re.findall(r"case_result ", text))),
    }


def extract_config_block(config_path: Path) -> str:
    text = config_path.read_text(encoding="utf-8")
    start = "# EXPERIMENT CONFIG"
    end = "# END CONFIG"
    start_idx = text.index(start)
    end_idx = text.index(end) + len(end)
    return text[start_idx:end_idx] + "\n"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_iteration(
    iteration_idx: int,
    iteration_dir: Path,
    base_cmd: list[str],
    minutes: float,
    benchmark_mode: str,
    cases: int,
    stop_on_error: bool,
) -> dict[str, float | int | str]:
    deadline = time.time() + minutes * 60.0
    summary_path = iteration_dir / "summary.tsv"
    summary_path.write_text(
        "run\trc\telapsed_s\tbenchmark_mode\tcases\tval_f1\tprecision\trecall\ttp\tfp\tfn\tcase_results\tneurons\tedges\tunresolved\tlog_path\n",
        encoding="utf-8",
    )

    f1_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    best_f1 = None
    run_idx = 0
    interrupted = False

    print(f"\n=== Iteration {iteration_idx:03d} ({minutes:.2f} min) ===")
    print(f"Iteration logs: {iteration_dir}")

    try:
        while time.time() < deadline:
            run_idx += 1
            run_started = time.time()
            proc = subprocess.run(
                list(base_cmd),
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            elapsed = time.time() - run_started
            output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
            metrics = extract_metrics(output)

            log_path = iteration_dir / f"run_{run_idx:03d}.log"
            log_path.write_text(output, encoding="utf-8")

            if metrics["val_f1"] is not None:
                f1_value = float(metrics["val_f1"])
                f1_values.append(f1_value)
                best_f1 = max(best_f1, f1_value) if best_f1 is not None else f1_value
            if metrics["precision"] is not None:
                precision_values.append(float(metrics["precision"]))
            if metrics["recall"] is not None:
                recall_values.append(float(metrics["recall"]))

            with summary_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    "\t".join(
                        [
                            str(run_idx),
                            str(proc.returncode),
                            f"{elapsed:.3f}",
                            benchmark_mode,
                            str(cases),
                            metrics["val_f1"] or "",
                            metrics["precision"] or "",
                            metrics["recall"] or "",
                            metrics["tp"] or "",
                            metrics["fp"] or "",
                            metrics["fn"] or "",
                            metrics["case_results"] or "",
                            metrics["neurons"] or "",
                            metrics["edges"] or "",
                            metrics["unresolved"] or "",
                            log_path.name,
                        ]
                    )
                    + "\n"
                )

            print(
                f"[it {iteration_idx:03d} run {run_idx:03d}] rc={proc.returncode} "
                f"elapsed={elapsed:.2f}s "
                f"val_f1={metrics['val_f1'] or 'n/a'} "
                f"P={metrics['precision'] or 'n/a'} "
                f"R={metrics['recall'] or 'n/a'} "
                f"TP={metrics['tp'] or 'n/a'} "
                f"FP={metrics['fp'] or 'n/a'} "
                f"FN={metrics['fn'] or 'n/a'} "
                f"best={f'{best_f1:.4f}' if best_f1 is not None else 'n/a'}"
            )

            if proc.returncode != 0 and stop_on_error:
                raise RuntimeError(f"iteration {iteration_idx} run {run_idx} failed; see {log_path}")
    except KeyboardInterrupt:
        interrupted = True

    stats = {
        "iteration": iteration_idx,
        "runs": run_idx,
        "mean_f1": mean(f1_values),
        "best_f1": max(f1_values) if f1_values else 0.0,
        "min_f1": min(f1_values) if f1_values else 0.0,
        "mean_precision": mean(precision_values),
        "mean_recall": mean(recall_values),
        "summary_path": summary_path.name,
        "interrupted": interrupted,
    }
    (iteration_dir / "iteration_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"Iteration {iteration_idx:03d} summary: "
        f"runs={run_idx} mean_f1={stats['mean_f1']:.4f} "
        f"best_f1={stats['best_f1']:.4f} min_f1={stats['min_f1']:.4f} "
        f"mean_P={stats['mean_precision']:.3f} mean_R={stats['mean_recall']:.3f}"
    )
    return stats


def main() -> int:
    args = parse_args()
    if not args.repeat_until_interrupt and args.iterations is None:
        args.iterations = 1

    started_at = time.time()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config_path)
    config_block = extract_config_block(config_path)
    config_hash = hashlib.sha256(config_block.encode("utf-8")).hexdigest()[:12]
    (log_dir / "config_snapshot.txt").write_text(config_block, encoding="utf-8")

    base_cmd = [
        args.python,
        *shlex.split(args.cmd),
        "--quiet",
        "--data-mode",
        args.data_mode,
        "--cases",
        str(args.cases),
        "--benchmark-mode",
        args.benchmark_mode,
        "--real-boxes-per-eval",
        str(args.real_boxes_per_eval),
        "--real-min-synapses",
        str(args.real_min_synapses),
        "--membrane-source",
        args.membrane_source,
        "--membrane-cache-dir",
        args.membrane_cache_dir,
    ]

    session = {
        "started_at_epoch": started_at,
        "iteration_minutes": args.minutes,
        "iterations": args.iterations,
        "repeat_until_interrupt": args.repeat_until_interrupt,
        "command": base_cmd,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "benchmark_mode": args.benchmark_mode,
        "data_mode": args.data_mode,
        "cases": args.cases,
        "real_boxes_per_eval": args.real_boxes_per_eval,
        "real_min_synapses": args.real_min_synapses,
        "membrane_source": args.membrane_source,
        "membrane_cache_dir": args.membrane_cache_dir,
    }
    (log_dir / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")

    iteration_summary = log_dir / "iteration_summary.tsv"
    iteration_summary.write_text(
        "iteration\truns\tmean_f1\tbest_f1\tmin_f1\tmean_precision\tmean_recall\tsummary_path\n",
        encoding="utf-8",
    )

    print(f"Per-iteration budget: {args.minutes:.2f} minutes")
    print(f"Command: {' '.join(base_cmd)}")
    print(f"Logs: {log_dir}")
    print(f"Config: {config_path} (sha256:{config_hash})")
    print(f"Data mode: {args.data_mode}")
    print(f"Benchmark mode: {args.benchmark_mode}")
    print(f"Cases per evaluation: {args.cases}")
    if args.data_mode == "real":
        print(f"Real boxes per evaluation: {args.real_boxes_per_eval}")
        print(f"Real min synapses: {args.real_min_synapses}")
        print(f"Membrane source: {args.membrane_source}")
        print(f"Membrane cache dir: {args.membrane_cache_dir}")
    if args.repeat_until_interrupt:
        print("Iterations: until Ctrl+C")
    else:
        print(f"Iterations: {args.iterations}")

    iteration_idx = 0
    all_iteration_means: list[float] = []
    try:
        while True:
            if not args.repeat_until_interrupt and iteration_idx >= int(args.iterations):
                break
            iteration_idx += 1
            iteration_dir = log_dir / f"iteration_{iteration_idx:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            stats = run_iteration(
                iteration_idx=iteration_idx,
                iteration_dir=iteration_dir,
                base_cmd=base_cmd,
                minutes=args.minutes,
                benchmark_mode=args.benchmark_mode,
                cases=args.cases,
                stop_on_error=args.stop_on_error,
            )
            all_iteration_means.append(float(stats["mean_f1"]))
            with iteration_summary.open("a", encoding="utf-8") as fh:
                fh.write(
                    "\t".join(
                        [
                            str(stats["iteration"]),
                            str(stats["runs"]),
                            f"{float(stats['mean_f1']):.4f}",
                            f"{float(stats['best_f1']):.4f}",
                            f"{float(stats['min_f1']):.4f}",
                            f"{float(stats['mean_precision']):.4f}",
                            f"{float(stats['mean_recall']):.4f}",
                            str(stats["summary_path"]),
                        ]
                    )
                    + "\n"
                )
            print(
                f"Overall progress after iteration {iteration_idx:03d}: "
                f"latest_mean_f1={float(stats['mean_f1']):.4f} "
                f"best_iteration_mean_f1={max(all_iteration_means):.4f}"
            )
            if bool(stats["interrupted"]):
                print("\nInterrupted by user. Writing final summaries.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user. Writing final summaries.")

    if all_iteration_means:
        print(f"Completed iterations: {len(all_iteration_means)}")
        print(f"Best iteration mean F1: {max(all_iteration_means):.4f}")
    print(f"Iteration summary: {iteration_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
