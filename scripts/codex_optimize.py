#!/usr/bin/env python3
"""Run a Codex-driven outer optimization loop over neuronauts/run.py."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_FILE = REPO_ROOT / "neuronauts" / "run.py"
PROGRAM_FILE = REPO_ROOT / "program.md"
TEST_CMD = [".venv/bin/python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
VALIDATION_CMD = [
    ".venv/bin/python",
    "-m",
    "neuronauts.run",
    "--quiet",
    "--data-mode",
    "real",
    "--real-boxes-per-eval",
    "3",
    "--real-min-synapses",
    "50",
    "--membrane-source",
    "auto",
    "--membrane-cache-dir",
    "cache/membranes",
]
LOOP_SCRIPT = REPO_ROOT / "scripts" / "iterative_loop.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minutes", type=float, default=5.0, help="Wall-clock budget for accepted iteration runs.")
    parser.add_argument("--iterations", type=int, default=None, help="Optional number of optimizer iterations.")
    parser.add_argument(
        "--repeat-until-interrupt",
        action="store_true",
        help="Keep proposing changes until Ctrl+C.",
    )
    parser.add_argument("--log-dir", default="run_logs/codex_optimize", help="Directory for optimizer artifacts.")
    parser.add_argument("--model", default=None, help="Optional Codex model override.")
    parser.add_argument("--codex-bin", default="codex", help="Codex CLI executable.")
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.0,
        help="Minimum fixed-validation F1 improvement required to keep a change.",
    )
    parser.add_argument(
        "--skip-live-loop",
        action="store_true",
        help="Skip the accepted-change timed loop and only use fixed validation.",
    )
    return parser.parse_args()


def resolve_executable(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    if Path(executable).exists():
        return str(Path(executable).resolve())
    fallback_paths = [
        Path.home() / ".cursor/extensions/openai.chatgpt-26.311.21342-darwin-arm64/bin/macos-aarch64/codex",
    ]
    for candidate in fallback_paths:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"could not find executable: {executable}")


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def extract_metric(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def parse_validation_metrics(text: str) -> dict[str, float | int | None]:
    def as_float(pattern: str) -> float | None:
        value = extract_metric(pattern, text)
        return float(value) if value is not None else None

    def as_int(pattern: str) -> int | None:
        value = extract_metric(pattern, text)
        return int(value) if value is not None else None

    return {
        "val_f1": as_float(r"val_f1\s*=\s*([0-9.]+)"),
        "precision": as_float(r"P=([0-9.]+)"),
        "recall": as_float(r"R=([0-9.]+)"),
        "tp": as_int(r"TP=([0-9]+)"),
        "fp": as_int(r"FP=([0-9]+)"),
        "fn": as_int(r"FN=([0-9]+)"),
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_recommendation(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "no recommendation text"


def diff_text(before_path: Path, after_path: Path, cwd: Path) -> str:
    proc = run_command(["git", "diff", "--no-index", "--", str(before_path), str(after_path)], cwd=cwd)
    return proc.stdout + proc.stderr


def build_prompt(
    *,
    program_text: str,
    baseline_metrics: dict[str, float | int | None],
    recent_summary: list[dict[str, object]],
) -> str:
    summary_lines = []
    for row in recent_summary[-5:]:
        summary_lines.append(
            "- iter {iteration}: decision={decision} baseline_f1={baseline_f1} candidate_f1={candidate_f1} note={note}".format(
                iteration=row["iteration"],
                decision=row["decision"],
                baseline_f1=row["baseline_f1"],
                candidate_f1=row["candidate_f1"],
                note=row["note"],
            )
        )
    recent_text = "\n".join(summary_lines) if summary_lines else "- no prior optimizer iterations yet"
    return f"""You are the outer optimization agent for this repository.

Read and follow this research brief:

{program_text}

Hard constraints for this iteration:
- Edit only neuronauts/run.py.
- Make exactly one focused experiment change.
- Do not edit any other files.
- Do not add shims, wrappers, or new abstractions.
- Keep the code simple.

Current accepted fixed-validation baseline:
- val_f1={baseline_metrics['val_f1']}
- precision={baseline_metrics['precision']}
- recall={baseline_metrics['recall']}
- TP={baseline_metrics['tp']} FP={baseline_metrics['fp']} FN={baseline_metrics['fn']}

Recent optimizer history:
{recent_text}

Task:
1. Inspect neuronauts/run.py.
2. Make one targeted change that is likely to improve real MICrONS fixed-validation performance.
3. Stop after editing neuronauts/run.py.

Do not run long benchmark loops yourself; the wrapper script will evaluate and decide whether to keep or revert your change.
"""


def main() -> int:
    args = parse_args()
    if not args.repeat_until_interrupt and args.iterations is None:
        args.iterations = 1

    codex_bin = resolve_executable(args.codex_bin)

    log_dir = REPO_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    session_path = log_dir / "session.json"
    summary_path = log_dir / "optimizer_summary.tsv"
    summary_path.write_text(
        "iteration\tdecision\tbaseline_f1\tcandidate_f1\tdelta_f1\tbaseline_precision\tcandidate_precision\tbaseline_recall\tcandidate_recall\trun_hash_before\trun_hash_after\trecommendation\tnote\n",
        encoding="utf-8",
    )

    session = {
        "started_at_epoch": time.time(),
        "minutes": args.minutes,
        "iterations": args.iterations,
        "repeat_until_interrupt": args.repeat_until_interrupt,
        "run_file": str(RUN_FILE),
        "program_file": str(PROGRAM_FILE),
        "validation_cmd": VALIDATION_CMD,
        "test_cmd": TEST_CMD,
    }
    write_text(session_path, json.dumps(session, indent=2))

    env = os.environ.copy()
    env.setdefault("SSL_CERT_FILE", str(REPO_ROOT / ".venv/lib/python3.14/site-packages/certifi/cacert.pem"))

    baseline_run_text = RUN_FILE.read_text(encoding="utf-8")
    baseline_eval = run_command(VALIDATION_CMD, cwd=REPO_ROOT, env=env)
    if baseline_eval.returncode != 0:
        write_text(log_dir / "baseline_validation.log", baseline_eval.stdout + baseline_eval.stderr)
        raise SystemExit("baseline validation failed; see run_logs for details")
    baseline_metrics = parse_validation_metrics(baseline_eval.stdout + baseline_eval.stderr)
    write_text(log_dir / "baseline_validation.log", baseline_eval.stdout + baseline_eval.stderr)

    history: list[dict[str, object]] = []
    iteration_idx = 0
    print("Optimizer baseline:")
    print(
        "fixed_validation val_f1={val_f1:.4f} P={precision:.3f} R={recall:.3f} TP={tp} FP={fp} FN={fn}".format(
            val_f1=baseline_metrics["val_f1"] or 0.0,
            precision=baseline_metrics["precision"] or 0.0,
            recall=baseline_metrics["recall"] or 0.0,
            tp=baseline_metrics["tp"] or 0,
            fp=baseline_metrics["fp"] or 0,
            fn=baseline_metrics["fn"] or 0,
        )
    )
    print(f"Logs: {log_dir}")

    try:
        while True:
            if not args.repeat_until_interrupt and iteration_idx >= int(args.iterations):
                break
            iteration_idx += 1
            iteration_dir = log_dir / f"iteration_{iteration_idx:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            accepted_before = dict(baseline_metrics)
            run_before = RUN_FILE.read_text(encoding="utf-8")
            run_before_hash = hash_text(run_before)
            write_text(iteration_dir / "run_before.py", run_before)

            prompt = build_prompt(
                program_text=PROGRAM_FILE.read_text(encoding="utf-8"),
                baseline_metrics=baseline_metrics,
                recent_summary=history,
            )
            write_text(iteration_dir / "prompt.txt", prompt)

            codex_cmd = [codex_bin, "exec", "--full-auto", "--skip-git-repo-check", "-C", str(REPO_ROOT)]
            if args.model:
                codex_cmd.extend(["--model", args.model])
            codex_cmd.extend(["-o", str(iteration_dir / "codex_last_message.txt"), "-"])

            print(f"\n=== Optimizer Iteration {iteration_idx:03d} ===")
            print(f"Accepted baseline F1: {baseline_metrics['val_f1'] or 0.0:.4f}")
            print("Running Codex proposal...")
            codex_proc = subprocess.run(
                codex_cmd,
                cwd=REPO_ROOT,
                env=env,
                input=prompt,
                text=True,
                capture_output=True,
            )
            write_text(iteration_dir / "codex_stdout.log", codex_proc.stdout)
            write_text(iteration_dir / "codex_stderr.log", codex_proc.stderr)
            recommendation_text = ""
            last_message_path = iteration_dir / "codex_last_message.txt"
            if last_message_path.exists():
                recommendation_text = extract_recommendation(last_message_path.read_text(encoding="utf-8"))
                print(f"Codex recommendation: {recommendation_text}")
            else:
                recommendation_text = "missing codex_last_message.txt"

            run_after = RUN_FILE.read_text(encoding="utf-8")
            run_after_hash = hash_text(run_after)
            write_text(iteration_dir / "run_after.py", run_after)
            write_text(iteration_dir / "run_diff.patch", diff_text(iteration_dir / "run_before.py", iteration_dir / "run_after.py", REPO_ROOT))

            if codex_proc.returncode != 0:
                note = f"codex exec failed rc={codex_proc.returncode}"
                RUN_FILE.write_text(run_before, encoding="utf-8")
                decision = "revert"
                candidate_metrics = baseline_metrics.copy()
            elif run_after == run_before:
                note = "no change to neuronauts/run.py"
                decision = "revert"
                candidate_metrics = baseline_metrics.copy()
            else:
                print("Running regression test...")
                test_proc = run_command(TEST_CMD, cwd=REPO_ROOT, env=env)
                write_text(iteration_dir / "test.log", test_proc.stdout + test_proc.stderr)
                if test_proc.returncode != 0:
                    note = f"tests failed rc={test_proc.returncode}"
                    RUN_FILE.write_text(run_before, encoding="utf-8")
                    decision = "revert"
                    candidate_metrics = baseline_metrics.copy()
                else:
                    print("Running fixed validation...")
                    candidate_eval = run_command(VALIDATION_CMD, cwd=REPO_ROOT, env=env)
                    write_text(iteration_dir / "candidate_validation.log", candidate_eval.stdout + candidate_eval.stderr)
                    if candidate_eval.returncode != 0:
                        note = f"validation failed rc={candidate_eval.returncode}"
                        RUN_FILE.write_text(run_before, encoding="utf-8")
                        decision = "revert"
                        candidate_metrics = baseline_metrics.copy()
                    else:
                        candidate_metrics = parse_validation_metrics(candidate_eval.stdout + candidate_eval.stderr)
                        delta = (candidate_metrics["val_f1"] or 0.0) - (accepted_before["val_f1"] or 0.0)
                        if delta > args.improvement_threshold:
                            decision = "keep"
                            note = "fixed validation improved"
                            baseline_metrics = candidate_metrics
                            if not args.skip_live_loop:
                                print("Running accepted 5-minute monitoring loop...")
                                loop_log_dir = iteration_dir / "accepted_loop"
                                loop_cmd = [
                                    ".venv/bin/python",
                                    str(LOOP_SCRIPT),
                                    "--minutes",
                                    str(args.minutes),
                                    "--iterations",
                                    "1",
                                    "--python",
                                    ".venv/bin/python",
                                    "--data-mode",
                                    "real",
                                    "--real-boxes-per-eval",
                                    "3",
                                    "--real-min-synapses",
                                    "50",
                                    "--log-dir",
                                    str(loop_log_dir),
                                ]
                                loop_proc = run_command(loop_cmd, cwd=REPO_ROOT, env=env)
                                write_text(iteration_dir / "accepted_loop.log", loop_proc.stdout + loop_proc.stderr)
                        else:
                            decision = "revert"
                            note = "no fixed-validation improvement"
                            RUN_FILE.write_text(run_before, encoding="utf-8")

            current_run = RUN_FILE.read_text(encoding="utf-8")
            current_hash = hash_text(current_run)
            delta_f1 = (candidate_metrics["val_f1"] or 0.0) - (accepted_before["val_f1"] or 0.0)

            row = {
                "iteration": iteration_idx,
                "decision": decision,
                "baseline_f1": accepted_before["val_f1"],
                "candidate_f1": candidate_metrics["val_f1"],
                "note": note,
            }
            history.append(row)
            with summary_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    "\t".join(
                        [
                            str(iteration_idx),
                            decision,
                            f"{accepted_before['val_f1'] or 0.0:.4f}",
                            f"{candidate_metrics['val_f1'] or 0.0:.4f}",
                            f"{delta_f1:.4f}",
                            f"{accepted_before['precision'] or 0.0:.4f}",
                            f"{candidate_metrics['precision'] or 0.0:.4f}",
                            f"{accepted_before['recall'] or 0.0:.4f}",
                            f"{candidate_metrics['recall'] or 0.0:.4f}",
                            run_before_hash,
                            current_hash,
                            recommendation_text.replace("\t", " ").replace("\n", " "),
                            note,
                        ]
                    )
                    + "\n"
                )
            print(
                f"Decision: {decision} | baseline_f1={accepted_before['val_f1'] or 0.0:.4f} "
                f"candidate_f1={candidate_metrics['val_f1'] or 0.0:.4f} | {note}"
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(f"Optimizer summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
