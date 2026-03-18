#!/usr/bin/env python3
"""Patch/evaluate/keep-or-revert loop — supports Codex, Claude, and Gemini backends.

This script implements a self-improving loop that proposes edits to a target
file (default: ``neuronauts/grammar.py``), runs the research cycle, and
keeps or reverts changes based on line-graph F1.

Supported backends
------------------
codex  (default)
    Uses the ``codex exec --full-auto`` CLI.  Requires the ``codex``
    executable to be available on PATH (or passed via ``--codex-bin``).

claude
    Uses the Anthropic Python API (``pip install anthropic``).  Reads the
    target file, sends a prompt to Claude Sonnet, parses the modified file
    from the model's response, and writes it back.  Requires either the
    ``ANTHROPIC_API_KEY`` environment variable or ``--api-key``.

gemini
    Uses the Google Generative AI Python API (``pip install google-generativeai``).
    Same approach as the Claude backend.  Requires either
    ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY`` env-var or ``--api-key``.
"""

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

from neuronauts.experiment_driver import (
    append_experiment_ledger,
    build_ledger_entry,
    compare_cycle_summaries,
    summarize_research_cycle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_FILE = REPO_ROOT / "program.md"
TEST_CMD = [".venv/bin/python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]


def build_cycle_cmd(output_dir: Path) -> list[str]:
    return [
        ".venv/bin/python",
        "scripts/run_research_cycle.py",
        "--python-bin",
        ".venv/bin/python",
        "--output-dir",
        str(output_dir),
        "--quiet",
    ]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=None,
                        help="Number of optimizer iterations (default: 1 unless --repeat-until-interrupt).")
    parser.add_argument(
        "--repeat-until-interrupt",
        action="store_true",
        help="Keep proposing changes until Ctrl+C.",
    )
    parser.add_argument("--log-dir", default="run_logs/codex_optimize",
                        help="Directory for optimizer artifacts.")
    parser.add_argument("--model", default=None,
                        help="LLM model override (e.g. claude-sonnet-4-5, gemini-2.0-flash, o4-mini).")
    parser.add_argument("--backend", default="codex",
                        choices=["codex", "claude", "gemini"],
                        help="LLM backend to use for proposals (default: codex).")
    parser.add_argument("--api-key", default=None,
                        help="API key for claude/gemini backends (falls back to env-var).")
    parser.add_argument("--codex-bin", default="codex",
                        help="Codex CLI executable path (only used with --backend=codex).")
    parser.add_argument("--target-file", default="neuronauts/grammar.py",
                        help="Repo-relative path to the file the optimizer may edit.")
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.0,
        help="Minimum fixed-validation F1 improvement required to keep a change.",
    )
    parser.add_argument("--ledger-path", default="run_logs/research_ledger.jsonl",
                        help="Shared experiment ledger path.")
    return parser.parse_args()


def resolve_executable(executable: str) -> str:
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    if Path(executable).exists():
        return str(Path(executable).resolve())
    fallback_paths = sorted(
        Path.home().glob(".cursor/extensions/openai.chatgpt-*/bin/macos-aarch64/codex"),
        reverse=True,
    )
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
- Edit only neuronauts/grammar.py.
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
1. Inspect neuronauts/grammar.py.
2. Make one targeted change that is likely to improve real MICrONS fixed-validation performance.
3. Stop after editing neuronauts/grammar.py.

Do not run long benchmark loops yourself; the wrapper script will evaluate and decide whether to keep or revert your change.
"""


def main() -> int:
    args = parse_args()
    if not args.repeat_until_interrupt and args.iterations is None:
        args.iterations = 1

    codex_bin = resolve_executable(args.codex_bin)

    log_dir = REPO_ROOT / args.log_dir
    ledger_path = REPO_ROOT / args.ledger_path
    leaderboard_path = ledger_path.with_suffix(".leaderboard.tsv")
    log_dir.mkdir(parents=True, exist_ok=True)
    session_path = log_dir / "session.json"
    summary_path = log_dir / "optimizer_summary.tsv"
    summary_path.write_text(
        "iteration\tdecision\tbaseline_f1\tcandidate_f1\tdelta_f1\tbaseline_merge_acc\tcandidate_merge_acc\tbaseline_atomicity_acc\tcandidate_atomicity_acc\tbaseline_reranker_corr\tcandidate_reranker_corr\trun_hash_before\trun_hash_after\trecommendation\tnote\n",
        encoding="utf-8",
    )

    session = {
        "started_at_epoch": time.time(),
        "iterations": args.iterations,
        "repeat_until_interrupt": args.repeat_until_interrupt,
        "target_file": str(TARGET_FILE),
        "program_file": str(PROGRAM_FILE),
        "validation_cmd": build_cycle_cmd(log_dir / "baseline_cycle"),
        "test_cmd": TEST_CMD,
    }
    write_text(session_path, json.dumps(session, indent=2))

    env = os.environ.copy()
    env.setdefault("SSL_CERT_FILE", str(REPO_ROOT / ".venv/lib/python3.14/site-packages/certifi/cacert.pem"))

    baseline_eval = run_command(build_cycle_cmd(log_dir / "baseline_cycle"), cwd=REPO_ROOT, env=env)
    if baseline_eval.returncode != 0:
        write_text(log_dir / "baseline_validation.log", baseline_eval.stdout + baseline_eval.stderr)
        raise SystemExit("baseline validation failed; see run_logs for details")
    baseline_summary = json.loads(baseline_eval.stdout)
    baseline_metrics = baseline_summary["metrics"]
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
            accepted_before_summary = dict(baseline_summary)
            candidate_summary = dict(baseline_summary)
            run_before = TARGET_FILE.read_text(encoding="utf-8")
            run_before_hash = hash_text(run_before)
            write_text(iteration_dir / "target_before.py", run_before)

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

            run_after = TARGET_FILE.read_text(encoding="utf-8")
            run_after_hash = hash_text(run_after)
            write_text(iteration_dir / "target_after.py", run_after)
            write_text(
                iteration_dir / "target_diff.patch",
                diff_text(iteration_dir / "target_before.py", iteration_dir / "target_after.py", REPO_ROOT),
            )

            if codex_proc.returncode != 0:
                note = f"codex exec failed rc={codex_proc.returncode}"
                TARGET_FILE.write_text(run_before, encoding="utf-8")
                decision = "revert"
                candidate_metrics = baseline_metrics.copy()
            elif run_after == run_before:
                note = "no change to neuronauts/grammar.py"
                decision = "revert"
                candidate_metrics = baseline_metrics.copy()
            else:
                print("Running regression test...")
                test_proc = run_command(TEST_CMD, cwd=REPO_ROOT, env=env)
                write_text(iteration_dir / "test.log", test_proc.stdout + test_proc.stderr)
                if test_proc.returncode != 0:
                    note = f"tests failed rc={test_proc.returncode}"
                    TARGET_FILE.write_text(run_before, encoding="utf-8")
                    decision = "revert"
                    candidate_metrics = baseline_metrics.copy()
                else:
                    print("Running fixed validation...")
                    candidate_eval = run_command(build_cycle_cmd(iteration_dir / "research_cycle"), cwd=REPO_ROOT, env=env)
                    write_text(iteration_dir / "candidate_validation.log", candidate_eval.stdout + candidate_eval.stderr)
                    if candidate_eval.returncode != 0:
                        note = f"validation failed rc={candidate_eval.returncode}"
                        TARGET_FILE.write_text(run_before, encoding="utf-8")
                        decision = "revert"
                        candidate_metrics = baseline_metrics.copy()
                        candidate_summary = baseline_summary
                    else:
                        candidate_summary = json.loads(candidate_eval.stdout)
                        candidate_metrics = candidate_summary["metrics"]
                        decision, note = compare_cycle_summaries(
                            accepted_before_summary,
                            candidate_summary,
                            improvement_threshold=args.improvement_threshold,
                        )
                        if decision == "keep":
                            baseline_metrics = candidate_metrics
                            baseline_summary = candidate_summary
                        else:
                            TARGET_FILE.write_text(run_before, encoding="utf-8")

            current_run = TARGET_FILE.read_text(encoding="utf-8")
            current_hash = hash_text(current_run)
            delta_f1 = (candidate_metrics["val_f1"] or 0.0) - (accepted_before["val_f1"] or 0.0)
            accepted_compact = summarize_research_cycle(accepted_before_summary)
            candidate_compact = summarize_research_cycle(candidate_summary)

            row = {
                "iteration": iteration_idx,
                "decision": decision,
                "baseline_f1": accepted_before["val_f1"],
                "candidate_f1": candidate_metrics["val_f1"],
                "note": note,
            }
            history.append(row)
            append_experiment_ledger(
                ledger_path,
                build_ledger_entry(
                    candidate_summary,
                    source="codex",
                    target_file=str(TARGET_FILE.relative_to(REPO_ROOT)),
                    hypothesis=recommendation_text,
                    decision=decision,
                    note=note,
                    iteration=iteration_idx,
                    run_dir=str(iteration_dir.relative_to(REPO_ROOT)),
                ),
                leaderboard_path=leaderboard_path,
            )
            with summary_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    "\t".join(
                        [
                            str(iteration_idx),
                            decision,
                            f"{accepted_before['val_f1'] or 0.0:.4f}",
                            f"{candidate_metrics['val_f1'] or 0.0:.4f}",
                            f"{delta_f1:.4f}",
                            f"{accepted_compact['merge_accuracy']:.4f}",
                            f"{candidate_compact['merge_accuracy']:.4f}",
                            f"{accepted_compact['atomicity_accuracy']:.4f}",
                            f"{candidate_compact['atomicity_accuracy']:.4f}",
                            f"{accepted_compact['reranker_corr']:.4f}",
                            f"{candidate_compact['reranker_corr']:.4f}",
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
                f"candidate_f1={candidate_metrics['val_f1'] or 0.0:.4f} "
                f"merge_acc={candidate_compact['merge_accuracy']:.3f} "
                f"atomicity_acc={candidate_compact['atomicity_accuracy']:.3f} "
                f"reranker_corr={candidate_compact['reranker_corr']:.3f} | {note}"
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    print(f"Optimizer summary: {summary_path}")
    print(f"Research ledger: {ledger_path}")
    print(f"Leaderboard: {leaderboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
