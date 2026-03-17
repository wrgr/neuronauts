"""Shared research-cycle driver for Codex and Gemini outer loops."""

from __future__ import annotations

from datetime import datetime, UTC
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchCycleConfig:
    repo_root: Path
    python_bin: str = ".venv/bin/python"
    output_dir: Path | None = None
    merge_dataset: str = "data/merge_dataset_smoke.npz"
    topology_dataset: str = "data/topology_dataset_smoke.npz"
    shared_model: str = "models/shared_grammar_smoke.pt"
    assembly_dataset: str = "data/assembly_ranking_smoke.npz"
    assembly_reranker: str = "models/assembly_reranker_smoke.npz"
    export_boxes: str = "0,1,2"
    assembly_cases: int = 3
    thresholds: str = "-0.5,0.0,0.5"
    beam_widths: str = "1,2,4"
    selection_box_indices: str = "0,1,2"
    holdout_box_indices: str = "3,4,5"
    run_data_mode: str = "real"
    real_boxes_per_eval: int = 3
    real_min_synapses: int = 50
    membrane_source: str = "auto"
    membrane_cache_dir: str = "cache/membranes"
    quiet: bool = True


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)


def parse_validation_metrics(text: str) -> dict[str, float | int | None]:
    def as_float(pattern: str) -> float | None:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    def as_int(pattern: str) -> int | None:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None

    return {
        "val_f1": as_float(r"val_f1\s*=\s*([0-9.]+)"),
        "precision": as_float(r"P=([0-9.]+)"),
        "recall": as_float(r"R=([0-9.]+)"),
        "tp": as_int(r"TP=([0-9]+)"),
        "fp": as_int(r"FP=([0-9]+)"),
        "fn": as_int(r"FN=([0-9]+)"),
    }


def load_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_research_cycle(summary: dict[str, object]) -> dict[str, float]:
    metrics = dict(summary.get("metrics", {}))
    shared_metrics = dict(summary.get("shared_training_metrics", {}))
    reranker_metrics = dict(summary.get("reranker_metrics", {}))
    merged = {
        "val_f1": float(metrics.get("val_f1") or 0.0),
        "holdout_f1": float(metrics.get("holdout_f1") or 0.0),
        "precision": float(metrics.get("precision") or 0.0),
        "recall": float(metrics.get("recall") or 0.0),
        "merge_accuracy": float(shared_metrics.get("merge_accuracy") or 0.0),
        "atomicity_accuracy": float(shared_metrics.get("atomicity_accuracy") or 0.0),
        "reranker_corr": float(reranker_metrics.get("corr") or 0.0),
        "reranker_mse": float(reranker_metrics.get("mse") or 0.0),
    }
    return merged


def build_ledger_entry(
    summary: dict[str, object],
    *,
    source: str,
    target_file: str | None = None,
    hypothesis: str | None = None,
    decision: str | None = None,
    note: str | None = None,
    iteration: int | None = None,
    run_dir: str | None = None,
) -> dict[str, object]:
    compact = summarize_research_cycle(summary)
    entry: dict[str, object] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "source": source,
        "decision": decision or ("keep" if summary.get("ok") else "failed"),
        "note": note or "",
        "target_file": target_file or "",
        "hypothesis": hypothesis or "",
        "iteration": iteration,
        "run_dir": run_dir or "",
        "val_f1": compact["val_f1"],
        "holdout_f1": compact["holdout_f1"],
        "precision": compact["precision"],
        "recall": compact["recall"],
        "merge_accuracy": compact["merge_accuracy"],
        "atomicity_accuracy": compact["atomicity_accuracy"],
        "reranker_corr": compact["reranker_corr"],
        "reranker_mse": compact["reranker_mse"],
        "ok": bool(summary.get("ok")),
    }
    failed_step = summary.get("failed_step")
    if failed_step is not None:
        entry["failed_step"] = failed_step
    return entry


def load_experiment_ledger(ledger_path: str | Path) -> list[dict[str, object]]:
    path = Path(ledger_path)
    if not path.exists():
        return []
    entries: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(json.loads(line))
    return entries


def write_experiment_leaderboard(leaderboard_path: str | Path, entries: list[dict[str, object]]) -> None:
    path = Path(leaderboard_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        entries,
        key=lambda entry: (
            float(entry.get("holdout_f1") or 0.0),
            float(entry.get("val_f1") or 0.0),
            float(entry.get("reranker_corr") or 0.0),
        ),
        reverse=True,
    )
    header = [
        "timestamp",
        "source",
        "decision",
        "iteration",
        "target_file",
        "val_f1",
        "holdout_f1",
        "merge_accuracy",
        "atomicity_accuracy",
        "reranker_corr",
        "reranker_mse",
        "note",
    ]
    lines = ["\t".join(header)]
    for entry in ordered:
        lines.append(
            "\t".join(
                [
                    str(entry.get("timestamp", "")),
                    str(entry.get("source", "")),
                    str(entry.get("decision", "")),
                    "" if entry.get("iteration") is None else str(entry.get("iteration")),
                    str(entry.get("target_file", "")),
                    f"{float(entry.get('val_f1') or 0.0):.4f}",
                    f"{float(entry.get('holdout_f1') or 0.0):.4f}",
                    f"{float(entry.get('merge_accuracy') or 0.0):.4f}",
                    f"{float(entry.get('atomicity_accuracy') or 0.0):.4f}",
                    f"{float(entry.get('reranker_corr') or 0.0):.4f}",
                    f"{float(entry.get('reranker_mse') or 0.0):.4f}",
                    str(entry.get("note", "")).replace("\t", " ").replace("\n", " "),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_experiment_ledger(
    ledger_path: str | Path,
    entry: dict[str, object],
    *,
    leaderboard_path: str | Path | None = None,
) -> None:
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    entries = load_experiment_ledger(path)
    board_path = Path(leaderboard_path) if leaderboard_path is not None else path.with_suffix(".leaderboard.tsv")
    write_experiment_leaderboard(board_path, entries)


def compare_cycle_summaries(
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    improvement_threshold: float = 0.0,
) -> tuple[str, str]:
    base = summarize_research_cycle(baseline)
    cand = summarize_research_cycle(candidate)
    f1_delta = cand["val_f1"] - base["val_f1"]
    holdout_delta = cand["holdout_f1"] - base["holdout_f1"]
    if f1_delta > improvement_threshold:
        if holdout_delta < -max(0.01, improvement_threshold):
            return "revert", f"selection improved but holdout regressed by {holdout_delta:.4f}"
        return "keep", f"val_f1 improved by {f1_delta:.4f}"
    if f1_delta < -max(0.001, improvement_threshold):
        return "revert", f"val_f1 regressed by {f1_delta:.4f}"

    # Tie-break region: allow small/no F1 change only if the inner-loop metrics improve coherently.
    merge_delta = cand["merge_accuracy"] - base["merge_accuracy"]
    atomicity_delta = cand["atomicity_accuracy"] - base["atomicity_accuracy"]
    reranker_delta = cand["reranker_corr"] - base["reranker_corr"]
    if merge_delta > 0.01 and atomicity_delta >= 0.0 and cand["reranker_mse"] <= base["reranker_mse"] + 1e-6:
        return "keep", "inner-loop metrics improved in tie region"
    if atomicity_delta > 0.01 and merge_delta >= 0.0 and cand["reranker_mse"] <= base["reranker_mse"] + 1e-6:
        return "keep", "atomicity improved in tie region"
    if reranker_delta > 0.01 and merge_delta >= 0.0 and atomicity_delta >= 0.0:
        if holdout_delta < -0.01:
            return "revert", f"tie-region improvement but holdout regressed by {holdout_delta:.4f}"
        return "keep", "reranker improved in tie region"
    return "revert", "no meaningful cycle-level improvement"


def build_research_cycle_commands(config: ResearchCycleConfig) -> dict[str, list[str]]:
    py = config.python_bin
    commands = {
        "export_merge": [
            py,
            "scripts/export_merge_dataset.py",
            "--output",
            config.merge_dataset,
            "--box-indices",
            config.export_boxes,
        ],
        "export_topology": [
            py,
            "scripts/export_topology_dataset.py",
            "--output",
            config.topology_dataset,
            "--box-indices",
            config.export_boxes,
            "--membrane-source",
            config.membrane_source,
            "--membrane-cache-dir",
            config.membrane_cache_dir,
        ],
        "train_shared": [
            py,
            "scripts/train_shared_grammar.py",
            "--merge-dataset",
            config.merge_dataset,
            "--topology-dataset",
            config.topology_dataset,
            "--output",
            config.shared_model,
        ],
        "export_assembly": [
            py,
            "scripts/export_assembly_ranking_dataset.py",
            "--output",
            config.assembly_dataset,
            "--cases",
            str(config.assembly_cases),
            "--thresholds",
            config.thresholds,
            "--beam-widths",
            config.beam_widths,
            "--shared-grammar-checkpoint",
            config.shared_model,
        ],
        "train_reranker": [
            py,
            "scripts/train_assembly_ranker.py",
            "--dataset",
            config.assembly_dataset,
            "--output",
            config.assembly_reranker,
        ],
        "validate_selection": [
            py,
            "-m",
            "neuronauts.run",
            "--data-mode",
            config.run_data_mode,
            "--shared-grammar-checkpoint",
            config.shared_model,
            "--assembly-reranker-checkpoint",
            config.assembly_reranker,
            "--real-boxes-per-eval",
            str(config.real_boxes_per_eval),
            "--real-box-indices",
            config.selection_box_indices,
            "--real-min-synapses",
            str(config.real_min_synapses),
            "--membrane-source",
            config.membrane_source,
            "--membrane-cache-dir",
            config.membrane_cache_dir,
        ],
        "validate_holdout": [
            py,
            "-m",
            "neuronauts.run",
            "--data-mode",
            config.run_data_mode,
            "--shared-grammar-checkpoint",
            config.shared_model,
            "--assembly-reranker-checkpoint",
            config.assembly_reranker,
            "--real-boxes-per-eval",
            str(config.real_boxes_per_eval),
            "--real-box-indices",
            config.holdout_box_indices,
            "--real-min-synapses",
            str(config.real_min_synapses),
            "--membrane-source",
            config.membrane_source,
            "--membrane-cache-dir",
            config.membrane_cache_dir,
        ],
    }
    if config.quiet:
        commands["validate_selection"].append("--quiet")
        commands["validate_holdout"].append("--quiet")
    return commands


def run_research_cycle(
    config: ResearchCycleConfig,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, object]:
    commands = build_research_cycle_commands(config)
    output_dir = config.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    step_logs: dict[str, dict[str, object]] = {}
    for step_name, cmd in commands.items():
        proc = run_command(cmd, cwd=config.repo_root, env=env)
        step_logs[step_name] = {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        if output_dir is not None:
            (output_dir / f"{step_name}.stdout.log").write_text(proc.stdout, encoding="utf-8")
            (output_dir / f"{step_name}.stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            return {
                "ok": False,
                "failed_step": step_name,
                "steps": step_logs,
            }

    selection_metrics = parse_validation_metrics(step_logs["validate_selection"]["stdout"] + step_logs["validate_selection"]["stderr"])
    holdout_metrics = parse_validation_metrics(step_logs["validate_holdout"]["stdout"] + step_logs["validate_holdout"]["stderr"])
    metrics = dict(selection_metrics)
    metrics["holdout_f1"] = holdout_metrics.get("val_f1")
    summary = {
        "ok": True,
        "failed_step": None,
        "metrics": metrics,
        "shared_training_metrics": load_json_if_exists(config.repo_root / Path(config.shared_model).with_suffix(".metrics.json")),
        "reranker_metrics": load_json_if_exists(config.repo_root / Path(config.assembly_reranker).with_suffix(".metrics.json")),
        "artifacts": {
            "merge_dataset": config.merge_dataset,
            "topology_dataset": config.topology_dataset,
            "shared_model": config.shared_model,
            "assembly_dataset": config.assembly_dataset,
            "assembly_reranker": config.assembly_reranker,
        },
        "steps": {name: {"returncode": step["returncode"]} for name, step in step_logs.items()},
        "selection_metrics": selection_metrics,
        "holdout_metrics": holdout_metrics,
    }
    if output_dir is not None:
        (output_dir / "research_cycle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
