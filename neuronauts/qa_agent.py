"""QA/QC agent for neuronauts runs.

Scans run logs, model metrics, training curves, and the research ledger to
surface warnings and errors as they accumulate.  Designed to be run
incrementally during a live experiment session.

Entry points
------------
- ``python scripts/qa_agent.py``              — full one-shot report
- ``python scripts/qa_agent.py --watch``      — poll every N seconds
- ``python scripts/qa_agent.py --session X``  — focus on one run session
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"ERROR": 0, "WARN": 1, "INFO": 2, "OK": 3}
SEVERITY_COLOR = {
    "ERROR": "\033[91m",   # red
    "WARN":  "\033[93m",   # yellow
    "INFO":  "\033[96m",   # cyan
    "OK":    "\033[92m",   # green
}
RESET = "\033[0m"


@dataclass
class Finding:
    severity: str          # ERROR | WARN | INFO | OK
    category: str          # LogAnalysis | MetricsHealth | TrainingCurve | etc.
    message: str
    detail: str = ""
    source: str = ""       # file path or context

    def __str__(self) -> str:
        color = SEVERITY_COLOR.get(self.severity, "")
        tag = f"[{self.severity:<5}]"
        src = f"  ({self.source})" if self.source else ""
        detail = f"\n         {self.detail}" if self.detail else ""
        return f"{color}{tag}{RESET} [{self.category}] {self.message}{src}{detail}"

    @property
    def sort_key(self) -> int:
        return SEVERITY_ORDER.get(self.severity, 99)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in _read_text(path).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _read_tsv(path: Path) -> list[dict[str, str]]:
    lines = _read_text(path).splitlines()
    if len(lines) < 2:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        rows.append(dict(zip(header, parts)))
    return rows


def _safe_float(value) -> float | None:
    try:
        v = float(value)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Log Analysis
# ---------------------------------------------------------------------------

# Patterns that indicate problems
_WARNING_RE   = re.compile(r"\[W\]")
_EXCEPTION_RE = re.compile(r"(Traceback|Error:|Exception:|raise )", re.IGNORECASE)
_NAN_RE       = re.compile(r"\bnan\b|\binf\b", re.IGNORECASE)
_ZERO_SYN_RE  = re.compile(r"synapses=0\b|0 synapses")
_BOX_FAIL_RE  = re.compile(r"failed to load ([a-f0-9]+)")
_GAT_FAIL_RE  = re.compile(r"\[W\] GAT step failed")
_VAL_FAIL_RE  = re.compile(r"\[W\] validation failed for ([a-f0-9]+)")
_KDTREE_RE    = re.compile(r"cKDTree.*fallback|fallback.*cKDTree", re.IGNORECASE)
_UNRESOLVED_RE = re.compile(r"(\d+) unresolved")
_HIT_RE       = re.compile(r"(\d+)/(\d+) synapses hit")
_F1_RE        = re.compile(r"val_f1\s*=\s*([0-9.]+)")
_RESULT_RE    = re.compile(r"LineGraph F1=([0-9.]+)\s+P=([0-9.]+)\s+R=([0-9.]+)")


def check_log_file(log_path: Path) -> list[Finding]:
    text = _read_text(log_path)
    if not text:
        return []
    findings: list[Finding] = []
    src = str(log_path)

    # Count warnings
    warns = _WARNING_RE.findall(text)
    if len(warns) > 10:
        findings.append(Finding("WARN", "LogAnalysis",
            f"{len(warns)} [W] warnings in log", source=src))
    elif warns:
        findings.append(Finding("INFO", "LogAnalysis",
            f"{len(warns)} [W] warning(s) in log", source=src))

    # Exceptions / tracebacks
    exc_lines = [l for l in text.splitlines() if _EXCEPTION_RE.search(l)]
    if exc_lines:
        findings.append(Finding("ERROR", "LogAnalysis",
            f"{len(exc_lines)} exception/error line(s)",
            detail=exc_lines[0][:200],
            source=src))

    # NaN / inf in output values
    nan_lines = [l for l in text.splitlines() if _NAN_RE.search(l)
                 and not l.strip().startswith("#")]
    if nan_lines:
        findings.append(Finding("ERROR", "LogAnalysis",
            "NaN or Inf detected in output",
            detail=nan_lines[0][:200],
            source=src))

    # cKDTree fallback — means scipy is unavailable, O(n²) merge
    if _KDTREE_RE.search(text):
        findings.append(Finding("WARN", "LogAnalysis",
            "cKDTree O(n²) fallback active (scipy missing?)",
            source=src))

    # Repeated box failures
    box_fails = _BOX_FAIL_RE.findall(text)
    if box_fails:
        findings.append(Finding("WARN", "LogAnalysis",
            f"{len(box_fails)} box load failure(s)",
            detail=", ".join(set(box_fails))[:200],
            source=src))

    # GAT step failures
    gat_fails = _GAT_FAIL_RE.findall(text)
    if len(gat_fails) > 3:
        findings.append(Finding("WARN", "LogAnalysis",
            f"GAT training step failed {len(gat_fails)} times",
            source=src))

    # Validation failures
    val_fails = _VAL_FAIL_RE.findall(text)
    if val_fails:
        findings.append(Finding("WARN", "LogAnalysis",
            f"{len(val_fails)} validation failure(s)",
            detail=", ".join(set(val_fails))[:200],
            source=src))

    # Zero synapses
    if _ZERO_SYN_RE.search(text):
        findings.append(Finding("WARN", "LogAnalysis",
            "Box(es) with zero synapses encountered",
            source=src))

    # Synapse hit rate — low hit rate means agents aren't covering the volume
    hit_matches = _HIT_RE.findall(text)
    for hit_str, total_str in hit_matches:
        hit, total = int(hit_str), int(total_str)
        if total > 0 and hit / total < 0.5:
            findings.append(Finding("WARN", "LogAnalysis",
                f"Low synapse hit rate: {hit}/{total} = {hit/total:.0%}",
                source=src))

    # Unresolved synapses
    unresolved_matches = _UNRESOLVED_RE.findall(text)
    for u in unresolved_matches:
        if int(u) > 5:
            findings.append(Finding("WARN", "LogAnalysis",
                f"{u} unresolved synapse assignments (OWNER_MARGIN=0.0 tie-breaking?)",
                source=src))

    # F1 from a single run
    f1_match = _F1_RE.search(text)
    result_match = _RESULT_RE.search(text)
    if f1_match:
        f1 = float(f1_match.group(1))
        if f1 == 0.0:
            findings.append(Finding("ERROR", "LogAnalysis",
                "val_f1 = 0.000 — evaluation may be broken",
                source=src))
        elif f1 < 0.2:
            findings.append(Finding("WARN", "LogAnalysis",
                f"Very low val_f1 = {f1:.3f}",
                source=src))

    if result_match:
        f1, p, r = float(result_match.group(1)), float(result_match.group(2)), float(result_match.group(3))
        # Check for precision/recall imbalance (> 3x ratio)
        if p > 0 and r > 0:
            ratio = max(p, r) / min(p, r)
            if ratio > 3.0:
                findings.append(Finding("WARN", "LogAnalysis",
                    f"Large P/R imbalance: P={p:.3f} R={r:.3f} (ratio {ratio:.1f}x) — "
                    "consider threshold or merge radius tuning",
                    source=src))

    return findings


# ---------------------------------------------------------------------------
# Metrics Health
# ---------------------------------------------------------------------------

def check_metrics_file(metrics_path: Path) -> list[Finding]:
    text = _read_text(metrics_path)
    if not text:
        return []
    findings: list[Finding] = []
    src = str(metrics_path)

    try:
        m = json.loads(text)
    except json.JSONDecodeError:
        findings.append(Finding("ERROR", "MetricsHealth",
            "Metrics file is not valid JSON", source=src))
        return findings

    def _check(key: str, lo: float, hi: float, label: str):
        v = _safe_float(m.get(key))
        if v is None:
            return
        if v < lo:
            findings.append(Finding("WARN", "MetricsHealth",
                f"{label} = {v:.4f} is suspiciously low (< {lo})",
                source=src))
        elif v > hi:
            findings.append(Finding("WARN", "MetricsHealth",
                f"{label} = {v:.4f} is suspiciously high (> {hi}) — possible overfitting",
                source=src))

    # Grammar model metrics
    _check("merge_accuracy",    lo=0.50, hi=0.98, label="merge_accuracy")
    _check("atomicity_accuracy", lo=0.50, hi=0.98, label="atomicity_accuracy")

    # Reranker metrics
    _check("corr", lo=0.0, hi=0.9995, label="reranker_corr")
    mse = _safe_float(m.get("mse"))
    if mse is not None and mse == 0.0:
        findings.append(Finding("WARN", "MetricsHealth",
            "reranker MSE = 0.0 exactly — possible trivial/constant dataset",
            source=src))

    # Validate training examples present
    n_merge = m.get("n_merge")
    n_topo  = m.get("n_topology")
    if n_merge is not None and int(n_merge) < 10:
        findings.append(Finding("WARN", "MetricsHealth",
            f"Only {n_merge} merge training examples — model may be underfit",
            source=src))
    if n_topo is not None and int(n_topo) < 10:
        findings.append(Finding("WARN", "MetricsHealth",
            f"Only {n_topo} topology training examples — model may be underfit",
            source=src))

    # Loss sanity: NaN in last_step
    last_step = m.get("last_step", {})
    for loss_key, loss_val in (last_step.items() if isinstance(last_step, dict) else []):
        v = _safe_float(loss_val)
        if v is None or math.isnan(v if v is not None else float("nan")):
            findings.append(Finding("ERROR", "MetricsHealth",
                f"NaN in last training step: {loss_key}",
                source=src))
        elif v is not None and v > 5.0:
            findings.append(Finding("WARN", "MetricsHealth",
                f"High last-step loss: {loss_key} = {v:.4f}",
                source=src))

    return findings


# ---------------------------------------------------------------------------
# Training Curve Analysis
# ---------------------------------------------------------------------------

def check_training_log(log_path: Path) -> list[Finding]:
    rows = _read_tsv(log_path)
    if not rows:
        return []
    findings: list[Finding] = []
    src = str(log_path)

    val_f1s = [_safe_float(r.get("val_f1")) for r in rows]
    val_f1s = [v for v in val_f1s if v is not None]

    if not val_f1s:
        return findings

    # Check for stagnation: no improvement over last 5 epochs
    if len(val_f1s) >= 5:
        recent = val_f1s[-5:]
        if max(recent) - min(recent) < 0.002:
            findings.append(Finding("WARN", "TrainingCurve",
                f"Val F1 stagnated over last 5 epochs "
                f"(range {min(recent):.4f}–{max(recent):.4f})",
                detail="Consider adjusting lr, regularization, or data diversity",
                source=src))

    # Check for regression: val F1 dropped significantly from best
    best_f1 = max(val_f1s)
    latest_f1 = val_f1s[-1]
    if best_f1 > 0 and (best_f1 - latest_f1) / best_f1 > 0.15:
        findings.append(Finding("WARN", "TrainingCurve",
            f"Val F1 regressed from best: {best_f1:.4f} → {latest_f1:.4f} "
            f"(-{(best_f1 - latest_f1) / best_f1:.0%})",
            detail="Model may be overfitting — checkpoint at best epoch is safeguarded",
            source=src))

    # Check for consistently zero F1
    if all(v == 0.0 for v in val_f1s):
        findings.append(Finding("ERROR", "TrainingCurve",
            "Val F1 = 0.0 across all epochs — validation pipeline may be broken",
            source=src))

    # Check train/val gap (overfitting signal)
    train_accs = [_safe_float(r.get("train_merge_acc")) for r in rows]
    train_accs = [v for v in train_accs if v is not None and v > 0]
    if train_accs and val_f1s:
        latest_train = train_accs[-1]
        if latest_train > 0.95 and val_f1s[-1] < 0.4:
            findings.append(Finding("WARN", "TrainingCurve",
                f"Train merge_acc={latest_train:.3f} vs val_f1={val_f1s[-1]:.3f} "
                "— large train/val gap suggests overfitting",
                detail="Consider more data, dropout, or weight decay",
                source=src))

    # n/a rows indicate boxes skipped
    na_rows = sum(1 for r in rows if r.get("train_merge_acc") == "n/a")
    if na_rows > len(rows) * 0.5:
        findings.append(Finding("WARN", "TrainingCurve",
            f"{na_rows}/{len(rows)} epochs had no grammar training examples (all n/a)",
            detail="Box cache may be too sparse or synapse filter too strict",
            source=src))

    if not findings:
        best_epoch = val_f1s.index(best_f1) + 1
        findings.append(Finding("OK", "TrainingCurve",
            f"{len(val_f1s)} epochs logged — best val_f1={best_f1:.4f} @ epoch {best_epoch}",
            source=src))

    return findings


# ---------------------------------------------------------------------------
# Research Ledger Analysis
# ---------------------------------------------------------------------------

def check_research_ledger(ledger_path: Path) -> list[Finding]:
    entries = _read_jsonl(ledger_path)
    if not entries:
        return [Finding("INFO", "LedgerAnalysis",
            "Research ledger is empty — no cycles logged yet",
            source=str(ledger_path))]

    findings: list[Finding] = []
    src = str(ledger_path)

    # Extract val_f1 time series
    f1_series: list[tuple[int, float, str]] = []  # (idx, f1, decision)
    for i, entry in enumerate(entries):
        compact = entry.get("compact_metrics", entry.get("metrics", {}))
        f1 = _safe_float(compact.get("val_f1") if isinstance(compact, dict) else None)
        decision = str(entry.get("decision", ""))
        if f1 is not None:
            f1_series.append((i + 1, f1, decision))

    if not f1_series:
        findings.append(Finding("INFO", "LedgerAnalysis",
            f"{len(entries)} ledger entries with no val_f1 field",
            source=src))
        return findings

    f1_values = [f for _, f, _ in f1_series]
    best_f1 = max(f1_values)
    latest_f1 = f1_values[-1]

    # Regression from peak
    if len(f1_values) >= 3 and best_f1 > 0 and (best_f1 - latest_f1) / best_f1 > 0.10:
        findings.append(Finding("WARN", "LedgerAnalysis",
            f"Ledger: val_f1 regressed from peak {best_f1:.4f} → {latest_f1:.4f}",
            detail=f"Over {len(f1_values)} research cycles",
            source=src))

    # Consecutive rejects — stuck in optimization loop
    decisions = [d for _, _, d in f1_series]
    if len(decisions) >= 3:
        recent_decisions = decisions[-4:]
        if all(d == "reject" for d in recent_decisions):
            findings.append(Finding("WARN", "LedgerAnalysis",
                f"Last {len(recent_decisions)} consecutive research cycles were 'reject'",
                detail="Optimizer may be stuck — consider resetting or changing strategy",
                source=src))

    # F1 = 0 in any cycle
    zero_cycles = [(i, f) for i, f, _ in f1_series if f == 0.0]
    if zero_cycles:
        findings.append(Finding("WARN", "LedgerAnalysis",
            f"{len(zero_cycles)} research cycle(s) with val_f1=0.0",
            detail=f"Cycle indices: {[i for i, _ in zero_cycles]}",
            source=src))

    # Summary
    findings.append(Finding("INFO", "LedgerAnalysis",
        f"{len(entries)} ledger entries | best val_f1={best_f1:.4f} | "
        f"latest val_f1={latest_f1:.4f}",
        source=src))

    return findings


# ---------------------------------------------------------------------------
# Session completeness
# ---------------------------------------------------------------------------

def check_session_dir(session_dir: Path) -> list[Finding]:
    findings: list[Finding] = []
    src = str(session_dir)

    summary_tsv = session_dir / "summary.tsv"
    session_json = session_dir / "session.json"

    if not summary_tsv.exists():
        findings.append(Finding("WARN", "RunCompleteness",
            "summary.tsv missing — session may not have completed",
            source=src))
    else:
        rows = _read_tsv(summary_tsv)
        if not rows:
            findings.append(Finding("WARN", "RunCompleteness",
                "summary.tsv exists but has no data rows",
                source=src))
        else:
            f1s = [_safe_float(r.get("val_f1")) for r in rows]
            f1s = [v for v in f1s if v is not None]
            failed = sum(1 for r in rows if r.get("rc", "0") != "0")
            if failed > 0:
                findings.append(Finding("WARN", "RunCompleteness",
                    f"{failed}/{len(rows)} run(s) in session had non-zero exit code",
                    source=src))
            if f1s:
                avg_f1 = sum(f1s) / len(f1s)
                findings.append(Finding("INFO", "RunCompleteness",
                    f"Session: {len(rows)} run(s), mean val_f1={avg_f1:.4f}, "
                    f"range [{min(f1s):.4f}, {max(f1s):.4f}]",
                    source=src))

    # Log files: count warnings across all runs in session
    log_files = sorted(session_dir.glob("run_*.log"))
    if not log_files:
        findings.append(Finding("INFO", "RunCompleteness",
            "No run_NNN.log files found in session",
            source=src))
    else:
        total_warns = 0
        for lf in log_files:
            text = _read_text(lf)
            total_warns += len(_WARNING_RE.findall(text))
        if total_warns > 0:
            findings.append(Finding("WARN" if total_warns > 5 else "INFO",
                "RunCompleteness",
                f"{total_warns} total [W] warnings across {len(log_files)} log(s)",
                source=src))

    return findings


# ---------------------------------------------------------------------------
# Model file health
# ---------------------------------------------------------------------------

def check_model_files(models_dir: Path) -> list[Finding]:
    findings: list[Finding] = []

    pt_files  = list(models_dir.glob("*.pt"))
    npz_files = list(models_dir.glob("*.npz"))
    json_files = list(models_dir.glob("*.metrics.json"))

    if not pt_files and not npz_files:
        findings.append(Finding("WARN", "ModelHealth",
            "No model checkpoint files found in models/",
            source=str(models_dir)))
        return findings

    # Every model should have a corresponding metrics file
    all_stems = {f.stem for f in pt_files + npz_files}
    metrics_stems = {f.name.replace(".metrics.json", "") for f in json_files}
    missing_metrics = all_stems - metrics_stems
    if missing_metrics:
        findings.append(Finding("INFO", "ModelHealth",
            f"Model(s) without a metrics file: {missing_metrics}",
            source=str(models_dir)))

    # Check age of models — stale model with fresh data could be an issue
    now = time.time()
    for f in pt_files + npz_files:
        age_hours = (now - f.stat().st_mtime) / 3600
        if age_hours > 48:
            findings.append(Finding("INFO", "ModelHealth",
                f"{f.name} is {age_hours:.0f}h old — may be stale",
                source=str(f)))

    # Check each metrics file
    for mf in json_files:
        findings.extend(check_metrics_file(mf))

    if not findings:
        findings.append(Finding("OK", "ModelHealth",
            f"{len(pt_files)} .pt + {len(npz_files)} .npz checkpoints with metrics",
            source=str(models_dir)))

    return findings


# ---------------------------------------------------------------------------
# Box cache health
# ---------------------------------------------------------------------------

def check_box_cache(cache_dir: Path) -> list[Finding]:
    if not cache_dir.exists():
        return [Finding("INFO", "DataQuality",
            f"Box cache not found at {cache_dir}",
            source=str(cache_dir))]

    findings: list[Finding] = []
    box_dirs = [d for d in cache_dir.iterdir() if d.is_dir()]
    if not box_dirs:
        return [Finding("INFO", "DataQuality",
            "Box cache directory is empty",
            source=str(cache_dir))]

    corrupt = 0
    empty_syn = 0
    total_syn_counts: list[int] = []

    for bd in box_dirs:
        vol_file = bd / "volume.npz"
        syn_file = bd / "synapses.npz"

        if not vol_file.exists() or not syn_file.exists():
            corrupt += 1
            continue

        # Check synapse count from file size as a proxy (0-byte = empty)
        if syn_file.stat().st_size < 100:
            empty_syn += 1
            continue

        total_syn_counts.append(syn_file.stat().st_size)

    if corrupt > 0:
        findings.append(Finding("WARN", "DataQuality",
            f"{corrupt}/{len(box_dirs)} box cache entries incomplete (missing npz files)",
            source=str(cache_dir)))

    if empty_syn > 0:
        findings.append(Finding("WARN", "DataQuality",
            f"{empty_syn}/{len(box_dirs)} box cache entries have near-empty synapse files",
            source=str(cache_dir)))

    if len(box_dirs) - corrupt - empty_syn > 0:
        findings.append(Finding("OK", "DataQuality",
            f"Box cache: {len(box_dirs) - corrupt - empty_syn} healthy box(es) "
            f"({corrupt} corrupt, {empty_syn} empty-synapse)",
            source=str(cache_dir)))

    return findings


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

@dataclass
class QAReport:
    timestamp: datetime
    findings: list[Finding] = field(default_factory=list)
    run_root: Path = Path(".")

    def add(self, findings: list[Finding]) -> None:
        self.findings.extend(findings)

    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "ERROR"]

    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "WARN"]

    def by_severity(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: f.sort_key)

    def summary_line(self) -> str:
        errors  = len(self.errors())
        warnings = len(self.warnings())
        infos   = sum(1 for f in self.findings if f.severity == "INFO")
        oks     = sum(1 for f in self.findings if f.severity == "OK")
        color = SEVERITY_COLOR["ERROR"] if errors else (
                SEVERITY_COLOR["WARN"]  if warnings else SEVERITY_COLOR["OK"])
        return (
            f"{color}{'ERROR' if errors else 'WARN' if warnings else 'OK'}{RESET}  "
            f"errors={errors}  warnings={warnings}  info={infos}  ok={oks}"
        )


def run_full_audit(run_root: Path, *, session: str | None = None) -> QAReport:
    report = QAReport(timestamp=datetime.now(timezone.utc), run_root=run_root)

    # Model files
    models_dir = run_root / "models"
    if models_dir.exists():
        report.add(check_model_files(models_dir))

    # Research ledger
    ledger = run_root / "run_logs" / "research_ledger.jsonl"
    if ledger.exists():
        report.add(check_research_ledger(ledger))

    # Training log
    train_log = run_root / "run_logs" / "train_log.tsv"
    if train_log.exists():
        report.add(check_training_log(train_log))

    # Session(s)
    run_logs_dir = run_root / "run_logs"
    if session:
        sessions = [run_logs_dir / session] if (run_logs_dir / session).is_dir() else []
    else:
        sessions = sorted(
            [d for d in run_logs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )[:5]  # 5 most-recent sessions

    for sess_dir in sessions:
        report.add(check_session_dir(sess_dir))
        for log_file in sorted(sess_dir.glob("run_*.log")):
            report.add(check_log_file(log_file))

    # Box cache
    for cache_dir in [run_root / "data" / "boxes", run_root / "data"]:
        if cache_dir.exists():
            report.add(check_box_cache(cache_dir))
            break

    return report


def run_incremental_audit(run_root: Path, since_mtime: float = 0.0) -> QAReport:
    """Only check files modified after since_mtime (unix timestamp)."""
    report = QAReport(timestamp=datetime.now(timezone.utc), run_root=run_root)

    run_logs_dir = run_root / "run_logs"

    # Always re-check ledger and model metrics (cheap)
    ledger = run_logs_dir / "research_ledger.jsonl"
    if ledger.exists() and ledger.stat().st_mtime > since_mtime:
        report.add(check_research_ledger(ledger))

    train_log = run_logs_dir / "train_log.tsv"
    if train_log.exists() and train_log.stat().st_mtime > since_mtime:
        report.add(check_training_log(train_log))

    for mf in (run_root / "models").glob("*.metrics.json"):
        if mf.stat().st_mtime > since_mtime:
            report.add(check_metrics_file(mf))

    # New/modified log files
    for log_file in sorted((run_logs_dir).rglob("run_*.log")):
        if log_file.stat().st_mtime > since_mtime:
            report.add(check_log_file(log_file))

    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_report(report: QAReport, *, verbose: bool = False) -> str:
    lines = [
        "",
        f"  {'='*60}",
        f"  Neuronauts QA/QC  —  {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"  {'='*60}",
    ]

    findings = report.by_severity()
    if not verbose:
        # In non-verbose mode, suppress OK findings unless there's nothing else
        meaningful = [f for f in findings if f.severity != "OK"]
        findings = meaningful if meaningful else findings

    for f in findings:
        lines.append(f"  {f}")

    lines += [
        "",
        f"  {report.summary_line()}",
        "",
    ]
    return "\n".join(lines)
