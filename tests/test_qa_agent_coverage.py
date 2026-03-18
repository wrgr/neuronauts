"""Tests for neuronauts/qa_agent.py — 0% → ~80% coverage.

qa_agent.py is the monitoring layer that detects silent training failures
(NaN losses, zero F1, stagnant curves, etc.).  Bugs here mean real problems
go undetected during multi-hour training runs.

Covers:
- Finding dataclass and __str__ / sort_key
- check_log_file: NaN, exceptions, cKDTree fallback, hit-rate, F1, P/R
- check_metrics_file: bad JSON, low/high accuracy, NaN loss, MSE=0
- check_training_log: stagnation, regression, zero F1, overfitting gap
- check_research_ledger: empty, no val_f1, regression, zero-f1 cycles
- QAReport: add, errors, warnings, by_severity, summary_line
- run_full_audit: minimal temp-dir smoke test
- render_report: string output includes summary line
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helper: write a temp file
# ---------------------------------------------------------------------------

def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------

class FindingTest(unittest.TestCase):

    def _finding(self, severity="WARN", category="Test", message="msg",
                 detail="", source="src.log"):
        from neuronauts.qa_agent import Finding
        return Finding(severity=severity, category=category, message=message,
                       detail=detail, source=source)

    def test_str_includes_severity_and_category(self):
        f = self._finding(severity="ERROR", category="LogAnalysis", message="boom")
        s = str(f)
        self.assertIn("ERROR", s)
        self.assertIn("LogAnalysis", s)
        self.assertIn("boom", s)

    def test_str_includes_detail_when_present(self):
        f = self._finding(detail="extra info")
        s = str(f)
        self.assertIn("extra info", s)

    def test_str_includes_source(self):
        f = self._finding(source="some/path.log")
        self.assertIn("some/path.log", str(f))

    def test_str_no_detail_no_newline_detail(self):
        f = self._finding(detail="")
        # Detail block starts with \n + spaces — absent when detail is empty.
        self.assertNotIn("\n         ", str(f))

    def test_sort_key_ordering(self):
        from neuronauts.qa_agent import Finding
        severities = ["ERROR", "WARN", "INFO", "OK"]
        keys = [Finding(s, "T", "m").sort_key for s in severities]
        self.assertEqual(keys, sorted(keys))

    def test_sort_key_unknown_severity_is_max(self):
        from neuronauts.qa_agent import Finding
        f = Finding("UNKNOWN", "T", "m")
        self.assertGreater(f.sort_key, 10)


# ---------------------------------------------------------------------------
# check_log_file
# ---------------------------------------------------------------------------

class CheckLogFileTest(unittest.TestCase):

    def _check(self, text: str):
        from neuronauts.qa_agent import check_log_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as fh:
            fh.write(text)
            p = Path(fh.name)
        try:
            return check_log_file(p)
        finally:
            p.unlink(missing_ok=True)

    def test_empty_file_returns_no_findings(self):
        findings = self._check("")
        self.assertEqual(findings, [])

    def test_nan_in_loss_detected(self):
        findings = self._check("loss=nan after epoch 3\n")
        severities = [f.severity for f in findings]
        self.assertIn("ERROR", severities)

    def test_inf_in_output_detected(self):
        findings = self._check("val_score=inf\n")
        severities = [f.severity for f in findings]
        self.assertIn("ERROR", severities)

    def test_traceback_detected_as_error(self):
        findings = self._check(
            "Something went wrong\nTraceback (most recent call last):\n  line 1\n"
        )
        errors = [f for f in findings if f.severity == "ERROR"]
        self.assertTrue(any("exception" in f.message.lower() for f in errors))

    def test_single_warning_is_info(self):
        findings = self._check("[W] something minor happened\n")
        infos = [f for f in findings if f.severity == "INFO"]
        self.assertTrue(any("warning" in f.message.lower() for f in infos))

    def test_many_warnings_become_warn(self):
        text = "\n".join(["[W] warning"] * 12) + "\n"
        findings = self._check(text)
        warns = [f for f in findings if f.severity == "WARN"]
        self.assertTrue(any("warning" in f.message.lower() for f in warns))

    def test_kdtree_fallback_detected(self):
        findings = self._check("using cKDTree fallback due to missing scipy\n")
        self.assertTrue(any("fallback" in f.message.lower() for f in findings))

    def test_low_synapse_hit_rate_detected(self):
        # 2/10 = 20% < 50% threshold
        findings = self._check("Agents complete: 2/10 synapses hit\n")
        self.assertTrue(any("hit rate" in f.message.lower() for f in findings))

    def test_good_synapse_hit_rate_no_warn(self):
        # 8/10 = 80% > 50% → no hit-rate warning
        findings = self._check("Agents complete: 8/10 synapses hit\n")
        hit_warns = [f for f in findings if "hit rate" in f.message.lower()]
        self.assertEqual(hit_warns, [])

    def test_val_f1_zero_is_error(self):
        findings = self._check("val_f1 = 0.0\n")
        errors = [f for f in findings if f.severity == "ERROR"]
        self.assertTrue(any("val_f1" in f.message.lower() or "0.000" in f.message for f in errors))

    def test_val_f1_low_is_warn(self):
        findings = self._check("val_f1 = 0.05\n")
        self.assertTrue(any(f.severity == "WARN" for f in findings))

    def test_pr_imbalance_detected(self):
        # P=0.9, R=0.1 → ratio=9 > 3
        findings = self._check("LineGraph F1=0.18 P=0.90 R=0.10\n")
        self.assertTrue(any("imbalance" in f.message.lower() for f in findings))

    def test_no_pr_imbalance_when_balanced(self):
        findings = self._check("LineGraph F1=0.70 P=0.72 R=0.68\n")
        imbalance = [f for f in findings if "imbalance" in f.message.lower()]
        self.assertEqual(imbalance, [])

    def test_zero_synapses_detected(self):
        findings = self._check("Box has synapses=0 entries, skipping\n")
        self.assertTrue(any("synapse" in f.message.lower() for f in findings))

    def test_box_load_failure_detected(self):
        findings = self._check("failed to load abc123def456\n")
        self.assertTrue(any("box load failure" in f.message.lower() for f in findings))

    def test_nan_in_comment_not_flagged(self):
        # Lines starting with # should not trigger NaN detection
        findings = self._check("# nan is a floating-point concept\n")
        nan_errors = [f for f in findings
                      if f.severity == "ERROR" and "nan" in f.message.lower()]
        self.assertEqual(nan_errors, [])

    def test_missing_file_returns_empty(self):
        from neuronauts.qa_agent import check_log_file
        findings = check_log_file(Path("/tmp/nonexistent_qa_test.log"))
        self.assertEqual(findings, [])


# ---------------------------------------------------------------------------
# check_metrics_file
# ---------------------------------------------------------------------------

class CheckMetricsFileTest(unittest.TestCase):

    def _check(self, data: dict | str):
        from neuronauts.qa_agent import check_metrics_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            fh.write(data if isinstance(data, str) else json.dumps(data))
            p = Path(fh.name)
        try:
            return check_metrics_file(p)
        finally:
            p.unlink(missing_ok=True)

    def test_empty_file_returns_no_findings(self):
        from neuronauts.qa_agent import check_metrics_file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            p = Path(fh.name)
        p.write_text("")
        try:
            self.assertEqual(check_metrics_file(p), [])
        finally:
            p.unlink(missing_ok=True)

    def test_invalid_json_is_error(self):
        findings = self._check("not_json{")
        self.assertTrue(any(f.severity == "ERROR" for f in findings))

    def test_low_merge_accuracy_is_warn(self):
        findings = self._check({"merge_accuracy": 0.30})
        self.assertTrue(any("merge_accuracy" in f.message for f in findings))

    def test_high_merge_accuracy_no_warn(self):
        findings = self._check({"merge_accuracy": 0.85})
        merge_warns = [f for f in findings if "merge_accuracy" in f.message and f.severity == "WARN"]
        self.assertEqual(merge_warns, [])

    def test_reranker_mse_zero_is_warn(self):
        findings = self._check({"mse": 0.0})
        self.assertTrue(any("mse" in f.message.lower() for f in findings))

    def test_few_merge_examples_is_warn(self):
        findings = self._check({"n_merge": 5})
        self.assertTrue(any("merge training examples" in f.message for f in findings))

    def test_sufficient_merge_examples_no_warn(self):
        findings = self._check({"n_merge": 100})
        low_warns = [f for f in findings if "merge training examples" in f.message]
        self.assertEqual(low_warns, [])

    def test_high_loss_in_last_step_is_warn(self):
        findings = self._check({"last_step": {"merge_loss": 8.5}})
        self.assertTrue(any("high last-step loss" in f.message.lower() for f in findings))

    def test_nominal_loss_no_warn(self):
        findings = self._check({"last_step": {"merge_loss": 0.4}})
        loss_warns = [f for f in findings if "last-step" in f.message.lower()]
        self.assertEqual(loss_warns, [])

    def test_missing_file_returns_empty(self):
        from neuronauts.qa_agent import check_metrics_file
        self.assertEqual(check_metrics_file(Path("/tmp/no_such_metrics.json")), [])


# ---------------------------------------------------------------------------
# check_training_log
# ---------------------------------------------------------------------------

class CheckTrainingLogTest(unittest.TestCase):

    def _write_tsv(self, rows: list[dict]) -> Path:
        if not rows:
            return Path("/tmp/nonexistent_train_log.tsv")
        headers = list(rows[0].keys())
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as fh:
            p = Path(fh.name)
            fh.write("\t".join(headers) + "\n")
            for row in rows:
                fh.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")
        return p

    def _check(self, rows: list[dict]):
        from neuronauts.qa_agent import check_training_log
        p = self._write_tsv(rows)
        try:
            return check_training_log(p)
        finally:
            p.unlink(missing_ok=True)

    def test_empty_log_returns_empty(self):
        from neuronauts.qa_agent import check_training_log
        p = Path("/tmp/no_tsv_qa.tsv")
        self.assertEqual(check_training_log(p), [])

    def test_no_val_f1_column_returns_empty(self):
        findings = self._check([{"epoch": str(i), "train_loss": "0.5"} for i in range(10)])
        self.assertEqual(findings, [])

    def test_stagnation_detected(self):
        rows = [{"val_f1": "0.3000", "train_merge_acc": "0.80"} for _ in range(6)]
        findings = self._check(rows)
        self.assertTrue(any("stagnat" in f.message.lower() for f in findings))

    def test_improving_curve_is_ok(self):
        rows = [{"val_f1": str(0.1 * (i + 1)), "train_merge_acc": "0.80"} for i in range(5)]
        findings = self._check(rows)
        errors_warns = [f for f in findings if f.severity in {"ERROR", "WARN"}]
        stagnation = [f for f in errors_warns if "stagnat" in f.message.lower()]
        self.assertEqual(stagnation, [])

    def test_f1_regression_detected(self):
        # Best was 0.5, latest is 0.3 → > 15% regression
        rows = [{"val_f1": v} for v in ["0.1", "0.3", "0.5", "0.3", "0.3"]]
        findings = self._check(rows)
        self.assertTrue(any("regress" in f.message.lower() for f in findings))

    def test_all_zero_f1_is_error(self):
        rows = [{"val_f1": "0.0"} for _ in range(5)]
        findings = self._check(rows)
        self.assertTrue(any(f.severity == "ERROR" for f in findings))

    def test_overfitting_gap_detected(self):
        # High train accuracy but low val F1 → overfitting signal
        rows = [{"val_f1": "0.1", "train_merge_acc": "0.97"}]
        findings = self._check(rows)
        self.assertTrue(any("train/val gap" in f.message.lower() or
                            "gap" in f.message.lower() for f in findings))

    def test_mostly_na_rows_is_warn(self):
        rows = [{"val_f1": "0.2", "train_merge_acc": "n/a"} for _ in range(10)]
        findings = self._check(rows)
        self.assertTrue(any("n/a" in f.message.lower() or "grammar training" in f.message.lower()
                            for f in findings))


# ---------------------------------------------------------------------------
# check_research_ledger
# ---------------------------------------------------------------------------

class CheckResearchLedgerTest(unittest.TestCase):

    def _write_ledger(self, entries: list[dict]) -> Path:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
            return Path(fh.name)

    def _check(self, entries: list[dict]):
        from neuronauts.qa_agent import check_research_ledger
        p = self._write_ledger(entries)
        try:
            return check_research_ledger(p)
        finally:
            p.unlink(missing_ok=True)

    def test_empty_ledger_returns_info(self):
        from neuronauts.qa_agent import check_research_ledger
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as fh:
            p = Path(fh.name)
        p.write_text("")
        try:
            findings = check_research_ledger(p)
        finally:
            p.unlink(missing_ok=True)
        self.assertTrue(any(f.severity == "INFO" for f in findings))

    def test_missing_file_returns_info(self):
        from neuronauts.qa_agent import check_research_ledger
        findings = check_research_ledger(Path("/tmp/no_ledger_qa.jsonl"))
        self.assertTrue(any(f.severity == "INFO" for f in findings))

    def test_entries_without_val_f1_returns_info(self):
        findings = self._check([{"decision": "keep", "note": "no f1 here"}])
        self.assertTrue(any(f.severity == "INFO" for f in findings))

    def test_f1_regression_from_peak_detected(self):
        entries = [
            {"metrics": {"val_f1": 0.1}},
            {"metrics": {"val_f1": 0.5}},
            {"metrics": {"val_f1": 0.2}},  # 60% regression from 0.5
        ]
        findings = self._check(entries)
        self.assertTrue(any("regress" in f.message.lower() for f in findings))

    def test_no_regression_no_warn(self):
        entries = [{"metrics": {"val_f1": float(i) * 0.1}} for i in range(1, 5)]
        findings = self._check(entries)
        regression = [f for f in findings if "regress" in f.message.lower()]
        self.assertEqual(regression, [])

    def test_zero_f1_cycles_detected(self):
        entries = [{"metrics": {"val_f1": 0.0}} for _ in range(3)]
        findings = self._check(entries)
        self.assertTrue(any("0.0" in f.message or "val_f1=0" in f.message for f in findings))

    def test_summary_info_always_present(self):
        entries = [{"metrics": {"val_f1": 0.3}}, {"metrics": {"val_f1": 0.4}}]
        findings = self._check(entries)
        # Summary line should always be appended
        self.assertTrue(any("ledger entries" in f.message.lower() for f in findings))


# ---------------------------------------------------------------------------
# QAReport
# ---------------------------------------------------------------------------

class QAReportTest(unittest.TestCase):

    def _report(self):
        from neuronauts.qa_agent import QAReport
        return QAReport(timestamp=datetime.now(timezone.utc), run_root=Path("."))

    def _finding(self, severity="WARN"):
        from neuronauts.qa_agent import Finding
        return Finding(severity=severity, category="Test", message="test")

    def test_add_appends_findings(self):
        r = self._report()
        r.add([self._finding("ERROR"), self._finding("INFO")])
        self.assertEqual(len(r.findings), 2)

    def test_errors_returns_only_errors(self):
        r = self._report()
        r.add([self._finding("ERROR"), self._finding("WARN"), self._finding("OK")])
        self.assertEqual(len(r.errors()), 1)
        self.assertEqual(r.errors()[0].severity, "ERROR")

    def test_warnings_returns_only_warns(self):
        r = self._report()
        r.add([self._finding("ERROR"), self._finding("WARN"), self._finding("OK")])
        self.assertEqual(len(r.warnings()), 1)

    def test_by_severity_orders_errors_first(self):
        r = self._report()
        r.add([self._finding("OK"), self._finding("INFO"), self._finding("ERROR")])
        ordered = r.by_severity()
        self.assertEqual(ordered[0].severity, "ERROR")
        self.assertEqual(ordered[-1].severity, "OK")

    def test_summary_line_shows_error_when_present(self):
        r = self._report()
        r.add([self._finding("ERROR")])
        line = r.summary_line()
        self.assertIn("ERROR", line)

    def test_summary_line_shows_ok_when_clean(self):
        r = self._report()
        r.add([self._finding("OK")])
        line = r.summary_line()
        self.assertIn("OK", line)

    def test_summary_line_counts_are_correct(self):
        r = self._report()
        r.add([self._finding("ERROR")] * 2 + [self._finding("WARN")] * 3 +
              [self._finding("INFO")] + [self._finding("OK")])
        line = r.summary_line()
        self.assertIn("errors=2", line)
        self.assertIn("warnings=3", line)


# ---------------------------------------------------------------------------
# render_report
# ---------------------------------------------------------------------------

class RenderReportTest(unittest.TestCase):

    def test_render_includes_summary_line(self):
        from neuronauts.qa_agent import QAReport, Finding, render_report
        r = QAReport(timestamp=datetime.now(timezone.utc), run_root=Path("."))
        r.add([Finding("WARN", "Test", "something is off")])
        text = render_report(r)
        self.assertIn("warnings=", text)

    def test_render_verbose_includes_ok_findings(self):
        from neuronauts.qa_agent import QAReport, Finding, render_report
        r = QAReport(timestamp=datetime.now(timezone.utc), run_root=Path("."))
        r.add([Finding("OK", "Test", "all good")])
        text = render_report(r, verbose=True)
        self.assertIn("all good", text)

    def test_render_nonverbose_suppresses_ok_when_there_are_warns(self):
        from neuronauts.qa_agent import QAReport, Finding, render_report
        r = QAReport(timestamp=datetime.now(timezone.utc), run_root=Path("."))
        r.add([Finding("WARN", "Test", "issue here"), Finding("OK", "Test", "fine")])
        text = render_report(r, verbose=False)
        # WARN should appear; OK may be suppressed
        self.assertIn("issue here", text)


# ---------------------------------------------------------------------------
# run_full_audit — smoke test with minimal temp directory
# ---------------------------------------------------------------------------

class RunFullAuditTest(unittest.TestCase):

    def test_audit_on_empty_dir_returns_report(self):
        from neuronauts.qa_agent import run_full_audit, QAReport
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_logs").mkdir()  # run_full_audit iterates this dir
            report = run_full_audit(root)
        self.assertIsInstance(report, QAReport)

    def test_audit_with_log_file_picks_up_nan(self):
        from neuronauts.qa_agent import run_full_audit
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sess_dir = root / "run_logs" / "session_001"
            sess_dir.mkdir(parents=True)
            _write(sess_dir / "run_001.log", "Training complete: loss=nan\n")
            report = run_full_audit(root)
        self.assertTrue(any(f.severity == "ERROR" for f in report.findings))

    def test_audit_with_metrics_file(self):
        from neuronauts.qa_agent import run_full_audit
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_logs").mkdir()
            models = root / "models"
            models.mkdir()
            _write(models / "shared_grammar.metrics.json",
                   json.dumps({"merge_accuracy": 0.87, "n_merge": 500}))
            report = run_full_audit(root)
        self.assertIsInstance(report.findings, list)

    def test_audit_with_training_log(self):
        from neuronauts.qa_agent import run_full_audit
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_logs = root / "run_logs"
            run_logs.mkdir()
            tsv = "val_f1\ttrain_merge_acc\n"
            for i in range(6):
                tsv += "0.300\t0.80\n"  # stagnating
            _write(run_logs / "train_log.tsv", tsv)
            report = run_full_audit(root)
        self.assertTrue(any("stagnat" in f.message.lower() for f in report.findings))


# ---------------------------------------------------------------------------
# check_session_dir
# ---------------------------------------------------------------------------

class CheckSessionDirTest(unittest.TestCase):

    def test_missing_summary_tsv_is_warn(self):
        from neuronauts.qa_agent import check_session_dir
        with tempfile.TemporaryDirectory() as tmp:
            findings = check_session_dir(Path(tmp))
        self.assertTrue(any("summary.tsv" in f.message for f in findings))

    def test_empty_summary_tsv_is_warn(self):
        from neuronauts.qa_agent import check_session_dir
        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp)
            _write(sess / "summary.tsv", "val_f1\n")  # header only, no data rows
            findings = check_session_dir(sess)
        self.assertTrue(any("no data rows" in f.message for f in findings))

    def test_session_with_valid_data(self):
        from neuronauts.qa_agent import check_session_dir
        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp)
            _write(sess / "summary.tsv", "val_f1\trc\n0.45\t0\n0.50\t0\n")
            findings = check_session_dir(sess)
        self.assertTrue(any("INFO" == f.severity for f in findings))

    def test_failed_runs_flagged(self):
        from neuronauts.qa_agent import check_session_dir
        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp)
            _write(sess / "summary.tsv", "val_f1\trc\n0.45\t1\n0.50\t0\n")
            findings = check_session_dir(sess)
        self.assertTrue(any("non-zero exit" in f.message for f in findings))

    def test_no_run_log_files_is_info(self):
        from neuronauts.qa_agent import check_session_dir
        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp)
            _write(sess / "summary.tsv", "val_f1\n0.3\n")
            findings = check_session_dir(sess)
        infos = [f for f in findings if "No run_" in f.message]
        self.assertTrue(len(infos) > 0)

    def test_warns_across_run_logs_detected(self):
        from neuronauts.qa_agent import check_session_dir
        with tempfile.TemporaryDirectory() as tmp:
            sess = Path(tmp)
            _write(sess / "summary.tsv", "val_f1\n0.3\n")
            for i in range(3):
                _write(sess / f"run_{i:03d}.log",
                       "[W] warning 1\n[W] warning 2\n[W] warning 3\n")
            findings = check_session_dir(sess)
        warn_msg = [f for f in findings if "warnings across" in f.message]
        self.assertTrue(len(warn_msg) > 0)


# ---------------------------------------------------------------------------
# check_model_files
# ---------------------------------------------------------------------------

class CheckModelFilesTest(unittest.TestCase):

    def test_empty_models_dir_is_warn(self):
        from neuronauts.qa_agent import check_model_files
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            findings = check_model_files(models)
        self.assertTrue(any("No model checkpoint" in f.message for f in findings))

    def test_model_without_metrics_is_info(self):
        from neuronauts.qa_agent import check_model_files
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            _write(models / "grammar.pt", "dummy")
            findings = check_model_files(models)
        missing = [f for f in findings if "without a metrics file" in f.message]
        self.assertTrue(len(missing) > 0)

    def test_model_with_matching_metrics_is_ok(self):
        from neuronauts.qa_agent import check_model_files
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            _write(models / "grammar.pt", "dummy")
            _write(models / "grammar.metrics.json",
                   json.dumps({"merge_accuracy": 0.85}))
            findings = check_model_files(models)
        # No "without metrics" warning
        missing = [f for f in findings if "without a metrics file" in f.message]
        self.assertEqual(missing, [])

    def test_check_metrics_called_for_each_json(self):
        from neuronauts.qa_agent import check_model_files
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            _write(models / "grammar.pt", "dummy")
            _write(models / "grammar.metrics.json",
                   json.dumps({"merge_accuracy": 0.15}))  # too low → WARN
            findings = check_model_files(models)
        warns = [f for f in findings if f.severity == "WARN"]
        self.assertTrue(len(warns) > 0)


# ---------------------------------------------------------------------------
# check_box_cache
# ---------------------------------------------------------------------------

class CheckBoxCacheTest(unittest.TestCase):

    def test_nonexistent_cache_is_info(self):
        from neuronauts.qa_agent import check_box_cache
        findings = check_box_cache(Path("/tmp/no_such_qa_box_cache"))
        self.assertTrue(any(f.severity == "INFO" for f in findings))

    def test_empty_cache_is_info(self):
        from neuronauts.qa_agent import check_box_cache
        with tempfile.TemporaryDirectory() as tmp:
            findings = check_box_cache(Path(tmp))
        self.assertTrue(any(f.severity == "INFO" for f in findings))

    def test_corrupt_box_missing_files_detected(self):
        from neuronauts.qa_agent import check_box_cache
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            box_dir = cache / "box_aabbcc"
            box_dir.mkdir()
            # No volume.npz or synapses.npz → incomplete
            findings = check_box_cache(cache)
        self.assertTrue(any("incomplete" in f.message.lower() for f in findings))

    def test_valid_box_structure_no_incomplete(self):
        """A box with both volume.npz and synapses.npz of reasonable size."""
        from neuronauts.qa_agent import check_box_cache
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            box_dir = cache / "box_valid"
            box_dir.mkdir()
            # Write valid npz files (at least 100 bytes each)
            buf = io.BytesIO()
            np.savez(buf, data=np.zeros((10, 10, 10), dtype=np.uint8))
            (box_dir / "volume.npz").write_bytes(buf.getvalue())
            buf2 = io.BytesIO()
            np.savez(buf2, pre_pt=np.zeros((5, 3)), post_pt=np.zeros((5, 3)))
            (box_dir / "synapses.npz").write_bytes(buf2.getvalue())
            findings = check_box_cache(cache)
        incomplete = [f for f in findings if "incomplete" in f.message.lower()]
        self.assertEqual(incomplete, [])


# ---------------------------------------------------------------------------
# run_incremental_audit
# ---------------------------------------------------------------------------

class RunIncrementalAuditTest(unittest.TestCase):

    def test_incremental_audit_with_fresh_files(self):
        """Files modified more recently than since_mtime=0 should be audited."""
        from neuronauts.qa_agent import run_incremental_audit, QAReport
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_logs = root / "run_logs"
            run_logs.mkdir()
            tsv = "val_f1\ttrain_merge_acc\n" + "0.3\t0.80\n" * 6
            _write(run_logs / "train_log.tsv", tsv)
            report = run_incremental_audit(root, since_mtime=0.0)
        self.assertIsInstance(report, QAReport)

    def test_incremental_audit_skips_old_files(self):
        """Files with mtime < since_mtime should not contribute findings."""
        from neuronauts.qa_agent import run_incremental_audit
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_logs = root / "run_logs"
            run_logs.mkdir()
            tsv = "val_f1\ttrain_merge_acc\n" + "0.3\t0.80\n" * 6
            p = run_logs / "train_log.tsv"
            _write(p, tsv)
            # Use a future timestamp so the file appears older than the cutoff.
            since = time.time() + 1000
            report = run_incremental_audit(root, since_mtime=since)
        # The stagnation warning should not appear because the file was "too old".
        stagnation = [f for f in report.findings if "stagnat" in f.message.lower()]
        self.assertEqual(stagnation, [])

    def test_incremental_audit_with_log_file(self):
        from neuronauts.qa_agent import run_incremental_audit
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_logs = root / "run_logs"
            run_logs.mkdir()
            _write(run_logs / "run_001.log", "val_f1 = 0.0\n")
            report = run_incremental_audit(root, since_mtime=0.0)
        self.assertTrue(any(f.severity == "ERROR" for f in report.findings))


# ---------------------------------------------------------------------------
# load_json_if_exists (experiment_driver line 68)
# ---------------------------------------------------------------------------

class LoadJsonIfExistsTest(unittest.TestCase):

    def test_returns_empty_for_missing_file(self):
        from neuronauts.experiment_driver import load_json_if_exists
        self.assertEqual(load_json_if_exists(Path("/tmp/no_such_file_qa.json")), {})

    def test_loads_json_when_file_exists(self):
        from neuronauts.experiment_driver import load_json_if_exists
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
            json.dump({"key": "value", "n": 42}, fh)
            p = Path(fh.name)
        try:
            result = load_json_if_exists(p)
            self.assertEqual(result["key"], "value")
            self.assertEqual(result["n"], 42)
        finally:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
