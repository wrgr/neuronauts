"""Tests for experiment_driver.py covering the 6 previously-missing lines.

Missing lines were:
  43   — run_command return statement (subprocess.run result)
  68   — as_int() inside parse_validation_metrics (integer extraction)
  121  — build_ledger_entry: entry["failed_step"] = failed_step
  128  — load_experiment_ledger: return [] when path doesn't exist
  132  — load_experiment_ledger: continue on blank lines
  216  — compare_cycle_summaries: "val_f1 regressed" revert path
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# run_command
# ---------------------------------------------------------------------------

class RunCommandTest(unittest.TestCase):

    def test_run_command_captures_stdout(self):
        from neuronauts.experiment_driver import run_command
        result = run_command(
            [sys.executable, "-c", "print('hello_qa')"],
            cwd=Path("."),
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello_qa", result.stdout)

    def test_run_command_nonzero_exit(self):
        from neuronauts.experiment_driver import run_command
        result = run_command(
            [sys.executable, "-c", "raise SystemExit(2)"],
            cwd=Path("."),
        )
        self.assertEqual(result.returncode, 2)

    def test_run_command_captures_stderr(self):
        from neuronauts.experiment_driver import run_command
        result = run_command(
            [sys.executable, "-c", "import sys; sys.stderr.write('err_out')"],
            cwd=Path("."),
        )
        self.assertIn("err_out", result.stderr)

    def test_run_command_with_env(self):
        import os
        from neuronauts.experiment_driver import run_command
        env = {**os.environ, "QA_TEST_VAR": "42"}
        result = run_command(
            [sys.executable, "-c",
             "import os; print(os.environ.get('QA_TEST_VAR','missing'))"],
            cwd=Path("."),
            env=env,
        )
        self.assertIn("42", result.stdout)


# ---------------------------------------------------------------------------
# parse_validation_metrics  (covers as_int path — line 68)
# ---------------------------------------------------------------------------

class ParseValidationMetricsTest(unittest.TestCase):

    def test_extracts_val_f1(self):
        from neuronauts.experiment_driver import parse_validation_metrics
        m = parse_validation_metrics("val_f1 = 0.4321\n")
        self.assertAlmostEqual(m["val_f1"], 0.4321, places=4)

    def test_extracts_precision_and_recall(self):
        from neuronauts.experiment_driver import parse_validation_metrics
        m = parse_validation_metrics("P=0.75 R=0.60\n")
        self.assertAlmostEqual(m["precision"], 0.75, places=4)
        self.assertAlmostEqual(m["recall"], 0.60, places=4)

    def test_extracts_integer_tp_fp_fn(self):
        from neuronauts.experiment_driver import parse_validation_metrics
        m = parse_validation_metrics("TP=120 FP=30 FN=15\n")
        self.assertEqual(m["tp"], 120)
        self.assertEqual(m["fp"], 30)
        self.assertEqual(m["fn"], 15)

    def test_missing_fields_return_none(self):
        from neuronauts.experiment_driver import parse_validation_metrics
        m = parse_validation_metrics("no metrics here")
        self.assertIsNone(m["val_f1"])
        self.assertIsNone(m["tp"])

    def test_all_fields_present(self):
        from neuronauts.experiment_driver import parse_validation_metrics
        text = "val_f1 = 0.55\nP=0.60 R=0.50\nTP=100 FP=20 FN=30\n"
        m = parse_validation_metrics(text)
        self.assertAlmostEqual(m["val_f1"], 0.55, places=4)
        self.assertEqual(m["tp"], 100)


# ---------------------------------------------------------------------------
# build_ledger_entry  (covers line 121: failed_step branch)
# ---------------------------------------------------------------------------

class BuildLedgerEntryTest(unittest.TestCase):

    def _minimal_summary(self, **kwargs) -> dict:
        base = {
            "ok": True,
            "failed_step": None,
            "metrics": {"val_f1": 0.3, "holdout_f1": 0.25,
                        "precision": 0.4, "recall": 0.2},
            "shared_training_metrics": {},
            "reranker_metrics": {},
        }
        base.update(kwargs)
        return base

    def test_failed_step_included_when_present(self):
        from neuronauts.experiment_driver import build_ledger_entry
        summary = self._minimal_summary(
            ok=False,
            failed_step="export_merge",
        )
        entry = build_ledger_entry(summary, source="test")
        self.assertIn("failed_step", entry)
        self.assertEqual(entry["failed_step"], "export_merge")

    def test_failed_step_excluded_when_none(self):
        from neuronauts.experiment_driver import build_ledger_entry
        summary = self._minimal_summary(ok=True, failed_step=None)
        entry = build_ledger_entry(summary, source="test")
        self.assertNotIn("failed_step", entry)

    def test_required_fields_present(self):
        from neuronauts.experiment_driver import build_ledger_entry
        summary = self._minimal_summary()
        entry = build_ledger_entry(summary, source="test", iteration=5,
                                   hypothesis="tweak lr", decision="keep")
        for key in ("timestamp", "source", "decision", "iteration", "val_f1",
                    "holdout_f1", "ok"):
            self.assertIn(key, entry)

    def test_decision_defaults_to_keep_on_success(self):
        from neuronauts.experiment_driver import build_ledger_entry
        summary = self._minimal_summary(ok=True)
        entry = build_ledger_entry(summary, source="test")
        self.assertEqual(entry["decision"], "keep")

    def test_decision_defaults_to_failed_on_failure(self):
        from neuronauts.experiment_driver import build_ledger_entry
        summary = self._minimal_summary(ok=False, failed_step="train_shared")
        entry = build_ledger_entry(summary, source="test")
        self.assertEqual(entry["decision"], "failed")


# ---------------------------------------------------------------------------
# load_experiment_ledger  (covers lines 128, 132)
# ---------------------------------------------------------------------------

class LoadExperimentLedgerTest(unittest.TestCase):

    def test_returns_empty_list_when_file_missing(self):
        from neuronauts.experiment_driver import load_experiment_ledger
        # Line 128: path.exists() is False → return []
        result = load_experiment_ledger(Path("/tmp/qa_no_such_ledger.jsonl"))
        self.assertEqual(result, [])

    def test_returns_empty_list_for_empty_file(self):
        from neuronauts.experiment_driver import load_experiment_ledger
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as fh:
            p = Path(fh.name)
        p.write_text("")
        try:
            result = load_experiment_ledger(p)
            self.assertEqual(result, [])
        finally:
            p.unlink(missing_ok=True)

    def test_blank_lines_are_skipped(self):
        from neuronauts.experiment_driver import load_experiment_ledger
        # Line 132: `continue` on empty/blank lines
        content = (
            json.dumps({"val_f1": 0.3}) + "\n"
            "\n"
            "   \n"
            + json.dumps({"val_f1": 0.4}) + "\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as fh:
            fh.write(content)
            p = Path(fh.name)
        try:
            result = load_experiment_ledger(p)
            self.assertEqual(len(result), 2)
        finally:
            p.unlink(missing_ok=True)

    def test_parses_json_entries(self):
        from neuronauts.experiment_driver import load_experiment_ledger
        entries = [{"val_f1": 0.1 * i, "ok": True} for i in range(3)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                         delete=False) as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
            p = Path(fh.name)
        try:
            result = load_experiment_ledger(p)
            self.assertEqual(len(result), 3)
            self.assertAlmostEqual(result[1]["val_f1"], 0.1, places=5)
        finally:
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# compare_cycle_summaries  (covers line 216: val_f1 regressed path)
# ---------------------------------------------------------------------------

class CompareCycleSummariesTest(unittest.TestCase):

    def _summary(self, val_f1=0.3, holdout_f1=0.3,
                 merge_acc=0.80, atom_acc=0.75, reranker_corr=0.5,
                 reranker_mse=0.1) -> dict:
        return {
            "ok": True,
            "failed_step": None,
            "metrics": {
                "val_f1": val_f1,
                "holdout_f1": holdout_f1,
                "precision": 0.5,
                "recall": 0.5,
            },
            "shared_training_metrics": {
                "merge_accuracy": merge_acc,
                "atomicity_accuracy": atom_acc,
            },
            "reranker_metrics": {
                "corr": reranker_corr,
                "mse": reranker_mse,
            },
        }

    def test_improved_val_f1_returns_keep(self):
        from neuronauts.experiment_driver import compare_cycle_summaries
        decision, _ = compare_cycle_summaries(
            self._summary(val_f1=0.3),
            self._summary(val_f1=0.5),
        )
        self.assertEqual(decision, "keep")

    def test_val_f1_regression_returns_revert(self):
        from neuronauts.experiment_driver import compare_cycle_summaries
        # Line 216: f1 dropped → "revert"
        decision, reason = compare_cycle_summaries(
            self._summary(val_f1=0.5),
            self._summary(val_f1=0.2),
        )
        self.assertEqual(decision, "revert")
        self.assertIn("regress", reason.lower())

    def test_selection_improved_but_holdout_regressed_reverts(self):
        from neuronauts.experiment_driver import compare_cycle_summaries
        decision, reason = compare_cycle_summaries(
            self._summary(val_f1=0.3, holdout_f1=0.5),
            self._summary(val_f1=0.5, holdout_f1=0.2),
        )
        self.assertEqual(decision, "revert")
        self.assertIn("holdout", reason.lower())

    def test_tie_region_merge_improvement_keeps(self):
        from neuronauts.experiment_driver import compare_cycle_summaries
        # Tiny F1 delta but merge_acc improved > 0.01
        decision, reason = compare_cycle_summaries(
            self._summary(val_f1=0.3000, merge_acc=0.70, reranker_mse=0.1),
            self._summary(val_f1=0.3000, merge_acc=0.85, reranker_mse=0.1),
        )
        self.assertEqual(decision, "keep")

    def test_no_improvement_returns_revert(self):
        from neuronauts.experiment_driver import compare_cycle_summaries
        decision, _ = compare_cycle_summaries(
            self._summary(val_f1=0.3, merge_acc=0.80, atom_acc=0.75,
                          reranker_corr=0.5, reranker_mse=0.1),
            self._summary(val_f1=0.3, merge_acc=0.80, atom_acc=0.75,
                          reranker_corr=0.5, reranker_mse=0.1),
        )
        self.assertEqual(decision, "revert")


# ---------------------------------------------------------------------------
# write_experiment_leaderboard  (bonus: covers leaderboard sort logic)
# ---------------------------------------------------------------------------

class WriteLeaderboardTest(unittest.TestCase):

    def test_leaderboard_orders_by_holdout_then_val_f1(self):
        from neuronauts.experiment_driver import write_experiment_leaderboard
        entries = [
            {"holdout_f1": 0.2, "val_f1": 0.3, "source": "a",
             "decision": "keep", "iteration": 1,
             "timestamp": "2026-01-01T00:00:00+00:00",
             "target_file": "f.py", "merge_accuracy": 0.8,
             "atomicity_accuracy": 0.75, "reranker_corr": 0.5,
             "reranker_mse": 0.1, "note": ""},
            {"holdout_f1": 0.5, "val_f1": 0.4, "source": "b",
             "decision": "keep", "iteration": 2,
             "timestamp": "2026-01-02T00:00:00+00:00",
             "target_file": "f.py", "merge_accuracy": 0.85,
             "atomicity_accuracy": 0.80, "reranker_corr": 0.6,
             "reranker_mse": 0.08, "note": ""},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv",
                                         delete=False) as fh:
            p = Path(fh.name)
        try:
            write_experiment_leaderboard(p, entries)
            text = p.read_text()
            lines = text.strip().splitlines()
            # Header + 2 data rows
            self.assertEqual(len(lines), 3)
            # First data row should be the entry with holdout_f1=0.5
            self.assertIn("0.5000", lines[1])
        finally:
            p.unlink(missing_ok=True)

    def test_leaderboard_creates_parent_dirs(self):
        from neuronauts.experiment_driver import write_experiment_leaderboard
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sub" / "dir" / "board.tsv"
            write_experiment_leaderboard(p, [])
            self.assertTrue(p.exists())


if __name__ == "__main__":
    unittest.main()
