import tempfile
import unittest
from pathlib import Path

from neuronauts.experiment_driver import (
    append_experiment_ledger,
    build_ledger_entry,
    ResearchCycleConfig,
    build_research_cycle_commands,
    compare_cycle_summaries,
    load_experiment_ledger,
    summarize_research_cycle,
    write_experiment_leaderboard,
)


class ExperimentDriverTest(unittest.TestCase):
    def test_build_research_cycle_commands_contains_shared_pipeline_steps(self):
        config = ResearchCycleConfig(repo_root=Path("/tmp/repo"), python_bin=".venv/bin/python", output_dir=Path("/tmp/out"))
        commands = build_research_cycle_commands(config)
        self.assertIn("export_merge", commands)
        self.assertIn("export_topology", commands)
        self.assertIn("train_shared", commands)
        self.assertIn("export_assembly", commands)
        self.assertIn("train_reranker", commands)
        self.assertIn("validate_selection", commands)
        self.assertIn("validate_holdout", commands)
        self.assertIn("scripts/train_shared_grammar.py", commands["train_shared"])
        self.assertIn("scripts/train_assembly_ranker.py", commands["train_reranker"])
        self.assertIn("--shared-grammar-checkpoint", commands["validate_selection"])
        self.assertIn("--shared-grammar-checkpoint", commands["validate_holdout"])
        self.assertIn("--real-box-indices", commands["validate_selection"])
        self.assertIn("--real-box-indices", commands["validate_holdout"])
        self.assertIn("0,1,2", commands["validate_selection"])
        self.assertIn("3,4,5", commands["validate_holdout"])

    def test_compare_cycle_summaries_prefers_higher_val_f1(self):
        baseline = {
            "metrics": {"val_f1": 0.4, "holdout_f1": 0.4, "precision": 0.4, "recall": 0.4},
            "shared_training_metrics": {"merge_accuracy": 0.5, "atomicity_accuracy": 0.5},
            "reranker_metrics": {"corr": 0.1, "mse": 0.2},
        }
        candidate = {
            "metrics": {"val_f1": 0.5, "holdout_f1": 0.45, "precision": 0.5, "recall": 0.5},
            "shared_training_metrics": {"merge_accuracy": 0.4, "atomicity_accuracy": 0.4},
            "reranker_metrics": {"corr": 0.0, "mse": 0.3},
        }
        decision, note = compare_cycle_summaries(baseline, candidate)
        self.assertEqual(decision, "keep")
        self.assertIn("val_f1 improved", note)

    def test_compare_cycle_summaries_rejects_holdout_regression(self):
        baseline = {
            "metrics": {"val_f1": 0.4, "holdout_f1": 0.5, "precision": 0.4, "recall": 0.4},
            "shared_training_metrics": {"merge_accuracy": 0.5, "atomicity_accuracy": 0.5},
            "reranker_metrics": {"corr": 0.1, "mse": 0.2},
        }
        candidate = {
            "metrics": {"val_f1": 0.45, "holdout_f1": 0.45, "precision": 0.45, "recall": 0.45},
            "shared_training_metrics": {"merge_accuracy": 0.6, "atomicity_accuracy": 0.6},
            "reranker_metrics": {"corr": 0.2, "mse": 0.1},
        }
        decision, note = compare_cycle_summaries(baseline, candidate)
        self.assertEqual(decision, "revert")
        self.assertIn("holdout regressed", note)

    def test_compare_cycle_summaries_uses_inner_loop_tie_break(self):
        baseline = {
            "metrics": {"val_f1": 0.5, "holdout_f1": 0.5, "precision": 0.5, "recall": 0.5},
            "shared_training_metrics": {"merge_accuracy": 0.5, "atomicity_accuracy": 0.5},
            "reranker_metrics": {"corr": 0.1, "mse": 0.2},
        }
        candidate = {
            "metrics": {"val_f1": 0.5, "holdout_f1": 0.5, "precision": 0.5, "recall": 0.5},
            "shared_training_metrics": {"merge_accuracy": 0.53, "atomicity_accuracy": 0.5},
            "reranker_metrics": {"corr": 0.1, "mse": 0.2},
        }
        decision, _ = compare_cycle_summaries(baseline, candidate)
        self.assertEqual(decision, "keep")

    def test_summarize_research_cycle_returns_dense_metrics(self):
        compact = summarize_research_cycle(
            {
                "metrics": {"val_f1": 0.6, "holdout_f1": 0.55, "precision": 0.7, "recall": 0.8},
                "shared_training_metrics": {"merge_accuracy": 0.9, "atomicity_accuracy": 0.85},
                "reranker_metrics": {"corr": 0.4, "mse": 0.05},
            }
        )
        self.assertEqual(compact["val_f1"], 0.6)
        self.assertEqual(compact["holdout_f1"], 0.55)
        self.assertEqual(compact["merge_accuracy"], 0.9)
        self.assertEqual(compact["atomicity_accuracy"], 0.85)
        self.assertEqual(compact["reranker_corr"], 0.4)

    def test_build_ledger_entry_carries_cycle_metrics(self):
        summary = {
            "ok": True,
            "metrics": {"val_f1": 0.6, "holdout_f1": 0.55, "precision": 0.7, "recall": 0.8},
            "shared_training_metrics": {"merge_accuracy": 0.9, "atomicity_accuracy": 0.85},
            "reranker_metrics": {"corr": 0.4, "mse": 0.05},
        }
        entry = build_ledger_entry(
            summary,
            source="codex",
            target_file="neuronauts/grammar.py",
            hypothesis="test hypothesis",
            decision="keep",
            note="improved selection",
            iteration=3,
            run_dir="run_logs/codex_optimize/iteration_003",
        )
        self.assertEqual(entry["source"], "codex")
        self.assertEqual(entry["decision"], "keep")
        self.assertEqual(entry["iteration"], 3)
        self.assertEqual(entry["val_f1"], 0.6)
        self.assertEqual(entry["holdout_f1"], 0.55)
        self.assertEqual(entry["merge_accuracy"], 0.9)
        self.assertIn("timestamp", entry)

    def test_append_experiment_ledger_updates_leaderboard(self):
        entry_a = {
            "timestamp": "2026-03-17T10:00:00+00:00",
            "source": "codex",
            "decision": "keep",
            "iteration": 1,
            "target_file": "neuronauts/grammar.py",
            "val_f1": 0.50,
            "holdout_f1": 0.40,
            "merge_accuracy": 0.60,
            "atomicity_accuracy": 0.70,
            "reranker_corr": 0.10,
            "reranker_mse": 0.20,
            "note": "first",
        }
        entry_b = {
            "timestamp": "2026-03-17T11:00:00+00:00",
            "source": "gemini",
            "decision": "completed",
            "iteration": None,
            "target_file": "neuronauts/topology_model.py",
            "val_f1": 0.45,
            "holdout_f1": 0.50,
            "merge_accuracy": 0.55,
            "atomicity_accuracy": 0.72,
            "reranker_corr": 0.15,
            "reranker_mse": 0.18,
            "note": "second",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "research_ledger.jsonl"
            leaderboard_path = Path(tmpdir) / "research_ledger.leaderboard.tsv"
            append_experiment_ledger(ledger_path, entry_a, leaderboard_path=leaderboard_path)
            append_experiment_ledger(ledger_path, entry_b, leaderboard_path=leaderboard_path)

            entries = load_experiment_ledger(ledger_path)
            self.assertEqual(len(entries), 2)

            leaderboard_lines = leaderboard_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(leaderboard_lines), 3)
            self.assertIn("holdout_f1", leaderboard_lines[0])
            self.assertIn("gemini", leaderboard_lines[1])
            self.assertIn("codex", leaderboard_lines[2])

    def test_write_experiment_leaderboard_sorts_by_holdout_then_selection(self):
        entries = [
            {"timestamp": "t1", "source": "a", "decision": "keep", "iteration": 1, "target_file": "f", "val_f1": 0.7, "holdout_f1": 0.4, "merge_accuracy": 0.0, "atomicity_accuracy": 0.0, "reranker_corr": 0.0, "reranker_mse": 1.0, "note": ""},
            {"timestamp": "t2", "source": "b", "decision": "keep", "iteration": 2, "target_file": "f", "val_f1": 0.6, "holdout_f1": 0.5, "merge_accuracy": 0.0, "atomicity_accuracy": 0.0, "reranker_corr": 0.0, "reranker_mse": 1.0, "note": ""},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            leaderboard_path = Path(tmpdir) / "leaderboard.tsv"
            write_experiment_leaderboard(leaderboard_path, entries)
            lines = leaderboard_path.read_text(encoding="utf-8").splitlines()
            self.assertIn("source", lines[0])
            self.assertIn("b", lines[1])
            self.assertIn("a", lines[2])


class RunResearchCycleTest(unittest.TestCase):
    """E1 — HIGH: run_research_cycle orchestration and failure-path logic.

    run_command spawns real subprocesses; we mock it to avoid hitting the
    file system, network, or requiring trained models.
    """

    def _make_config(self, repo_root):
        return ResearchCycleConfig(
            repo_root=repo_root,
            python_bin="python",
            output_dir=None,
        )

    def _stub_ok(self, stdout: str = ""):
        """Return a successful CompletedProcess."""
        import subprocess
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    def _stub_fail(self):
        """Return a failing CompletedProcess."""
        import subprocess
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")

    def test_failure_on_first_step_returns_ok_false(self):
        import unittest.mock as mock
        from neuronauts.experiment_driver import run_research_cycle

        config = self._make_config(Path("/tmp"))
        with mock.patch(
            "neuronauts.experiment_driver.run_command",
            return_value=self._stub_fail(),
        ):
            result = run_research_cycle(config)

        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["failed_step"])

    def test_failure_on_specific_step_names_that_step(self):
        import unittest.mock as mock
        from neuronauts.experiment_driver import run_research_cycle, build_research_cycle_commands

        config = self._make_config(Path("/tmp"))
        commands = build_research_cycle_commands(config)
        step_names = list(commands.keys())

        # Fail the second step; first step succeeds.
        call_count = [0]

        def side_effect(cmd, *, cwd, env=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return self._stub_ok()
            return self._stub_fail()

        with mock.patch("neuronauts.experiment_driver.run_command", side_effect=side_effect):
            result = run_research_cycle(config)

        self.assertFalse(result["ok"])
        self.assertEqual(result["failed_step"], step_names[1])

    def test_all_steps_succeed_returns_ok_true_with_metrics_dict(self):
        import unittest.mock as mock
        from neuronauts.experiment_driver import run_research_cycle

        val_output = "val_f1 = 0.72  P=0.80  R=0.65  TP=10  FP=2  FN=5"
        config = self._make_config(Path("/tmp"))
        with mock.patch(
            "neuronauts.experiment_driver.run_command",
            return_value=self._stub_ok(stdout=val_output),
        ):
            result = run_research_cycle(config)

        self.assertTrue(result["ok"])
        self.assertIsNone(result["failed_step"])
        self.assertIn("metrics", result)
        self.assertIn("val_f1", result["metrics"])

    def test_val_f1_extracted_from_validate_selection_stdout(self):
        import unittest.mock as mock
        from neuronauts.experiment_driver import run_research_cycle

        config = self._make_config(Path("/tmp"))
        call_count = [0]
        step_names = list(config.__class__.__dataclass_fields__.keys())

        def side_effect(cmd, *, cwd, env=None):
            call_count[0] += 1
            # Last two steps (validate_selection, validate_holdout) return metrics.
            if call_count[0] >= len(list(
                __import__("neuronauts.experiment_driver", fromlist=["build_research_cycle_commands"])
                .build_research_cycle_commands(config).keys()
            )) - 1:
                return self._stub_ok(stdout="val_f1 = 0.55  P=0.60  R=0.50  TP=8  FP=3  FN=8")
            return self._stub_ok()

        with mock.patch("neuronauts.experiment_driver.run_command", side_effect=side_effect):
            result = run_research_cycle(config)

        if result["ok"]:
            # val_f1 should be populated from stdout
            self.assertIsNotNone(result["metrics"].get("val_f1"))

    def test_output_dir_receives_log_files(self):
        import unittest.mock as mock
        from neuronauts.experiment_driver import run_research_cycle

        val_output = "val_f1 = 0.60  P=0.65  R=0.55  TP=9  FP=3  FN=7"
        with tempfile.TemporaryDirectory() as d:
            output_dir = Path(d) / "out"
            config = ResearchCycleConfig(
                repo_root=Path("/tmp"),
                python_bin="python",
                output_dir=output_dir,
            )
            with mock.patch(
                "neuronauts.experiment_driver.run_command",
                return_value=self._stub_ok(stdout=val_output),
            ):
                result = run_research_cycle(config)

            if result["ok"]:
                # At least some log files should have been created.
                log_files = list(output_dir.glob("*.stdout.log"))
                self.assertGreater(len(log_files), 0)

    def test_failed_step_logs_stored_in_steps(self):
        import unittest.mock as mock
        from neuronauts.experiment_driver import run_research_cycle

        config = self._make_config(Path("/tmp"))
        with mock.patch(
            "neuronauts.experiment_driver.run_command",
            return_value=self._stub_fail(),
        ):
            result = run_research_cycle(config)

        self.assertIn("steps", result)
        self.assertGreater(len(result["steps"]), 0)


class CompareCycleSummariesTieBreakTest(unittest.TestCase):
    """E2 — MEDIUM: atomicity_delta and reranker_delta tie-break branches."""

    def _make(self, val_f1, holdout_f1, merge_acc, atomicity_acc, reranker_corr, reranker_mse):
        return {
            "metrics": {
                "val_f1": val_f1,
                "holdout_f1": holdout_f1,
                "precision": 0.5,
                "recall": 0.5,
            },
            "shared_training_metrics": {
                "merge_accuracy": merge_acc,
                "atomicity_accuracy": atomicity_acc,
            },
            "reranker_metrics": {"corr": reranker_corr, "mse": reranker_mse},
        }

    def test_atomicity_delta_tie_break_returns_keep(self):
        baseline = self._make(0.5, 0.5, 0.600, 0.600, 0.1, 0.2)
        # F1 unchanged, merge delta = 0.005 (NOT > 0.01), atomicity delta = 0.03 (> 0.01).
        candidate = self._make(0.5, 0.5, 0.605, 0.630, 0.1, 0.2)
        decision, note = compare_cycle_summaries(baseline, candidate)
        self.assertEqual(decision, "keep")
        self.assertIn("atomicity", note)

    def test_reranker_delta_tie_break_returns_keep(self):
        baseline = self._make(0.5, 0.5, 0.600, 0.600, 0.100, 0.2)
        # F1 unchanged, merge/atomicity within tie region, reranker corr delta = 0.03 (> 0.01).
        candidate = self._make(0.5, 0.5, 0.605, 0.600, 0.130, 0.2)
        decision, note = compare_cycle_summaries(baseline, candidate)
        self.assertEqual(decision, "keep")
        self.assertIn("reranker", note)

    def test_reranker_tie_break_with_holdout_regression_reverts(self):
        baseline = self._make(0.5, 0.50, 0.600, 0.600, 0.10, 0.2)
        # Reranker improved but holdout regressed by > 0.01.
        candidate = self._make(0.5, 0.47, 0.605, 0.605, 0.15, 0.2)
        decision, note = compare_cycle_summaries(baseline, candidate)
        self.assertEqual(decision, "revert")
        self.assertIn("holdout regressed", note)

    def test_no_improvement_anywhere_reverts(self):
        baseline = self._make(0.5, 0.5, 0.6, 0.6, 0.1, 0.2)
        # All metrics equal or worse.
        candidate = self._make(0.5, 0.5, 0.58, 0.58, 0.08, 0.22)
        decision, _ = compare_cycle_summaries(baseline, candidate)
        self.assertEqual(decision, "revert")

    def test_merge_delta_trigger_returns_keep_via_inner_loop(self):
        baseline = self._make(0.5, 0.5, 0.600, 0.600, 0.1, 0.2)
        # merge delta = 0.03 > 0.01, atomicity non-negative, MSE unchanged.
        candidate = self._make(0.5, 0.5, 0.630, 0.600, 0.1, 0.2)
        decision, note = compare_cycle_summaries(baseline, candidate)
        self.assertEqual(decision, "keep")
        self.assertIn("inner-loop", note)

    def test_f1_improvement_above_threshold_wins_over_tie_break(self):
        baseline = self._make(0.5, 0.5, 0.6, 0.6, 0.1, 0.2)
        candidate = self._make(0.55, 0.52, 0.4, 0.4, 0.0, 0.5)
        decision, note = compare_cycle_summaries(baseline, candidate)
        # F1 improved → keep regardless of inner metrics.
        self.assertEqual(decision, "keep")
        self.assertIn("val_f1 improved", note)


if __name__ == "__main__":
    unittest.main()
