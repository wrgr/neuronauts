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


if __name__ == "__main__":
    unittest.main()
