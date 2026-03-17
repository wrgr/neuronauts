import tempfile
import unittest
from pathlib import Path

from neuronauts.experiment_driver import append_experiment_ledger
from scripts.view_research_ledger import filter_entries, format_table, sort_entries


class ResearchLedgerViewerTest(unittest.TestCase):
    def test_filter_entries_by_source_and_threshold(self):
        entries = [
            {"source": "codex", "decision": "keep", "target_file": "neuronauts/grammar.py", "val_f1": 0.6, "holdout_f1": 0.5},
            {"source": "gemini", "decision": "completed", "target_file": "neuronauts/topology_model.py", "val_f1": 0.4, "holdout_f1": 0.6},
        ]

        class Args:
            source = "codex"
            decision = None
            target_file = "grammar"
            min_val_f1 = 0.5
            min_holdout_f1 = 0.4

        filtered = filter_entries(entries, Args())
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["source"], "codex")

    def test_sort_entries_prefers_higher_holdout_by_default(self):
        entries = [
            {"timestamp": "2026-03-17T10:00:00+00:00", "holdout_f1": 0.4},
            {"timestamp": "2026-03-17T11:00:00+00:00", "holdout_f1": 0.6},
        ]
        ordered = sort_entries(entries, sort_by="holdout_f1", ascending=False)
        self.assertEqual(float(ordered[0]["holdout_f1"]), 0.6)

    def test_format_table_renders_headers(self):
        table = format_table(
            [
                {
                    "timestamp": "2026-03-17T10:00:00+00:00",
                    "source": "codex",
                    "decision": "keep",
                    "iteration": 1,
                    "target_file": "neuronauts/grammar.py",
                    "val_f1": 0.6,
                    "holdout_f1": 0.5,
                    "merge_accuracy": 0.7,
                    "atomicity_accuracy": 0.8,
                    "reranker_corr": 0.2,
                    "note": "good run",
                }
            ]
        )
        self.assertIn("timestamp", table)
        self.assertIn("holdout_f1", table)
        self.assertIn("codex", table)

    def test_append_and_viewer_helpers_share_ledger_format(self):
        entry = {
            "timestamp": "2026-03-17T10:00:00+00:00",
            "source": "codex",
            "decision": "keep",
            "iteration": 1,
            "target_file": "neuronauts/grammar.py",
            "val_f1": 0.6,
            "holdout_f1": 0.5,
            "merge_accuracy": 0.7,
            "atomicity_accuracy": 0.8,
            "reranker_corr": 0.2,
            "reranker_mse": 0.1,
            "note": "good run",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger_path = Path(tmpdir) / "research_ledger.jsonl"
            append_experiment_ledger(ledger_path, entry)
            text = ledger_path.read_text(encoding="utf-8")
            self.assertIn('"source": "codex"', text)


if __name__ == "__main__":
    unittest.main()
