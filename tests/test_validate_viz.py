"""Smoke tests for scripts/validate_viz.py — ensure CLI runs without crashing."""

import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class ValidateVizSmokeTest(unittest.TestCase):
    """Smoke test that validate_viz CLI runs."""

    def test_help_exits_zero(self):
        """--help should run and exit 0."""
        result = subprocess.run(
            [sys.executable, "scripts/validate_viz.py", "--help"],
            cwd=os.path.join(os.path.dirname(__file__), ".."),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--cache-dir", result.stdout)
        self.assertIn("--grammar-path", result.stdout)
        self.assertIn("--list-boxes", result.stdout)

    def test_list_boxes_with_empty_cache_exits_nonzero_but_no_crash(self):
        """--list-boxes with empty cache dir exits 1 (no boxes) but does not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "scripts/validate_viz.py", "--cache-dir", tmpdir, "--list-boxes"],
                cwd=os.path.join(os.path.dirname(__file__), ".."),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("No boxes", result.stdout)


if __name__ == "__main__":
    unittest.main()
