#!/usr/bin/env python3
"""Optional Gemini-driven outer research loop for Neuronauts.

This script is intentionally thin: it edits one target file, runs the current
export/train/eval path, and reports the resulting metrics. It is not used by
the test suite and requires external credentials and dependencies.
"""

from __future__ import annotations

import os
import json
import re
import subprocess
from pathlib import Path

from neuronauts.experiment_driver import append_experiment_ledger, build_ledger_entry


TARGET_GRAMMAR = Path("neuronauts/grammar.py")
TARGET_TOPOLOGY = Path("neuronauts/topology_model.py")
PROGRAM_MD = Path("program.md")
LEDGER_PATH = Path("run_logs/research_ledger.jsonl")


def _extract_code(text: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    return match.group(1) if match else text


def main() -> int:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY before running this script.")

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise SystemExit("Install google-generativeai to use this script.") from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-002")

    prompt = f"""
{PROGRAM_MD.read_text()}

--- CURRENT GRAMMAR ---
{TARGET_GRAMMAR.read_text()}

--- CURRENT TOPOLOGY MODEL ---
{TARGET_TOPOLOGY.read_text()}

Mission:
- improve the shared grammar path while preserving coordinate-free behavior
- prefer sparse or hierarchical ideas over quadratic global comparisons
- optimize for terminal line-graph F1

Return one hypothesis and one full Python file to replace, wrapped in a
```python``` block.
""".strip()

    response = model.generate_content(prompt)
    new_code = _extract_code(response.text)
    target = TARGET_GRAMMAR if "class PathEncoder" in new_code else TARGET_TOPOLOGY
    target.write_text(new_code, encoding="utf-8")
    result = subprocess.run(
        [
            "python3",
            "scripts/run_research_cycle.py",
            "--python-bin",
            ".venv/bin/python",
            "--output-dir",
            "run_logs/gemini_research_cycle",
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print(response.text)
    if result.stdout.strip():
        try:
            summary = json.loads(result.stdout)
            append_experiment_ledger(
                LEDGER_PATH,
                build_ledger_entry(
                    summary,
                    source="gemini",
                    target_file=str(target),
                    hypothesis=response.text.strip().splitlines()[0] if response.text.strip() else "",
                    decision="completed" if summary.get("ok") else "failed",
                    note="gemini_researcher cycle",
                    run_dir="run_logs/gemini_research_cycle",
                ),
            )
            print(json.dumps(summary, indent=2))
        except json.JSONDecodeError:
            print(result.stdout)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
