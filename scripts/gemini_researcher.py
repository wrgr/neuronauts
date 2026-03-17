#!/usr/bin/env python3
"""Optional Gemini-driven outer research loop for Neuronauts.

This script is intentionally thin: it edits one target file, runs the current
export/train/eval path, and reports the resulting metrics. It is not used by
the test suite and requires external credentials and dependencies.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


TARGET_GRAMMAR = Path("neuronauts/grammar.py")
TARGET_TOPOLOGY = Path("neuronauts/topology_model.py")
PROGRAM_MD = Path("program.md")
EXPORT_CMD = ["python3", "scripts/export_topology_dataset.py", "--output", "data/topology_dataset_multi.npz"]
TRAIN_CMD = ["python3", "scripts/train_topology_model.py", "--dataset", "data/topology_dataset_multi.npz"]
EVAL_CMD = ["python3", "-m", "neuronauts.run", "--data-mode", "real", "--quiet"]


def _extract_code(text: str) -> str:
    match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
    return match.group(1) if match else text


def _parse_metrics(output: str) -> dict[str, float]:
    match = re.search(r"F1=([0-9.]+)", output)
    return {"val_f1": float(match.group(1)) if match else 0.0}


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

    subprocess.run(EXPORT_CMD, check=True)
    subprocess.run(TRAIN_CMD, check=True)
    result = subprocess.run(EVAL_CMD, capture_output=True, text=True, check=False)
    print(response.text)
    print(_parse_metrics(result.stdout + result.stderr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
