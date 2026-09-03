"""A committed credential must never reappear.

A CAVE token was hard-coded in ten files of this repository, which is public, so
it reached pushed history. This test is the guard: any 32-hex string literal in
tracked Python fails it. Tokens come from CAVE_TOKEN or
~/.cloudvolume/secrets/cave-secret.json via ``neuronauts.auth.cave_token``.
"""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET_SHAPED = re.compile(r'["\'][0-9a-fA-F]{32,}["\']')


def test_no_hardcoded_secrets():
    files = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                           capture_output=True, text=True).stdout.split()
    offenders = []
    for rel in files:
        if rel == "tests/test_no_hardcoded_secrets.py":
            continue
        p = ROOT / rel
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if SECRET_SHAPED.search(line):
                offenders.append(f"{rel}:{i}")
    assert not offenders, (
        "secret-shaped literals found; use neuronauts.auth.cave_token(): "
        + ", ".join(offenders)
    )
