#!/usr/bin/env python3
"""Provenance lint — catches the failure modes behind the synthetic-data incident.

Each rule below exists because a specific, verified defect got through review.
Run with no arguments to lint the repo; exits non-zero on any violation.

    python scripts/lint_provenance.py            # lint everything
    python scripts/lint_provenance.py --paths a.py b.py

Rules
-----
LEAK001  A ground-truth label is passed into a scorer/verifier as an input.
         Origin: cloudvolume_em_sampler.sample_bridge_volume(...,
         is_true_continuation, ...) returned a Gaussian conditioned on the
         answer, producing the entire "Selective Micro-EM" results table.

LEAK002  A prediction is computed by offsetting the ground-truth label.
         Origin: generate_dashboard.py -> `if is_same_cell: p += 0.40`.

SYNTH001 A synthetic generator is called outside quarantine without an opt-in
         flag. Origin: _split_skeleton_n_pieces() manufacturing "v117
         fragments" in 31 benchmark scripts.

SYNTH002 A fetch failure silently falls back to generated data.
         Origin: benchmark_exp049 called generate_dense_subvolume_fallback()
         unconditionally under a docstring claiming real v117 data.

SYNTH003 The identity-encoding synapse-partner idiom.
         Origin: `partner_base = obj_counter * 100` in 31 files.

SEC001   A hardcoded credential.
         Origin: a CAVE bearer token committed in 11 files.

SPLIT001 A random shuffle used to build a train/val/test split.
         Origin: scripts/train.py shuffled box records, putting the same
         neuron in train and val (arbors span many boxes).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Directories exempt from all rules.
EXEMPT_DIRS = {"quarantine", ".git", "build", "dist", "node_modules",
               "__pycache__", ".venv", "venv", ".pytest_cache", "docs"}

# Files allowed to *define* the patterns they describe.
SELF_REFERENTIAL = {
    "scripts/lint_provenance.py",
    "neuronauts/data/auth.py",
    "neuronauts/results_schema.py",
    "treestitch/data.py",       # defines _split_skeleton_n_pieces (warned)
    "treestitch/synthetic.py",  # declared synthetic world generator
    "treestitch/worldbuild.py", # defines frankenmerge_adjacent
    "tests/test_worldbuild.py", # unit tests for the generator itself
    "tests/test_provenance_guardrails.py",  # asserts each rule fires
}

# Encoder-representation ablations: bisection is the stated task, not a
# simulated v117. See quarantine/README.md for the rule.
SYNTH_OPTIN_ALLOWED = {
    "scripts/half_split_ablation.py",
    "scripts/within_type_ablation.py",
    "scripts/multi_fragment_ablation.py",
    "scripts/ablate_dna.py",
    "scripts/compare_partition_methods.py",
    "scripts/optimize_tree_stitch.py",
    "scripts/global_gnn_ablation.py",
    "scripts/half_synapse_ablation.py",
}

LABEL_NAMES = r"(is_true_continuation|is_same_cell|gt_target_id|gt_label|gt_map)"

Rule = tuple[str, re.Pattern, str]

RULES: list[Rule] = [
    ("LEAK001",
     re.compile(rf"\b(score|verify|sample|predict|rerank|infill)\w*\s*\([^)]*\b{LABEL_NAMES}\b",
                re.IGNORECASE | re.DOTALL),
     "ground-truth label passed into a scorer/verifier as an input"),
    # NOTE: `evaluate*` is deliberately excluded — evaluation functions are
    # supposed to receive ground truth. The defect is inference-time access.
    ("LEAK002",
     re.compile(rf"if\s+{LABEL_NAMES}\s*:[^\n]*\n\s*\w+\s*=.*(clip|\+\s*0\.|-\s*0\.)"),
     "prediction computed by offsetting the ground-truth label"),
    ("SYNTH001",
     re.compile(r"_split_skeleton_n_pieces|split_skeleton_n_pieces|frankenmerge_adjacent"),
     "synthetic fragment/corruption generator used outside quarantine"),
    ("SYNTH002",
     re.compile(r"(except\s+(?!(ImportError|ModuleNotFoundError|AttributeError|np\.linalg\.LinAlgError|LinAlgError|KeyboardInterrupt))[^\n]*:\s*\n(?:[^\n]*\n){0,6}?[^\n]*\b\w*(fallback|synthetic|generate_\w*)\w*\s*\()"
                r"|(\bgenerate_\w*_fallback\s*\()"),
     "fetch failure falls back to generated data"),
    ("SYNTH003",
     re.compile(r"partner_base\s*=|obj_counter\s*\*\s*100"),
     "identity-encoding synthetic synapse-partner idiom"),
    ("SEC001",
     re.compile(r"""(?i)(token|api_key|secret|password)\s*[:=]\s*["'][A-Za-z0-9_\-]{24,}["']"""),
     "hardcoded credential"),
    ("SPLIT001",
     re.compile(r"(rng|np\.random|random)\.(shuffle|permutation)\s*\([^)]*"
                r"\w*(record|box|boxes|sample|neuron|frag|obs)\w*"),
     "random shuffle used to build a data split (split by neuron/region instead)"),
]

ALLOW_COMMENT = "provenance-lint: allow"


def iter_files(paths: list[str] | None) -> list[Path]:
    if paths:
        return [Path(p).resolve() for p in paths]
    out: list[Path] = []
    for p in REPO.rglob("*.py"):
        if any(part in EXEMPT_DIRS for part in p.relative_to(REPO).parts):
            continue
        out.append(p)
    return sorted(out)


def _rel(path: Path) -> str:
    """Repo-relative path, or the absolute path for files outside the repo."""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def lint_file(path: Path) -> list[tuple[int, str, str]]:
    rel = _rel(path)
    if rel in SELF_REFERENTIAL:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    lines = text.splitlines()
    violations: list[tuple[int, str, str]] = []

    declares_synthetic_optin = bool(
        re.search(r'add_argument\(\s*["\']--synthetic', text)
    )

    for code, pattern, message in RULES:
        for m in pattern.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            line = lines[lineno - 1] if lineno <= len(lines) else ""

            # Inline waiver, e.g.  # provenance-lint: allow SYNTH001 - reason
            # Look back a few lines so a waiver whose reason wraps onto a
            # second comment line still applies.
            window = "\n".join(lines[max(0, lineno - 4):lineno])
            if ALLOW_COMMENT in window and code in window:
                continue
            # Skip comment/docstring-only mentions.
            if line.lstrip().startswith("#"):
                continue
            if code == "SYNTH001":
                if rel in SYNTH_OPTIN_ALLOWED and declares_synthetic_optin:
                    continue
                if rel in SYNTH_OPTIN_ALLOWED:
                    message = ("synthetic generator used without an explicit "
                               "--synthetic opt-in flag")
            violations.append((lineno, code, f"{message}\n      {line.strip()[:120]}"))
    return violations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", nargs="*", default=None,
                    help="specific files to lint (default: whole repo)")
    args = ap.parse_args()

    total = 0
    for f in iter_files(args.paths):
        for lineno, code, message in lint_file(f):
            print(f"{_rel(f)}:{lineno}: {code}: {message}")
            total += 1

    if total:
        print(f"\n{total} provenance violation(s).")
        print("Each rule maps to a verified past defect; see the module "
              "docstring. To waive one, add a line comment:")
        print(f"    # {ALLOW_COMMENT} <CODE> - <reason>")
        return 1
    print("provenance lint: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
