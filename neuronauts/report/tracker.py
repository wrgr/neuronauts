"""Where the consolidation and the experiment program actually stand.

A hand-maintained checklist goes stale within a day and then quietly lies. This
derives every line from the repository instead: a phase is done when the thing
it promised is on disk, an experiment has a state because its result file says
so, and the deprecation counters are file counts. Nothing here is asserted by a
human and then trusted.

The one number it cannot derive is whether a result is *good*; that stays with
the predeclared criterion in each experiment's spec.

    uv run python scripts/status.py            # terminal
    uv run python scripts/status.py --markdown # for a doc or an issue
    uv run python scripts/status.py --json     # for a machine
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from neuronauts.report.provenance import repo_root


@dataclass
class Check:
    """One derived fact: what was promised, and what is on disk.

    ``done`` may be ``None`` for "not checked" -- reporting a skipped check as
    passing is the failure mode this whole file exists to avoid.
    """

    label: str
    done: Optional[bool]
    detail: str = ""


@dataclass
class Phase:
    key: str
    title: str
    checks: list[Check] = field(default_factory=list)

    @property
    def done(self) -> int:
        return sum(1 for c in self.checks if c.done is True)

    @property
    def unknown(self) -> int:
        return sum(1 for c in self.checks if c.done is None)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def state(self) -> str:
        if self.done == self.total:
            return "done"
        if self.done == 0 and self.unknown == 0:
            return "not started"
        return "partial"


def _exists(root: Path, *rel: str) -> bool:
    """Case-sensitive existence.

    macOS is case-insensitive, so a plain ``Path.exists()`` reported
    ``docs/ARCHITECTURE.md`` as present when only ``docs/architecture.md``
    existed -- a check that passed without the thing being done.
    """
    for r in rel:
        p = root / r
        if not p.exists():
            return False
        try:
            if p.name not in (q.name for q in p.parent.iterdir()):
                return False
        except OSError:
            return False
    return True


def _count(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern)))


def _delegates(root: Path, rel: str) -> bool:
    p = root / rel
    return p.exists() and "neuronauts.metrics" in p.read_text(errors="ignore")


def consolidation_phases(root: Optional[Path] = None) -> list[Phase]:
    """The six phases of docs/consolidation_plan.md §5, checked against disk."""
    root = root or repo_root()

    p0 = Phase("0", "Freeze")
    p0.checks = [
        Check("harness/report/meshing/metrics committed",
              _exists(root, "neuronauts/harness", "neuronauts/report",
                      "neuronauts/meshing", "neuronauts/metrics"),
              "the four packages are on disk"),
        Check("EXPERIMENT_LOG.md marked superseded",
              "SUPERSEDED" in (root / "EXPERIMENT_LOG.md").read_text(errors="ignore")
              if (root / "EXPERIMENT_LOG.md").exists() else False),
        Check("paper drafts carry a provenance notice",
              _exists(root, "docs/paper/README.md")),
        Check("pytest collects with no errors",
              None if _collected(root) < 0 else _collection_errors(root) == 0,
              "not checked (--fast)" if _collected(root) < 0 else
              f"{_collected(root)} tests, {_collection_errors(root)} errors"),
    ]

    p1 = Phase("1", "One metric package")
    shims = ["neuronauts/line_graph.py", "treestitch/partition.py",
             "treestitch/connectivity.py", "treestitch/calibration.py",
             "neuronauts/global_merge/eval/benchmark.py",
             "experiments/pcfg/conn_metric.py"]
    delegating = [s for s in shims if _delegates(root, s)]
    p1.checks = [
        Check("neuronauts/metrics/ exists", _exists(root, "neuronauts/metrics")),
        Check("legacy callers delegate",
              len(delegating) == len(shims),
              f"{len(delegating)}/{len(shims)}: missing "
              f"{', '.join(Path(s).name for s in shims if s not in delegating) or 'none'}"),
        Check("docs/metrics.md written", _exists(root, "docs/metrics.md")),
        Check("agreement test exists (EXP-059)",
              _exists(root, "results/EXP-059/result.json"),
              "run EXP-059 to close this"),
    ]

    p2 = Phase("2", "Attic")
    n_attic_bench = _count(root, "attic/benchmarks_semi_synthetic/*.py")
    n_left = _count(root, "scripts/benchmark_*.py")
    p2.checks = [
        Check("morpho_grammar retired", _exists(root, "attic/morpho_grammar"),
              f"{_count(root, 'attic/morpho_grammar/*.py')} engines"),
        Check("semi-synthetic benchmarks retired", n_attic_bench > 0,
              f"{n_attic_bench} retired, {n_left} kept in scripts/"),
        Check("outer-loop and viz retired",
              _exists(root, "attic/outer_loop_and_viz")),
        Check("import shim keeps old paths working",
              _exists(root, "neuronauts/morpho_grammar/__init__.py")),
        Check("attic excluded from the default test run",
              _count(root, "attic/tests/*.py") > 0
              and "attic" not in _testpaths(root)),
        Check("legacy/ retired", _exists(root, "attic/legacy"),
              "still in neuronauts/legacy/; drags train.py and 13 test files, "
              "so it belongs with Phase 3"),
    ]

    # Each of these asks for positive evidence that the new thing exists, not
    # merely that the old file is gone -- on an empty tree the absence test
    # passes, which is a check that flatters rather than measures.
    p3 = Phase("3", "Fold into stage packages")
    p3.checks = [
        Check("treestitch folded into assemble/",
              _exists(root, "neuronauts/assemble/partition.py")
              and not _exists(root, "treestitch/partition.py"),
              f"treestitch/ still has {_count(root, 'treestitch/*.py')} modules"),
        Check("kimimaro skeleton path extracted",
              _exists(root, "neuronauts/harness/skeleton.py"),
              f"still inside cell_graph.py ({_lines(root, 'neuronauts/cell_graph.py')} lines)"),
        Check("scripts/train.py split into stage CLIs",
              _exists(root, "scripts/neuronauts_cli.py")
              and not _exists(root, "scripts/train.py"),
              f"{_lines(root, 'scripts/train.py')} lines, 17 subcommands"),
    ]

    p4 = Phase("4", "Docs")
    p4.checks = [
        Check("superseded direction docs archived",
              _exists(root, "docs/archive/2026-09")),
        Check("no broken markdown links",
              _broken_links(root) == 0 and _count(root, "docs/*.md") > 0,
              f"{_broken_links(root)} broken across "
              f"{_count(root, 'docs/*.md')} docs/*.md"),
        Check("ARCHITECTURE.md is the single entry point",
              _exists(root, "docs/ARCHITECTURE.md"),
              "docs/architecture.md + roadmap_global_assembly.md not yet merged"),
        Check("RESULTS.md ledger exists", _exists(root, "results/RESULTS.md"),
              "written by the runner on the first registered run"),
    ]

    p5 = Phase("5", "Runner")
    p5.checks = [
        Check("runner implemented",
              _exists(root, "neuronauts/experiments/_runner.py")),
        Check("experiment program registered",
              _exists(root, "neuronauts/experiments/registry.py")),
        Check("EXP-051..056 ported onto the runner",
              _count(root, "results/EXP-05*/result.json") >= 6,
              f"{_count(root, 'results/EXP-*/result.json')} registered results"),
    ]

    return [p0, p1, p2, p3, p4, p5]


# ---------------------------------------------------------------------------
# derived repo facts
# ---------------------------------------------------------------------------

_cache: dict = {}


def _testpaths(root: Path) -> str:
    p = root / "pyproject.toml"
    return p.read_text(errors="ignore") if p.exists() else ""


def _lines(root: Path, rel: str) -> int:
    p = root / rel
    return len(p.read_text(errors="ignore").splitlines()) if p.exists() else 0


def _pytest_collect(root: Path) -> tuple[int, int]:
    """(collected, errors) from a real collection run, cached per process."""
    if "collect" in _cache:
        return _cache["collect"]
    try:
        out = subprocess.run(
            [str(root / ".venv/bin/python"), "-m", "pytest", "-q", "--co",
             "-p", "no:cacheprovider"],
            cwd=root, capture_output=True, text=True, timeout=300).stdout
    except (OSError, subprocess.SubprocessError):
        _cache["collect"] = (0, -1)
        return _cache["collect"]
    n = err = 0
    for line in out.splitlines():
        if "tests collected" in line or "test collected" in line:
            for tok in line.split():
                if tok.isdigit():
                    n = int(tok)
                    break
        if "error" in line.lower() and "collected" in line.lower():
            for tok in line.split():
                if tok.isdigit():
                    err = int(tok)
    err = out.count("ERROR ")
    _cache["collect"] = (n, err)
    return _cache["collect"]


def _collected(root: Path) -> int:
    return _pytest_collect(root)[0]


def _collection_errors(root: Path) -> int:
    return _pytest_collect(root)[1]


def _broken_links(root: Path) -> int:
    if "links" in _cache:
        return _cache["links"]
    import re
    link = re.compile(r"\[[^\]]*\]\(([^)#]+?)(?:#[^)]*)?\)")
    n = 0
    for md in root.rglob("*.md"):
        if any(p in md.parts for p in (".venv", "node_modules", ".git")):
            continue
        for m in link.finditer(md.read_text(errors="ignore")):
            t = m.group(1).strip()
            if t.startswith(("http", "mailto:", "file:")) or not t:
                continue
            if not (md.parent / t).resolve().exists():
                n += 1
    _cache["links"] = n
    return n


def git_log(root: Path, n: int = 12) -> list[str]:
    try:
        return subprocess.run(["git", "log", f"-{n}", "--oneline"], cwd=root,
                              capture_output=True, text=True,
                              timeout=30).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return []


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_BOX = {"done": "[x]", "partial": "[~]", "not started": "[ ]"}


def render_text(root: Optional[Path] = None, *, show_experiments: bool = True) -> str:
    from neuronauts.experiments.registry import status_table

    root = root or repo_root()
    phases = consolidation_phases(root)
    done = sum(p.done for p in phases)
    total = sum(p.total for p in phases)

    out = ["", "=" * 72,
           f" neuronauts status   consolidation {done}/{total} checks", "=" * 72]
    for ph in phases:
        out.append(f"\n {_BOX[ph.state]} Phase {ph.key} - {ph.title}"
                   f"   ({ph.done}/{ph.total})")
        for c in ph.checks:
            mark = {True: "x", False: " ", None: "?"}[c.done]
            line = f"      [{mark}] {c.label}"
            if c.detail:
                line += f"  -- {c.detail}"
            out.append(line)

    if show_experiments:
        out += ["", "-" * 72, ""]
        out.append(status_table(root))

    out += ["", "-" * 72, " recent commits"]
    out += [f"   {c}" for c in git_log(root, 8)]
    out.append("")
    return "\n".join(out)


def render_markdown(root: Optional[Path] = None) -> str:
    from neuronauts.experiments.registry import REGISTRY, state, summary

    root = root or repo_root()
    phases = consolidation_phases(root)
    done = sum(p.done for p in phases)
    total = sum(p.total for p in phases)
    s = summary(root)

    md = [f"# Status\n",
          f"Derived from the repository, not hand-maintained "
          f"(`scripts/status.py`).\n",
          f"**Consolidation:** {done}/{total} checks. "
          f"**Experiments:** {s['passed']}/{s['total']} passed.\n",
          "## Consolidation\n",
          "| Phase | State | Checks | Outstanding |", "|---|---|---:|---|"]
    for ph in phases:
        out = "; ".join(c.label for c in ph.checks if c.done is not True) or "—"
        md.append(f"| {ph.key} · {ph.title} | {ph.state} | "
                  f"{ph.done}/{ph.total} | {out} |")

    md += ["\n## Experiment program\n",
           "| ID | Title | Series | State | Bar |", "|---|---|---|---|---|"]
    for e in REGISTRY:
        st, why = state(e, root)
        md.append(f"| {e.id} | {e.spec.title} | {e.series} | {st} | "
                  f"{e.spec.criterion} |")
    return "\n".join(md) + "\n"


def as_dict(root: Optional[Path] = None) -> dict:
    from neuronauts.experiments.registry import REGISTRY, state, summary

    root = root or repo_root()
    phases = consolidation_phases(root)
    return {
        "consolidation": [
            {"phase": p.key, "title": p.title, "state": p.state,
             "done": p.done, "total": p.total,
             "checks": [{"label": c.label, "done": c.done, "detail": c.detail}
                        for c in p.checks]}
            for p in phases],
        "experiments": [
            {"id": e.id, "title": e.spec.title, "series": e.series,
             "state": state(e, root)[0], "reasons": state(e, root)[1],
             "criterion": e.spec.criterion, "requires": e.spec.requires,
             "est_minutes": e.est_minutes}
            for e in REGISTRY],
        "summary": summary(root),
        "repo": {"tests_collected": _collected(root),
                 "collection_errors": _collection_errors(root),
                 "broken_links": _broken_links(root)},
    }


def write_json(path: str | Path, root: Optional[Path] = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(as_dict(root), indent=2) + "\n")
    return p
