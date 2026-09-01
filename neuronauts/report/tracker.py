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


# ---------------------------------------------------------------------------
# shareable board
# ---------------------------------------------------------------------------

_HTML_HEAD = """<title>Neuronauts Progress</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root{--bg:#F7F7F5;--panel:#FFF;--ink:#16191C;--ink2:#4C555C;--muted:#7C868D;
--rule:#DFE2DE;--accent:#1F6F6B;--accent2:#C25E00;--ok:#2E7D4F;--okbg:#E3F0E7;
--no:#8A9299;--nobg:#EDEFEE;--warn:#B4541C;--warnbg:#F8E6D9;--shadow:0 1px 2px rgba(22,25,28,.05)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--bg:#111416;--panel:#191D20;--ink:#E6E9E7;--ink2:#AEB7B3;--muted:#7E8688;
--rule:#272D30;--accent:#5FB8B2;--accent2:#E08A45;--ok:#63BC8A;--okbg:#152C20;
--no:#8B9399;--nobg:#20262A;--warn:#DE8A4E;--warnbg:#34210F;--shadow:none}}
:root[data-theme="dark"]{--bg:#111416;--panel:#191D20;--ink:#E6E9E7;--ink2:#AEB7B3;
--muted:#7E8688;--rule:#272D30;--accent:#5FB8B2;--accent2:#E08A45;--ok:#63BC8A;
--okbg:#152C20;--no:#8B9399;--nobg:#20262A;--warn:#DE8A4E;--warnbg:#34210F;--shadow:none}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;
font-size:15px;line-height:1.5;font-variant-numeric:tabular-nums}
.wrap{max-width:1060px;margin:0 auto;padding:36px 22px 80px}
h1{font-size:29px;font-weight:700;letter-spacing:-.015em;margin:8px 0 6px;text-wrap:balance}
.sub{color:var(--ink2);margin:0 0 6px;max-width:64ch}
.stamp{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted);margin-top:12px}
h2{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.09em;
color:var(--muted);margin:38px 0 12px;padding-bottom:7px;border-bottom:1px solid var(--rule)}
.bars{display:grid;gap:9px;margin-bottom:6px}
.row{display:grid;grid-template-columns:210px 1fr 74px;gap:14px;align-items:center;
background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:11px 14px;box-shadow:var(--shadow)}
.row .name{font-weight:600;font-size:14px}
.row .name small{display:block;font-weight:400;color:var(--muted);font-size:12px;margin-top:1px}
.track{height:7px;background:var(--nobg);border-radius:4px;overflow:hidden}
.fill{height:100%;background:var(--accent);border-radius:4px}
.frac{font-family:"IBM Plex Mono",monospace;font-size:13px;color:var(--ink2);text-align:right}
.checks{margin:6px 0 20px 0;display:grid;gap:3px}
.chk{display:grid;grid-template-columns:20px 1fr;gap:8px;font-size:13.5px;
padding:3px 14px;color:var(--ink2)}
.chk .m{font-family:"IBM Plex Mono",monospace;font-weight:600}
.chk.on .m{color:var(--ok)} .chk.off .m{color:var(--no)} .chk.unk .m{color:var(--warn)}
.chk .d{color:var(--muted);font-size:12.5px}
table{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--panel);
border:1px solid var(--rule);border-radius:5px;overflow:hidden;box-shadow:var(--shadow)}
th{text-align:left;font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;
color:var(--muted);font-weight:600;padding:9px 12px;border-bottom:1px solid var(--rule)}
td{padding:8px 12px;border-bottom:1px solid var(--rule);vertical-align:top}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:11px;
font-weight:600;padding:2px 7px;border-radius:3px;text-transform:uppercase;letter-spacing:.04em}
.p-passed{background:var(--okbg);color:var(--ok)}
.p-blocked{background:var(--nobg);color:var(--no)}
.p-ready{background:var(--okbg);color:var(--ok);outline:1px solid var(--ok)}
.p-todo{background:var(--warnbg);color:var(--warn)}
.p-failed{background:var(--warnbg);color:var(--warn)}
.crit{color:var(--muted);font-size:12.5px;max-width:44ch}
.why{color:var(--muted);font-size:12px;font-family:"IBM Plex Mono",monospace}
.ser{font-family:"IBM Plex Mono",monospace;color:var(--accent2);font-weight:600}
.id{font-family:"IBM Plex Mono",monospace;font-weight:600;white-space:nowrap}
.foot{margin-top:34px;padding-top:14px;border-top:1px solid var(--rule);
color:var(--muted);font-size:12.5px}
code{font-family:"IBM Plex Mono",monospace;background:var(--nobg);padding:1px 5px;border-radius:3px;font-size:.9em}
@media(max-width:640px){.row{grid-template-columns:1fr;gap:7px}}
</style>
"""


def render_html(root: Optional[Path] = None) -> str:
    """A shareable board, generated from the same derived data as the CLI."""
    import datetime as _dt
    import html as _h

    root = root or repo_root()
    d = as_dict(root)
    phases, exps, s, repo = (d["consolidation"], d["experiments"],
                             d["summary"], d["repo"])
    cdone = sum(p["done"] for p in phases)
    ctot = sum(p["total"] for p in phases)

    out = [_HTML_HEAD, '<div class="wrap">',
           '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
           'letter-spacing:.08em;text-transform:uppercase;color:var(--muted)">'
           'neuronauts</div>',
           "<h1>Progress</h1>",
           '<p class="sub">Every line is read from the repository. A phase is '
           'done when the thing it promised is on disk; an experiment has a '
           'state because its result file says so. Nothing here is asserted by '
           'hand and then trusted.</p>',
           f'<p class="stamp">consolidation {cdone}/{ctot} checks &nbsp;·&nbsp; '
           f'experiments {s["passed"]}/{s["total"]} passed &nbsp;·&nbsp; '
           f'{repo["tests_collected"]} tests, {repo["collection_errors"]} '
           f'collection errors &nbsp;·&nbsp; {repo["broken_links"]} broken links'
           f' &nbsp;·&nbsp; generated '
           f'{_dt.datetime.now().strftime("%Y-%m-%d %H:%M")}</p>',
           "<h2>Consolidation</h2>", '<div class="bars">']

    for p in phases:
        pct = 100 * p["done"] / max(p["total"], 1)
        out.append(
            f'<div class="row"><div class="name">Phase {p["phase"]} · '
            f'{_h.escape(p["title"])}<small>{p["state"]}</small></div>'
            f'<div class="track"><div class="fill" style="width:{pct:.0f}%"></div></div>'
            f'<div class="frac">{p["done"]}/{p["total"]}</div></div>')
        rows = []
        for c in p["checks"]:
            cls, m = ({True: ("on", "&check;"), False: ("off", "·"),
                       None: ("unk", "?")}[c["done"]])
            det = (f' <span class="d">— {_h.escape(c["detail"])}</span>'
                   if c["detail"] else "")
            rows.append(f'<div class="chk {cls}"><span class="m">{m}</span>'
                        f'<span>{_h.escape(c["label"])}{det}</span></div>')
        out.append('<div class="checks">' + "".join(rows) + "</div>")
    out.append("</div>")

    out += ["<h2>Experiment program</h2>",
            "<table><thead><tr><th>ID</th><th>Experiment</th><th>Series</th>"
            "<th>State</th><th>Bar it must clear</th></tr></thead><tbody>"]
    pill = {"passed": "p-passed", "ready": "p-ready", "blocked": "p-blocked",
            "not_implemented": "p-todo", "failed": "p-failed",
            "prerequisite_failed": "p-blocked", "error": "p-failed"}
    for e in exps:
        why = ""
        if e["state"] in ("blocked", "not_implemented") and e["reasons"]:
            why = f'<div class="why">{_h.escape(e["reasons"][0][:96])}</div>'
        label = "todo" if e["state"] == "not_implemented" else e["state"]
        out.append(
            f'<tr><td class="id">{_h.escape(e["id"])}</td>'
            f'<td>{_h.escape(e["title"])}{why}</td>'
            f'<td class="ser">{_h.escape(e["series"])}</td>'
            f'<td><span class="pill {pill.get(e["state"], "p-blocked")}">'
            f'{_h.escape(label)}</span></td>'
            f'<td class="crit">{_h.escape(e["criterion"])}</td></tr>')
    out.append("</tbody></table>")

    out.append('<p class="foot">Regenerate with '
               '<code>uv run python scripts/status.py --html --out FILE</code>. '
               'Series A substrate · B candidates · C cuts · D scoring · '
               'E assembly · F re-derivation; order is a strict dependency '
               'chain, which is why running them out of order is how EXP-053B, '
               '054 and 055 all died on prerequisites.</p>')
    out.append("</div>")
    return "\n".join(out)
