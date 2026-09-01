"""Markdown rendering: one report per experiment plus an index.

A report answers, in order, the questions a reader brings to a result file:
what is this, did it pass, exactly what produced it (and can it be re-run),
what was the population, what are the numbers, what do they look like, and
where can I look at the underlying data in Neuroglancer. The evaluation note
the experimenter wrote is linked and excerpted, never replaced -- the report
is generated and the note carries the interpretation.

Links are written relative to the report directory so they resolve on GitHub
and in an editor alike.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional

from neuronauts.report.figures import FigureSpec
from neuronauts.report.registry import ExperimentRecord, fmt

GITHUB_COMMIT = "https://github.com/wrgr/neuronauts/commit/{sha}"
_STATUS_BADGE = {
    "passed": "PASSED",
    "failed": "FAILED (gate not met)",
    "prerequisite_failed": "PREREQUISITE FAILED (did not run)",
    "completed": "COMPLETED (no gate)",
}


def _rel(target: Path, start: Path) -> str:
    return os.path.relpath(Path(target).resolve(), Path(start).resolve()).replace(os.sep, "/")


def _kv_table(items: Iterable[tuple[str, object]], key_head: str = "field",
              val_head: str = "value", max_len: int = 60) -> str:
    rows = [f"| `{k}` | {fmt(v, max_len)} |" for k, v in items]
    if not rows:
        return "*none recorded*"
    return "\n".join([f"| {key_head} | {val_head} |", "|---|---|", *rows])


def _excerpt(md_path: Path, max_lines: int = 30) -> str:
    """The body under the first ``##`` heading of an evaluation note."""
    lines = md_path.read_text().splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("## ")), None)
    if start is None:
        return "\n".join(lines[:max_lines])
    body = []
    for line in lines[start + 1:]:
        if line.startswith("## ") or line.startswith("```"):
            break
        body.append(line)
    body = [l for l in body]
    while body and not body[0].strip():
        body.pop(0)
    return "\n".join(body[:max_lines]).rstrip()


def _provenance_section(rec: ExperimentRecord) -> str:
    prov = rec.provenance
    grade = rec.provenance_grade
    parts = []
    sha = prov.get("git_commit")
    if sha:
        line = f"**Commit** [`{sha[:12]}`]({GITHUB_COMMIT.format(sha=sha)})"
        if rec.commit_info:
            line += f" — {rec.commit_info['date'][:10]} · {rec.commit_info['subject']}"
        else:
            line += " — *not present in the local repository*"
        parts.append(line)
    else:
        parts.append("**Commit** *not recorded*")
    if "git_dirty" in prov:
        parts.append(f"**Tree at run time** {'dirty' if prov['git_dirty'] else 'clean'}"
                     + (f" ({len(prov.get('git_modified_files', []))} modified tracked files)"
                        if prov.get("git_dirty") else ""))
    parts.append(f"**Result file written** {rec.source_mtime_utc} (file mtime)")
    score = grade.get("score", 0)
    missing = ", ".join(f"`{m}`" for m in grade.get("missing", []))
    parts.append(f"**Provenance completeness** {score:.0%}"
                 + (f" — missing {missing}" if missing else " — complete"))
    flags = [(k, v) for k, v in prov.items()
             if isinstance(v, bool) and k not in ("git_dirty",)]
    other = [(k, v) for k, v in prov.items()
             if k not in ("git_commit", "git_dirty", "git_modified_files",
                          "git_untracked_count", "inputs", "packages", "argv",
                          "params") and not isinstance(v, bool)]
    out = "\n".join(f"- {p}" for p in parts)
    if flags:
        out += "\n\n**Honesty flags** (recorded by the script itself)\n\n" + \
            _kv_table(flags, "flag", "value")
    if other:
        out += "\n\n**Recorded run parameters**\n\n" + _kv_table(other, max_len=160)
    inputs = prov.get("inputs") or []
    hashed = [i for i in inputs if isinstance(i, dict) and i.get("hash")]
    if hashed:
        out += "\n\n**Input files**\n\n| path | algo | hash | bytes |\n|---|---|---|---:|\n" + \
            "\n".join(f"| `{i['path']}` | {i['algo']} | `{i['hash'][:16]}…` | {i['bytes']:,} |"
                      for i in hashed)
    if prov.get("packages"):
        out += "\n\n**Environment** " + ", ".join(
            f"{k} {v}" for k, v in prov["packages"].items() if v)
    return out


def _gate_section(rec: ExperimentRecord) -> str:
    if not rec.gate:
        return "*No gate recorded.*"
    rows = ["| gate | requirement | required | observed |", "|---|---|---:|---:|"]
    for g in rec.gate:
        rows.append(f"| {g['gate']} | {g['requirement']} | {fmt(g['required'])} | "
                    f"{fmt(g['observed'])} |")
    return "\n".join(rows)


def render_experiment(rec: ExperimentRecord, figures: list[FigureSpec],
                      ngl_views: dict[str, dict], out_dir: str | Path,
                      root: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{rec.id}.md"
    badge = _STATUS_BADGE.get(rec.status, rec.status.upper())

    md: list[str] = [f"# {rec.id} — {rec.title}", ""]
    md.append(f"**Status: {badge}** — {rec.status_reason}")
    md.append("")
    md.append(f"- **Result file** [`{_rel(rec.source, root)}`]({_rel(rec.source, out_dir)})")
    if rec.script:
        md.append(f"- **Script** [`{_rel(rec.script, root)}`]({_rel(rec.script, out_dir)})")
    if rec.evaluation_md:
        md.append(f"- **Evaluation note** [`{_rel(rec.evaluation_md, root)}`]"
                  f"({_rel(rec.evaluation_md, out_dir)})")
    if rec.dependencies:
        md.append("- **Consumes** " + ", ".join(f"`{d}`" for d in rec.dependencies))
    if rec.elapsed_seconds is not None:
        md.append(f"- **Elapsed** {rec.elapsed_seconds / 60:.1f} min")
    md.append("")

    md += ["## What this experiment does", ""]
    if rec.docstring:
        md.append(rec.docstring.strip())
    else:
        md.append("*The script has no module docstring.*")
    md.append("")
    if rec.evaluation_md:
        md += ["### From the evaluation note", "",
               _excerpt(rec.evaluation_md), "",
               f"*Full note: [`{rec.evaluation_md.name}`]({_rel(rec.evaluation_md, out_dir)})*", ""]
    for key, text in rec.notes.items():
        md += [f"### {key}", "", text.strip(), ""]

    md += ["## Provenance", "", _provenance_section(rec), ""]
    md += ["## Gate", "", _gate_section(rec), ""]
    md += ["## Population", "", _kv_table(rec.population.items()), ""]
    if rec.headline:
        md += ["## Headline values", "",
               _kv_table(sorted(rec.headline.items())), ""]

    if rec.tables:
        md += ["## Tables", ""]
        for table in rec.tables:
            cols = table.columns
            md += [f"### `{table.name}`", "",
                   f"{len(table.rows)} rows × {len(cols)} columns.", "",
                   table.to_markdown(max_rows=40), ""]

    md += ["## Figures", ""]
    if figures:
        for spec in figures:
            md += [f"![{spec.caption}]({_rel(spec.path, out_dir)})", "",
                   f"{spec.caption} **Check:** {spec.check}", ""]
    else:
        md += ["*No sweep-like table or percentile series to plot.*", ""]

    md += ["## Neuroglancer", ""]
    if ngl_views:
        for name, view in ngl_views.items():
            line = f"- **{name}** — [state JSON]({_rel(Path(view['json']), out_dir)})"
            if view.get("url"):
                line += f" · [open in Neuroglancer]({view['url']})"
            if view.get("note"):
                line += f" — {view['note']}"
            md.append(line)
        md.append("")
        md.append("Load a state JSON by pasting it into the viewer's `{}` (edit JSON "
                  "state) panel, or serve it locally with "
                  "`scripts/ngl_view.py state <file> --serve`.")
    else:
        md.append("*No spatial provenance (bounding box, anchor, or atom id) recorded, "
                  "so no view was built.*")
    md.append("")

    md += ["## Reproduce", ""]
    if rec.script:
        md += ["```bash", f"uv run --extra cave python {_rel(rec.script, root)}", "```", ""]
    md += ["Regenerate this report: `uv run python scripts/build_reports.py"
           f" --only {rec.id}`", ""]
    path.write_text("\n".join(md))
    return path


def _mermaid(records: list[ExperimentRecord], root: Path) -> str:
    by_source = {_rel(r.source, root): r.id for r in records}
    lines = ["```mermaid", "graph LR"]
    edges: set[tuple[str, str]] = set()
    externals: dict[str, str] = {}
    for r in records:
        node = r.id.replace("-", "_").replace("/", "_")
        label = f'{r.id}<br/>{_STATUS_BADGE.get(r.status, r.status).split(" (")[0]}'
        lines.append(f'  {node}["{label}"]')
        for dep in r.dependencies:
            src = by_source.get(dep)
            if src:
                edges.add((src.replace("-", "_").replace("/", "_"), node))
            else:
                ext = re.sub(r"[^A-Za-z0-9]+", "_", dep)
                externals[ext] = dep
                edges.add((ext, node))
    for ext, dep in externals.items():
        lines.append(f'  {ext}(["{dep}"])')
    lines += [f"  {a} --> {b}" for a, b in sorted(edges)]
    lines.append("```")
    return "\n".join(lines)


def render_index(records: list[ExperimentRecord], report_paths: dict[str, Path],
                 out_dir: str | Path, root: Path) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / "README.md"
    md = ["# Experiment reports", "",
          "Generated by `scripts/build_reports.py` from every JSON under `results/`. "
          "Each row links to a report with provenance, gate, tables, figures and "
          "Neuroglancer views. Regenerate after any experiment writes a result.", "",
          "| id | title | status | commit | commit date | elapsed | provenance | report |",
          "|---|---|---|---|---|---:|---:|---|"]
    for r in records:
        sha = r.provenance.get("git_commit") or ""
        commit = f"[`{sha[:8]}`]({GITHUB_COMMIT.format(sha=sha)})" if sha else "—"
        date = r.commit_info["date"][:10] if r.commit_info else "—"
        elapsed = f"{r.elapsed_seconds / 60:.1f} min" if r.elapsed_seconds else "—"
        md.append(f"| {r.id} | {r.title} | {_STATUS_BADGE.get(r.status, r.status)} | "
                  f"{commit} | {date} | {elapsed} | {r.provenance_grade.get('score', 0):.0%} | "
                  f"[report]({_rel(report_paths[r.id], out_dir)}) |")
    md += ["", "## How results feed each other", "",
           "Edges come from the `provenance.input(s)` and `population` paths each "
           "result recorded. A failed prerequisite propagates: nothing downstream "
           "of a failed gate is a measurement.", "", _mermaid(records, root), ""]
    md += ["## Provenance grading", "",
           "A result is reproducible when it records the commit, whether the tree "
           "was dirty, a timestamp, hashed inputs, the command line, package versions "
           "and its synthetic-fallback flag. Older results recorded a commit and honesty "
           "flags only, so they grade partially; new scripts should call "
           "`neuronauts.report.write_result` to record the rest.", ""]
    path.write_text("\n".join(md))
    return path
