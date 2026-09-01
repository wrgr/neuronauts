"""Discover result files and normalise them into ``ExperimentRecord``.

Every experiment JSON in ``results/`` has its own shape, but they share a few
habits: an ``experiment`` title, a ``provenance`` block, a ``population``
block, a gate (``success_criterion`` / ``prerequisite_gate`` / ``status``),
and one or more *sweep-like* dicts -- a dict of dicts where the inner dicts
share numeric keys (``sweep``, ``grid``, ``checkpoints``). This module reads
any such file, without a per-experiment schema, and returns:

  status       passed / failed / prerequisite_failed / completed
  provenance   the recorded block, plus a completeness grade and, when the
               local repo has the commit, its date and subject
  population   scalar block as recorded
  gate         required-vs-observed rows from the gate block
  headline     every remaining scalar, dotted by path
  tables       every sweep-like dict (and nested ones), rows in natural order

Nothing here interprets metrics. A table of ``pair_recall`` values is a table;
whether 0.94 is good is the evaluation note's job, not this module's.
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from neuronauts.report.provenance import completeness, git_commit_info, repo_root

_EXP_RE = re.compile(r"^exp(\d{3})([a-z]?)_", re.I)
_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")
_LABEL_KEYS = ("k", "tier", "name", "rule", "root", "id", "cell", "version")
_HARNESS_SCRIPTS = {
    "atom_topology": "scripts/build_atom_topology.py",
    "atom_geometry_tiers": "scripts/fetch_atom_geometry.py",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def natural_key(s: Any) -> tuple:
    """Sort key that orders ``absolute_0.25um < absolute_0.5um < absolute_10um``."""
    out = []
    for tok in _NUM_RE.split(str(s)):
        if not tok:
            continue
        try:
            out.append((0, float(tok)))
        except ValueError:
            out.append((1, tok.lower()))
    return tuple(out)


def is_scalar(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


def is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Recursive dotted flatten of the scalar leaves of a dict."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif is_scalar(v):
            out[key] = v
        elif isinstance(v, list) and len(v) <= 8 and all(is_scalar(x) for x in v):
            out[key] = json.dumps(v)
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

@dataclass
class Table:
    """A sweep: named rows sharing (mostly) the same scalar columns.

    ``parent``/``parent_row``/``leaf`` are set only on a nested table --
    ``checkpoints.<name>.threshold_sweep`` becomes
    ``parent="checkpoints"``, ``parent_row="<name>"``, ``leaf="threshold_sweep"``.
    They are kept as fields rather than parsed back out of ``name`` because
    row names contain dots (``shared_grammar_real.pt``).
    """

    name: str
    rows: list[str]
    columns: list[str]
    cells: dict[str, dict[str, Any]]
    parent: Optional[str] = None
    parent_row: Optional[str] = None
    leaf: Optional[str] = None

    def value(self, row: str, col: str, default: Any = None) -> Any:
        return self.cells.get(row, {}).get(col, default)

    def column(self, col: str) -> list[Any]:
        return [self.cells[r].get(col) for r in self.rows]

    def numeric_columns(self) -> list[str]:
        half = max(1, len(self.rows) // 2)
        return [c for c in self.columns
                if sum(is_number(self.cells[r].get(c)) for r in self.rows) >= half]

    def to_markdown(self, columns: Optional[list[str]] = None,
                    max_rows: Optional[int] = None) -> str:
        cols = columns or self.columns
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        head = "| " + " | ".join([self.name.split(".")[-1], *cols]) + " |"
        sep = "|---|" + "|".join("---:" for _ in cols) + "|"
        body = ["| " + " | ".join([f"`{r}`", *(fmt(self.cells[r].get(c)) for c in cols)]) + " |"
                for r in rows]
        note = "" if max_rows is None or len(self.rows) <= max_rows else \
            f"\n\n*{len(self.rows) - max_rows} more rows in the source JSON.*"
        return "\n".join([head, sep, *body]) + note


def fmt(v: Any, max_len: int = 60) -> str:
    """Compact scalar formatting for Markdown cells."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v == 0:
            return "0"
        if abs(v) >= 1000:
            return f"{v:,.1f}"
        if abs(v) >= 1:
            return f"{v:.4g}"
        return f"{v:.4f}" if abs(v) >= 1e-4 else f"{v:.2e}"
    s = str(v)
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


def dict_table(name: str, d: Any) -> Optional[tuple[Table, list[Table]]]:
    """A dict of dicts sharing numeric keys, plus any sub-tables nested in it."""
    if not isinstance(d, dict) or len(d) < 2:
        return None
    if not all(isinstance(v, dict) for v in d.values()):
        return None
    subtables: list[Table] = []
    cells: dict[str, dict[str, Any]] = {}
    for row, rd in d.items():
        flat: dict[str, Any] = {}
        for k, v in rd.items():
            sub = dict_table(f"{name}.{row}.{k}", v)
            if sub is not None:
                sub[0].parent, sub[0].parent_row, sub[0].leaf = name, str(row), k
                subtables.append(sub[0])
                subtables.extend(sub[1])
                continue
            if isinstance(v, dict):
                flat.update(flatten(v, k + "."))
            elif is_scalar(v):
                flat[k] = v
            elif isinstance(v, list) and len(v) <= 8 and all(is_scalar(x) for x in v):
                flat[k] = json.dumps(v)
        cells[str(row)] = flat
    columns: list[str] = []
    for flat in cells.values():
        columns.extend(c for c in flat if c not in columns)
    table = Table(name, sorted(cells, key=natural_key), columns, cells)
    if not table.numeric_columns():
        return None
    return table, subtables


def list_table(name: str, lst: Any) -> Optional[tuple[Table, list[Table]]]:
    """A list of dicts, labelled by the first id-like key each row carries."""
    if not isinstance(lst, list) or not lst or not all(isinstance(x, dict) for x in lst):
        return None
    cells: dict[str, dict[str, Any]] = {}
    for i, item in enumerate(lst):
        label = next((str(item[k]) for k in _LABEL_KEYS
                      if k in item and is_scalar(item[k])), str(i))
        cells[label] = flatten(item)
    columns: list[str] = []
    for flat in cells.values():
        columns.extend(c for c in flat if c not in columns)
    table = Table(name, list(cells), columns, cells)
    if not table.numeric_columns():
        return None
    return table, []


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    id: str
    title: str
    family: str                      # benchmark | harness | probe
    source: Path
    raw: dict
    status: str = "unknown"
    status_reason: str = ""
    provenance: dict = field(default_factory=dict)
    provenance_grade: dict = field(default_factory=dict)
    commit_info: Optional[dict] = None
    population: dict = field(default_factory=dict)
    gate: list[dict] = field(default_factory=list)
    headline: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    percentiles: dict[str, dict[float, float]] = field(default_factory=dict)
    tables: list[Table] = field(default_factory=list)
    elapsed_seconds: Optional[float] = None
    source_mtime_utc: str = ""
    evaluation_md: Optional[Path] = None
    script: Optional[Path] = None
    docstring: str = ""
    dependencies: list[str] = field(default_factory=list)

    @property
    def top_tables(self) -> list[Table]:
        """Tables that are direct children of the JSON root (no nesting)."""
        return [t for t in self.tables if "." not in t.name]

    def table(self, name: str) -> Optional[Table]:
        return next((t for t in self.tables if t.name == name), None)

    def percentile_series(self, min_points: int = 4) -> dict[str, dict[float, float]]:
        """Blocks keyed entirely by numbers, e.g. ``leaf_len_nm_pct``."""
        return {k: v for k, v in self.percentiles.items() if len(v) >= min_points}


def _percentile_block(flat: dict) -> Optional[dict[float, float]]:
    """``{"10": 383.5, "50": 1477.8, ...}`` -> ``{10.0: 383.5, ...}``, else None."""
    pts: dict[float, float] = {}
    for k, v in flat.items():
        if not is_number(v):
            return None
        try:
            pts[float(k)] = float(v)
        except ValueError:
            return None
    return dict(sorted(pts.items())) if len(pts) >= 4 else None


def _sibling_tables(scalar_dicts: dict[str, dict], skip: set[str] = frozenset(),
                    min_keys: int = 3, min_overlap: float = 0.5) -> list[Table]:
    numeric = {k: {c for c, v in d.items() if is_number(v)}
               for k, d in scalar_dicts.items() if k not in skip}
    numeric = {k: s for k, s in numeric.items() if len(s) >= min_keys}
    tables: list[Table] = []
    remaining = list(numeric)
    while remaining:
        seed = remaining.pop(0)
        group = [seed]
        for other in list(remaining):
            shared = len(numeric[seed] & numeric[other])
            if shared / max(len(numeric[seed]), 1) >= min_overlap:
                group.append(other)
                remaining.remove(other)
        if len(group) < 2:
            continue
        cells = {k: dict(scalar_dicts[k]) for k in group}
        columns: list[str] = []
        for flat in cells.values():
            columns.extend(c for c in flat if c not in columns)
        tables.append(Table("summary" if not tables else f"summary_{len(tables) + 1}",
                            group, columns, cells))
    return tables


def _identify(path: Path) -> tuple[str, str]:
    m = _EXP_RE.match(path.stem)
    if m:
        return f"EXP-{m.group(1)}{m.group(2).upper()}", "benchmark"
    if path.stem.startswith("probe_"):
        return path.stem, "probe"
    return path.stem, "harness"


def _status(raw: dict) -> tuple[str, str]:
    if isinstance(raw.get("status"), str):
        return raw["status"], "explicit `status` field"
    for key in ("success_criterion", "prerequisite_gate"):
        g = raw.get(key)
        if isinstance(g, dict) and "passed" in g:
            return ("passed" if g["passed"] else "failed"), f"`{key}.passed` = {g['passed']}"
    return "completed", "no gate recorded; the run finished and wrote its result"


def _gate_rows(raw: dict) -> list[dict]:
    rows: list[dict] = []
    for key in ("success_criterion", "prerequisite_gate"):
        g = raw.get(key)
        if not isinstance(g, dict):
            continue
        for k, v in g.items():
            if k == "passed" or not is_scalar(v):
                continue
            if k.startswith("required_"):
                rows.append({"gate": key, "requirement": k[9:], "required": v,
                             "observed": g.get("observed_" + k[9:])})
            elif not k.startswith("observed_"):
                rows.append({"gate": key, "requirement": k, "required": v,
                             "observed": None})
        for k, v in g.items():
            if k.startswith("observed_") and "required_" + k[9:] not in g:
                rows.append({"gate": key, "requirement": k[9:], "required": None,
                             "observed": v})
        if isinstance(g.get("failures"), list):
            for f in g["failures"]:
                rows.append({"gate": key, "requirement": "failure", "required": None,
                             "observed": f})
    return rows


def _find_script(root: Path, rec_id: str, stem: str) -> Optional[Path]:
    if rec_id.startswith("EXP-"):
        tag = rec_id[4:].lower()
        hits = sorted((root / "scripts").glob(f"benchmark_exp{tag}*.py"))
        return hits[0] if hits else None
    for prefix, script in _HARNESS_SCRIPTS.items():
        if stem.startswith(prefix):
            p = root / script
            return p if p.exists() else None
    p = root / "scripts" / f"{stem}.py"
    return p if p.exists() else None


def _docstring(script: Optional[Path]) -> str:
    if script is None:
        return ""
    try:
        return ast.get_docstring(ast.parse(script.read_text())) or ""
    except (OSError, SyntaxError):
        return ""


def _dependencies(raw: dict, prov: dict) -> list[str]:
    out: list[str] = []
    if isinstance(prov.get("input"), str):
        out.append(prov["input"])
    for item in prov.get("inputs") or []:
        path = item.get("path") if isinstance(item, dict) else item
        if isinstance(path, str):
            out.append(path)
    for key in ("population", "input"):
        if isinstance(raw.get(key), str):
            out.append(raw[key])
    return list(dict.fromkeys(out))


def load_record(path: str | Path, root: Optional[Path] = None) -> ExperimentRecord:
    path = Path(path)
    root = root or repo_root()
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        raw = {"rows": raw}
    rec_id, family = _identify(path)
    prov = raw.get("provenance") if isinstance(raw.get("provenance"), dict) else {}
    population = raw.get("population") if isinstance(raw.get("population"), dict) else {}
    rec = ExperimentRecord(
        id=rec_id, family=family, source=path, raw=raw,
        title=str(raw.get("experiment") or path.stem.replace("_", " ")),
        provenance=prov, provenance_grade=completeness(prov),
        commit_info=git_commit_info(prov.get("git_commit", ""), root),
        population=flatten(population), gate=_gate_rows(raw),
        elapsed_seconds=raw.get("elapsed_seconds"),
        source_mtime_utc=_dt.datetime.fromtimestamp(
            path.stat().st_mtime, _dt.timezone.utc).isoformat(timespec="seconds"),
        dependencies=_dependencies(raw, prov),
    )
    rec.status, rec.status_reason = _status(raw)

    scalar_dicts: dict[str, dict] = {}
    for key, val in raw.items():
        if key in ("provenance", "population", "success_criterion",
                   "prerequisite_gate", "elapsed_seconds", "experiment"):
            continue
        hit = dict_table(key, val) or list_table(key, val)
        if hit is not None:
            rec.tables.append(hit[0])
            rec.tables.extend(hit[1])
        elif isinstance(val, dict):
            scalar_dicts[key] = flatten(val)
        elif isinstance(val, str) and len(val) > 80:
            rec.notes[key] = val
        elif is_scalar(val):
            rec.headline[key] = val
        elif isinstance(val, list) and len(val) <= 8 and all(is_scalar(x) for x in val):
            rec.headline[key] = json.dumps(val)

    # A block keyed entirely by numbers is a distribution, not a sweep: keep it
    # for the percentile curve and out of the sibling grouping below.
    for key, flat in scalar_dicts.items():
        pts = _percentile_block(flat)
        if pts is not None:
            rec.percentiles[key] = pts

    # Sibling scalar blocks that share numeric keys -- a baseline next to a
    # chosen rule, say -- read better as rows of one table than as a wall of
    # dotted keys. Anything left over is a headline value.
    for table in _sibling_tables(scalar_dicts, skip=set(rec.percentiles)):
        rec.tables.append(table)
        for row in table.rows:
            scalar_dicts.pop(row, None)
    for key, flat in scalar_dicts.items():
        rec.headline.update({f"{key}.{k}": v for k, v in flat.items()})

    stem = path.stem
    eval_md = path.parent / f"{stem.split('_')[0]}_evaluation.md"
    rec.evaluation_md = eval_md if eval_md.exists() and family == "benchmark" else None
    rec.script = _find_script(root, rec_id, stem)
    rec.docstring = _docstring(rec.script)
    return rec


def discover(results_dir: str | Path = "results",
             root: Optional[Path] = None) -> list[ExperimentRecord]:
    """Every result JSON directly under ``results_dir``, benchmarks first."""
    root = root or repo_root()
    rdir = Path(results_dir)
    if not rdir.is_absolute():
        rdir = root / rdir
    order = {"benchmark": 0, "harness": 1, "probe": 2}
    records = [load_record(p, root) for p in sorted(rdir.glob("*.json"))]
    records.sort(key=lambda r: (order.get(r.family, 9), natural_key(r.id)))
    return records
