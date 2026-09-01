"""Run an experiment so that its result can be trusted without reading it.

EXP-051 through EXP-056 established a discipline by hand: declare the bar
before the run, refuse to report metrics when a prerequisite is missing, and
say plainly that a run failed rather than quietly reporting whatever came out.
Three of those five experiments correctly refused to produce numbers. That is
the behaviour worth keeping, so this module makes it the runner's job instead
of each author's.

An experiment is a module with a :class:`Spec` and a ``run`` function::

    SPEC = Spec(
        id="EXP-060",
        title="endpoint filter",
        question="Which endpoints are real split sites rather than spines?",
        criterion="pair recall >= 0.90 at median panel size <= 20",
        requires=["EXP-057"],
        inputs=["data/substrate/topology/k10.npz", "results/atom_labels_v1822.json"],
        flags={"synthetic_fallback": False,
               "labels_used_only_for_evaluation": True},
    )

    def run(ctx):
        ...
        return Outcome(passed=..., observed={...}, tables={...},
                       population={...})

The runner then, in order:

1. **Checks prerequisites.** If a required experiment has no result, or its
   result did not pass, this one does not run. It writes a
   ``prerequisite_failed`` result naming what was missing -- the EXP-054 and
   EXP-055 behaviour -- so a failure is a recorded fact rather than an absence.
2. **Checks declared inputs exist** before any work starts, so a two-hour run
   does not die at the end on a missing file.
3. **Runs**, capturing wall-clock and any exception. An exception is recorded
   as ``status="error"`` with the traceback, never swallowed.
4. **Writes** ``results/<id>/result.json`` with full provenance via
   :mod:`neuronauts.report.provenance`, so ``scripts/build_reports.py`` renders
   it and ``completeness`` grades it.
5. **Appends one ledger row** to ``results/RESULTS.md``. A run without a row
   does not exist.

The runner never decides whether a number is good. ``Spec.criterion`` is a
sentence the author wrote before seeing the data, and ``Outcome.passed`` is the
author's own verdict against it; the runner only records both and refuses to
let a failed prerequisite masquerade as a result.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from neuronauts.report.provenance import repo_root, write_result

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_PREREQ = "prerequisite_failed"
STATUS_ERROR = "error"

#: Statuses that let a downstream experiment run.
_SATISFIES = {STATUS_PASSED}


@dataclass
class Spec:
    """What an experiment declares *before* it is run."""

    id: str
    title: str
    question: str
    criterion: str
    requires: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    budget_minutes: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.criterion.strip():
            raise ValueError(
                f"{self.id}: criterion must be declared before the run; an "
                "experiment with no bar cannot fail, so it cannot inform.")


@dataclass
class Outcome:
    """What the experiment observed, and its own verdict on the criterion."""

    passed: bool
    observed: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    population: dict = field(default_factory=dict)
    note: str = ""


@dataclass
class Context:
    """Handed to ``run``: the spec, the result directory, and prior results."""

    spec: Spec
    out_dir: Path
    root: Path
    upstream: dict[str, dict] = field(default_factory=dict)


def result_path(exp_id: str, root: Optional[Path] = None) -> Path:
    return (root or repo_root()) / "results" / exp_id / "result.json"


def load_result(exp_id: str, root: Optional[Path] = None) -> Optional[dict]:
    p = result_path(exp_id, root)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def check_prerequisites(spec: Spec, root: Path) -> list[str]:
    """Reasons this experiment must not run. Empty means it may."""
    problems: list[str] = []
    for dep in spec.requires:
        res = load_result(dep, root)
        if res is None:
            problems.append(f"{dep}: no result at {result_path(dep, root)}")
        elif res.get("status") not in _SATISFIES:
            problems.append(f"{dep}: status={res.get('status')!r}, needs "
                            f"one of {sorted(_SATISFIES)}")
    return problems


def check_inputs(spec: Spec, root: Path) -> list[str]:
    """Declared inputs that are missing, checked before any work starts."""
    return [str(p) for p in spec.inputs if not (root / p).exists()
            and not Path(p).exists()]


def _ledger_row(payload: dict) -> str:
    s = payload["status"]
    mark = {STATUS_PASSED: "pass", STATUS_FAILED: "fail",
            STATUS_PREREQ: "blocked", STATUS_ERROR: "error"}.get(s, s)
    obs = payload.get("observed") or {}
    headline = ", ".join(
        f"{k}={v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool)
        else f"{k}={v}" for k, v in list(obs.items())[:3]) or "--"
    prov = payload.get("provenance") or {}
    commit = (prov.get("commit") or prov.get("git_commit") or "")[:9] or "?"
    dirty = " (dirty)" if prov.get("dirty") else ""
    return (f"| {payload['id']} | {payload['title']} | {mark} | {headline} | "
            f"{payload.get('elapsed_min', 0):.1f} | `{commit}`{dirty} | "
            f"{prov.get('timestamp_utc', '')} |")


_LEDGER_HEADER = """# Results ledger

One row per experiment run, appended by `neuronauts.experiments._runner`.
**A run without a row does not exist.** `blocked` means a prerequisite had no
passing result, so no metrics were computed -- that is a recorded outcome, not
a gap. Full reports: [`results/reports/`](reports/).

| ID | Title | Status | Headline | Min | Commit | When (UTC) |
|---|---|---|---|---:|---|---|
"""


def append_ledger(payload: dict, root: Path) -> Path:
    led = root / "results" / "RESULTS.md"
    led.parent.mkdir(parents=True, exist_ok=True)
    if not led.exists():
        led.write_text(_LEDGER_HEADER)
    with led.open("a") as fh:
        fh.write(_ledger_row(payload) + "\n")
    return led


def run_experiment(spec: Spec, run: Callable[[Context], Outcome], *,
                   root: Optional[Path] = None, verbose: bool = True) -> dict:
    """Execute one experiment under the discipline described in the module docstring."""
    root = root or repo_root()
    out_dir = root / "results" / spec.id
    payload: dict[str, Any] = {
        "id": spec.id, "title": spec.title, "question": spec.question,
        "success_criterion": spec.criterion, "requires": list(spec.requires),
        "elapsed_min": 0.0,
    }

    def finish(status: str, **extra) -> dict:
        payload["status"] = status
        payload.update(extra)
        write_result(out_dir / "result.json", payload,
                     inputs=spec.inputs, params=spec.params,
                     quick_hash=True, **spec.flags)
        append_ledger(payload, root)
        if verbose:
            print(f"[{spec.id}] {status.upper()}"
                  + (f" -- {extra.get('note')}" if extra.get("note") else ""),
                  flush=True)
        return payload

    problems = check_prerequisites(spec, root)
    if problems:
        return finish(STATUS_PREREQ, prerequisite_gate={"unmet": problems},
                      note="; ".join(problems))

    missing = check_inputs(spec, root)
    if missing:
        return finish(STATUS_PREREQ,
                      prerequisite_gate={"missing_inputs": missing},
                      note=f"missing inputs: {', '.join(missing)}")

    upstream = {d: load_result(d, root) or {} for d in spec.requires}
    ctx = Context(spec=spec, out_dir=out_dir, root=root, upstream=upstream)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[{spec.id}] {spec.title}\n  question  : {spec.question}\n"
              f"  criterion : {spec.criterion}", flush=True)

    t0 = time.time()
    try:
        outcome = run(ctx)
    except Exception:                                    # noqa: BLE001
        return finish(STATUS_ERROR, traceback=traceback.format_exc(),
                      elapsed_min=(time.time() - t0) / 60,
                      note="run raised; recorded rather than swallowed")

    payload["elapsed_min"] = (time.time() - t0) / 60
    if not isinstance(outcome, Outcome):
        return finish(STATUS_ERROR,
                      note=f"run returned {type(outcome).__name__}, not Outcome")

    return finish(STATUS_PASSED if outcome.passed else STATUS_FAILED,
                  observed=outcome.observed, tables=outcome.tables,
                  population=outcome.population, note=outcome.note)


def main(spec: Spec, run: Callable[[Context], Outcome]) -> int:
    """``sys.exit(main(SPEC, run))`` -- non-zero unless the criterion was met."""
    payload = run_experiment(spec, run)
    return 0 if payload["status"] == STATUS_PASSED else 1
