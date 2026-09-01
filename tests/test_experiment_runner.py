"""The runner's job is to refuse, honestly, in the cases that matter.

EXP-053B, 054 and 055 each declined to report metrics because a prerequisite
failed, and that was the right behaviour. These tests pin it down so it cannot
regress into "report whatever came out".
"""

from __future__ import annotations

import json

import pytest

from neuronauts.experiments import (
    Context, Outcome, Spec, check_inputs, check_prerequisites, load_result,
    result_path, run_experiment,
)
from neuronauts.experiments._runner import (
    STATUS_ERROR, STATUS_FAILED, STATUS_PASSED, STATUS_PREREQ,
)


def spec(**kw):
    base = dict(id="EXP-999", title="t", question="q?", criterion="x >= 1")
    base.update(kw)
    return Spec(**base)


def ok(_ctx):
    return Outcome(passed=True, observed={"x": 2.0}, population={"n": 3})


def write_upstream(root, exp_id, status):
    p = root / "results" / exp_id
    p.mkdir(parents=True, exist_ok=True)
    (p / "result.json").write_text(json.dumps({"id": exp_id, "status": status}))


# ---------------------------------------------------------------------------
# declaring the bar
# ---------------------------------------------------------------------------

def test_an_experiment_without_a_criterion_is_rejected():
    """A run with no bar cannot fail, so it cannot inform."""
    with pytest.raises(ValueError, match="criterion"):
        Spec(id="EXP-1", title="t", question="q", criterion="   ")


# ---------------------------------------------------------------------------
# prerequisites
# ---------------------------------------------------------------------------

def test_missing_prerequisite_blocks_the_run(tmp_path):
    ran = []
    payload = run_experiment(spec(requires=["EXP-057"]),
                             lambda c: ran.append(1) or ok(c),
                             root=tmp_path, verbose=False)
    assert payload["status"] == STATUS_PREREQ
    assert not ran, "run must not execute when a prerequisite is missing"
    assert "EXP-057" in payload["prerequisite_gate"]["unmet"][0]
    assert "observed" not in payload, "no metrics may be reported"


def test_failed_prerequisite_blocks_the_run(tmp_path):
    write_upstream(tmp_path, "EXP-057", STATUS_FAILED)
    payload = run_experiment(spec(requires=["EXP-057"]), ok,
                             root=tmp_path, verbose=False)
    assert payload["status"] == STATUS_PREREQ
    assert "status='failed'" in payload["prerequisite_gate"]["unmet"][0]


def test_passed_prerequisite_lets_it_run(tmp_path):
    write_upstream(tmp_path, "EXP-057", STATUS_PASSED)
    payload = run_experiment(spec(requires=["EXP-057"]), ok,
                             root=tmp_path, verbose=False)
    assert payload["status"] == STATUS_PASSED
    assert payload["observed"] == {"x": 2.0}


def test_blocked_result_is_still_written_and_ledgered(tmp_path):
    """A failure is a recorded fact, not an absence."""
    run_experiment(spec(requires=["EXP-057"]), ok, root=tmp_path, verbose=False)
    assert result_path("EXP-999", tmp_path).exists()
    assert "blocked" in (tmp_path / "results" / "RESULTS.md").read_text()


def test_upstream_results_are_handed_to_the_run(tmp_path):
    write_upstream(tmp_path, "EXP-057", STATUS_PASSED)
    seen = {}

    def r(ctx: Context):
        seen.update(ctx.upstream)
        return ok(ctx)

    run_experiment(spec(requires=["EXP-057"]), r, root=tmp_path, verbose=False)
    assert seen["EXP-057"]["status"] == STATUS_PASSED


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def test_missing_input_is_caught_before_the_run(tmp_path):
    ran = []
    payload = run_experiment(spec(inputs=["data/nope.npz"]),
                             lambda c: ran.append(1) or ok(c),
                             root=tmp_path, verbose=False)
    assert payload["status"] == STATUS_PREREQ
    assert not ran, "a two-hour run must not die at the end on a missing file"
    assert payload["prerequisite_gate"]["missing_inputs"] == ["data/nope.npz"]


def test_present_input_passes_the_check(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.npz").write_bytes(b"0")
    assert check_inputs(spec(inputs=["data/x.npz"]), tmp_path) == []


# ---------------------------------------------------------------------------
# verdict and errors
# ---------------------------------------------------------------------------

def test_not_meeting_the_criterion_is_failed_not_passed(tmp_path):
    payload = run_experiment(
        spec(), lambda c: Outcome(passed=False, observed={"x": 0.1}),
        root=tmp_path, verbose=False)
    assert payload["status"] == STATUS_FAILED
    assert payload["observed"] == {"x": 0.1}, "a failed run still reports what it saw"


def test_an_exception_is_recorded_never_swallowed(tmp_path):
    def boom(_ctx):
        raise RuntimeError("kaboom")

    payload = run_experiment(spec(), boom, root=tmp_path, verbose=False)
    assert payload["status"] == STATUS_ERROR
    assert "kaboom" in payload["traceback"]


def test_returning_the_wrong_type_is_an_error(tmp_path):
    payload = run_experiment(spec(), lambda c: {"passed": True},
                             root=tmp_path, verbose=False)
    assert payload["status"] == STATUS_ERROR


# ---------------------------------------------------------------------------
# the record
# ---------------------------------------------------------------------------

def test_result_carries_provenance_and_declared_criterion(tmp_path):
    payload = run_experiment(spec(flags={"synthetic_fallback": False}), ok,
                             root=tmp_path, verbose=False)
    on_disk = json.loads(result_path("EXP-999", tmp_path).read_text())
    assert on_disk["success_criterion"] == "x >= 1"
    assert on_disk["provenance"]["synthetic_fallback"] is False
    assert "timestamp_utc" in on_disk["provenance"]
    assert payload["status"] == STATUS_PASSED


def test_every_run_appends_exactly_one_ledger_row(tmp_path):
    for _ in range(3):
        run_experiment(spec(), ok, root=tmp_path, verbose=False)
    body = (tmp_path / "results" / "RESULTS.md").read_text()
    assert body.count("| EXP-999 |") == 3
    assert body.count("# Results ledger") == 1


def test_load_result_tolerates_a_corrupt_file(tmp_path):
    d = tmp_path / "results" / "EXP-1"
    d.mkdir(parents=True)
    (d / "result.json").write_text("{not json")
    assert load_result("EXP-1", tmp_path) is None


def test_check_prerequisites_reports_every_unmet_dependency(tmp_path):
    write_upstream(tmp_path, "EXP-A", STATUS_PASSED)
    write_upstream(tmp_path, "EXP-B", STATUS_FAILED)
    problems = check_prerequisites(
        spec(requires=["EXP-A", "EXP-B", "EXP-C"]), tmp_path)
    assert len(problems) == 2
    assert any("EXP-B" in p for p in problems) and any("EXP-C" in p for p in problems)


# ---------------------------------------------------------------------------
# "must have passed" vs "must have run"
# ---------------------------------------------------------------------------

def test_requires_ran_is_satisfied_by_a_failed_upstream(tmp_path):
    """A failed bar must not halt a path its failure does not touch.

    EXP-057 failed a bar about label *density*. Candidate generation needs the
    overlay it produced, not the density it measured.
    """
    write_upstream(tmp_path, "EXP-057", STATUS_FAILED)
    payload = run_experiment(spec(requires_ran=["EXP-057"]), ok,
                             root=tmp_path, verbose=False)
    assert payload["status"] == STATUS_PASSED


def test_requires_ran_still_blocks_when_upstream_never_ran(tmp_path):
    payload = run_experiment(spec(requires_ran=["EXP-057"]), ok,
                             root=tmp_path, verbose=False)
    assert payload["status"] == STATUS_PREREQ
    assert "has not run" in payload["prerequisite_gate"]["unmet"][0]


def test_requires_ran_blocks_when_upstream_produced_no_artifact(tmp_path):
    """Blocked or errored upstream means there is nothing to build on."""
    for status in (STATUS_PREREQ, STATUS_ERROR):
        write_upstream(tmp_path, "EXP-057", status)
        payload = run_experiment(spec(requires_ran=["EXP-057"]), ok,
                                 root=tmp_path, verbose=False)
        assert payload["status"] == STATUS_PREREQ, status


def test_requires_and_requires_ran_are_both_enforced(tmp_path):
    write_upstream(tmp_path, "EXP-A", STATUS_FAILED)
    write_upstream(tmp_path, "EXP-B", STATUS_PASSED)
    # strict dep failed -> blocked
    p = run_experiment(spec(requires=["EXP-A"], requires_ran=["EXP-B"]), ok,
                       root=tmp_path, verbose=False)
    assert p["status"] == STATUS_PREREQ
    # swap: strict dep passed, ran-dep failed -> allowed
    p = run_experiment(spec(requires=["EXP-B"], requires_ran=["EXP-A"]), ok,
                       root=tmp_path, verbose=False)
    assert p["status"] == STATUS_PASSED


def test_upstream_results_include_ran_dependencies(tmp_path):
    write_upstream(tmp_path, "EXP-057", STATUS_FAILED)
    seen = {}
    run_experiment(spec(requires_ran=["EXP-057"]),
                   lambda c: (seen.update(c.upstream), ok(c))[1],
                   root=tmp_path, verbose=False)
    assert seen["EXP-057"]["status"] == STATUS_FAILED
