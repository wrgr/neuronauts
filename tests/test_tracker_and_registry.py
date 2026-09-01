"""The tracker's value is that it cannot flatter us.

Every check must be derived from disk, a skipped check must read as skipped
rather than as passing, and the experiment DAG must refuse to call something
ready when its prerequisites are not.
"""

from __future__ import annotations

import json

import pytest

from neuronauts.experiments import Outcome, Spec, run_experiment
from neuronauts.experiments import registry as reg
from neuronauts.report import tracker


# ---------------------------------------------------------------------------
# the registry as a dependency graph
# ---------------------------------------------------------------------------

def test_every_experiment_declares_a_bar():
    for e in reg.REGISTRY:
        assert e.spec.criterion.strip(), f"{e.id} has no criterion"


def test_ids_are_unique():
    ids = [e.id for e in reg.REGISTRY]
    assert len(ids) == len(set(ids))


def test_every_prerequisite_is_a_registered_experiment():
    known = {e.id for e in reg.REGISTRY}
    for e in reg.REGISTRY:
        for dep in e.spec.requires:
            assert dep in known, f"{e.id} requires unregistered {dep}"


def test_the_dependency_graph_is_acyclic():
    dep = {e.id: set(e.spec.requires) for e in reg.REGISTRY}
    resolved: set[str] = set()
    for _ in range(len(dep) + 1):
        step = {k for k, v in dep.items() if v <= resolved}
        if step == resolved:
            break
        resolved = step
    assert resolved == set(dep), f"cycle or unreachable: {set(dep) - resolved}"


def test_prerequisites_come_earlier_in_program_order():
    """Reading the registry top to bottom must be a runnable order."""
    seen: set[str] = set()
    for e in reg.REGISTRY:
        for d in e.spec.requires:
            assert d in seen, f"{e.id} requires {d}, which is listed later"
        seen.add(e.id)


def test_an_experiment_with_unmet_prerequisites_is_never_ready(tmp_path):
    entry = reg.by_id("EXP-064")
    assert entry is not None
    state, why = reg.state(entry, tmp_path)
    assert state != "ready"
    assert why, "a blocked experiment must say why"


def test_state_reads_a_passing_result_from_disk(tmp_path):
    d = tmp_path / "results" / "EXP-057"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"id": "EXP-057",
                                               "status": "passed"}))
    assert reg.state(reg.by_id("EXP-057"), tmp_path)[0] == "passed"


def test_next_runnable_unlocks_as_upstream_results_land(tmp_path):
    """The autonomy primitive: satisfy a prerequisite, more becomes ready."""
    for name in ("population.npz",):
        (tmp_path / "data" / "substrate" / "c100um").mkdir(parents=True,
                                                           exist_ok=True)
        (tmp_path / "data" / "substrate" / "c100um" / name).write_bytes(b"0")
    (tmp_path / "data" / "substrate" / "topology").mkdir(parents=True)
    (tmp_path / "data" / "substrate" / "topology" / "k10.npz").write_bytes(b"0")
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "atom_labels_v1822.json").write_text("{}")

    before = {e.id for e in reg.next_runnable(tmp_path)}
    d = tmp_path / "results" / "EXP-057"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"status": "passed"}))
    after = {e.id for e in reg.next_runnable(tmp_path)}
    assert after >= before
    # EXP-058 depends only on EXP-057 and the two files above
    assert "EXP-058" in after or reg.by_id("EXP-058").module is None


def test_a_failed_upstream_does_not_unlock_downstream(tmp_path):
    d = tmp_path / "results" / "EXP-057"
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"status": "failed"}))
    state, why = reg.state(reg.by_id("EXP-058"), tmp_path)
    assert state == "blocked"
    assert any("EXP-057" in w for w in why)


def test_summary_counts_every_entry(tmp_path):
    s = reg.summary(tmp_path)
    assert s["total"] == len(reg.REGISTRY)
    assert sum(s["by_state"].values()) == s["total"]


# ---------------------------------------------------------------------------
# the tracker
# ---------------------------------------------------------------------------

def test_a_skipped_check_is_not_reported_as_passing():
    """--fast must read as unknown; a skipped check that reads as done lies."""
    c = tracker.Check("x", None, "not checked")
    ph = tracker.Phase("9", "t", [c])
    assert ph.done == 0
    assert ph.unknown == 1
    assert ph.state == "partial", "unknown is not 'not started'"


def test_phase_states():
    done = tracker.Phase("1", "t", [tracker.Check("a", True)])
    none = tracker.Phase("2", "t", [tracker.Check("a", False)])
    part = tracker.Phase("3", "t", [tracker.Check("a", True),
                                    tracker.Check("b", False)])
    assert (done.state, none.state, part.state) == ("done", "not started",
                                                    "partial")


def test_exists_is_case_sensitive(tmp_path):
    """macOS is case-insensitive; a check for ARCHITECTURE.md must not be
    satisfied by architecture.md."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "architecture.md").write_text("x")
    assert tracker._exists(tmp_path, "docs/architecture.md")
    assert not tracker._exists(tmp_path, "docs/ARCHITECTURE.md")


def test_checks_are_derived_from_disk_not_asserted(tmp_path):
    """An empty tree must report almost nothing done."""
    phases = tracker.consolidation_phases(tmp_path)
    done = sum(p.done for p in phases)
    assert done <= 3, f"empty repo reported {done} checks done"


def test_rendering_does_not_crash_on_an_empty_tree(tmp_path):
    tracker._cache["collect"] = (-1, 0)
    assert "neuronauts status" in tracker.render_text(tmp_path,
                                                      show_experiments=False)
    assert "| Phase |" in tracker.render_markdown(tmp_path)
    d = tracker.as_dict(tmp_path)
    assert {"consolidation", "experiments", "summary", "repo"} <= set(d)


def test_json_round_trips(tmp_path):
    tracker._cache["collect"] = (-1, 0)
    p = tracker.write_json(tmp_path / "s.json", tmp_path)
    assert json.loads(p.read_text())["summary"]["total"] == len(reg.REGISTRY)


def test_tracker_sees_a_result_the_runner_wrote(tmp_path):
    """End to end: the runner writes, the tracker reads the same fact."""
    spec = Spec(id="EXP-057", title="t", question="q", criterion="c")
    run_experiment(spec, lambda ctx: Outcome(passed=True, observed={"a": 1}),
                   root=tmp_path, verbose=False)
    assert reg.state(reg.by_id("EXP-057"), tmp_path)[0] == "passed"
    d = tracker.as_dict(tmp_path)
    assert d["summary"]["passed"] == 1
