"""Provenance capture/grading, result normalisation, figures and rendering.

The registry is exercised on a synthetic result shaped like EXP-056 (a sweep
of rules with nested pair counts, a gate, a best-rule block) and one shaped
like EXP-053B (a radius x cone grid), because those are the two table shapes
the real files use.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from neuronauts.report import provenance as P
from neuronauts.report.figures import figures_for_record, pick_metrics
from neuronauts.report.registry import (
    Table, dict_table, discover, flatten, list_table, load_record, natural_key,
)
from neuronauts.report.render import render_experiment, render_index

ROOT = Path(__file__).resolve().parents[1]


def _sweep_row(p, r, split, perfect, tp, fp, fn, tn):
    return {"pair_precision": p, "pair_recall": r, "cross_lineage_split_recall": split,
            "perfect_roots": perfect, "pair_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def exp056_like() -> dict:
    return {
        "experiment": "EXP-999 synthetic sweep",
        "provenance": {"git_commit": "0" * 40, "bbox_nm": [[0, 0, 0], [30000, 30000, 30000]],
                       "synthetic_fallback": False, "labels_used_only_for_evaluation": True,
                       "input": "results/exp998_upstream.json"},
        "population": {"evaluated_roots": 116, "min_observations": 10},
        "success_criterion": {"required_pair_recall": 0.9,
                              "required_cross_lineage_split_recall": 0.5, "passed": False},
        "best_rule_with_pair_recall_at_least_0.9": {"rule": "absolute_10um",
                                                    **_sweep_row(.87, .94, .23, 13, 1, 2, 3, 4)},
        "sweep": {
            "atomic": _sweep_row(.84, 1.0, 0.0, 0, 145014, 27280, 0, 0),
            "absolute_10um": _sweep_row(.87, .94, .23, 13, 136425, 21049, 8589, 6231),
            "absolute_1um": _sweep_row(.99, .02, 1.0, 0, 2713, 9, 142301, 27271),
            "absolute_0.5um": _sweep_row(.99, .004, 1.0, 0, 616, 2, 144398, 27278),
            "quantile_0.99": _sweep_row(.88, .78, .45, 25, 113503, 14927, 31511, 12353),
        },
        "interpretation": "x" * 100,
        "elapsed_seconds": 186.0,
    }


def grid_like() -> dict:
    grid = {}
    for r in (0.5, 1, 2.5, 5, 10):
        for c in (30, 90, 180):
            grid[f"r{r:g}_cone{c}"] = {"candidate_pairs": int(r * c),
                                       "recall_all_true_pairs": 0.0,
                                       "panel_size": {"median": r, "p90": r * 2}}
    return {"experiment": "EXP-998 grid", "grid": grid,
            "provenance": {"git_commit": "1" * 40, "synthetic_fallback": False},
            "success_criterion": {"required_recall": 0.9, "passed": False}}


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------

def test_git_state_matches_git():
    st = P.git_state(ROOT)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert st["git_commit"] == head
    assert isinstance(st["git_dirty"], bool)


def test_hash_file_is_sha256(tmp_path):
    f = tmp_path / "in.bin"
    f.write_bytes(b"neuronauts" * 1000)
    rec = P.hash_file(f)
    assert rec["algo"] == "sha256"
    assert rec["hash"] == hashlib.sha256(b"neuronauts" * 1000).hexdigest()
    assert rec["bytes"] == 10000


def test_quick_hash_is_labelled(tmp_path):
    f = tmp_path / "big.bin"
    f.write_bytes(b"\0" * (9 << 20))
    assert P.hash_file(f, quick=True)["algo"] == "quick-sha256"
    assert P.hash_file(f, quick=False)["algo"] == "sha256"


def test_capture_and_write_result(tmp_path):
    inp = tmp_path / "input.json"
    inp.write_text("{}")
    out = tmp_path / "out.json"
    P.write_result(out, {"experiment": "t", "value": 1},
                   inputs=[inp, tmp_path / "missing.npz"], params={"k": 10},
                   synthetic_fallback=False)
    data = json.loads(out.read_text())
    prov = data["provenance"]
    assert len(prov["git_commit"]) == 40
    assert prov["inputs"][0]["hash"] and prov["inputs"][1]["missing"] is True
    assert prov["params"] == {"k": 10}
    assert prov["synthetic_fallback"] is False
    assert P.completeness(prov)["score"] == 1.0
    assert not out.with_suffix(".json.tmp").exists()


def test_completeness_grades_partial_blocks():
    old = {"git_commit": "abc", "synthetic_fallback": False}
    g = P.completeness(old)
    assert g["score"] == pytest.approx(0.35)
    assert "inputs" in g["missing"] and "git_dirty" in g["missing"]
    assert P.completeness(None)["score"] == 0.0
    # a bare path is not a hashed input
    assert "inputs" in P.completeness({"inputs": [{"path": "x"}]})["missing"]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def test_natural_key_orders_thresholds():
    names = ["absolute_10um", "absolute_1um", "absolute_0.5um", "absolute_0.25um", "atomic"]
    assert sorted(names, key=natural_key) == [
        "absolute_0.25um", "absolute_0.5um", "absolute_1um", "absolute_10um", "atomic"]


def test_flatten_dotted_keys():
    assert flatten({"a": {"b": 1, "c": {"d": 2.5}}, "e": "x", "f": [1, 2]}) == \
        {"a.b": 1, "a.c.d": 2.5, "e": "x", "f": "[1, 2]"}


def test_dict_table_detects_sweep_and_nested_counts():
    hit = dict_table("sweep", exp056_like()["sweep"])
    assert hit is not None
    t, subs = hit
    assert t.rows[0] == "absolute_0.5um" and t.rows[-1] == "quantile_0.99"
    assert "pair_counts.tp" in t.columns and "pair_precision" in t.numeric_columns()
    assert t.value("atomic", "pair_counts.fp") == 27280
    assert subs == []


def test_dict_table_rejects_scalar_dicts():
    assert dict_table("population", {"a": 1, "b": 2}) is None
    assert dict_table("prov", {"a": {"x": "s"}, "b": {"x": "t"}}) is None


def test_nested_table_becomes_subtable():
    raw = {"ckpt_a": {"edges": 5, "sweep": {"0": {"f1": .1}, "1": {"f1": .2}}},
           "ckpt_b": {"edges": 6, "sweep": {"0": {"f1": .3}, "1": {"f1": .4}}}}
    t, subs = dict_table("checkpoints", raw)
    assert t.columns == ["edges"]
    assert [s.name for s in subs] == ["checkpoints.ckpt_a.sweep", "checkpoints.ckpt_b.sweep"]


def test_list_table_labels_by_id_key():
    t, _ = list_table("tiers", [{"k": 10, "n_atoms": 5}, {"k": 5, "n_atoms": 9}])
    assert t.rows == ["10", "5"] and t.value("5", "n_atoms") == 9


def test_load_record_exp_like(tmp_path):
    src = tmp_path / "exp999_synthetic.json"
    src.write_text(json.dumps(exp056_like()))
    rec = load_record(src, ROOT)
    assert rec.id == "EXP-999" and rec.family == "benchmark"
    assert rec.status == "failed" and "success_criterion" in rec.status_reason
    assert rec.population == {"evaluated_roots": 116, "min_observations": 10}
    assert [t.name for t in rec.tables] == ["sweep"]
    assert rec.headline["best_rule_with_pair_recall_at_least_0.9.rule"] == "absolute_10um"
    assert "interpretation" in rec.notes and "interpretation" not in rec.headline
    assert rec.dependencies == ["results/exp998_upstream.json"]
    assert {g["requirement"] for g in rec.gate} == {"pair_recall", "cross_lineage_split_recall"}
    assert rec.provenance_grade["score"] == pytest.approx(0.35)
    assert rec.commit_info is None  # fake sha is not in the repo


def test_load_record_status_field_and_list_root(tmp_path):
    src = tmp_path / "exp997_gate.json"
    src.write_text(json.dumps({"status": "prerequisite_failed",
                               "prerequisite_gate": {"required_covered_positives": 10,
                                                     "observed_covered_positives": 1,
                                                     "passed": False,
                                                     "failures": ["too few"]}}))
    rec = load_record(src, ROOT)
    assert rec.status == "prerequisite_failed"
    reqs = {g["requirement"]: g for g in rec.gate}
    assert reqs["covered_positives"]["required"] == 10
    assert reqs["covered_positives"]["observed"] == 1
    assert reqs["failure"]["observed"] == "too few"

    lst = tmp_path / "atom_geometry_tiers.json"
    lst.write_text(json.dumps([{"k": 10, "n_atoms": 3}, {"k": 5, "n_atoms": 4}]))
    rec = load_record(lst, ROOT)
    assert rec.family == "harness" and rec.status == "completed"
    assert rec.tables[0].rows == ["10", "5"]


def test_percentile_series(tmp_path):
    src = tmp_path / "atom_topology_k10.json"
    src.write_text(json.dumps({"tier": 10, "leaf_len_nm_pct": {"10": 1, "50": 2, "90": 3, "99": 9},
                               "two": {"1": 1, "2": 2}}))
    rec = load_record(src, ROOT)
    series = rec.percentile_series()
    assert list(series) == ["leaf_len_nm_pct"]
    assert series["leaf_len_nm_pct"] == {10.0: 1.0, 50.0: 2.0, 90.0: 3.0, 99.0: 9.0}


def test_discover_real_results_parse():
    """Every checked-in result file must load; this is the regression net."""
    recs = discover("results", ROOT)
    ids = [r.id for r in recs]
    assert "EXP-056" in ids and "EXP-053B" in ids
    exp056 = next(r for r in recs if r.id == "EXP-056")
    assert exp056.status == "failed"
    assert exp056.table("sweep") is not None and len(exp056.table("sweep").rows) == 16
    assert exp056.script is not None and exp056.evaluation_md is not None
    exp055 = next(r for r in recs if r.id == "EXP-055")
    assert exp055.status == "prerequisite_failed"
    assert exp055.dependencies == ["results/exp054_fixed_panel_scorers.json"]


# ---------------------------------------------------------------------------
# figures + rendering
# ---------------------------------------------------------------------------

def test_pick_metrics_prefers_priority_and_skips_counts():
    t, _ = dict_table("sweep", exp056_like()["sweep"])
    m = pick_metrics(t)
    assert m[:3] == ["pair_precision", "pair_recall", "cross_lineage_split_recall"]
    assert not any(x.startswith("pair_counts.") for x in m)


def test_figures_and_report_for_sweep(tmp_path):
    pytest.importorskip("matplotlib")
    src = tmp_path / "exp999_synthetic.json"
    src.write_text(json.dumps(exp056_like()))
    rec = load_record(src, ROOT)
    figs = figures_for_record(rec, tmp_path / "figures")
    names = {f.path.name for f in figs}
    assert "EXP-999_sweep_panels.png" in names
    assert "EXP-999_sweep_operating.png" in names
    assert "EXP-999_sweep_pairs.png" in names
    assert all(f.path.stat().st_size > 1000 for f in figs)

    md = render_experiment(rec, figs, {"box": {"json": tmp_path / "b.json", "url": "http://x"}},
                           tmp_path / "reports", ROOT)
    text = md.read_text()
    assert text.startswith("# EXP-999 — EXP-999 synthetic sweep")
    assert "FAILED (gate not met)" in text
    assert "| `atomic` |" in text and "pair_precision" in text
    assert "figures/EXP-999_sweep_operating.png" in text
    assert "open in Neuroglancer" in text
    assert "Provenance completeness** 35%" in text


def test_grid_heatmap_and_index(tmp_path):
    pytest.importorskip("matplotlib")
    src = tmp_path / "exp998_grid.json"
    src.write_text(json.dumps(grid_like()))
    rec = load_record(src, ROOT)
    figs = figures_for_record(rec, tmp_path / "figures")
    assert any(f.path.name.endswith("candidate_pairs_grid.png") for f in figs)

    other = tmp_path / "exp999_synthetic.json"
    other.write_text(json.dumps(exp056_like()))
    recs = [rec, load_record(other, ROOT)]
    paths = {r.id: render_experiment(r, [], {}, tmp_path / "reports", ROOT) for r in recs}
    index = render_index(recs, paths, tmp_path / "reports", ROOT)
    text = index.read_text()
    assert "| EXP-998 |" in text and "| EXP-999 |" in text
    assert "```mermaid" in text and "--> EXP_999" in text


def test_sibling_scalar_blocks_become_one_table(tmp_path):
    """A baseline block beside a chosen-rule block reads as two rows, not 20 keys."""
    payload = exp056_like()
    payload["atomic_baseline"] = _sweep_row(.84, 1.0, 0.0, 0, 145014, 27280, 0, 0)
    src = tmp_path / "exp996_siblings.json"
    src.write_text(json.dumps(payload))
    rec = load_record(src, ROOT)
    summary = rec.table("summary")
    assert summary is not None
    assert set(summary.rows) == {"atomic_baseline", "best_rule_with_pair_recall_at_least_0.9"}
    assert summary.value("atomic_baseline", "pair_recall") == 1.0
    assert summary.value("best_rule_with_pair_recall_at_least_0.9", "rule") == "absolute_10um"
    # and those keys are no longer duplicated as headline scalars
    assert not any(k.startswith("atomic_baseline.") for k in rec.headline)


def test_unrelated_scalar_blocks_stay_headline(tmp_path):
    src = tmp_path / "exp995_mixed.json"
    src.write_text(json.dumps({
        "experiment": "x",
        "timing": {"graphs_s": 1.0, "coords_s": 2.0, "v117_s": 3.0},
        "config": {"version": 1822, "side_um": 200.0, "n_cells": 5},
    }))
    rec = load_record(src, ROOT)
    assert rec.table("summary") is None
    assert rec.headline["timing.graphs_s"] == 1.0 and rec.headline["config.n_cells"] == 5


def test_two_row_table_gets_no_bar_chart(tmp_path):
    pytest.importorskip("matplotlib")
    src = tmp_path / "exp994_two.json"
    src.write_text(json.dumps({"experiment": "x", "sweep": {
        "a": {"pair_precision": .1, "pair_recall": .2},
        "b": {"pair_precision": .3, "pair_recall": .4}}}))
    rec = load_record(src, ROOT)
    assert rec.table("sweep") is not None          # still tabulated
    assert figures_for_record(rec, tmp_path / "figures") == []


def test_sibling_subtables_get_a_comparison_figure(tmp_path):
    pytest.importorskip("matplotlib")
    ckpts = {}
    for i, name in enumerate(["a.pt", "b.pt", "c.pt"]):
        ckpts[name] = {"candidate_edges": 100 + i,
                       "threshold_sweep": {str(t): {"merge_recall": 0.9 - 0.1 * t - 0.05 * i,
                                                    "merge_precision": 0.01 * t,
                                                    "erl_um": 80 + t}
                                           for t in range(5)}}
    src = tmp_path / "exp993_bakeoff.json"
    src.write_text(json.dumps({"experiment": "bake-off", "checkpoints": ckpts}))
    rec = load_record(src, ROOT)
    figs = figures_for_record(rec, tmp_path / "figures")
    names = {f.path.name for f in figs}
    assert "EXP-993_checkpoints_threshold_sweep_compare.png" in names
    spec = next(f for f in figs if f.path.name.endswith("_compare.png"))
    assert "one line per checkpoints entry" in spec.caption
    assert spec.path.stat().st_size > 1000
