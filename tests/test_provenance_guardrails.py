"""Tests for the provenance guardrails added after the synthetic-data incident.

These assert that the *safety properties* hold — a results record cannot be
written without provenance, the version pins reject the expired materialization,
a missing token raises instead of silently sending an empty credential, and the
lint still detects the original defects.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

from neuronauts.data import auth, versions  # noqa: E402
from neuronauts.results_schema import (  # noqa: E402
    ProvenanceError,
    ResultsRecord,
    read_results,
    write_results,
)


# ---------------------------------------------------------------------------
# results schema
# ---------------------------------------------------------------------------

def _valid_record(**over):
    kw = dict(
        experiment="unit_test",
        split="test",
        metrics={"ari": 0.5},
        base_version=117,
        label_version=1718,
        synthetic=False,
        data_manifest_sha="deadbeef",
    )
    kw.update(over)
    return ResultsRecord(**kw)


def test_valid_record_roundtrips(tmp_path):
    p = write_results(_valid_record(), tmp_path / "r.json")
    d = read_results(p)
    assert d["metrics"]["ari"] == 0.5
    assert d["synthetic"] is False
    assert "real data" in d["provenance_caption"]


def test_real_record_requires_data_manifest():
    """A real result must name the dataset it ran on."""
    with pytest.raises(ProvenanceError, match="data_manifest_sha"):
        _valid_record(data_manifest_sha=None).to_dict()


def test_synthetic_record_requires_notes():
    """Synthetic runs are allowed, but must say what was generated."""
    with pytest.raises(ProvenanceError, match="notes"):
        _valid_record(synthetic=True, data_manifest_sha=None).to_dict()


def test_synthetic_record_is_captioned(tmp_path):
    rec = _valid_record(synthetic=True, data_manifest_sha=None,
                        notes="generated fragments for a smoke test")
    d = rec.to_dict()
    assert d["provenance_caption"].startswith("SYNTHETIC")


def test_expired_label_version_rejected():
    """v1412 was removed server-side; a 'real' result claiming it is impossible."""
    with pytest.raises(ProvenanceError, match="1412"):
        _valid_record(label_version=1412).to_dict()


def test_metrics_are_required():
    with pytest.raises(ProvenanceError, match="no metrics"):
        _valid_record(metrics={}).to_dict()


def test_unknown_split_rejected():
    with pytest.raises(ProvenanceError, match="split"):
        _valid_record(split="holdout-ish").to_dict()


def test_read_results_rejects_unstamped(tmp_path):
    p = tmp_path / "bare.json"
    p.write_text(json.dumps({"metrics": {"ari": 0.9}}))
    with pytest.raises(ProvenanceError):
        read_results(p)


# ---------------------------------------------------------------------------
# version + coordinate contract
# ---------------------------------------------------------------------------

def test_pinned_versions():
    assert versions.BASE_VERSION == 117
    assert versions.LABEL_VERSION == 1718
    assert 1412 in versions.EXPIRED_VERSIONS


def test_expired_version_rejected_offline():
    """Guarded before any network call, so it fails fast and without a token."""
    with pytest.raises(versions.VersionContractError, match="expired"):
        versions.verify_version_contract(label=1412)


def test_coordinate_roundtrip():
    import numpy as np

    pt = (818_500.0, 685_000.0, 794_000.0)
    back = versions.voxel_to_nm(versions.nm_to_voxel(pt))
    assert np.allclose(back, pt)


def test_synapse_and_nucleus_frames_differ():
    """The (8,8,40) nucleus frame once put ~93% of box centres outside the
    volume when used as the synapse frame; they must stay distinct."""
    assert versions.SYNAPSE_VOXEL_NM != versions.NUCLEUS_CSV_VOXEL_NM


# ---------------------------------------------------------------------------
# token handling
# ---------------------------------------------------------------------------

def test_missing_token_raises(monkeypatch):
    for var in ("CAVE_TOKEN", "CAVE_BEARER_TOKEN", "token"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(auth.MissingCaveToken):
        auth.cave_token()


def test_missing_token_optional(monkeypatch):
    for var in ("CAVE_TOKEN", "CAVE_BEARER_TOKEN", "token"):
        monkeypatch.delenv(var, raising=False)
    assert auth.cave_token(required=False) is None


def test_token_read_from_env(monkeypatch):
    monkeypatch.setenv("CAVE_TOKEN", "abc123")
    assert auth.cave_token() == "abc123"


def test_redact_hides_the_secret():
    out = auth.redact("0123456789abcdef0123456789abcdef")
    assert "0123456789abcdef" not in out
    assert "len=32" in out


def test_headers_refuse_bearer_none(monkeypatch):
    """A missing token must raise, never send 'Bearer None' and get an empty
    result that downstream code could mistake for data."""
    for var in ("CAVE_TOKEN", "CAVE_BEARER_TOKEN", "token"):
        monkeypatch.delenv(var, raising=False)
    from neuronauts.data.lineage import _headers

    with pytest.raises(auth.MissingCaveToken):
        _headers(None)


# ---------------------------------------------------------------------------
# provenance lint
# ---------------------------------------------------------------------------

def _run_lint(*paths):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "lint_provenance.py"),
         "--paths", *[str(p) for p in paths]],
        capture_output=True, text=True, cwd=REPO,
    )


def test_lint_is_clean_on_main_tree():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "lint_provenance.py")],
        capture_output=True, text=True, cwd=REPO,
    )
    assert r.returncode == 0, f"provenance lint regressed:\n{r.stdout}"


@pytest.mark.parametrize("code,snippet", [
    ("LEAK001", "def score_bridge(a, b, is_true_continuation, rng):\n    return 1.0\n"),
    ("LEAK002", "if is_same_cell:\n    p = clip(p + 0.40, 0.7, 0.98)\n"),
    ("SYNTH001", "from treestitch.data import _split_skeleton_n_pieces\n"),
    ("SYNTH003", "partner_base = obj_counter * 100\n"),
    ("SEC001", 'token = "' + "a08c" + "dcba8581846f48d5742a75c53311" + '"\n'),
    ("SPLIT001", "rng.shuffle(all_records)\n"),
])
def test_lint_detects_each_defect_class(tmp_path, code, snippet):
    """Each rule must still fire — a rule that cannot fire is not a guardrail."""
    f = tmp_path / f"bad_{code.lower()}.py"
    f.write_text(snippet)
    r = _run_lint(f)
    assert code in r.stdout, f"{code} did not fire on:\n{snippet}\n{r.stdout}"
    assert r.returncode == 1


def test_lint_detects_historical_defects_in_quarantine():
    """Regression guard against a refactor silently neutering the rules."""
    sampler = (REPO / "quarantine" / "neuronauts" / "global_merge" /
               "represent" / "cloudvolume_em_sampler.py")
    dashboard = REPO / "quarantine" / "scripts" / "generate_dashboard.py"
    exp049 = REPO / "quarantine" / "scripts" / "benchmark_exp049_dense_subvolume.py"
    if not sampler.exists():
        pytest.skip("quarantine not present in this checkout")
    out = _run_lint(sampler, dashboard, exp049).stdout
    for code in ("LEAK001", "LEAK002", "SYNTH001", "SYNTH002", "SYNTH003"):
        assert code in out, f"{code} no longer detected in known-bad code"


def test_lint_allows_inline_waiver(tmp_path):
    f = tmp_path / "waived.py"
    f.write_text(
        "# provenance-lint: allow SYNTH003 - documented test fixture\n"
        "partner_base = obj_counter * 100\n"
    )
    assert _run_lint(f).returncode == 0


# ---------------------------------------------------------------------------
# quarantine stays sealed
# ---------------------------------------------------------------------------

def test_no_live_code_imports_quarantined_modules():
    bad = []
    for p in REPO.rglob("*.py"):
        rel = p.relative_to(REPO).as_posix()
        if (rel.startswith("quarantine/") or "lint_provenance" in rel
                or rel == "tests/test_provenance_guardrails.py"):
            continue  # these name the modules in order to police them
        if any(part in {".venv", "build", "dist", "__pycache__"}
               for part in p.relative_to(REPO).parts):
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for mod in ("morpho_grammar", "cloudvolume_em_sampler", "local_em_verifier"):
            if mod in text:
                bad.append(f"{rel} -> {mod}")
    assert not bad, "live code references quarantined modules: " + ", ".join(bad)
