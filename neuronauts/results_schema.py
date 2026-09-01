"""Provenance-stamped results records.

Every reported metric must carry enough provenance to answer, months later and
without reading the script: *what data produced this, which model, and was any
of it synthetic?*

The synthetic-data incident was possible partly because results files recorded
numbers and nothing else. A table saying "merge_P 0.75" is indistinguishable
from a real measurement unless the artifact itself says what it ran on. This
module refuses to write a record that cannot answer that question.

Usage::

    from neuronauts.results_schema import ResultsRecord, write_results

    rec = ResultsRecord(
        experiment="bench_v1_baseline_union_find",
        split="test",
        metrics={"ari": 0.61, "merge_precision": 0.95},
        data_manifest_sha="ab12...",
        base_version=117,
        label_version=1718,
        synthetic=False,
        checkpoint_sha="c0ffee...",
        notes="cc_bias=-2.0 calibrated on val",
    )
    write_results(rec, "results/bench_v1_baseline.json")
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1


class ProvenanceError(ValueError):
    """Raised when a results record lacks required provenance."""


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, for stamping checkpoints and manifests."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ResultsRecord:
    """One reported result, with the provenance needed to trust it."""

    experiment: str
    split: str                      # "train" | "val" | "test" | "ooc" | ...
    metrics: dict[str, Any]

    # --- provenance: all required, none defaulted to something permissive ---
    base_version: int
    label_version: int
    synthetic: bool
    data_manifest_sha: Optional[str] = None
    checkpoint_sha: Optional[str] = None

    # --- context ---
    notes: str = ""
    n_observations: Optional[int] = None
    n_fragments: Optional[int] = None
    gates: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    # --- filled automatically ---
    schema_version: int = SCHEMA_VERSION
    git_commit: str = field(default_factory=_git_commit)
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def validate(self) -> None:
        """Fail closed on missing or self-contradictory provenance."""
        if not self.experiment:
            raise ProvenanceError("experiment name is required")
        if self.split not in {"train", "val", "test", "ooc", "diagnostic"}:
            raise ProvenanceError(
                f"split must be one of train/val/test/ooc/diagnostic, "
                f"got {self.split!r}"
            )
        if not isinstance(self.synthetic, bool):
            raise ProvenanceError("synthetic must be an explicit bool")
        if not self.metrics:
            raise ProvenanceError("a results record with no metrics is not a result")

        if self.synthetic:
            # Synthetic runs are allowed, but may never masquerade as real.
            if not self.notes:
                raise ProvenanceError(
                    "a synthetic record must carry notes saying what was "
                    "generated and why"
                )
        else:
            # Real runs must name the data they ran on.
            if not self.data_manifest_sha:
                raise ProvenanceError(
                    "a non-synthetic record must carry data_manifest_sha "
                    "identifying the dataset it ran on"
                )
            if self.label_version in (1412,):
                raise ProvenanceError(
                    "label_version 1412 is expired server-side; results "
                    "claiming it cannot be real"
                )

    def to_dict(self) -> dict:
        self.validate()
        d = asdict(self)
        # Make the caption impossible to miss when a human opens the file.
        d["provenance_caption"] = (
            "SYNTHETIC — not a measurement on real data"
            if self.synthetic
            else f"real data; base v{self.base_version} -> labels v{self.label_version}"
        )
        return d


def write_results(record: ResultsRecord, path: str | Path) -> Path:
    """Validate and write a results record as JSON. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True) + "\n")
    return p


def read_results(path: str | Path) -> dict:
    """Read a results record, validating that it carries provenance."""
    d = json.loads(Path(path).read_text())
    for key in ("experiment", "split", "metrics", "base_version",
                "label_version", "synthetic"):
        if key not in d:
            raise ProvenanceError(f"{path}: missing required field {key!r}")
    return d
