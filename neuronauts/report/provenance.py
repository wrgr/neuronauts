"""Run provenance: what code, on what inputs, in what environment, produced a result.

Two directions:

* **Capture** (for new scripts): :func:`capture_provenance` records the git
  commit *and* whether the tree was dirty, the command line, the interpreter
  and key package versions, and a content hash of every declared input file.
  :func:`stamp` folds that into a result payload; :func:`write_result` writes
  the payload atomically. A result that carries this block can be reproduced
  or, when it cannot, the block says why (dirty tree, changed input).

* **Audit** (for existing results): :func:`completeness` grades a provenance
  dict against the fields a reproducible run needs and lists what is missing.
  The EXP-05x benchmarks recorded a commit and honesty flags but no input
  hashes and no dirty flag, so their scores are deliberately partial -- the
  report shows that rather than hiding it.

Hashing is full SHA-256 by default. Pass ``quick=True`` for multi-gigabyte
inputs; the record then says ``"algo": "quick-sha256"`` so nobody mistakes a
sampled hash for a full one.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

_PACKAGES = ("numpy", "scipy", "torch", "caveclient", "pandas", "matplotlib",
             "neuroglancer", "networkx")

#: Fields a provenance block needs before a run can be called reproducible.
#: ``weight`` is the share of the completeness score each carries.
REQUIRED_FIELDS: dict[str, float] = {
    "git_commit": 0.25,
    "git_dirty": 0.15,
    "timestamp_utc": 0.10,
    "inputs": 0.20,
    "argv": 0.10,
    "packages": 0.10,
    "synthetic_fallback": 0.10,
}


def repo_root(start: Optional[Path] = None) -> Path:
    """Nearest ancestor containing ``.git``; falls back to the package root."""
    here = Path(start or __file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".git").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


def _git(args: list[str], root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_state(root: Optional[Path] = None, max_files: int = 20) -> dict:
    """Commit, branch, and dirtiness of the working tree.

    ``git_dirty`` counts only modifications to *tracked* files, which is what
    makes a commit hash untrustworthy; untracked files are counted separately
    because this repo routinely carries uncommitted scripts alongside a clean
    tracked tree.
    """
    root = root or repo_root()
    commit = _git(["rev-parse", "HEAD"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _git(["status", "--porcelain", "--untracked-files=all"], root)
    modified: list[str] = []
    untracked: list[str] = []
    for line in (status or "").splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:]
        (untracked if code == "??" else modified).append(path)
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_dirty": bool(modified),
        "git_modified_files": modified[:max_files],
        "git_untracked_count": len(untracked),
    }


def git_commit_info(commit: str, root: Optional[Path] = None) -> Optional[dict]:
    """Date and subject of ``commit`` if the local repo has it, else None."""
    if not commit:
        return None
    root = root or repo_root()
    out = _git(["show", "-s", "--format=%cI%x1f%s", commit], root)
    if not out:
        return None
    date, _, subject = out.partition("\x1f")
    return {"commit": commit, "date": date, "subject": subject}


def hash_file(path: str | Path, *, quick: bool = False,
              chunk: int = 1 << 20) -> dict:
    """Content hash plus size and mtime of one file.

    ``quick`` hashes size + first/last 4 MiB only and labels the result so a
    sampled hash is never confused with a full one.
    """
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    with p.open("rb") as f:
        if quick and st.st_size > 8 * chunk:
            h.update(str(st.st_size).encode())
            h.update(f.read(4 * chunk))
            f.seek(-4 * chunk, os.SEEK_END)
            h.update(f.read(4 * chunk))
            algo = "quick-sha256"
        else:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
            algo = "sha256"
    return {
        "path": str(p),
        "algo": algo,
        "hash": h.hexdigest(),
        "bytes": st.st_size,
        "mtime_utc": _dt.datetime.fromtimestamp(
            st.st_mtime, _dt.timezone.utc).isoformat(timespec="seconds"),
    }


def package_versions(names: Iterable[str] = _PACKAGES) -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def capture_provenance(*, inputs: Iterable[str | Path] = (),
                       params: Optional[dict] = None,
                       root: Optional[Path] = None,
                       quick_hash: bool = False,
                       **flags) -> dict:
    """Everything needed to say what produced a result.

    ``flags`` are experiment honesty flags recorded verbatim, e.g.
    ``synthetic_fallback=False`` or ``labels_used_only_for_evaluation=True``.
    Missing input files are recorded as ``{"path": ..., "missing": True}``
    rather than raising, so a report can still be written.
    """
    root = root or repo_root()
    prov: dict = {
        **git_state(root),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": package_versions(),
        "inputs": [],
        "params": dict(params or {}),
    }
    for item in inputs:
        p = Path(item)
        if p.exists():
            prov["inputs"].append(hash_file(p, quick=quick_hash))
        else:
            prov["inputs"].append({"path": str(p), "missing": True})
    prov.update(flags)
    return prov


def stamp(payload: dict, **kwargs) -> dict:
    """Add (or merge into) ``payload["provenance"]``; returns ``payload``."""
    existing = payload.get("provenance") or {}
    payload["provenance"] = {**existing, **capture_provenance(**kwargs)}
    return payload


def write_result(path: str | Path, payload: dict, **kwargs) -> Path:
    """Stamp provenance and write JSON atomically (tmp file + rename)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stamp(payload, **kwargs)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True,
                              default=_json_default) + "\n")
    os.replace(tmp, p)
    return p


def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def completeness(prov: Optional[dict]) -> dict:
    """Grade a provenance block: score in [0, 1], present and missing fields.

    ``inputs`` only counts when at least one entry carries a hash; a bare
    path is a pointer, not provenance. ``synthetic_fallback`` is accepted at
    the top level of the block (where the EXP-05x scripts put it).
    """
    prov = prov or {}
    present, missing = [], []
    score = 0.0
    for field, weight in REQUIRED_FIELDS.items():
        value = prov.get(field)
        ok = value is not None
        if field == "inputs":
            ok = isinstance(value, list) and any(
                isinstance(v, dict) and v.get("hash") for v in value)
        elif field == "argv":
            ok = bool(value) or bool(prov.get("command"))
        if ok:
            present.append(field)
            score += weight
        else:
            missing.append(field)
    return {"score": round(score, 3), "present": present, "missing": missing}
