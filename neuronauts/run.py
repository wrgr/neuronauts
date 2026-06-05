"""Deprecated shim — the v1 agent/membrane orchestrator moved to
:mod:`neuronauts.legacy.run`.

This module belongs to the legacy (pre-tree-DNA) agent-simulation pipeline and
is **not** used by the active assembly pipeline; ``import neuronauts`` does not
load it. It is retained only so existing imports keep working while the
quarantine completes — ``from neuronauts.run import ...``, the ``neuronauts``
console script, and the v1 tests all resolve through here to
``neuronauts.legacy.run``. New code must import from
:mod:`neuronauts.legacy.run` directly. See ``docs/stage_ownership.md``.
"""

from __future__ import annotations

from .legacy import run as _run


def __getattr__(name: str):
    # Delegate every attribute (functions, constants, underscore-prefixed
    # internals) to the relocated module so existing importers are unaffected.
    return getattr(_run, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_run)))
