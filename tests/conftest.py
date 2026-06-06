"""Shared pytest configuration.

Tags the v1 agent/membrane *simulation* test modules — the ones that exercise
``neuronauts.legacy.*`` (the quarantined pre-tree-DNA pipeline) — with the
``legacy`` marker, so they can be excluded with ``pytest -m 'not legacy'``.

This is centralized here rather than as a ``pytestmark`` in each module to keep
the v1 test files untouched. See ``docs/stage_ownership.md`` for the quarantine.
"""

from pathlib import Path

import pytest

# Test modules whose tests drive the legacy agent/membrane/synthetic-benchmark
# pipeline (import neuronauts.legacy.{run,fields,vectorized,agent,agent_merge}).
_LEGACY_TEST_MODULES = {
    "test_run",
    "test_run_batch",
    "test_run_extras",
    "test_vectorized_extras",
    "test_fields",
    "test_scaffold",
    "test_heuristic_config",
    "test_bridge",
    "test_gat_assembly",
    "test_gat_training",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        stem = Path(item.nodeid.split("::", 1)[0]).stem
        if stem in _LEGACY_TEST_MODULES:
            item.add_marker(pytest.mark.legacy)
