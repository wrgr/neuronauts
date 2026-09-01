"""Registered experiments: one module per EXP, run under a common discipline.

See :mod:`neuronauts.experiments._runner` for the contract. Each experiment
declares a ``Spec`` (id, question, predeclared criterion, prerequisites,
inputs) and a ``run`` function returning an ``Outcome``; the runner refuses to
report metrics when a prerequisite is unmet, stamps provenance, and appends a
row to ``results/RESULTS.md``.
"""

from neuronauts.experiments._runner import (
    Context, Outcome, Spec, append_ledger, check_inputs, check_prerequisites,
    load_result, main, result_path, run_experiment,
)

__all__ = [
    "Context", "Outcome", "Spec", "append_ledger", "check_inputs",
    "check_prerequisites", "load_result", "main", "result_path",
    "run_experiment",
]
