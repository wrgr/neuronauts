# superseded_modules — predecessors a consolidation replaced

**Era.** Early 2026, the v1412-materialization CAVE fetchers.

| Module | What it did |
|---|---|
| `cave_synapse_counts_v1412.py` | Per-root synapse counts against materialization v1412. |
| `cave_synapse_degrees_v1412.py` | Per-root synaptic degree against the same version. |

**Why they are here.** `neuronauts/cave_synapse.py`'s own docstring names these
as two of the three single-purpose scripts it consolidated. Nothing imports them
and no test imports them, and the numerical equivalence was checked before the
move — the audit is in [`../README.md`](../README.md).

**What replaced them.** [`neuronauts/cave_synapse.py`](../../neuronauts/cave_synapse.py),
one API instead of three scripts. `docs/consolidation_plan.md` §4.1 asks for a
further fold of the whole family behind one `data.synapses` API, with
`CLAUDE.md`'s "validate counts against a trusted query" rule as an actual test.
That fold is not done.

**Route back.** No experiment gates these — they are not a research pathway. They
return only if `neuronauts/cave_synapse.py` turns out to have lost a behavior
they had, and the equivalence check in the parent README is the thing to re-run
first.
