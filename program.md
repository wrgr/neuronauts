# neuronauts autoresearch

This repo is set up for autonomous iteration in the style of `autoresearch`.

The important idea is simple:

- the benchmark harness stays small and mostly fixed
- the human edits `program.md`
- the agent edits `neuronauts/run.py`
- each experiment runs for a fixed wall-clock budget
- changes are kept only if the metric improves

## Setup

Before starting a new run, work with the human to:

1. Confirm the repo is in a clean enough state to start iterating.
2. Read these files for full context:
   - `README.md`
   - `program.md`
   - `neuronauts/fetch.py`
   - `neuronauts/run.py`
3. Verify the environment works:
   - `python -m unittest discover -s tests -p 'test_*.py'`
   - `python -m neuronauts.run --cases 5 --benchmark-mode fixed_validation --quiet`
4. Confirm setup looks good.

## Scope

Treat the repo as having three conceptual parts:

- Fixed prep / eval / utilities:
  - `neuronauts/fetch.py`
  - `neuronauts/fields.py`
  - `neuronauts/line_graph.py`
  - `neuronauts/vectorized.py`
  - `tests/`
- Agent-edited experiment surface:
  - `neuronauts/run.py`
- Human-edited research instructions:
  - `program.md`

Default rule: only edit `neuronauts/run.py`.

Only widen beyond that if the human explicitly asks for structural work or you discover a benchmark bug that makes the search invalid.

## Goal

Optimize connectome recovery quality on real MICrONS boxes.

Primary scalar:

- `val_f1` from `python -m neuronauts.run --data-mode real`

Secondary diagnostics:

- precision
- recall
- TP / FP / FN

The real objective is not a lucky single run. The real objective is stronger average behavior over repeated 5-minute iterations.

## What the agent may do

- edit `neuronauts/run.py`
- change hyperparameters
- change merge / ownership / assignment logic inside `neuronauts/run.py`
- run fixed-validation checks
- run 5-minute real-data iterations
- use synthetic mode only for smoke tests and debugging
- keep or discard changes based on results

## What the agent should avoid

- adding unnecessary complexity to the benchmark harness
- spreading the editable surface across many files
- changing the scoring definition unless the existing scorer is broken
- changing the outer-loop philosophy away from fixed-time experiments

## The loop

The outer optimizer should think in 5-minute iterations.

For each iteration:

1. Propose one experiment idea.
2. Edit `neuronauts/run.py`.
3. Run the regression test.
4. Run a quick fixed-validation comparison.
5. If promising, run a 5-minute real-data iteration.
6. Keep the change only if the iteration-level metrics improved enough to justify the complexity.

Repeat until the human stops the process.

## Simplicity rule

All else being equal, simpler is better.

Keep changes that:

- improve mean F1
- improve precision / recall tradeoff clearly
- remove code while preserving performance

Reject changes that:

- add complexity for negligible gains
- improve only a lucky batch but hurt the mean
- expand the editable surface without necessity

## Current benchmark policy

- primary target: real MICrONS boxes
- use a small fixed candidate pool of `~6 x 6 x 6 um` boxes
- require at least `50` synapses in a box for it to count
- evaluate multiple boxes per run for robustness
- keep synthetic mode only as a smoke-test / regression path

## Current real-data convention

Use approximately a `6 x 6 x 6 um` cube:

- `bbox_nm` side length: `6000 nm`

Validation policy:

- use the same real validation boxes every iteration for apples-to-apples comparison
- include a few additional candidate boxes in reserve
- if a candidate box has fewer than `50` synapses, skip it and move to the next one

## Important note

The loop runner in `scripts/iterative_loop.py` is only the benchmark harness.

It is not the optimizer.

The optimizer is the external Codex session that reads `program.md`, edits `neuronauts/run.py`, runs 5-minute iterations, and decides whether to keep or discard changes.
