# tests — the test that moved with its subject

`test_morpho_grammar.py` exercises the 26 engines in
[`../morpho_grammar/`](../morpho_grammar/README.md). It moved here with them so
the default suite stays clean: `pyproject.toml` sets `testpaths = ["tests"]`, so
nothing in this directory is collected.

**Why its subject is archived.** No engine loads a trained checkpoint — 25 of 26
contain no `torch.load` or `.pt` path at all — and every benchmark that scored
them built its test world by synthetically cutting and frankenmerging real
skeletons at 45%. The reported accuracies are those of randomly initialized
models on fabricated damage.

**What replaced it.** Nothing here: this is a unit test for retired code, not a
capability. The live suite is `tests/` at the repository root.

**Route back.** With its subject, and only through **EXP-069** — the registered
experiment that asks whether any archived engine earns its numbers back on the
real harness substrate under the EXP-064 fixed-panel protocol.
