# CLAUDE.md — working agreement for agents in this repo

## 0. The most important rule: assume the bug is yours

When something fails — a timeout, an error, an empty result, a wrong number,
slowness — **assume your own code, call, or usage is wrong before blaming
anything external** (the library, CAVE, the network, the egress proxy, the
container, the server, the data, "rate limiting", "environmental variance").

Empirically, in this project it is the agent's mistake ~99% of the time.
Confidently blaming an external system when the fault is yours wastes hours and
erodes trust. Treat "it's the environment" as a claim that requires proof, not a
default explanation.

### Banned behavior (this is the thing we keep hitting)
- ❌ Declaring an external system "broken / down / degraded / rate-limited /
  too slow / environmental / blocking us" **without a minimal reproduction that
  also rules out your own usage.**
- ❌ Certainty words you haven't earned: "conclusive", "definitively",
  "confirmed", "it's clearly X", "this proves". If you haven't proven it, say
  "I haven't ruled out ___" instead.
- ❌ Calling something "working" because it *returned something*. It is not
  working until the **output is verified correct** (right count, right values,
  compared to a known-good reference).
- ❌ Stacking workarounds (`bounded=False`, longer timeouts, more retries,
  `try/except` that swallows) to get past a failure you don't understand. A
  workaround that hides an unexplained error usually hides *your* bug.

### Required behavior instead
1. **Read before you blame.** Re-read the actual API signature / docs / your own
   call. Most "the tool is broken" moments are a wrong argument, wrong source,
   a forgotten flag, or a units/coordinate mistake. (Real examples from this
   repo: wrong segmentation source; forgetting `agglomerate=True`; wrong query
   chunk size silently changing results; a `pkill` pattern that matched its own
   command.)
2. **Minimal repro or it didn't happen.** Before claiming an external cause,
   write the smallest call that isolates it *and* shows your code is not the
   cause. If you can't produce that, you may not assert the external cause —
   report "still investigating; here's what I've ruled out."
3. **Change one thing at a time and verify it** before changing the next. No
   shotgun debugging.
4. **Verify correctness against ground truth**, not vibes. If a fast path and a
   slow/known path disagree, the fast one is suspect until proven equal.
5. **Check your own command for self-sabotage** (a kill that matches its own
   process, a glob that deletes your output, a cwd assumption, a stale cache).
6. **When stuck, escalate honestly.** Say "I have not found the cause; here is
   what I tried and ruled out, and what I'd check next" — never invent an
   external culprit to close the loop.

### A useful phrasing test
Before you write "the problem is <external thing>", ask: *"Have I proven my own
code is correct here, with a check I could show?"* If no, don't write it.

## 1. Other standing habits
- Don't present unverified results as results. Label estimates, partial runs,
  and unproven claims as such.
- Prefer the smallest experiment that answers the question over a big run.
- Keep secrets (tokens) out of files and commits; pass via env only.
- Clean up scratch files; don't commit them.

## 2. Project context
- `docs/roadmap_global_assembly.md` — canonical pipeline overview and
  direction (supersedes `docs/archive/2026-09/program.md`, see
  `docs/consolidation_plan.md` §4.4).
- `README.md` — project summary and setup.
- `experiments/pcfg_synapse_partitions/README.md` — PCFG synapse-partition
  experiment.
- Data access goes through CAVE (`neuronauts/fetch.py`). The synapse table
  `synapses_pni_2` is ~337M rows; unfiltered spatial queries are heavy and the
  MICrONS docs recommend filtering by root id. Any region/synapse fetch helper
  must have its **counts validated against a trusted query** before being relied
  on.
