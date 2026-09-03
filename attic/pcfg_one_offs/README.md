# pcfg_one_offs — single-run probes from the probabilistic context-free grammar thread

**Era.** July–August 2026, inside `experiments/pcfg/`.

**What it is.** Fifteen command-line scripts, each written to answer one question
once and recorded in
[`experiments/pcfg/FINDINGS_synapse_correction.md`](../../experiments/pcfg/FINDINGS_synapse_correction.md).
`docs/consolidation_plan.md` §4.2 marks exactly this group ATTIC: "the ~20 one-off
`*_merge.py` / `*_cut.py` / `seam_*` scripts."

| Script | The one question it asked |
|---|---|
| `ablate_merge.py` | Which features carry the merge signal? |
| `close_loop_merge.py` | Does closing the loop on merges beat scoring them open-loop? |
| `continuation_merge.py` | Does a continuation prior improve the merge decision? |
| `cut_report.py` | What do the cut decisions look like, written out? |
| `grammar_regime.py` | Which grammar regime (alphabet, order) fits the token stream? |
| `group_eval.py` | How does the model score under grouped-by-cell cross-validation? |
| `join_corrector.py` | Can a corrector repair the join side specifically? |
| `learned_grammar_neural.py` | Does a neural grammar beat the counted bigram grammar? |
| `recursive_corrector.py` | Does applying the corrector recursively converge? |
| `seam_hash.py` | Does a hash of the seam re-identify partners? |
| `seam_ssl.py` | Does self-supervision on seams help the detector? |
| `selfsup_grammar.py` | Can the grammar be learned self-supervised? |
| `skel_ssl_grammar.py`, `skel_ssl_grammar_v2.py` | Same, over skeleton tokens rather than synapse tokens. |
| `synapse_grammar_ar.py` | Does an autoregressive synapse grammar beat the bigram one? |

**Why exactly these fifteen.** Each satisfies all four tests: no module remaining
in `experiments/pcfg/` imports it; it is not in that package's README `Files`
table or any documented `python -m` command line; no test imports it; and nothing
in `neuronauts/`, `scripts/`, `tests/` or `run_bigdata.sh` references it. Each was
imported from its new location after the move to prove it still resolves.

**They still run.** Every script keeps its `from experiments.pcfg.<core> import …`
lines, and those resolve because the core stayed put. Nesting depth is unchanged,
so the `sys.path.insert(…, parents[2])` bootstrap at the top of each file still
points at the repository root:

```
python -m attic.pcfg_one_offs.grammar_regime --help
```

**What stayed in `experiments/pcfg/`** — the thread's core and its documented
entry points: `pcfg_partitions.py`, `synapse_correction.py`, `skeleton_tokens.py`,
`learned_grammar.py`, `run_experiment.py`, `run_synapse_correction.py`,
`v117_pcfg.py`, `fetch_bigdata.py` (driven by `run_bigdata.sh`),
`close_loop_cut.py`, `seam_detector.py`, `skeleton_cut_op.py`,
`atomicity_detector.py`, `skeleton_topology_merge.py`, and two modules the live
package depends on — `conn_metric.py` (cited by
`neuronauts/report/tracker.py` as a metric shim) and `global_shape_merge.py`
(imported by `neuronauts/harness/atom_features.py`).

**What replaced them.** The cross-region holdout result these probes fed into
(`experiments/pcfg/HOLDOUT_RESULTS.md`, AUC 0.816) stands, but is graded
**semi-synthetic**: real skeletons and synapses, synthetically introduced break
points. The registered replacement is EXP-064, which scores every candidate signal
— grammar included — on one fixed panel.

**Route back.** EXP-064. A grammar scorer returns by winning that bake-off.
