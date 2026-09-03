# superseded_training — training entry points with no real-data result

**Era.** April–June 2026: the shared-grammar / graph-attention track and the
topology-validator track.

| Script | What it trained | Status |
|---|---|---|
| `train_shared_grammar.py` | `SharedGrammarModel` — the pairwise merge scorer behind the "~85–87% merge accuracy" figure. | That figure is an **in-sample cross-validation number on a curated candidate panel**, and the harder later test contradicts it: EXP-053A found *no* checkpoint separates real continuation pairs from dense confusers. `docs/threads/experiment_survey.md` Part 3 item 10 says not to quote it as a real-data result. |
| `train_topology_model.py` | `neuronauts/topology_model.py`, the atomicity validator that flags clusters formed by merging two roots. | `docs/threads/topology.md` calls the thread "optional… smoke only." No checkpoint tracked, no real-data result recorded. |
| `export_topology_dataset.py` | The dataset the above consumes. | Moves with its trainer. |
| `inspect_topology_metric.py` | A one-off inspector for the same metric. | Nothing references it. |

**What replaced them.** The scoring question moved into the registered program:
EXP-064 (fixed-panel scorer bake-off) and EXP-065 (ablation) are the declared
home for "which signal separates true continuations," and both are still unrun
because candidate *generation* has not cleared its bar. The atomicity question is
EXP-062 (real level-2 cuts and seam location), also unrun.

**Note on what did not move.** The model code itself —
`neuronauts/topology_model.py`, `neuronauts/shared_grammar_model.py`,
`neuronauts/grammar.py` — is still in the package and still imported by tests.
Only the training command-line entry points moved. `docs/consolidation_plan.md`
§4.1 marks those modules SPLIT/ATTIC, which is package surgery and a separate
change.

**Route back.** EXP-064 for the grammar scorer; EXP-062 for the topology
validator. A checkpoint earns its way back by winning one of those, not by being
imported again.
