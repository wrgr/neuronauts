# ⚠️ RETRACTED — every paper artifact in this directory

**Do not submit, circulate, or cite any file in `docs/paper/` or `docs/latex/`.**

The results reported in these manuscripts derive from EXP-020, EXP-023,
EXP-025, EXP-026 and EXP-035. All of them ran on a pipeline that:

1. **manufactured its own fragments** by bisecting real proofread skeletons into
   equal thirds and calling the thirds "v117 segments" — real v117 structure is
   "one trunk + slivers" (88% of somata are already a single v117 root);
2. **fabricated its own synapses**, with partner IDs assigned as
   `partner_base = obj_counter * 100`, making synaptic partner overlap a
   deterministic function of the ground-truth neuron identity;
3. **used that leaked identity as a scoring feature** with weight 3.0;
4. **scored "micro-EM verification" without reading any EM** — the sampler takes
   the ground-truth label as an argument and returns a Gaussian conditioned on
   it;
5. **ran an untrained model** — the "Tree-Grammar Transformer" is random
   matrices; no checkpoint is ever loaded;
6. compared against **RNG stubs** standing in for AutoProof and NEURD.

The claimed "strict 3-way inductive protocol (Train 60% / Val 20% / Held-Out
Test 20%)" is not implemented. The validation set is never materialized in code
(`n_val` only offsets the test slice; no `val_pieces` variable exists), and the
same seed-42 population and its "held-out" test slice were reused across ~28
experiments.

Specifically retracted: the SOTA comparison table, merge precision 0.75,
synapse/line-graph precision 95.4% and 99.1%, ERL 3.37–3.60 mm, 556,799 TP
edges, "88.33% Top-3 / 46.67% Top-1 / 60.55% circuit recall / 482k synapses",
"75% Top-3 / 81.46% path precision", and the reported confusion matrices.

Full audit: [`../synthetic_data_audit_and_dataset_plan.md`](../synthetic_data_audit_and_dataset_plan.md).
Frozen defective code: [`../../quarantine/README.md`](../../quarantine/README.md).

Neither manuscript disclosed that fragments, synapses, and frankenmerges were
generated; the word "synthetic" does not appear in either. If either was shared
outside the project, it needs an explicit correction notice — that is an open
decision for the project owner.
