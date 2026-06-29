# Project principles — read before modeling

## Features: LEARN them, do NOT hand-build them (hard rule)

The models here must **learn their representations from raw inputs**. Do **not** hand-engineer
features and feed them to a model.

- **Legal (raw inputs):** data fields that exist on the entity — vertex `xyz`, skeleton `radius`,
  a synapse's `pre`/`post` side, the raw synapse count attached to a vertex. Splitting a synapse
  count into its raw pre/post channels is raw input, not a feature — that is fine and encouraged.
- **NOT legal (hand-engineered features):** anything *derived* that the model could compute itself
  — caliber gradients, turn/branch angles, tortuosity, collinearity, polarity-contrast ratios,
  local densities, kNN statistics. A GNN over the skeleton graph is meant to learn exactly these
  via message passing. Precomputing them is the anti-pattern.
- If a derived quantity seems necessary, the answer is **more model capacity / depth or a
  self-supervised objective**, not a hand-coded feature. When in doubt, ask — do not add the feature.
- Prefer **self-supervised** objectives that can train on the **noisy raw segmentation** (v117)
  without proofreading labels; labels (v1718) are for *evaluation/targets*, not as the only signal.

(This rule was set explicitly and violated once by adding kink/caliber-gradient features to the
seam GNN — which also did not work. Don't repeat it.)

## Evaluation: the do-nothing guardrail, pre/post explicit

Always evaluate corrections against the **do-nothing baseline** (net within-object pair-error /
Rand-disagreement reduction), and always **pre/post explicitly**: a synapse connects a pre (axon)
cell to a post (dendrite) cell, so report pre-side and post-side separately plus connectivity
(a synapse is correct iff BOTH sides land in the right cell). Pooled-only or AUC-only numbers have
been mirages here. Report #splits and #merges applied. Shared util: `conn_metric.py`.

## Environment gotchas (remote/ephemeral)

- The container is **reclaimed on inactivity**; background jobs die between turns and the agent
  proxy changes port. Things worth keeping must be **committed/pushed** or checkpointed to `data/`
  (which persists across reclaims). Re-derive the live proxy port from `/root/.ccr/README.md`.
- **Never** `pgrep -f` / `pkill -f` a pattern that also appears in your own shell command — it
  matches and SIGTERMs your own shell. Use explicit PIDs or a pidfile.

## Where things are

Error-and-correction model lives in `experiments/pcfg_synapse_partitions/`; running findings in
`FINDINGS_synapse_correction.md`. Data (gitignored): `data/sidetable_*.npz`, `data/skel_v117`,
`data/skel_v1718`.
