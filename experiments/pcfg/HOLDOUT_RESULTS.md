# Cross-region held-out evaluation of the learned skeleton–synapse grammar

**Question.** Can a *data-driven learned grammar* (not hand-crafted features)
reassemble fragmented neurons — and does it generalise to tissue it has never
seen? CV within one box cannot answer this: pieces of the same cut neuron leak
across folds. So we train in one region and test in a spatially **disjoint**
one.

**Answer (MICrONS `minnie65_public`, v117, 60 µm boxes):**

> **Cross-region held-out ROC-AUC = 0.816, 95% CI [0.754, 0.874]** (pure grammar,
> no distance feature; epoch chosen on a third disjoint region). Robust across
> seeds (0.78–0.82). At a deployable threshold it recovers **60 % of the true
> cross-region merges (39/65) vs 0 for doing nothing** (F1 0.57 vs 0.00).

---

## The task — self-supervised split → merge

1. Take a dense volume; combine each neuron's pre- and post-synaptic sites.
2. **Fragment** each neuron with *realistic* breaks: cut at thin necks and
   branch points, drop a ~1.5 µm connector at each cut (the lost piece that
   makes closest-point distance evidence rather than a giveaway), and keep only
   fragments with **≥ K = 5 synapses**.
3. **Reassemble** with the learned grammar (`SkeletonSynapseNet`): each
   inter-synapse skeleton path is encoded (`PathEdgeEncoder`), the sequence of
   path tokens is encoded into a fragment embedding (`TorchPathEncoder`), and an
   MLP scores `concat(emb_A, emb_B, |emb_A−emb_B|, log_dist)` → merge probability.
   - **Positives** = fragments adjacent across a single break (a real
     reconnect-the-break candidate).
   - **Negatives** = spatially nearest fragments of *different* neurons.

No proofreading is needed — the supervision is the synthetic cut itself.

## The three-region protocol

| Region | Center (nm) | Offset | Role |
|--------|-------------|--------|------|
| **A** train | (733592, 513592, 595640) | 0 | fit the model |
| **B** test  | (933592, 513592, 595640) | +200 µm x | held-out AUC (never touched in training/selection) |
| **C** val   | (583592, 513592, 595640) | −150 µm x | **select the training epoch** |

All three are mutually disjoint (60 µm boxes, ≥90 µm gaps), same cortical layer,
density-matched. Region C exists because a validation split taken from region A
*overfits together with training* — its AUC keeps rising while true cross-region
AUC falls, so it cannot detect region-specific overfit (see findings). An
independent region can, and selecting on it never peeks at B.

Run it:

```bash
python scripts/v117_pcfg.py --token "$CAVE_TOKEN" \
  --side-um 60 --synapse-source bulk \
  --synapse-cache-dir syn_cache --skeleton-cache-dir skel_cache \
  --use-learned --no-learned-use-distance \
  --eval-offset-um 200 --val-offset-um=-150 \
  --max-neurons 0 --learned-epochs 30 \
  --checkpoint holdout.pt
```

## Results

Pure morphology grammar (distance feature zeroed), 60 µm:

| Method | Evaluation | ROC-AUC | 95% CI |
|--------|------------|---------|--------|
| Chance | — | 0.50 | — |
| Hand-crafted **bigram features** (F/B/L/R) | within-region CV *(leaky, easier)* | 0.56–0.63 | — |
| **Learned grammar** | held-out, same-region val' selection | 0.70 | [0.63, 0.77] |
| **Learned grammar** | **held-out, 3rd-region selection** | **0.82** | **[0.75, 0.87]** |

Train = region A, 1048 neurons → 900 pairs (225 pos). Test = region B, 590
neurons → 260 pairs (65 pos / 195 neg). Val = region C, 439 neurons → 188 pairs.
The learned grammar wins **despite** facing the harder cross-region test while
the bigram baseline gets the easier (leaky) within-region CV.

### vs. doing nothing (deployable merge decision)

Threshold chosen on region C (F1-max), applied to region B:

| | Precision | Recall | F1 | Accuracy | Merges |
|--|-----------|--------|----|----------|--------|
| **Learned grammar** | 0.55 | **0.60** | **0.57** | 0.78 | 39 correct / 32 wrong |
| **Do nothing** | 1.00 | 0.00 | 0.00 | 0.75 | 0 |

The grammar recovers **60 % of true cross-region merges** where doing nothing
recovers none (**ΔF1 = +0.57**). Note "do nothing" already scores **75 %
accuracy** — because 75 % of candidate pairs are genuine non-merges — so
accuracy is the wrong metric; recall/F1 on the *merge* class is the honest one,
and there doing nothing is worthless by construction.

## Key methodological findings

1. **Distance is a *backwards* confound in dense tissue.** Across-break positives
   sit at median **4.9 µm** (the dropped connector), but the nearest
   different-neuron negative is at **0.8 µm**. So "distance-only" scores ~0.93 —
   *in the wrong direction* (far ⇒ same neuron). Using distance as a feature
   teaches the deploy-time opposite of the truth, so the honest test zeroes it
   and lets the morphology grammar decide.

2. **Same-region validation cannot detect cross-region overfit.** A val split
   from the training region keeps improving (AUC 0.76 → 0.89) while the true
   cross-region AUC *falls* — they overfit region-specific structure together.
   An independent region C **tracks** the test region (both peak ~epoch 10 and
   decline together), so it is a faithful selector. This is *why* the 3rd region
   matters: same-region selection reported 0.70; 3rd-region selection 0.82.

3. **Overfit is a small-data artifact.** At 40 µm only ~13 % of usable neurons
   cut into adjacent ≥5-synapse fragments → 404 train / 96 eval pairs, the eval
   curve peaks early (~0.75) then decays to ~0.66. At 60 µm (900 train / 260
   eval pairs) the decay is mild and the held-out estimate is stable with a
   tighter CI.

## Caveats

- **Skeleton coverage is the operating domain.** Only ~10–23 % of randomly
  sampled roots have a usable v117 skeleton (≥10 vertices); the grammar can only
  be applied to fragments that do. Root sampling is **random** (representative),
  not biggest-first — a biggest-neuron subset can't be reproduced at deploy and
  inflates the estimate.
- **Positive scarcity → finite CI.** 65 held-out positives give CI ≈ ±0.06;
  larger boxes / more regions would tighten it further.
- **Seed sensitivity.** Held-out AUC varies 0.78–0.82 across seeds (model init +
  which epoch region C selects); `torch.manual_seed(seed)` makes a given run
  reproducible.

## Code

- `experiments/pcfg/learned_grammar.py`
  - `train_and_eval_holdout(train, eval, val_partitions=…)` — three-region train
    / select / test, bootstrap CI, merge-decision vs do-nothing, checkpoint.
  - `_fragment_neuron`, `build_training_data` — realistic split → merge data.
- `scripts/v117_pcfg.py` — flags `--eval-offset-um`, `--val-offset-um`,
  `--checkpoint`, `--no-learned-use-distance`, `--max-neurons` (random sampling).

Checkpoints (`torch.save`, kept out of the repo as binaries) store the
early-stopped weights plus a `meta` block with the AUC, CI, selection method,
and merge-decision metrics.
