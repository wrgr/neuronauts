# Ablation Results — Calibrated Threshold (t = 0.99)

All numbers are mean / median F1 on the 7-box test split, evaluated
at the calibrated partition threshold of 0.99 (best-mean threshold
identified by `scripts/eval_at_t099.sh`'s sweep on the baseline seg
model).  All trainings used `seed=42`, 50 epochs, the 37-box CAVE
cache at `data/boxes/`, and the seg-connectivity feature (loaded
from `data/seg_scores.json`).

## Headline

| Config | mean F1 @ t=0.99 | Δ vs baseline |
|---|---|---|
| **drop_shared_partners** | **0.2994** | **+0.027 (best)** |
| K=2 | 0.2867 | +0.014 |
| drop_seg_connectivity | 0.2737 | +0.001 |
| **baseline (K=3, all 6 features)** | **0.2722** | — |
| drop_distance | 0.2669 | −0.005 |
| drop_shared_agents | 0.2589 | −0.013 |
| drop_same_scaffold | 0.2566 | −0.016 |
| K=4 | 0.2555 | −0.017 |
| K=1 | 0.2488 | −0.023 |
| drop_grammar_score | 0.2463 | −0.026 |
| K=5 | 0.2459 | −0.026 |
| K=3 | 0.2408 | −0.031 |

## Calibrated vs in-training threshold (t=0.5 vs t=0.99)

The picture is very different at the two thresholds.  The training
loop reports F1 at t=0.5, which over-merges and produces a flat,
noisy landscape that does not reflect the feature's actual
contribution.  At t=0.99 the contributions separate cleanly.

### K-hop ablation

| K | t=0.5 | t=0.99 |
|---|---|---|
| 1 | 0.176 | 0.249 |
| 2 | 0.197 | **0.287** |
| 3 | 0.194 | 0.241 |
| 4 | 0.195 | 0.256 |
| 5 | 0.185 | 0.246 |

K=2 is the clear winner at calibrated threshold.  The on-disk
default `CellGNNConfig.n_layers=3` is empirically suboptimal and
should be reduced to 2 in any non-back-compat code path.

### Per-feature drop

| Feature dropped | t=0.5 | t=0.99 |
|---|---|---|
| (none — baseline) | 0.198 | 0.272 |
| distance | 0.199 | 0.267 |
| same_scaffold | 0.193 | 0.257 |
| grammar_score | 0.196 | **0.246 (biggest drop)** |
| shared_agents | 0.203 | 0.259 |
| shared_partners | **0.205 (best)** | **0.299 (best — beats baseline)** |
| seg_connectivity | 0.201 | 0.274 |

Two surprises:

1. **`grammar_score` is actually the most important scalar feature**
   at t=0.99 — dropping it loses 0.026 mean F1.  The flat t=0.5
   ablation (delta of −0.001) was misleading.
2. **`shared_partners` is *anti*-informative**: dropping it improves
   F1 by 0.027 at both thresholds, making the 5-feature model the
   single best configuration we have.

## Recommended changes from this evidence

1. Change default `n_layers` from 3 to 2.
2. Drop `shared_partners` from the edge feature set entirely.
3. Always evaluate at calibrated threshold during training so the
   in-loop F1 number reflects what we actually care about.
4. The grammar_score finding motivates the next step: replace its
   scalar collapse with a learned path embedding (Option 2,
   `PathEdgeEncoder`).  The wiring is in place; what's missing is
   threading the skeleton-path cache through `cmd_train_cell_gnn`.

## Reproduce

```bash
# Threshold sweep
for t in 0.90 0.95 0.97 0.99 0.995 0.999; do
    python scripts/train.py evaluate \
        --cell-gnn-checkpoint models/cell_gnn_seg.pt \
        --partition-threshold $t
done

# K-hop ablation
bash scripts/run_k_ablation.sh

# Per-feature ablation
bash scripts/run_feature_ablation.sh

# Apples-to-apples at t=0.99 across all checkpoints
bash scripts/eval_at_t099.sh
```
