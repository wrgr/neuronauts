# EXP-053A — fixed-population checkpoint bake-off

## Result

No existing checkpoint separates real continuation pairs from dense spatial
confusers on the EXP-052 population. Permissive thresholds recover most of the
14 real merge pairs while collapsing hundreds of unrelated roots; the first
threshold that avoids collapse recovers no real merge pair.

| Checkpoint | Last high-recall threshold | Recall | Precision | Largest cluster | First non-collapse threshold | Recall there |
|---|---:|---:|---:|---:|---:|---:|
| raw skeleton | 0 | 0.929 | 0.000026 | 997 | 3 | 0 |
| raw skeleton + GAT | 0 | 0.929 | 0.000026 | 999 | 4 | 0 |
| legacy real | 0 | 0.786 | 0.000023 | 979 | 3 | 0 |
| root neighborhood | 0 | 0.929 | 0.000027 | 981 | 1 | 0 |

The raw-skeleton checkpoint at threshold 3 makes only two joins, both false.
The root-neighborhood checkpoint at threshold 1 makes 15 pairwise joins, all
false. At abstention, the untouched-v117 baseline is restored: expected run
length (ERL) 81.34 um and circuit F1 0.9868, with zero merge recall.

## Controls

- Real P1 v117 roots in the edit-anchored 30 um box: 11,241
- Synapses: 24,573
- Active path roots: 1,023
- Soma seeds: 6
- Real merge pairs under v1412 lineage: 14
- Mixed-lineage v117 roots, unsplittable in this experiment: 116
- Candidate edges per checkpoint: 29,985
- Synthetic fallback: disabled
- Ground-truth lineage used by inference: no

The common failure across feature modes points upstream of checkpoint choice.
The current candidate graph is built from minimum-spanning trees (MSTs) of
observed synapse endpoints, not dense L2 geometry. Soma exclusion prevents
multi-soma clusters but does not prevent erroneous chaining among non-soma
roots. EXP-053B therefore tests real L2 candidate-panel recall before assembly.

```bash
uv run --extra cave --extra topology \
  python scripts/benchmark_exp053a_checkpoint_bakeoff.py
```

The machine-readable aggregate is `results/exp053a_checkpoint_bakeoff.json`.
Thresholds are post-hoc diagnostics, not calibrated operating points.
