# EXP-056 — real-root atomization

## Result

Simple geometry-only cuts do not provide a safe atomization operating point on
the 116 real mixed-lineage v117 roots. The experiment used actual pre- and
postsynaptic endpoint observations; target lineage was withheld during cutting
and used only for evaluation. No synthetic fallback was available.

| Rule | Pair precision | Pair recall | Cross-lineage split recall | Perfect roots |
|---|---:|---:|---:|---:|
| Atomic baseline | 0.842 | 1.000 | 0.000 | 0/116 |
| Absolute 10 um | 0.866 | 0.941 | 0.228 | 13/116 |
| Absolute 5 um | 0.906 | 0.630 | 0.652 | 8/116 |
| Edge-length q=0.99 | 0.884 | 0.783 | 0.453 | 25/116 |
| Robust median + 4 MAD | 0.895 | 0.644 | 0.599 | 15/116 |

The predeclared criterion required at least 90% same-lineage pair recall and at
least 50% cross-lineage split recall. No rule met both. The conservative 10 um
cut improves precision modestly and exactly resolves 13 roots, but leaves more
than three quarters of cross-lineage pairs joined. More aggressive cuts split
the error signal but over-fragment valid within-lineage observations.

The atomic baseline's high aggregate pair F1 (0.914) is class-imbalance driven:
it preserves all 145,014 same-lineage pairs while failing to separate any of
the 27,280 cross-lineage pairs. Exact-root recovery and split recall expose that
failure directly.

## Interpretation

This rules out a single global edge-length threshold as the atomizer. The q=0.99
result is still useful: 25 roots can be perfectly separated by removing only
their longest one percent of minimum-spanning-tree edges. A next atomization
model should target the remaining root-specific ambiguity using local caliber,
branch context, synapse role/direction, and ultimately membrane or L2 adjacency
evidence. It must be evaluated separately from cross-root joining.

The sweep is post-hoc and uses synapse-observation geometry rather than dense
voxel membranes or true L2 adjacency. It is a diagnostic, not a calibrated
deployment rule.

```bash
uv run --extra cave python scripts/benchmark_exp056_real_root_atomization.py
```
