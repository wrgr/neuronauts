# EXP-052 evaluation: proofread-soma-anchored 30 um v117 run

## Validity contract

EXP-052 used a 30 x 30 x 30 um box centered on a real changed soma in the
proofread-dense P1 region. Candidates are every v117 root at either endpoint of
every real synapse centered in the box. Target lineage was used only for box
selection, a pre-inference signal gate, and evaluation; it was not supplied to
the grammar or assembler. The gate required at least 10 true v117 fragment
pairs sharing a v1412 root and found 14 before the checkpoint was loaded. There
is no synthetic fallback.

## Population

| Quantity | Result |
|---|---:|
| Real synapses | 24,573 |
| Synapse-bearing v117 roots | 11,241 |
| Usable path roots | 1,023 |
| Singleton confusers | 10,218 |
| Exact soma seeds | 6 |
| Candidate joins | 29,985 |
| Distinct active v1412 targets | 1,009 |
| True fragment-merge pairs | **14** |
| Mixed-lineage v117 roots | **116** |

## Results

| Score threshold | Merge precision | Merge recall | ERL, um |
|---:|---:|---:|---:|
| 0 | 0.000026 | **0.929** | 5.49 |
| 1 | 0.000025 | 0.857 | 7.94 |
| 2 | 0.000024 | 0.643 | 18.39 |
| 3 | 0.000000 | 0.000 | 79.71 |
| 4-6 | 1.000000 | 0.000 | 81.34 |
| Untouched v117 | 1.000000 | 0.000 | **81.34** |

At threshold 0 the assembler recovers 13 of 14 true pairs but predicts 496,510
joined fragment pairs. One soma-owned cluster contains 997 of 1,023 active
roots. Adjusted Rand Index is -0.0000012 and circuit F1 is 0.000190. The
single-soma constraint holds, but it does not prevent one soma from claiming
nearly the entire non-soma population.

## Interpretation

This is an edit-bearing positive test, unlike EXP-051. The checkpoint is not
calibrated for blind dense adjacency: no tested global threshold separates true
continuations from the much larger false-candidate population.

Two substrate limitations remain explicit. Path geometry is a
minimum-spanning graph of real synapse endpoint observations, not complete v117
L2 geometry. Also, v117 roots are atomic, so the 116 mixed-lineage roots can be
detected but not cleaved.

