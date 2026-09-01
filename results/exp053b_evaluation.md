# EXP-053B — real-L2 candidate-panel recall

## Result

The experiment failed its prerequisite coverage gate. Of 1,023 eligible v117
roots selected from the real synapse table, only 284 (27.8%) had at least two
bounded L2 representative coordinates in the 30 um box plus a 10 um halo. Only
one of the 14 true two-root lineages had L2 geometry on both sides. That one
pair was not proposed at any tested radius/cone setting, including the fully
open 10 um / 180 degree panel.

Therefore this is **not evidence that L2 endpoint geometry cannot recover real
continuations**. It is evidence that the current bounded v117 L2 retrieval path
does not provide a valid positive panel for this experiment.

## Population and controls

- Real synapses: 24,573
- Eligible synapse-bearing v117 roots: 1,023
- L2-covered roots: 284
- Endpoint paths: 2,178
- True target lineages with two eligible v117 roots: 14
- True pairs with both roots L2-covered: 1
- Synthetic fallback: disabled
- Target lineage used during candidate generation: no
- Root universe restricted to roots appearing in the synapse table: yes

The fourteen positives are fourteen independent two-root target lineages, not
all-pairs combinations inside larger fragment groups. The pairwise target is
therefore appropriate here.

## Panel sweep

All 30 radius/cone configurations recovered 0/14 true pairs. Representative
panel sizes were:

| Radius | Cone | Candidate pairs | Median panel | P90 panel | Recall |
|---:|---:|---:|---:|---:|---:|
| 2.5 um | 180 deg | 412 | 3 | 5 | 0 |
| 5 um | 90 deg | 1,207 | 8 | 13 | 0 |
| 10 um | 60 deg | 2,309 | 16 | 24.7 | 0 |
| 10 um | 180 deg | 15,292 | 106 | 146.7 | 0 |

The predeclared criterion—at least 90% total recall with median panel size at
most 20—failed.

## Retrieval diagnostic

A label-blind sample of ten bounded misses was sent through the unbounded
complete-root L2 route and clipped back to the same halo. The probe did not
complete after more than 14 minutes and was stopped. At this API behavior, that
route is not a practical 1,023-root dense substrate without a bulk L2 graph
export or a persistent precomputed cache.

## Consequence for the sequence

EXP-054 requires a fixed candidate panel containing enough positives to compare
scorers. With zero recovered positives and only one geometrically covered true
pair, a scorer bake-off would be undefined or misleading. EXP-054 must therefore
record a prerequisite-gate failure rather than report model performance.

```bash
uv run --extra cave --extra topology \
  python scripts/benchmark_exp053b_l2_candidate_panel.py --workers 8
```

The aggregate grid is `results/exp053b_l2_candidate_panel.json`. Root-level L2
and candidate caches remain local and are excluded from git.
