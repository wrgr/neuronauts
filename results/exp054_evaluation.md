# EXP-054 — fixed-panel scorer bake-off

## Result

Status: **prerequisite failed; no scorer comparison run**.

EXP-054 required at least 10 L2-covered positive pairs and at least 90% recall
from the fixed, label-blind EXP-053B candidate panel. Observed values were one
covered positive and zero recall. A panel with no recovered positives cannot
estimate scorer discrimination, ranking, precision-recall behavior, or
calibration.

No distance, tangent, grammar, root-neighborhood, or combined scorer metric is
reported. This is deliberate fail-closed behavior, not a missing run.

```bash
python scripts/benchmark_exp054_fixed_panel_scorers.py
```
