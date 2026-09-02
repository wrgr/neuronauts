# Results ledger

One row per experiment run, appended by `neuronauts.experiments._runner`.
**A run without a row does not exist.** `blocked` means a prerequisite had no
passing result, so no metrics were computed -- that is a recorded outcome, not
a gap. Full reports: [`results/reports/`](reports/).

| ID | Title | Status | Headline | Min | Commit | When (UTC) |
|---|---|---|---|---:|---|---|
| EXP-057 | GT overlay and spatial split | fail | mass_frac_pure_proofread=0.1619, mass_frac_pure_gold=0.06158, n_seam_positives=56 | 0.0 | `06e8b44a6` | 2026-09-01T23:47:31+00:00 |
| EXP-058 | Baseline ladder | pass | oracle_ari=1, best_proximity_ari=0, random_ari=-1e-06 | 1.6 | `95d67c25a` | 2026-09-02T00:00:11+00:00 |
| EXP-058 | Baseline ladder | pass | oracle_ari=1, best_proximity_ari=0, random_ari=-1e-06 | 0.2 | `95d67c25a` | 2026-09-02T00:01:21+00:00 |
