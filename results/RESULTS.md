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
| EXP-060 | Endpoint filter | fail | best_setting=none, best_recall=0, unfiltered_recall_ceiling=0.1748 | 2.3 | `8a148bdba` | 2026-09-02T00:06:31+00:00 |
| EXP-060 | Endpoint filter | fail | best_setting=none, best_recall=0, unfiltered_recall_ceiling=0.1748 | 2.5 | `3f8c6bb98` | 2026-09-02T00:22:22+00:00 |
| EXP-061 | Directed cone vs proximity ball | fail | best_cone=none, best_reach=0, best_median_panel=None | 4.7 | `677d3d9bf` | 2026-09-02T00:29:51+00:00 |
| EXP-061 | Directed cone vs proximity ball | fail | best_cone=none, best_reach=0, best_median_panel=None | 4.7 | `677d3d9bf` | 2026-09-02T00:35:17+00:00 |
| EXP-059 | Metric agreement | pass | n_cases=200, n_quantities_checked=11, n_disagreeing=0 | 0.0 | `06e069e0b` | 2026-09-02T00:47:43+00:00 |
| EXP-060B | Object-space atom-pair panel | fail | tier10_recall_5um=0.12, tier10_recall_2um=0.12, tier10_median_panel_5um=20 | 1.4 | `b67bd27fd` | 2026-09-02T01:23:49+00:00 |
| EXP-060B | Object-space atom-pair panel | pass | tier10_5um_recall_at_cap_20=0.12, tier10_5um_recall_uncapped=0.6457, tier10_5um_median_panel_uncapped=3870 | 1.6 | `b67bd27fd` | 2026-09-02T01:33:40+00:00 |
| EXP-060B | Object-space atom-pair panel | pass | tier10_5um_recall_at_cap_20=0.12, tier10_5um_recall_uncapped=0.6457, tier10_5um_median_panel_uncapped=3870 | 3.8 | `28039e9f3` | 2026-09-02T02:03:07+00:00 |
