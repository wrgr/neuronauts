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
| EXP-057B | ConnectomeBench2 intake | pass | decisions_mapped_to_population_atom=2392, criterion_min_decisions=1000, split_edit_before_decisions=1508 | 0.0 | `fa5db41f8` | 2026-09-02T08:27:46+00:00 |
| EXP-070 | Object vs endpoint distance | pass | control_reproduces=True, ordering_violations=0, tier10_mst_within_5um_endpoint=0.6486 | 0.1 | `fa5db41f8` | 2026-09-02T08:36:11+00:00 |
| EXP-063 | Frankenmerge detection | pass | best_feature_set=all/gbdt, best_val_auc_strict=0.9576, size_only_val_auc_strict=0.4834 | 2.0 | `8b509ef12` (dirty) | 2026-09-02T12:56:48+00:00 |
| EXP-070 | Object vs endpoint distance | error | -- | 0.2 | `8b509ef12` (dirty) | 2026-09-02T12:59:35+00:00 |
| EXP-070 | Object vs endpoint distance | pass | control_reproduces=True, ordering_violations=0, tier10_mst_within_5um_endpoint=0.6486 | 0.2 | `8b509ef12` (dirty) | 2026-09-02T13:02:32+00:00 |
| EXP-061 | Directed cone vs proximity ball | fail | best_cone=none, best_reach=0, best_median_panel=None | 5.1 | `8b509ef12` (dirty) | 2026-09-02T13:06:07+00:00 |
| EXP-071 | Contact adjacency and the connective gap | pass | n_cells=40, nearest_sibling_median_hops=3, nearest_sibling_frac_within_3=0.6008 | 2.6 | `867cba04b` (dirty) | 2026-09-02T14:26:51+00:00 |
| EXP-073 | Constrained chaining: does structure prune the panel? | blocked | -- | 0.0 | `81aec2b81` (dirty) | 2026-09-02T15:46:46+00:00 |
| EXP-072 | Object-level proposal on the widened substrate | fail | widened_chained_recall_at_bar=0.6365, population_only_chained_recall_at_bar=0.7109, gain_from_widening=-0.07445 | 10.4 | `81aec2b81` (dirty) | 2026-09-02T15:54:19+00:00 |
| EXP-063 | Frankenmerge detection | pass | best_feature_set=all/gbdt, best_val_auc_strict=0.9576, size_only_val_auc_strict=0.6538 | 2.0 | `81aec2b81` (dirty) | 2026-09-02T17:08:53+00:00 |
| EXP-074 | Soma-seeded growth, distance only | fail | recovery_at_bar=0.00431, purity_at_bar=0.000609, abstention_at_bar=0 | 15.1 | `2c92d4efd` (dirty) | 2026-09-02T20:35:59+00:00 |
