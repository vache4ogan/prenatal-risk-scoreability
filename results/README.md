# Results

Only small article-facing tables are committed. Models, predictions, replicate-level bootstrap output, and reproducibility artifacts remain in ClearML or artifact storage. The canonical historical ClearML task is `29717840f05144579af240f6ded4852a`.

- `singleton_fixed_budget_results.csv`: policy-level primary singleton results.
- `singleton_fixed_budget_comparisons.csv`: primary equal-budget policy contrasts.
- `paired_bootstrap_summary.csv`: paired full-size bootstrap confidence intervals.
- `mechanism_table_10pct.csv`: group-level mechanism counts at 10%.
- `mechanism_table_10pct_comparisons.csv`: mechanism contrasts at 10%.
- `lightgbm_calibration_metrics.csv`: raw and calibrated probability metrics.
- `lightgbm_platt_calibrators.csv`: fitted calibrator parameters.
- `lightgbm_raw_vs_calibrated.csv`: calibration comparison; calibration did not improve the primary LightGBM.
- `lightgbm_reliability_curve_points.csv`: reliability-curve bins.
- `all_births_fixed_budget_results.csv`: all-birth sensitivity results.
- `all_births_fixed_budget_comparisons.csv`: all-birth sensitivity contrasts.
- `article_revision_summary.md`: human-readable canonical summary.

The singleton fixed-budget outputs are primary. All-birth outputs are sensitivity only; calibration and mechanism tables are supporting analyses. Older parent-repository result folders are legacy and are not part of the primary article estimand.
