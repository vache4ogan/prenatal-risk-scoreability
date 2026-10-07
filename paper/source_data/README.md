# Source-data notes

This directory contains aggregate, non-record-level inputs for the manuscript generators.

- `cohort_counts_validation.csv`: target-specific CDC 2022/2023 cohort flow and event counts.
- `model_diagnostics.csv`: frozen LightGBM ranking diagnostics.
- `canonical_protocol_results.csv`: CDC 2023 candidate-set and policy results.
- `shapley_main_results.csv`: six point-estimate/interval rows for three outcomes and q in {5%, 10%}.
- `shapley_bootstrap_replicates.csv`: 500 paired bootstrap replicates per outcome and budget.
- `replication_2024_results.csv`: corrected CDC 2024 workload at the full-precision CDC 2022 q=10% score thresholds, including exact counts and arithmetic fields.
- `temporal_2024/`: complete frozen-model replication: 24 four-cell estimates, six decomposition summaries, 3,000 paired-resample rows, full-precision and rounded threshold transport, and `run_manifest.json` with provenance and hashes.
- `sensitivity_no_priorterm.csv`: complete q=5%/10% Shapley results after removing prior other pregnancy outcomes.
- `sensitivity_no_race.csv`: q=10% comparison after removing maternal race; this supporting run has point estimates and no bootstrap interval.
- `synthetic_figure3_data.csv`: complete 81-condition mechanism grid summary.
- `synthetic_manifest.json`: locked simulation design, provenance, and invariant results.
- `figure1_headline_data.csv` and `figure_manifest.json`: derived by `generate_figures.py`.

The reviewed threshold-only CDC 2024 calculation represented numeric maternal-race codes as strings. The frozen encoder therefore treated them as unknown categories. The camera-ready results supersede that calculation with numeric codes consistent with CDC 2022 model fitting. The corrected full-precision selected counts are 372,390 / 373,434 / 358,394 for preterm / NICU / low birth weight; the reviewed counts were 373,946 / 333,413 / 291,726. Using six-decimal thresholds on the corrected scores changes the full-precision counts by +1 / -8 / 0, respectively: rounding does not explain the correction.

The complete replication freezes the 2022 models, preprocessing, and thresholds. Its four-cell audit holds the score vector fixed, uses 2024 cohort-derived budgets, and conditions paired bootstrap intervals on that vector and the observed budgets. Replaying the same pipeline on CDC 2023 reproduced all 24 original four-cell estimates exactly. Threshold-transport workload changes are descriptive and are not assigned to a particular drift mechanism.

The value `temporal_test_2023` in the archived canonical CSV is a legacy run label. In the manuscript, CDC 2023 is described as the temporal evaluation cohort because the audit hypothesis was developed and evaluated on these scores; model fitting and threshold selection used CDC 2022.

Run `generate_figures.py`, `generate_tables.py`, and `verify_submission.py --skip-hashes` after deliberate source changes; verify the delivered package without `--skip-hashes`. `analysis_code/temporal/run_2024_threshold_transport.py` reconstructs the corrected threshold table when harmonized public-use records and frozen model bundles are available. The public repository's `scripts/experiments/run_temporal_four_cell.py` reconstructs the complete replication.
