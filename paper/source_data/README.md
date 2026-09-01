# Source-data notes

This directory contains aggregate, non-record-level inputs for the manuscript generators.

- `cohort_counts_validation.csv`: target-specific CDC 2022/2023 cohort flow and event counts.
- `model_diagnostics.csv`: frozen LightGBM ranking diagnostics.
- `canonical_protocol_results.csv`: CDC 2023 candidate-set and policy results.
- `shapley_main_results.csv`: six point-estimate/interval rows for three outcomes and q in {5%, 10%}.
- `shapley_bootstrap_replicates.csv`: 500 paired bootstrap replicates per outcome and budget.
- `replication_2024_results.csv`: target-specific CDC 2024 workload realized by the stored six-decimal CDC 2022 q=10% score cutoffs, including exact counts and arithmetic fields.
- `sensitivity_no_priorterm.csv`: complete q=5%/10% Shapley results after removing prior other pregnancy outcomes.
- `sensitivity_no_race.csv`: q=10% comparison after removing maternal race; this supporting run has point estimates and no bootstrap interval.
- `synthetic_figure3_data.csv`: complete 81-condition mechanism grid summary.
- `synthetic_manifest.json`: locked simulation design, provenance, and invariant results.
- `figure1_care_entry_rates.csv`, `figure2_capacity_share.csv`, `figure_manifest.json`: derived by `generate_figures.py`.

The CDC 2024 table records both the exact canonical threshold and the six-decimal cutoff used by the archived transport run. Its claim is the realized workload under cutoff transport. The table does not assign that workload change to a specific upstream drift mechanism.

The value `temporal_test_2023` in the archived canonical CSV is a legacy run label. In the manuscript, CDC 2023 is described as the temporal evaluation cohort because the audit hypothesis was developed and evaluated on these scores; model fitting and threshold selection used CDC 2022.

Run `generate_figures.py`, `generate_tables.py`, and `verify_submission.py` after changing any source file. `analysis_code/temporal/run_2024_threshold_transport.py` reconstructs the CDC 2024 table when harmonized public-use records and the frozen model bundles are available.
