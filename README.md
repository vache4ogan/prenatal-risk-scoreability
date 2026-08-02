# Project overview

This publication-oriented directory isolates the canonical prenatal scoreability article analysis from historical experiments and large or private artifacts.

# Research question

At the same absolute referral capacity, how much additional population event capture is obtained when a fixed prenatal risk model can rank the full eligible cohort rather than only people entering prenatal care in months 1–3?

# Primary estimand

The primary estimand is full-cohort minus early-entry population event capture at fixed absolute budgets of 5% and 10% of each full target-specific population. `recall_among_eligible` is conditional on policy eligibility and is not population event capture.

# Data and temporal design

CDC Natality 2022 supports model development. CDC Natality 2023 is a temporal test. The primary population is U.S.-resident singleton births, with target-specific missing-outcome filtering.

# Canonical analysis

One fixed LightGBM model per target uses the primary `booking_strict` feature set. The same fitted model ranks early-entry and full-cohort candidates, and both policies receive the same absolute budget. A paired full-size bootstrap uses 500 replicates. Mechanism analysis is at the 10% budget, with a LightGBM calibration audit.

# Canonical results

At the 10% CDC 2023 absolute budget, full-cohort access increases population event capture by +2.58 percentage points for preterm birth, +2.73 pp for NICU admission, and +3.43 pp for low birth weight. Full precision is higher at equal budget.

# Repository layout

`configs/` contains the canonical editable configuration; `scripts/` contains the standalone analysis and preprocessing; `metadata/` contains design, harmonization, and the sanitized run manifest; `results/canonical_tables/` contains article-facing outputs; `docs/` records interpretation and reproduction; `tests/` records pending scientific invariants; and `repo_migration/` contains provenance and validation records.

# Reproduction

See `docs/reproduction.md`. Large data, model files, predictions, replicate-level artifacts, and secrets are deliberately excluded.

# Primary, sensitivity, secondary, and legacy analyses

Singleton fixed-model equal-budget analysis is primary. All-birth analysis is sensitivity only. `pregnancy_oracle` is secondary and retrospective, not available at booking. Shared-threshold effects around +5.5 to +8.5 pp used unequal realized referral counts and are legacy or supplementary, not the primary equal-budget effect.

# Interpretation guardrails

Do not describe CDC 2023 as external validation, conditional eligible recall as population capture, or class balancing as improving training. The design does not use class weighting or oversampling. Calibration did not improve the primary LightGBM.

# Data availability

CDC source data are not committed. Users must obtain authorized source files and produce the expected harmonized inputs locally or through the referenced ClearML dataset.

# Reproducibility status

Clean-clone reproducibility is not yet complete: exact historical package versions are absent here, the canonical full datasets and artifact-level `package_versions.txt` are external, skipped test skeletons require refactoring or fixtures, two requested canonical narrative documents were unavailable, and the historical manifest lacks Git commit identity.

# Citation

TODO: add the article citation and persistent identifier after publication metadata are finalized.

# License

TODO: confirm and add the repository license before redistribution.
